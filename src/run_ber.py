"""BER và NMSE vs SNR — load dataset có sẵn, không sinh kênh lại."""
import sys, json, glob
from pathlib import Path
import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "cebed"))

from sionna.channel import ApplyOFDMChannel
from cebed.envs import OfdmEnv, EnvConfig
from cebed.models import get_model_class
from cebed.baselines import linear_ls_baseline, lmmse_baseline
from cebed.utils import unflatten_last_dim
from cebed.datasets.sionna import preprocess_inputs
from cebed.datasets.utils import postprocess
from src.cn_config import CN_HPARAMS
from src.ber_extension import zf_equalize, qpsk_demap
from src.utils import nmse_db

SNR_RANGE  = [0, 5, 10, 15, 20]
BATCH_SIZE = 100
N_SAMPLES  = 500   # lấy N_SAMPLES từ mỗi domain trong dataset
OUT_DIR    = ROOT / "results" / "train_output"
DATA_DIR   = ROOT / "results" / "datasets"
BER_DIR    = ROOT / "results" / "ber"
FIG_DIR    = ROOT / "results" / "figures"
BER_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

EXPS = [
    dict(name="rayleigh_block",     exp_name="siso_1_rayleigh_block_1_ps2_p72",
         scenario="Rayleigh", pilot_pattern="block",     p_spacing=2, ue_speed=3),
    dict(name="rayleigh_kronecker", exp_name="siso_1_rayleigh_kronecker_1_ps1_p72",
         scenario="Rayleigh", pilot_pattern="kronecker", p_spacing=1, ue_speed=3),
    dict(name="uma_block",          exp_name="siso_1_uma_block_1_ps2_p72",
         scenario="uma",      pilot_pattern="block",     p_spacing=2, ue_speed=3),
    dict(name="uma_kronecker",      exp_name="siso_1_uma_kronecker_1_ps1_p72",
         scenario="uma",      pilot_pattern="kronecker", p_spacing=1, ue_speed=3),
]

METHODS = ["LS", "LMMSE", "ChannelNet"]
STYLES  = {
    "LS":         ("steelblue", "o", "--", 6, 1),
    "ChannelNet": ("red",       "^", "-",  7, 2),
    "LMMSE":      ("green",     "D", "-.", 9, 3),
}

apply_noiseless = ApplyOFDMChannel(add_awgn=False, dtype=tf.complex64)


def make_env(e):
    cfg = EnvConfig()
    cfg.scenario, cfg.pilot_pattern = e["scenario"], e["pilot_pattern"]
    cfg.p_spacing, cfg.ue_speed     = e["p_spacing"], e["ue_speed"]
    cfg.carrier_frequency           = 3.0e9
    return OfdmEnv(cfg)


def load_channelnet(env, e):
    model = get_model_class("ChannelNet")(CN_HPARAMS)
    model.build(tf.TensorShape(
        [None, env.n_pilot_symbols, env.n_pilot_subcarriers, 2]))
    ckpt = OUT_DIR / e["name"] / e["exp_name"] / "0" / "ChannelNet" / "cp.ckpt"
    model.load_weights(str(ckpt)).expect_partial()
    return model


def load_snr_data(name, snr_idx, n=N_SAMPLES):
    """Load h, y, x cho 1 SNR domain từ HDF5 dataset."""
    f_path = glob.glob(str(DATA_DIR / name / "**" / "data.hdf5"), recursive=True)[0]
    with h5py.File(f_path, "r") as f:
        h = np.array(f["h"][snr_idx, :n])
        y = np.array(f["y"][snr_idx, :n])
        x = np.array(f["x"][snr_idx, :n])
    return h, y, x


all_results = {}

for e in EXPS:
    print(f"\n--- {e['name']} ---")
    env     = make_env(e)
    model   = load_channelnet(env, e)
    mask    = env.get_mask()
    pilots  = env.rg.pilot_pattern.pilots
    mask_np = mask.numpy() if hasattr(mask, "numpy") else np.array(mask)
    data_mask = ~(np.squeeze(np.asarray(mask_np)).astype(bool))

    results = {m: {"ber": {}, "nmse": {}} for m in METHODS}

    for snr_idx, snr in enumerate(SNR_RANGE):
        # Load từ dataset — không chạy Sionna lại
        h_all, y_all, x_all = load_snr_data(e["name"], snr_idx)

        acc_nmse   = {m: [] for m in METHODS}
        acc_errors = {m: 0  for m in METHODS}
        acc_bits   = {m: 0  for m in METHODS}

        for i in range(0, len(h_all), BATCH_SIZE):
            h_b = tf.constant(h_all[i:i+BATCH_SIZE], dtype=tf.complex64)
            y_b = tf.constant(y_all[i:i+BATCH_SIZE], dtype=tf.complex64)
            x_b = tf.constant(x_all[i:i+BATCH_SIZE], dtype=tf.complex64)

            # ── LS ──────────────────────────────────────────────────────────
            h_ls = unflatten_last_dim(
                tf.math.divide_no_nan(env.extract_at_pilot_locations(y_b), pilots),
                (env.n_pilot_symbols, env.n_pilot_subcarriers))
            h_ls_full = linear_ls_baseline(
                h_ls, env.config.num_ofdm_symbols, env.config.fft_size)

            # ── LMMSE (dùng noiseless y từ dataset h và x) ───────────────────
            y_nl    = apply_noiseless([x_b, h_b])
            h_nl_ls = unflatten_last_dim(
                tf.math.divide_no_nan(env.extract_at_pilot_locations(y_nl), pilots),
                (env.n_pilot_symbols, env.n_pilot_subcarriers))
            h_lmmse = lmmse_baseline(
                h_nl_ls, h_b, h_ls, snr,
                env.pilot_ofdm_symbol_indices,
                env.config.num_ofdm_symbols, env.config.fft_size)

            # ── ChannelNet ───────────────────────────────────────────────────
            pre  = tf.map_fn(
                lambda z: preprocess_inputs(z, input_type="low", mask=mask),
                env.estimate_at_pilot_locations(y_b),
                fn_output_signature=tf.float32)
            h_cn = tf.map_fn(postprocess, model(pre, training=False),
                             fn_output_signature=tf.complex64)

            h_np = h_b.numpy()
            y_np = np.squeeze(y_b.numpy())
            x_np = np.squeeze(x_b.numpy())
            if y_np.ndim == 2: y_np = y_np[np.newaxis]
            if x_np.ndim == 2: x_np = x_np[np.newaxis]

            for method, h_hat_raw in [
                ("LS",         h_ls_full.numpy()),
                ("LMMSE",      np.array(h_lmmse)),
                ("ChannelNet", h_cn.numpy()),
            ]:
                h_hat = np.squeeze(h_hat_raw)
                if h_hat.ndim == 2: h_hat = h_hat[np.newaxis]

                acc_nmse[method].append(nmse_db(h_np, h_hat))

                x_hat     = zf_equalize(y_np, h_hat)
                bits_hat  = qpsk_demap(x_hat[:, data_mask])
                bits_sent = qpsk_demap(x_np[:, data_mask])
                acc_errors[method] += int(np.sum(bits_hat != bits_sent))
                acc_bits[method]   += bits_hat.size

        for method in METHODS:
            results[method]["nmse"][snr] = float(np.mean(acc_nmse[method]))
            results[method]["ber"][snr]  = (
                float(acc_errors[method] / acc_bits[method])
                if acc_bits[method] > 0 else float("nan"))

        print(f"  SNR={snr:2d}dB  "
              f"LS  nmse={results['LS']['nmse'][snr]:5.1f}dB ber={results['LS']['ber'][snr]:.4f}  "
              f"LMMSE nmse={results['LMMSE']['nmse'][snr]:5.1f}dB ber={results['LMMSE']['ber'][snr]:.4f}  "
              f"CN  nmse={results['ChannelNet']['nmse'][snr]:5.1f}dB ber={results['ChannelNet']['ber'][snr]:.4f}")

    with open(BER_DIR / f"{e['name']}.json", "w") as f:
        json.dump({m: {
            "ber":  {str(s): v for s, v in results[m]["ber"].items()},
            "nmse": {str(s): v for s, v in results[m]["nmse"].items()},
        } for m in METHODS}, f, indent=2)

    all_results[e["name"]] = results


# ── Vẽ hình ───────────────────────────────────────────────────────────────────
for e in EXPS:
    res = all_results[e["name"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(e["name"].replace("_", " ").title())

    for method, (color, marker, ls, ms, zorder) in STYLES.items():
        bers  = [res[method]["ber"][s]  for s in SNR_RANGE]
        nmses = [res[method]["nmse"][s] for s in SNR_RANGE]

        snr_b = [s for s, b in zip(SNR_RANGE, bers) if b > 0]
        ber_b = [b for b in bers if b > 0]
        ax1.semilogy(snr_b, ber_b, color=color, marker=marker, linestyle=ls,
                     markersize=ms, zorder=zorder, linewidth=1.8,
                     markeredgecolor="white", markeredgewidth=0.5, label=method)

        ax2.plot(SNR_RANGE, nmses, color=color, marker=marker, linestyle=ls,
                 markersize=ms, zorder=zorder, linewidth=1.8,
                 markeredgecolor="white", markeredgewidth=0.5, label=method)

    ax1.set_xlabel("SNR (dB)"); ax1.set_ylabel("BER")
    ax1.set_title("BER vs SNR")
    ax1.legend(); ax1.grid(True, which="both", linestyle="--", alpha=0.4)

    ax2.set_xlabel("SNR (dB)"); ax2.set_ylabel("NMSE (dB)")
    ax2.set_title("NMSE vs SNR")
    ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{e['name']}.png", dpi=150)
    plt.close()
    print(f"  → results/figures/{e['name']}.png")
