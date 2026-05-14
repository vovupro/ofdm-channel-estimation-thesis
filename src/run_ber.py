"""BER và NMSE vs SNR — dùng dataset HDF5 có sẵn, cùng 1 vòng lặp.
Bao gồm ablation Doppler mismatch vs matched (UMa block, speed=30).
"""
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
N_SAMPLES  = 200        # test_datasets: 200 mẫu/SNR, seed=99, tách biệt train
OUT_DIR    = ROOT / "results" / "train_output"
DATA_DIR   = ROOT / "results" / "test_datasets"
BER_DIR    = ROOT / "results" / "ber"
FIG_DIR    = ROOT / "results" / "figures"
ABL_DIR    = ROOT / "results" / "ablation"
for d in (BER_DIR, FIG_DIR, ABL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 4 experiment chính
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

STYLES_MAIN = {
    "LS":         ("steelblue", "o", "--", 6, 1),
    "ChannelNet": ("red",       "^", "-",  7, 2),
    "LMMSE":      ("green",     "D", "-.", 9, 3),
}

STYLES_ABL = {
    "LS":                  ("steelblue", "o", "--", 6, 1),
    "LMMSE":               ("green",     "D", "-.", 9, 2),
    "ChannelNet_mismatch": ("red",       "^", "-",  7, 3),
    "ChannelNet_matched":  ("purple",    "s", "-",  7, 4),
}

apply_noiseless = ApplyOFDMChannel(add_awgn=False, dtype=tf.complex64)


def make_env(scenario, pilot_pattern, p_spacing, ue_speed):
    cfg = EnvConfig()
    cfg.scenario, cfg.pilot_pattern = scenario, pilot_pattern
    cfg.p_spacing, cfg.ue_speed     = p_spacing, ue_speed
    cfg.carrier_frequency           = 3.0e9
    return OfdmEnv(cfg)


def load_channelnet(env, train_name, exp_name):
    model = get_model_class("ChannelNet")(CN_HPARAMS)
    model.build(tf.TensorShape([None, env.n_pilot_symbols, env.n_pilot_subcarriers, 2]))
    ckpt = OUT_DIR / train_name / exp_name / "0" / "ChannelNet" / "cp.ckpt"
    model.load_weights(str(ckpt)).expect_partial()
    return model


def load_domain(dataset_name, snr_idx, n=N_SAMPLES):
    path = glob.glob(str(DATA_DIR / dataset_name / "**" / "data.hdf5"), recursive=True)[0]
    with h5py.File(path, "r") as f:
        return (np.array(f["h"][snr_idx, :n]),
                np.array(f["y"][snr_idx, :n]),
                np.array(f["x"][snr_idx, :n]))


def get_h_hats(env, models_dict, mask, pilots, h_b, y_b, x_b, snr):
    """Trả về {method: h_hat} cho tất cả phương pháp trong models_dict."""
    h_ls = unflatten_last_dim(
        tf.math.divide_no_nan(env.extract_at_pilot_locations(y_b), pilots),
        (env.n_pilot_symbols, env.n_pilot_subcarriers))
    h_ls_full = linear_ls_baseline(h_ls, env.config.num_ofdm_symbols, env.config.fft_size)

    y_nl    = apply_noiseless([x_b, h_b])
    h_nl_ls = unflatten_last_dim(
        tf.math.divide_no_nan(env.extract_at_pilot_locations(y_nl), pilots),
        (env.n_pilot_symbols, env.n_pilot_subcarriers))
    h_lmmse = lmmse_baseline(h_nl_ls, h_b, h_ls, snr,
                             env.pilot_ofdm_symbol_indices,
                             env.config.num_ofdm_symbols, env.config.fft_size)

    pre = tf.map_fn(lambda z: preprocess_inputs(z, input_type="low", mask=mask),
                    env.estimate_at_pilot_locations(y_b), fn_output_signature=tf.float32)

    result = {"LS": np.squeeze(h_ls_full.numpy()),
              "LMMSE": np.squeeze(np.array(h_lmmse))}
    for name, model in models_dict.items():
        h_cn = tf.map_fn(postprocess, model(pre, training=False), fn_output_signature=tf.complex64)
        result[name] = np.squeeze(h_cn.numpy())
    return result


def run_experiment(dataset_name, env, models_dict, styles):
    """Chạy BER+NMSE cho 1 experiment, trả về results dict."""
    mask      = env.get_mask()
    pilots    = env.rg.pilot_pattern.pilots
    data_mask = ~np.squeeze(np.asarray(
        mask.numpy() if hasattr(mask, "numpy") else np.array(mask)
    )).astype(bool)

    results = {m: {"ber": {}, "nmse": {}} for m in styles}

    for snr_idx, snr in enumerate(SNR_RANGE):
        h_all, y_all, x_all = load_domain(dataset_name, snr_idx)
        acc = {m: {"nmse": [], "errors": 0, "bits": 0} for m in styles}

        for i in range(0, len(h_all), BATCH_SIZE):
            h_b = tf.constant(h_all[i:i+BATCH_SIZE], dtype=tf.complex64)
            y_b = tf.constant(y_all[i:i+BATCH_SIZE], dtype=tf.complex64)
            x_b = tf.constant(x_all[i:i+BATCH_SIZE], dtype=tf.complex64)

            h_hats = get_h_hats(env, models_dict, mask, pilots, h_b, y_b, x_b, snr)

            h_np = h_b.numpy()
            y_np = np.squeeze(y_b.numpy())
            x_np = np.squeeze(x_b.numpy())
            if y_np.ndim == 2: y_np = y_np[np.newaxis]
            if x_np.ndim == 2: x_np = x_np[np.newaxis]

            for method, h_hat in h_hats.items():
                if h_hat.ndim == 2: h_hat = h_hat[np.newaxis]
                acc[method]["nmse"].append(nmse_db(h_np, h_hat))
                bits_hat  = qpsk_demap(zf_equalize(y_np, h_hat)[:, data_mask])
                bits_sent = qpsk_demap(x_np[:, data_mask])
                acc[method]["errors"] += int(np.sum(bits_hat != bits_sent))
                acc[method]["bits"]   += bits_hat.size

        for method in styles:
            results[method]["nmse"][snr] = float(np.mean(acc[method]["nmse"]))
            results[method]["ber"][snr]  = (
                float(acc[method]["errors"] / acc[method]["bits"])
                if acc[method]["bits"] > 0 else float("nan"))

        print(f"  SNR={snr:2d}dB  " + "  ".join(
            f"{m} nmse={results[m]['nmse'][snr]:5.1f}dB ber={results[m]['ber'][snr]:.4f}"
            for m in styles))

    return results


def plot_results(res, styles, title, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(title)
    for method, (color, marker, ls, ms, zorder) in styles.items():
        kw = dict(color=color, marker=marker, linestyle=ls, markersize=ms,
                  zorder=zorder, linewidth=1.8, markeredgecolor="white",
                  markeredgewidth=0.5, label=method.replace("_", " "))
        bers  = [res[method]["ber"][s]  for s in SNR_RANGE]
        nmses = [res[method]["nmse"][s] for s in SNR_RANGE]
        snr_b = [s for s, b in zip(SNR_RANGE, bers) if b > 0]
        ax1.semilogy(snr_b, [b for b in bers if b > 0], **kw)
        ax2.plot(SNR_RANGE, nmses, **kw)
    for ax, ylabel, title_ax in [
        (ax1, "BER",       "BER vs SNR"),
        (ax2, "NMSE (dB)", "NMSE vs SNR"),
    ]:
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel(ylabel); ax.set_title(title_ax)
        ax.legend(); ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → {out_path.relative_to(ROOT)}")


# ── 4 experiment chính ────────────────────────────────────────────────────────
all_results = {}
for e in EXPS:
    print(f"\n--- {e['name']} ---")
    env   = make_env(e["scenario"], e["pilot_pattern"], e["p_spacing"], e["ue_speed"])
    model = load_channelnet(env, e["name"], e["exp_name"])
    res   = run_experiment(e["name"], env, {"ChannelNet": model}, STYLES_MAIN)
    with open(BER_DIR / f"{e['name']}.json", "w") as f:
        json.dump({m: {"ber":  {str(s): v for s, v in res[m]["ber"].items()},
                       "nmse": {str(s): v for s, v in res[m]["nmse"].items()}}
                   for m in STYLES_MAIN}, f, indent=2)
    plot_results(res, STYLES_MAIN,
                 e["name"].replace("_", " ").title(),
                 FIG_DIR / f"{e['name']}.png")
    all_results[e["name"]] = res


# ── Ablation Doppler: mismatch (speed=3) vs matched (speed=30) ────────────────
print("\n--- Ablation Doppler (UMa block, eval speed=30) ---")
EXP_NAME = "siso_1_uma_block_1_ps2_p72"
env_30   = make_env("uma", "block", 2, 30)
models_abl = {
    "ChannelNet_mismatch": load_channelnet(env_30, "uma_block",          EXP_NAME),
    "ChannelNet_matched":  load_channelnet(env_30, "uma_block_doppler30", EXP_NAME),
}
res_abl = run_experiment("uma_block_doppler30", env_30, models_abl, STYLES_ABL)
with open(ABL_DIR / "doppler_ablation.json", "w") as f:
    json.dump({m: {"ber":  {str(s): v for s, v in res_abl[m]["ber"].items()},
                   "nmse": {str(s): v for s, v in res_abl[m]["nmse"].items()}}
               for m in STYLES_ABL}, f, indent=2)
plot_results(res_abl, STYLES_ABL,
             "Ablation Doppler — UMa Block (eval speed=30 m/s)",
             FIG_DIR / "ablation_doppler.png")
