"""Tính BER vs SNR — ZF + QPSK. Chạy sau khi train.py xong."""
import sys, json
from pathlib import Path
import numpy as np
import tensorflow as tf
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
from src.ber_extension import compute_ber

SNR_RANGE  = [0, 5, 10, 15, 20]
N_BATCHES  = 20
BATCH_SIZE = 50
OUT_DIR    = ROOT / "results" / "train_output"
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


all_ber = {}

for e in EXPS:
    print(f"\n--- BER: {e['name']} ---")
    env    = make_env(e)
    model  = load_channelnet(env, e)
    mask   = env.get_mask()
    pilots = env.rg.pilot_pattern.pilots
    mask_np = mask.numpy() if hasattr(mask, "numpy") else np.array(mask)
    results = {m: {} for m in ["LS", "LMMSE", "ChannelNet"]}

    for snr in SNR_RANGE:
        y_list, x_list, ls_list, lmmse_list, cn_list = [], [], [], [], []

        for _ in range(N_BATCHES):
            x_rg, y, h = env(BATCH_SIZE, snr, return_x=True)

            # LS — giống Trainer.evaluate_ls()
            h_ls = unflatten_last_dim(
                tf.math.divide_no_nan(env.extract_at_pilot_locations(y), pilots),
                (env.n_pilot_symbols, env.n_pilot_subcarriers))
            h_ls_full = linear_ls_baseline(
                h_ls, env.config.num_ofdm_symbols, env.config.fft_size)

            # LMMSE — giống Trainer.evaluate_lmmse()
            y_nl = apply_noiseless([x_rg, h])
            h_nl_ls = unflatten_last_dim(
                tf.math.divide_no_nan(env.extract_at_pilot_locations(y_nl), pilots),
                (env.n_pilot_symbols, env.n_pilot_subcarriers))
            h_lmmse = lmmse_baseline(
                h_nl_ls, h, h_ls, snr,
                env.pilot_ofdm_symbol_indices,
                env.config.num_ofdm_symbols, env.config.fft_size)

            # ChannelNet — giống Trainer.evaluate()
            pre = tf.map_fn(
                lambda z: preprocess_inputs(z, input_type="low", mask=mask),
                env.estimate_at_pilot_locations(y),
                fn_output_signature=tf.float32)
            h_cn = tf.map_fn(postprocess, model(pre, training=False),
                             fn_output_signature=tf.complex64)

            y_list.append(y.numpy());    x_list.append(x_rg.numpy())
            ls_list.append(h_ls_full.numpy())
            lmmse_list.append(np.array(h_lmmse))
            cn_list.append(h_cn.numpy())

        results["LS"][snr]         = compute_ber(y_list, ls_list,    x_list, mask_np)
        results["LMMSE"][snr]      = compute_ber(y_list, lmmse_list, x_list, mask_np)
        results["ChannelNet"][snr] = compute_ber(y_list, cn_list,    x_list, mask_np)
        print(f"  SNR={snr:2d}dB  LS={results['LS'][snr]:.4f}  "
              f"LMMSE={results['LMMSE'][snr]:.4f}  "
              f"CN={results['ChannelNet'][snr]:.4f}")

    with open(BER_DIR / f"{e['name']}.json", "w") as f:
        json.dump({k: {str(s): v for s, v in d.items()}
                   for k, d in results.items()}, f, indent=2)
    all_ber[e["name"]] = results

# Vẽ BER
STYLES = {
    "LS":         ("steelblue", "o", "--",  6, 1),
    "ChannelNet": ("red",       "^", "-",   7, 2),
    "LMMSE":      ("green",     "D", "-.",  9, 3),
}
for e in EXPS:
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, (color, marker, ls, ms, zorder) in STYLES.items():
        bers = [all_ber[e["name"]][method][snr] for snr in SNR_RANGE]
        snr_plot = [s for s, b in zip(SNR_RANGE, bers) if b > 0]
        ber_plot = [b for b in bers if b > 0]
        ax.semilogy(snr_plot, ber_plot, color=color, marker=marker, linestyle=ls,
                    markersize=ms, zorder=zorder, linewidth=1.8,
                    markeredgecolor="white", markeredgewidth=0.5, label=method)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER")
    ax.set_title(f"BER vs SNR — {e['name']}")
    ax.legend(); ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"ber_{e['name']}.png", dpi=150)
    plt.close()
    print(f"  → results/figures/ber_{e['name']}.png")
