"""
Ablation Doppler: so sánh ChannelNet trained speed=3 (mismatch)
vs speed=30 (matched), cả hai evaluate trên môi trường speed=30.
"""
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
from src.utils import nmse_db

SNR_RANGE  = [0, 5, 10, 15, 20]
N_BATCHES  = 20
BATCH_SIZE = 50
OUT_DIR    = ROOT / "results" / "train_output"
ABL_DIR    = ROOT / "results" / "ablation"
FIG_DIR    = ROOT / "results" / "figures"
ABL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

EXP_NAME = "siso_1_uma_block_1_ps2_p72"

apply_noiseless = ApplyOFDMChannel(add_awgn=False, dtype=tf.complex64)

cfg = EnvConfig()
cfg.scenario, cfg.pilot_pattern = "uma", "block"
cfg.p_spacing, cfg.ue_speed     = 2, 30
cfg.carrier_frequency           = 3.0e9
env = OfdmEnv(cfg)

mask   = env.get_mask()
pilots = env.rg.pilot_pattern.pilots


def load_channelnet(output_name):
    model = get_model_class("ChannelNet")(CN_HPARAMS)
    model.build(tf.TensorShape(
        [None, env.n_pilot_symbols, env.n_pilot_subcarriers, 2]))
    ckpt = OUT_DIR / output_name / EXP_NAME / "0" / "ChannelNet" / "cp.ckpt"
    model.load_weights(str(ckpt)).expect_partial()
    return model


model_mismatch = load_channelnet("uma_block")
model_matched  = load_channelnet("uma_block_doppler30")

print("Ablation: evaluate trên env speed=30 m/s")
results = {key: [] for key in ["LS", "LMMSE", "ChannelNet_mismatch", "ChannelNet_matched"]}

for snr in SNR_RANGE:
    acc = {key: [] for key in results}

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
        h_mismatch = tf.map_fn(postprocess, model_mismatch(pre, training=False),
                               fn_output_signature=tf.complex64)
        h_matched  = tf.map_fn(postprocess, model_matched(pre,  training=False),
                               fn_output_signature=tf.complex64)

        h_np = h.numpy()
        acc["LS"].append(nmse_db(h_np, h_ls_full.numpy()))
        acc["LMMSE"].append(nmse_db(h_np, np.array(h_lmmse)))
        acc["ChannelNet_mismatch"].append(nmse_db(h_np, h_mismatch.numpy()))
        acc["ChannelNet_matched"].append(nmse_db(h_np, h_matched.numpy()))

    for key in results:
        results[key].append(float(np.mean(acc[key])))
    print(f"  SNR={snr:2d}dB  LS={results['LS'][-1]:.2f}  "
          f"LMMSE={results['LMMSE'][-1]:.2f}  "
          f"Mismatch={results['ChannelNet_mismatch'][-1]:.2f}  "
          f"Matched={results['ChannelNet_matched'][-1]:.2f} dB")

with open(ABL_DIR / "doppler_ablation.json", "w") as f:
    json.dump({"snr": SNR_RANGE, **results}, f, indent=2)

STYLES = {
    "LS":                  ("blue",   "o", "--"),
    "LMMSE":               ("orange", "s", "-."),
    "ChannelNet_mismatch": ("red",    "^", "-"),
    "ChannelNet_matched":  ("green",  "D", "-"),
}
fig, ax = plt.subplots(figsize=(6, 4))
for method, (color, marker, ls) in STYLES.items():
    ax.plot(SNR_RANGE, results[method], color=color, marker=marker,
            linestyle=ls, label=method.replace("_", " "))
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("NMSE (dB)")
ax.set_title("Ablation: Doppler mismatch vs matched (UMa, speed=30 m/s)")
ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR / "ablation_doppler.png", dpi=150)
plt.close()
print("  → results/ablation/doppler_ablation.json")
print("  → results/figures/ablation_doppler.png")
