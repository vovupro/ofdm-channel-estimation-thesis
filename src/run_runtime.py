"""Đo thời gian inference và số params. Chạy sau khi train.py xong."""
import sys, json
from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "cebed"))

from cebed.envs import OfdmEnv, EnvConfig
from cebed.models import get_model_class
from src.cn_config import CN_HPARAMS
from src.runtime_meter import measure_all

OUT_DIR = ROOT / "results" / "train_output"
EXP = dict(name="uma_block", exp_name="siso_1_uma_block_1_ps2_p72",
           scenario="uma", pilot_pattern="block", p_spacing=2, ue_speed=3)

cfg = EnvConfig()
cfg.scenario, cfg.pilot_pattern = EXP["scenario"], EXP["pilot_pattern"]
cfg.p_spacing, cfg.ue_speed     = EXP["p_spacing"], EXP["ue_speed"]
cfg.carrier_frequency           = 3.0e9
env = OfdmEnv(cfg)

model = get_model_class("ChannelNet")(CN_HPARAMS)
model.build(tf.TensorShape([None, env.n_pilot_symbols, env.n_pilot_subcarriers, 2]))
ckpt = OUT_DIR / EXP["name"] / EXP["exp_name"] / "0" / "ChannelNet" / "cp.ckpt"
model.load_weights(str(ckpt)).expect_partial()

print("Đo runtime (50 lần mỗi phương pháp)...")
result = measure_all(env, model, batch_size=32, snr_db=10)

print(f"  LS:         {result['LS_time_s']*1000:.2f} ms/batch")
print(f"  LMMSE:      {result['LMMSE_time_s']*1000:.2f} ms/batch")
print(f"  ChannelNet: {result['ChannelNet_time_s']*1000:.2f} ms/batch"
      f"  ({result['ChannelNet_params']:,} params)")

out = ROOT / "results" / "runtime_params.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(f"  → results/runtime_params.json")
