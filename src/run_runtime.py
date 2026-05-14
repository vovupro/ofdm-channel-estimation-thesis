"""
Đo thời gian inference và đếm params cho LS / LMMSE / ChannelNet.
Chạy sau khi train.py đã hoàn thành.
"""
import sys, json
from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "cebed"))

from cebed.envs import OfdmEnv, EnvConfig
from cebed.models import get_model_class
from src.runtime_meter import measure_all_runtimes

OUT_DIR     = ROOT / "results" / "train_output"
RUNTIME_DIR = ROOT / "results"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

# Đo trên uma_block làm đại diện (cùng kiến trúc, cùng kích thước)
EXP = dict(
    name="uma_block", exp_name="siso_1_uma_block_1_ps2_p72",
    scenario="uma", pilot_pattern="block", p_spacing=2, ue_speed=3,
)

CN_HPARAMS = dict(
    dropout_rate=0.1, sr_hidden_size=[64, 32], sr_kernels=[9, 1, 5],
    dc_hidden=64, num_dc_layers=18, input_type="low",
    lr=0.001, int_type="bilinear", output_dim=[14, 72, 2],
)

cfg = EnvConfig()
cfg.scenario, cfg.pilot_pattern = EXP["scenario"], EXP["pilot_pattern"]
cfg.p_spacing, cfg.ue_speed     = EXP["p_spacing"], EXP["ue_speed"]
cfg.carrier_frequency           = 3.0e9
env = OfdmEnv(cfg)

model = get_model_class("ChannelNet")(CN_HPARAMS)
model.build(tf.TensorShape([None, env.n_pilot_symbols, env.n_pilot_subcarriers, 2]))
ckpt = OUT_DIR / EXP["name"] / EXP["exp_name"] / "0" / "ChannelNet" / "cp.ckpt"
model.load_weights(str(ckpt)).expect_partial()

print("Đang đo runtime (50 lần chạy mỗi phương pháp)...")
result = measure_all_runtimes(env, model, batch_size=32, snr_db=10)

print(f"  LS:         {result['LS_time_s']*1000:.2f} ms/batch")
print(f"  LMMSE:      {result['LMMSE_time_s']*1000:.2f} ms/batch")
print(f"  ChannelNet: {result['ChannelNet_time_s']*1000:.2f} ms/batch  "
      f"({result['ChannelNet_params']:,} params)")

out_path = RUNTIME_DIR / "runtime_params.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"  → results/runtime_params.json")
