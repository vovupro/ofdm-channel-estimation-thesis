"""
run_all_experiments.py
Gọi các script CeBed có sẵn để chạy pipeline benchmark.
Chỉ thêm BER, Runtime, Ablation — phần đóng góp của đồ án.

Cách dùng:
    python run_all_experiments.py
    python run_all_experiments.py --skip-generate
    python run_all_experiments.py --skip-train
"""

import subprocess, sys, yaml
from pathlib import Path

ROOT   = Path(__file__).parent
CEBED  = ROOT / "cebed"
DATA   = ROOT / "results" / "datasets"
OUTPUT = ROOT / "results" / "train_output"
PYTHON = sys.executable

EXPS = [
    dict(name="rayleigh_block",     exp_name="siso_1_rayleigh_block_1_ps2_p72",
         scenario="rayleigh", pilot_pattern="block",     p_spacing=2, ue_speed=3),
    dict(name="rayleigh_kronecker", exp_name="siso_1_rayleigh_kronecker_1_ps1_p72",
         scenario="rayleigh", pilot_pattern="kronecker", p_spacing=1, ue_speed=3),
    dict(name="uma_block",          exp_name="siso_1_uma_block_1_ps2_p72",
         scenario="uma",      pilot_pattern="block",     p_spacing=2, ue_speed=3),
    dict(name="uma_kronecker",      exp_name="siso_1_uma_kronecker_1_ps1_p72",
         scenario="uma",      pilot_pattern="kronecker", p_spacing=1, ue_speed=3),
]

# Ablation: train matched (speed=30) để so sánh với mismatch (speed=3)
ABLATION_MATCHED = dict(name="uma_block_doppler30",
                        exp_name="siso_1_uma_block_1_ps2_p72",
                        scenario="uma", pilot_pattern="block", p_spacing=2, ue_speed=30)

# ── Helpers ───────────────────────────────────────────────────────────────────
def run(cmd, cwd=CEBED):
    cmd = [str(x) for x in cmd]
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd))


def find_data(base: Path) -> Path:
    hits = list(base.rglob("data.hdf5"))
    if not hits:
        raise FileNotFoundError(f"Không thấy data.hdf5 trong {base}.")
    return hits[0].parent


def gen_args(e: dict) -> list:
    return [
        PYTHON, CEBED / "scripts/generate_datasets_from_sionna.py",
        "--scenario",          e["scenario"],
        "--pilot_pattern",     e["pilot_pattern"],
        "--p_spacing",         e["p_spacing"],
        "--carrier_frequency", 3.0e9,
        "--ue_speed",          e["ue_speed"],
        "--num_domains",       5,
        "--start_ds",          0,
        "--end_ds",            25,
        "--size",              2000,
        "--output_dir",        DATA / e["name"],
    ]


def patch_channelnet_yaml():
    """Thêm experiment thiếu vào ChannelNet.yaml nếu chưa có (tránh KeyError)."""
    yaml_path = CEBED / "hyperparams" / "ChannelNet.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        hparams = yaml.safe_load(f)
    needed = [
        "siso_1_rayleigh_block_1_ps2_p72",
        "siso_1_rayleigh_kronecker_1_ps1_p72",
        "siso_1_uma_kronecker_1_ps1_p72",
    ]
    missing = [k for k in needed if k not in hparams]
    if not missing:
        return
    entry = (
        "\n{key}:\n"
        "  default:\n"
        "    dropout_rate: 0.1\n"
        "    sr_hidden_size: [64, 32]\n"
        "    sr_kernels: [9, 1, 5]\n"
        "    dc_hidden: 64\n"
        "    num_dc_layers: 18\n"
        "    input_type: low\n"
        "    lr: 0.001\n"
        "    int_type: bilinear\n"
        "    output_dim: [14, 72, 2]\n"
    )
    with open(yaml_path, "a", encoding="utf-8") as f:
        for key in missing:
            f.write(entry.format(key=key))
    print(f"Patched ChannelNet.yaml: {missing}")


# ── CLI flags ─────────────────────────────────────────────────────────────────
SKIP_GEN   = "--skip-generate" in sys.argv
SKIP_TRAIN = "--skip-train"    in sys.argv

# Fix blocker: đảm bảo ChannelNet.yaml có đủ keys trước khi train
patch_channelnet_yaml()

# ── Bước 1: Sinh dataset ──────────────────────────────────────────────────────
if not SKIP_GEN:
    print("\n" + "="*60)
    print("BƯỚC 1: Sinh dataset")
    print("="*60)
    for e in EXPS + [ABLATION_MATCHED]:
        run(gen_args(e))

# ── Bước 2: Train ChannelNet (CeBed tự eval LS & LMMSE → test_mses.csv) ─────
if not SKIP_TRAIN:
    print("\n" + "="*60)
    print("BƯỚC 2: Train ChannelNet (4 experiment + 1 ablation matched)")
    print("="*60)
    for e in EXPS + [ABLATION_MATCHED]:
        run([
            PYTHON, CEBED / "scripts/train.py",
            "--experiment_name", e["exp_name"],
            "--model_name",      "ChannelNet",
            "--data_dir",        find_data(DATA / e["name"]),
            "--output_dir",      OUTPUT / e["name"],
            "--epochs",          50,
            "--dataset_name",    "SionnaOfflineMD",
        ])

# ── Bước 3: Ablation Doppler (script riêng, evaluate.py không đủ) ─────────────
print("\n" + "="*60)
print("BƯỚC 3: Ablation Doppler mismatch vs matched (src/run_ablation.py)")
print("="*60)
run([PYTHON, ROOT / "src/run_ablation.py"], cwd=ROOT)

# ── Bước 4: BER (đóng góp đồ án) ─────────────────────────────────────────────
print("\n" + "="*60)
print("BƯỚC 4: BER via ZF + QPSK")
print("="*60)
run([PYTHON, ROOT / "src/run_ber.py"], cwd=ROOT)

# ── Bước 5: Runtime ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("BƯỚC 5: Runtime & Params")
print("="*60)
run([PYTHON, ROOT / "src/run_runtime.py"], cwd=ROOT)

# ── Bước 6: NMSE plot (cebed/scripts/plot.py) ─────────────────────────────────
print("\n" + "="*60)
print("BƯỚC 6: Vẽ NMSE vs SNR (cebed/scripts/plot.py)")
print("="*60)
run([PYTHON, CEBED / "scripts/plot.py", str(OUTPUT), "thesis_benchmark"])

# ── Bước 7: Sinh bảng CSV cho luận văn ───────────────────────────────────────
print("\n" + "="*60)
print("BƯỚC 7: Sinh bảng tổng hợp (src/make_tables.py)")
print("="*60)
run([PYTHON, ROOT / "src/make_tables.py"], cwd=ROOT)

print("\n=== Hoàn tất! Kết quả trong results/ và benchmark/figures/ ===")
