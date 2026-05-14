"""
run_all_experiments.py
Gọi các script CeBed có sẵn để chạy pipeline benchmark.
Chỉ thêm BER và Runtime — phần đóng góp của đồ án.

Cách dùng (chạy từ thư mục gốc my-cebed-thesis/):
    python run_all_experiments.py
    python run_all_experiments.py --skip-generate   # nếu dataset đã có
    python run_all_experiments.py --skip-train      # nếu model đã train xong
"""

import subprocess, sys
from pathlib import Path

ROOT   = Path(__file__).parent
CEBED  = ROOT / "cebed"          # thư mục submodule CeBed
DATA   = ROOT / "results" / "datasets"
OUTPUT = ROOT / "results" / "train_output"
PYTHON = sys.executable

# ── Danh sách experiment ─────────────────────────────────────────────────────
EXPS = [
    dict(name="rayleigh_block",
         exp_name="siso_1_rayleigh_block_1_ps2_p72",
         scenario="rayleigh", pilot_pattern="block", p_spacing=2, ue_speed=3),
    dict(name="rayleigh_kronecker",
         exp_name="siso_1_rayleigh_kronecker_1_ps1_p72",
         scenario="rayleigh", pilot_pattern="kronecker", p_spacing=1, ue_speed=3),
    dict(name="uma_block",
         exp_name="siso_1_uma_block_1_ps2_p72",
         scenario="uma", pilot_pattern="block", p_spacing=2, ue_speed=3),
    dict(name="uma_kronecker",
         exp_name="siso_1_uma_kronecker_1_ps1_p72",
         scenario="uma", pilot_pattern="kronecker", p_spacing=1, ue_speed=3),
]

# Ablation: train trên speed=3, test trên speed=30
ABLATION = dict(name="uma_block_doppler30",
                exp_name="siso_1_uma_block_1_ps2_p72",
                scenario="uma", pilot_pattern="block", p_spacing=2, ue_speed=30)

# ── Helper ───────────────────────────────────────────────────────────────────
def run(cmd, cwd=CEBED):
    """Chạy lệnh subprocess, in ra màn hình."""
    cmd = [str(x) for x in cmd]
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd))


def find_data(base: Path) -> Path:
    """Tìm thư mục chứa data.hdf5 bên trong base."""
    hits = list(base.rglob("data.hdf5"))
    if not hits:
        raise FileNotFoundError(f"Không thấy data.hdf5 trong {base}. Hãy chạy bước generate trước.")
    return hits[0].parent


def gen_args(e: dict) -> list:
    """Tạo list args cho generate_datasets_from_sionna.py."""
    return [
        PYTHON, CEBED / "scripts/generate_datasets_from_sionna.py",
        "--scenario",          e["scenario"],
        "--pilot_pattern",     e["pilot_pattern"],
        "--p_spacing",         e["p_spacing"],
        "--carrier_frequency", 3.0e9,
        "--ue_speed",          e["ue_speed"],
        "--num_domains",       5,          # 5 SNR: 0,5,10,15,20 dB
        "--start_ds",          0,
        "--end_ds",            25,
        "--size",              2000,       # 2000 samples/domain
        "--output_dir",        DATA / e["name"],
    ]


# ── CLI flags đơn giản ───────────────────────────────────────────────────────
SKIP_GEN   = "--skip-generate" in sys.argv
SKIP_TRAIN = "--skip-train"    in sys.argv

# ── Bước 1: Sinh dataset ─────────────────────────────────────────────────────
if not SKIP_GEN:
    print("\n" + "="*60)
    print("BƯỚC 1: Sinh dataset (generate_datasets_from_sionna.py)")
    print("="*60)
    for e in EXPS + [ABLATION]:
        run(gen_args(e))

# ── Bước 2: Train + Evaluate ChannelNet (CeBed tự so sánh LS & LMMSE) ───────
if not SKIP_TRAIN:
    print("\n" + "="*60)
    print("BƯỚC 2: Train ChannelNet + Evaluate vs LS/LMMSE (train.py)")
    print("="*60)
    for e in EXPS:
        run([
            PYTHON, CEBED / "scripts/train.py",
            "--experiment_name", e["exp_name"],
            "--model_name",      "ChannelNet",
            "--data_dir",        find_data(DATA / e["name"]),
            "--output_dir",      OUTPUT / e["name"],
            "--epochs",          50,
            "--dataset_name",    "SionnaOfflineMD",
        ])

# ── Bước 3: Ablation — load model uma_block, test trên env speed=30 ──────────
print("\n" + "="*60)
print("BƯỚC 3: Ablation Doppler (evaluate.py — mismatch speed)")
print("="*60)

# uma_block model đã train ở bước 2, giờ evaluate trên dataset speed=30
ablation_model_dir = OUTPUT / "uma_block" / ABLATION["exp_name"] / "0" / "ChannelNet"
run([
    PYTHON, CEBED / "scripts/evaluate.py",
    str(ablation_model_dir),
    "LS", "LMMSE",
])

# ── Bước 4: BER (đóng góp đồ án — CeBed không có) ───────────────────────────
print("\n" + "="*60)
print("BƯỚC 4: BER via ZF + QPSK (src/ber_extension.py)")
print("="*60)
run([PYTHON, ROOT / "src/run_ber.py"], cwd=ROOT)

# ── Bước 5: Runtime (đóng góp đồ án — CeBed không có) ───────────────────────
print("\n" + "="*60)
print("BƯỚC 5: Runtime & Params (src/runtime_meter.py)")
print("="*60)
run([PYTHON, ROOT / "src/run_runtime.py"], cwd=ROOT)


# ── Bước 6: Vẽ NMSE (dùng plot.py CeBed) ────────────────────────────────────
print("\n" + "="*60)
print("BƯỚC 6: Vẽ NMSE vs SNR (cebed/scripts/plot.py)")
print("="*60)
run([PYTHON, CEBED / "scripts/plot.py",
     str(OUTPUT), "thesis_benchmark"])

print("\n=== Hoàn tất! Kết quả trong results/ và benchmark/figures/ ===")
