"""Tổng hợp kết quả ra CSV cho luận văn. Chạy sau tất cả các bước khác."""
import json, csv
from pathlib import Path

ROOT      = Path(__file__).parent.parent
BER_DIR   = ROOT / "results" / "ber"
TRAIN_DIR = ROOT / "results" / "train_output"
TABLE_DIR = ROOT / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

EXPS = ["rayleigh_block", "rayleigh_kronecker", "uma_block", "uma_kronecker"]
SNR_REF  = 10
METHODS  = ["LS", "LMMSE", "ChannelNet"]


# ── Table 1: NMSE tại SNR=10 dB ──────────────────────────────────────────────
# Đọc từ test_mses.csv mà train.py sinh ra
import csv as _csv

def read_nmse(exp_name_folder, exp_name):
    """Đọc MSE tại SNR_REF từ test_mses.csv, trả về dict {method: mse}."""
    import pandas as pd
    pattern = list((TRAIN_DIR / exp_name_folder).rglob("test_mses.csv"))
    if not pattern:
        return {}
    df = pd.read_csv(pattern[0])
    row = df[df["snr"] == SNR_REF]
    return {r["method"]: round(float(r["mse"]), 6) for _, r in row.iterrows()}

with open(TABLE_DIR / "table1_nmse_snr10.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["experiment"] + METHODS)
    for exp in EXPS:
        exp_name = {
            "rayleigh_block":     "siso_1_rayleigh_block_1_ps2_p72",
            "rayleigh_kronecker": "siso_1_rayleigh_kronecker_1_ps1_p72",
            "uma_block":          "siso_1_uma_block_1_ps2_p72",
            "uma_kronecker":      "siso_1_uma_kronecker_1_ps1_p72",
        }[exp]
        nmse = read_nmse(exp, exp_name)
        w.writerow([exp] + [nmse.get(m, "N/A") for m in METHODS])
print("  → results/tables/table1_nmse_snr10.csv")


# ── Table 2: BER tại SNR=10 dB ───────────────────────────────────────────────
with open(TABLE_DIR / "table2_ber_snr10.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["experiment"] + METHODS)
    for exp in EXPS:
        ber_file = BER_DIR / f"{exp}.json"
        if not ber_file.exists():
            w.writerow([exp] + ["N/A"] * len(METHODS))
            continue
        with open(ber_file) as jf:
            data = json.load(jf)
        w.writerow([exp] + [
            f"{data.get(m, {}).get(str(SNR_REF), 'N/A'):.2e}"
            if data.get(m, {}).get(str(SNR_REF)) != "N/A" else "N/A"
            for m in METHODS
        ])
print("  → results/tables/table2_ber_snr10.csv")


# ── Table 3: Runtime & Params ─────────────────────────────────────────────────
runtime_file = ROOT / "results" / "runtime_params.json"
with open(TABLE_DIR / "table3_runtime_params.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "time_ms_per_batch", "params"])
    if runtime_file.exists():
        with open(runtime_file) as jf:
            rt = json.load(jf)
        batch = rt.get("batch_size", 32)
        w.writerow(["LS",         round(rt["LS_time_s"] * 1000, 2),         0])
        w.writerow(["LMMSE",      round(rt["LMMSE_time_s"] * 1000, 2),      0])
        w.writerow(["ChannelNet", round(rt["ChannelNet_time_s"] * 1000, 2), rt["ChannelNet_params"]])
    else:
        w.writerow(["N/A", "N/A", "N/A"])
print("  → results/tables/table3_runtime_params.csv")
