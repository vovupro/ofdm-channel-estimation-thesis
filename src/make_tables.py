"""Tổng hợp kết quả ra CSV cho luận văn. Chạy sau tất cả các bước khác."""
import json, csv
from pathlib import Path
import pandas as pd

ROOT      = Path(__file__).parent.parent
BER_DIR   = ROOT / "results" / "ber"
TRAIN_DIR = ROOT / "results" / "train_output"
TABLE_DIR = ROOT / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

EXPS = ["rayleigh_block", "rayleigh_kronecker", "uma_block", "uma_kronecker"]
EXP_NAMES = {
    "rayleigh_block":     "siso_1_rayleigh_block_1_ps2_p72",
    "rayleigh_kronecker": "siso_1_rayleigh_kronecker_1_ps1_p72",
    "uma_block":          "siso_1_uma_block_1_ps2_p72",
    "uma_kronecker":      "siso_1_uma_kronecker_1_ps1_p72",
}
SNR_REF = 10
METHODS = ["LS", "LMMSE", "ChannelNet"]


def read_nmse(exp_folder):
    """Đọc MSE tại SNR_REF từ test_mses.csv (cột: snr, mse, method, seed)."""
    hits = list((TRAIN_DIR / exp_folder).rglob("test_mses.csv"))
    if not hits:
        return {}
    df = pd.read_csv(hits[0])
    row = df[df["snr"] == SNR_REF]
    return {r["method"]: round(float(r["mse"]), 6) for _, r in row.iterrows()}


# ── Table 1: NMSE tại SNR=10 dB ──────────────────────────────────────────────
with open(TABLE_DIR / "table1_nmse_snr10.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["experiment"] + METHODS)
    for exp in EXPS:
        nmse = read_nmse(exp)
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
        row = []
        for m in METHODS:
            val = data.get(m, {}).get(str(SNR_REF))
            row.append(f"{val:.2e}" if isinstance(val, (int, float)) else "N/A")
        w.writerow([exp] + row)
print("  → results/tables/table2_ber_snr10.csv")


# ── Table 3: Runtime & Params ─────────────────────────────────────────────────
runtime_file = ROOT / "results" / "runtime_params.json"
with open(TABLE_DIR / "table3_runtime_params.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "time_ms_per_batch", "params"])
    if runtime_file.exists():
        with open(runtime_file) as jf:
            rt = json.load(jf)
        w.writerow(["LS",         round(rt["LS_time_s"] * 1000, 2),         0])
        w.writerow(["LMMSE",      round(rt["LMMSE_time_s"] * 1000, 2),      0])
        w.writerow(["ChannelNet", round(rt["ChannelNet_time_s"] * 1000, 2), rt["ChannelNet_params"]])
    else:
        w.writerow(["N/A", "N/A", "N/A"])
print("  → results/tables/table3_runtime_params.csv")
