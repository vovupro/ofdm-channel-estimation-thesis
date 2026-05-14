# Benchmark ước lượng kênh OFDM bằng CeBed

Đồ án tốt nghiệp ngành Viễn thông — so sánh các phương pháp ước lượng kênh OFDM
(LS, LMMSE, ChannelNet) trên hai mô hình kênh (Rayleigh, UMa 3GPP TR 38.901)
và hai kiểu pilot (block, kronecker). Phần đóng góp bổ sung: BER (ZF + QPSK)
và ablation ảnh hưởng Doppler.

## Yêu cầu hệ thống

- Python 3.10
- GPU NVIDIA (CUDA 11.8+) hoặc Google Colab T4
- Hệ điều hành: Linux / macOS / Windows (WSL2 khuyến nghị)

## Cài đặt

```bash
git clone --recursive https://github.com/vovupro/ofdm-channel-estimation-thesis.git
cd ofdm-channel-estimation-thesis
pip install -r requirements.txt
pip install -e cebed/
```

## Cách chạy

```bash
# Chạy toàn bộ pipeline (sinh data → train → BER → runtime → bảng → đồ thị)
python run_all_experiments.py

# Bỏ qua bước sinh dataset nếu đã có
python run_all_experiments.py --skip-generate

# Bỏ qua cả sinh dataset lẫn training
python run_all_experiments.py --skip-train
```

## Output

```
results/
├── datasets/          # HDF5 dataset mỗi experiment
├── train_output/      # Checkpoint + test_mses.csv (NMSE vs SNR)
├── ber/               # BER vs SNR theo JSON
├── ablation/          # NMSE ablation Doppler JSON
├── figures/           # PNG: ber_*.png, ablation_doppler.png
└── tables/            # CSV: table1_nmse.csv, table2_ber.csv, table3_runtime.csv
```

## Tham chiếu

- **CeBed** (SAIC-MONTREAL): https://github.com/SAIC-MONTREAL/CeBed — commit `d1a5e43`
- **ChannelNet**: M. Soltani et al., "Deep Learning-Based Channel Estimation," IEEE Commun. Lett., 2019. DOI: 10.1109/LCOMM.2019.2916797
- **Sionna 0.16.2**: https://nvlabs.github.io/sionna/

## Đóng góp của đồ án

| Phần | Mô tả |
|------|-------|
| `src/ber_extension.py` | Tính BER sau ZF equalization + QPSK hard decision (CeBed không có) |
| `src/run_ablation.py` | So sánh ChannelNet matched vs mismatch theo tốc độ Doppler |
| `src/make_tables.py` | Tổng hợp kết quả ra CSV cho luận văn |
