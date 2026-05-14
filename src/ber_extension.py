"""
BER Extension: ZF equalization + QPSK hard decision → tính BER vs SNR.
Đây là phần đóng góp của đồ án (CeBed gốc không có BER).

Pipeline:
  y (received) + h_hat (estimated channel) → ZF → hard QPSK decision
  so sánh với bits từ x_rg (transmitted RG) → BER
"""
import numpy as np
from typing import Tuple


def zf_equalize(y: np.ndarray, h_hat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    ZF equalization với regularization tránh chia cho số nhỏ.
    x_hat = y * conj(h_hat) / (|h_hat|^2 + eps)

    Tham số eps=1e-8 để tránh khuếch đại nhiễu tại deep null.
    """
    h_conj = np.conj(h_hat)
    return y * h_conj / (np.abs(h_hat) ** 2 + eps)


def qpsk_hard_bits(symbols: np.ndarray) -> np.ndarray:
    """
    Hard decision QPSK (Gray coded):
      bit0 = 0 nếu Re(symbol) > 0, else 1
      bit1 = 0 nếu Im(symbol) > 0, else 1

    Trả về: array shape [..., 2] kiểu int32
    """
    bit0 = (np.real(symbols) < 0).astype(np.int32)
    bit1 = (np.imag(symbols) < 0).astype(np.int32)
    return np.stack([bit0, bit1], axis=-1)


def _get_data_mask(x_rg_squeezed: np.ndarray, pilot_mask_squeezed: np.ndarray) -> np.ndarray:
    """
    Tìm mask của các vị trí DATA (không phải pilot, không phải null subcarrier).

    - pilot_mask_squeezed: [n_symbols, fft_size] bool — True tại pilot
    - x_rg_squeezed: [batch, n_symbols, fft_size] complex

    Null subcarrier: trung bình |x_rg| theo batch ≈ 0
    """
    not_pilot = ~pilot_mask_squeezed.astype(bool)
    mean_abs_x = np.mean(np.abs(x_rg_squeezed), axis=0)  # [n_symbols, fft_size]
    not_null = mean_abs_x > 1e-6
    return not_pilot & not_null  # [n_symbols, fft_size]


def compute_ber(
    y: np.ndarray,
    h_hat: np.ndarray,
    x_rg: np.ndarray,
    pilot_mask: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Tính BER sau ZF equalization và hard QPSK decision.

    Tham số:
    - y:          [batch, n_rx, n_rx_ant, n_symbols, fft_size] complex
    - h_hat:      tensor bất kỳ shape, squeeze về [batch, n_symbols, fft_size]
    - x_rg:       [batch, n_ues, n_streams, n_symbols, fft_size] complex
    - pilot_mask: [n_ues, n_streams, n_symbols, fft_size] bool (True=pilot)
    - eps:        regularization ZF

    Trả về: BER float trong [0, 1]
    """
    # Chuyển về numpy nếu là tensor
    if hasattr(y, "numpy"):
        y = y.numpy()
    if hasattr(h_hat, "numpy"):
        h_hat = h_hat.numpy()
    if hasattr(x_rg, "numpy"):
        x_rg = x_rg.numpy()
    if hasattr(pilot_mask, "numpy"):
        pilot_mask = pilot_mask.numpy()

    # Squeeze về [batch, n_symbols, fft_size] cho SISO
    y_s      = np.squeeze(y)       # [batch, n_symbols, fft_size]
    h_s      = np.squeeze(h_hat)   # [batch, n_symbols, fft_size]
    x_s      = np.squeeze(x_rg)    # [batch, n_symbols, fft_size]
    mask_s   = np.squeeze(pilot_mask).astype(bool)  # [n_symbols, fft_size]

    # Đảm bảo đủ 3 chiều (batch, symbols, subcarriers)
    if y_s.ndim == 2:
        y_s = y_s[np.newaxis]
    if h_s.ndim == 2:
        h_s = h_s[np.newaxis]
    if x_s.ndim == 2:
        x_s = x_s[np.newaxis]

    # Tìm data mask
    data_mask = _get_data_mask(x_s, mask_s)  # [n_symbols, fft_size]
    if not data_mask.any():
        return float("nan")

    # ZF equalization
    x_hat = zf_equalize(y_s, h_s, eps=eps)  # [batch, n_symbols, fft_size]

    # Lấy các vị trí data
    x_hat_data = x_hat[:, data_mask]  # [batch, n_data_symbols]
    x_rg_data  = x_s[:, data_mask]   # [batch, n_data_symbols]

    # Hard QPSK decision
    bits_hat  = qpsk_hard_bits(x_hat_data)   # [batch, n_data, 2]
    bits_sent = qpsk_hard_bits(x_rg_data)    # [batch, n_data, 2]

    ber = float(np.mean(bits_hat != bits_sent))
    return ber


def compute_ber_batch(
    y_list, h_hat_list, x_rg_list, pilot_mask, eps: float = 1e-8
) -> float:
    """
    Tính BER trung bình trên nhiều batch.
    y_list, h_hat_list, x_rg_list: list of arrays (mỗi phần tử 1 batch)
    """
    total_errors = 0
    total_bits = 0

    for y, h_hat, x_rg in zip(y_list, h_hat_list, x_rg_list):
        if hasattr(y, "numpy"):
            y = y.numpy()
        if hasattr(h_hat, "numpy"):
            h_hat = h_hat.numpy()
        if hasattr(x_rg, "numpy"):
            x_rg = x_rg.numpy()
        if hasattr(pilot_mask, "numpy"):
            pilot_mask_np = pilot_mask.numpy()
        else:
            pilot_mask_np = pilot_mask

        y_s    = np.squeeze(y)
        h_s    = np.squeeze(h_hat)
        x_s    = np.squeeze(x_rg)
        mask_s = np.squeeze(pilot_mask_np).astype(bool)

        if y_s.ndim == 2:
            y_s = y_s[np.newaxis]
        if h_s.ndim == 2:
            h_s = h_s[np.newaxis]
        if x_s.ndim == 2:
            x_s = x_s[np.newaxis]

        data_mask = _get_data_mask(x_s, mask_s)
        if not data_mask.any():
            continue

        x_hat      = zf_equalize(y_s, h_s, eps=eps)
        x_hat_data = x_hat[:, data_mask]
        x_rg_data  = x_s[:, data_mask]

        bits_hat  = qpsk_hard_bits(x_hat_data)
        bits_sent = qpsk_hard_bits(x_rg_data)

        total_errors += int(np.sum(bits_hat != bits_sent))
        total_bits   += bits_hat.size

    if total_bits == 0:
        return float("nan")
    return float(total_errors / total_bits)
