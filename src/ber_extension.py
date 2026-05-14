"""
BER sau ZF equalization + QPSK hard decision.
Công thức Ch.3 luận văn — không có trong CeBed gốc.
"""
import numpy as np


def zf_equalize(y: np.ndarray, h_hat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """x̂ = y · conj(h) / (|h|² + ε)  — Công thức (3.x) luận văn."""
    return y * np.conj(h_hat) / (np.abs(h_hat) ** 2 + eps)


def qpsk_demap(symbols: np.ndarray) -> np.ndarray:
    """Gray-coded QPSK: bit0 = Re<0, bit1 = Im<0. Trả về [..., 2] int32."""
    return np.stack([
        (np.real(symbols) < 0).astype(np.int32),
        (np.imag(symbols) < 0).astype(np.int32),
    ], axis=-1)


def compute_ber(y_list, h_hat_list, x_list, pilot_mask) -> float:
    """
    BER trung bình trên nhiều batch.
    Data mask = ~pilot_mask (không có null subcarrier trong cấu hình CeBed mặc định).
    """
    pilot_mask = np.squeeze(np.asarray(pilot_mask)).astype(bool)
    data_mask  = ~pilot_mask   # [n_symbols, fft_size]

    total_errors = total_bits = 0
    for y, h_hat, x_rg in zip(y_list, h_hat_list, x_list):
        y     = np.squeeze(np.asarray(y))
        h_hat = np.squeeze(np.asarray(h_hat))
        x_rg  = np.squeeze(np.asarray(x_rg))

        if y.ndim == 2:     y     = y[np.newaxis]
        if h_hat.ndim == 2: h_hat = h_hat[np.newaxis]
        if x_rg.ndim == 2:  x_rg  = x_rg[np.newaxis]

        x_hat = zf_equalize(y, h_hat)               # [batch, symbols, subcarriers]
        bits_hat  = qpsk_demap(x_hat[:, data_mask])  # [batch, n_data, 2]
        bits_sent = qpsk_demap(x_rg[:, data_mask])

        total_errors += int(np.sum(bits_hat != bits_sent))
        total_bits   += bits_hat.size

    return float(total_errors / total_bits) if total_bits > 0 else float("nan")
