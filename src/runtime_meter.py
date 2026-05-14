"""
Đo thời gian inference và đếm số tham số mô hình.
"""
import time
from typing import Callable, Any, Dict

import numpy as np


def measure_inference_time(
    model_fn: Callable,
    sample_input: Any,
    n_warmup: int = 5,
    n_runs: int = 50,
) -> float:
    """
    Đo thời gian inference trung bình (giây/batch).

    - n_warmup: số lần chạy khởi động (bỏ qua overhead lần đầu)
    - n_runs:   số lần chạy lấy trung bình
    """
    # Warm-up
    for _ in range(n_warmup):
        model_fn(sample_input)

    # Đo thời gian
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model_fn(sample_input)
        times.append(time.perf_counter() - t0)

    return float(np.mean(times))


def count_params(model) -> int:
    """
    Đếm số tham số có thể huấn luyện của Keras model.
    Trả về 0 nếu model không phải Keras model.
    """
    try:
        return int(model.count_params())
    except Exception:
        return 0


def measure_all_runtimes(
    env,
    channelnet_model,
    batch_size: int = 32,
    snr_db: int = 10,
    n_warmup: int = 5,
    n_runs: int = 50,
) -> Dict[str, Any]:
    """
    Đo runtime inference (giây/batch) cho LS, LMMSE và ChannelNet.
    Đếm số params của ChannelNet.

    Trả về dict với keys: LS_time, LMMSE_time, ChannelNet_time, ChannelNet_params
    """
    import tensorflow as tf
    from sionna.ofdm import ApplyOFDMChannel
    from cebed.baselines import linear_ls_baseline, lmmse_baseline
    from cebed.utils import unflatten_last_dim
    from cebed.datasets.sionna import preprocess_inputs as cebed_preprocess
    from cebed.datasets.utils import postprocess

    apply_noisy = ApplyOFDMChannel(add_awgn=True, dtype=tf.complex64)
    apply_noiseless = ApplyOFDMChannel(add_awgn=False, dtype=tf.complex64)
    mask = env.get_mask()
    noise_lin = tf.pow(10.0, -snr_db / 10.0)

    # Sinh 1 batch test data
    x_rg, y, h = env(batch_size, snr_db, return_x=True)

    # --- LS ---
    def ls_fn(y_in):
        pilots = env.rg.pilot_pattern.pilots
        y_p = env.extract_at_pilot_locations(y_in)
        h_ls = tf.math.divide_no_nan(y_p, pilots)
        h_ls = unflatten_last_dim(h_ls, (env.n_pilot_symbols, env.n_pilot_subcarriers))
        return linear_ls_baseline(h_ls, env.config.num_ofdm_symbols, env.config.fft_size)

    ls_time = measure_inference_time(ls_fn, y, n_warmup, n_runs)

    # --- LMMSE ---
    y_noiseless = apply_noiseless([x_rg, h])

    def lmmse_fn(y_in):
        pilots = env.rg.pilot_pattern.pilots
        # LS noiseless (để tính cross-covariance)
        y_p_nl = env.extract_at_pilot_locations(y_noiseless)
        h_nl_ls = tf.math.divide_no_nan(y_p_nl, pilots)
        h_nl_ls = unflatten_last_dim(h_nl_ls, (env.n_pilot_symbols, env.n_pilot_subcarriers))
        # LS noisy
        y_p = env.extract_at_pilot_locations(y_in)
        h_ls = tf.math.divide_no_nan(y_p, pilots)
        h_ls = unflatten_last_dim(h_ls, (env.n_pilot_symbols, env.n_pilot_subcarriers))
        return lmmse_baseline(
            h_nl_ls, h, h_ls, snr_db,
            env.pilot_ofdm_symbol_indices,
            env.config.num_ofdm_symbols, env.config.fft_size,
        )

    lmmse_time = measure_inference_time(lmmse_fn, y, n_warmup, n_runs)

    # --- ChannelNet ---
    inputs = env.estimate_at_pilot_locations(y)
    pre_inputs = tf.map_fn(
        lambda x: cebed_preprocess(x, input_type="low", mask=mask),
        inputs,
        fn_output_signature=tf.float32,
    )

    def cn_fn(pre_in):
        return channelnet_model(pre_in, training=False)

    cn_time = measure_inference_time(cn_fn, pre_inputs, n_warmup, n_runs)
    cn_params = count_params(channelnet_model)

    return {
        "LS_time_s":          round(ls_time, 6),
        "LMMSE_time_s":       round(lmmse_time, 6),
        "ChannelNet_time_s":  round(cn_time, 6),
        "ChannelNet_params":  cn_params,
        "LS_params":          0,
        "LMMSE_params":       0,
        "batch_size":         batch_size,
        "snr_db":             snr_db,
        "n_runs":             n_runs,
    }
