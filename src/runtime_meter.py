"""Đo thời gian inference và đếm params — không có trong CeBed gốc."""
import time
import numpy as np
import tensorflow as tf


def _timeit(fn, arg, n_warmup, n_runs):
    for _ in range(n_warmup):
        fn(arg)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(arg)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


def measure_all(env, model, batch_size: int = 32, snr_db: int = 10,
                n_warmup: int = 5, n_runs: int = 50) -> dict:
    """
    Đo runtime (giây/batch) cho LS, LMMSE, ChannelNet.
    Trả về dict: LS_time_s, LMMSE_time_s, ChannelNet_time_s, ChannelNet_params.
    """
    from sionna.channel import ApplyOFDMChannel
    from cebed.baselines import linear_ls_baseline, lmmse_baseline
    from cebed.utils import unflatten_last_dim
    from cebed.datasets.sionna import preprocess_inputs

    apply_noiseless = ApplyOFDMChannel(add_awgn=False, dtype=tf.complex64)
    pilots = env.rg.pilot_pattern.pilots
    mask   = env.get_mask()

    x_rg, y, h = env(batch_size, snr_db, return_x=True)
    y_nl = apply_noiseless([x_rg, h])

    def run_ls(_y):
        h_ls = unflatten_last_dim(
            tf.math.divide_no_nan(env.extract_at_pilot_locations(_y), pilots),
            (env.n_pilot_symbols, env.n_pilot_subcarriers))
        return linear_ls_baseline(h_ls, env.config.num_ofdm_symbols, env.config.fft_size)

    def run_lmmse(_y):
        h_ls = unflatten_last_dim(
            tf.math.divide_no_nan(env.extract_at_pilot_locations(_y), pilots),
            (env.n_pilot_symbols, env.n_pilot_subcarriers))
        h_nl = unflatten_last_dim(
            tf.math.divide_no_nan(env.extract_at_pilot_locations(y_nl), pilots),
            (env.n_pilot_symbols, env.n_pilot_subcarriers))
        return lmmse_baseline(h_nl, h, h_ls, snr_db,
                              env.pilot_ofdm_symbol_indices,
                              env.config.num_ofdm_symbols, env.config.fft_size)

    pre = tf.map_fn(
        lambda z: preprocess_inputs(z, input_type="low", mask=mask),
        env.estimate_at_pilot_locations(y), fn_output_signature=tf.float32)

    return {
        "LS_time_s":         round(_timeit(run_ls,              y,   n_warmup, n_runs), 6),
        "LMMSE_time_s":      round(_timeit(run_lmmse,           y,   n_warmup, n_runs), 6),
        "ChannelNet_time_s": round(_timeit(lambda p: model(p, training=False), pre, n_warmup, n_runs), 6),
        "ChannelNet_params": int(model.count_params()),
        "batch_size":        batch_size,
        "snr_db":            snr_db,
        "n_runs":            n_runs,
    }
