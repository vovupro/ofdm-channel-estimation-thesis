"""Hyperparameter ChannelNet SISO — dùng chung cho run_ber, run_runtime, run_ablation."""
CN_HPARAMS = dict(
    dropout_rate=0.1,
    sr_hidden_size=[64, 32],
    sr_kernels=[9, 1, 5],
    dc_hidden=64,
    num_dc_layers=18,
    input_type="low",
    lr=0.001,
    int_type="bilinear",
    output_dim=[14, 72, 2],
)
