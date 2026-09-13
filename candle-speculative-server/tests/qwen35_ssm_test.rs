use candle::{
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Device, Result, Tensor,
};
use candle_speculative_server::{
    qwen35_ssm::Qwen35SsmLayer,
    Qwen35Config, Qwen35LayerState,
};
use candle_transformers::quantized_nn::RmsNorm;

fn make_qmatmul_eye(in_dim: usize, out_dim: usize, device: &Device) -> Result<QMatMul> {
    let mut data = vec![0f32; out_dim * in_dim];
    let min_dim = in_dim.min(out_dim);
    for i in 0..min_dim {
        data[i * in_dim + i] = 1.0;
    }
    let t = Tensor::from_vec(data, (out_dim, in_dim), device)?;
    Ok(QMatMul::Tensor(t))
}

fn make_qmatmul_zeros(in_dim: usize, out_dim: usize, device: &Device) -> Result<QMatMul> {
    let t = Tensor::zeros((out_dim, in_dim), DType::F32, device)?;
    Ok(QMatMul::Tensor(t))
}

fn make_rmsnorm_ones(dim: usize, eps: f64, device: &Device) -> Result<RmsNorm> {
    let t = Tensor::ones(dim, DType::F32, device)?;
    let q = QTensor::quantize(&t, GgmlDType::F32)?;
    RmsNorm::from_qtensor(q, eps)
}

fn create_test_ssm_layer(config: &Qwen35Config, device: &Device) -> Result<Qwen35SsmLayer> {
    let hidden_size = config.hidden_size;
    let conv_dim = config.ssm_conv_dim();
    let inner_size = config.ssm_inner_size;
    let dt_rank = config.ssm_dt_rank;
    let d_state = config.ssm_d_state;

    let attn_norm = make_rmsnorm_ones(hidden_size, config.rms_norm_eps, device)?;
    let attn_qkv = make_qmatmul_eye(hidden_size, conv_dim, device)?;
    let attn_gate = make_qmatmul_eye(hidden_size, inner_size, device)?;

    // ssm_conv1d: [10240, 4] with 1.0 on last tap (current token)
    let mut conv_w = vec![0f32; conv_dim * config.ssm_conv_kernel];
    for c in 0..conv_dim {
        conv_w[c * config.ssm_conv_kernel + (config.ssm_conv_kernel - 1)] = 1.0;
    }
    let ssm_conv1d = Tensor::from_vec(conv_w, (conv_dim, config.ssm_conv_kernel), device)?;

    // ssm_a: log decay rates for 48 heads, set to -0.1
    let ssm_a = Tensor::full(-0.1f32, dt_rank, device)?;

    let ssm_alpha = make_qmatmul_zeros(hidden_size, dt_rank, device)?;
    let ssm_beta = make_qmatmul_zeros(hidden_size, dt_rank, device)?;
    let ssm_dt = Tensor::zeros(dt_rank, DType::F32, device)?;

    let ssm_norm = make_rmsnorm_ones(d_state, config.rms_norm_eps, device)?;
    let ssm_out = make_qmatmul_eye(inner_size, hidden_size, device)?;

    Ok(Qwen35SsmLayer::new(
        attn_norm,
        attn_qkv,
        attn_gate,
        ssm_conv1d,
        ssm_a,
        ssm_alpha,
        ssm_beta,
        ssm_dt,
        ssm_norm,
        ssm_out,
        config.clone(),
    ))
}

#[test]
fn test_qwen35_ssm_layer_dimensions_and_initialization() -> Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();
    let layer = create_test_ssm_layer(&config, &device)?;

    assert_eq!(layer.config.hidden_size, 5120);
    assert_eq!(layer.config.ssm_conv_dim(), 10240);
    assert_eq!(layer.config.ssm_inner_size, 6144);
    assert_eq!(layer.config.ssm_dt_rank, 48);
    assert_eq!(layer.config.ssm_d_state, 128);
    assert_eq!(layer.ssm_conv1d.dims2()?, (10240, 4));
    assert_eq!(layer.ssm_a.dims1()?, 48);
    assert_eq!(layer.ssm_dt.dims1()?, 48);

    Ok(())
}

#[test]
fn test_qwen35_ssm_depthwise_conv1d_sliding_window() -> Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();
    let mut layer = create_test_ssm_layer(&config, &device)?;
    let mut state = Qwen35LayerState::new(&config, &device)?;

    // Initially conv_state is all zeros
    let conv_sum = state.conv_state.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert_eq!(conv_sum, 0.0);

    // Run step 1: input token with all ones
    let xs1 = Tensor::ones((1, 1, config.hidden_size), DType::F32, &device)?;
    let _ = layer.forward_decode(&xs1, &mut state)?;

    // conv_state should now hold the history
    assert_eq!(state.conv_state.dims3()?, (1, 3, 10240));
    // Last position in state (index 2) should correspond to xs1 projection
    let last_slice = state.conv_state.narrow(1, 2, 1)?;
    let last_val = last_slice.flatten_all()?.to_vec1::<f32>()?[0];
    assert!(last_val > 0.0, "Last conv history slot must be populated");

    // Run 3 more steps to completely fill the 3-step history buffer
    for _ in 0..3 {
        let xs = Tensor::ones((1, 1, config.hidden_size), DType::F32, &device)?;
        let _ = layer.forward_decode(&xs, &mut state)?;
    }

    // Now all 3 slots in conv_state should be populated (non-zero)
    let slot0_val = state.conv_state.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?[0];
    let slot1_val = state.conv_state.narrow(1, 1, 1)?.flatten_all()?.to_vec1::<f32>()?[0];
    let slot2_val = state.conv_state.narrow(1, 2, 1)?.flatten_all()?.to_vec1::<f32>()?[0];
    assert!(slot0_val > 0.0);
    assert!(slot1_val > 0.0);
    assert!(slot2_val > 0.0);

    Ok(())
}

#[test]
fn test_qwen35_ssm_delta_net_recurrence_and_decay() -> Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();
    let mut layer = create_test_ssm_layer(&config, &device)?;
    let mut state = Qwen35LayerState::new(&config, &device)?;

    // ssm_state is initially zero: [48, 128, 128]
    assert_eq!(state.ssm_state.dims3()?, (48, 128, 128));
    let initial_sum = state.ssm_state.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert_eq!(initial_sum, 0.0);

    // Step 1: feed non-zero input
    let xs = Tensor::full(0.5f32, (1, 1, config.hidden_size), &device)?;
    let out1 = layer.forward_decode(&xs, &mut state)?;
    assert_eq!(out1.dims3()?, (1, 1, config.hidden_size));

    // ssm_state should now be updated with outer product k (x) d
    let state_sum1 = state.ssm_state.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert!(state_sum1 > 0.0, "ssm_state must be updated after forward_decode");

    // Step 2: feed input with zeros (decay step)
    let xs_zero = Tensor::zeros((1, 1, config.hidden_size), DType::F32, &device)?;
    let out2 = layer.forward_decode(&xs_zero, &mut state)?;
    assert_eq!(out2.dims3()?, (1, 1, config.hidden_size));

    Ok(())
}

#[test]
fn test_qwen35_ssm_forward_decode_full_pass() -> Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();
    let mut layer = create_test_ssm_layer(&config, &device)?;
    let mut state = Qwen35LayerState::new(&config, &device)?;

    let xs = Tensor::randn(0.0f32, 1.0f32, (1, 1, config.hidden_size), &device)?;
    let out = layer.forward_decode(&xs, &mut state)?;

    assert_eq!(out.dims3()?, (1, 1, 5120));
    assert_eq!(out.dtype(), DType::F32);

    // Verify output has finite values
    let sum = out.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert!(sum.is_finite());
    assert!(sum > 0.0);

    Ok(())
}

#[test]
fn test_qwen35_ssm_2d_and_3d_input_support() -> Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();
    let mut layer = create_test_ssm_layer(&config, &device)?;
    let mut state1 = Qwen35LayerState::new(&config, &device)?;
    let mut state2 = Qwen35LayerState::new(&config, &device)?;

    let xs_2d = Tensor::full(0.3f32, (1, config.hidden_size), &device)?;
    let xs_3d = Tensor::full(0.3f32, (1, 1, config.hidden_size), &device)?;

    let out_2d = layer.forward_decode(&xs_2d, &mut state1)?;
    let out_3d = layer.forward_decode(&xs_3d, &mut state2)?;

    assert_eq!(out_2d.dims3()?, (1, 1, 5120));
    assert_eq!(out_3d.dims3()?, (1, 1, 5120));

    let diff = (out_2d.sub(&out_3d)?).abs()?.sum_all()?.to_scalar::<f32>()?;
    assert_eq!(diff, 0.0);

    Ok(())
}

#[test]
fn test_qwen35_ssm_snapshot_and_speculative_rollback() -> Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();
    let mut layer = create_test_ssm_layer(&config, &device)?;
    let mut state = Qwen35LayerState::new(&config, &device)?;

    // Step 1
    let xs1 = Tensor::randn(0.0f32, 1.0f32, (1, 1, config.hidden_size), &device)?;
    let _ = layer.forward_decode(&xs1, &mut state)?;

    // Take snapshot at step 1
    let snap = state.snapshot()?;

    // Speculative forward steps: step 2 and step 3
    let xs2 = Tensor::randn(0.0f32, 1.0f32, (1, 1, config.hidden_size), &device)?;
    let xs3 = Tensor::randn(0.0f32, 1.0f32, (1, 1, config.hidden_size), &device)?;
    let _ = layer.forward_decode(&xs2, &mut state)?;
    let _ = layer.forward_decode(&xs3, &mut state)?;

    // Rollback to snapshot
    state.restore(&snap);

    // Verify rollback restored exact tensor contents
    let diff_conv = (state.conv_state.sub(&snap.conv_state)?).abs()?.sum_all()?.to_scalar::<f32>()?;
    let diff_ssm = (state.ssm_state.sub(&snap.ssm_state)?).abs()?.sum_all()?.to_scalar::<f32>()?;
    assert_eq!(diff_conv, 0.0);
    assert_eq!(diff_ssm, 0.0);

    // Re-running step 2 after rollback should produce deterministic identical output
    let mut state_clone = snap.clone();
    let out_a = layer.forward_decode(&xs2, &mut state)?;
    let out_b = layer.forward_decode(&xs2, &mut state_clone)?;

    let diff_out = (out_a.sub(&out_b)?).abs()?.sum_all()?.to_scalar::<f32>()?;
    assert_eq!(diff_out, 0.0);

    Ok(())
}
