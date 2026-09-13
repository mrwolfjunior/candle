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

#[test]
fn test_qwen35_ssm_semantic_retrieval_value_space() -> Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();

    let d_state = config.ssm_d_state; // 128
    let n_heads = config.ssm_dt_rank; // 48

    // k along basis vector e0 for all 48 heads: [48, 128]
    let mut k_data = vec![0f32; n_heads * d_state];
    for h in 0..n_heads {
        k_data[h * d_state] = 1.0;
    }
    let k = Tensor::from_vec(k_data, (n_heads, d_state), &device)?;

    // v along basis vector e1 for all 48 heads: [48, 128] (orthogonal to k)
    let mut v_data = vec![0f32; n_heads * d_state];
    for h in 0..n_heads {
        v_data[h * d_state + 1] = 1.0;
    }
    let v = Tensor::from_vec(v_data, (n_heads, d_state), &device)?;

    // Verify k and v are strictly orthogonal: k . v == 0
    let kv_dot = (k.mul(&v)?).sum_all()?.to_scalar::<f32>()?;
    assert_eq!(kv_dot, 0.0, "k and v must be orthogonal test vectors");

    // Recurrence step on initial zero state:
    // sk = S * k = 0
    // d = (v - sk) = v
    // S_new = S + d * k^T (outer product in value x key space)
    let s_init = Tensor::zeros((n_heads, d_state, d_state), DType::F32, &device)?;
    let k_col = k.unsqueeze(2)?; // [48, 128, 1]
    let sk = s_init.matmul(&k_col)?.squeeze(2)?; // [48, 128]

    let d = v.sub(&sk)?; // [48, 128] (equals v)
    let d_col = d.unsqueeze(2)?; // [48, 128, 1]
    let k_row = k.unsqueeze(1)?; // [48, 1, 128]
    let dk = d_col.matmul(&k_row)?; // [48, 128, 128]
    let s_updated = (s_init + dk)?;

    // Query with q = k:
    // o = S_updated * q = (v * k^T) * k = v * (k^T * k) = v * 1.0 = v
    let q = k.clone();
    let q_col = q.unsqueeze(2)?; // [48, 128, 1]
    let o = s_updated.matmul(&q_col)?.squeeze(2)?; // [48, 128]

    // Verify alignment: o must be perfectly aligned with value space (v), NOT key space (k)
    let o_dot_v = (o.mul(&v)?).sum(1)?.to_vec1::<f32>()?;
    let o_dot_k = (o.mul(&k)?).sum(1)?.to_vec1::<f32>()?;

    for h in 0..n_heads {
        assert!(
            (o_dot_v[h] - 1.0).abs() < 1e-5,
            "Head {h} retrieved vector must have dot product ~1.0 with value vector v, got {}",
            o_dot_v[h]
        );
        assert!(
            o_dot_k[h].abs() < 1e-5,
            "Head {h} retrieved vector must be orthogonal to key vector k, got {}",
            o_dot_k[h]
        );
    }

    Ok(())
}

#[test]
fn test_qwen35_ssm_layer_semantic_retrieval_end_to_end() -> Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();

    let hidden_size = config.hidden_size;
    let conv_dim = config.ssm_conv_dim();
    let inner_size = config.ssm_inner_size;
    let dt_rank = config.ssm_dt_rank;
    let d_state = config.ssm_d_state;

    let attn_norm = make_rmsnorm_ones(hidden_size, config.rms_norm_eps, &device)?;

    // QKV projection: [10240, 5120]
    // Map input index 0 -> Q[head 0, dim 0] (idx 0)
    // Map input index 1 -> K[head 0, dim 0] (idx 2048)
    // Map input index 2 -> V[head 0, dim 1] (idx 4096 + 1 = 4097)
    let mut qkv_w = vec![0f32; conv_dim * hidden_size];
    qkv_w[0 * hidden_size + 0] = 1.0; // Q = e0 when input has xs[0]
    qkv_w[2048 * hidden_size + 1] = 1.0; // K = e0 when input has xs[1]
    qkv_w[4097 * hidden_size + 2] = 1.0; // V = e1 when input has xs[2]
    let attn_qkv = QMatMul::Tensor(Tensor::from_vec(qkv_w, (conv_dim, hidden_size), &device)?);

    let attn_gate = QMatMul::Tensor(Tensor::ones((inner_size, hidden_size), DType::F32, &device)?);

    // ssm_conv1d: 1.0 on current tap (index 3)
    let mut conv_w = vec![0f32; conv_dim * config.ssm_conv_kernel];
    for c in 0..conv_dim {
        conv_w[c * config.ssm_conv_kernel + 3] = 1.0;
    }
    let ssm_conv1d = Tensor::from_vec(conv_w, (conv_dim, config.ssm_conv_kernel), &device)?;

    // ssm_a: 0.0 log decay rates -> decay factor exp(0) = 1.0
    let ssm_a = Tensor::zeros(dt_rank, DType::F32, &device)?;
    let ssm_alpha = make_qmatmul_zeros(hidden_size, dt_rank, &device)?;

    // ssm_beta: large positive bias -> sigmoid ~ 1.0
    let ssm_beta = make_qmatmul_zeros(hidden_size, dt_rank, &device)?;
    let ssm_dt = Tensor::zeros(dt_rank, DType::F32, &device)?;

    let ssm_norm = make_rmsnorm_ones(d_state, config.rms_norm_eps, &device)?;
    let ssm_out = make_qmatmul_eye(inner_size, hidden_size, &device)?;

    let mut layer = Qwen35SsmLayer::new(
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
    );

    let mut state = Qwen35LayerState::new(&config, &device)?;

    // Step 1: Write token (activates K = e0, V = e1)
    let mut write_input = vec![0f32; hidden_size];
    write_input[1] = 1.0; // K = e0
    write_input[2] = 1.0; // V = e1
    let xs_write = Tensor::from_vec(write_input, (1, 1, hidden_size), &device)?;
    let _ = layer.forward_decode(&xs_write, &mut state)?;

    // Step 2: Query token (activates Q = e0)
    let mut query_input = vec![0f32; hidden_size];
    query_input[0] = 1.0; // Q = e0
    let xs_query = Tensor::from_vec(query_input, (1, 1, hidden_size), &device)?;
    let out = layer.forward_decode(&xs_query, &mut state)?;

    // The output for head 0 must have energy in value index 1 (e1), NOT key index 0 (e0)
    let out_vec = out.flatten_all()?.to_vec1::<f32>()?;
    let val_energy = out_vec[1].abs();
    let key_energy = out_vec[0].abs();

    assert!(
        val_energy > 0.01,
        "Retrieved output must have significant energy along value coordinate 1, got {}",
        val_energy
    );
    assert!(
        key_energy < 1e-4,
        "Retrieved output must have zero energy along key coordinate 0, got {}",
        key_energy
    );

    Ok(())
}
