use candle::{
    quantized::{gguf_file::{self, Value}, GgmlDType, QMatMul, QTensor},
    DType, Device, Result, Tensor,
};
use candle_speculative_server::{
    kv_cache::InPlaceKvCache,
    model::precompute_freqs_cis,
    qwen35_attn::Qwen35AttnLayer,
    Qwen35Config,
};
use candle_transformers::quantized_nn::RmsNorm;

fn make_rmsnorm_ones(dim: usize, eps: f64, device: &Device) -> Result<RmsNorm> {
    let t = Tensor::ones(dim, DType::F32, device)?;
    let q = QTensor::quantize(&t, GgmlDType::F32)?;
    RmsNorm::from_qtensor(q, eps)
}

fn make_qmatmul_eye(in_dim: usize, out_dim: usize, device: &Device) -> Result<QMatMul> {
    let mut data = vec![0f32; out_dim * in_dim];
    let min_dim = in_dim.min(out_dim);
    for i in 0..min_dim {
        data[i * in_dim + i] = 1.0;
    }
    let t = Tensor::from_vec(data, (out_dim, in_dim), device)?;
    Ok(QMatMul::Tensor(t))
}


fn create_test_attn_layer(
    config: &Qwen35Config,
    kv_capacity: usize,
    device: &Device,
) -> Result<Qwen35AttnLayer> {
    let hidden_size = config.hidden_size; // 5120
    let q_dim = config.num_attention_heads * config.head_dim; // 24 * 256 = 6144
    let q_gate_dim = q_dim * 2; // 12288
    let kv_dim = config.num_key_value_heads * config.head_dim; // 4 * 256 = 1024
    let head_dim = config.head_dim; // 256

    let attn_norm = make_rmsnorm_ones(hidden_size, config.rms_norm_eps, device)?;
    let attn_q = make_qmatmul_eye(hidden_size, q_gate_dim, device)?;
    let attn_k = make_qmatmul_eye(hidden_size, kv_dim, device)?;
    let attn_v = make_qmatmul_eye(hidden_size, kv_dim, device)?;
    let attn_q_norm = make_rmsnorm_ones(head_dim, config.rms_norm_eps, device)?;
    let attn_k_norm = make_rmsnorm_ones(head_dim, config.rms_norm_eps, device)?;
    let attn_output = make_qmatmul_eye(q_dim, hidden_size, device)?;

    let kv_cache = InPlaceKvCache::new(
        1,
        config.num_key_value_heads,
        config.head_dim,
        kv_capacity,
        DType::F32,
        device,
    )?;

    Ok(Qwen35AttnLayer::new(
        attn_norm,
        attn_q,
        attn_k,
        attn_v,
        attn_q_norm,
        attn_k_norm,
        attn_output,
        kv_cache,
        config.clone(),
    ))
}

fn write_gguf<W: std::io::Seek + std::io::Write>(
    w: &mut W,
    metadata: &[(&str, Value)],
    tensors: &[(&str, &QTensor)],
) -> Result<()> {
    let md_refs: Vec<(&str, &Value)> = metadata.iter().map(|(k, v)| (*k, v)).collect();
    gguf_file::write(w, &md_refs, tensors)
}

#[test]
fn test_qwen35_attn_layer_dimensions_and_initialization() -> Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();
    let layer = create_test_attn_layer(&config, 1024, &device)?;

    assert_eq!(layer.config.hidden_size, 5120);
    assert_eq!(layer.config.num_attention_heads, 24);
    assert_eq!(layer.config.num_key_value_heads, 4);
    assert_eq!(layer.config.head_dim, 256);
    assert_eq!(layer.kv_cache.b_sz(), 1);
    assert_eq!(layer.kv_cache.n_kv_head(), 4);
    assert_eq!(layer.kv_cache.head_dim(), 256);
    assert_eq!(layer.kv_cache.current_pos(), 0);

    Ok(())
}

#[test]
fn test_qwen35_attn_fused_q_gate_and_forward_single_token() -> Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();
    let mut layer = create_test_attn_layer(&config, 1024, &device)?;

    let (cos, sin) = precompute_freqs_cis(
        config.head_dim,
        config.rope_theta as f32,
        1024,
        &device,
    )?;

    // Step 0: Input token [1, 1, 5120]
    let xs0 = Tensor::ones((1, 1, 5120), DType::F32, &device)?;
    let out0 = layer.forward(&xs0, &cos, &sin, 0)?;

    assert_eq!(out0.dims3()?, (1, 1, 5120));
    assert_eq!(layer.kv_cache.current_pos(), 1);

    // Verify output is non-zero and finite
    let out0_vec = out0.flatten_all()?.to_vec1::<f32>()?;
    let non_zero_count = out0_vec.iter().filter(|&&v| v.abs() > 1e-6).count();
    assert!(non_zero_count > 0, "output should have non-zero elements");
    for &v in &out0_vec {
        assert!(v.is_finite(), "output elements must be finite");
    }

    // Step 1: Second token [1, 1, 5120] at pos 1
    let xs1 = Tensor::full(2.0f32, (1, 1, 5120), &device)?;
    let out1 = layer.forward(&xs1, &cos, &sin, 1)?;

    assert_eq!(out1.dims3()?, (1, 1, 5120));
    assert_eq!(layer.kv_cache.current_pos(), 2);

    Ok(())
}

#[test]
fn test_qwen35_attn_gate_suppression() -> Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();
    let hidden_size = config.hidden_size;
    let q_dim = config.num_attention_heads * config.head_dim; // 6144
    let q_gate_dim = q_dim * 2; // 12288
    let kv_dim = config.num_key_value_heads * config.head_dim; // 1024
    let head_dim = config.head_dim; // 256

    let attn_norm = make_rmsnorm_ones(hidden_size, config.rms_norm_eps, &device)?;

    // Set gate portion (rows 6144..12288) to large negative bias
    let mut q_gate_weights = vec![0f32; q_gate_dim * hidden_size];
    // First 6144 rows: identity for Q
    for i in 0..q_dim.min(hidden_size) {
        q_gate_weights[i * hidden_size + i] = 1.0;
    }
    // Rows 6144..12288: set constant negative values so gate is ~ -50.0
    for i in q_dim..q_gate_dim {
        for j in 0..hidden_size {
            q_gate_weights[i * hidden_size + j] = -0.01;
        }
    }
    let attn_q = QMatMul::Tensor(Tensor::from_vec(
        q_gate_weights,
        (q_gate_dim, hidden_size),
        &device,
    )?);

    let attn_k = make_qmatmul_eye(hidden_size, kv_dim, &device)?;
    let attn_v = make_qmatmul_eye(hidden_size, kv_dim, &device)?;
    let attn_q_norm = make_rmsnorm_ones(head_dim, config.rms_norm_eps, &device)?;
    let attn_k_norm = make_rmsnorm_ones(head_dim, config.rms_norm_eps, &device)?;
    let attn_output = make_qmatmul_eye(q_dim, hidden_size, &device)?;

    let kv_cache = InPlaceKvCache::new(
        1,
        config.num_key_value_heads,
        config.head_dim,
        1024,
        DType::F32,
        &device,
    )?;

    let mut layer = Qwen35AttnLayer::new(
        attn_norm,
        attn_q,
        attn_k,
        attn_v,
        attn_q_norm,
        attn_k_norm,
        attn_output,
        kv_cache,
        config.clone(),
    );

    let (cos, sin) = precompute_freqs_cis(
        config.head_dim,
        config.rope_theta as f32,
        1024,
        &device,
    )?;

    // Input with ones: gate projection will be ~ -0.01 * 5120 = -51.2
    // sigmoid(-51.2) ≈ 0.0, output should be practically zero
    let xs = Tensor::ones((1, 1, 5120), DType::F32, &device)?;
    let out = layer.forward(&xs, &cos, &sin, 0)?;

    let max_abs = out.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(
        max_abs < 1e-4,
        "strongly negative gate must suppress attention output, got max_abs={max_abs}"
    );

    Ok(())
}

#[test]
fn test_qwen35_attn_multi_token_prefill_causal_mask() -> Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();
    let mut layer = create_test_attn_layer(&config, 1024, &device)?;

    let (cos, sin) = precompute_freqs_cis(
        config.head_dim,
        config.rope_theta as f32,
        1024,
        &device,
    )?;

    // Prefill 4 tokens at once
    let xs = Tensor::randn(0f32, 1f32, (1, 4, 5120), &device)?;
    let out = layer.forward(&xs, &cos, &sin, 0)?;

    assert_eq!(out.dims3()?, (1, 4, 5120));
    assert_eq!(layer.kv_cache.current_pos(), 4);

    // Follow with single token decode at pos 4
    let xs_next = Tensor::randn(0f32, 1f32, (1, 1, 5120), &device)?;
    let out_next = layer.forward(&xs_next, &cos, &sin, 4)?;

    assert_eq!(out_next.dims3()?, (1, 1, 5120));
    assert_eq!(layer.kv_cache.current_pos(), 5);

    Ok(())
}

#[test]
fn test_qwen35_attn_rolling_window() -> Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();
    let mut layer = create_test_attn_layer(&config, 1024, &device)?;

    let (cos, sin) = precompute_freqs_cis(
        config.head_dim,
        config.rope_theta as f32,
        1024,
        &device,
    )?;

    let rolling_window = 4;

    // Push 6 tokens with rolling window of 4
    for pos in 0..6 {
        let xs = Tensor::randn(0f32, 1f32, (1, 1, 5120), &device)?;
        let _ = layer.forward_with_rolling(&xs, &cos, &sin, pos, Some(rolling_window))?;
    }

    // After 6 tokens with window 4, KV cache position should be capped at 4
    assert_eq!(layer.kv_cache.current_pos(), 4);

    Ok(())
}

#[test]
fn test_qwen35_attn_2d_input_support() -> Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();
    let mut layer = create_test_attn_layer(&config, 1024, &device)?;

    let (cos, sin) = precompute_freqs_cis(
        config.head_dim,
        config.rope_theta as f32,
        1024,
        &device,
    )?;

    // 2D input [1, 5120]
    let xs_2d = Tensor::ones((1, 5120), DType::F32, &device)?;
    let out = layer.forward(&xs_2d, &cos, &sin, 0)?;

    assert_eq!(out.dims2()?, (1, 5120));
    assert_eq!(layer.kv_cache.current_pos(), 1);

    Ok(())
}

#[test]
fn test_qwen35_attn_from_gguf_loader() -> Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();
    let hidden_size = config.hidden_size; // 5120
    let q_dim = config.num_attention_heads * config.head_dim; // 6144
    let q_gate_dim = q_dim * 2; // 12288
    let kv_dim = config.num_key_value_heads * config.head_dim; // 1024
    let head_dim = config.head_dim; // 256

    let t_norm = Tensor::ones(hidden_size, DType::F32, &device)?;
    let q_norm = QTensor::quantize(&t_norm, GgmlDType::F32)?;

    let t_q = Tensor::zeros((q_gate_dim, hidden_size), DType::F32, &device)?;
    let q_q = QTensor::quantize(&t_q, GgmlDType::F32)?;

    let t_k = Tensor::zeros((kv_dim, hidden_size), DType::F32, &device)?;
    let q_k = QTensor::quantize(&t_k, GgmlDType::F32)?;

    let t_v = Tensor::zeros((kv_dim, hidden_size), DType::F32, &device)?;
    let q_v = QTensor::quantize(&t_v, GgmlDType::F32)?;

    let t_q_norm = Tensor::ones(head_dim, DType::F32, &device)?;
    let q_q_norm = QTensor::quantize(&t_q_norm, GgmlDType::F32)?;

    let t_k_norm = Tensor::ones(head_dim, DType::F32, &device)?;
    let q_k_norm = QTensor::quantize(&t_k_norm, GgmlDType::F32)?;

    let t_output = Tensor::zeros((hidden_size, q_dim), DType::F32, &device)?;
    let q_output = QTensor::quantize(&t_output, GgmlDType::F32)?;

    let mut cursor = std::io::Cursor::new(Vec::new());
    let metadata = vec![
        ("general.architecture", Value::String("qwen35".to_string())),
    ];
    let tensors: Vec<(&str, &QTensor)> = vec![
        ("blk.3.attn_norm.weight", &q_norm),
        ("blk.3.attn_q.weight", &q_q),
        ("blk.3.attn_k.weight", &q_k),
        ("blk.3.attn_v.weight", &q_v),
        ("blk.3.attn_q_norm.weight", &q_q_norm),
        ("blk.3.attn_k_norm.weight", &q_k_norm),
        ("blk.3.attn_output.weight", &q_output),
    ];

    write_gguf(&mut cursor, &metadata, &tensors)?;
    cursor.set_position(0);

    let content = gguf_file::Content::read(&mut cursor)?;
    let layer = Qwen35AttnLayer::from_gguf(&content, &mut cursor, 3, &config, &device)?;

    assert_eq!(layer.config.hidden_size, 5120);
    assert_eq!(layer.config.num_attention_heads, 24);
    assert_eq!(layer.config.num_key_value_heads, 4);
    assert_eq!(layer.kv_cache.n_kv_head(), 4);
    assert_eq!(layer.kv_cache.head_dim(), 256);
    assert_eq!(layer.kv_cache.current_pos(), 0);

    Ok(())
}

#[test]
fn test_qwen35_attn_kv_cache_rollback_and_reset() -> Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();
    let mut layer = create_test_attn_layer(&config, 1024, &device)?;

    let (cos, sin) = precompute_freqs_cis(
        config.head_dim,
        config.rope_theta as f32,
        1024,
        &device,
    )?;

    // Push 5 tokens sequentially
    for pos in 0..5 {
        let xs = Tensor::randn(0f32, 1f32, (1, 1, 5120), &device)?;
        let _ = layer.forward(&xs, &cos, &sin, pos)?;
    }
    assert_eq!(layer.kv_cache.current_pos(), 5);

    // Rollback to pos 2 (speculative rejection of 3 tokens)
    layer.rollback_kv_cache(2)?;
    assert_eq!(layer.kv_cache.current_pos(), 2);

    // Decode new token at pos 2
    let xs_new = Tensor::randn(0f32, 1f32, (1, 1, 5120), &device)?;
    let _ = layer.forward(&xs_new, &cos, &sin, 2)?;
    assert_eq!(layer.kv_cache.current_pos(), 3);

    // Reset back to 0
    layer.reset_kv_cache();
    assert_eq!(layer.kv_cache.current_pos(), 0);

    Ok(())
}

