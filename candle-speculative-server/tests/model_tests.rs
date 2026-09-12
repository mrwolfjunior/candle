use candle::{DType, Device, Tensor};
use candle_speculative_server::kv_cache::InPlaceKvCache;
use candle_speculative_server::model::{precompute_freqs_cis, Config, Layer, QuantizedQwen2WithKv};

#[test]
fn test_qwen2_config_dimensions() {
    let config = Config::qwen2_5_1_5b();
    assert_eq!(config.hidden_size, 1536);
    assert_eq!(config.num_attention_heads, 12);
    assert_eq!(config.num_key_value_heads, 2);
    assert_eq!(config.head_dim(), 128);

    let config_14b = Config::qwen2_5_14b();
    assert_eq!(config_14b.hidden_size, 5120);
    assert_eq!(config_14b.num_attention_heads, 40);
    assert_eq!(config_14b.num_key_value_heads, 8);
    assert_eq!(config_14b.head_dim(), 128);
}

#[test]
fn test_precompute_freqs_cis() -> candle::Result<()> {
    let device = Device::Cpu;
    let head_dim = 128;
    let context_len = 16;
    let (cos, sin) = precompute_freqs_cis(head_dim, 1_000_000.0, context_len, &device)?;

    assert_eq!(cos.dims(), &[context_len, head_dim / 2]);
    assert_eq!(sin.dims(), &[context_len, head_dim / 2]);
    assert_eq!(cos.dtype(), DType::F32);
    assert_eq!(sin.dtype(), DType::F32);
    Ok(())
}

#[test]
fn test_quantized_qwen2_rollback_and_reset() -> candle::Result<()> {
    let device = Device::Cpu;
    let b_sz = 1;
    let n_kv_head = 2;
    let head_dim = 128;
    let max_seq_len = 32;

    // Construct 2 dummy layers with InPlaceKvCache
    let mut layers = Vec::new();
    for _ in 0..2 {
        let dummy_w = Tensor::zeros((head_dim, head_dim), DType::F32, &device)?;
        let dummy_q = candle::quantized::QMatMul::Tensor(dummy_w);
        let dummy_norm_w = candle::quantized::QTensor::quantize(
            &Tensor::ones(head_dim, DType::F32, &device)?,
            candle::quantized::GgmlDType::F32,
        )?;
        let dummy_norm =
            candle_transformers::quantized_nn::RmsNorm::from_qtensor(dummy_norm_w, 1e-6)?;
        let kv_cache = InPlaceKvCache::new(
            b_sz,
            n_kv_head,
            head_dim,
            max_seq_len,
            DType::F32,
            &device,
        )?;

        layers.push(Layer {
            attention_wq: dummy_q.clone(),
            attention_wk: dummy_q.clone(),
            attention_wv: dummy_q.clone(),
            attention_wo: dummy_q.clone(),
            attention_norm: dummy_norm.clone(),
            ffn_gate: dummy_q.clone(),
            ffn_down: dummy_q.clone(),
            ffn_up: dummy_q.clone(),
            ffn_norm: dummy_norm,
            kv_cache,
            n_head: 12,
            n_kv_head,
            head_dim,
        });
    }

    // Append 5 tokens into each layer's KV cache
    let k = Tensor::zeros((b_sz, n_kv_head, 5, head_dim), DType::F32, &device)?;
    let v = Tensor::zeros((b_sz, n_kv_head, 5, head_dim), DType::F32, &device)?;
    for layer in &mut layers {
        layer.kv_cache.append(&k, &v)?;
    }

    let (cos, sin) = precompute_freqs_cis(head_dim, 1_000_000.0, max_seq_len, &device)?;
    let dummy_embed_w = Tensor::zeros((100, head_dim), DType::F32, &device)?;
    let tok_embeddings = candle_nn::Embedding::new(dummy_embed_w, head_dim);
    let dummy_out_w = Tensor::zeros((100, head_dim), DType::F32, &device)?;
    let output = candle::quantized::QMatMul::Tensor(dummy_out_w);
    let norm_w = candle::quantized::QTensor::quantize(
        &Tensor::ones(head_dim, DType::F32, &device)?,
        candle::quantized::GgmlDType::F32,
    )?;
    let norm = candle_transformers::quantized_nn::RmsNorm::from_qtensor(norm_w, 1e-6)?;

    let mut model = QuantizedQwen2WithKv {
        tok_embeddings,
        layers,
        norm,
        output,
        config: Config::qwen2_5_1_5b(),
        device,
        cos,
        sin,
    };

    assert_eq!(model.current_kv_pos(), 5);

    // Rollback to pos 3 across all layers
    model.rollback_kv(3)?;
    assert_eq!(model.current_kv_pos(), 3);
    for layer in &model.layers {
        assert_eq!(layer.kv_cache.current_pos(), 3);
    }

    // Invalid rollback should return Error
    assert!(model.rollback_kv(10).is_err());

    // Reset across all layers
    model.reset_kv();
    assert_eq!(model.current_kv_pos(), 0);
    for layer in &model.layers {
        assert_eq!(layer.kv_cache.current_pos(), 0);
    }

    Ok(())
}
