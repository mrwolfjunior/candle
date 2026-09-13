use candle::{
    quantized::{gguf_file::{self, Value}, GgmlDType, QTensor},
    DType, Device, Tensor,
};
use candle_speculative_server::{
    kv_cache::InPlaceKvCache,
    model::{precompute_freqs_cis, Bonsai27BWithKv, Config, Layer, QuantizedQwen2WithKv},
    Bonsai27BWithKv as ExportedBonsai27BWithKv,
};

fn write_gguf<W: std::io::Seek + std::io::Write>(
    w: &mut W,
    metadata: &[(&str, Value)],
    tensors: &[(&str, &QTensor)],
) -> candle::Result<()> {
    let md_refs: Vec<(&str, &Value)> = metadata.iter().map(|(k, v)| (*k, v)).collect();
    gguf_file::write(w, &md_refs, tensors)
}

#[test]
fn test_bonsai_27b_config_dimensions() {
    let config = Config::bonsai_27b();
    assert_eq!(config.hidden_size, 5120);
    assert_eq!(config.intermediate_size, 13824);
    assert_eq!(config.vocab_size, 248320);
    assert_eq!(config.num_hidden_layers, 64);
    assert_eq!(config.num_attention_heads, 40);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.max_position_embeddings, 32768);
    assert_eq!(config.rope_theta, 1_000_000.0);
    assert_eq!(config.head_dim(), 128);
}

#[test]
fn test_bonsai_gguf_metadata_parser_qwen35() -> candle::Result<()> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    let metadata = vec![
        ("general.architecture", Value::String("qwen35".to_string())),
        ("qwen35.embedding_length", Value::U32(5120)),
        ("qwen35.feed_forward_length", Value::U32(13824)),
        ("qwen35.vocab_size", Value::U32(248320)),
        ("qwen35.block_count", Value::U32(64)),
        ("qwen35.attention.head_count", Value::U32(40)),
        ("qwen35.attention.head_count_kv", Value::U32(8)),
        ("qwen35.context_length", Value::U32(32768)),
        ("qwen35.rope.freq_base", Value::F32(1_000_000.0)),
        ("qwen35.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
    ];
    write_gguf(&mut cursor, &metadata, &[])?;
    cursor.set_position(0);

    let content = gguf_file::Content::read(&mut cursor)?;
    let config = Config::from_gguf(&content)?;

    assert_eq!(config.hidden_size, 5120);
    assert_eq!(config.intermediate_size, 13824);
    assert_eq!(config.vocab_size, 248320);
    assert_eq!(config.num_hidden_layers, 64);
    assert_eq!(config.num_attention_heads, 40);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.max_position_embeddings, 32768);
    assert_eq!(config.rope_theta, 1_000_000.0);
    assert_eq!(config.head_dim(), 128);
    Ok(())
}

#[test]
fn test_bonsai_gguf_metadata_parser_qwen2_fallback() -> candle::Result<()> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    let metadata = vec![
        ("general.architecture", Value::String("qwen2".to_string())),
        ("qwen2.embedding_length", Value::U32(5120)),
        ("qwen2.feed_forward_length", Value::U32(13824)),
        ("qwen2.vocab_size", Value::U32(248320)),
        ("qwen2.block_count", Value::U32(64)),
        ("qwen2.attention.head_count", Value::U32(40)),
        ("qwen2.attention.head_count_kv", Value::U32(8)),
        ("qwen2.context_length", Value::U32(32768)),
        ("qwen2.rope.freq_base", Value::F32(1_000_000.0)),
        ("qwen2.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
    ];
    write_gguf(&mut cursor, &metadata, &[])?;
    cursor.set_position(0);

    let content = gguf_file::Content::read(&mut cursor)?;
    let config = Config::from_gguf(&content)?;

    assert_eq!(config.hidden_size, 5120);
    assert_eq!(config.intermediate_size, 13824);
    assert_eq!(config.vocab_size, 248320);
    assert_eq!(config.num_hidden_layers, 64);
    assert_eq!(config.num_attention_heads, 40);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.max_position_embeddings, 32768);
    assert_eq!(config.rope_theta, 1_000_000.0);
    Ok(())
}

#[test]
fn test_bonsai_loader_instantiation_on_device() -> candle::Result<()> {
    let device = Device::Cpu;
    let hidden_size = 16;
    let intermediate_size = 32;
    let vocab_size = 32;
    let num_layers = 1;
    let num_heads = 2;
    let num_kv_heads = 1;

    let mut cursor = std::io::Cursor::new(Vec::new());
    let metadata = vec![
        ("general.architecture", Value::String("qwen35".to_string())),
        ("qwen35.embedding_length", Value::U32(hidden_size as u32)),
        ("qwen35.feed_forward_length", Value::U32(intermediate_size as u32)),
        ("qwen35.vocab_size", Value::U32(vocab_size as u32)),
        ("qwen35.block_count", Value::U32(num_layers as u32)),
        ("qwen35.attention.head_count", Value::U32(num_heads as u32)),
        ("qwen35.attention.head_count_kv", Value::U32(num_kv_heads as u32)),
        ("qwen35.context_length", Value::U32(256)),
        ("qwen35.rope.freq_base", Value::F32(1_000_000.0)),
        ("qwen35.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
    ];

    let t_embed = QTensor::quantize(&Tensor::zeros((vocab_size, hidden_size), DType::F32, &device)?, GgmlDType::F32)?;
    let t_out_norm = QTensor::quantize(&Tensor::ones(hidden_size, DType::F32, &device)?, GgmlDType::F32)?;
    let t_out = QTensor::quantize(&Tensor::zeros((vocab_size, hidden_size), DType::F32, &device)?, GgmlDType::F32)?;

    let t_wq = QTensor::quantize(&Tensor::zeros((hidden_size, hidden_size), DType::F32, &device)?, GgmlDType::F32)?;
    let t_wk = QTensor::quantize(&Tensor::zeros((8, hidden_size), DType::F32, &device)?, GgmlDType::F32)?;
    let t_wv = QTensor::quantize(&Tensor::zeros((8, hidden_size), DType::F32, &device)?, GgmlDType::F32)?;
    let t_wo = QTensor::quantize(&Tensor::zeros((hidden_size, hidden_size), DType::F32, &device)?, GgmlDType::F32)?;
    let t_attn_norm = QTensor::quantize(&Tensor::ones(hidden_size, DType::F32, &device)?, GgmlDType::F32)?;

    let t_gate = QTensor::quantize(&Tensor::zeros((intermediate_size, hidden_size), DType::F32, &device)?, GgmlDType::F32)?;
    let t_down = QTensor::quantize(&Tensor::zeros((hidden_size, intermediate_size), DType::F32, &device)?, GgmlDType::F32)?;
    let t_up = QTensor::quantize(&Tensor::zeros((intermediate_size, hidden_size), DType::F32, &device)?, GgmlDType::F32)?;
    let t_ffn_norm = QTensor::quantize(&Tensor::ones(hidden_size, DType::F32, &device)?, GgmlDType::F32)?;

    let tensors: Vec<(&str, &QTensor)> = vec![
        ("token_embd.weight", &t_embed),
        ("output_norm.weight", &t_out_norm),
        ("output.weight", &t_out),
        ("blk.0.attn_q.weight", &t_wq),
        ("blk.0.attn_k.weight", &t_wk),
        ("blk.0.attn_v.weight", &t_wv),
        ("blk.0.attn_output.weight", &t_wo),
        ("blk.0.attn_norm.weight", &t_attn_norm),
        ("blk.0.ffn_gate.weight", &t_gate),
        ("blk.0.ffn_down.weight", &t_down),
        ("blk.0.ffn_up.weight", &t_up),
        ("blk.0.ffn_norm.weight", &t_ffn_norm),
    ];

    write_gguf(&mut cursor, &metadata, &tensors)?;
    cursor.set_position(0);

    let content = gguf_file::Content::read(&mut cursor)?;
    let bonsai = Bonsai27BWithKv::from_gguf_with_window(&content, &mut cursor, 128, &device)?;

    assert_eq!(bonsai.rolling_window(), 128);
    assert_eq!(bonsai.current_kv_pos(), 0);
    assert_eq!(bonsai.model.layers.len(), 1);
    assert_eq!(bonsai.model.config.hidden_size, 16);
    assert_eq!(bonsai.model.config.intermediate_size, 32);
    Ok(())
}

#[test]
fn test_bonsai_rolling_window_kv_logic() -> candle::Result<()> {
    let device = Device::Cpu;
    let b_sz = 1;
    let n_kv_head = 2;
    let head_dim = 4;
    let rolling_window = 8;

    let dummy_w = Tensor::zeros((head_dim, head_dim), DType::F32, &device)?;
    let dummy_q = candle::quantized::QMatMul::Tensor(dummy_w);
    let dummy_norm_w = candle::quantized::QTensor::quantize(
        &Tensor::ones(head_dim, DType::F32, &device)?,
        candle::quantized::GgmlDType::F32,
    )?;
    let dummy_norm = candle_transformers::quantized_nn::RmsNorm::from_qtensor(dummy_norm_w, 1e-6)?;
    let kv_cache = InPlaceKvCache::new(
        b_sz,
        n_kv_head,
        head_dim,
        rolling_window,
        DType::F32,
        &device,
    )?;

    let layer = Layer {
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
        n_head: 4,
        n_kv_head,
        head_dim,
    };

    let (cos, sin) = precompute_freqs_cis(head_dim, 1_000_000.0, rolling_window, &device)?;
    let dummy_embed_w = Tensor::zeros((10, head_dim), DType::F32, &device)?;
    let tok_embeddings = candle_nn::Embedding::new(dummy_embed_w, head_dim);
    let dummy_out_w = Tensor::zeros((10, head_dim), DType::F32, &device)?;
    let output = candle::quantized::QMatMul::Tensor(dummy_out_w);
    let norm_w = candle::quantized::QTensor::quantize(
        &Tensor::ones(head_dim, DType::F32, &device)?,
        candle::quantized::GgmlDType::F32,
    )?;
    let norm = candle_transformers::quantized_nn::RmsNorm::from_qtensor(norm_w, 1e-6)?;

    let model = QuantizedQwen2WithKv {
        tok_embeddings,
        layers: vec![layer],
        norm,
        output,
        config: Config::bonsai_27b(),
        device: device.clone(),
        cos,
        sin,
        total_tokens_seen: 0,
    };

    let mut bonsai = Bonsai27BWithKv::new(model, rolling_window);
    assert_eq!(bonsai.rolling_window(), 8);
    assert_eq!(bonsai.current_kv_pos(), 0);

    // Helper to generate identifiable tensor with values filled by token index
    let make_kv = |token_start: f32, len: usize| -> candle::Result<(Tensor, Tensor)> {
        let mut data = Vec::with_capacity(b_sz * n_kv_head * len * head_dim);
        for t in 0..len {
            let val = token_start + t as f32;
            for _ in 0..(b_sz * n_kv_head * head_dim) {
                data.push(val);
            }
        }
        // Shape (len, b_sz, n_kv_head, head_dim) -> permute to (b_sz, n_kv_head, len, head_dim)
        let t = Tensor::from_vec(data, (len, b_sz, n_kv_head, head_dim), &device)?;
        let t = t.permute((1, 2, 0, 3))?.contiguous()?;
        Ok((t.clone(), t))
    };

    // Append 5 tokens (values 1..=5)
    let (k1, v1) = make_kv(1.0, 5)?;
    bonsai.append_kv(&k1, &v1)?;
    assert_eq!(bonsai.current_kv_pos(), 5);

    // Append 3 more tokens (values 6..=8) -> reaches capacity 8
    let (k2, v2) = make_kv(6.0, 3)?;
    bonsai.append_kv(&k2, &v2)?;
    assert_eq!(bonsai.current_kv_pos(), 8);

    // Now append 2 more tokens (values 9..=10) -> exceeds rolling window 8!
    // Evicts oldest 2 tokens (1.0, 2.0). Cache should now hold 3.0..=10.0 (8 tokens)
    let (k3, v3) = make_kv(9.0, 2)?;
    bonsai.append_kv(&k3, &v3)?;
    assert_eq!(bonsai.current_kv_pos(), 10);
    assert_eq!(bonsai.kv_buffer_len(), 8);

    // Inspect KV cache contents via view
    let (k_view, _) = bonsai.model.layers[0].kv_cache.current_view()?;
    assert_eq!(k_view.dims(), &[1, 2, 8, 4]);
    // The first token in the rolling cache should now have value 3.0
    let first_token = k_view.narrow(2, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(first_token[0], 3.0);
    // The last token in the rolling cache should have value 10.0
    let last_token = k_view.narrow(2, 7, 1)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(last_token[0], 10.0);

    // Test rollback within window
    // Rolled back from 10 to 6 (discarded 4 tokens: 10, 9, 8, 7).
    // The rolling cache originally held tokens 3..=10 (8 tokens).
    // After discarding the 4 tail tokens, it correctly retains tokens 3..=6 (4 tokens).
    bonsai.rollback_kv(6)?;
    assert_eq!(bonsai.current_kv_pos(), 6);
    assert_eq!(bonsai.kv_buffer_len(), 4);

    // Test reset
    bonsai.reset_kv();
    assert_eq!(bonsai.current_kv_pos(), 0);

    // Verify lib export type
    let _ = std::any::type_name::<ExportedBonsai27BWithKv>();

    Ok(())
}
