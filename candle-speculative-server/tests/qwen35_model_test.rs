use candle::{
    quantized::{gguf_file::{self, Value}, GgmlDType, QMatMul, QTensor},
    DType, Device, Result, Tensor,
};
use candle_transformers::quantized_nn::RmsNorm;
use candle_speculative_server::{
    engine::SuperDraftSpeculativeEngine,
    kv_cache::InPlaceKvCache,
    model::{precompute_freqs_cis, Bonsai27BWithKv, BonsaiBackend, Config as Qwen2Config, Layer as Qwen2Layer, QuantizedQwen2WithKv},
    qwen35_attn::Qwen35AttnLayer,
    qwen35_model::{Qwen35Block, Qwen35Model, SwiGluFfn},
    qwen35_ssm::Qwen35SsmLayer,
    qwen35_state::{Qwen35Config, Qwen35RecurrentState},
};

fn create_test_config() -> Qwen35Config {
    Qwen35Config {
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 4,
        full_attn_interval: 4,
        ssm_conv_kernel: 4,
        ssm_d_state: 16,
        ssm_n_group: 2,
        ssm_dt_rank: 4,
        ssm_inner_size: 64,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        vocab_size: 128,
        rope_theta: 10_000_000.0,
        rms_norm_eps: 1e-6,
    }
}

fn make_rmsnorm_ones(dim: usize, eps: f64, device: &Device) -> Result<RmsNorm> {
    let t = Tensor::ones(dim, DType::F32, device)?;
    let q = QTensor::quantize(&t, GgmlDType::F32)?;
    RmsNorm::from_qtensor(q, eps)
}

fn create_mock_ffn(config: &Qwen35Config, device: &Device) -> Result<SwiGluFfn> {
    let gate_w = Tensor::zeros((config.intermediate_size, config.hidden_size), DType::F32, device)?;
    let up_w = Tensor::zeros((config.intermediate_size, config.hidden_size), DType::F32, device)?;
    let down_w = Tensor::zeros((config.hidden_size, config.intermediate_size), DType::F32, device)?;
    let post_attention_norm = make_rmsnorm_ones(config.hidden_size, config.rms_norm_eps, device)?;

    let ffn_gate = QMatMul::Tensor(gate_w);
    let ffn_up = QMatMul::Tensor(up_w);
    let ffn_down = QMatMul::Tensor(down_w);

    Ok(SwiGluFfn::new(ffn_gate, ffn_up, ffn_down, post_attention_norm))
}

fn create_mock_ssm_layer(config: &Qwen35Config, device: &Device) -> Result<Qwen35SsmLayer> {
    let attn_norm = make_rmsnorm_ones(config.hidden_size, config.rms_norm_eps, device)?;
    let attn_qkv = QMatMul::Tensor(Tensor::zeros((config.ssm_conv_dim(), config.hidden_size), DType::F32, device)?);
    let attn_gate = QMatMul::Tensor(Tensor::zeros((config.ssm_inner_size, config.hidden_size), DType::F32, device)?);
    let ssm_conv1d = Tensor::zeros((config.ssm_conv_dim(), config.ssm_conv_kernel), DType::F32, device)?;
    let ssm_a = Tensor::zeros(config.ssm_dt_rank, DType::F32, device)?;
    let ssm_alpha = QMatMul::Tensor(Tensor::zeros((config.ssm_dt_rank, config.hidden_size), DType::F32, device)?);
    let ssm_beta = QMatMul::Tensor(Tensor::zeros((config.ssm_dt_rank, config.hidden_size), DType::F32, device)?);
    let ssm_dt = Tensor::zeros(config.ssm_dt_rank, DType::F32, device)?;
    let ssm_norm = make_rmsnorm_ones(config.ssm_d_state, config.rms_norm_eps, device)?;
    let ssm_out = QMatMul::Tensor(Tensor::zeros((config.hidden_size, config.ssm_inner_size), DType::F32, device)?);

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

fn create_mock_attn_layer(config: &Qwen35Config, max_seq_len: usize, device: &Device) -> Result<Qwen35AttnLayer> {
    let attn_norm = make_rmsnorm_ones(config.hidden_size, config.rms_norm_eps, device)?;
    let attn_q = QMatMul::Tensor(Tensor::zeros((config.q_gate_dim(), config.hidden_size), DType::F32, device)?);
    let attn_k = QMatMul::Tensor(Tensor::zeros((config.kv_dim(), config.hidden_size), DType::F32, device)?);
    let attn_v = QMatMul::Tensor(Tensor::zeros((config.kv_dim(), config.hidden_size), DType::F32, device)?);
    let attn_q_norm = make_rmsnorm_ones(config.head_dim, config.rms_norm_eps, device)?;
    let attn_k_norm = make_rmsnorm_ones(config.head_dim, config.rms_norm_eps, device)?;
    let attn_output = QMatMul::Tensor(Tensor::zeros((config.hidden_size, config.q_dim()), DType::F32, device)?);
    let kv_cache = InPlaceKvCache::new(
        1,
        config.num_key_value_heads,
        config.head_dim,
        max_seq_len,
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

fn create_test_model(config: &Qwen35Config, device: &Device) -> Result<Qwen35Model> {
    let max_seq_len = 64;
    let embed_w = Tensor::zeros((config.vocab_size, config.hidden_size), DType::F32, device)?;
    let tok_embeddings = candle_nn::Embedding::new(embed_w, config.hidden_size);

    let output_norm = make_rmsnorm_ones(config.hidden_size, config.rms_norm_eps, device)?;
    let output = QMatMul::Tensor(Tensor::zeros((config.vocab_size, config.hidden_size), DType::F32, device)?);

    let (cos, sin) = precompute_freqs_cis(config.head_dim, config.rope_theta as f32, max_seq_len, device)?;
    let recurrent_state = Qwen35RecurrentState::new(config, device)?;

    let mut blocks = Vec::with_capacity(config.num_hidden_layers);
    for layer_idx in 0..config.num_hidden_layers {
        let ffn = create_mock_ffn(config, device)?;
        if config.is_full_attn(layer_idx) {
            let layer = create_mock_attn_layer(config, max_seq_len, device)?;
            blocks.push(Qwen35Block::Attn { layer, ffn });
        } else {
            let layer = create_mock_ssm_layer(config, device)?;
            blocks.push(Qwen35Block::Ssm { layer, ffn });
        }
    }

    Ok(Qwen35Model {
        tok_embeddings,
        blocks,
        output_norm,
        output,
        recurrent_state,
        cos,
        sin,
        total_tokens_seen: 0,
        device: device.clone(),
        config: config.clone(),
        state_snapshots: Vec::new(),
    })
}

#[test]
fn test_qwen35_config_bonsai_27b_layer_interleaving() {
    let config = Qwen35Config::bonsai_27b();
    assert_eq!(config.num_hidden_layers, 64);
    assert_eq!(config.num_ssm_layers(), 48);
    assert_eq!(config.num_full_attn_layers(), 16);

    let mut ssm_count = 0;
    let mut attn_count = 0;
    for i in 0..64 {
        if (i + 1) % 4 == 0 {
            assert!(config.is_full_attn(i), "layer {i} should be full attention");
            assert!(!config.is_ssm(i), "layer {i} should not be ssm");
            assert_eq!(config.ssm_layer_index(i), None);
            attn_count += 1;
        } else {
            assert!(config.is_ssm(i), "layer {i} should be ssm");
            assert!(!config.is_full_attn(i), "layer {i} should not be full attention");
            assert!(config.ssm_layer_index(i).is_some());
            ssm_count += 1;
        }
    }
    assert_eq!(ssm_count, 48);
    assert_eq!(attn_count, 16);
}

#[test]
fn test_qwen35_model_construction_and_interleaved_blocks() -> Result<()> {
    let device = Device::Cpu;
    let config = create_test_config();
    let model = create_test_model(&config, &device)?;

    assert_eq!(model.blocks.len(), 4);
    // Layers 0, 1, 2 should be SSM, layer 3 should be Attn
    assert!(model.blocks[0].is_ssm());
    assert!(model.blocks[1].is_ssm());
    assert!(model.blocks[2].is_ssm());
    assert!(model.blocks[3].is_attn());

    assert_eq!(model.recurrent_state.len(), 3);
    assert_eq!(model.current_kv_pos(), 0);
    assert_eq!(model.kv_buffer_len(), 0);
    Ok(())
}

#[test]
fn test_qwen35_model_forward_single_token_logits_shape() -> Result<()> {
    let device = Device::Cpu;
    let config = create_test_config();
    let mut model = create_test_model(&config, &device)?;

    let input = Tensor::new(&[[42u32]], &device)?;
    let logits = model.forward(&input)?;

    assert_eq!(logits.dims(), &[1, 1, config.vocab_size]);
    assert_eq!(model.current_kv_pos(), 1);
    assert_eq!(model.kv_buffer_len(), 1);

    // Forward second token
    let input2 = Tensor::new(&[[7u32]], &device)?;
    let logits2 = model.forward(&input2)?;

    assert_eq!(logits2.dims(), &[1, 1, config.vocab_size]);
    assert_eq!(model.current_kv_pos(), 2);
    assert_eq!(model.kv_buffer_len(), 2);
    Ok(())
}

#[test]
fn test_qwen35_model_forward_multi_token_prefill() -> Result<()> {
    let device = Device::Cpu;
    let config = create_test_config();
    let mut model = create_test_model(&config, &device)?;

    let input = Tensor::new(&[[1u32, 2u32, 3u32, 4u32]], &device)?;
    let logits = model.forward(&input)?;

    assert_eq!(logits.dims(), &[1, 4, config.vocab_size]);
    assert_eq!(model.current_kv_pos(), 4);
    assert_eq!(model.kv_buffer_len(), 4);
    Ok(())
}

#[test]
fn test_qwen35_model_speculative_rollback_both_attn_and_recurrent() -> Result<()> {
    let device = Device::Cpu;
    let config = create_test_config();
    let mut model = create_test_model(&config, &device)?;

    // Advance 4 tokens
    for tok in [10u32, 20u32, 30u32, 40u32] {
        let input = Tensor::new(&[[tok]], &device)?;
        let _ = model.forward(&input)?;
    }
    assert_eq!(model.current_kv_pos(), 4);
    assert_eq!(model.kv_buffer_len(), 4);

    // Rollback to position 2 (e.g. divergence at draft token index 1)
    model.rollback_kv(2)?;
    assert_eq!(model.current_kv_pos(), 2);
    assert_eq!(model.kv_buffer_len(), 2);

    // Rollback to 0 (clean reset to initial state)
    model.rollback_kv(0)?;
    assert_eq!(model.current_kv_pos(), 0);
    assert_eq!(model.kv_buffer_len(), 0);

    // After reset, model can step again cleanly
    let input = Tensor::new(&[[99u32]], &device)?;
    let logits = model.forward(&input)?;
    assert_eq!(logits.dims(), &[1, 1, config.vocab_size]);
    assert_eq!(model.current_kv_pos(), 1);
    assert_eq!(model.kv_buffer_len(), 1);

    Ok(())
}

#[test]
fn test_bonsai_backend_dispatch_and_unified_methods() -> Result<()> {
    let device = Device::Cpu;
    let config = create_test_config();
    let model = create_test_model(&config, &device)?;

    let mut bonsai = Bonsai27BWithKv::new_qwen35(model, 16);
    assert_eq!(bonsai.rolling_window(), 16);
    assert_eq!(bonsai.current_kv_pos(), 0);
    assert_eq!(bonsai.kv_buffer_len(), 0);
    assert!(matches!(bonsai.backend, BonsaiBackend::Qwen35(_)));

    let input = Tensor::new(&[[12u32]], &device)?;
    let logits = bonsai.forward(&input)?;
    assert_eq!(logits.dims(), &[1, 1, config.vocab_size]);
    assert_eq!(bonsai.current_kv_pos(), 1);
    assert_eq!(bonsai.kv_buffer_len(), 1);

    bonsai.rollback_kv(0)?;
    assert_eq!(bonsai.current_kv_pos(), 0);
    assert_eq!(bonsai.kv_buffer_len(), 0);

    bonsai.reset_kv();
    assert_eq!(bonsai.current_kv_pos(), 0);

    Ok(())
}

#[test]
fn test_bonsai_from_gguf_dispatch_qwen35_vs_qwen2() -> Result<()> {
    let device = Device::Cpu;
    let hidden_size = 16;
    let intermediate_size = 32;
    let vocab_size = 32;
    let num_layers = 1;
    let conv_kernel = 4;
    let d_state = 8;
    let n_group = 1;
    let dt_rank = 2;
    let inner_size = 16;
    let conv_dim = inner_size + 2 * n_group * d_state; // 16 + 16 = 32

    let mut cursor = std::io::Cursor::new(Vec::new());
    let metadata = vec![
        ("general.architecture", Value::String("qwen35".to_string())),
        ("qwen35.embedding_length", Value::U32(hidden_size as u32)),
        ("qwen35.feed_forward_length", Value::U32(intermediate_size as u32)),
        ("qwen35.vocab_size", Value::U32(vocab_size as u32)),
        ("qwen35.block_count", Value::U32(num_layers as u32)),
        ("qwen35.attention.head_count", Value::U32(2)),
        ("qwen35.attention.head_count_kv", Value::U32(1)),
        ("qwen35.ssm_conv_kernel", Value::U32(conv_kernel as u32)),
        ("qwen35.ssm_d_state", Value::U32(d_state as u32)),
        ("qwen35.ssm_n_group", Value::U32(n_group as u32)),
        ("qwen35.ssm_dt_rank", Value::U32(dt_rank as u32)),
        ("qwen35.ssm_inner_size", Value::U32(inner_size as u32)),
        ("qwen35.context_length", Value::U32(64)),
        ("qwen35.rope.freq_base", Value::F32(10_000_000.0)),
        ("qwen35.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
    ];

    let t_embed = QTensor::quantize(&Tensor::zeros((vocab_size, hidden_size), DType::F32, &device)?, GgmlDType::F32)?;
    let t_out_norm = QTensor::quantize(&Tensor::ones(hidden_size, DType::F32, &device)?, GgmlDType::F32)?;
    let t_out = QTensor::quantize(&Tensor::zeros((vocab_size, hidden_size), DType::F32, &device)?, GgmlDType::F32)?;

    // SSM Layer 0 tensors
    let t_attn_norm = QTensor::quantize(&Tensor::ones(hidden_size, DType::F32, &device)?, GgmlDType::F32)?;
    let t_attn_qkv = QTensor::quantize(&Tensor::zeros((conv_dim, hidden_size), DType::F32, &device)?, GgmlDType::F32)?;
    let t_attn_gate = QTensor::quantize(&Tensor::zeros((inner_size, hidden_size), DType::F32, &device)?, GgmlDType::F32)?;
    let t_ssm_conv1d = QTensor::quantize(&Tensor::zeros((conv_dim, conv_kernel), DType::F32, &device)?, GgmlDType::F32)?;
    let t_ssm_a = QTensor::quantize(&Tensor::zeros(dt_rank, DType::F32, &device)?, GgmlDType::F32)?;
    let t_ssm_alpha = QTensor::quantize(&Tensor::zeros((dt_rank, hidden_size), DType::F32, &device)?, GgmlDType::F32)?;
    let t_ssm_beta = QTensor::quantize(&Tensor::zeros((dt_rank, hidden_size), DType::F32, &device)?, GgmlDType::F32)?;
    let t_ssm_dt = QTensor::quantize(&Tensor::zeros(dt_rank, DType::F32, &device)?, GgmlDType::F32)?;
    let t_ssm_norm = QTensor::quantize(&Tensor::ones(d_state, DType::F32, &device)?, GgmlDType::F32)?;
    let t_ssm_out = QTensor::quantize(&Tensor::zeros((hidden_size, inner_size), DType::F32, &device)?, GgmlDType::F32)?;

    // FFN tensors
    let t_gate = QTensor::quantize(&Tensor::zeros((intermediate_size, hidden_size), DType::F32, &device)?, GgmlDType::F32)?;
    let t_down = QTensor::quantize(&Tensor::zeros((hidden_size, intermediate_size), DType::F32, &device)?, GgmlDType::F32)?;
    let t_up = QTensor::quantize(&Tensor::zeros((intermediate_size, hidden_size), DType::F32, &device)?, GgmlDType::F32)?;
    let t_ffn_norm = QTensor::quantize(&Tensor::ones(hidden_size, DType::F32, &device)?, GgmlDType::F32)?;

    let tensors: Vec<(&str, &QTensor)> = vec![
        ("token_embd.weight", &t_embed),
        ("output_norm.weight", &t_out_norm),
        ("output.weight", &t_out),
        ("blk.0.attn_norm.weight", &t_attn_norm),
        ("blk.0.attn_qkv.weight", &t_attn_qkv),
        ("blk.0.attn_gate.weight", &t_attn_gate),
        ("blk.0.ssm_conv1d.weight", &t_ssm_conv1d),
        ("blk.0.ssm_a", &t_ssm_a),
        ("blk.0.ssm_alpha.weight", &t_ssm_alpha),
        ("blk.0.ssm_beta.weight", &t_ssm_beta),
        ("blk.0.ssm_dt.bias", &t_ssm_dt),
        ("blk.0.ssm_norm.weight", &t_ssm_norm),
        ("blk.0.ssm_out.weight", &t_ssm_out),
        ("blk.0.ffn_gate.weight", &t_gate),
        ("blk.0.ffn_down.weight", &t_down),
        ("blk.0.ffn_up.weight", &t_up),
        ("blk.0.ffn_norm.weight", &t_ffn_norm),
    ];

    let md_refs: Vec<(&str, &Value)> = metadata.iter().map(|(k, v)| (*k, v)).collect();
    gguf_file::write(&mut cursor, &md_refs, &tensors)?;
    cursor.set_position(0);

    let content = gguf_file::Content::read(&mut cursor)?;
    let bonsai = Bonsai27BWithKv::from_gguf_with_window(&content, &mut cursor, 64, &device)?;

    assert!(matches!(bonsai.backend, BonsaiBackend::Qwen35(_)));
    assert_eq!(bonsai.rolling_window(), 64);
    assert_eq!(bonsai.current_kv_pos(), 0);

    Ok(())
}

#[test]
fn test_superdraft_engine_with_qwen35_draft() -> Result<()> {
    let device = Device::Cpu;
    let config = create_test_config();
    let draft_model = create_test_model(&config, &device)?;
    let draft_bonsai = Bonsai27BWithKv::new_qwen35(draft_model, 16);

    // Create small QuantizedQwen2WithKv target
    let head_dim = config.head_dim;
    let dummy_w = Tensor::zeros((config.hidden_size, config.hidden_size), DType::F32, &device)?;
    let dummy_q = QMatMul::Tensor(dummy_w.clone());
    let dummy_kv_w = Tensor::zeros((config.num_key_value_heads * head_dim, config.hidden_size), DType::F32, &device)?;
    let dummy_kv_q = QMatMul::Tensor(dummy_kv_w);
    let dummy_norm = make_rmsnorm_ones(config.hidden_size, 1e-6, &device)?;
    let kv_cache = InPlaceKvCache::new(
        1,
        config.num_key_value_heads,
        head_dim,
        64,
        DType::F32,
        &device,
    )?;

    let target_layer = Qwen2Layer {
        attention_wq: dummy_q.clone(),
        attention_wk: dummy_kv_q.clone(),
        attention_wv: dummy_kv_q,
        attention_wo: dummy_q.clone(),
        attention_norm: dummy_norm.clone(),
        ffn_gate: dummy_q.clone(),
        ffn_down: dummy_q.clone(),
        ffn_up: dummy_q,
        ffn_norm: dummy_norm.clone(),
        kv_cache,
        n_head: config.num_attention_heads,
        n_kv_head: config.num_key_value_heads,
        head_dim,
    };

    let (cos, sin) = precompute_freqs_cis(head_dim, 1_000_000.0, 64, &device)?;
    let embed_w = Tensor::zeros((config.vocab_size, config.hidden_size), DType::F32, &device)?;
    let tok_embeddings = candle_nn::Embedding::new(embed_w, config.hidden_size);
    let out_w = Tensor::zeros((config.vocab_size, config.hidden_size), DType::F32, &device)?;
    let output = QMatMul::Tensor(out_w);
    let norm = make_rmsnorm_ones(config.hidden_size, 1e-6, &device)?;

    let target_config = Qwen2Config {
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        vocab_size: config.vocab_size,
        num_hidden_layers: 1,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        max_position_embeddings: 64,
    };

    let target_verifier = QuantizedQwen2WithKv {
        tok_embeddings,
        layers: vec![target_layer],
        norm,
        output,
        config: target_config,
        device: device.clone(),
        cos,
        sin,
        total_tokens_seen: 0,
    };

    let mut engine = SuperDraftSpeculativeEngine::new(draft_bonsai, target_verifier, 2);
    let res = engine.speculative_step(1)?;

    assert!(!res.accepted_tokens.is_empty());
    assert_eq!(engine.draft_bonsai.current_kv_pos() >= 1, true);
    assert_eq!(engine.target_verifier.current_kv_pos() >= 1, true);

    Ok(())
}
