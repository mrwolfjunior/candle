use candle::{DType, Device, Tensor};
use candle_speculative_server::{
    engine::{verify_greedy, SuperDraftSpeculativeEngine},
    kv_cache::InPlaceKvCache,
    model::{precompute_freqs_cis, Bonsai27BWithKv, Config, Layer, QuantizedQwen2WithKv},
};

fn create_deterministic_model(
    device: &Device,
    vocab_size: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    max_seq_len: usize,
    next_token_map: &[(u32, u32)],
) -> candle::Result<QuantizedQwen2WithKv> {
    let head_dim = hidden_size / num_heads;

    let dummy_w = Tensor::zeros((hidden_size, hidden_size), DType::F32, device)?;
    let dummy_q = candle::quantized::QMatMul::Tensor(dummy_w.clone());
    let dummy_kv_w = Tensor::zeros((num_kv_heads * head_dim, hidden_size), DType::F32, device)?;
    let dummy_kv_q = candle::quantized::QMatMul::Tensor(dummy_kv_w);

    let dummy_norm_w = candle::quantized::QTensor::quantize(
        &Tensor::ones(hidden_size, DType::F32, device)?,
        candle::quantized::GgmlDType::F32,
    )?;
    let dummy_norm =
        candle_transformers::quantized_nn::RmsNorm::from_qtensor(dummy_norm_w, 1e-6)?;

    let kv_cache = InPlaceKvCache::new(
        1,
        num_kv_heads,
        head_dim,
        max_seq_len,
        DType::F32,
        device,
    )?;

    let layer = Layer {
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
        n_head: num_heads,
        n_kv_head: num_kv_heads,
        head_dim,
    };

    let (cos, sin) = precompute_freqs_cis(head_dim, 1_000_000.0, max_seq_len, device)?;

    // Embeddings: identity matrix so token t maps to one-hot vector t
    let embed_w = Tensor::eye(hidden_size, DType::F32, device)?;
    let tok_embeddings = candle_nn::Embedding::new(embed_w, hidden_size);

    // Output projection: maps one-hot vector t to logits with peak at next_token_map[t]
    let mut out_data = vec![0f32; vocab_size * hidden_size];
    for &(from_tok, to_tok) in next_token_map {
        let from_idx = from_tok as usize;
        let to_idx = to_tok as usize;
        if from_idx < hidden_size && to_idx < vocab_size {
            // output weight has shape (vocab_size, hidden_size).
            // logits = xs.matmul(output^T).
            // row `to_idx`, col `from_idx` in output_weight
            out_data[to_idx * hidden_size + from_idx] = 100.0;
        }
    }
    let out_w = Tensor::from_vec(out_data, (vocab_size, hidden_size), device)?;
    let output = candle::quantized::QMatMul::Tensor(out_w);

    let norm_w = candle::quantized::QTensor::quantize(
        &Tensor::ones(hidden_size, DType::F32, device)?,
        candle::quantized::GgmlDType::F32,
    )?;
    let norm = candle_transformers::quantized_nn::RmsNorm::from_qtensor(norm_w, 1e-6)?;

    let config = Config {
        hidden_size,
        intermediate_size: hidden_size * 2,
        vocab_size,
        num_hidden_layers: 1,
        num_attention_heads: num_heads,
        num_key_value_heads: num_kv_heads,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        max_position_embeddings: max_seq_len,
    };

    Ok(QuantizedQwen2WithKv {
        tok_embeddings,
        layers: vec![layer],
        norm,
        output,
        config,
        device: device.clone(),
        cos,
        sin,
        total_tokens_seen: 0,
    })
}

#[test]
fn test_superdraft_engine_divergence_and_rollback() -> candle::Result<()> {
    let device = Device::Cpu;
    let vocab_size = 16;
    let hidden_size = 16;
    let num_heads = 2;
    let num_kv_heads = 2;
    let max_seq_len = 64;
    let gamma = 4;

    // Draft model transition: 1->2, 2->3, 3->4, 4->5, 5->6
    let draft_transitions = vec![(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)];
    let draft_qwen = create_deterministic_model(
        &device,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        max_seq_len,
        &draft_transitions,
    )?;
    let draft_bonsai = Bonsai27BWithKv::new(draft_qwen, 16);

    // Target model transition: agrees on 1->2, 2->3, but diverges at 3->9 (instead of 3->4)
    let target_transitions = vec![(1, 2), (2, 3), (3, 9), (4, 5), (5, 6)];
    let target_verifier = create_deterministic_model(
        &device,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        max_seq_len,
        &target_transitions,
    )?;

    let mut engine = SuperDraftSpeculativeEngine::new(draft_bonsai, target_verifier, gamma);
    assert_eq!(engine.gamma, 4);
    assert_eq!(engine.draft_bonsai.current_kv_pos(), 0);
    assert_eq!(engine.target_verifier.current_kv_pos(), 0);

    // Run a speculative step starting from token 1:
    // 1. Draft proposes 4 tokens from token 1: [2, 3, 4, 5]
    // 2. Target verifies in parallel batch: [2, 3, 9, 5, 6]
    // 3. Greedy verification:
    //    - index 0: 2 == 2 (accepted)
    //    - index 1: 3 == 3 (accepted)
    //    - index 2: 4 != 9 (divergence at k = 2!)
    //    accepted_tokens: [2, 3, 9], num_accepted_draft: 2, bonus_token: None
    // 4. Rollback: k = 2 < gamma -> rollback both models to pos + k + 1 = 0 + 2 + 1 = 3!
    let result = engine.speculative_step(1)?;

    assert_eq!(result.num_accepted_draft, 2);
    assert_eq!(result.accepted_tokens, vec![2, 3, 9]);
    assert_eq!(result.bonus_token, None);

    // Verify O(1) rollback synchronized both KV caches to position 3
    assert_eq!(engine.draft_bonsai.current_kv_pos(), 3);
    assert_eq!(engine.target_verifier.current_kv_pos(), 3);

    Ok(())
}

#[test]
fn test_superdraft_engine_all_accepted_with_bonus() -> candle::Result<()> {
    let device = Device::Cpu;
    let vocab_size = 16;
    let hidden_size = 16;
    let num_heads = 2;
    let num_kv_heads = 2;
    let max_seq_len = 64;
    let gamma = 4;

    // Both draft and target agree on: 1->2, 2->3, 3->4, 4->5, 5->6
    let transitions = vec![(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)];
    let draft_qwen = create_deterministic_model(
        &device,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        max_seq_len,
        &transitions,
    )?;
    let draft_bonsai = Bonsai27BWithKv::new(draft_qwen, 16);

    let target_verifier = create_deterministic_model(
        &device,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        max_seq_len,
        &transitions,
    )?;

    let mut engine = SuperDraftSpeculativeEngine::new(draft_bonsai, target_verifier, gamma);

    // Run speculative step from token 1:
    // Draft proposes [2, 3, 4, 5]
    // Target produces [2, 3, 4, 5, 6] (all match + bonus token 6)
    let result = engine.speculative_step(1)?;

    assert_eq!(result.num_accepted_draft, 4);
    assert_eq!(result.accepted_tokens, vec![2, 3, 4, 5, 6]);
    assert_eq!(result.bonus_token, Some(6));

    // Both models should be synchronized at position 5 (pos 0 + gamma 4 + 1)
    assert_eq!(engine.draft_bonsai.current_kv_pos(), 5);
    assert_eq!(engine.target_verifier.current_kv_pos(), 5);

    Ok(())
}

#[test]
fn test_superdraft_engine_immediate_divergence() -> candle::Result<()> {
    let device = Device::Cpu;
    let vocab_size = 16;
    let hidden_size = 16;
    let num_heads = 2;
    let num_kv_heads = 2;
    let max_seq_len = 64;
    let gamma = 4;

    let draft_transitions = vec![(1, 2), (2, 3), (3, 4), (4, 5)];
    let draft_qwen = create_deterministic_model(
        &device,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        max_seq_len,
        &draft_transitions,
    )?;
    let draft_bonsai = Bonsai27BWithKv::new(draft_qwen, 16);

    // Target disagrees immediately at token 1: predicts 7 instead of 2
    let target_transitions = vec![(1, 7), (2, 3), (3, 4), (4, 5)];
    let target_verifier = create_deterministic_model(
        &device,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        max_seq_len,
        &target_transitions,
    )?;

    let mut engine = SuperDraftSpeculativeEngine::new(draft_bonsai, target_verifier, gamma);

    let result = engine.speculative_step(1)?;

    assert_eq!(result.num_accepted_draft, 0);
    assert_eq!(result.accepted_tokens, vec![7]);
    assert_eq!(result.bonus_token, None);

    // Both models rolled back to pos + 0 + 1 = 1
    assert_eq!(engine.draft_bonsai.current_kv_pos(), 1);
    assert_eq!(engine.target_verifier.current_kv_pos(), 1);

    Ok(())
}

#[test]
fn test_superdraft_engine_reexport_and_individual_methods() -> candle::Result<()> {
    use candle_speculative_server::SuperDraftSpeculativeEngine as RootExportEngine;

    let device = Device::Cpu;
    let vocab_size = 16;
    let hidden_size = 16;
    let num_heads = 2;
    let num_kv_heads = 2;
    let max_seq_len = 64;
    let gamma = 4;

    let transitions = vec![(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)];
    let draft_qwen = create_deterministic_model(
        &device,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        max_seq_len,
        &transitions,
    )?;
    let draft_bonsai = Bonsai27BWithKv::new(draft_qwen, 16);
    let target_verifier = create_deterministic_model(
        &device,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        max_seq_len,
        &transitions,
    )?;

    let mut engine = RootExportEngine::new(draft_bonsai, target_verifier, gamma);
    assert_eq!(engine.gamma(), 4);

    // Test propose directly
    let draft_tokens = engine.propose(1)?;
    assert_eq!(draft_tokens, vec![2, 3, 4, 5]);
    assert_eq!(engine.draft_bonsai.current_kv_pos(), 4);

    // Test target_forward_batch directly
    let target_argmax = engine.target_forward_batch(1, &draft_tokens)?;
    assert_eq!(target_argmax, vec![2, 3, 4, 5, 6]);
    assert_eq!(engine.target_verifier.current_kv_pos(), 5);

    // Test verify_greedy directly
    let result = verify_greedy(&draft_tokens, &target_argmax);
    assert_eq!(result.num_accepted_draft, 4);
    assert_eq!(result.accepted_tokens, vec![2, 3, 4, 5, 6]);

    // Test rollback_kv directly
    engine.rollback_kv(2)?;
    assert_eq!(engine.draft_bonsai.current_kv_pos(), 2);
    assert_eq!(engine.target_verifier.current_kv_pos(), 2);

    // Test reset_kv directly
    engine.reset_kv();
    assert_eq!(engine.draft_bonsai.current_kv_pos(), 0);
    assert_eq!(engine.target_verifier.current_kv_pos(), 0);

    Ok(())
}

#[test]
fn test_superdraft_engine_prefill() -> candle::Result<()> {
    let device = Device::Cpu;
    let vocab_size = 16;
    let hidden_size = 16;
    let num_heads = 2;
    let num_kv_heads = 2;
    let max_seq_len = 64;
    let gamma = 4;

    let transitions = vec![(1, 2), (2, 3), (3, 4)];
    let draft_qwen = create_deterministic_model(
        &device,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        max_seq_len,
        &transitions,
    )?;
    let draft_bonsai = Bonsai27BWithKv::new(draft_qwen, 16);
    let target_verifier = create_deterministic_model(
        &device,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        max_seq_len,
        &transitions,
    )?;

    let mut engine = SuperDraftSpeculativeEngine::new(draft_bonsai, target_verifier, gamma);

    let next_token = engine.prefill(&[1, 2])?;
    // Input prompt [1, 2] ends at token 2, which transitions to 3
    assert_eq!(next_token, 3);
    assert_eq!(engine.draft_bonsai.current_kv_pos(), 2);
    assert_eq!(engine.target_verifier.current_kv_pos(), 2);

    Ok(())
}

#[test]
fn test_superdraft_engine_divergence_when_draft_rolling_window_full() -> candle::Result<()> {
    let device = Device::Cpu;
    let vocab_size = 16;
    let hidden_size = 16;
    let num_heads = 2;
    let num_kv_heads = 2;
    let max_seq_len = 64;
    let window_size = 8;
    let gamma = 4;

    // Both models transition 1..15 cyclically
    let mut transitions = Vec::new();
    for i in 1..15 {
        transitions.push((i, i + 1));
    }
    let draft_qwen = create_deterministic_model(
        &device,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        max_seq_len,
        &transitions,
    )?;
    let draft_bonsai = Bonsai27BWithKv::new(draft_qwen, window_size);

    // Target verifier diverges at token 12: predicts 9 instead of 13
    let mut target_transitions = transitions.clone();
    for t in &mut target_transitions {
        if t.0 == 12 {
            t.1 = 9;
        }
    }
    let target_verifier = create_deterministic_model(
        &device,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        max_seq_len,
        &target_transitions,
    )?;

    let mut engine = SuperDraftSpeculativeEngine::new(draft_bonsai, target_verifier, gamma);

    // Prefill 10 tokens: both are at sequence position 10, draft rolling buffer is capped at window_size 8
    let prompt: Vec<u32> = (1..=10).collect();
    let current_token = engine.prefill(&prompt)?;
    assert_eq!(current_token, 11);
    assert_eq!(engine.target_verifier.current_kv_pos(), 10);
    assert_eq!(engine.draft_bonsai.current_kv_pos(), 10);
    assert_eq!(engine.draft_bonsai.kv_buffer_len(), window_size);

    // Step starting from token 11:
    // Draft proposes [12, 13, 14, 15]
    // Target expects [12, 9, ...] -> divergence at k = 1!
    let res = engine.speculative_step(current_token)?;
    assert_eq!(res.num_accepted_draft, 1);
    assert_eq!(res.accepted_tokens, vec![12, 9]);

    // Target accepted 1 draft token (12) + emitted correction token (9)
    // Both models rolled back to sequence position 12 (holding prompt 1..10 + draft token 11 + token 12)
    assert_eq!(engine.target_verifier.current_kv_pos(), 12);
    assert_eq!(engine.draft_bonsai.current_kv_pos(), 12);
    assert_eq!(engine.draft_bonsai.kv_buffer_len(), 6);

    // Execute subsequent step starting with correction token 9:
    // Both models ingest 9 at position 12 without token duplication!
    let next_res = engine.speculative_step(9)?;
    assert_eq!(engine.target_verifier.current_kv_pos() >= 13, true);
    assert_eq!(engine.draft_bonsai.current_kv_pos() >= 13, true);
    assert!(!next_res.accepted_tokens.is_empty());

    Ok(())
}

#[test]
fn test_superdraft_engine_multi_step_divergence_no_duplication_and_rope_monotonicity() -> candle::Result<()> {
    let device = Device::Cpu;
    let vocab_size = 32;
    let hidden_size = 32;
    let num_heads = 2;
    let num_kv_heads = 2;
    let max_seq_len = 64;
    let window_size = 8;
    let gamma = 4;

    // Draft predictably transitions t -> t + 1 for all t
    let mut draft_transitions = Vec::new();
    for i in 1..31 {
        draft_transitions.push((i, i + 1));
    }
    let draft_qwen = create_deterministic_model(
        &device,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        max_seq_len,
        &draft_transitions,
    )?;
    let draft_bonsai = Bonsai27BWithKv::new(draft_qwen, window_size);

    // Target agrees on 1..8, but at token 8 it diverges: predicts 20 instead of 9!
    // Then at token 20 it predicts 21.
    let mut target_transitions = draft_transitions.clone();
    for t in &mut target_transitions {
        if t.0 == 8 {
            t.1 = 20;
        }
    }
    target_transitions.push((20, 21));
    target_transitions.push((21, 22));

    let target_verifier = create_deterministic_model(
        &device,
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        max_seq_len,
        &target_transitions,
    )?;

    let mut engine = SuperDraftSpeculativeEngine::new(draft_bonsai, target_verifier, gamma);

    // Step 1: start at token 1
    // Draft proposes [2, 3, 4, 5]
    // Target matches all 4 and emits bonus token 6
    let res1 = engine.speculative_step(1)?;
    assert_eq!(res1.num_accepted_draft, 4);
    assert_eq!(res1.accepted_tokens, vec![2, 3, 4, 5, 6]);
    assert_eq!(res1.bonus_token, Some(6));
    assert_eq!(engine.target_verifier.current_kv_pos(), 5);
    assert_eq!(engine.draft_bonsai.current_kv_pos(), 5);

    // Step 2: continue from bonus token 6
    // Draft proposes [7, 8, 9, 10]
    // Target checks: 6->7, 7->8, 8->20! -> divergence at k = 2 (token 8 predicts 20, draft had 9)!
    // Accepted tokens: [7, 8, 20]
    let res2 = engine.speculative_step(6)?;
    assert_eq!(res2.num_accepted_draft, 2);
    assert_eq!(res2.accepted_tokens, vec![7, 8, 20]);
    assert_eq!(res2.bonus_token, None);

    // Both models roll back to position 5 + 2 + 1 = 8 (holding prompt 1 + [2,3,4,5,6] + [7,8])
    assert_eq!(engine.target_verifier.current_kv_pos(), 8);
    assert_eq!(engine.draft_bonsai.current_kv_pos(), 8);

    // Step 3: continue from correction token 20
    // Neither model should have duplicate token 20 in their KV cache!
    // Draft proposes from 20 -> [21, 22, 23, 24]
    // Target checks: 20->21, 21->22 ...
    let res3 = engine.speculative_step(20)?;
    assert_eq!(res3.accepted_tokens[0], 21);
    assert_eq!(engine.target_verifier.current_kv_pos() >= 9, true);
    assert_eq!(engine.draft_bonsai.current_kv_pos() >= 9, true);

    Ok(())
}
