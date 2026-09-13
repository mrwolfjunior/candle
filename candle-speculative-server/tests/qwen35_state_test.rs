use candle::{DType, Device, Tensor};
use candle_speculative_server::{
    qwen35_state::{Qwen35Config, Qwen35LayerState, Qwen35RecurrentState, Qwen35StateSnapshot},
};

#[test]
fn test_qwen35_config_dimensions_and_helpers() {
    let config = Qwen35Config::bonsai_27b();
    assert_eq!(config.hidden_size, 5120);
    assert_eq!(config.intermediate_size, 17408);
    assert_eq!(config.num_hidden_layers, 64);
    assert_eq!(config.full_attn_interval, 4);
    assert_eq!(config.ssm_conv_kernel, 4);
    assert_eq!(config.ssm_d_state, 128);
    assert_eq!(config.ssm_n_group, 16);
    assert_eq!(config.ssm_dt_rank, 48);
    assert_eq!(config.ssm_inner_size, 6144);
    assert_eq!(config.num_attention_heads, 24);
    assert_eq!(config.num_key_value_heads, 4);
    assert_eq!(config.head_dim, 256);
    assert_eq!(config.vocab_size, 248320);
    assert_eq!(config.rope_theta, 10000000.0);
    assert_eq!(config.rms_norm_eps, 1e-6);

    // Default trait
    let default_cfg = Qwen35Config::default();
    assert_eq!(config, default_cfg);

    // Architectural counts
    assert_eq!(config.num_ssm_layers(), 48);
    assert_eq!(config.num_full_attn_layers(), 16);
    assert_eq!(config.ssm_conv_dim(), 10240);
    assert_eq!(config.ssm_conv_len(), 3);

    // Layer interleaving: every 4th layer is full attention ((i + 1) % 4 == 0)
    // Layers 0, 1, 2 are SSM; layer 3 is Full Attn
    assert!(!config.is_full_attn(0));
    assert!(!config.is_full_attn(1));
    assert!(!config.is_full_attn(2));
    assert!(config.is_full_attn(3));

    assert!(!config.is_full_attn(4));
    assert!(!config.is_full_attn(5));
    assert!(!config.is_full_attn(6));
    assert!(config.is_full_attn(7));

    assert!(config.is_full_attn(63));
    assert!(!config.is_full_attn(62));

    // SSM layer index mapping
    assert_eq!(config.ssm_layer_index(0), Some(0));
    assert_eq!(config.ssm_layer_index(1), Some(1));
    assert_eq!(config.ssm_layer_index(2), Some(2));
    assert_eq!(config.ssm_layer_index(3), None);
    assert_eq!(config.ssm_layer_index(4), Some(3));
    assert_eq!(config.ssm_layer_index(7), None);
    assert_eq!(config.ssm_layer_index(62), Some(47));
    assert_eq!(config.ssm_layer_index(63), None);
}

#[test]
fn test_qwen35_layer_state_allocation_f32_and_f16() -> candle::Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();

    // Default F32 layer state
    let layer_state = Qwen35LayerState::new(&config, &device)?;
    assert_eq!(layer_state.conv_state.dims(), &[1, 3, 10240]);
    assert_eq!(layer_state.conv_state.dtype(), DType::F32);
    assert_eq!(layer_state.ssm_state.dims(), &[48, 128, 128]);
    assert_eq!(layer_state.ssm_state.dtype(), DType::F32);

    // Conv state in F16
    let layer_state_f16 = Qwen35LayerState::new_with_dtype(&config, DType::F16, &device)?;
    assert_eq!(layer_state_f16.conv_state.dims(), &[1, 3, 10240]);
    assert_eq!(layer_state_f16.conv_state.dtype(), DType::F16);
    assert_eq!(layer_state_f16.ssm_state.dims(), &[48, 128, 128]);
    assert_eq!(layer_state_f16.ssm_state.dtype(), DType::F32);

    Ok(())
}

#[test]
fn test_qwen35_recurrent_state_allocation_and_stacked_views() -> candle::Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();

    let state = Qwen35RecurrentState::new(&config, &device)?;
    assert_eq!(state.layers.len(), 48);
    assert_eq!(state.len(), 48);
    assert!(!state.is_empty());

    for (idx, layer) in state.layers.iter().enumerate() {
        assert_eq!(
            layer.conv_state.dims(),
            &[1, 3, 10240],
            "layer {idx} conv_state dim mismatch"
        );
        assert_eq!(
            layer.ssm_state.dims(),
            &[48, 128, 128],
            "layer {idx} ssm_state dim mismatch"
        );
    }

    // Stacked tensor shapes match plan specification: [48, 3, 10240] and [48, 48, 128, 128]
    let stacked_conv = state.stacked_conv_states()?;
    assert_eq!(stacked_conv.dims(), &[48, 3, 10240]);

    let stacked_ssm = state.stacked_ssm_states()?;
    assert_eq!(stacked_ssm.dims(), &[48, 48, 128, 128]);

    Ok(())
}

#[test]
fn test_qwen35_state_snapshot_and_restore() -> candle::Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();

    let mut state = Qwen35RecurrentState::new(&config, &device)?;

    // Modify state at layer 0 and layer 47 with distinctive values
    let val_conv_0 = Tensor::full(1.5f32, (1, 3, 10240), &device)?;
    let val_ssm_0 = Tensor::full(2.5f32, (48, 128, 128), &device)?;
    state.layers[0].conv_state = val_conv_0;
    state.layers[0].ssm_state = val_ssm_0;

    let val_conv_47 = Tensor::full(3.5f32, (1, 3, 10240), &device)?;
    let val_ssm_47 = Tensor::full(4.5f32, (48, 128, 128), &device)?;
    state.layers[47].conv_state = val_conv_47;
    state.layers[47].ssm_state = val_ssm_47;

    // Take snapshot
    let snapshot: Qwen35StateSnapshot = state.snapshot()?;
    assert_eq!(snapshot.layers.len(), 48);

    // Overwrite state with speculative garbage (simulating rejected draft tokens)
    let bad_conv = Tensor::full(999.0f32, (1, 3, 10240), &device)?;
    let bad_ssm = Tensor::full(888.0f32, (48, 128, 128), &device)?;
    for layer in &mut state.layers {
        layer.conv_state = bad_conv.clone();
        layer.ssm_state = bad_ssm.clone();
    }

    // Verify current state is dirty
    let curr_val_0 = state.layers[0].conv_state.flatten_all()?.to_vec1::<f32>()?[0];
    assert_eq!(curr_val_0, 999.0);

    // Verify snapshot was not mutated
    let snap_val_0 = snapshot.layers[0].conv_state.flatten_all()?.to_vec1::<f32>()?[0];
    assert_eq!(snap_val_0, 1.5);
    let snap_ssm_0 = snapshot.layers[0].ssm_state.flatten_all()?.to_vec1::<f32>()?[0];
    assert_eq!(snap_ssm_0, 2.5);

    // Restore state from snapshot
    state.restore(&snapshot);

    // Verify values restored
    let restored_conv_0 = state.layers[0].conv_state.flatten_all()?.to_vec1::<f32>()?[0];
    assert_eq!(restored_conv_0, 1.5);
    let restored_ssm_0 = state.layers[0].ssm_state.flatten_all()?.to_vec1::<f32>()?[0];
    assert_eq!(restored_ssm_0, 2.5);

    let restored_conv_47 = state.layers[47].conv_state.flatten_all()?.to_vec1::<f32>()?[0];
    assert_eq!(restored_conv_47, 3.5);
    let restored_ssm_47 = state.layers[47].ssm_state.flatten_all()?.to_vec1::<f32>()?[0];
    assert_eq!(restored_ssm_47, 4.5);

    Ok(())
}

#[test]
fn test_qwen35_state_reset() -> candle::Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();

    let mut state = Qwen35RecurrentState::new(&config, &device)?;

    // Fill with non-zero
    for layer in &mut state.layers {
        layer.conv_state = Tensor::full(42.0f32, (1, 3, 10240), &device)?;
        layer.ssm_state = Tensor::full(7.0f32, (48, 128, 128), &device)?;
    }

    // Reset
    state.reset();

    // Verify all zeros
    for layer in &state.layers {
        let conv_max = layer.conv_state.abs()?.max_all()?.to_scalar::<f32>()?;
        let ssm_max = layer.ssm_state.abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(conv_max, 0.0);
        assert_eq!(ssm_max, 0.0);
    }

    Ok(())
}

#[test]
fn test_qwen35_multi_step_speculative_rollback() -> candle::Result<()> {
    let device = Device::Cpu;
    let config = Qwen35Config::bonsai_27b();

    let mut state = Qwen35RecurrentState::new(&config, &device)?;

    // Step 0 initial snapshot
    let snap_0 = state.snapshot()?;

    // Step 1: Draft token 1
    state.layers[0].conv_state = Tensor::full(1.0f32, (1, 3, 10240), &device)?;
    let snap_1 = state.snapshot()?;

    // Step 2: Draft token 2
    state.layers[0].conv_state = Tensor::full(2.0f32, (1, 3, 10240), &device)?;
    let snap_2 = state.snapshot()?;

    // Step 3: Draft token 3
    state.layers[0].conv_state = Tensor::full(3.0f32, (1, 3, 10240), &device)?;

    // Target verifier accepts up to Step 1, rejects Step 2 and 3
    state.restore(&snap_1);

    let current_val = state.layers[0].conv_state.flatten_all()?.to_vec1::<f32>()?[0];
    assert_eq!(current_val, 1.0);

    // Target verifier rejects all tokens, rolls back to Step 0
    state.restore(&snap_0);
    let initial_val = state.layers[0].conv_state.flatten_all()?.to_vec1::<f32>()?[0];
    assert_eq!(initial_val, 0.0);

    // Can still restore to snap_2 if needed
    state.restore(&snap_2);
    let step2_val = state.layers[0].conv_state.flatten_all()?.to_vec1::<f32>()?[0];
    assert_eq!(step2_val, 2.0);

    Ok(())
}
