use candle::{Device, DType, Tensor};
use candle_speculative_server::kv_cache::InPlaceKvCache;

#[test]
fn test_in_place_kv_cache_append_and_rollback() -> candle::Result<()> {
    let dev = Device::Cpu;
    let b_sz = 1;
    let n_kv_head = 2;
    let head_dim = 4;
    let max_seq_len = 16;
    let dtype = DType::F32;

    let mut cache = InPlaceKvCache::new(b_sz, n_kv_head, head_dim, max_seq_len, dtype, &dev)?;
    assert_eq!(cache.current_pos(), 0);

    // Append 3 tokens
    let k1 = Tensor::zeros((b_sz, n_kv_head, 3, head_dim), dtype, &dev)?;
    let v1 = Tensor::zeros((b_sz, n_kv_head, 3, head_dim), dtype, &dev)?;
    cache.append(&k1, &v1)?;
    assert_eq!(cache.current_pos(), 3);

    let (k_view, v_view) = cache.current_view()?;
    assert_eq!(k_view.dims(), &[b_sz, n_kv_head, 3, head_dim]);
    assert_eq!(v_view.dims(), &[b_sz, n_kv_head, 3, head_dim]);

    // Append 4 more tokens (speculative draft)
    let k2 = Tensor::zeros((b_sz, n_kv_head, 4, head_dim), dtype, &dev)?;
    let v2 = Tensor::zeros((b_sz, n_kv_head, 4, head_dim), dtype, &dev)?;
    cache.append(&k2, &v2)?;
    assert_eq!(cache.current_pos(), 7);

    // Roll back to 5 tokens (speculative rejection of last 2)
    cache.rollback(5)?;
    assert_eq!(cache.current_pos(), 5);

    let (k_view2, _) = cache.current_view()?;
    assert_eq!(k_view2.dims(), &[b_sz, n_kv_head, 5, head_dim]);

    // Reset
    cache.reset();
    assert_eq!(cache.current_pos(), 0);

    Ok(())
}

#[test]
fn test_in_place_kv_cache_data_preservation_and_overwrite() -> candle::Result<()> {
    let dev = Device::Cpu;
    let (b_sz, n_kv_head, head_dim, max_seq_len) = (1, 1, 2, 8);
    let dtype = DType::F32;

    let mut cache = InPlaceKvCache::new(b_sz, n_kv_head, head_dim, max_seq_len, dtype, &dev)?;

    // Initial empty view
    let (k_empty, v_empty) = cache.current_view()?;
    assert_eq!(k_empty.dims(), &[1, 1, 0, 2]);
    assert_eq!(v_empty.dims(), &[1, 1, 0, 2]);

    // Write token 0 and 1 with known values
    let k_data = Tensor::new(&[[[[1.0f32, 2.0], [3.0, 4.0]]]], &dev)?;
    let v_data = Tensor::new(&[[[[10.0f32, 20.0], [30.0, 40.0]]]], &dev)?;
    cache.append(&k_data, &v_data)?;
    assert_eq!(cache.current_pos(), 2);

    let (k_view, v_view) = cache.current_view()?;
    assert_eq!(k_view.flatten_all()?.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(v_view.flatten_all()?.to_vec1::<f32>()?, vec![10.0, 20.0, 30.0, 40.0]);

    // Append token 2 (draft token)
    let k_draft = Tensor::new(&[[[[5.0f32, 6.0]]]], &dev)?;
    let v_draft = Tensor::new(&[[[[50.0f32, 60.0]]]], &dev)?;
    cache.append(&k_draft, &v_draft)?;
    assert_eq!(cache.current_pos(), 3);

    // Roll back token 2
    cache.rollback(2)?;
    assert_eq!(cache.current_pos(), 2);

    // Append different token 2 (replacement accepted token)
    let k_correct = Tensor::new(&[[[[7.0f32, 8.0]]]], &dev)?;
    let v_correct = Tensor::new(&[[[[70.0f32, 80.0]]]], &dev)?;
    cache.append(&k_correct, &v_correct)?;
    assert_eq!(cache.current_pos(), 3);

    let (k_view, v_view) = cache.current_view()?;
    assert_eq!(
        k_view.flatten_all()?.to_vec1::<f32>()?,
        vec![1.0, 2.0, 3.0, 4.0, 7.0, 8.0]
    );
    assert_eq!(
        v_view.flatten_all()?.to_vec1::<f32>()?,
        vec![10.0, 20.0, 30.0, 40.0, 70.0, 80.0]
    );

    Ok(())
}

#[test]
fn test_in_place_kv_cache_boundary_checks() -> candle::Result<()> {
    let dev = Device::Cpu;
    let mut cache = InPlaceKvCache::new(1, 1, 2, 4, DType::F32, &dev)?;

    // Append 3 tokens
    let k = Tensor::zeros((1, 1, 3, 2), DType::F32, &dev)?;
    let v = Tensor::zeros((1, 1, 3, 2), DType::F32, &dev)?;
    cache.append(&k, &v)?;

    // Append 2 tokens -> should fail because 3 + 2 > 4
    let k_overflow = Tensor::zeros((1, 1, 2, 2), DType::F32, &dev)?;
    let v_overflow = Tensor::zeros((1, 1, 2, 2), DType::F32, &dev)?;
    assert!(cache.append(&k_overflow, &v_overflow).is_err());

    // Rollback past current_pos (e.g. 5 > 3) -> should fail
    assert!(cache.rollback(5).is_err());

    // Valid rollback to 1
    assert!(cache.rollback(1).is_ok());
    assert_eq!(cache.current_pos(), 1);

    Ok(())
}
