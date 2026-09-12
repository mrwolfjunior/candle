use candle_speculative_server::engine::{verify_greedy, SpeculativeEngineConfig, SpeculativeResult};

#[test]
fn test_verify_greedy_all_accepted() {
    let draft_tokens = vec![101, 102, 103, 104];
    // Target produces matching argmax tokens for indices 0..3, plus bonus token 105
    let target_argmax = vec![101, 102, 103, 104, 105];

    let result: SpeculativeResult = verify_greedy(&draft_tokens, &target_argmax);
    assert_eq!(result.accepted_tokens, vec![101, 102, 103, 104, 105]);
    assert_eq!(result.num_accepted_draft, 4);
    assert_eq!(result.bonus_token, Some(105));
    assert_eq!(result.accepted_count(), 5);
}

#[test]
fn test_verify_greedy_partial_accepted() {
    let draft_tokens = vec![101, 102, 103, 104];
    // Target agrees on 101, 102, but diverges at index 2 (expects 999 instead of 103)
    let target_argmax = vec![101, 102, 999, 500, 600];

    let result: SpeculativeResult = verify_greedy(&draft_tokens, &target_argmax);
    assert_eq!(result.accepted_tokens, vec![101, 102, 999]);
    assert_eq!(result.num_accepted_draft, 2);
    assert_eq!(result.bonus_token, None);
    assert_eq!(result.accepted_count(), 3);
}

#[test]
fn test_verify_greedy_none_accepted() {
    let draft_tokens = vec![101, 102, 103];
    // Target diverges immediately at index 0 (expects 777 instead of 101)
    let target_argmax = vec![777, 888, 999, 1000];

    let result: SpeculativeResult = verify_greedy(&draft_tokens, &target_argmax);
    assert_eq!(result.accepted_tokens, vec![777]);
    assert_eq!(result.num_accepted_draft, 0);
    assert_eq!(result.bonus_token, None);
    assert_eq!(result.accepted_count(), 1);
}

#[test]
fn test_verify_greedy_all_accepted_no_bonus() {
    let draft_tokens = vec![101, 102];
    // Exactly matches draft tokens, but target does not provide a bonus token
    let target_argmax = vec![101, 102];

    let result = verify_greedy(&draft_tokens, &target_argmax);
    assert_eq!(result.accepted_tokens, vec![101, 102]);
    assert_eq!(result.num_accepted_draft, 2);
    assert_eq!(result.bonus_token, None);
    assert_eq!(result.accepted_count(), 2);
}

#[test]
#[should_panic(expected = "Target argmax length (2) must be at least draft token count (3)")]
fn test_verify_greedy_target_shorter_than_draft_panics() {
    let draft_tokens = vec![101, 102, 103];
    let target_argmax = vec![101, 102];
    let _ = verify_greedy(&draft_tokens, &target_argmax);
}

#[test]
fn test_speculative_engine_config_default() {
    let config = SpeculativeEngineConfig::default();
    assert_eq!(config.gamma, 4);
    assert!((config.temperature - 0.0).abs() < f64::EPSILON);
    assert!((config.top_p - 1.0).abs() < f64::EPSILON);
    assert_eq!(config.max_context, 65536);
}
