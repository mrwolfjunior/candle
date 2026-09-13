#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeResult {
    pub accepted_tokens: Vec<u32>,
    pub num_accepted_draft: usize,
    pub bonus_token: Option<u32>,
}

impl SpeculativeResult {
    pub fn accepted_count(&self) -> usize {
        self.accepted_tokens.len()
    }
}

/// Pure algorithmic greedy verification between draft proposals and target argmax predictions.
///
/// `draft_tokens`: proposed tokens [d_1, d_2, ..., d_gamma]
/// `target_argmax`: target model's highest-probability tokens for each position:
///                 target_argmax[0] corresponds to position of d_1,
///                 target_argmax[gamma-1] corresponds to d_gamma,
///                 target_argmax[gamma] is the bonus next-token prediction if all match.
pub fn verify_greedy(draft_tokens: &[u32], target_argmax: &[u32]) -> SpeculativeResult {
    let gamma = draft_tokens.len();
    assert!(
        target_argmax.len() >= gamma,
        "Target argmax length ({}) must be at least draft token count ({})",
        target_argmax.len(),
        gamma
    );

    let mut accepted = Vec::with_capacity(gamma + 1);
    let mut num_accepted_draft = 0;

    for i in 0..gamma {
        let expected = target_argmax[i];
        let proposed = draft_tokens[i];

        if proposed == expected {
            accepted.push(proposed);
            num_accepted_draft += 1;
        } else {
            // Divergence: accept target's correction token and terminate speculative cycle
            accepted.push(expected);
            return SpeculativeResult {
                accepted_tokens: accepted,
                num_accepted_draft,
                bonus_token: None,
            };
        }
    }

    // All gamma draft tokens matched! Accept the bonus token from target's final logits if available.
    let bonus_token = if target_argmax.len() > gamma {
        let bonus = target_argmax[gamma];
        accepted.push(bonus);
        Some(bonus)
    } else {
        None
    };

    SpeculativeResult {
        accepted_tokens: accepted,
        num_accepted_draft,
        bonus_token,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeEngineConfig {
    pub gamma: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub max_context: usize,
}

impl Default for SpeculativeEngineConfig {
    fn default() -> Self {
        Self {
            gamma: 4,
            temperature: 0.0,
            top_p: 1.0,
            max_context: 65536,
        }
    }
}

use candle::{IndexOp, Result, Tensor};
use crate::model::{Bonsai27BWithKv, QuantizedQwen2WithKv};

pub struct SuperDraftSpeculativeEngine {
    pub draft_bonsai: Bonsai27BWithKv,
    pub target_verifier: QuantizedQwen2WithKv,
    pub gamma: usize,
    pub draft_time: std::time::Duration,
    pub target_time: std::time::Duration,
}

impl SuperDraftSpeculativeEngine {
    pub fn new(
        draft_bonsai: Bonsai27BWithKv,
        target_verifier: QuantizedQwen2WithKv,
        gamma: usize,
    ) -> Self {
        Self {
            draft_bonsai,
            target_verifier,
            gamma,
            draft_time: std::time::Duration::ZERO,
            target_time: std::time::Duration::ZERO,
        }
    }

    pub fn reset_timings(&mut self) {
        self.draft_time = std::time::Duration::ZERO;
        self.target_time = std::time::Duration::ZERO;
    }

    pub fn gamma(&self) -> usize {
        self.gamma
    }

    /// Autoregressively propose `gamma` draft tokens using `draft_bonsai`.
    ///
    /// Input `current_token` is fed at current position `pos`.
    /// Proposes `gamma` tokens [d_1, d_2, ..., d_gamma].
    /// Draft's KV cache appends [current_token, d_1, ..., d_{gamma-1}],
    /// ending at position `pos + gamma`.
    pub fn propose(&mut self, current_token: u32) -> Result<Vec<u32>> {
        let mut draft_tokens = Vec::with_capacity(self.gamma);
        let mut next_in = current_token;

        for _ in 0..self.gamma {
            let input_tensor = Tensor::new(&[[next_in]], &self.draft_bonsai.model.device)?;
            let logits = self.draft_bonsai.forward(&input_tensor)?;
            let pred_token = logits
                .squeeze(0)?
                .squeeze(0)?
                .argmax(candle::D::Minus1)?
                .to_scalar::<u32>()?;
            draft_tokens.push(pred_token);
            next_in = pred_token;
        }

        Ok(draft_tokens)
    }

    /// Parallel batch forward pass on `target_verifier` to verify draft tokens.
    ///
    /// Evaluates `[current_token, d_1, d_2, ..., d_gamma]` in a single batch pass of length `gamma + 1`.
    /// Returns `target_argmax` containing `gamma + 1` predicted tokens.
    pub fn target_forward_batch(
        &mut self,
        current_token: u32,
        draft_tokens: &[u32],
    ) -> Result<Vec<u32>> {
        let mut target_input = Vec::with_capacity(draft_tokens.len() + 1);
        target_input.push(current_token);
        target_input.extend_from_slice(draft_tokens);

        let input_tensor = Tensor::from_slice(
            &target_input,
            (1, target_input.len()),
            &self.target_verifier.device,
        )?;
        let logits = self.target_verifier.forward(&input_tensor)?;
        let argmax_tensor = logits.squeeze(0)?.argmax(candle::D::Minus1)?;
        let target_argmax = argmax_tensor.to_vec1::<u32>()?;
        Ok(target_argmax)
    }

    /// Execute a single speculative iteration:
    /// 1. Propose `gamma` tokens with draft on its device.
    /// 2. Parallel batch forward pass on target verifier.
    /// 3. Verify tokens via `verify_greedy`.
    /// 4. If discrepancy at index `k < gamma`, rollback KV cache on both models to `pos + k + 1`.
    ///    If all accepted ($k == gamma$), synchronize draft KV cache with the final draft token.
    pub fn step(&mut self, current_token: u32) -> Result<SpeculativeResult> {
        let start_pos = self.target_verifier.current_kv_pos();

        // 1. Propose gamma tokens with draft on its device
        let t_draft_start = std::time::Instant::now();
        let draft_tokens = self.propose(current_token)?;
        self.draft_time += t_draft_start.elapsed();

        // 2. Parallel batch forward pass on target verifier
        let t_target_start = std::time::Instant::now();
        let target_argmax = self.target_forward_batch(current_token, &draft_tokens)?;
        self.target_time += t_target_start.elapsed();

        // 3. Verify tokens via verify_greedy
        let result = verify_greedy(&draft_tokens, &target_argmax);

        // 4. Rollback or synchronize KV caches
        let k = result.num_accepted_draft;
        if k < self.gamma {
            // Discrepancy at index k < gamma:
            // Rollback both models to start_pos + k + 1.
            // Draft proposed gamma tokens, and appended [current_token, d_0, ..., d_{gamma-2}] (gamma tokens).
            // Exactly k draft tokens were accepted: d_0, ..., d_{k-1}.
            // Rolling back to start_pos + k + 1 discards the (gamma - k - 1) rejected draft tokens.
            // Both models now contain [current_token, d_0, ..., d_{k-1}].
            // The target's correction token (result.accepted_tokens[k]) is returned to the caller
            // and will be ingested as current_token on the subsequent step.
            let rollback_pos = start_pos + k + 1;
            self.draft_bonsai.rollback_kv(rollback_pos)?;
            self.target_verifier.rollback_kv(rollback_pos)?;
        } else {
            // All gamma draft tokens accepted!
            // Append the final accepted draft token into draft's cache to synchronize
            let t_sync_start = std::time::Instant::now();
            let last_draft_token = draft_tokens[self.gamma - 1];
            let input_tensor = Tensor::new(&[[last_draft_token]], &self.draft_bonsai.model.device)?;
            let _ = self.draft_bonsai.forward(&input_tensor)?;
            self.draft_time += t_sync_start.elapsed();
        }

        Ok(result)
    }

    /// Speculative step alias
    pub fn speculative_step(&mut self, current_token: u32) -> Result<SpeculativeResult> {
        self.step(current_token)
    }

    /// Prefill prompt on both models and return the first generated token from target.
    pub fn prefill(&mut self, prompt: &[u32]) -> Result<u32> {
        if prompt.is_empty() {
            return Err(candle::Error::Msg("Cannot prefill empty prompt".into()));
        }

        let chunk_size = 2048;
        let mut last_next_token = 0;

        for chunk in prompt.chunks(chunk_size) {
            let draft_input =
                Tensor::from_slice(chunk, (1, chunk.len()), &self.draft_bonsai.model.device)?;
            let _ = self.draft_bonsai.forward(&draft_input)?;

            let target_input =
                Tensor::from_slice(chunk, (1, chunk.len()), &self.target_verifier.device)?;
            let logits = self.target_verifier.forward(&target_input)?;
            last_next_token = logits
                .squeeze(0)?
                .i(chunk.len() - 1)?
                .argmax(candle::D::Minus1)?
                .to_scalar::<u32>()?;
        }

        Ok(last_next_token)
    }

    /// Rollback KV caches on both models
    pub fn rollback_kv(&mut self, pos: usize) -> Result<()> {
        self.draft_bonsai.rollback_kv(pos)?;
        self.target_verifier.rollback_kv(pos)?;
        Ok(())
    }

    /// Reset KV caches on both models
    pub fn reset_kv(&mut self) {
        self.draft_bonsai.reset_kv();
        self.target_verifier.reset_kv();
    }
}
