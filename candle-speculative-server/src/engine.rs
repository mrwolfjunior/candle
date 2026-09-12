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
