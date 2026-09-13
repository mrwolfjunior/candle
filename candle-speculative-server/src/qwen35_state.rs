use candle::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

/// Architectural dimensions and hyper-parameters for Qwen3.5 hybrid models (e.g. Bonsai-27B).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Qwen35Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub full_attn_interval: usize,
    pub ssm_conv_kernel: usize,
    pub ssm_d_state: usize,
    pub ssm_n_group: usize,
    pub ssm_dt_rank: usize,
    pub ssm_inner_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
}

impl Default for Qwen35Config {
    fn default() -> Self {
        Self::bonsai_27b()
    }
}

impl Qwen35Config {
    /// Architectural dimensions for `Bonsai-27B` (hybrid 64 layers: 48 SSM + 16 full GQA attention).
    pub fn bonsai_27b() -> Self {
        Self {
            hidden_size: 5120,
            intermediate_size: 17408,
            num_hidden_layers: 64,
            full_attn_interval: 4,
            ssm_conv_kernel: 4,
            ssm_d_state: 128,
            ssm_n_group: 16,
            ssm_dt_rank: 48,
            ssm_inner_size: 6144,
            num_attention_heads: 24,
            num_key_value_heads: 4,
            head_dim: 256,
            vocab_size: 248320,
            rope_theta: 10_000_000.0,
            rms_norm_eps: 1e-6,
        }
    }

    /// Whether the given layer index (0..num_hidden_layers) is a full GQA attention layer.
    /// Interleaving rule: every 4th layer: `(layer_idx + 1) % full_attn_interval == 0`.
    #[inline]
    pub fn is_full_attn(&self, layer_idx: usize) -> bool {
        (layer_idx + 1) % self.full_attn_interval == 0
    }

    /// Whether the given layer index (0..num_hidden_layers) is a recurrent SSM layer.
    #[inline]
    pub fn is_ssm(&self, layer_idx: usize) -> bool {
        !self.is_full_attn(layer_idx)
    }

    /// Total count of recurrent SSM layers in the model (e.g. 48 for Bonsai-27B).
    pub fn num_ssm_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&i| self.is_ssm(i))
            .count()
    }

    /// Total count of full attention layers in the model (e.g. 16 for Bonsai-27B).
    pub fn num_full_attn_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&i| self.is_full_attn(i))
            .count()
    }

    /// Map a 0-based global layer index (0..64) to its 0-based SSM layer index (0..48).
    /// Returns `None` if the layer is a full attention layer.
    #[inline]
    pub fn ssm_layer_index(&self, layer_idx: usize) -> Option<usize> {
        if self.is_full_attn(layer_idx) {
            None
        } else {
            Some(layer_idx - (layer_idx + 1) / self.full_attn_interval)
        }
    }

    /// Channel dimension for the SSM depthwise 1D convolution.
    /// In Qwen3.5 SSM: `ssm_inner_size + 2 * (ssm_n_group * ssm_d_state)`
    /// = `6144 + 2 * (16 * 128) = 6144 + 4096 = 10240`.
    #[inline]
    pub fn ssm_conv_dim(&self) -> usize {
        self.ssm_inner_size + 2 * self.ssm_n_group * self.ssm_d_state
    }

    /// Context history length kept in the depthwise conv state: `ssm_conv_kernel - 1 = 3`.
    #[inline]
    pub fn ssm_conv_len(&self) -> usize {
        self.ssm_conv_kernel.saturating_sub(1)
    }

    /// Query dimension: `num_attention_heads * head_dim` = `24 * 256 = 6144`.
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// Fused Query + Gate dimension: `2 * q_dim()` = `12288`.
    #[inline]
    pub fn q_gate_dim(&self) -> usize {
        self.q_dim() * 2
    }

    /// Key/Value projection dimension: `num_key_value_heads * head_dim` = `4 * 256 = 1024`.
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Grouped-query repetition factor: `num_attention_heads / num_key_value_heads` = `24 / 4 = 6`.
    #[inline]
    pub fn gqa_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Parse config from GGUF metadata, falling back to standard Bonsai-27B dimensions.
    pub fn from_gguf(ct: &candle::quantized::gguf_file::Content) -> Result<Self> {
        let arch = if let Some(candle::quantized::gguf_file::Value::String(arch)) =
            ct.metadata.get("general.architecture")
        {
            Some(arch.as_str())
        } else {
            None
        };

        let mut prefixes = Vec::new();
        if let Some(a) = arch {
            prefixes.push(a);
        }
        if !prefixes.contains(&"qwen35") {
            prefixes.push("qwen35");
        }
        if !prefixes.contains(&"qwen2") {
            prefixes.push("qwen2");
        }

        let find_u32 = |suffix: &str| -> Option<u32> {
            for prefix in &prefixes {
                let key = format!("{prefix}.{suffix}");
                if let Some(val) = ct.metadata.get(&key) {
                    if let Ok(v) = val.to_u32() {
                        return Some(v);
                    }
                    if let Ok(v) = val.to_u64() {
                        return Some(v as u32);
                    }
                }
            }
            if let Some(val) = ct.metadata.get(suffix) {
                if let Ok(v) = val.to_u32() {
                    return Some(v);
                }
                if let Ok(v) = val.to_u64() {
                    return Some(v as u32);
                }
            }
            None
        };

        let find_f32 = |suffix: &str| -> Option<f32> {
            for prefix in &prefixes {
                let key = format!("{prefix}.{suffix}");
                if let Some(val) = ct.metadata.get(&key) {
                    if let Ok(v) = val.to_f32() {
                        return Some(v);
                    }
                    if let Ok(v) = val.to_f64() {
                        return Some(v as f32);
                    }
                }
            }
            if let Some(val) = ct.metadata.get(suffix) {
                if let Ok(v) = val.to_f32() {
                    return Some(v);
                }
                if let Ok(v) = val.to_f64() {
                    return Some(v as f32);
                }
            }
            None
        };

        let hidden_size = find_u32("embedding_length").unwrap_or(5120) as usize;
        let intermediate_size = find_u32("feed_forward_length").unwrap_or(17408) as usize;
        let num_hidden_layers = find_u32("block_count").unwrap_or(64) as usize;
        let full_attn_interval = find_u32("full_attention_interval").unwrap_or(4) as usize;
        let ssm_conv_kernel = find_u32("ssm_conv_kernel").unwrap_or(4) as usize;
        let ssm_d_state = find_u32("ssm_d_state").unwrap_or(128) as usize;
        let ssm_n_group = find_u32("ssm_n_group").unwrap_or(16) as usize;
        let ssm_dt_rank = find_u32("ssm_dt_rank").unwrap_or(48) as usize;
        let ssm_inner_size = find_u32("ssm_inner_size").unwrap_or(6144) as usize;
        let num_attention_heads = find_u32("attention.head_count").unwrap_or(24) as usize;
        let num_key_value_heads = find_u32("attention.head_count_kv").unwrap_or(4) as usize;
        let head_dim = find_u32("attention.key_length").unwrap_or(256) as usize;
        let vocab_size = find_u32("vocab_size").unwrap_or(248320) as usize;
        let rope_theta = find_f32("rope.freq_base").unwrap_or(10_000_000.0) as f64;
        let rms_norm_eps = find_f32("attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f64;

        Ok(Self {
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            full_attn_interval,
            ssm_conv_kernel,
            ssm_d_state,
            ssm_n_group,
            ssm_dt_rank,
            ssm_inner_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            vocab_size,
            rope_theta,
            rms_norm_eps,
        })
    }
}

/// Recurrent and convolutional state for a single SSM layer.
#[derive(Debug, Clone)]
pub struct Qwen35LayerState {
    /// 1D depthwise conv state of shape `[1, 3, 10240]`
    /// (retains the last `ssm_conv_kernel - 1` tokens of mixed QKV projection).
    pub conv_state: Tensor,
    /// Gated Delta Net recurrent state for 48 heads of shape `[48, 128, 128]` (dtype F32).
    pub ssm_state: Tensor,
}

impl Qwen35LayerState {
    /// Allocate zero-initialized layer state on `device` with default F32 precision.
    pub fn new(config: &Qwen35Config, device: &Device) -> Result<Self> {
        Self::new_with_dtype(config, DType::F32, device)
    }

    /// Allocate zero-initialized layer state with custom precision for conv state.
    pub fn new_with_dtype(config: &Qwen35Config, conv_dtype: DType, device: &Device) -> Result<Self> {
        let conv_len = config.ssm_conv_len();
        let conv_dim = config.ssm_conv_dim();
        let conv_state = Tensor::zeros((1, conv_len, conv_dim), conv_dtype, device)?;
        let ssm_state = Tensor::zeros(
            (config.ssm_dt_rank, config.ssm_d_state, config.ssm_d_state),
            DType::F32,
            device,
        )?;
        Ok(Self {
            conv_state,
            ssm_state,
        })
    }

    /// Reset layer state tensors to zeros.
    pub fn reset(&mut self) {
        if let Ok(zeros) = Tensor::zeros(
            self.conv_state.shape(),
            self.conv_state.dtype(),
            self.conv_state.device(),
        ) {
            self.conv_state = zeros;
        }
        if let Ok(zeros) = Tensor::zeros(
            self.ssm_state.shape(),
            self.ssm_state.dtype(),
            self.ssm_state.device(),
        ) {
            self.ssm_state = zeros;
        }
    }

    /// Fallible variant of `reset`.
    pub fn try_reset(&mut self) -> Result<()> {
        self.conv_state = Tensor::zeros(
            self.conv_state.shape(),
            self.conv_state.dtype(),
            self.conv_state.device(),
        )?;
        self.ssm_state = Tensor::zeros(
            self.ssm_state.shape(),
            self.ssm_state.dtype(),
            self.ssm_state.device(),
        )?;
        Ok(())
    }

    /// Create a shallow snapshot of this layer's state in $O(1)$ time.
    pub fn snapshot(&self) -> Result<Self> {
        Ok(Self {
            conv_state: self.conv_state.clone(),
            ssm_state: self.ssm_state.clone(),
        })
    }

    /// Restore state from a snapshot.
    pub fn restore(&mut self, snapshot: &Self) {
        self.conv_state = snapshot.conv_state.clone();
        self.ssm_state = snapshot.ssm_state.clone();
    }
}

/// Lightweight snapshot of all SSM layers for $O(1)$ speculative rollback.
#[derive(Debug, Clone)]
pub struct Qwen35StateSnapshot {
    pub layers: Vec<Qwen35LayerState>,
}

impl Qwen35StateSnapshot {
    pub fn new(layers: Vec<Qwen35LayerState>) -> Self {
        Self { layers }
    }

    pub fn layers(&self) -> &[Qwen35LayerState] {
        &self.layers
    }
}

/// Collection of recurrent states for all 48 SSM layers in Qwen3.5 (Bonsai-27B).
#[derive(Debug, Clone)]
pub struct Qwen35RecurrentState {
    pub layers: Vec<Qwen35LayerState>,
}

impl Qwen35RecurrentState {
    /// Allocate zero-initialized recurrent states for all SSM layers on `device`.
    pub fn new(config: &Qwen35Config, device: &Device) -> Result<Self> {
        Self::new_with_dtype(config, DType::F32, device)
    }

    /// Allocate zero-initialized recurrent states with custom conv precision.
    pub fn new_with_dtype(
        config: &Qwen35Config,
        conv_dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let num_ssm = config.num_ssm_layers();
        let mut layers = Vec::with_capacity(num_ssm);
        for _ in 0..num_ssm {
            layers.push(Qwen35LayerState::new_with_dtype(config, conv_dtype, device)?);
        }
        Ok(Self { layers })
    }

    /// Total number of SSM layer states (48 for Bonsai-27B).
    #[inline]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Get SSM layer state by 0-based SSM index (0..48).
    #[inline]
    pub fn get_ssm_layer(&self, ssm_idx: usize) -> Option<&Qwen35LayerState> {
        self.layers.get(ssm_idx)
    }

    /// Get mutable SSM layer state by 0-based SSM index (0..48).
    #[inline]
    pub fn get_ssm_layer_mut(&mut self, ssm_idx: usize) -> Option<&mut Qwen35LayerState> {
        self.layers.get_mut(ssm_idx)
    }

    /// Get SSM layer state by 0-based global layer index (0..64).
    /// Returns `None` for full attention layers or out-of-range indices.
    pub fn get_layer_state(&self, layer_idx: usize, config: &Qwen35Config) -> Option<&Qwen35LayerState> {
        let ssm_idx = config.ssm_layer_index(layer_idx)?;
        self.layers.get(ssm_idx)
    }

    /// Get mutable SSM layer state by 0-based global layer index (0..64).
    pub fn get_layer_state_mut(
        &mut self,
        layer_idx: usize,
        config: &Qwen35Config,
    ) -> Option<&mut Qwen35LayerState> {
        let ssm_idx = config.ssm_layer_index(layer_idx)?;
        self.layers.get_mut(ssm_idx)
    }

    /// Reset all 48 SSM layer states to zero.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Fallible variant of `reset`.
    pub fn try_reset(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.try_reset()?;
        }
        Ok(())
    }

    /// Take an $O(1)$ snapshot of all recurrent states for speculative rollback.
    pub fn snapshot(&self) -> Result<Qwen35StateSnapshot> {
        let mut snap_layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            snap_layers.push(layer.snapshot()?);
        }
        Ok(Qwen35StateSnapshot::new(snap_layers))
    }

    /// Restore all recurrent states from a snapshot in $O(1)$ pointer copy time.
    pub fn restore(&mut self, snapshot: &Qwen35StateSnapshot) {
        for (layer, snap) in self.layers.iter_mut().zip(&snapshot.layers) {
            layer.restore(snap);
        }
    }

    /// Stacks all 48 conv states into a single tensor of shape `[48, 3, 10240]`.
    pub fn stacked_conv_states(&self) -> Result<Tensor> {
        let squeezed: Result<Vec<Tensor>> = self
            .layers
            .iter()
            .map(|l| l.conv_state.squeeze(0))
            .collect();
        let squeezed = squeezed?;
        let refs: Vec<&Tensor> = squeezed.iter().collect();
        Tensor::stack(&refs, 0)
    }

    /// Stacks all 48 SSM recurrent states into a single tensor of shape `[48, 48, 128, 128]`.
    pub fn stacked_ssm_states(&self) -> Result<Tensor> {
        let refs: Vec<&Tensor> = self.layers.iter().map(|l| &l.ssm_state).collect();
        Tensor::stack(&refs, 0)
    }
}
