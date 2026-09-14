use candle::{
    quantized::{gguf_file, QMatMul, QTensor},
    Device, Module, Result, Tensor,
};
use candle_transformers::quantized_nn::RmsNorm;

use crate::{
    model::precompute_freqs_cis,
    qwen35_attn::Qwen35AttnLayer,
    qwen35_ssm::Qwen35SsmLayer,
    qwen35_state::{Qwen35Config, Qwen35RecurrentState, Qwen35StateSnapshot},
};

/// SwiGLU Feed-Forward Network layer for Qwen3.5.
///
/// Mathematical formulation:
/// `norm_xs = post_attention_norm(xs)`
/// `gate = ffn_gate(norm_xs)`
/// `up = ffn_up(norm_xs)`
/// `h = silu(gate) * up`
/// `out = ffn_down(h)`
#[derive(Debug, Clone)]
pub struct SwiGluFfn {
    pub ffn_gate: QMatMul,
    pub ffn_up: QMatMul,
    pub ffn_down: QMatMul,
    pub post_attention_norm: RmsNorm,
}

impl SwiGluFfn {
    pub fn new(
        ffn_gate: QMatMul,
        ffn_up: QMatMul,
        ffn_down: QMatMul,
        post_attention_norm: RmsNorm,
    ) -> Self {
        Self {
            ffn_gate,
            ffn_up,
            ffn_down,
            post_attention_norm,
        }
    }

    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &gguf_file::Content,
        reader: &mut R,
        layer_idx: usize,
        rms_norm_eps: f64,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}.");
        let mut find_qtensor = |names: &[String]| -> Result<QTensor> {
            for name in names {
                if ct.tensor_infos.contains_key(name) {
                    return ct.tensor(reader, name, device);
                }
            }
            candle::bail!("cannot find tensor info for {}", names.join(" | "))
        };

        let post_attention_norm_q = find_qtensor(&[
            format!("{prefix}post_attention_norm.weight"),
            format!("{prefix}post_attention_norm"),
            format!("{prefix}ffn_norm.weight"),
            format!("{prefix}ffn_norm"),
            format!("{prefix}post_attention_layernorm.weight"),
            format!("{prefix}post_attention_layernorm"),
        ])?;
        let post_attention_norm = RmsNorm::from_qtensor(post_attention_norm_q, rms_norm_eps)?;

        let ffn_gate_q = find_qtensor(&[
            format!("{prefix}ffn_gate.weight"),
            format!("{prefix}ffn_gate"),
        ])?;
        let ffn_gate = QMatMul::from_qtensor(ffn_gate_q)?;

        let ffn_up_q = find_qtensor(&[
            format!("{prefix}ffn_up.weight"),
            format!("{prefix}ffn_up"),
        ])?;
        let ffn_up = QMatMul::from_qtensor(ffn_up_q)?;

        let ffn_down_q = find_qtensor(&[
            format!("{prefix}ffn_down.weight"),
            format!("{prefix}ffn_down"),
        ])?;
        let ffn_down = QMatMul::from_qtensor(ffn_down_q)?;

        Ok(Self::new(ffn_gate, ffn_up, ffn_down, post_attention_norm))
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let norm_xs = self.post_attention_norm.forward(xs)?;
        let gate = self.ffn_gate.forward(&norm_xs)?;
        let up = self.ffn_up.forward(&norm_xs)?;
        let silu_gate = candle_nn::ops::silu(&gate)?;
        let h = silu_gate.broadcast_mul(&up)?;
        self.ffn_down.forward(&h)
    }
}

/// An individual block in the Qwen3.5 hybrid architecture.
///
/// In Bonsai-27B (64 layers total):
/// - 48 layers are `Ssm`: Gated Delta Net recurrent SSM + SwiGLU FFN
/// - 16 layers are `Attn`: GQA attention with fused Q-Gate + SwiGLU FFN (`(il + 1) % 4 == 0`)
#[derive(Debug, Clone)]
pub enum Qwen35Block {
    Ssm {
        layer: Qwen35SsmLayer,
        ffn: SwiGluFfn,
    },
    Attn {
        layer: Qwen35AttnLayer,
        ffn: SwiGluFfn,
    },
}

impl Qwen35Block {
    #[inline]
    pub fn is_ssm(&self) -> bool {
        matches!(self, Self::Ssm { .. })
    }

    #[inline]
    pub fn is_attn(&self) -> bool {
        matches!(self, Self::Attn { .. })
    }
}

/// Full Qwen3.5 hybrid model (e.g. Bonsai-27B) supporting single-token decode,
/// multi-token prefill, rolling KV caching, and exact speculative rollback.
pub struct Qwen35Model {
    pub tok_embeddings: candle_nn::Embedding,
    pub blocks: Vec<Qwen35Block>,
    pub output_norm: RmsNorm,
    pub output: QMatMul,
    pub recurrent_state: Qwen35RecurrentState,
    pub cos: Tensor,
    pub sin: Tensor,
    pub total_tokens_seen: usize,
    pub device: Device,
    pub config: Qwen35Config,
    /// History of recurrent state snapshots for exact speculative rollback
    pub state_snapshots: Vec<(usize, Qwen35StateSnapshot)>,
}

impl Qwen35Model {
    /// Roll back attention KV cache on all 16 attention layers and restore recurrent state
    /// of all 48 SSM layers to position `pos`.
    pub fn rollback_kv(&mut self, pos: usize) -> Result<()> {
        if pos > self.total_tokens_seen {
            return Err(candle::Error::Msg(format!(
                "Cannot rollback to position {pos} greater than current pos {}",
                self.total_tokens_seen
            )));
        }
        if pos == self.total_tokens_seen {
            return Ok(());
        }

        let count = self.total_tokens_seen - pos;
        for block in &mut self.blocks {
            if let Qwen35Block::Attn { layer, .. } = block {
                layer.kv_cache.discard_tail(count);
            }
        }

        if pos == 0 {
            self.recurrent_state.reset();
            self.state_snapshots.clear();
        } else if let Some(idx) = self.state_snapshots.iter().rposition(|(p, _)| *p == pos) {
            self.recurrent_state.restore(&self.state_snapshots[idx].1);
            self.state_snapshots.truncate(idx);
        } else {
            candle::bail!("No recurrent state snapshot found for position {pos}");
        }

        self.total_tokens_seen = pos;
        Ok(())
    }

    /// Reset all attention KV caches and recurrent SSM states to position 0.
    pub fn reset_kv(&mut self) {
        for block in &mut self.blocks {
            if let Qwen35Block::Attn { layer, .. } = block {
                layer.reset_kv_cache();
            }
        }
        self.recurrent_state.reset();
        self.state_snapshots.clear();
        self.total_tokens_seen = 0;
    }

    /// Current global token sequence position.
    #[inline]
    pub fn current_kv_pos(&self) -> usize {
        self.total_tokens_seen
    }

    /// Current resident buffer length in the attention KV caches.
    pub fn kv_buffer_len(&self) -> usize {
        for block in &self.blocks {
            if let Qwen35Block::Attn { layer, .. } = block {
                return layer.kv_cache.current_pos();
            }
        }
        self.total_tokens_seen
    }

    /// Append key and value tensors to all attention layers without rolling eviction.
    pub fn append_kv(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let seq_len = k.dim(2)?;
        for block in &mut self.blocks {
            if let Qwen35Block::Attn { layer, .. } = block {
                layer.kv_cache.append(k, v)?;
            }
        }
        self.total_tokens_seen += seq_len;
        Ok(())
    }

    /// Append key and value tensors with rolling window eviction.
    pub fn append_kv_rolling(&mut self, k: &Tensor, v: &Tensor, window: usize) -> Result<()> {
        let seq_len = k.dim(2)?;
        for block in &mut self.blocks {
            if let Qwen35Block::Attn { layer, .. } = block {
                layer.kv_cache.append_rolling(k, v, window)?;
            }
        }
        self.total_tokens_seen += seq_len;
        Ok(())
    }

    /// Standard forward pass without rolling window eviction.
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_internal(input_ids, None)
    }

    fn embed_input(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embed_dev = self.tok_embeddings.embeddings().device();
        let input_on_embed = if input_ids.device().same_device(embed_dev) {
            input_ids.clone()
        } else {
            input_ids.to_device(embed_dev)?
        };
        let xs_embed = self.tok_embeddings.forward(&input_on_embed)?;
        if xs_embed.device().same_device(&self.device) {
            Ok(xs_embed)
        } else {
            xs_embed.to_device(&self.device)
        }
    }

    /// Forward pass with optional rolling window eviction on attention layers.
    ///
    /// Saves a snapshot of `recurrent_state` at each token step to enable $O(1)$ speculative rollback.
    pub fn forward_internal(
        &mut self,
        input_ids: &Tensor,
        rolling_window: Option<usize>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids.dims2()?;
        if b_sz != 1 {
            candle::bail!("Qwen35Model currently supports batch_size = 1, got {b_sz}");
        }

        if seq_len == 1 {
            let current_pos = self.total_tokens_seen;
            self.state_snapshots.push((current_pos, self.recurrent_state.snapshot()?));
            if self.state_snapshots.len() > 16 {
                self.state_snapshots.drain(0..self.state_snapshots.len() - 8);
            }

            let mut xs = self.embed_input(input_ids)?; // [1, 1, hidden_size]
            let mut ssm_idx = 0;
            for block in &mut self.blocks {
                match block {
                    Qwen35Block::Ssm { layer, ffn } => {
                        let state = self.recurrent_state.get_ssm_layer_mut(ssm_idx).ok_or_else(|| {
                            candle::Error::Msg(format!("SSM state out of range at index {ssm_idx}"))
                        })?;
                        let attn_out = layer.forward_decode(&xs, state)?;
                        let h = (&xs + &attn_out)?;
                        let ffn_out = ffn.forward(&h)?;
                        xs = (&h + &ffn_out)?;
                        ssm_idx += 1;
                    }
                    Qwen35Block::Attn { layer, ffn } => {
                        let attn_out = layer.forward_with_rolling(
                            &xs,
                            &self.cos,
                            &self.sin,
                            current_pos,
                            rolling_window,
                        )?;
                        let h = (&xs + &attn_out)?;
                        let ffn_out = ffn.forward(&h)?;
                        xs = (&h + &ffn_out)?;
                    }
                }
            }
            self.total_tokens_seen += 1;
            let xs = self.output_norm.forward(&xs)?;
            let logits = self.output.forward(&xs)?;
            Ok(logits)
        } else {
            let mut all_logits = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let token_pos = self.total_tokens_seen;
                self.state_snapshots.push((token_pos, self.recurrent_state.snapshot()?));
                if self.state_snapshots.len() > 16 {
                    self.state_snapshots.drain(0..self.state_snapshots.len() - 8);
                }

                let single_token = input_ids.narrow(1, t, 1)?; // [1, 1]
                let mut xs = self.embed_input(&single_token)?;
                let mut ssm_idx = 0;
                for block in &mut self.blocks {
                    match block {
                        Qwen35Block::Ssm { layer, ffn } => {
                            let state = self.recurrent_state.get_ssm_layer_mut(ssm_idx).ok_or_else(|| {
                                candle::Error::Msg(format!("SSM state out of range at index {ssm_idx}"))
                            })?;
                            let attn_out = layer.forward_decode(&xs, state)?;
                            let h = (&xs + &attn_out)?;
                            let ffn_out = ffn.forward(&h)?;
                            xs = (&h + &ffn_out)?;
                            ssm_idx += 1;
                        }
                        Qwen35Block::Attn { layer, ffn } => {
                            let attn_out = layer.forward_with_rolling(
                                &xs,
                                &self.cos,
                                &self.sin,
                                token_pos,
                                rolling_window,
                            )?;
                            let h = (&xs + &attn_out)?;
                            let ffn_out = ffn.forward(&h)?;
                            xs = (&h + &ffn_out)?;
                        }
                    }
                }
                self.total_tokens_seen += 1;
                let xs = self.output_norm.forward(&xs)?;
                let logits = self.output.forward(&xs)?;
                all_logits.push(logits);
            }
            let refs: Vec<&Tensor> = all_logits.iter().collect();
            Tensor::cat(&refs, 1)
        }
    }

    /// Load Qwen35Model from GGUF content using default maximum sequence capacity.
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_with_max_seq_len(ct, reader, None, device)
    }

    /// Load Qwen35Model from GGUF content with custom max sequence length.
    pub fn from_gguf_with_max_seq_len<R: std::io::Seek + std::io::Read>(
        ct: &gguf_file::Content,
        reader: &mut R,
        max_seq_len: Option<usize>,
        device: &Device,
    ) -> Result<Self> {
        let config = Qwen35Config::from_gguf(ct)?;
        let head_dim = config.head_dim;
        let rope_max_len = max_seq_len.map(|m| m + 1024).unwrap_or(32768).max(65536);
        let kv_max_len = max_seq_len.map(|m| m + 1024).unwrap_or(32768);

        // 1. Embeddings - kept in CPU memory to save ~5.1 GB of GPU VRAM
        let tok_embeddings = ct.tensor(reader, "token_embd.weight", &Device::Cpu)?;
        let tok_embeddings = tok_embeddings.dequantize(&Device::Cpu)?;
        let tok_embeddings = candle_nn::Embedding::new(tok_embeddings, config.hidden_size);

        // 2. Output RMSNorm
        let output_norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            config.rms_norm_eps,
        )?;

        // 3. Output projection
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(v) => QMatMul::from_qtensor(v)?,
            _ => QMatMul::from_qtensor(ct.tensor(reader, "token_embd.weight", device)?)?,
        };

        // 4. Precompute RoPE tables
        let (cos, sin) = precompute_freqs_cis(
            head_dim,
            config.rope_theta as f32,
            rope_max_len,
            device,
        )?;

        // 5. Zero-initialized recurrent state
        let recurrent_state = Qwen35RecurrentState::new(&config, device)?;

        // 6. Interleaved blocks (48 SSM + 16 full GQA attention)
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let ffn = SwiGluFfn::from_gguf(ct, reader, layer_idx, config.rms_norm_eps, device)?;
            if config.is_full_attn(layer_idx) {
                let layer = Qwen35AttnLayer::from_gguf_with_max_seq_len(
                    ct,
                    reader,
                    layer_idx,
                    &config,
                    kv_max_len,
                    device,
                )?;
                blocks.push(Qwen35Block::Attn { layer, ffn });
            } else {
                let layer = Qwen35SsmLayer::from_gguf(
                    ct,
                    reader,
                    layer_idx,
                    &config,
                    device,
                )?;
                blocks.push(Qwen35Block::Ssm { layer, ffn });
            }
        }

        Ok(Self {
            tok_embeddings,
            blocks,
            output_norm,
            output,
            recurrent_state,
            cos,
            sin,
            total_tokens_seen: 0,
            device: device.clone(),
            config,
            state_snapshots: Vec::new(),
        })
    }
}
