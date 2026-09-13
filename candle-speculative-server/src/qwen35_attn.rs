use candle::{
    quantized::{gguf_file, QMatMul, QTensor},
    DType, Device, Module, Result, Tensor,
};
use candle_transformers::quantized_nn::RmsNorm;

use crate::{
    kv_cache::InPlaceKvCache,
    qwen35_state::Qwen35Config,
};

/// Builds causal attention mask for prefill or batched tokens.
fn build_causal_mask(
    q_len: usize,
    kv_len: usize,
    q_start_pos: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    if q_len == 1 {
        return Tensor::zeros((1, 1, 1, kv_len), dtype, device);
    }
    let mut mask = vec![0f32; q_len * kv_len];
    let kv_offset = if q_start_pos + q_len > kv_len {
        (q_start_pos + q_len) - kv_len
    } else {
        0
    };
    for q in 0..q_len {
        let query_pos = q_start_pos + q;
        for k in 0..kv_len {
            let key_pos = k + kv_offset;
            if key_pos > query_pos {
                mask[q * kv_len + k] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(mask, (1, 1, q_len, kv_len), device)?.to_dtype(dtype)
}

/// Full Grouped-Query Attention (GQA) layer with fused Query-Gate projection
/// for Qwen3.5 hybrid architectures (e.g. Bonsai-27B).
///
/// In Bonsai-27B, every 4th layer (`(il + 1) % 4 == 0`, e.g. layers 3, 7, 11, ..., 63)
/// is a full attention layer equipped with:
/// - Fused Q + Gate projection (`[12288, 5120]`) splitting into Query (`[6144]`) and Gate (`[6144]`)
/// - Key and Value projections (`[1024, 5120]`) for 4 KV heads
/// - Per-head RMSNorm on Q and K (`[256]`)
/// - Rotary Position Embedding (RoPE)
/// - Grouped-Query Attention with 24 query heads and 4 KV heads (6x repetition)
/// - Sigmoid gating on the attention output prior to final projection (`[5120, 6144]`)
#[derive(Debug, Clone)]
pub struct Qwen35AttnLayer {
    /// Pre-attention layer normalization [5120], eps = 1e-6
    pub attn_norm: RmsNorm,
    /// Fused Query + Gate projection [12288, 5120] (24 heads * 256 head_dim * 2)
    pub attn_q: QMatMul,
    /// Key projection [1024, 5120] (4 KV heads * 256 head_dim)
    pub attn_k: QMatMul,
    /// Value projection [1024, 5120] (4 KV heads * 256 head_dim)
    pub attn_v: QMatMul,
    /// Per-head Query RMSNorm [256], eps = 1e-6
    pub attn_q_norm: RmsNorm,
    /// Per-head Key RMSNorm [256], eps = 1e-6
    pub attn_k_norm: RmsNorm,
    /// Attention output projection [5120, 6144]
    pub attn_output: QMatMul,
    /// Dedicated in-place Key-Value cache
    pub kv_cache: InPlaceKvCache,
    /// Architectural hyper-parameters
    pub config: Qwen35Config,
}

impl Qwen35AttnLayer {
    /// Construct an attention layer with explicit component weights and cache.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        attn_norm: RmsNorm,
        attn_q: QMatMul,
        attn_k: QMatMul,
        attn_v: QMatMul,
        attn_q_norm: RmsNorm,
        attn_k_norm: RmsNorm,
        attn_output: QMatMul,
        kv_cache: InPlaceKvCache,
        config: Qwen35Config,
    ) -> Self {
        Self {
            attn_norm,
            attn_q,
            attn_k,
            attn_v,
            attn_q_norm,
            attn_k_norm,
            attn_output,
            kv_cache,
            config,
        }
    }

    /// Load an attention layer from GGUF content at block index `layer_idx` with default 32768 KV capacity.
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &gguf_file::Content,
        reader: &mut R,
        layer_idx: usize,
        config: &Qwen35Config,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_with_max_seq_len(ct, reader, layer_idx, config, 32768, device)
    }

    /// Load an attention layer from GGUF content with custom KV cache max capacity.
    pub fn from_gguf_with_max_seq_len<R: std::io::Seek + std::io::Read>(
        ct: &gguf_file::Content,
        reader: &mut R,
        layer_idx: usize,
        config: &Qwen35Config,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}.");

        let find_qtensor = |ct: &gguf_file::Content, reader: &mut R, names: &[String]| -> Result<QTensor> {
            for name in names {
                if ct.tensor_infos.contains_key(name) {
                    return ct.tensor(reader, name, device);
                }
            }
            candle::bail!("cannot find tensor info for {}", names.join(" | "))
        };

        // 1. attn_norm.weight: RmsNorm [5120], eps = config.rms_norm_eps
        let attn_norm_q = find_qtensor(
            ct,
            reader,
            &[
                format!("{prefix}attn_norm.weight"),
                format!("{prefix}attn_norm"),
            ],
        )?;
        let attn_norm = RmsNorm::from_qtensor(attn_norm_q, config.rms_norm_eps)?;

        // 2. attn_q.weight: QMatMul [12288, 5120]
        let attn_q_q = find_qtensor(
            ct,
            reader,
            &[
                format!("{prefix}attn_q.weight"),
                format!("{prefix}attn_q"),
            ],
        )?;
        let attn_q = QMatMul::from_qtensor(attn_q_q)?;

        // 3. attn_k.weight: QMatMul [1024, 5120]
        let attn_k_q = find_qtensor(
            ct,
            reader,
            &[
                format!("{prefix}attn_k.weight"),
                format!("{prefix}attn_k"),
            ],
        )?;
        let attn_k = QMatMul::from_qtensor(attn_k_q)?;

        // 4. attn_v.weight: QMatMul [1024, 5120]
        let attn_v_q = find_qtensor(
            ct,
            reader,
            &[
                format!("{prefix}attn_v.weight"),
                format!("{prefix}attn_v"),
            ],
        )?;
        let attn_v = QMatMul::from_qtensor(attn_v_q)?;

        // 5. attn_q_norm.weight: RmsNorm [256], eps = config.rms_norm_eps
        let attn_q_norm_q = find_qtensor(
            ct,
            reader,
            &[
                format!("{prefix}attn_q_norm.weight"),
                format!("{prefix}attn_q_norm"),
            ],
        )?;
        let attn_q_norm = RmsNorm::from_qtensor(attn_q_norm_q, config.rms_norm_eps)?;

        // 6. attn_k_norm.weight: RmsNorm [256], eps = config.rms_norm_eps
        let attn_k_norm_q = find_qtensor(
            ct,
            reader,
            &[
                format!("{prefix}attn_k_norm.weight"),
                format!("{prefix}attn_k_norm"),
            ],
        )?;
        let attn_k_norm = RmsNorm::from_qtensor(attn_k_norm_q, config.rms_norm_eps)?;

        // 7. attn_output.weight: QMatMul [5120, 6144]
        let attn_output_q = find_qtensor(
            ct,
            reader,
            &[
                format!("{prefix}attn_output.weight"),
                format!("{prefix}attn_output"),
                format!("{prefix}attn_out.weight"),
                format!("{prefix}attn_out"),
            ],
        )?;
        let attn_output = QMatMul::from_qtensor(attn_output_q)?;

        // 8. InPlaceKvCache: (1, 4, 256, max_seq_len)
        let kv_dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };
        let kv_cache = InPlaceKvCache::new(
            1,
            config.num_key_value_heads,
            config.head_dim,
            max_seq_len,
            kv_dtype,
            device,
        )?;

        Ok(Self::new(
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

    /// Access reference to configuration.
    #[inline]
    pub fn config(&self) -> &Qwen35Config {
        &self.config
    }

    /// Access reference to KV cache.
    #[inline]
    pub fn kv_cache(&self) -> &InPlaceKvCache {
        &self.kv_cache
    }

    /// Access mutable reference to KV cache.
    #[inline]
    pub fn kv_cache_mut(&mut self) -> &mut InPlaceKvCache {
        &mut self.kv_cache
    }

    /// Reset KV cache to position 0.
    #[inline]
    pub fn reset_kv_cache(&mut self) {
        self.kv_cache.reset();
    }

    /// Roll back KV cache to a previous position.
    #[inline]
    pub fn rollback_kv_cache(&mut self, pos: usize) -> Result<()> {
        self.kv_cache.rollback(pos)
    }

    /// Forward pass without rolling window:
    /// `xs`: `[b_sz, seq_len, 5120]` (or `[b_sz, 5120]`).
    /// `cos`, `sin`: precomputed rotary table.
    /// `pos`: sequence offset in tokens.
    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        pos: usize,
    ) -> Result<Tensor> {
        self.forward_with_rolling(xs, cos, sin, pos, None)
    }

    /// Forward pass with optional rolling window eviction:
    ///
    /// # Mathematical Steps:
    /// 1. `norm_xs = self.attn_norm.forward(xs)`
    /// 2. `q_full = self.attn_q.forward(&norm_xs)` -> `[b_sz, seq_len, 12288]`
    ///    `q = q_full.narrow(2, 0, 6144).reshape([b_sz, seq_len, 24, 256])`
    ///    `gate = q_full.narrow(2, 6144, 6144)` -> `[b_sz, seq_len, 6144]`
    ///    `k = self.attn_k.forward(&norm_xs).reshape([b_sz, seq_len, 4, 256])`
    ///    `v = self.attn_v.forward(&norm_xs).reshape([b_sz, seq_len, 4, 256])`
    /// 3. Per-head RMSNorm on Q and K along head_dim (256):
    ///    `q = self.attn_q_norm.forward(&q)`
    ///    `k = self.attn_k_norm.forward(&k)`
    /// 4. RoPE on Q and K after transposing to `[b_sz, n_heads, seq_len, head_dim]`
    /// 5. Append K and V to KV cache, perform GQA (repeat KV 6x), compute causal attention
    /// 6. Sigmoid gate modulation: `gated_attn = attn_out * sigmoid(gate)`
    /// 7. Output projection: `out = self.attn_output.forward(&gated_attn)` -> `[b_sz, seq_len, 5120]`
    pub fn forward_with_rolling(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        pos: usize,
        rolling_window: Option<usize>,
    ) -> Result<Tensor> {
        let is_2d = xs.dims().len() == 2;
        let xs_3d = if is_2d {
            xs.unsqueeze(1)?
        } else {
            xs.clone()
        };

        let (b_sz, seq_len, hidden) = xs_3d.dims3()?;
        if hidden != self.config.hidden_size {
            candle::bail!(
                "expected hidden_size {}, got {}",
                self.config.hidden_size,
                hidden
            );
        }

        let num_heads = self.config.num_attention_heads; // 24
        let num_kv_heads = self.config.num_key_value_heads; // 4
        let head_dim = self.config.head_dim; // 256
        let q_dim = num_heads * head_dim; // 6144

        // 1. Pre-norm
        let norm_xs = self.attn_norm.forward(&xs_3d)?; // [b_sz, seq_len, 5120]

        // 2. Projections
        let q_full = self.attn_q.forward(&norm_xs)?; // [b_sz, seq_len, 12288]
        let q = q_full
            .narrow(2, 0, q_dim)?
            .reshape((b_sz, seq_len, num_heads, head_dim))?;
        let gate = q_full.narrow(2, q_dim, q_dim)?; // [b_sz, seq_len, 6144]

        let k = self
            .attn_k
            .forward(&norm_xs)?
            .reshape((b_sz, seq_len, num_kv_heads, head_dim))?;
        let v = self
            .attn_v
            .forward(&norm_xs)?
            .reshape((b_sz, seq_len, num_kv_heads, head_dim))?;

        // 3. Per-head RMSNorm on Q and K along head_dim
        let q = if q.is_contiguous() { q } else { q.contiguous()? };
        let q = self.attn_q_norm.forward(&q)?;
        let k = if k.is_contiguous() { k } else { k.contiguous()? };
        let k = self.attn_k_norm.forward(&k)?;

        // 4. Transpose to [b_sz, n_heads, seq_len, head_dim] for RoPE & Attention
        let q = q.transpose(1, 2)?.contiguous()?; // [b_sz, 24, seq_len, 256]
        let k = k.transpose(1, 2)?.contiguous()?; // [b_sz, 4, seq_len, 256]
        let v = v.transpose(1, 2)?.contiguous()?; // [b_sz, 4, seq_len, 256]

        // Rotary Embedding (RoPE)
        let cos_pos = if cos.dim(0)? == seq_len {
            cos.clone()
        } else {
            cos.narrow(0, pos, seq_len)?
        };
        let sin_pos = if sin.dim(0)? == seq_len {
            sin.clone()
        } else {
            sin.narrow(0, pos, seq_len)?
        };

        let cos_pos = if cos_pos.dtype() != q.dtype() {
            cos_pos.to_dtype(q.dtype())?
        } else {
            cos_pos
        };
        let sin_pos = if sin_pos.dtype() != q.dtype() {
            sin_pos.to_dtype(q.dtype())?
        } else {
            sin_pos
        };
        let cos_pos = if cos_pos.is_contiguous() {
            cos_pos
        } else {
            cos_pos.contiguous()?
        };
        let sin_pos = if sin_pos.is_contiguous() {
            sin_pos
        } else {
            sin_pos.contiguous()?
        };

        let q = candle_nn::rotary_emb::rope(&q, &cos_pos, &sin_pos)?;
        let k = candle_nn::rotary_emb::rope(&k, &cos_pos, &sin_pos)?;

        // 5. Append K and V to InPlaceKvCache
        let k_to_append = if k.dtype() != self.kv_cache.dtype() {
            k.to_dtype(self.kv_cache.dtype())?
        } else {
            k
        };
        let v_to_append = if v.dtype() != self.kv_cache.dtype() {
            v.to_dtype(self.kv_cache.dtype())?
        } else {
            v
        };

        if let Some(window) = rolling_window {
            self.kv_cache
                .append_rolling(&k_to_append, &v_to_append, window)?;
        } else {
            self.kv_cache.append(&k_to_append, &v_to_append)?;
        }

        // Retrieve full KV cache view
        let (k_all, v_all) = self.kv_cache.current_view()?;

        // Grouped-Query Attention (GQA): repeat KV heads 6x (24 / 4)
        let n_rep = num_heads / num_kv_heads;
        let k_all = candle_transformers::utils::repeat_kv(k_all, n_rep)?;
        let v_all = candle_transformers::utils::repeat_kv(v_all, n_rep)?;

        let k_all = if k_all.dtype() != q.dtype() {
            k_all.to_dtype(q.dtype())?
        } else {
            k_all
        };
        let v_all = if v_all.dtype() != q.dtype() {
            v_all.to_dtype(q.dtype())?
        } else {
            v_all
        };

        // Scaled dot-product attention
        let kv_len = k_all.dim(2)?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut att = (q.matmul(&k_all.t()?)? * scale)?; // [b_sz, num_heads, seq_len, kv_len]

        if seq_len > 1 {
            let mask = build_causal_mask(seq_len, kv_len, pos, att.device(), att.dtype())?;
            att = att.broadcast_add(&mask)?;
        }

        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let v_cont = if v_all.is_contiguous() {
            v_all
        } else {
            v_all.contiguous()?
        };
        let y = att.matmul(&v_cont)?; // [b_sz, num_heads, seq_len, head_dim]
        let y = y
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, seq_len, q_dim))?; // [b_sz, seq_len, 6144]

        // 6. Gate Multiplication: y * sigmoid(gate)
        let gate_typed = if gate.dtype() != y.dtype() {
            gate.to_dtype(y.dtype())?
        } else {
            gate
        };
        let gate_sigmoid = candle_nn::ops::sigmoid(&gate_typed)?;
        let gated_attn = y.broadcast_mul(&gate_sigmoid)?; // [b_sz, seq_len, 6144]

        // 7. Output Projection: [b_sz, seq_len, 5120]
        let out = self.attn_output.forward(&gated_attn)?;

        if is_2d {
            out.squeeze(1)
        } else {
            Ok(out)
        }
    }
}
