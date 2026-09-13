use candle::{
    quantized::{gguf_file, QMatMul, QTensor},
    DType, Device, Module, Result, Tensor,
};
use candle_transformers::quantized_nn::RmsNorm;

use crate::qwen35_state::{Qwen35Config, Qwen35LayerState};

/// Gated Delta Net linear attention SSM layer for Qwen3.5 (e.g. Bonsai-27B).
///
/// Implements depthwise 1D convolution over mixed QKV projections, followed by
/// L2 normalization, head expansion, and the Gated Delta Net recurrence step.
#[derive(Debug, Clone)]
pub struct Qwen35SsmLayer {
    /// Pre-attention layer normalization [5120], eps = 1e-6
    pub attn_norm: RmsNorm,
    /// Mixed QKV projection [10240, 5120]
    pub attn_qkv: QMatMul,
    /// Gate projection (z) [6144, 5120]
    pub attn_gate: QMatMul,
    /// Depthwise 1D conv weights [10240, 4]
    pub ssm_conv1d: Tensor,
    /// Log decay rates [48]
    pub ssm_a: Tensor,
    /// Alpha projection [48, 5120]
    pub ssm_alpha: QMatMul,
    /// Beta projection [48, 5120]
    pub ssm_beta: QMatMul,
    /// Delta-t bias [48]
    pub ssm_dt: Tensor,
    /// Post-recurrent head normalization [128]
    pub ssm_norm: RmsNorm,
    /// Output projection [5120, 6144]
    pub ssm_out: QMatMul,
    /// Architectural config
    pub config: Qwen35Config,
}

#[inline]
fn softplus(xs: &Tensor) -> Result<Tensor> {
    (xs.exp()? + 1.0)?.log()
}

impl Qwen35SsmLayer {
    /// Create a new SSM layer from explicit component weights.
    pub fn new(
        attn_norm: RmsNorm,
        attn_qkv: QMatMul,
        attn_gate: QMatMul,
        ssm_conv1d: Tensor,
        ssm_a: Tensor,
        ssm_alpha: QMatMul,
        ssm_beta: QMatMul,
        ssm_dt: Tensor,
        ssm_norm: RmsNorm,
        ssm_out: QMatMul,
        config: Qwen35Config,
    ) -> Self {
        // Ensure ssm_conv1d is [conv_dim, conv_kernel]
        let mut conv1d = ssm_conv1d;
        if conv1d.dims().len() == 3 {
            if let Ok(sq) = conv1d.squeeze(1) {
                conv1d = sq;
            }
        }
        if conv1d.dims().len() == 2 && conv1d.dims2().map_or(false, |(d0, d1)| d0 == config.ssm_conv_kernel && d1 == config.ssm_conv_dim()) {
            if let Ok(t) = conv1d.t().and_then(|t| t.contiguous()) {
                conv1d = t;
            }
        }

        Self {
            attn_norm,
            attn_qkv,
            attn_gate,
            ssm_conv1d: conv1d,
            ssm_a,
            ssm_alpha,
            ssm_beta,
            ssm_dt,
            ssm_norm,
            ssm_out,
            config,
        }
    }

    /// Load an SSM layer from GGUF content at block index `layer_idx`.
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &gguf_file::Content,
        reader: &mut R,
        layer_idx: usize,
        config: &Qwen35Config,
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

        // 1. attn_norm.weight: RmsNorm [5120], eps = 1e-6
        let attn_norm_q = find_qtensor(ct, reader, &[format!("{prefix}attn_norm.weight")])?;
        let attn_norm = RmsNorm::from_qtensor(attn_norm_q, config.rms_norm_eps)?;

        // 2. attn_qkv.weight: QMatMul [10240, 5120]
        let attn_qkv_q = find_qtensor(ct, reader, &[format!("{prefix}attn_qkv.weight")])?;
        let attn_qkv = QMatMul::from_qtensor(attn_qkv_q)?;

        // 3. attn_gate.weight: QMatMul [6144, 5120]
        let attn_gate_q = find_qtensor(ct, reader, &[format!("{prefix}attn_gate.weight")])?;
        let attn_gate = QMatMul::from_qtensor(attn_gate_q)?;

        // 4. ssm_conv1d.weight: Tensor [10240, 4]
        let ssm_conv1d_q = find_qtensor(ct, reader, &[format!("{prefix}ssm_conv1d.weight")])?;
        let mut ssm_conv1d = ssm_conv1d_q.dequantize(device)?;
        if ssm_conv1d.dims().len() == 3 {
            ssm_conv1d = ssm_conv1d.squeeze(1)?;
        }
        if ssm_conv1d.dims2()? == (config.ssm_conv_kernel, config.ssm_conv_dim()) {
            ssm_conv1d = ssm_conv1d.t()?.contiguous()?;
        }

        // 5. ssm_a: Tensor [48]
        let ssm_a_q = find_qtensor(
            ct,
            reader,
            &[format!("{prefix}ssm_a"), format!("{prefix}ssm_a.weight")],
        )?;
        let ssm_a = ssm_a_q.dequantize(device)?.flatten_all()?;

        // 6. ssm_alpha.weight: QMatMul [48, 5120]
        let ssm_alpha_q = find_qtensor(ct, reader, &[format!("{prefix}ssm_alpha.weight")])?;
        let ssm_alpha = QMatMul::from_qtensor(ssm_alpha_q)?;

        // 7. ssm_beta.weight: QMatMul [48, 5120]
        let ssm_beta_q = find_qtensor(ct, reader, &[format!("{prefix}ssm_beta.weight")])?;
        let ssm_beta = QMatMul::from_qtensor(ssm_beta_q)?;

        // 8. ssm_dt.bias: Tensor [48]
        let ssm_dt_q = find_qtensor(
            ct,
            reader,
            &[format!("{prefix}ssm_dt.bias"), format!("{prefix}ssm_dt.weight")],
        )?;
        let ssm_dt = ssm_dt_q.dequantize(device)?.flatten_all()?;

        // 9. ssm_norm.weight: RmsNorm [128]
        let ssm_norm_q = find_qtensor(ct, reader, &[format!("{prefix}ssm_norm.weight")])?;
        let ssm_norm = RmsNorm::from_qtensor(ssm_norm_q, config.rms_norm_eps)?;

        // 10. ssm_out.weight: QMatMul [5120, 6144]
        let ssm_out_q = find_qtensor(ct, reader, &[format!("{prefix}ssm_out.weight")])?;
        let ssm_out = QMatMul::from_qtensor(ssm_out_q)?;

        Ok(Self::new(
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

    /// Architectural configuration reference.
    #[inline]
    pub fn config(&self) -> &Qwen35Config {
        &self.config
    }

    /// Single-token autoregressive decode forward pass:
    /// `xs: [B=1, L=1, hidden_size]` (or `[1, hidden_size]`).
    ///
    /// Updates convolutional state `state.conv_state` and recurrent delta net state
    /// `state.ssm_state` in place, returning the output projection tensor of shape `[1, 1, hidden_size]`.
    pub fn forward_decode(
        &mut self,
        xs: &Tensor,
        state: &mut Qwen35LayerState,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden) = match xs.dims() {
            [b, s, h] => (*b, *s, *h),
            [b, h] => (*b, 1, *h),
            _ => candle::bail!("expected 2D or 3D tensor for xs, got {:?}", xs.shape()),
        };

        if b_sz != 1 || seq_len != 1 {
            candle::bail!(
                "forward_decode currently supports single-token decode ([1, 1, hidden]), got shape {:?}",
                xs.shape()
            );
        }
        if hidden != self.config.hidden_size {
            candle::bail!(
                "expected hidden_size {}, got {}",
                self.config.hidden_size,
                hidden
            );
        }

        let xs_3d = if xs.dims().len() == 2 {
            xs.unsqueeze(1)?
        } else {
            xs.clone()
        };

        // 1. Pre-norm
        let norm_xs = self.attn_norm.forward(&xs_3d)?; // [1, 1, 5120]

        // 2. Projections
        let qkv_mixed = self.attn_qkv.forward(&norm_xs)?; // [1, 1, 10240]
        let z = self.attn_gate.forward(&norm_xs)?;         // [1, 1, 6144]

        let beta_raw = self.ssm_beta.forward(&norm_xs)?;  // [1, 1, 48]
        let beta = candle_nn::ops::sigmoid(&beta_raw)?;   // [1, 1, 48]

        let alpha_raw = self.ssm_alpha.forward(&norm_xs)?; // [1, 1, 48]
        let dt = self.ssm_dt.reshape((1, 1, self.config.ssm_dt_rank))?;
        let alpha_biased = alpha_raw.broadcast_add(&dt)?;  // [1, 1, 48]
        let alpha_softplus = softplus(&alpha_biased)?;     // [1, 1, 48]
        let a = self.ssm_a.reshape((1, 1, self.config.ssm_dt_rank))?;
        let decay_gate = alpha_softplus.broadcast_mul(&a)?; // [1, 1, 48]

        // 3. 1D Depthwise Conv
        let qkv_conv_dtype = if qkv_mixed.dtype() != state.conv_state.dtype() {
            qkv_mixed.to_dtype(state.conv_state.dtype())?
        } else {
            qkv_mixed
        };

        let conv_input = Tensor::cat(&[&state.conv_state, &qkv_conv_dtype], 1)?; // [1, 4, 10240]
        state.conv_state = conv_input.narrow(1, 1, self.config.ssm_conv_len())?.contiguous()?;

        let conv_in = conv_input.squeeze(0)?.t()?.to_dtype(self.ssm_conv1d.dtype())?; // [10240, 4]
        let conv_out = (conv_in * &self.ssm_conv1d)?.sum(1)?; // [10240]
        let conv_out_silu = candle_nn::ops::silu(&conv_out)?; // [10240]

        // 4. Split into Q, K, V
        let key_dim = self.config.ssm_n_group * self.config.ssm_d_state; // 16 * 128 = 2048
        let val_dim = self.config.ssm_inner_size; // 6144
        let n_qk_heads = self.config.ssm_n_group; // 16
        let n_v_heads = self.config.ssm_dt_rank;   // 48
        let d_state = self.config.ssm_d_state;     // 128

        let q = conv_out_silu.narrow(0, 0, key_dim)?.reshape((n_qk_heads, d_state))?;
        let k = conv_out_silu.narrow(0, key_dim, key_dim)?.reshape((n_qk_heads, d_state))?;
        let v = conv_out_silu.narrow(0, 2 * key_dim, val_dim)?.reshape((n_v_heads, d_state))?;

        // 5. L2 Normalization on Q and K
        let eps = self.config.rms_norm_eps;
        let q_norm_denom = q.sqr()?.sum_keepdim(1)?.affine(1.0, eps)?.sqrt()?;
        let q_norm = q.broadcast_div(&q_norm_denom)?; // [16, 128]

        let k_norm_denom = k.sqr()?.sum_keepdim(1)?.affine(1.0, eps)?.sqrt()?;
        let k_norm = k.broadcast_div(&k_norm_denom)?; // [16, 128]

        // 6. Head Expansion (16 -> 48)
        let n_rep = n_v_heads / n_qk_heads; // 3
        let q_exp = Tensor::cat(&vec![&q_norm.unsqueeze(1)?; n_rep], 1)?.reshape((n_v_heads, d_state))?; // [48, 128]
        let k_exp = Tensor::cat(&vec![&k_norm.unsqueeze(1)?; n_rep], 1)?.reshape((n_v_heads, d_state))?; // [48, 128]

        // 7. Recurrent Gated Delta Net Step
        let g = decay_gate.reshape((n_v_heads, 1, 1))?.exp()?;
        let mut s = state.ssm_state.to_dtype(DType::F32)?.broadcast_mul(&g)?; // [48, 128, 128]

        // Prediction: sk_h = s_h * k_h
        let k_col = k_exp.unsqueeze(2)?; // [48, 128, 1]
        let sk = s.matmul(&k_col)?.squeeze(2)?; // [48, 128]

        // Error & Delta: d_h = (v_h - sk_h) * beta_h
        let beta_col = beta.reshape((n_v_heads, 1))?; // [48, 1]
        let d = (v.sub(&sk)?).broadcast_mul(&beta_col)?; // [48, 128]

        // Update: s_h = s_h + (k_h (x) d_h)
        let d_row = d.unsqueeze(1)?; // [48, 1, 128]
        let kd = k_col.matmul(&d_row)?; // [48, 128, 128]
        s = (s + kd)?;

        // Head Output: o_h = s_h * q_h
        let q_col = q_exp.unsqueeze(2)?; // [48, 128, 1]
        let o = s.matmul(&q_col)?.squeeze(2)?; // [48, 128]

        // Update recurrent state
        state.ssm_state = if s.dtype() != state.ssm_state.dtype() {
            s.to_dtype(state.ssm_state.dtype())?
        } else {
            s
        };

        // 8. Gated RMSNorm
        let normed = self.ssm_norm.forward(&o)?; // [48, 128]
        let normed_flat = normed.reshape((1, 1, val_dim))?; // [1, 1, 6144]

        let z_silu = candle_nn::ops::silu(&z.to_dtype(normed_flat.dtype())?)?;
        let gated_out = (normed_flat * z_silu)?; // [1, 1, 6144]

        // 9. Output Projection
        self.ssm_out.forward(&gated_out) // [1, 1, 5120]
    }
}
