use candle::{
    quantized::QMatMul,
    DType, Device, Module, Result, Tensor,
};
use candle_transformers::quantized_nn::RmsNorm;
use crate::kv_cache::InPlaceKvCache;

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn qwen2_5_1_5b() -> Self {
        Self {
            hidden_size: 1536,
            intermediate_size: 8960,
            vocab_size: 151936,
            num_hidden_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 32768,
        }
    }

    pub fn qwen2_5_14b() -> Self {
        Self {
            hidden_size: 5120,
            intermediate_size: 13824,
            vocab_size: 152064,
            num_hidden_layers: 48,
            num_attention_heads: 40,
            num_key_value_heads: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 131072,
        }
    }

    pub fn bonsai_27b() -> Self {
        Self {
            hidden_size: 5120,
            intermediate_size: 13824,
            vocab_size: 248320,
            num_hidden_layers: 64,
            num_attention_heads: 40,
            num_key_value_heads: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 32768,
        }
    }

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

        let hidden_size = find_u32("embedding_length")
            .ok_or_else(|| candle::Error::Msg("missing embedding_length in GGUF metadata".into()))?
            as usize;

        let num_hidden_layers = find_u32("block_count")
            .ok_or_else(|| candle::Error::Msg("missing block_count in GGUF metadata".into()))?
            as usize;

        let num_attention_heads = find_u32("attention.head_count")
            .ok_or_else(|| candle::Error::Msg("missing attention.head_count in GGUF metadata".into()))?
            as usize;

        let num_key_value_heads = find_u32("attention.head_count_kv")
            .map(|v| v as usize)
            .unwrap_or(num_attention_heads);

        let intermediate_size = find_u32("feed_forward_length")
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                if hidden_size == 5120 {
                    13824
                } else {
                    hidden_size * 4
                }
            });

        let max_position_embeddings = find_u32("context_length")
            .map(|v| v as usize)
            .unwrap_or(32768);

        let rope_theta = find_f32("rope.freq_base").unwrap_or(1_000_000.0);

        let rms_norm_eps = find_f32("attention.layer_norm_rms_epsilon")
            .map(|v| v as f64)
            .unwrap_or(1e-6);

        let vocab_size = if let Some(v) = find_u32("vocab_size") {
            v as usize
        } else if let Some(candle::quantized::gguf_file::Value::Array(tokens)) =
            ct.metadata.get("tokenizer.ggml.tokens")
        {
            tokens.len()
        } else if let Some(tinfo) = ct.tensor_infos.get("token_embd.weight") {
            tinfo.shape.dims()[0]
        } else if prefixes.contains(&"qwen35") {
            248320
        } else {
            151936
        };

        Ok(Self {
            hidden_size,
            intermediate_size,
            vocab_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            rms_norm_eps,
            rope_theta,
            max_position_embeddings,
        })
    }
}

pub struct Layer {
    pub attention_wq: QMatMul,
    pub attention_wk: QMatMul,
    pub attention_wv: QMatMul,
    pub attention_wo: QMatMul,
    pub attention_norm: RmsNorm,
    pub ffn_gate: QMatMul,
    pub ffn_down: QMatMul,
    pub ffn_up: QMatMul,
    pub ffn_norm: RmsNorm,
    pub kv_cache: InPlaceKvCache,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
}

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

impl Layer {
    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rolling_window: Option<usize>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, n_embd) = xs.dims3()?;
        let index_pos = self.kv_cache.current_pos();

        let norm_xs = self.attention_norm.forward(xs)?;
        let q = self.attention_wq.forward(&norm_xs)?;
        let k = self.attention_wk.forward(&norm_xs)?;
        let v = self.attention_wv.forward(&norm_xs)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let cos_pos = cos.narrow(0, index_pos, seq_len)?;
        let sin_pos = sin.narrow(0, index_pos, seq_len)?;
        let q = candle_nn::rotary_emb::rope(&q, &cos_pos, &sin_pos)?;
        let k = candle_nn::rotary_emb::rope(&k, &cos_pos, &sin_pos)?;

        if let Some(window) = rolling_window {
            self.kv_cache.append_rolling(&k, &v, window)?;
        } else {
            self.kv_cache.append(&k, &v)?;
        }

        let (k_all, v_all) = self.kv_cache.current_view()?;

        let n_rep = self.n_head / self.n_kv_head;
        let k_all = candle_transformers::utils::repeat_kv(k_all, n_rep)?;
        let v_all = candle_transformers::utils::repeat_kv(v_all, n_rep)?;

        let kv_len = k_all.dim(2)?;
        let mut att = (q.matmul(&k_all.t()?)? / (self.head_dim as f64).sqrt())?;

        if seq_len > 1 {
            let mask = build_causal_mask(seq_len, kv_len, index_pos, att.device(), att.dtype())?;
            att = att.broadcast_add(&mask)?;
        }

        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let v_cont = if v_all.is_contiguous() {
            v_all
        } else {
            v_all.contiguous()?
        };
        let y = att.matmul(&v_cont)?;
        let y = y.transpose(1, 2)?.reshape((b_sz, seq_len, n_embd))?;
        let y = self.attention_wo.forward(&y)?;
        let xs = (&y + xs)?;

        // FFN
        let norm_xs = self.ffn_norm.forward(&xs)?;
        let gate = self.ffn_gate.forward(&norm_xs)?;
        let up = self.ffn_up.forward(&norm_xs)?;
        let silu_gate = candle_nn::ops::silu(&gate)?;
        let h = silu_gate.broadcast_mul(&up)?;
        let down = self.ffn_down.forward(&h)?;
        let xs = (&down + &xs)?;

        Ok(xs)
    }
}

pub struct QuantizedQwen2WithKv {
    pub tok_embeddings: candle_nn::Embedding,
    pub layers: Vec<Layer>,
    pub norm: RmsNorm,
    pub output: QMatMul,
    pub config: Config,
    pub device: Device,
    pub cos: Tensor,
    pub sin: Tensor,
}

impl QuantizedQwen2WithKv {
    pub fn rollback_kv(&mut self, pos: usize) -> Result<()> {
        for layer in &mut self.layers {
            layer.kv_cache.rollback(pos)?;
        }
        Ok(())
    }

    pub fn reset_kv(&mut self) {
        for layer in &mut self.layers {
            layer.kv_cache.reset();
        }
    }

    pub fn current_kv_pos(&self) -> usize {
        self.layers.first().map(|l| l.kv_cache.current_pos()).unwrap_or(0)
    }

    pub fn append_kv(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        for layer in &mut self.layers {
            layer.kv_cache.append(k, v)?;
        }
        Ok(())
    }

    pub fn append_kv_rolling(&mut self, k: &Tensor, v: &Tensor, window: usize) -> Result<()> {
        for layer in &mut self.layers {
            layer.kv_cache.append_rolling(k, v, window)?;
        }
        Ok(())
    }

    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_internal(input_ids, None)
    }

    pub fn forward_internal(
        &mut self,
        input_ids: &Tensor,
        rolling_window: Option<usize>,
    ) -> Result<Tensor> {
        let (_b_sz, _seq_len) = input_ids.dims2()?;
        let mut xs = self.tok_embeddings.forward(input_ids)?;

        for layer in &mut self.layers {
            xs = layer.forward(&xs, &self.cos, &self.sin, rolling_window)?;
        }

        let xs = self.norm.forward(&xs)?;
        let logits = self.output.forward(&xs)?;
        Ok(logits)
    }

    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &candle::quantized::gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_with_max_seq_len(ct, reader, None, device)
    }

    pub fn from_gguf_with_max_seq_len<R: std::io::Seek + std::io::Read>(
        ct: &candle::quantized::gguf_file::Content,
        reader: &mut R,
        max_seq_len: Option<usize>,
        device: &Device,
    ) -> Result<Self> {
        let config = Config::from_gguf(ct)?;
        let head_dim = config.head_dim();
        let kv_max_len = max_seq_len
            .map(|m| m + 1024)
            .unwrap_or(config.max_position_embeddings);

        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let tok_embeddings = candle_nn::Embedding::new(tok_embeddings, config.hidden_size);

        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            config.rms_norm_eps,
        )?;

        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(v) => QMatMul::from_qtensor(v)?,
            _ => QMatMul::from_qtensor(ct.tensor(reader, "token_embd.weight", device)?)?,
        };

        let (cos, sin) = precompute_freqs_cis(head_dim, config.rope_theta, kv_max_len, device)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo = ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;
            let attention_norm = ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;

            let ffn_gate = ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
            let ffn_down = ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
            let ffn_up = ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;

            let kv_cache = InPlaceKvCache::new(
                1,
                config.num_key_value_heads,
                head_dim,
                kv_max_len,
                DType::F32,
                device,
            )?;

            layers.push(Layer {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, config.rms_norm_eps)?,
                ffn_gate: QMatMul::from_qtensor(ffn_gate)?,
                ffn_down: QMatMul::from_qtensor(ffn_down)?,
                ffn_up: QMatMul::from_qtensor(ffn_up)?,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, config.rms_norm_eps)?,
                kv_cache,
                n_head: config.num_attention_heads,
                n_kv_head: config.num_key_value_heads,
                head_dim,
            });
        }

        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output,
            config,
            device: device.clone(),
            cos,
            sin,
        })
    }
}

pub struct Bonsai27BWithKv {
    pub model: QuantizedQwen2WithKv,
    pub rolling_window: usize,
}

impl Bonsai27BWithKv {
    pub const DEFAULT_ROLLING_WINDOW: usize = 8192;

    pub fn new(model: QuantizedQwen2WithKv, rolling_window: usize) -> Self {
        Self {
            model,
            rolling_window,
        }
    }

    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &candle::quantized::gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_with_window(ct, reader, Self::DEFAULT_ROLLING_WINDOW, device)
    }

    pub fn from_gguf_with_window<R: std::io::Seek + std::io::Read>(
        ct: &candle::quantized::gguf_file::Content,
        reader: &mut R,
        rolling_window: usize,
        device: &Device,
    ) -> Result<Self> {
        let model = QuantizedQwen2WithKv::from_gguf_with_max_seq_len(
            ct,
            reader,
            Some(rolling_window),
            device,
        )?;
        Ok(Self {
            model,
            rolling_window,
        })
    }

    pub fn rolling_window(&self) -> usize {
        self.rolling_window
    }

    pub fn rollback_kv(&mut self, pos: usize) -> Result<()> {
        let clamped = pos.min(self.rolling_window).min(self.model.current_kv_pos());
        self.model.rollback_kv(clamped)
    }

    pub fn reset_kv(&mut self) {
        self.model.reset_kv()
    }

    pub fn current_kv_pos(&self) -> usize {
        self.model.current_kv_pos()
    }

    pub fn append_kv(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        self.model.append_kv_rolling(k, v, self.rolling_window)
    }

    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.model.forward_internal(input_ids, Some(self.rolling_window))
    }
}

pub fn precompute_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}
