use candle::{
    quantized::QMatMul,
    DType, Device, Result, Tensor,
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
