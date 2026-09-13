pub mod engine;
pub mod kv_cache;
pub mod model;
pub mod server;

pub use engine::SuperDraftSpeculativeEngine;
pub use model::{Bonsai27BWithKv, QuantizedQwen2WithKv};

use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Asymmetric speculative inference server for dual-GPU")]
pub struct CliArgs {
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    #[arg(long, default_value = "cuda:0")]
    pub draft_device: String,

    #[arg(long, default_value = "cuda:1")]
    pub target_device: String,

    #[arg(long)]
    pub draft_model: Option<String>,

    #[arg(long)]
    pub target_model: Option<String>,

    #[arg(long)]
    pub tokenizer: Option<String>,

    #[arg(long, default_value_t = 4)]
    pub gamma: usize,

    #[arg(long, default_value_t = 65536)]
    pub max_context: usize,

    #[arg(long, default_value_t = false)]
    pub mock: bool,
}

pub fn version() -> &'static str {
    "0.1.0"
}
