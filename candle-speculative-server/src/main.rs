use candle_speculative_server::{
    server::{create_app, AppState},
    Bonsai27BWithKv, CliArgs, QuantizedQwen2WithKv, SuperDraftSpeculativeEngine,
};
use clap::Parser;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn parse_device(device_str: &str) -> anyhow::Result<candle::Device> {
    if device_str == "cpu" {
        Ok(candle::Device::Cpu)
    } else if let Some(id) = device_str.strip_prefix("cuda:") {
        let id: usize = id.parse()?;
        Ok(candle::Device::new_cuda(id)?)
    } else if device_str == "cuda" {
        Ok(candle::Device::new_cuda(0)?)
    } else {
        anyhow::bail!("Unsupported device: {device_str}")
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = CliArgs::parse();
    tracing::info!("Starting speculative server on {}:{}", args.host, args.port);
    tracing::info!("Draft device: {}, Target device: {}", args.draft_device, args.target_device);
    tracing::info!("Speculative lookahead gamma: {}, Max context: {}", args.gamma, args.max_context);

    let _engine = if let (Some(draft_path), Some(target_path)) = (&args.draft_model, &args.target_model) {
        if !args.mock {
            let draft_dev = parse_device(&args.draft_device)?;
            let target_dev = parse_device(&args.target_device)?;

            tracing::info!("Loading Bonsai-27B draft model from {draft_path} on {draft_dev:?}");
            let mut draft_file = std::fs::File::open(draft_path)?;
            let draft_content = candle::quantized::gguf_file::Content::read(&mut draft_file)?;
            let draft_bonsai = Bonsai27BWithKv::from_gguf_with_window(
                &draft_content,
                &mut draft_file,
                8192,
                &draft_dev,
            )?;

            tracing::info!("Loading target verifier from {target_path} on {target_dev:?}");
            let mut target_file = std::fs::File::open(target_path)?;
            let target_content = candle::quantized::gguf_file::Content::read(&mut target_file)?;
            let target_verifier = QuantizedQwen2WithKv::from_gguf_with_max_seq_len(
                &target_content,
                &mut target_file,
                Some(args.max_context),
                &target_dev,
            )?;

            let engine = SuperDraftSpeculativeEngine::new(draft_bonsai, target_verifier, args.gamma);
            tracing::info!("Super-Draft speculative engine initialized with gamma = {}", args.gamma);
            Some(engine)
        } else {
            tracing::info!("Mock mode enabled; skipping GGUF weights loading");
            None
        }
    } else {
        tracing::info!("No model paths provided; running in mock mode");
        None
    };

    let state = Arc::new(AppState {
        model_name: "qwen2.5-14b-instruct-speculative".to_string(),
        is_mock: args.mock,
    });

    let app = create_app(state);
    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("Server listening on http://{}", addr);

    axum::serve(listener, app).await?;
    Ok(())
}
