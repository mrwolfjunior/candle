use candle_speculative_server::{
    server::{create_app, AppState},
    CliArgs,
};
use clap::Parser;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

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
