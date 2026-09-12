use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json,
    },
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub stream: bool,
}

fn default_temperature() -> f64 {
    0.0
}

fn default_max_tokens() -> usize {
    2048
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelsListResponse {
    pub object: String,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

pub struct AppState {
    pub model_name: String,
    pub is_mock: bool,
}

impl AppState {
    pub fn mock() -> Self {
        Self {
            model_name: "qwen2.5-14b-instruct-speculative".to_string(),
            is_mock: true,
        }
    }
}

pub fn create_app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .with_state(state)
}

async fn health_handler() -> &'static str {
    "OK"
}

async fn models_handler(State(state): State<Arc<AppState>>) -> Json<ModelsListResponse> {
    Json(ModelsListResponse {
        object: "list".to_string(),
        data: vec![ModelCard {
            id: state.model_name.clone(),
            object: "model".to_string(),
            created: 1710000000,
            owned_by: "candle-speculative-server".to_string(),
        }],
    })
}

async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    if req.stream {
        let stream = tokio_stream::iter(vec![
            Ok::<Event, Infallible>(Event::default().data(
                serde_json::to_string(&ChatCompletionChunk {
                    id: "chatcmpl-test".to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created: 1710000000,
                    model: state.model_name.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: Some("assistant".to_string()),
                            content: Some("Hello".to_string()),
                        },
                        finish_reason: None,
                    }],
                })
                .unwrap(),
            )),
            Ok::<Event, Infallible>(Event::default().data("[DONE]")),
        ]);

        Sse::new(stream)
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
            .into_response()
    } else {
        Json(ChatCompletionResponse {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion".to_string(),
            created: 1710000000,
            model: state.model_name.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: "Hello from candle speculative server!".to_string(),
                },
                finish_reason: "stop".to_string(),
            }],
        })
        .into_response()
    }
}
