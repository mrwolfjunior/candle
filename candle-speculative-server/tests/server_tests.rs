use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use candle_speculative_server::server::{create_app, AppState, ChatCompletionResponse};
use std::sync::Arc;
use tower::ServiceExt;

#[tokio::test]
async fn test_health_endpoint() {
    let state = Arc::new(AppState::mock());
    let app = create_app(state);

    let response = app
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_models_endpoint() {
    let state = Arc::new(AppState::mock());
    let app = create_app(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_chat_completions_non_stream() {
    let state = Arc::new(AppState::mock());
    let app = create_app(state);

    let req_body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "stream": false
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let res: ChatCompletionResponse = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(res.choices.len(), 1);
    assert_eq!(res.choices[0].finish_reason, "stop");
}

#[tokio::test]
async fn test_chat_completions_stream() {
    let state = Arc::new(AppState::mock());
    let app = create_app(state);

    let req_body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "stream": true
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .unwrap()
        .to_str()
        .unwrap();
    assert!(content_type.contains("text/event-stream"));
}
