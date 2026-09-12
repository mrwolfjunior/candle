use candle_speculative_server::CliArgs;
use clap::Parser;

#[test]
fn test_cli_args_parsing_defaults() {
    let args = CliArgs::parse_from(["speculative-server"]);
    assert_eq!(args.host, "0.0.0.0");
    assert_eq!(args.port, 8080);
    assert_eq!(args.gamma, 4);
    assert_eq!(args.max_context, 65536);
    assert_eq!(args.draft_device, "cuda:0");
    assert_eq!(args.target_device, "cuda:1");
}

#[test]
fn test_cli_args_parsing_custom() {
    let args = CliArgs::parse_from([
        "speculative-server",
        "--host",
        "127.0.0.1",
        "--port",
        "9000",
        "--draft-device",
        "cuda:2",
        "--target-device",
        "cuda:3",
        "--gamma",
        "8",
        "--max-context",
        "32768",
        "--draft-model",
        "draft.gguf",
        "--target-model",
        "target.gguf",
        "--tokenizer",
        "tokenizer.json",
        "--mock",
    ]);
    assert_eq!(args.host, "127.0.0.1");
    assert_eq!(args.port, 9000);
    assert_eq!(args.gamma, 8);
    assert_eq!(args.max_context, 32768);
    assert_eq!(args.draft_device, "cuda:2");
    assert_eq!(args.target_device, "cuda:3");
    assert_eq!(args.draft_model.as_deref(), Some("draft.gguf"));
    assert_eq!(args.target_model.as_deref(), Some("target.gguf"));
    assert_eq!(args.tokenizer.as_deref(), Some("tokenizer.json"));
    assert!(args.mock);
}
