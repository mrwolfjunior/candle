use candle::{DType, Device, Tensor};
use candle_speculative_server::{
    kv_cache::InPlaceKvCache,
    model::{precompute_freqs_cis, Config, Layer},
    Bonsai27BWithKv as BonsaiModel, QuantizedQwen2WithKv as TargetModel,
    SuperDraftSpeculativeEngine, TargetVerifier,
};
use clap::Parser;
use std::time::Instant;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
#[command(author, version, about = "Benchmark Super-Draft Bonsai 27B across varying context depths up to 64k")]
struct BenchmarkArgs {
    /// Path to Bonsai-27B Q1_0 GGUF draft model
    #[arg(long)]
    draft_model: Option<String>,

    /// Path to Qwen3.8-27B or Qwen2.5 Q4_K_M GGUF target model
    #[arg(long)]
    target_model: Option<String>,

    /// Draft device (e.g. "cuda:0", "cpu")
    #[arg(long, default_value = "cuda:0")]
    draft_device: String,

    /// Target device (e.g. "cuda:1", "cpu")
    #[arg(long, default_value = "cuda:1")]
    target_device: String,

    /// Speculative lookahead gamma (draft proposals per step)
    #[arg(long, default_value_t = 4)]
    gamma: usize,

    /// Draft rolling window context size (default: 8192)
    #[arg(long, default_value_t = 8192)]
    draft_window: usize,

    /// Target max context length (default: 65536)
    #[arg(long, default_value_t = 65536)]
    max_context: usize,

    /// Comma-separated list of context depths to benchmark
    #[arg(long, default_value = "512,1024,4096,8192,16384,65536")]
    context_lens: String,

    /// Number of tokens to generate during decode evaluation
    #[arg(long, default_value_t = 64)]
    gen_tokens: usize,

    /// Run in mock mode with lightweight synthetic models
    #[arg(long, default_value_t = false)]
    mock: bool,

    /// Simulate realistic ~75% draft acceptance rate with periodic rollbacks in mock mode
    #[arg(long, default_value_t = false)]
    simulate_divergence: bool,

    /// Only verify draft model loading and single-token decode on draft device
    #[arg(long, default_value_t = false)]
    verify_draft_only: bool,
}

fn get_gpu_memory_info() -> Option<String> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"])
        .output()
        .ok()?;
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut lines = Vec::new();
        for line in stdout.lines() {
            let line = line.trim();
            if !line.is_empty() {
                lines.push(format!("    GPU {line}"));
            }
        }
        Some(lines.join("\n"))
    } else {
        None
    }
}

fn parse_device(device_str: &str) -> anyhow::Result<Device> {
    if device_str == "cpu" {
        Ok(Device::Cpu)
    } else if let Some(id) = device_str.strip_prefix("cuda:") {
        let id: usize = id.parse()?;
        Ok(Device::new_cuda(id)?)
    } else if device_str == "cuda" {
        Ok(Device::new_cuda(0)?)
    } else {
        anyhow::bail!("Unsupported device: {device_str}")
    }
}

fn create_mock_target(device: &Device, max_context: usize) -> anyhow::Result<TargetModel> {
    let vocab_size = 128;
    let hidden_size = 64;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = hidden_size / num_heads;

    let dummy_w = Tensor::zeros((hidden_size, hidden_size), DType::F32, device)?;
    let dummy_q = candle::quantized::QMatMul::Tensor(dummy_w.clone());
    let dummy_kv_w = Tensor::zeros((num_kv_heads * head_dim, hidden_size), DType::F32, device)?;
    let dummy_kv_q = candle::quantized::QMatMul::Tensor(dummy_kv_w);

    let dummy_norm_w = candle::quantized::QTensor::quantize(
        &Tensor::ones(hidden_size, DType::F32, device)?,
        candle::quantized::GgmlDType::F32,
    )?;
    let dummy_norm = candle_transformers::quantized_nn::RmsNorm::from_qtensor(dummy_norm_w, 1e-6)?;

    let total_context = max_context + 1024;
    let kv_cache = InPlaceKvCache::new(1, num_kv_heads, head_dim, total_context, DType::F32, device)?;

    let layer = Layer {
        attention_wq: dummy_q.clone(),
        attention_wk: dummy_kv_q.clone(),
        attention_wv: dummy_kv_q,
        attention_wo: dummy_q.clone(),
        attention_norm: dummy_norm.clone(),
        ffn_gate: dummy_q.clone(),
        ffn_down: dummy_q.clone(),
        ffn_up: dummy_q,
        ffn_norm: dummy_norm.clone(),
        kv_cache,
        n_head: num_heads,
        n_kv_head: num_kv_heads,
        head_dim,
    };

    let (cos, sin) = precompute_freqs_cis(head_dim, 1_000_000.0, total_context, device)?;
    let mut embed_data = vec![0.0f32; vocab_size * hidden_size];
    for i in 0..hidden_size.min(vocab_size) {
        embed_data[i * hidden_size + i] = 1.0;
    }
    let embed_w = Tensor::from_vec(embed_data, (vocab_size, hidden_size), device)?;
    let tok_embeddings = candle_nn::Embedding::new(embed_w, hidden_size);

    let mut trans_matrix = vec![0.0f32; hidden_size * vocab_size];
    for in_tok in 0..hidden_size.min(vocab_size) {
        let out_tok = (in_tok + 1) % vocab_size;
        trans_matrix[in_tok * vocab_size + out_tok] = 100.0;
    }
    let lm_head_t = Tensor::from_vec(trans_matrix, (vocab_size, hidden_size), device)?;
    let lm_head = candle::quantized::QMatMul::Tensor(lm_head_t);

    let config = Config {
        vocab_size,
        hidden_size,
        intermediate_size: hidden_size * 4,
        num_hidden_layers: 1,
        num_attention_heads: num_heads,
        num_key_value_heads: num_kv_heads,
        max_position_embeddings: total_context,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
    };

    Ok(TargetModel {
        tok_embeddings,
        layers: vec![layer],
        norm: dummy_norm,
        output: lm_head,
        cos,
        sin,
        config,
        device: device.clone(),
        total_tokens_seen: 0,
    })
}

fn create_mock_draft(
    device: &Device,
    window_size: usize,
    max_context: usize,
    simulate_divergence: bool,
) -> anyhow::Result<BonsaiModel> {
    let mut target = create_mock_target(device, max_context)?;
    if simulate_divergence {
        let vocab_size = target.config.vocab_size;
        let hidden_size = target.config.hidden_size;
        let mut trans_matrix = vec![0.0f32; hidden_size * vocab_size];
        for in_tok in 0..hidden_size.min(vocab_size) {
            let out_tok = if in_tok % 4 == 0 {
                (in_tok + 2) % vocab_size // 25% divergence
            } else {
                (in_tok + 1) % vocab_size // 75% match
            };
            trans_matrix[in_tok * vocab_size + out_tok] = 100.0;
        }
        let lm_head_t = Tensor::from_vec(trans_matrix, (vocab_size, hidden_size), device)?;
        target.output = candle::quantized::QMatMul::Tensor(lm_head_t);
    }
    for layer in &mut target.layers {
        layer.kv_cache = InPlaceKvCache::new(
            1,
            target.config.num_key_value_heads,
            layer.head_dim,
            window_size,
            DType::F32,
            device,
        )?;
    }
    Ok(BonsaiModel::new(target, window_size))
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = BenchmarkArgs::parse();

    println!("===============================================================================");
    println!("     Super-Draft Bonsai-27B Dual-GPU Speculative Benchmark Suite             ");
    println!("===============================================================================");
    println!("Draft Device:       {}", args.draft_device);
    println!("Target Device:      {}", args.target_device);
    println!("Speculative Lookahead (gamma): {}", args.gamma);
    println!("Draft Window Size:  {} tokens", args.draft_window);
    println!("Target Max Context: {} tokens", args.max_context);
    println!("Tokens to Generate: {}", args.gen_tokens);
    println!("Mock Mode:          {}", args.mock);
    println!("Simulate Divergence: {}", args.simulate_divergence);
    println!("Verify Draft Only:  {}", args.verify_draft_only);
    println!("-------------------------------------------------------------------------------");

    let draft_dev = parse_device(&args.draft_device)?;
    let target_dev = parse_device(&args.target_device)?;

    if args.verify_draft_only {
        let draft_path = args.draft_model.unwrap_or_else(|| {
            "/mnt/data/LMStudio/lmstudio-community/Bonsai-27B-GGUF/Bonsai-27B-Q1_0.gguf".to_string()
        });
        println!("=== Draft Sanity Verification: {} on {} ===", draft_path, args.draft_device);
        let t_load = Instant::now();
        let mut draft_file = std::fs::File::open(&draft_path)?;
        let draft_content = candle::quantized::gguf_file::Content::read(&mut draft_file)?;
        let mut draft = BonsaiModel::from_gguf_with_window(
            &draft_content,
            &mut draft_file,
            args.draft_window,
            &draft_dev,
        )?;
        let load_duration = t_load.elapsed();
        println!("  Draft model loaded successfully in {:.2?}", load_duration);
        if let Some(gpu_mem) = get_gpu_memory_info() {
            println!("  Resident GPU Memory:\n{}", gpu_mem);
        }

        println!("  Executing single-token decode test on {}...", args.draft_device);
        let input_tensor = Tensor::new(&[[151644u32]], &draft_dev)?;
        let t_warmup = Instant::now();
        let logits = draft.forward(&input_tensor)?;
        let warmup_lat = t_warmup.elapsed();
        println!("  Warmup forward completed in {:.2?}, logits shape: {:?}", warmup_lat, logits.shape());
        let mut current_token = logits.squeeze(0)?.squeeze(0)?.argmax(candle::D::Minus1)?.to_scalar::<u32>()?;
        println!("  Initial decoded token: {}", current_token);

        println!("  Running {} decode iterations...", args.gen_tokens);
        let t_decode = Instant::now();
        let mut tokens = vec![current_token];
        for _ in 1..args.gen_tokens {
            let input = Tensor::new(&[[current_token]], &draft_dev)?;
            let l = draft.forward(&input)?;
            current_token = l.squeeze(0)?.squeeze(0)?.argmax(candle::D::Minus1)?.to_scalar::<u32>()?;
            tokens.push(current_token);
        }
        let decode_duration = t_decode.elapsed();
        let tok_per_sec = args.gen_tokens as f64 / decode_duration.as_secs_f64();
        let ms_per_tok = decode_duration.as_secs_f64() * 1000.0 / args.gen_tokens as f64;

        println!("===============================================================================");
        println!("  Draft Single-Token Decode Verification Summary");
        println!("===============================================================================");
        println!("  Draft Device:       {}", args.draft_device);
        println!("  Tokens Generated:   {}", args.gen_tokens);
        println!("  Total Decode Time:  {:.3} s", decode_duration.as_secs_f64());
        println!("  Latency:            {:.2} ms/token", ms_per_tok);
        println!("  Throughput:         {:.2} tokens/sec", tok_per_sec);
        println!("  Current KV Pos:     {}", draft.current_kv_pos());
        if let Some(gpu_mem) = get_gpu_memory_info() {
            println!("  Final GPU Memory:\n{}", gpu_mem);
        }
        println!("  Tokens Decoded:     {:?}", tokens);
        println!("===============================================================================");
        println!("Draft Sanity Verification: PASSED!");
        return Ok(());
    }

    let (draft_model, target_model) = if args.mock || args.draft_model.is_none() || args.target_model.is_none() {
        tracing::info!("Initializing mock/synthetic models for benchmark demonstration");
        let draft = create_mock_draft(&draft_dev, args.draft_window, args.max_context, args.simulate_divergence)?;
        let target = create_mock_target(&target_dev, args.max_context)?;
        (draft, TargetVerifier::from(target))
    } else {
        let draft_path = args.draft_model.as_ref().unwrap();
        let target_path = args.target_model.as_ref().unwrap();

        tracing::info!("Loading Bonsai-27B draft model from {draft_path} on {draft_dev:?} (window={})", args.draft_window);
        let mut draft_file = std::fs::File::open(draft_path)?;
        let draft_content = candle::quantized::gguf_file::Content::read(&mut draft_file)?;
        tracing::info!("Draft tensor count: {}", draft_content.tensor_infos.len());
        tracing::info!("Draft metadata count: {}", draft_content.metadata.len());
        for (k, v) in draft_content.metadata.iter() {
            if k.contains("general.architecture") || k.contains("block_count") || k.contains("context") || k.contains("attention") {
                tracing::info!("  Draft Meta: {} = {:?}", k, v);
            }
        }

        tracing::info!("Loading Target model from {target_path} on {target_dev:?} (max_context={})", args.max_context);
        let mut target_file = std::fs::File::open(target_path)?;
        let target_content = candle::quantized::gguf_file::Content::read(&mut target_file)?;
        tracing::info!("Target tensor count: {}", target_content.tensor_infos.len());
        tracing::info!("Target metadata count: {}", target_content.metadata.len());
        for (k, v) in target_content.metadata.iter() {
            if k.contains("general.architecture") || k.contains("block_count") || k.contains("context") || k.contains("attention") {
                tracing::info!("  Target Meta: {} = {:?}", k, v);
            }
        }

        let draft = BonsaiModel::from_gguf_with_window(
            &draft_content,
            &mut draft_file,
            args.draft_window,
            &draft_dev,
        )?;

        let target = BonsaiModel::from_gguf_with_window(
            &target_content,
            &mut target_file,
            args.max_context,
            &target_dev,
        )?;

        if let Some(gpu_mem) = get_gpu_memory_info() {
            println!("Resident GPU Memory after model loading:\n{}", gpu_mem);
        }

        (draft, TargetVerifier::from(target))
    };

    let mut engine = SuperDraftSpeculativeEngine::new(draft_model, target_model, args.gamma);

    let context_lengths: Vec<usize> = args
        .context_lens
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();

    println!();
    println!("+---------------+----------------------+----------------------+------------+------------+------------+------------+--------------+---------------+");
    println!("| Context Depth | Prefill Throughput   | Speculative Decode   | Accept (α) | Toks/Step  | Draft Lat. | Target Lat | Target/Draft | Target KV Mem |");
    println!("+---------------+----------------------+----------------------+------------+------------+------------+------------+--------------+---------------+");

    for &ctx_len in &context_lengths {
        if ctx_len > args.max_context {
            tracing::warn!("Skipping context length {} > max_context {}", ctx_len, args.max_context);
            continue;
        }

        engine.reset_kv();
        engine.reset_timings();

        // Construct synthetic input prompt of length ctx_len
        let prompt: Vec<u32> = (0..ctx_len as u32).map(|i| i % 60 + 1).collect();

        // 1. Prefill Benchmark
        let prefill_start = Instant::now();
        let first_token = engine.prefill(&prompt)?;
        let prefill_elapsed = prefill_start.elapsed();
        let prefill_tok_per_sec = ctx_len as f64 / prefill_elapsed.as_secs_f64().max(1e-6);

        // 2. Decode Speculative Benchmark
        let mut current_token = first_token;
        let mut tokens_emitted = 0;
        let mut steps = 0;
        let mut total_proposed = 0;
        let mut total_accepted_draft = 0;

        let decode_start = Instant::now();
        while tokens_emitted < args.gen_tokens {
            let res = engine.step(current_token)?;
            steps += 1;
            total_proposed += args.gamma;
            total_accepted_draft += res.num_accepted_draft;
            tokens_emitted += res.accepted_count();
            current_token = *res.accepted_tokens.last().unwrap();
        }
        let decode_elapsed = decode_start.elapsed();
        let decode_tok_per_sec = tokens_emitted as f64 / decode_elapsed.as_secs_f64().max(1e-6);
        let alpha = (total_accepted_draft as f64 / total_proposed as f64) * 100.0;
        let tau = tokens_emitted as f64 / steps as f64;

        let draft_ms = engine.draft_time.as_secs_f64() * 1000.0 / steps.max(1) as f64;
        let target_ms = engine.target_time.as_secs_f64() * 1000.0 / steps.max(1) as f64;
        let ratio = target_ms / draft_ms.max(1e-6);

        let target_kv_bytes = engine.target_verifier.kv_cache_bytes(ctx_len);
        let target_kv_mb = target_kv_bytes as f64 / (1024.0 * 1024.0);

        println!(
            "| {:>10} ctx | {:>13.1} tok/s | {:>13.1} tok/s | {:>9.1}% | {:>9.2}  | {:>7.2} ms | {:>7.2} ms | {:>10.2}x | {:>10.1} MB |",
            format!("{ctx_len}"),
            prefill_tok_per_sec,
            decode_tok_per_sec,
            alpha,
            tau,
            draft_ms,
            target_ms,
            ratio,
            target_kv_mb,
        );
    }

    println!("+---------------+----------------------+----------------------+------------+------------+------------+------------+--------------+---------------+");
    println!();
    if let Some(gpu_mem) = get_gpu_memory_info() {
        println!("Final GPU Memory:\n{}", gpu_mem);
    }
    println!("Benchmark run complete.");
    Ok(())
}
