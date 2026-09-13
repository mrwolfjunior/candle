# candle-quantized-qwen3-moe

Candle implementation for running quantized Qwen3 MoE models (e.g. `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` or `Qwen3-30B-A3B-Instruct-2507-GGUF`) in pure Rust.

## Features & Optimizations
- **Architectural Compatibility**: Runs seamlessly on both modern Tensor Core GPUs (Turing `sm_75`, Ampere `sm_80+`) and Pascal GPUs (`sm_61`, e.g. Tesla P40) via dynamic DP4A/FP32 GEMM routing.
- **CPU Expert Sorting**: Automatically offloads expert top-$k$ sorting to CPU on sequences longer than 1,024 tokens, avoiding GPU shared memory limits.
- **Chunked Prefill (`--chunk-size <N>`)**: Allows processing large prompt files (e.g. 16k+ context) in manageable slices (e.g. `--chunk-size 512`), capping attention activation memory and preventing $O(L^2)$ VRAM spikes.
- **Embedded Tokenizer Support**: Automatically reads and extracts tokenizers directly from the `.gguf` file metadata if a separate `tokenizer.json` is not provided.

## Running the Example

### Basic CLI Inference
```bash
cargo run --features cuda --example quantized-qwen3-moe --release -- \
  --model /path/to/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  --prompt "Write an asynchronous TCP server in Rust using tokio." \
  --chunk-size 512 \
  --device-id 0 \
  --sample-len 512 \
  --temperature 0.0
```

### Long-Context / Repository Ingestion with Prompt File
```bash
cargo run --features cuda --example quantized-qwen3-moe --release -- \
  --model /path/to/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  --prompt-file /path/to/repo_context.txt \
  --chunk-size 512 \
  --device-id 0 \
  --sample-len 1024 \
  --temperature 0.0
```

### Using Hugging Face Pre-downloaded Models
Models available via `--which` argument: `16b_q2k`, `16b_q4k`, `16b_q6k`, `16b_q80`; `32b_q2k`, `32b_q4k`, `32b_q6k`, `32b_q80`:
```bash
cargo run --features cuda --example quantized-qwen3-moe --release -- \
  --which 32b_q4k \
  --prompt "A train is travelling at 120mph, how far does it travel in 3 minutes 30 seconds?"
```
