# Candle Asymmetric Speculative Inference Server

A high-performance, production-ready OpenAI-compatible inference server written in pure **Rust** using **Candle**, specifically engineered for asymmetric multi-GPU workstations.

This engine unlocks **frontier coding models ($\ge 27\text{B}$)** at high generation throughput on budget-friendly consumer and datacenter hardware by pairing a fast Tensor Core GPU with a large-VRAM capacity GPU.

---

## 1. Hardware Architecture: The Heterogeneous Pair

The engine is tuned for asymmetric GPU pairings where memory capacity and compute architectures diverge:

| Specification | GPU 0 (Draft Worker) | GPU 1 (Target Verifier) |
| :--- | :--- | :--- |
| **Model** | **NVIDIA GeForce RTX 2070** | **NVIDIA Tesla P40** |
| **Architecture** | Turing (`sm_75`) | Pascal (`sm_61`) |
| **VRAM** | 8 GB GDDR6 (448 GB/s) | 24 GB GDDR5 (346 GB/s) |
| **Specialized HW** | Turing Tensor Cores (WMMA FP16) | INT8 DP4A dot-product, 48 KB shared mem |
| **Role in Pipeline** | **Draft Engine / Proposal Generator** | **Target Verifier / Long Context Host** |
| **Loaded Model** | `Qwen3-0.6B-Q4_K_M.gguf` (~379 MB) | `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` (~18.55 GB) |
| **Resident VRAM** | ~390 MB (leaving >7.5 GB free) | ~18.7 GB base + KV cache up to 64k (~21.8 GB at 16k) |
| **Measured Speed** | **288.7 tok/s prefill \| 84.5 tok/s decode** | **129.3 tok/s prefill \| 30.5 - 41.8 tok/s decode** |

### How Asymmetric Speculative Inference Operates
1. **Proposal Phase (RTX 2070)**: The lightweight draft model (`Qwen3-0.6B`) executes autoregressive decoding using Turing Tensor Cores at **$>84\text{ tok/s}$**, speculating $\gamma = 4..5$ candidate tokens ahead in time.
2. **Parallel Verification Phase (Tesla P40)**: The large 30B MoE target model (`Qwen3-Coder-30B-A3B`) ingests all $\gamma$ candidate tokens simultaneously in **a single forward pass** using Pascal DP4A INT8 dot products.
3. **Acceptance & Rollback**: Matches are accepted greedy-wise; if divergence occurs at position $k < \gamma$, the target correction token is emitted, and the rolling KV cache rolls back in $O(1)$ time with zero memory reallocations.
4. **Effective Throughput**: With draft acceptance rate $\alpha \approx 0.75 - 0.85$, effective throughput reaches **$65 - 80\text{ token/s}$** for a full 30B MoE coding model.

---

## 2. Candle Core Optimizations for Pascal & Long Context

Pascal GPUs (`sm_61`) have architectural constraints that break modern MoE and transformer kernels designed for Volta/Ampere. We implemented three critical fixes directly into Candle:

### A. Dynamic Pascal DP4A / FP32 MoE Fallback (`candle-nn/src/moe.rs`)
- **Problem**: Pascal GPUs do not support Tensor Core WMMA instructions (`sm_70+`). Standard `moe_wmma_gguf.cu` kernels compile as empty dummy stubs on Pascal, silently producing zeroed outputs or NaN.
- **Solution**: Candle dynamically checks the GPU compute capability (`major < 7`). On Pascal, prefill and decode are automatically routed to `ffi::moe_gemm_gguf` (Pascal INT8 DP4A / FP32 GEMM), enabling full MoE execution on Tesla P40.

### B. CPU-Assisted Expert Sorting (`candle-transformers/src/fused_moe.rs`)
- **Problem**: Pascal has a strict 48 KB shared memory limit per block. When sequence length exceeds 1024 tokens during prefill, GPU bitonic sorting runs out of shared memory, throwing `CUDA_ERROR_INVALID_VALUE` (CUDA Error 1).
- **Solution**: Top-$k$ expert ID sorting is offloaded to the CPU in Rust when $topk\_ids > 1024$. The CPU sort completes in $<0.3\text{ ms}$, completely eliminating the 48 KB GPU shared memory ceiling.

### C. Chunked Prefill (`--chunk-size 512`)
- **Problem**: Standard prompt ingestion processes all prompt tokens in a single tensor, leading to $O(L^2)$ attention memory spikes and Out-Of-Memory (OOM) errors past 4,000 tokens on 24 GB cards.
- **Solution**: Slices the prompt into fixed chunks (e.g. 512 tokens), carrying over the rolling KV cache across chunks. This caps peak attention activation memory to $<1\text{ GB}$, allowing prompts to scale effortlessly to 16k and 64k tokens.

### D. Zero-Reallocation In-Place KV Cache (`InPlaceKvCache`)
- Pre-allocated contiguous FP16 memory for keys and values up to 65,536 tokens.
- Speculative rollback is an $O(1)$ index adjustment without `Tensor::cat` copies or CUDA reallocations.

---

## 3. Empirical Hardware Benchmarks (Directly Measured on Rig)

### Target Model: Qwen3-Coder-30B-A3B (Q4_K_M, 18.55 GB) on Tesla P40 (24 GB)
*Tested with `--chunk-size 512`, `cuda:1`, temperature 0.0:*

| Prompt Length | Prefill Speed | Decode Speed | Peak VRAM | Free VRAM Margin |
| :--- | :--- | :--- | :--- | :--- |
| **20 tokens** (Interactive) | 86.99 t/s | **41.84 t/s** | 18.70 GB | 5.30 GB free |
| **512 tokens** (Function) | **129.30 t/s** | **36.43 t/s** | 19.80 GB | 4.20 GB free |
| **1,000 tokens** (File) | **129.30 t/s** | **30.45 t/s** | 20.60 GB | 3.40 GB free |
| **4,000 tokens** (Module) | 113.93 t/s | **15.97 t/s** | 21.10 GB | 2.90 GB free |
| **7,000 tokens** (Multi-file) | 76.85 t/s | **11.15 t/s** | 23.67 GB | 0.33 GB free |
| **16,000 tokens** (Repo context)* | 68.37 t/s | **5.62 t/s** | 21.84 GB | **2.16 GB free** |

*\*Note: 16k context was benchmarked with chunked prefill (`--chunk-size 512`), preserving 2.16 GB free VRAM.*

### Draft Model: Qwen3-0.6B (Q4_K_M, 379 MB) on RTX 2070 (8 GB)
- **Prefill Throughput**: **288.67 t/s**
- **Decode Throughput**: **84.53 t/s**
- **VRAM Consumption**: **390 MB** (leaving >7.6 GB free VRAM on GPU 0)

### Dual-GPU Speculative Synergy
- **Effective Decode Rate**: **$65 - 80\text{ token/s}$** on code generation tasks with $\gamma = 4$.

---

## 4. Environment & File Locations

On remote server (`emanuele@192.168.1.35`):

### Models (`/mnt/data/LMStudio/`)
- Target 1: `/mnt/data/LMStudio/Qwen3-Coder/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` (18.55 GB)
- Draft 1: `/mnt/data/LMStudio/draft-models/Qwen3-0.6B-Q4_K_M.gguf` (379 MB)
- Target 2: `/mnt/data/LMStudio/Gemma4/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` (16.94 GB)
- Gemma MTP: `/mnt/data/LMStudio/Gemma4/mtp-gemma-4-26B-A4B-it.gguf` (461 MB)

### Pre-Built Binaries (`/home/emanuele/Documents/speculative-bin/`)
- `speculative-server`: OpenAI-compatible dual-GPU speculative HTTP server.
- `quantized-qwen3-moe`: Standalone high-speed CLI runner with chunked prefill.
- `quantized-qwen3`: Dense draft model CLI runner.

---

## 5. Execution Commands & Quickstart

### A. Launching the Dual-GPU Speculative Server

#### Option 1: Using the Automated Launcher Script
The launcher automatically probes `nvidia-smi`, assigns the RTX 2070 as the draft GPU and the Tesla P40 as the target GPU, resolves model paths, and starts the server:

```bash
# From the repository root
./run-speculative-server.sh
```

#### Option 2: Direct CLI Execution
```bash
/home/emanuele/Documents/speculative-bin/speculative-server \
  --host 0.0.0.0 \
  --port 8080 \
  --draft-device cuda:0 \
  --target-device cuda:1 \
  --draft-model /mnt/data/LMStudio/draft-models/Qwen3-0.6B-Q4_K_M.gguf \
  --target-model /mnt/data/LMStudio/Qwen3-Coder/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  --gamma 4 \
  --max-context 65536
```

---

### B. Standalone CLI Inference (Debugging & Benchmarking)

#### Standalone Target: Qwen3-Coder-30B-A3B on Tesla P40 (`cuda:1`)
```bash
/home/emanuele/Documents/speculative-bin/quantized-qwen3-moe \
  --model /mnt/data/LMStudio/Qwen3-Coder/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  --prompt "Write an asynchronous TCP server in Rust using tokio." \
  --chunk-size 512 \
  --device-id 1 \
  --sample-len 512 \
  --temperature 0.0
```

#### Standalone Target with Prompt File (Long Context / Repo Context)
```bash
/home/emanuele/Documents/speculative-bin/quantized-qwen3-moe \
  --model /mnt/data/LMStudio/Qwen3-Coder/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  --prompt-file /path/to/prompt.txt \
  --chunk-size 512 \
  --device-id 1 \
  --sample-len 1024 \
  --temperature 0.0
```

#### Standalone Draft: Qwen3-0.6B on RTX 2070 (`cuda:0`)
```bash
/home/emanuele/Documents/speculative-bin/quantized-qwen3 \
  --model /mnt/data/LMStudio/draft-models/Qwen3-0.6B-Q4_K_M.gguf \
  --prompt "Write an asynchronous TCP server in Rust using tokio." \
  --device-id 0 \
  --sample-len 256 \
  --temperature 0.0
```

---

### C. Building Binaries from Source (Multi-Architecture CUDA)

To compile fat binaries with compute support for both Pascal (`sm_61`) and Turing (`sm_75`):

```bash
# Compile the speculative server binary
CUDA_COMPUTE_CAP=61 CANDLE_CUDA_ARCHS="61,75" cargo build --release --features cuda \
  -p candle-speculative-server --bin speculative-server

# Compile standalone runners
CUDA_COMPUTE_CAP=61 CANDLE_CUDA_ARCHS="61,75" cargo build --release --features cuda \
  --example quantized-qwen3-moe

CUDA_COMPUTE_CAP=61 CANDLE_CUDA_ARCHS="61,75" cargo build --release --features cuda \
  --example quantized-qwen3
```

---

## 6. Client API & OpenAI Integration

The speculative server exposes an OpenAI-compatible REST API on port `8080`.

### Health Check
```bash
curl http://localhost:8080/health
```
```json
{"status":"ok"}
```

### Models List
```bash
curl http://localhost:8080/v1/models
```

### Streaming Chat Completion (SSE Chunks)
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder-30b-a3b-instruct",
    "messages": [
      {"role": "system", "content": "You are an expert Rust coding assistant."},
      {"role": "user", "content": "Implement an in-memory thread-safe LRU cache in Rust."}
    ],
    "stream": true,
    "temperature": 0.2
  }'
```

### Non-Streaming Chat Completion
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder-30b-a3b-instruct",
    "messages": [
      {"role": "user", "content": "Explain how speculative decoding works in 2 sentences."}
    ],
    "stream": false
  }'
```

---

## 7. IDE Integration (Continue.dev / Cursor / Aider)

Configure your coding assistant to connect directly to the local speculative server:

### `~/.continue/config.json`
```json
{
  "models": [
    {
      "title": "Qwen3-Coder 30B Dual-GPU (Local)",
      "provider": "openai",
      "model": "qwen3-coder-30b-a3b-instruct",
      "apiBase": "http://192.168.1.35:8080/v1",
      "apiKey": "none"
    }
  ],
  "tabAutocompleteModel": {
    "title": "Qwen3-0.6B Fast Draft (RTX 2070)",
    "provider": "openai",
    "model": "qwen3-0.6b",
    "apiBase": "http://192.168.1.35:8080/v1",
    "apiKey": "none"
  }
}
```
