# Design Document: Asymmetric Speculative Inference Server for Dual-GPU (Tesla P40 + RTX 2070) in Candle

- **Date**: 2026-09-12
- **Status**: Validated Design
- **Target Hardware**:
  - GPU 0 / Draft: NVIDIA GeForce RTX 2070 (8 GB VRAM, Turing microarchitecture, Compute Capability 7.5)
  - GPU 1 / Target: NVIDIA Tesla P40 (24 GB VRAM, Pascal microarchitecture, Compute Capability 6.1)
  - Host System: Intel Xeon CPU, 32 GB ECC DDR RAM, Linux x86_64, PCIe Gen3
- **Development Environment**: macOS (Darwin arm64, CPU / mock testing)

---

## 1. Executive Summary & Problem Diagnosis

### The Problem
Standard open-source inference servers (Ollama, LMStudio, default llama.cpp) produce severely degraded tokens-per-second (often < 5-8 tok/s) when deployed on heterogeneous dual-GPU setups combining a Pascal-era GPU (Tesla P40) and a Turing GPU (RTX 2070):
1. **Tesla P40 Compute Constraints (Pascal sm_61)**: Pascal lacks Tensor Cores. Its native FP16 execution rate is 1:64 of its FP32 rate. Standard CUDA kernels compiling against FP16 paths or dequantizing to half precision cause near-complete pipeline stalls. For high performance, Pascal requires **INT8 DP4A (`__dp4a`)** dot products or FP32 accumulation.
2. **Autoregressive Decoding Memory-Bandwidth Starvation**: At batch size 1, autoregressive token generation is memory-bandwidth bound. With ~346 GB/s theoretical bandwidth on the P40, scanning a ~9-10 GB model per token caps single-token generation at ~12-15 tok/s, irrespective of compute capability.
3. **PCIe Latency Penalty in Layer Pipelining**: Naive layer splitting across the PCIe bus transmits intermediate activations for every single generated token, multiplying transfer latency by the sequence length.

### The Solution: Asymmetric Speculative Decoding
- **RTX 2070 (Draft Model)**: Runs a compact model (`Qwen2.5-1.5B-Instruct` in GGUF Q8_0 or FP16) leveraging Turing Tensor Cores to emit $\gamma = 4..5$ speculative candidate tokens at high speed (> 90-120 tok/s).
- **Tesla P40 (Target Model)**: Runs the large model (`Qwen2.5-14B-Instruct` in GGUF Q4_K_M) with a native **64k FP16 In-Place KV Cache**. The P40 performs a **single parallel batch forward pass** on the $\gamma + 1$ candidate sequence. This increases the arithmetic intensity by $(\gamma + 1)\times$, transforming the memory-bandwidth bottleneck into a compute-dense verification step.
- **Inter-GPU PCIe Traffic**: Minimal. Only candidate token IDs (a few `u32` integers, ~20 bytes per speculative cycle) cross the host/PCIe interface.
- **Serving Interface**: An OpenAI-compatible HTTP server (`/v1/chat/completions`) built with Axum and Tokio supporting Server-Sent Events (SSE) streaming.

---

## 2. System Architecture & Topology

```
+-------------------------------------------------------------------------+
|                        Client (Open WebUI / curl)                       |
+------------------------------------+------------------------------------+
                                     | HTTP POST /v1/chat/completions (SSE)
                                     v
+-------------------------------------------------------------------------+
|                   Axum HTTP Server & Request Handler                    |
|  - Tokenizer & ChatML template application                              |
|  - SSE event generator & Cancellation detector                          |
+------------------------------------+------------------------------------+
                                     | Prompt Token IDs
                                     v
+-------------------------------------------------------------------------+
|                    Asymmetric Speculative Engine                        |
|                                                                         |
|  +-------------------------------+   +-------------------------------+  |
|  | Draft Engine (RTX 2070 sm_75) |   | Target Engine (P40 sm_61)     |  |
|  | - Qwen 2.5 1.5B (Q8_0 / FP16) |   | - Qwen 2.5 14B (Q4_K_M)       |  |
|  | - Turing Tensor Cores (MMQ)   |   | - Pascal DP4A INT8 & FP32     |  |
|  | - Autoregressive draft:       |   | - Batch verification forward  |  |
|  |   generates [d_1 ... d_gamma] |   |   pass on [t_last, d_1..d_g]  |  |
|  | - In-place KV cache (~65 MB)  |   | - In-place KV cache (~12.3GB) |  |
|  +---------------+---------------+   +---------------+---------------+  |
|                  |                                   |                  |
|                  +-------------[ PCIe ]--------------+                  |
|                       Transmit draft token IDs                          |
|                       Rollback / Synchronize KV                         |
+------------------------------------+------------------------------------+
                                     | Accepted Token Stream
                                     v
                          Client Streaming Output
```

### Device Placement & Defaults
- `--draft-device`: `cuda:0` (default) or specified device ID.
- `--target-device`: `cuda:1` (default) or specified device ID.
- Fallback flag `--device cpu`: Enables local execution on macOS or non-CUDA systems for development, automated tests, and mock benchmarking.

---

## 3. In-Place 64k Rolling KV Cache Design

### The Need
`candle-transformers` currently implements attention caching via dynamic concatenation:
```rust
let k = Tensor::cat(&[k_cache, &k], 2)?;
let v = Tensor::cat(&[v_cache, &v], 2)?;
```
At 64,000 tokens with 48 layers:
- Re-allocating and copying up to 12.3 GB of tensors per step leads to continuous `cudaMalloc` thrashing, VRAM fragmentation, and inevitable OOM errors.
- Truncating on rejection requires slicing allocations.

### `InPlaceKvCache` Specification
1. **Pre-Allocation**:
   - For each layer, pre-allocate contiguous tensors:
     - `k_cache`: Shape `(1, n_kv_heads, max_context_len, head_dim)`, dtype `F16`.
     - `v_cache`: Shape `(1, n_kv_heads, max_context_len, head_dim)`, dtype `F16`.
   - For Qwen 2.5 14B with 64k tokens ($n_{kv} = 8$, $d_{head} = 128$, $L = 48$):
     $$\text{Size} = 48 \times 2 \times 8 \times 65536 \times 128 \times 2 \text{ bytes} \approx 12.88 \text{ GB}$$
     Fits within the 24 GB VRAM alongside the ~9.0 GB Q4_K_M model weights and ~1.5 GB CUDA scratch workspace.
2. **In-Place Updates**:
   - Write incoming keys and values directly into the buffer slice at `[current_pos .. current_pos + seq_len]` using in-place slice copy / view updates.
   - Attention computes over `k_cache.narrow(2, 0, current_pos + seq_len)?` without copying memory.
3. **Instant $O(1)$ Rollback**:
   - When the Target engine rejects $M$ tokens, rollback is strictly an integer update:
     $$\text{current\_pos} = \text{accepted\_pos} + 1$$
   - Zero reallocation, zero tensor slicing, zero overhead.

---

## 4. Speculative Verification Protocol

Given prompt length $N$ and lookahead parameter $\gamma$ (default $\gamma = 4$):

### Step 1: Initialization / Prefill
1. Target model (P40) prefills prompt tokens $0..N-1$, populates Target KV cache to $N$, produces initial logits $L_{target}^{(0)}$.
2. Draft model (2070) prefills prompt tokens $0..N-1$, populates Draft KV cache to $N$.
3. Sample initial token $t_0$ from $L_{target}^{(0)}$ and emit to client.

### Step 2: Draft Generation Loop (RTX 2070)
1. For $i = 1..\gamma$:
   - Draft model runs autoregressive forward pass on token $d_{i-1}$ (where $d_0 = t_0$).
   - Samples candidate token $d_i$.
   - Draft KV cache advances to $N + \gamma$.

### Step 3: Batch Verification Forward Pass (Tesla P40)
1. Transmit token IDs $[t_0, d_1, d_2, \dots, d_\gamma]$ to Target engine.
2. Target model executes a **single forward pass** with sequence length $\gamma + 1$ at position index $N - 1$.
3. Target computes logit matrix $L_{target}$ of shape $(\gamma + 1, \text{vocab\_size})$.

### Step 4: Acceptance & Correction Logic
- **Greedy Verification** (temperature = 0):
  - For $j = 1..\gamma$:
    - Let $expected = \text{argmax}(L_{target}[j-1])$.
    - If $expected == d_j$:
      - Accept $d_j$.
    - Else:
      - Reject $d_j$ and all subsequent draft tokens.
      - Next accepted token is $expected$.
      - Break loop.
  - If all $\gamma$ tokens matched:
    - Sample an extra bonus token $t_{bonus} = \text{argmax}(L_{target}[\gamma])$.
- **Stochastic Verification** (temperature > 0):
  - Standard Leviathan speculative rejection sampling comparing $P_{target}(x)$ and $P_{draft}(x)$.

### Step 5: State Synchronization & KV Realignment
- Number of accepted tokens: $K_{acc} \in [1, \gamma + 1]$.
- Target KV pointer advances to $N + K_{acc}$.
- Draft KV pointer rolls back to $N + K_{acc}$.
- If a correction token was inserted, Draft engine updates its KV cache with the corrected token at the divergence position.
- Emit accepted tokens via SSE stream.

---

## 5. OpenAI-Compatible HTTP Server (Axum)

### Endpoints
- `GET /v1/models`
  - Returns JSON containing `id: "qwen2.5-14b-instruct-speculative"`.
- `POST /v1/chat/completions`
  - Request body:
    ```json
    {
      "model": "qwen2.5-14b-instruct",
      "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
      ],
      "temperature": 0.0,
      "max_tokens": 4096,
      "stream": true
    }
    ```
- `GET /health`
  - Returns engine status, VRAM usage summary, and cumulative acceptance rate metrics.

### SSE Stream Framing
- Events formatted as `text/event-stream`:
  ```
  data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}
  
  data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}
  
  data: [DONE]
  ```
- Client disconnection is monitored via `axum` stream drop; cancels the active inference task immediately to prevent GPU lockup.

---

## 6. Multi-Architecture CUDA Build Strategy (Target sm_61 + sm_75)

### Build Configuration
On the target machine, `nvcc` flags must compile for both microarchitectures:
```bash
CUDA_COMPUTE_CAP="61,75"
```
Flags passed to `nvcc`:
- `-gencode arch=compute_61,code=sm_61` (Enables DP4A INT8 SIMD intrinsics for Pascal P40).
- `-gencode arch=compute_75,code=sm_75` (Enables Turing Tensor Cores for RTX 2070).

### Deployment Script: `run-speculative-server.sh`
- Automates device detection:
  - Parses `nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader`
  - Detects Tesla P40 (24 GB) -> sets `--target-device`
  - Detects RTX 2070 (8 GB) -> sets `--draft-device`
- Downloads or verifies GGUF models:
  - `Qwen/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m.gguf`
  - `Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q8_0.gguf`
- Starts the server on `0.0.0.0:8080`.

---

## 7. Testing & Verification Plan

1. **Local Compilation & Type Checks (macOS)**:
   - `cargo check --features ""`: Compiles core and server in CPU fallback mode.
   - `cargo test --bin speculative-server`: Verifies:
     - `InPlaceKvCache` slice update and rollback behavior.
     - Greedy speculative acceptance logic (all accepted, first rejected, mid rejected).
     - OpenAI JSON request parsing and SSE serialization.
2. **Target CUDA Verification (Xeon Workstation)**:
   - Verify device creation: `Device::new_cuda(0)` and `Device::new_cuda(1)`.
   - Benchmark throughput against standard single-GPU autoregressive decoding.
   - Test 64k token context window to ensure steady-state VRAM residency without OOM.
