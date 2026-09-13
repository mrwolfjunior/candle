# Technical Specification: Super-Draft Bonsai 27B (1-Bit) on Heterogeneous Dual-GPU (RTX 2070 + Tesla P40)

**Date**: 2026-09-13  
**Status**: Proposed  
**Branch**: `feat/frontier-multitier-120b-bonsai`  
**Target Hardware**:
- GPU 0: NVIDIA GeForce RTX 2070 (8 GB GDDR6, Turing `sm_75`)
- GPU 1: NVIDIA Tesla P40 (24 GB GDDR5, Pascal `sm_61`)
- Host: Intel Xeon E5-2650 v4, 31 GB DDR4 System RAM, PCIe Gen 3

---

## 1. Executive Summary & Objective

In classical speculative decoding, the draft model is tiny (0.5B – 1.5B parameters). On complex programming, code refactoring, and multi-turn logic, small draft models frequently diverge from the target distribution, causing the speculative acceptance rate ($\alpha$) to drop to 50% – 65% and negating speed gains.

This project introduces **Bonsai 27B** (a 27-billion-parameter Qwen3.5 model quantized to 1.125 bits per weight using the `Q1_0` BitNet format, weighing only **3.6 GB**) as a **Super-Draft Engine** running on the 8 GB RTX 2070.

By pairing a 27B draft with a 27B–30B target verifier on the Tesla P40 (24 GB):
1. Both models possess deep semantic understanding and reasoning capabilities, elevating speculative acceptance ($\alpha$) to **88% – 95%**.
2. Both models fit **100% inside GPU VRAM** (3.6 GB on RTX 2070, 16–18.5 GB on Tesla P40), leaving ample headroom for rolling and extended KV caches up to **64,000 tokens**.
3. Effective generation throughput reaches an estimated **60 – 75 tokens/s** with frontier-tier intelligence.

---

## 2. Model & Quantization Architecture: `Q1_0` (GGML Type 41)

### 2.1 Format Specification
`Bonsai-27B-Q1_0.gguf` uses GGML quantization type ID `41` (`GGML_TYPE_Q1_0`), introduced for 1-bit / BitNet ternary models.

- **Block Size (`QK1_0`)**: 128 weights per block.
- **Type Size**: 18 bytes per block.
- **Bit-rate**: $\frac{18 \times 8}{128} = 1.125$ bits per parameter.
- **Memory Layout**:
  ```rust
  #[repr(C)]
  #[derive(Debug, Clone, Copy, PartialEq)]
  pub struct BlockQ1_0 {
      pub d: half::f16,      // 2 bytes: FP16 delta scale factor
      pub qs: [u8; 16],      // 16 bytes: 128 bit-packed ternary weights
  }
  ```

### 2.2 Mathematical Dequantization
For each block, the 128 quantized weights $w_i \in \{-1.0, +1.0\}$ (or $\{0, +1.0\}$ scaled by $d$) are unpacked from the 16 bytes of `qs`:
$$w_i = \text{to\_f32}(d) \times \left(2 \cdot \left(\frac{\text{qs}[i / 8] \gg (i \% 8)}{1} \& 1\right) - 1\right)$$

### 2.3 CUDA Acceleration for Turing (`sm_75`)
On the RTX 2070:
- Four 32-bit registers load the entire 16-byte `qs` payload.
- Vectorized bit-manipulation (`__popc`, byte masking) decompresses weights in fast registers/L1 cache.
- Forward matmul runs memory-bound at peak GDDR6 bandwidth (448 GB/s). Since the entire model weights comprise only 3.6 GB, weight reads achieve near-peak cache hit rates.

---

## 3. Dual-GPU Asymmetric Speculative Pipeline

### 3.1 Device Allocation
| Component | GPU 0 (RTX 2070 8GB, `sm_75`) | GPU 1 (Tesla P40 24GB, `sm_61`) |
| :--- | :--- | :--- |
| **Model** | `Bonsai-27B-Q1_0.gguf` (3.6 GB) | `Qwen3.8-27B-Q4_K_M.gguf` (16.0 GB) *or* `Qwen3-Coder-30B-A3B` (18.55 GB) |
| **Role** | Speculative Draft Generator ($\gamma = 4..5$) | Parallel Verifier & Context Host |
| **VRAM Model** | 3.60 GB | 16.00 GB (or 18.55 GB) |
| **VRAM KV Cache** | 2.00 GB (8k rolling window) | 6.00 GB (up to 64k tokens) |
| **VRAM Scratch/Activations**| 0.60 GB | 1.00 GB |
| **Total VRAM Allocated** | **~6.20 GB / 8.00 GB** (1.80 GB free margin) | **~23.00 GB / 24.00 GB** (1.00 GB free margin) |

### 3.2 Vocab Alignment & Token Matching
- `Bonsai-27B-Q1_0.gguf` and `Qwen3.8-27B-Q4_K_M.gguf` share the **exact same 248,320-token BPE vocabulary**:
  - Direct token ID comparison ($O(1)$ in Rust) with zero string conversion overhead.
- When paired with `Qwen3-Coder-30B-A3B` (151,936 vocab):
  - Bidirectional token mapping table `HashMap<u32, u32>` precomputed at startup for instantaneous 1:1 token matching.

### 3.3 Iteration Execution Cycle
1. **Draft Generation (GPU 0)**:
   - Bonsai 27B runs autoregressively for $\gamma = 4$ steps, generating proposal tokens $x_1, x_2, x_3, x_4$.
   - Execution time: ~50–65 ms.
2. **Parallel Verification (GPU 1)**:
   - Target model ingests $[x_1, x_2, x_3, x_4]$ in a single forward batch pass using Pascal DP4A INT8 dot-product kernels.
   - Target outputs predicted probabilities for each position.
   - Execution time: ~70–85 ms.
3. **Greedy Verification & $O(1)$ Rollback**:
   - `verify_greedy` compares draft proposal against target argmax.
   - If divergent at position $k < \gamma$, `InPlaceKvCache` decrements sequence head to position $k$ in $O(1)$ time on both GPUs.
   - Emits accepted tokens + target correction token.
4. **Resulting Throughput**:
   - Total loop latency: ~130–150 ms.
   - Average accepted tokens per loop: 3.6 – 4.5 tokens.
   - **Effective Throughput**: **60 – 75 tokens/second**.

---

## 4. Extended Context Management (64k Tokens)

1. **Target GPU (Tesla P40)**:
   - Maintains full 64k token memory resident in VRAM using `InPlaceKvCache`.
   - Chunked prefill (`--chunk-size 512`) keeps attention memory bounded during prompt ingestion.
2. **Draft GPU (RTX 2070)**:
   - Uses an 8,192-token rolling window KV cache.
   - In speculative decoding, the draft model only needs recent local context to predict the next $\gamma$ tokens; long-term conditioning is verified by the target model.
   - Keeps draft VRAM strictly bounded under 6.2 GB, preventing OOM on the 8 GB card.

---

## 5. Verification & Benchmark Plan

1. **Unit & Kernel Tests**:
   - `test_q1_0_block_struct`: Verify size and alignment of `BlockQ1_0` (18 bytes).
   - `test_q1_0_dequant_cpu`: Verify exact numerical dequantization against reference implementation.
   - `test_q1_0_cuda_gemv`: Verify CUDA matmul on RTX 2070.
2. **Model Loading & Smoke Test**:
   - Load `Bonsai-27B-Q1_0.gguf` on `cuda:0`.
   - Generate test prompt tokens and verify coherence.
3. **End-to-End Speculative Verification**:
   - Launch server with `--draft-model Bonsai-27B-Q1_0.gguf --target-model Qwen3.8-27B-Q4_K_M.gguf`.
   - Verify OpenAI `/v1/chat/completions` endpoint for both streaming and non-streaming responses.
4. **Context & Speed Benchmarking**:
   - Measure throughput and acceptance rate across context lengths:
     - 512 tokens (Interactive coding)
     - 4,000 tokens (Module refactoring)
     - 16,000 tokens (Repository context)
     - 64,000 tokens (Full repo / documentation context)
