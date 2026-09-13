# Super-Draft Bonsai 27B (1-Bit) Dual-GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement native `Q1_0` (1-bit / BitNet) quantization in Candle and deploy Bonsai-27B on NVIDIA RTX 2070 (8 GB) as a Super-Draft speculative engine paired with a 27B–30B target verifier on NVIDIA Tesla P40 (24 GB), supporting up to 64k tokens context.

**Architecture:** Extend Candle's core quantization subsystem (`candle-core` & `candle-kernels`) with GGML Type 41 (`Q1_0`). Load Bonsai-27B (3.6 GB) onto GPU 0 with an 8k rolling KV cache, and the target verifier onto GPU 1 with a 64k in-place KV cache. Verify draft tokens in parallel using greedy speculative decoding and measure end-to-end throughput and acceptance rate.

**Tech Stack:** Rust, Candle (`candle-core`, `candle-nn`, `candle-transformers`, `candle-kernels`), CUDA C++ (`sm_75` Turing, `sm_61` Pascal), Axum, Tokio.

**Spec:** `docs/superpowers/specs/2026-09-13-superdraft-bonsai-27b-design.md`

## Global Constraints

- Must be pure Candle in Rust (zero Python or llama.cpp runtime dependencies).
- `Q1_0` must conform to GGML Type 41 specifications: `QK1_0 = 128`, 18 bytes per block (16 bytes bit-packed weights + 2 bytes FP16 delta).
- GPU 0 (RTX 2070 8GB) VRAM must not exceed 6.5 GB total allocation.
- GPU 1 (Tesla P40 24GB) VRAM must support up to 64k tokens context using `InPlaceKvCache` and chunked prefill (`--chunk-size 512`).
- All code must build cleanly with `CANDLE_CUDA_ARCHS="61,75"`.

---

### Task 1: Data Type & Structs for `Q1_0` in `candle-core`

**Files:**
- Modify: `candle-core/src/quantized/mod.rs:280-345`
- Modify: `candle-core/src/quantized/ggml_file.rs:170-225`
- Modify: `candle-core/src/quantized/k_quants.rs`
- Test: `candle-core/tests/quantized_tests.rs`

**Interfaces:**
- Produces:
  ```rust
  pub const QK1_0: usize = 128;
  #[repr(C)]
  #[derive(Debug, Clone, Copy, PartialEq)]
  pub struct BlockQ1_0 {
      pub d: half::f16,
      pub qs: [u8; 16],
  }
  impl GgmlDType { pub const Q1_0: Self; }
  ```

- [ ] **Step 1: Write the failing unit test**

Create or update test in `candle-core/tests/quantized_tests.rs`:
```rust
#[test]
fn test_q1_0_block_size_and_dequant() {
    use candle_core::quantized::{GgmlDType, k_quants::BlockQ1_0};
    assert_eq!(std::mem::size_of::<BlockQ1_0>(), 18);
    let dtype = GgmlDType::from_u32(41).expect("GgmlDType 41 should be Q1_0");
    assert_eq!(dtype.block_size(), 128);
    assert_eq!(dtype.type_size(), 18);

    // Test dequantization of a known block: d = 2.0, qs all 0xAA (alternating bits)
    let block = BlockQ1_0 {
        d: half::f16::from_f32(2.0),
        qs: [0xAA; 16],
    };
    let mut out = vec![0f32; 128];
    block.dequantize(&mut out);
    // Bit 0 of 0xAA (10101010b) is 0 -> -2.0, Bit 1 is 1 -> +2.0
    assert_eq!(out[0], -2.0);
    assert_eq!(out[1], 2.0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p candle-core --test quantized_tests test_q1_0_block_size_and_dequant`
Expected: FAIL with unknown type or unhandled dtype 41.

- [ ] **Step 3: Implement `BlockQ1_0` and update `GgmlDType`**

In `candle-core/src/quantized/mod.rs`:
1. Add `Q1_0` to `enum GgmlDType` (mapped to `41` in `from_u32` and `to_u32`).
2. Add block_size (128) and type_size (18) to `GgmlDType`.
3. In `candle-core/src/quantized/k_quants.rs`:
   Define `BlockQ1_0` with `dequantize(&self, out: &mut [f32])`.
4. In `candle-core/src/quantized/ggml_file.rs`:
   Add `GgmlDType::Q1_0 => from_raw_data::<k_quants::BlockQ1_0>(raw_data, size_in_bytes, dims, device)` to `qtensor_from_ggml`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p candle-core --test quantized_tests test_q1_0_block_size_and_dequant`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add candle-core/
git commit -m "feat(quantized): add Q1_0 (type 41) data layout and CPU dequantization"
```

---

### Task 2: CUDA Dequantization & GEMV Kernel for `Q1_0` in `candle-kernels`

**Files:**
- Modify: `candle-kernels/src/quantized.cu`
- Modify: `candle-core/src/cuda_backend.rs`
- Modify: `candle-core/src/quantized/cuda.rs`
- Test: `candle-core/tests/quantized_tests.rs`

**Interfaces:**
- Produces:
  ```cuda
  extern "C" __global__ void dequantize_row_q1_0_cuda(const void* vx, half* vy, int64_t k);
  ```

- [ ] **Step 1: Write CUDA kernel in `candle-kernels/src/quantized.cu`**

Implement vectorized `dequantize_row_q1_0_cuda`:
- Each thread processes elements in parallel.
- Reads `BlockQ1_0`, extracts delta `d`, uses bit shifts and bitwise AND to decode each bit into $\pm d$ in FP16/FP32.

- [ ] **Step 2: Bind kernel in `candle-core/src/quantized/cuda.rs`**

Add dispatcher for `GgmlDType::Q1_0` in CUDA dequantize and QMatMul paths.

- [ ] **Step 3: Write test verifying CUDA dequantization against CPU**

In `candle-core/tests/quantized_tests.rs`:
Compare output tensor dequantized on CPU vs CUDA device.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p candle-core --features cuda --test quantized_tests test_q1_0_cuda`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add candle-kernels/ candle-core/
git commit -m "feat(cuda): add Q1_0 CUDA dequantization and matmul support"
```

---

### Task 3: Bonsai-27B GGUF Model Loading (`qwen35`)

**Files:**
- Modify: `candle-speculative-server/src/model.rs`
- Modify: `candle-speculative-server/src/lib.rs`
- Test: `candle-speculative-server/tests/bonsai_loader_test.rs`

**Interfaces:**
- Consumes: `candle-core::quantized::GgmlDType::Q1_0`
- Produces:
  ```rust
  pub struct Bonsai27BWithKv {
      pub model: QuantizedQwen2WithKv,
      pub rolling_window: usize,
  }
  ```

- [ ] **Step 1: Write failing test for Bonsai loader**

In `candle-speculative-server/tests/bonsai_loader_test.rs`:
Verify parser reads `Bonsai-27B-Q1_0.gguf` config (hidden_size 5120, num_layers 64, num_heads 40, num_kv_heads 8, vocab_size 248320) and instantiates model on device.

- [ ] **Step 2: Run test to verify failure**

Run: `cargo test -p candle-speculative-server --test bonsai_loader_test`
Expected: FAIL.

- [ ] **Step 3: Implement `qwen35` configuration and rolling window adapter**

In `candle-speculative-server/src/model.rs`:
Add preset `Config::bonsai_27b()` and support 8,192-token rolling KV cache window.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p candle-speculative-server --test bonsai_loader_test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add candle-speculative-server/
git commit -m "feat(model): add Bonsai-27B model loader and rolling window support"
```

---

### Task 4: Dual-GPU Super-Draft Engine Integration

**Files:**
- Modify: `candle-speculative-server/src/engine.rs`
- Modify: `candle-speculative-server/src/main.rs`
- Test: `candle-speculative-server/tests/superdraft_engine_test.rs`

**Interfaces:**
- Produces:
  ```rust
  pub struct SuperDraftSpeculativeEngine {
      pub draft_bonsai: Bonsai27BWithKv, // on cuda:0
      pub target_verifier: QuantizedQwen2WithKv, // on cuda:1
      pub gamma: usize,
  }
  ```

- [ ] **Step 1: Write integration test for dual-GPU speculative loop**

In `candle-speculative-server/tests/superdraft_engine_test.rs`:
Test proposal of $\gamma = 4$ tokens by Bonsai on GPU 0, verification by Target on GPU 1, and $O(1)$ KV rollback on simulated discrepancy.

- [ ] **Step 2: Run test to verify failure**

Run: `cargo test -p candle-speculative-server --test superdraft_engine_test`
Expected: FAIL.

- [ ] **Step 3: Implement `SuperDraftSpeculativeEngine` in `src/engine.rs`**

Wire the speculative loop:
1. Autoregressive proposal on GPU 0 for $\gamma$ steps.
2. Parallel forward pass on GPU 1.
3. Verification via `verify_greedy`.
4. In-place rollback on both caches when $k < \gamma$.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p candle-speculative-server --test superdraft_engine_test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add candle-speculative-server/
git commit -m "feat(engine): integrate Super-Draft Bonsai 27B dual-GPU speculative loop"
```

---

### Task 5: 64k Context Verification & Empirical Benchmarks on Hardware

**Files:**
- Create: `candle-speculative-server/examples/benchmark_superdraft.rs`
- Modify: `candle-speculative-server/README.md`
- Modify: `run-speculative-server.sh`

**Interfaces:**
- Produces:
  - Benchmark binary measuring tokens/sec, acceptance rate ($\alpha$), and VRAM usage at 512, 1k, 4k, 8k, 16k, and 64k context.
  - Complete documentation and reproduction commands.

- [ ] **Step 1: Write benchmark runner `benchmark_superdraft.rs`**

Create `candle-speculative-server/examples/benchmark_superdraft.rs`:
Ingests prompt files of varying context sizes (up to 64k tokens), triggers speculative generation for 256 tokens, records prompt prefill speed, decode throughput, speculative acceptance rate, and peak VRAM on both GPUs.

- [ ] **Step 2: Compile with multi-architecture CUDA flags**

```bash
CUDA_COMPUTE_CAP=61 CANDLE_CUDA_ARCHS="61,75" cargo build --release --features cuda \
  -p candle-speculative-server --example benchmark_superdraft
```

- [ ] **Step 3: Deploy to remote server and run benchmark suite**

Execute across context windows:
- 512 tokens
- 4,000 tokens
- 16,000 tokens
- 64,000 tokens

- [ ] **Step 4: Update documentation with empirical results**

Record benchmark table in `candle-speculative-server/README.md` and commit.

- [ ] **Step 5: Commit**

```bash
git add candle-speculative-server/ README.md
git commit -m "docs: add empirical benchmarks for Super-Draft Bonsai 27B across 64k context"
```
