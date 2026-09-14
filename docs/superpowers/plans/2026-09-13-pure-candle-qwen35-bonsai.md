# Pure Candle Qwen3.5 (Bonsai-27B) Architecture & Real Weight Inference Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement full `qwen35` hybrid architecture (Gated Delta Net SSM recurrent layers + GQA Attention layers) in pure Candle Rust, load real `Bonsai-27B-Q1_0.gguf` weights on RTX 2070 (8GB), pair with `Qwen3.8-27B-Q4_K_M.gguf` on Tesla P40 (24GB), and benchmark real empirical speculative decode without mocks.

**Architecture:** 
- Hybrid 64-layer decoder: 48 linear attention layers (Gated Delta Net recurrent SSM) interleaved with 16 standard GQA attention layers (every 4th layer: `(i + 1) % 4 == 0`).
- SSM state: 1D depthwise conv state (`[3, 10240]`) + Gated Delta Net recurrent state (`[48 heads, 128, 128]`) per SSM layer (~156 MB total resident VRAM).
- Attention state: FP16 zero-reallocation `InPlaceKvCache` on the 16 attention layers.
- Zero external runtime dependencies: 100% pure Candle Rust using existing compiled `Q1_0` CUDA dequantization kernels.

**Tech Stack:** Rust 1.75+, Candle (`candle-core`, `candle-nn`, `candle-transformers`), CUDA Compute 61 & 75 (`CANDLE_CUDA_ARCHS="61,75"`).

**Spec:** Dual-GPU speculative pipeline: RTX 2070 (8GB) draft, Tesla P40 (24GB) target, verified on physical hardware at `192.168.1.35`.

## Global Constraints
- Pure Candle in Rust: zero runtime C++/Python dependencies.
- Exact GGUF tensor compatibility with `Bonsai-27B-Q1_0.gguf` and `Qwen3.8-27B-Q4_K_M.gguf`.
- All tests must pass: `cargo test -p candle-speculative-server`.
- Authentic empirical benchmarks on physical hardware: no synthetic `--mock` fallback.

---

### Task 1: Qwen3.5 Config & Recurrent State Manager

**Files:**
- Create: `candle-speculative-server/src/qwen35_state.rs`
- Modify: `candle-speculative-server/src/lib.rs`
- Test: `candle-speculative-server/tests/qwen35_state_test.rs`

**Interfaces:**
- Produces: `Qwen35Config`, `Qwen35RecurrentState`, `Qwen35StateSnapshot`
- Methods: `new()`, `reset()`, `snapshot()`, `restore(&snapshot)`

- [ ] **Step 1: Write the failing test**
Create `candle-speculative-server/tests/qwen35_state_test.rs` testing state allocation, snapshotting, and rollback.

- [ ] **Step 2: Run test to verify it fails**
Run: `cargo test -p candle-speculative-server --test qwen35_state_test`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement Qwen35Config and Qwen35RecurrentState**
Implement `candle-speculative-server/src/qwen35_state.rs`:
- `Qwen35Config`: `hidden_size: 5120`, `intermediate_size: 17408`, `num_hidden_layers: 64`, `full_attn_interval: 4`, `ssm_conv_kernel: 4`, `ssm_d_state: 128`, `ssm_n_group: 16`, `ssm_dt_rank: 48`, `ssm_inner_size: 6144`, `num_attention_heads: 24`, `num_key_value_heads: 4`, `head_dim: 256`, `vocab_size: 248320`.
- `Qwen35RecurrentState`: Conv buffer (`[48, 3, 10240]`) and SSM state buffer (`[48, 48, 128, 128]`).
- Snapshot and restore methods for speculative rollback.

- [ ] **Step 4: Run test to verify it passes**
Run: `cargo test -p candle-speculative-server --test qwen35_state_test`
Expected: PASS

- [ ] **Step 5: Commit**
`git add candle-speculative-server/ && git commit -m "feat(qwen35): add config and recurrent state snapshot manager"`

---

### Task 2: Linear Attention / Gated Delta Net SSM Layer

**Files:**
- Create: `candle-speculative-server/src/qwen35_ssm.rs`
- Modify: `candle-speculative-server/src/lib.rs`
- Test: `candle-speculative-server/tests/qwen35_ssm_test.rs`

**Interfaces:**
- Consumes: `Qwen35Config`, `Qwen35RecurrentState`
- Produces: `Qwen35SsmLayer` with `forward_decode(&mut self, xs: &Tensor, state: &mut Qwen35LayerState) -> Result<Tensor>`

- [ ] **Step 1: Write the failing test**
Create `candle-speculative-server/tests/qwen35_ssm_test.rs` testing depthwise conv1d, L2 norm, and delta net recurrence.

- [ ] **Step 2: Run test to verify it fails**
Run: `cargo test -p candle-speculative-server --test qwen35_ssm_test`
Expected: FAIL

- [ ] **Step 3: Implement Qwen35SsmLayer**
Implement `candle-speculative-server/src/qwen35_ssm.rs`:
- QKV projection (`attn_qkv`), Gate projection (`attn_gate`), Beta (`ssm_beta`), Alpha (`ssm_alpha`), dt bias (`ssm_dt`), decay (`ssm_a`).
- Conv1d update with state buffer.
- L2 norm on Q and K, head repeat (16 -> 48).
- Recurrent Gated Delta Net math: $s \leftarrow s \cdot g$, $sk = s \cdot k$, $d = (v - sk) \cdot \beta$, $s \leftarrow s + k \otimes d$, $o = s \cdot q$.
- Gated RMSNorm: `norm(o, ssm_norm) * silu(z)`.
- Output projection: `ssm_out`.

- [ ] **Step 4: Run test to verify it passes**
Run: `cargo test -p candle-speculative-server --test qwen35_ssm_test`
Expected: PASS

- [ ] **Step 5: Commit**
`git add candle-speculative-server/ && git commit -m "feat(qwen35): implement Gated Delta Net linear attention layer"`

---

### Task 3: Qwen3.5 Attention Layer with Fused Q-Gate

**Files:**
- Create: `candle-speculative-server/src/qwen35_attn.rs`
- Modify: `candle-speculative-server/src/lib.rs`
- Test: `candle-speculative-server/tests/qwen35_attn_test.rs`

**Interfaces:**
- Consumes: `InPlaceKvCache`, `Qwen35Config`
- Produces: `Qwen35AttnLayer` with `forward(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, pos: usize) -> Result<Tensor>`

- [ ] **Step 1: Write the failing test**
Create `candle-speculative-server/tests/qwen35_attn_test.rs` testing fused Q-Gate projection (`12288 -> [6144, 6144]`), Q/K norm, RoPE, and GQA attention.

- [ ] **Step 2: Run test to verify it fails**
Run: `cargo test -p candle-speculative-server --test qwen35_attn_test`
Expected: FAIL

- [ ] **Step 3: Implement Qwen35AttnLayer**
Implement `candle-speculative-server/src/qwen35_attn.rs`:
- `attn_q` (12288): split into Q (6144) and Gate (6144).
- `attn_k` (1024), `attn_v` (1024), `attn_output` (5120).
- Q RMSNorm (`attn_q_norm`) and K RMSNorm (`attn_k_norm`).
- Rotary embedding (`rope_theta = 10,000,000.0`, head_dim = 256).
- GQA with `InPlaceKvCache` (24 query heads, 4 KV heads).
- Sigmoid gate activation and elementwise multiply.

- [ ] **Step 4: Run test to verify it passes**
Run: `cargo test -p candle-speculative-server --test qwen35_attn_test`
Expected: PASS

- [ ] **Step 5: Commit**
`git add candle-speculative-server/ && git commit -m "feat(qwen35): implement full GQA attention layer with fused Q-Gate"`

---

### Task 4: Qwen3.5 Model Loader & Bonsai27B Integration

**Files:**
- Create: `candle-speculative-server/src/qwen35_model.rs`
- Modify: `candle-speculative-server/src/model.rs`
- Modify: `candle-speculative-server/src/engine.rs`
- Test: `candle-speculative-server/tests/qwen35_model_test.rs`

**Interfaces:**
- Consumes: `Qwen35SsmLayer`, `Qwen35AttnLayer`, `Qwen35RecurrentState`, `QMatMul`
- Produces: `Qwen35Model`, `Bonsai27BWithKv` updated to load either `qwen35` hybrid or `qwen2` standard models seamlessly from GGUF.

- [ ] **Step 1: Write the failing test**
Create `candle-speculative-server/tests/qwen35_model_test.rs` verifying 64-layer construction and forward step.

- [ ] **Step 2: Run test to verify it fails**
Run: `cargo test -p candle-speculative-server --test qwen35_model_test`
Expected: FAIL

- [ ] **Step 3: Implement Qwen35Model and GGUF loader**
Implement `candle-speculative-server/src/qwen35_model.rs`:
- Read `token_embd.weight`, `output_norm.weight`, `output.weight`.
- For `layer_idx` in 0..64:
  - If `(layer_idx + 1) % 4 == 0`: load `Qwen35AttnLayer`.
  - Else: load `Qwen35SsmLayer`.
  - SwiGLU FFN (`ffn_gate`, `ffn_up`, `ffn_down`, `ffn_norm`).
- Integrate into `Bonsai27BWithKv` and `SuperDraftSpeculativeEngine`.

- [ ] **Step 4: Run test to verify it passes**
Run: `cargo test -p candle-speculative-server --test qwen35_model_test`
Expected: PASS

- [ ] **Step 5: Commit**
`git add candle-speculative-server/ && git commit -m "feat(qwen35): assemble full Qwen35Model with GGUF weight loading"`

---

### Task 5: Physical Dual-GPU Real-Weight Benchmark & Verification

**Files:**
- Modify: `candle-speculative-server/examples/benchmark_superdraft.rs`
- Modify: `candle-speculative-server/README.md`
- Script: `run-speculative-server.sh`

**Interfaces:**
- Executes real weight inference on remote hardware (`192.168.1.35`):
  - Draft: `/mnt/data/LMStudio/lmstudio-community/Bonsai-27B-GGUF/Bonsai-27B-Q1_0.gguf` on `cuda:0` (RTX 2070 8GB).
  - Target: `/mnt/data/LMStudio/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf` on `cuda:1` (Tesla P40 24GB).
- Produces: Real tokens/sec, acceptance rate $\alpha$, latency breakdown, and resident memory.

- [x] **Step 1: Sync code and build on remote server**
Build release binary with `CANDLE_CUDA_ARCHS="61,75"`.

- [x] **Step 2: Run single-token decode sanity test on RTX 2070**
Verify `Bonsai-27B-Q1_0.gguf` loads into VRAM and produces valid logits without error.

- [x] **Step 3: Run dual-GPU speculative benchmark with real weights**
Run `benchmark_superdraft` with real models across `cuda:0` and `cuda:1`.

- [x] **Step 4: Record authentic empirical measurements in README.md**
Update `candle-speculative-server/README.md` with true hardware numbers.

- [x] **Step 5: Commit and push**
`git add . && git commit -m "docs(benchmark): update README with authentic dual-GPU Bonsai-27B measurements"`
