# Candle Asymmetric Speculative Inference Server

A high-performance OpenAI-compatible inference server written in Rust using Candle, designed for heterogeneous multi-GPU setups.

## Hardware Setup
- **Draft GPU (RTX 2070 8GB, sm_75)**: Runs Qwen 2.5 1.5B (GGUF Q8_0 / FP16) using Turing Tensor Cores to rapidly propose $\gamma = 4..5$ tokens.
- **Target GPU (Tesla P40 24GB, sm_61)**: Runs Qwen 2.5 14B (GGUF Q4_K_M) using Pascal DP4A INT8 dot products to verify all candidate tokens in a single parallel batch pass.
- **Context**: 64,000 tokens ($65,536$) resident in VRAM using zero-reallocation `InPlaceKvCache`.

## Quick Start (Target Machine)

```bash
# Launch server (auto-detects P40 and RTX 2070)
./run-speculative-server.sh
```

## API Usage

```bash
# OpenAI Chat Completions endpoint
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-14b-instruct",
    "messages": [{"role": "user", "content": "Explain speculative decoding in 3 sentences."}],
    "stream": true
  }'
```
