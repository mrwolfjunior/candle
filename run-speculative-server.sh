#!/usr/bin/env bash
set -euo pipefail

echo "=== Asymmetric Speculative Inference Server Launcher ==="

PORT=${PORT:-8080}
HOST=${HOST:-"0.0.0.0"}
GAMMA=${GAMMA:-4}
MAX_CONTEXT=${MAX_CONTEXT:-65536}

TARGET_MODEL=${TARGET_MODEL:-"qwen2.5-14b-instruct-q4_k_m.gguf"}
DRAFT_MODEL=${DRAFT_MODEL:-"qwen2.5-1.5b-instruct-q8_0.gguf"}

# Detect GPUs via nvidia-smi if available
DRAFT_DEV="cuda:0"
TARGET_DEV="cuda:1"

if command -v nvidia-smi &> /dev/null; then
    echo "Querying NVIDIA GPUs..."
    GPU_INFO=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits)
    echo "$GPU_INFO"

    # Identify which GPU has >= 20GB VRAM (Tesla P40) and which has <= 10GB (RTX 2070)
    while IFS=',' read -r idx name mem; do
        idx=$(echo "$idx" | xargs)
        [ -z "$idx" ] && continue
        mem=$(echo "$mem" | xargs)
        if [ -n "$mem" ] && [ "$mem" -ge 20000 ]; then
            TARGET_DEV="cuda:$idx"
            echo "-> Detected Target GPU (Large VRAM >= 20GB): cuda:$idx ($name, ${mem}MB)"
        elif [ -n "$mem" ] && [ "$mem" -le 10000 ]; then
            DRAFT_DEV="cuda:$idx"
            echo "-> Detected Draft GPU (Small VRAM <= 10GB): cuda:$idx ($name, ${mem}MB)"
        fi
    done <<< "$GPU_INFO"
else
    echo "nvidia-smi not found. Using defaults: Draft=$DRAFT_DEV, Target=$TARGET_DEV"
fi

echo "Selected configuration:"
echo "  Draft Device:  $DRAFT_DEV"
echo "  Target Device: $TARGET_DEV"
echo "  Lookahead (gamma): $GAMMA"
echo "  Context:       $MAX_CONTEXT tokens"

if [ -d "$HOME/.cargo/bin" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Build with multi-arch flags for Pascal (sm_61) and Turing (sm_75)
if [ "${BUILD_RELEASE:-0}" = "1" ] || [ ! -f "./target/release/speculative-server" ]; then
    echo "Compiling binary with multi-architecture CUDA support (sm_61 + sm_75)..."
    CUDA_COMPUTE_CAP=61 CANDLE_CUDA_ARCHS="61,75" cargo build --release --features cuda -p candle-speculative-server --bin speculative-server
fi

BIN="./target/release/speculative-server"
if [ ! -f "$BIN" ] && [ -f "./target/debug/speculative-server" ]; then
    BIN="./target/debug/speculative-server"
fi

echo "Starting server on http://$HOST:$PORT ..."
exec "$BIN" \
    --host "$HOST" \
    --port "$PORT" \
    --draft-device "$DRAFT_DEV" \
    --target-device "$TARGET_DEV" \
    --gamma "$GAMMA" \
    --max-context "$MAX_CONTEXT" \
    ${TARGET_MODEL:+--target-model "$TARGET_MODEL"} \
    ${DRAFT_MODEL:+--draft-model "$DRAFT_MODEL"} \
    "$@"
