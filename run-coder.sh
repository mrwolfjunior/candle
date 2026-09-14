#!/usr/bin/env bash
set -euo pipefail

MODEL="/mnt/data/LMStudio/Qwen3-Coder/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
BIN=""

if [ -f "./target/release/examples/quantized-qwen3-moe" ]; then
    BIN="./target/release/examples/quantized-qwen3-moe"
elif [ -f "/home/emanuele/Documents/speculative-bin/quantized-qwen3-moe" ]; then
    BIN="/home/emanuele/Documents/speculative-bin/quantized-qwen3-moe"
elif [ -f "/home/emanuele/Documents/candle/target/release/examples/quantized-qwen3-moe" ]; then
    BIN="/home/emanuele/Documents/candle/target/release/examples/quantized-qwen3-moe"
else
    echo "Error: quantized-qwen3-moe binary not found." >&2
    exit 1
fi

GPU_ID=${CUDA_VISIBLE_DEVICES:-1}
SAMPLE_LEN=${SAMPLE_LEN:-1024}
CHUNK_SIZE=${CHUNK_SIZE:-512}
TEMP=${TEMPERATURE:-0.2}

if [ $# -eq 0 ]; then
    echo "=== Qwen3-Coder-30B-A3B Runner (Tesla P40) ==="
    echo "Uso rapido: $0 \"Il tuo prompt di codice\""
    echo "File prompt: $0 --prompt-file /percorso/prompt.txt"
    echo ""
    echo "Opzioni opzionali d'ambiente:"
    echo "  SAMPLE_LEN=2048 $0 \"...\"    (lunghezza max output)"
    echo "  TEMPERATURE=0.0 $0 \"...\"   (0 per greedy deterministico)"
    exit 1
fi

if [[ "$1" == --* ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$BIN" \
        --model "$MODEL" \
        --chunk-size "$CHUNK_SIZE" \
        -n "$SAMPLE_LEN" \
        --temperature "$TEMP" \
        "$@"
else
    PROMPT="$1"
    shift
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$BIN" \
        --model "$MODEL" \
        --prompt "$PROMPT" \
        --chunk-size "$CHUNK_SIZE" \
        -n "$SAMPLE_LEN" \
        --temperature "$TEMP" \
        "$@"
fi
