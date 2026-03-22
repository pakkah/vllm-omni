#!/bin/bash
# Lingbot-World online serving startup script

MODEL="${MODEL:-./lingbot-world-base-cam}"
PORT="${PORT:-8099}"

echo "Starting Lingbot-World server..."
echo "Model: $MODEL"
echo "Port: $PORT"

vllm serve "$MODEL" --omni \
    --port "$PORT"
