#!/usr/bin/env bash
# Reward-model vLLM server (Qwen3-32B). GPU and port are overridable via the
# CUDA_VISIBLE_DEVICES / PORT env vars so launch_servers.sh can give each job
# its own port (a sweep co-locates jobs on one node); defaults match the
# original standalone behavior (GPU 1, port 8002).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PORT="${PORT:-8002}"

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B \
    --served-model-name qwen3-32b \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size 1
