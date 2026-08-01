#!/usr/bin/env bash
# Launch Qwen3.6-27B as a TEXT-ONLY OpenAI-compatible vLLM server.
#
# Qwen3.6-27B is a Qwen3_5ForConditionalGeneration (vision-language) checkpoint,
# but memory construction / judging is text-only. The flags below mirror the
# verified-working LLM(**kwargs) from test_load_qwen.py, translated to the
# api_server CLI:
#   --limit-mm-per-prompt '{"image":0,"video":0}'  -> skip the multimodal warmup
#                                                     forward pass that HANGS startup
#   --gdn-prefill-backend triton                   -> skip the FlashInfer Gated-
#                                                     DeltaNet JIT that stalls prefill
#   --enforce-eager                                -> skip torch.compile + CUDA-graph
#                                                     capture (slow/hangs on this model)
#
# Reuse the same HF cache as the other scripts so weights are not re-downloaded.
export HF_HOME="${HF_HOME:-/checkpoint/seamless/tuochao/Models/huggingface/}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/checkpoint/seamless/tuochao/Models/huggingface/}"

# Pick the GPU(s) for the server. Keep this OFF the GPU running the memory agent
# to avoid OOM (e.g. agent on GPU0 -> server on GPU1).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
PORT="${PORT:-8002}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype bfloat16 \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --served-model-name qwen3-32b \
    --enforce-eager \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --gdn-prefill-backend triton
    # --max-model-len 32768   # uncomment to cap context if you hit KV-cache OOM
