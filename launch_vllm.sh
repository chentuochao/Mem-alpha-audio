CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B \
    --host 0.0.0.0 \
    --port 8002 \
    --tensor-parallel-size 1
