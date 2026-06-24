#!/usr/bin/env bash
# Run the cascade error tracer for one constructed instance.
#
# Usage:
#   bash diagnostic/run_trace_errors.sh [INSTANCE_DIR]
#
# INSTANCE_DIR defaults to the example below; pass a different one to override.
# All other knobs (qa_file, dialog_root, ...) fall back to the script defaults
# inside trace_errors_clean.py.
set -euo pipefail

# --- matcher backends ------------------------------------------------------ #
# LLM judge ON: needs OPENROUTER_API_KEY + a Qwen endpoint. For a local vLLM
# server (bash launch_vllm.sh) the key value is ignored but must be non-empty.
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-EMPTY}"
export QWEN_URL="${QWEN_URL:-http://localhost:8002/v1}"
export QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-Qwen/Qwen3-32B}"

# Embedding matcher OFF: it enables only when OPENAI_API_KEY is set.
unset OPENAI_API_KEY

# --- target instance ------------------------------------------------------- #
INSTANCE_DIR="${1:-./agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_Season1_vibevoice_extracted_name_no_thinking_tokens_2048/0}"

PYTHONPATH=. python diagnostic/trace_errors_clean.py \
    --instance_dir "$INSTANCE_DIR"
