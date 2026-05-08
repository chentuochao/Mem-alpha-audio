#!/usr/bin/env bash
# Full pipeline: memory construction -> QA evaluation -> metric evaluation
# Usage:
#   bash run_pipeline.sh <dataset> <parquet_path>
#
# Positional arguments:
#   1. dataset        Dataset name (e.g. seamlessinteraction, memalpha, LOCOMO, ...)
#   2. parquet_path   Path to parquet file

set -euo pipefail

# ---------------------------------------------------------------------------- #
# Parse positional arguments
# ---------------------------------------------------------------------------- #
if [[ $# -lt 2 ]]; then
    echo "Usage: bash run_pipeline.sh <dataset> <parquet_path>"
    exit 1
fi

DATASET="$1"
PARQUET_PATH="$2"

# ---------------------------------------------------------------------------- #
# Build shared args
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Step 1: Memory construction
# ---------------------------------------------------------------------------- #
# echo "================================================================"
# echo "[STEP 1] Memory construction"
# echo "================================================================"

# MEM_ARGS=("${COMMON_ARGS[@]}")

# python run_memory_construction.py \
#   --agent_config config/memalpha-qwen3-4b_agent_0.05-0.1.yaml \
#   --dataset $DATASET \
#   --parquet_path $PARQUET_PATH \
#   --batch_size 1 \

# ---------------------------------------------------------------------------- #
# Step 2: QA evaluation
# ---------------------------------------------------------------------------- #
echo "================================================================"
echo "[STEP 2] QA evaluation"
echo "================================================================"

QA_ARGS=("${COMMON_ARGS[@]}")

python run_qa_evaluation.py --agent_config config/memalpha-qwen3-4b_agent_0.05-0.1.yaml \
    --force_reanswer_questions \
    --dataset $DATASET \
    --parquet_path $PARQUET_PATH \
    --batch_size 1  \

# ---------------------------------------------------------------------------- #
# Step 3: Metric evaluation
# ---------------------------------------------------------------------------- #
# echo "================================================================"
# echo "[STEP 3] Metric evaluation"
# echo "================================================================"

# QWEN_URL="http://localhost:8002/v1" python evaluate_agent_results.py --base_dir ./agents

# echo "================================================================"
# echo "[DONE] Pipeline completed successfully."
# echo "================================================================"
