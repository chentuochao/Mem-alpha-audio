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
if [[ $# -lt 1 ]]; then
    echo "Usage: bash run_pipeline.sh <parquet_path> [dataset] [custom_qa_dir]"
    exit 1
fi

PARQUET_PATH="$1"
DATASET="${2:-seamlessinteraction_options}"
CUSTOM_QA_DIR="${3:-outputs/step3_anony/qas/}"

echo "parquet_path : ${PARQUET_PATH}"
echo "dataset      : ${DATASET}"
echo "custom_qa_dir: ${CUSTOM_QA_DIR}"



for STRAT in x2 x3 x5; do
    python run_memory_construction_new.py \
        --agent_config config/qwen3.6-27B_agent.yaml \
        --dataset seamlessinteraction_options \
        --parquet_path ./outputs/bazinga_data/Season1_vibevoice_gt_name.parquet \
        --compression_strategy $STRAT
done
