#!/usr/bin/env bash
# Full pipeline: memory construction -> QA evaluation -> metric evaluation
# Usage:
#   bash run_pipeline.sh <parquet_path> [dataset] [custom_qa_dir]
#
# Positional arguments:
#   1. parquet_path   Path to parquet file (required)
#   2. dataset        Dataset name (default: seamlessinteraction_options;
#                     use 'perltqa' for the PerLTQA audio pipeline)
#   3. custom_qa_dir  Directory of custom QA JSON/JSONL files
#                     (default: outputs/step3_anony/qas/)
#
# Environment variables:
#   COMPRESSION_STRATEGY  Memory compression strategy for step 1 + step 2.
#                         Default: None -> the flag is omitted and the Python
#                         default ('default') applies. Valid keys come from
#                         config/prompts_wrt_datasource_compression.yaml
#                         (default, x1.5, x2, x3, x5).
#
# Example (PerLTQA):
#   bash run_pipeline.sh outputs/perltqa_step3/bundle_0/dataset_pred_name.parquet \
#        perltqa outputs/perltqa_step3/bundle_0/qas/
#
# Example (with compression):
#   COMPRESSION_STRATEGY=x3 bash run_pipeline.sh <parquet> seamlessinteraction_options outputs/step3_anony/qas/

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
# Compression is an ENV var, not a positional arg. empty/None -> omit the flag
# (Python default: 'default'). e.g. COMPRESSION_STRATEGY=x3 bash run_pipeline.sh ...
COMPRESSION_STRATEGY="${COMPRESSION_STRATEGY:-}"

# Extra ENV knobs for seed-variance runs (all optional; empty -> flag omitted).
#   SEED             random seed for step 1 memory construction
#   MEM_TEMPERATURE  sampling temperature for step 1 (must be > 0 for seeds to
#                    actually diverge; the default config decodes greedily)
#   ROLLOUT_LABEL    output subdirectory (.../<name>/<label>/<idx>) so per-seed
#                    runs don't overwrite; applied to BOTH step 1 and step 2 so QA
#                    reads the matching memory folder.
SEED="${SEED:-}"
MEM_TEMPERATURE="${MEM_TEMPERATURE:-}"
ROLLOUT_LABEL="${ROLLOUT_LABEL:-}"

# Whether step 2 (QA) re-answers even when results.json already exists. Default
# OFF -> existing results are reused (QA skipped). Set FORCE_REANSWER=1/true to
# overwrite. (Step 1 memory construction is always skipped when agent_state.json
# already exists, regardless of this flag.)
FORCE_REANSWER="${FORCE_REANSWER:-}"

# Only pass --compression_strategy when a value is given, so the default run
# behaves exactly as before (no _comp_ postfix on the output folder).
COMPRESSION_ARGS=()
if [[ -n "$COMPRESSION_STRATEGY" ]]; then
    COMPRESSION_ARGS=(--compression_strategy "$COMPRESSION_STRATEGY")
fi

# step 1 (construction) knobs: seed + temperature.
CONSTRUCTION_ARGS=()
if [[ -n "$SEED" ]]; then
    CONSTRUCTION_ARGS+=(--seed "$SEED")
fi
if [[ -n "$MEM_TEMPERATURE" ]]; then
    CONSTRUCTION_ARGS+=(--temperature "$MEM_TEMPERATURE")
fi

# rollout label: shared by step 1 + step 2 so folder names match.
ROLLOUT_ARGS=()
if [[ -n "$ROLLOUT_LABEL" ]]; then
    ROLLOUT_ARGS=(--rollout_label "$ROLLOUT_LABEL")
fi

# step 2 (QA) force-reanswer: only pass the flag when explicitly enabled.
REANSWER_ARGS=()
case "$FORCE_REANSWER" in
    1|true|True|TRUE|yes|YES) REANSWER_ARGS=(--force_reanswer_questions) ;;
esac

echo "parquet_path        : ${PARQUET_PATH}"
echo "dataset             : ${DATASET}"
echo "custom_qa_dir       : ${CUSTOM_QA_DIR}"
echo "compression_strategy: ${COMPRESSION_STRATEGY:-None}"
echo "seed                : ${SEED:-None}"
echo "temperature         : ${MEM_TEMPERATURE:-None (config default)}"
echo "rollout_label       : ${ROLLOUT_LABEL:-None}"
echo "force_reanswer      : ${FORCE_REANSWER:-None (reuse existing results)}"

# ---------------------------------------------------------------------------- #
# Build shared args
# ---------------------------------------------------------------------------- #
# ──────────────────────────────────────────────────────────────────────
#  Locate conda
# ──────────────────────────────────────────────────────────────────────
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -n "${CONDA_EXE:-}" ]; then
    source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
else
    echo "ERROR: Cannot locate conda. Set CONDA_EXE or adjust the script."
    exit 1
fi

# ---------------------------------------------------------------------------- #
# Step 1: Memory construction
# ---------------------------------------------------------------------------- #
echo "================================================================"
echo "[STEP 1] Memory construction"
echo "================================================================"

# MEM_ARGS=("${COMMON_ARGS[@]}")
conda activate vllm
python run_memory_construction_new.py \
  --agent_config config/qwen3.6-27B_agent.yaml \
  --dataset "$DATASET" \
  --parquet_path "$PARQUET_PATH" \
  --batch_size 1 \
  "${COMPRESSION_ARGS[@]}" \
  "${CONSTRUCTION_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}"



### using Qwem3.6
# python run_memory_construction_new.py --agent_config config/qwen3.6-27B_agent.yaml --dataset seamlessinteraction_options --parquet_path ./outputs/bazinga_data/Season1_vibevoice_gt_name.parquet

# ---------------------------------------------------------------------------- #
# Step 2: QA evaluation
# ---------------------------------------------------------------------------- #
echo "================================================================"
echo "[STEP 2] QA evaluation"
echo "================================================================"

conda activate mem
python run_qa_evaluation.py --agent_config config/qwen3.6-27B_agent.yaml \
    --dataset "$DATASET" \
    --parquet_path "$PARQUET_PATH" \
    --batch_size 1  \
    --custom_qa_dir "$CUSTOM_QA_DIR" \
    "${REANSWER_ARGS[@]}" \
    "${COMPRESSION_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}"

# python run_qa_evaluation.py --agent_config config/qwen3.6-27B_agent.yaml --dataset seamlessinteraction_options --parquet_path  ./outputs/bazinga_data/Season1_vibevoice_gt_name.parquet --custom_qa_dir outputs/tmp_folder_for_95_qs/merged_95.jsonl

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
