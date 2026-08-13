#!/usr/bin/env bash
# Anonymized-speaker pipeline: memory construction (SpeakerX tags + registry in
# core) -> post-hoc name substitution + re-embed -> QA evaluation.
#
# Same positional args and ENV knobs as run_pipeline.sh, but --anon_speaker is
# ALWAYS enabled (step 1 uses the anon multispeaker prompt; both steps write/read
# the '_anonspk' output folder), and a substitution step runs in between.
#
# Usage:
#   bash run_pipeline_speaker_name.sh <parquet_path> [dataset] [custom_qa_dir]
#
# ENV knobs (all optional): COMPRESSION_STRATEGY, SEED, MEM_TEMPERATURE, ROLLOUT_LABEL
#   INCLUDE_CANDIDATES=0  -> substitute only [confirmed] names (default: include candidates)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash run_pipeline_speaker_name.sh <parquet_path> [dataset] [custom_qa_dir]"
    exit 1
fi

PARQUET_PATH="$1"
DATASET="${2:-seamlessinteraction_options}"
CUSTOM_QA_DIR="${3:-outputs/step3_anony/qas/}"

COMPRESSION_STRATEGY="${COMPRESSION_STRATEGY:-}"
SEED="${SEED:-}"
MEM_TEMPERATURE="${MEM_TEMPERATURE:-}"
ROLLOUT_LABEL="${ROLLOUT_LABEL:-}"
INCLUDE_CANDIDATES="${INCLUDE_CANDIDATES:-1}"

AGENT_CONFIG="config/qwen3.6-27B_agent.yaml"

COMPRESSION_ARGS=()
if [[ -n "$COMPRESSION_STRATEGY" ]]; then
    COMPRESSION_ARGS=(--compression_strategy "$COMPRESSION_STRATEGY")
fi
CONSTRUCTION_ARGS=()
if [[ -n "$SEED" ]]; then
    CONSTRUCTION_ARGS+=(--seed "$SEED")
fi
if [[ -n "$MEM_TEMPERATURE" ]]; then
    CONSTRUCTION_ARGS+=(--temperature "$MEM_TEMPERATURE")
fi
ROLLOUT_ARGS=()
if [[ -n "$ROLLOUT_LABEL" ]]; then
    ROLLOUT_ARGS=(--rollout_label "$ROLLOUT_LABEL")
fi

echo "parquet_path        : ${PARQUET_PATH}"
echo "dataset             : ${DATASET}"
echo "custom_qa_dir       : ${CUSTOM_QA_DIR}"
echo "compression_strategy: ${COMPRESSION_STRATEGY:-None}"
echo "rollout_label       : ${ROLLOUT_LABEL:-None}"
echo "anon_speaker        : ALWAYS ON"

# ---------------------------------------------------------------------------- #
# Locate conda
# ---------------------------------------------------------------------------- #
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
# Step 1: Memory construction (anon)
# ---------------------------------------------------------------------------- #
echo "================================================================"
echo "[STEP 1] Memory construction (--anon_speaker)"
echo "================================================================"
conda activate vllm
python run_memory_construction_new.py \
  --agent_config "$AGENT_CONFIG" \
  --dataset "$DATASET" \
  --parquet_path "$PARQUET_PATH" \
  --batch_size 1 \
  --anon_speaker \
  "${COMPRESSION_ARGS[@]}" \
  "${CONSTRUCTION_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}"

# ---------------------------------------------------------------------------- #
# Resolve the run's output folder (must match get_out_dir with anon_speaker=True)
# ---------------------------------------------------------------------------- #
conda activate mem
OUT_DIR=$(python - "$AGENT_CONFIG" "$PARQUET_PATH" "$DATASET" "${COMPRESSION_STRATEGY:-default}" "${ROLLOUT_LABEL:-}" <<'PY'
import sys, types, yaml
from conversation_creator import get_out_dir
agent_config = yaml.safe_load(open(sys.argv[1]))
comp = sys.argv[4] or "default"
rollout = sys.argv[5] or None
args = types.SimpleNamespace(
    parquet_path=sys.argv[2], dataset=sys.argv[3],
    exclude_memory=set(), rollout_label=rollout,
    compression_strategy=comp, anon_speaker=True,
)
print(get_out_dir(agent_config, args, 0))
PY
)
echo "run folder: $OUT_DIR"

# ---------------------------------------------------------------------------- #
# Step 1.5: Substitute inferred speaker names + re-embed
# ---------------------------------------------------------------------------- #
echo "================================================================"
echo "[STEP 1.5] Speaker-name substitution + re-embed"
echo "================================================================"
CAND_ARG=()
if [[ "$INCLUDE_CANDIDATES" == "0" ]]; then
    CAND_ARG=(--confirmed_only)
fi
python postprocess_speaker_substitute.py --agent_dir "$OUT_DIR" "${CAND_ARG[@]}"

# ---------------------------------------------------------------------------- #
# Step 2: QA evaluation (anon folder)
# ---------------------------------------------------------------------------- #
echo "================================================================"
echo "[STEP 2] QA evaluation"
echo "================================================================"
python run_qa_evaluation.py --agent_config "$AGENT_CONFIG" \
    --dataset "$DATASET" \
    --parquet_path "$PARQUET_PATH" \
    --batch_size 1 \
    --custom_qa_dir "$CUSTOM_QA_DIR" \
    --force_reanswer_questions \
    --anon_speaker \
    "${COMPRESSION_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}"

echo "================================================================"
echo "[DONE] Anon-speaker pipeline completed."
echo "================================================================"
