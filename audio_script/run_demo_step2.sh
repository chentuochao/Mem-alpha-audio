#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
#  Configuration — edit these to match your setup
# ──────────────────────────────────────────────────────────────────────

# Conda environments
ENV2="mem"   # WeSpeaker / Resemblyzer environment (speaker embeddings)


EMBEDDING_MODEL_DIR="/checkpoint/seamless/tuochao/Models/huggingface/wespeaker-voxceleb-resnet293-LM"

# Options
MAX_NUM_OF_SPKS=4
SIMILARITY_THRESHOLD=0.65
EMBEDDING_DEVICE="cuda:0"
DATA_DIR="/storage/home/tuochao/Mem-alpha-audio/Audio_Results/vibevoice/TheBigBangTheory/step1"
OUTPUT_DIR="/storage/home/tuochao/Mem-alpha-audio/Audio_Results/vibevoice/TheBigBangTheory/step2"

# Incremental (season-by-season) cross-run state. Set POOL_PATH to a single
# .npz file shared across runs to preserve the global speaker pool; leave empty
# for a one-shot run. SEASON_FILTER limits which samples are processed this run
# (space-separated substrings, e.g. "Season02"); empty = all.
POOL_PATH="${POOL_PATH:-}"
SEASON_FILTER=(${SEASON_FILTER:-})

# Working directory (where the python scripts live)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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


# ──────────────────────────────────────────────────────────────────────
#  Step 2: Speaker embedding + global matching  (env2)
# ──────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Step 2: Speaker matching  (conda env: ${ENV2})"
echo "============================================================"

conda activate "${ENV2}"
export PYTHONPATH="/storage/home/tuochao/Mem-alpha-audio"

STEP2_ARGS=(
    --data_dir            "${DATA_DIR}"
    --embedding_model_dir "${EMBEDDING_MODEL_DIR}"
    --similarity_threshold "${SIMILARITY_THRESHOLD}"
    --embedding_device    "${EMBEDDING_DEVICE}"
    --output_dir          "${OUTPUT_DIR}"
)
if [ -n "${POOL_PATH}" ]; then
    STEP2_ARGS+=(--pool_path "${POOL_PATH}")
fi
if [ "${#SEASON_FILTER[@]}" -gt 0 ]; then
    STEP2_ARGS+=(--season_filter "${SEASON_FILTER[@]}")
fi

python "${SCRIPT_DIR}/Speaker_Track/step2_speaker_match_v2.py" "${STEP2_ARGS[@]}"

conda deactivate
