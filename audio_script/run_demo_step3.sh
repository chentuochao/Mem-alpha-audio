#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
#  Configuration — edit these to match your setup
# ──────────────────────────────────────────────────────────────────────

# Conda environments
ENV1="mem"   # NeMo environment (diarization + ASR)
# Model paths

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

conda activate "${ENV1}"
export PYTHONPATH="/storage/home/tuochao/Mem-alpha-audio"
export QWEN_URL="http://localhost:8002/v1"

DATA_PATH="./Audio_Results/nemo-offline/TheBigBangTheory/step2/"

# Incremental (season-by-season) cross-run state: a single JSON file holding the
# speaker pool + evidence registry + processed chunks. Set STATE_PATH to reuse
# it across runs (only new chunks hit the LLM); leave empty for a one-shot run.
STATE_PATH="${DATA_PATH}/speakers_name_pool.json"   #"${STATE_PATH:-}"

STEP3_ARGS=(--data_dir "${DATA_PATH}")
if [ -n "${STATE_PATH}" ]; then
    STEP3_ARGS+=(--state_path "${STATE_PATH}")
    STEP3_ARGS+=(--update_pool)
fi

python audio_script/Speaker_Track/step3_speaker_name_extract.py "${STEP3_ARGS[@]}"
