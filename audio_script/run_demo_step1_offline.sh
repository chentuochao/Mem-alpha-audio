#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
#  Configuration — edit these to match your setup
# ──────────────────────────────────────────────────────────────────────

# Conda environments
ENV1="nemo"   # NeMo environment (diarization + ASR)
# Model paths
DIAR_MODEL_PATH="/checkpoint/seamless/tuochao/Models/huggingface/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1.nemo"
ASR_MODEL_PATH="/checkpoint/seamless/tuochao/Models/huggingface/multitalker-parakeet-streaming-0.6b-v1/multitalker-parakeet-streaming-0.6b-v1.nemo"

# Options
MAX_NUM_OF_SPKS=4

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
#  Step 1a (OFFLINE): Diarization + ASR inference for Bazinga! dataset
# ──────────────────────────────────────────────────────────────────────
DATA_PATH="/checkpoint/seamless/tuochao/data/bazinga/data/TheBigBangTheory/"
OUTPUT_DIR="/storage/home/tuochao/mem_projects/Outputs/TheBigBangTheory/step1_offline/"

echo "============================================================"
echo "  Step 1a (OFFLINE): Diarization + ASR  (conda env: ${ENV1})"
echo "============================================================"

conda activate "${ENV1}"
export PYTHONPATH="/storage/home/tuochao/Mem-alpha-audio"
# Serialize CUDA ops so graph-capture races don't cause illegal memory access
export CUDA_LAUNCH_BLOCKING=1

python -m audio_script.Multi_ASR.step1_diarize_asr_bazinga_offline \
    --data_dir        "${DATA_PATH}" \
    --diar_model_path "${DIAR_MODEL_PATH}" \
    --asr_model_path  "${ASR_MODEL_PATH}" \
    --max_num_of_spks "${MAX_NUM_OF_SPKS}" \
    --output_dir      "${OUTPUT_DIR}"
