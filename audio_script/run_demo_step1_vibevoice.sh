#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
#  Configuration — edit these to match your setup
# ──────────────────────────────────────────────────────────────────────

# Conda environment
ENV1="vibevoice"   # VibeVoice environment (diarization + ASR)

DATA_PATH="/checkpoint/seamless/tuochao/data/Mix_Mosaic/naturalistic/test/"

# Model path (single combined VibeVoice checkpoint)
MODEL_PATH="/checkpoint/seamless/tuochao/Models/huggingface/vibevoice"

# Options
DEVICE="cuda"
MAX_NEW_TOKENS=32768
TEMPERATURE=0.0
TOP_P=1.0
NUM_BEAMS=1
ATTN_IMPL="auto"   # auto | flash_attention_2 | sdpa | eager

OUTPUT_DIR="/storage/home/tuochao/mem_projects/demo_output_vibevoice/"

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
#  Step 1a: Diarization + ASR inference  (VibeVoice, needs GPU)
# ──────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "  Step 1a: Diarization + ASR inference  (conda env: ${ENV1})"
echo "============================================================"

conda activate "${ENV1}"
export PYTHONPATH="/storage/home/tuochao/Mem-alpha-audio"
python "${SCRIPT_DIR}/step1_vibevoice.py" \
    --data_dir         "${DATA_PATH}" \
    --model_path       "${MODEL_PATH}" \
    --device           "${DEVICE}" \
    --max_new_tokens   "${MAX_NEW_TOKENS}" \
    --temperature      "${TEMPERATURE}" \
    --top_p            "${TOP_P}" \
    --num_beams        "${NUM_BEAMS}" \
    --attn_implementation "${ATTN_IMPL}" \
    --output_dir       "${OUTPUT_DIR}"

# ──────────────────────────────────────────────────────────────────────
#  Step 1b: Evaluate DER + cpWER  (same env, no GPU needed)
# ──────────────────────────────────────────────────────────────────────

# echo ""
# echo "============================================================"
# echo "  Step 1b: Evaluate DER + cpWER"
# echo "============================================================"

# python "${SCRIPT_DIR}/step1_eval.py" \
#     --data_dir "${OUTPUT_DIR}"

# conda deactivate
