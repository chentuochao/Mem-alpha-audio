#!/usr/bin/env bash
#
# Step 1 (Bazinga) — unified driver for audio_script/Multi_ASR/step1_bazinga.py
#
# Picks one of three inference backends via the METHOD variable:
#   - nemo-streaming   NeMo streaming SortFormer + cache-aware ASR   (env: nemo)
#   - nemo-offline     NeMo offline  SortFormer + offline ASR        (env: nemo)
#   - vibevoice        VibeVoice end-to-end multi-talker ASR         (env: vibevoice)
#
# Override the METHOD without editing the file:
#   METHOD=nemo-offline ./audio_script/run_demo_step1_bazinga.sh
#

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
#  Configuration — edit these to match your setup
# ──────────────────────────────────────────────────────────────────────

# Which backend to run (override via env: METHOD=vibevoice ./run...)
METHOD="${METHOD:-nemo-streaming}"

# I/O
DATA_PATH="/checkpoint/seamless/tuochao/data/bazinga/data/TheBigBangTheory/"
OUTPUT_BASE="/storage/home/tuochao/mem_projects/Outputs/TheBigBangTheory"
PYTHONPATH_ROOT="/storage/home/tuochao/Mem-alpha-audio"
DEVICE="cuda"

# Optional episode filters (leave empty to disable)
INCLUDE_SUBSTR=""   # e.g. "Season01"
EXCLUDE_SUBSTR=""   # e.g. "Season02"

# Chunking (shared by all backends)
CHUNK_MIN_DUR=60.0
CHUNK_MAX_DUR=300.0
CHUNK_GAP_THRESHOLD=3.0

# ── NeMo backend args (used by nemo-streaming and nemo-offline) ──────
ENV_NEMO="nemo"
# 4-speaker model
# DIAR_MODEL_PATH="/checkpoint/seamless/tuochao/Models/huggingface/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1.nemo"
# MAX_NUM_OF_SPKS=4
# 8-speaker model
DIAR_MODEL_PATH="/checkpoint/seamless/tuochao/Models/huggingface/ultra_diar_streaming_sortformer_8spk_v1/ultra_diar_streaming_sortformer_8spk_v1.nemo"
MAX_NUM_OF_SPKS=8
# 5-speaker model
# DIAR_MODEL_PATH="/checkpoint/seamless/tuochao/Models/huggingface/ultra_diar_streaming_sortformer_5spk_v1/ultra_diar_streaming_sortformer_5spk_v1.nemo"
# MAX_NUM_OF_SPKS=5

ASR_MODEL_PATH="/checkpoint/seamless/tuochao/Models/huggingface/multitalker-parakeet-streaming-0.6b-v1/multitalker-parakeet-streaming-0.6b-v1.nemo"

# ── VibeVoice backend args ───────────────────────────────────────────
ENV_VIBEVOICE="vibevoice"
VV_MODEL_PATH="microsoft/VibeVoice-ASR"
MAX_NEW_TOKENS=32768
TEMPERATURE=0.0
TOP_P=1.0
NUM_BEAMS=1
ATTN_IMPL="auto"   # auto | flash_attention_2 | sdpa | eager

# ──────────────────────────────────────────────────────────────────────
#  Pick conda env + per-method output directory
# ──────────────────────────────────────────────────────────────────────

case "${METHOD}" in
    nemo-streaming)
        CONDA_ENV="${ENV_NEMO}"
        OUTPUT_DIR="${OUTPUT_BASE}/step1_nemo_streaming/"
        ;;
    nemo-offline)
        CONDA_ENV="${ENV_NEMO}"
        OUTPUT_DIR="${OUTPUT_BASE}/step1_nemo_offline/"
        ;;
    vibevoice)
        CONDA_ENV="${ENV_VIBEVOICE}"
        OUTPUT_DIR="${OUTPUT_BASE}/step1_vibevoice/"
        ;;
    *)
        echo "ERROR: unknown METHOD='${METHOD}'"
        echo "       choose one of: nemo-streaming | nemo-offline | vibevoice"
        exit 1
        ;;
esac

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
#  Run
# ──────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "  Step 1 (Bazinga): method=${METHOD}   conda env=${CONDA_ENV}"
echo "  Data       : ${DATA_PATH}"
echo "  Output     : ${OUTPUT_DIR}"
echo "============================================================"

conda activate "${CONDA_ENV}"
export PYTHONPATH="${PYTHONPATH_ROOT}"

# Offline NeMo needs serialized CUDA ops to avoid graph-capture races.
if [ "${METHOD}" = "nemo-offline" ]; then
    export CUDA_LAUNCH_BLOCKING=1
fi

# Common args shared by every backend.
COMMON_ARGS=(
    --method               "${METHOD}"
    --data_dir             "${DATA_PATH}"
    --output_dir           "${OUTPUT_DIR}"
    --device               "${DEVICE}"
    --chunk_min_dur        "${CHUNK_MIN_DUR}"
    --chunk_max_dur        "${CHUNK_MAX_DUR}"
    --chunk_gap_threshold  "${CHUNK_GAP_THRESHOLD}"
)
if [ -n "${INCLUDE_SUBSTR}" ]; then
    COMMON_ARGS+=(--include_substr "${INCLUDE_SUBSTR}")
fi
if [ -n "${EXCLUDE_SUBSTR}" ]; then
    COMMON_ARGS+=(--exclude_substr "${EXCLUDE_SUBSTR}")
fi

# Backend-specific args.
case "${METHOD}" in
    nemo-streaming|nemo-offline)
        BACKEND_ARGS=(
            --diar_model_path  "${DIAR_MODEL_PATH}"
            --asr_model_path   "${ASR_MODEL_PATH}"
            --max_num_of_spks  "${MAX_NUM_OF_SPKS}"
        )
        ;;
    vibevoice)
        BACKEND_ARGS=(
            --model_path           "${VV_MODEL_PATH}"
            --max_new_tokens       "${MAX_NEW_TOKENS}"
            --temperature          "${TEMPERATURE}"
            --top_p                "${TOP_P}"
            --num_beams            "${NUM_BEAMS}"
            --attn_implementation  "${ATTN_IMPL}"
        )
        ;;
esac

python -m audio_script.Multi_ASR.step1_bazinga \
    "${COMMON_ARGS[@]}" \
    "${BACKEND_ARGS[@]}"
