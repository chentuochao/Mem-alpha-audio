#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
#  Configuration — edit these to match your setup
# ──────────────────────────────────────────────────────────────────────

# Conda environments
ENV1="nemo"   # NeMo environment (diarization + ASR)
ENV2="mem"   # WeSpeaker / Resemblyzer environment (speaker embeddings)

# Audio files to process
AUDIO_FILES=(
   "./mix_conversation_dataset/V00_S0062_I00000125_P0092_V00_S0062_I00000125_P0093.wav"
   "./mix_conversation_dataset/V00_S0062_I00000126_P0092_V00_S0062_I00000126_P0093.wav"
   "./mix_conversation_dataset/V00_S0062_I00000128_P0092_V00_S0062_I00000128_P0093.wav"
   "./mix_conversation_dataset/V00_S0062_I00000129_P0092_V00_S0062_I00000129_P0093.wav"
   "./mix_conversation_dataset/V00_S0062_I00000130_P0092_V00_S0062_I00000130_P0093.wav"
)

# Model paths
DIAR_MODEL_PATH="/checkpoint/seamless/tuochao/Models/huggingface/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1.nemo"
ASR_MODEL_PATH="/checkpoint/seamless/tuochao/Models/huggingface/multitalker-parakeet-streaming-0.6b-v1/multitalker-parakeet-streaming-0.6b-v1.nemo"
EMBEDDING_MODEL_DIR="/checkpoint/seamless/tuochao/Models/huggingface//wespeaker-voxceleb-resnet293-LM"

# Options
MAX_NUM_OF_SPKS=4
SIMILARITY_THRESHOLD=0.65
EMBEDDING_DEVICE="cuda:0"
OUTPUT_DIR="./demo_output"

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
#  Step 1: Diarization + ASR  (env1)
# ──────────────────────────────────────────────────────────────────────

# echo "============================================================"
# echo "  Step 1: Diarization + ASR  (conda env: ${ENV1})"
# echo "============================================================"

# conda activate "${ENV1}"

# python "${SCRIPT_DIR}/step1_diarize_asr.py" \
#     --audio_files "${AUDIO_FILES[@]}" \
#     --diar_model_path "${DIAR_MODEL_PATH}" \
#     --asr_model_path  "${ASR_MODEL_PATH}" \
#     --max_num_of_spks "${MAX_NUM_OF_SPKS}" \
#     --output_dir      "${OUTPUT_DIR}"

# conda deactivate

# ──────────────────────────────────────────────────────────────────────
#  Step 2: Speaker embedding + global matching  (env2)
# ──────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Step 2: Speaker matching  (conda env: ${ENV2})"
echo "============================================================"

conda activate "${ENV2}"

python "${SCRIPT_DIR}/step2_speaker_match.py" \
    --manifest            "${OUTPUT_DIR}/step1_manifest.json" \
    --embedding_model_dir "${EMBEDDING_MODEL_DIR}" \
    --similarity_threshold "${SIMILARITY_THRESHOLD}" \
    --embedding_device    "${EMBEDDING_DEVICE}" \
    --output_dir          "${OUTPUT_DIR}"

conda deactivate

echo ""
echo "============================================================"
echo "  Done! Results in: ${OUTPUT_DIR}/global_speaker_results.json"
echo "============================================================"
