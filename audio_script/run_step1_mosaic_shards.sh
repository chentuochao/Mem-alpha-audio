#!/usr/bin/env bash
#
# Data-parallel Step1 (diar+ASR) for Mix_Mosaic: split the conversation list into
# NUM_SHARDS round-robin shards and run one process per GPU, all writing into the
# SAME output_dir. Results are stored per conv_id, so shards never collide and
# the output tree is byte-for-byte the layout a single-process run produces —
# Step2/Step3 (run_demo_pipeline_mosaic.sh with RUN_STEP1=0) work unchanged.
#
# Usage:
#   RAW_DATA_PATH=/checkpoint/.../test_interf_SNR0 ./audio_script/run_step1_mosaic_shards.sh
#   NUM_SHARDS=4 GPUS="0 1 2 3" METHOD=vibevoice ./audio_script/run_step1_mosaic_shards.sh
#
set -euo pipefail

METHOD="${METHOD:-vibevoice}"
RAW_DATA_PATH="${RAW_DATA_PATH:-/checkpoint/seamless/tuochao/data/Mix_Mosaic/naturalistic/test}"

# One shard per GPU. GPUS is a space-separated list of CUDA device ids.
GPUS="${GPUS:-0 1 2 3}"
read -ra GPU_ARR <<< "${GPUS}"
NUM_SHARDS="${NUM_SHARDS:-${#GPU_ARR[@]}}"

PYTHONPATH_ROOT="/storage/home/tuochao/Mem-alpha-audio"
RESULTS_ROOT="${PYTHONPATH_ROOT}/Audio_Results"

# ── Step1 (NeMo backend) models ──────────────────────────────────────
ENV_NEMO="nemo"
ENV_VIBEVOICE="vibevoice"
DIAR_MODEL_PATH="/checkpoint/seamless/tuochao/Models/huggingface/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1.nemo"
ASR_MODEL_PATH="/checkpoint/seamless/tuochao/Models/huggingface/multitalker-parakeet-streaming-0.6b-v1/multitalker-parakeet-streaming-0.6b-v1.nemo"
MAX_NUM_OF_SPKS=4
# ── Step1 (VibeVoice backend) ────────────────────────────────────────
VV_MODEL_PATH="microsoft/VibeVoice-ASR"
# Greedy decoding on noisy audio can collapse into a repetition loop that never
# emits EOS; the chunk then runs to MAX_NEW_TOKENS and yields an empty/salvaged
# transcript (see backends/vibevoice.py). Healthy chunks here emit ~300-3200
# tokens, so 8192 is ample headroom and caps the cost of a runaway chunk.
# NO_REPEAT_NGRAM_SIZE>0 breaks the loop outright (0 = off).
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.2}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-0}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
NUM_BEAMS="${NUM_BEAMS:-1}"
ATTN_IMPL="${ATTN_IMPL:-auto}"

DATASET_TAG="$(basename "${RAW_DATA_PATH}")"
STEP1_OUT="${RESULTS_ROOT}/${METHOD}/${DATASET_TAG}/step1"
LOG_DIR="${RESULTS_ROOT}/logs"
mkdir -p "${STEP1_OUT}" "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

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

case "${METHOD}" in
    nemo-streaming|nemo-offline) STEP1_ENV="${ENV_NEMO}" ;;
    vibevoice)                   STEP1_ENV="${ENV_VIBEVOICE}" ;;
    *) echo "ERROR: unknown METHOD='${METHOD}'"; exit 1 ;;
esac
conda activate "${STEP1_ENV}"

export PYTHONPATH="${PYTHONPATH_ROOT}"
export PYTHONUNBUFFERED=1
[ "${METHOD}" = "nemo-offline" ] && export CUDA_LAUNCH_BLOCKING=1

echo "============================================================"
echo "  Mix_Mosaic Step1 (sharded)   method=${METHOD}  env=${STEP1_ENV}"
echo "  Data dir  : ${RAW_DATA_PATH}"
echo "  Step1 out : ${STEP1_OUT}"
echo "  Shards    : ${NUM_SHARDS} on GPU(s) ${GPUS}"
echo "============================================================"

PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    GPU="${GPU_ARR[$(( i % ${#GPU_ARR[@]} ))]}"
    LOG="${LOG_DIR}/step1_mosaic_${DATASET_TAG}_${METHOD}_shard${i}_${STAMP}.log"

    ARGS=(
        --method      "${METHOD}"
        --data_dir    "${RAW_DATA_PATH}"
        --output_dir  "${STEP1_OUT}"
        --device      "cuda"
        --num_shards  "${NUM_SHARDS}"
        --shard_index "${i}"
    )
    case "${METHOD}" in
        nemo-streaming|nemo-offline)
            ARGS+=(
                --diar_model_path "${DIAR_MODEL_PATH}"
                --asr_model_path  "${ASR_MODEL_PATH}"
                --max_num_of_spks "${MAX_NUM_OF_SPKS}"
            ) ;;
        vibevoice)
            ARGS+=(
                --model_path            "${VV_MODEL_PATH}"
                --max_new_tokens        "${MAX_NEW_TOKENS}"
                --temperature           "${TEMPERATURE}"
                --top_p                 "${TOP_P}"
                --num_beams             "${NUM_BEAMS}"
                --repetition_penalty    "${REPETITION_PENALTY}"
                --no_repeat_ngram_size  "${NO_REPEAT_NGRAM_SIZE}"
                --attn_implementation   "${ATTN_IMPL}"
            ) ;;
    esac

    echo ">>> shard ${i} -> GPU ${GPU}   log: ${LOG}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
        python -m audio_script.Multi_ASR.step1_mosaic "${ARGS[@]}" > "${LOG}" 2>&1 &
    PIDS+=($!)
done

# Wait for all shards; report every failure instead of dying on the first.
FAIL=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "shard ${i} OK"
    else
        echo "!!! shard ${i} FAILED (see ${LOG_DIR}/step1_mosaic_${DATASET_TAG}_${METHOD}_shard${i}_${STAMP}.log)"
        FAIL=1
    fi
done

echo ""
echo "Step1 shards done (${DATASET_TAG}, ${METHOD}). Output: ${STEP1_OUT}"
echo "Now run Step2/3:  RAW_DATA_PATH=${RAW_DATA_PATH} RUN_STEP1=0 ./audio_script/run_demo_pipeline_mosaic.sh"
exit "${FAIL}"
