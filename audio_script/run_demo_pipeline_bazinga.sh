#!/usr/bin/env bash
#
# End-to-end Bazinga audio pipeline driver: Step1 (diar+ASR) -> Step2 (speaker
# matching) -> Step3 (speaker-name extraction), in one script.
#
# Incremental season-by-season state for Step2 + Step3:
#   - Step1 runs ONCE over every season in SEASONS (no cross-season state).
#   - Step2 and Step3 then run season by season. The FIRST season is run with
#     --update_pool so it builds the initial speaker pool (Step2 pool.npz) and
#     name state (Step3 json). Every LATER season runs WITHOUT --update_pool, so
#     it is matched/resolved against that frozen initial pool/state and does not
#     mutate it.
#
# Override anything inline, e.g.:
#   METHOD=vibevoice ./audio_script/run_demo_pipeline_bazinga.sh
#   RAW_DATA_PATH=/.../TheBigBangTheory_SNR10 ./audio_script/run_demo_pipeline_bazinga.sh
#
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
#  Configuration — edit to match your setup
# ──────────────────────────────────────────────────────────────────────

# Backend for Step1: nemo-streaming | nemo-offline | vibevoice
METHOD="${METHOD:-vibevoice}"

# Raw Bazinga audio folder (point at a *_SNRx folder to run on noisy audio).
RAW_DATA_PATH="${RAW_DATA_PATH:-/checkpoint/seamless/tuochao/data/bazinga/data/TheBigBangTheory}"

# Seasons to process, in order. The FIRST entry gets --update_pool (builds the
# initial pool/state); the rest reuse it frozen.
SEASONS=("Season01" "Season02" "Season03")

# Per-phase enable switches (set to 0 to skip a phase).
RUN_STEP1="${RUN_STEP1:-1}"
RUN_STEP2="${RUN_STEP2:-1}"
RUN_STEP3="${RUN_STEP3:-1}"

# Wipe any existing Step2 pool / Step3 state before the loop so the "initial"
# pool truly starts from Season01. Set to 0 to resume on top of existing state.
RESET_STATE="${RESET_STATE:-1}"

# Repo root (added to PYTHONPATH for every phase).
PYTHONPATH_ROOT="/storage/home/tuochao/Mem-alpha-audio"
RESULTS_ROOT="${PYTHONPATH_ROOT}/Audio_Results"
DEVICE="cuda"

# ── Step1 (NeMo backend) models ──────────────────────────────────────
ENV_NEMO="nemo"
ENV_VIBEVOICE="vibevoice"
DIAR_MODEL_PATH="/checkpoint/seamless/tuochao/Models/huggingface/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1.nemo"
ASR_MODEL_PATH="/checkpoint/seamless/tuochao/Models/huggingface/multitalker-parakeet-streaming-0.6b-v1/multitalker-parakeet-streaming-0.6b-v1.nemo"
MAX_NUM_OF_SPKS=4
# ── Step1 (VibeVoice backend) ────────────────────────────────────────
VV_MODEL_PATH="microsoft/VibeVoice-ASR"
MAX_NEW_TOKENS=32768
TEMPERATURE=0.0
TOP_P=1.0
NUM_BEAMS=1
ATTN_IMPL="auto"   # auto | flash_attention_2 | sdpa | eager

# ── Step2 (speaker matching) ─────────────────────────────────────────
ENV_MEM="mem"
EMBEDDING_MODEL_DIR="${EMBEDDING_MODEL_DIR:-/checkpoint/seamless/tuochao/Models/huggingface/wespeaker-voxceleb-resnet293-LM}"
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-0.5}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda:0}"

# ── Step3 (speaker-name extraction) ──────────────────────────────────
QWEN_URL="http://localhost:8002/v1"

# ──────────────────────────────────────────────────────────────────────
#  Derived paths (separate result tree per backend + dataset folder, so
#  SNR variants land in their own dirs automatically).
# ──────────────────────────────────────────────────────────────────────
DATASET_TAG="$(basename "${RAW_DATA_PATH}")"
STEP1_OUT="${RESULTS_ROOT}/${METHOD}/${DATASET_TAG}/step1"
STEP2_OUT="${RESULTS_ROOT}/${METHOD}/${DATASET_TAG}/step2"
POOL_PATH="${STEP2_OUT}/pool.npz"                       # Step2 cross-run state
STATE_PATH="${STEP2_OUT}/speakers_name_pool.json"       # Step3 cross-run state

# ──────────────────────────────────────────────────────────────────────
#  Logging — tee all output to a timestamped log file, and on any failure
#  report which stage/line died (set -e still aborts the run afterwards).
# ──────────────────────────────────────────────────────────────────────
LOG_DIR="${RESULTS_ROOT}/logs"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
# Send stdout+stderr of everything below to the terminal AND the log file.
exec > >(tee -a "${RUN_LOG}") 2>&1

STAGE="init"
trap 'status=$?; echo ""; echo "############################################################"; echo "!!! PIPELINE FAILED (exit ${status}) at stage: ${STAGE} [line ${LINENO}]"; echo "!!! Full log: ${RUN_LOG}"; echo "############################################################"' ERR

echo "Logging to ${RUN_LOG}"

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

export PYTHONPATH="${PYTHONPATH_ROOT}"

echo "============================================================"
echo "  Bazinga pipeline   method=${METHOD}"
echo "  Raw audio : ${RAW_DATA_PATH}"
echo "  Seasons   : ${SEASONS[*]}"
echo "  Step1 out : ${STEP1_OUT}"
echo "  Step2 out : ${STEP2_OUT}"
echo "  Pool/state: ${POOL_PATH} | ${STATE_PATH}"
echo "============================================================"

# ──────────────────────────────────────────────────────────────────────
#  Step 1 — diarization + ASR (once, all seasons)
# ──────────────────────────────────────────────────────────────────────
if [ "${RUN_STEP1}" = "1" ]; then
    case "${METHOD}" in
        nemo-streaming|nemo-offline) STEP1_ENV="${ENV_NEMO}" ;;
        vibevoice)                   STEP1_ENV="${ENV_VIBEVOICE}" ;;
        *) echo "ERROR: unknown METHOD='${METHOD}'"; exit 1 ;;
    esac

    echo ""
    echo ">>> Step 1 (diar+ASR)  env=${STEP1_ENV}"
    conda activate "${STEP1_ENV}"
    [ "${METHOD}" = "nemo-offline" ] && export CUDA_LAUNCH_BLOCKING=1

    STEP1_ARGS=(
        --method     "${METHOD}"
        --data_dir   "${RAW_DATA_PATH}"
        --output_dir "${STEP1_OUT}"
        --device     "${DEVICE}"
        --season_filter "${SEASONS[@]}"
    )
    case "${METHOD}" in
        nemo-streaming|nemo-offline)
            STEP1_ARGS+=(
                --diar_model_path "${DIAR_MODEL_PATH}"
                --asr_model_path  "${ASR_MODEL_PATH}"
                --max_num_of_spks "${MAX_NUM_OF_SPKS}"
            ) ;;
        vibevoice)
            STEP1_ARGS+=(
                --model_path          "${VV_MODEL_PATH}"
                --max_new_tokens      "${MAX_NEW_TOKENS}"
                --temperature         "${TEMPERATURE}"
                --top_p               "${TOP_P}"
                --num_beams           "${NUM_BEAMS}"
                --attn_implementation "${ATTN_IMPL}"
            ) ;;
    esac

    STAGE="Step1 (diar+ASR) seasons=${SEASONS[*]}"
    python -m audio_script.Multi_ASR.step1_bazinga "${STEP1_ARGS[@]}"
    conda deactivate
fi

# ──────────────────────────────────────────────────────────────────────
#  Optionally reset cross-run state so Season01 builds the initial pool.
# ──────────────────────────────────────────────────────────────────────
if [ "${RESET_STATE}" = "1" ]; then
    echo ""
    echo ">>> Resetting cross-run state (RESET_STATE=1)"
    rm -f "${POOL_PATH}" "${STATE_PATH}"
fi

# ──────────────────────────────────────────────────────────────────────
#  Step 2 + Step 3 — season by season, --update_pool only on the first.
# ──────────────────────────────────────────────────────────────────────
if [ "${RUN_STEP2}" = "1" ] || [ "${RUN_STEP3}" = "1" ]; then
    conda activate "${ENV_MEM}"
    export QWEN_URL="${QWEN_URL}"
    mkdir -p "${STEP2_OUT}"

    for idx in "${!SEASONS[@]}"; do
        SEASON="${SEASONS[$idx]}"
        if [ "${idx}" -eq 0 ]; then UPDATE=1; else UPDATE=0; fi

        echo ""
        echo "============================================================"
        echo "  Season ${SEASON}   (update_pool=${UPDATE})"
        echo "============================================================"

        # ── Step 2: speaker matching ──────────────────────────────────
        if [ "${RUN_STEP2}" = "1" ]; then
            echo ">>> Step 2 (speaker matching)  season=${SEASON}"
            STEP2_ARGS=(
                --data_dir             "${STEP1_OUT}"
                --output_dir           "${STEP2_OUT}"
                --embedding_model_dir  "${EMBEDDING_MODEL_DIR}"
                --similarity_threshold "${SIMILARITY_THRESHOLD}"
                --embedding_device     "${EMBEDDING_DEVICE}"
                --pool_path            "${POOL_PATH}"
                --season_filter        "${SEASON}"
            )
            [ "${UPDATE}" -eq 1 ] && STEP2_ARGS+=(--update_pool)
            STAGE="Step2 (speaker matching) season=${SEASON} update_pool=${UPDATE}"
            python "audio_script/Speaker_Track/step2_speaker_match_v2.py" "${STEP2_ARGS[@]}"
        fi

        # ── Step 3: speaker-name extraction ───────────────────────────
        if [ "${RUN_STEP3}" = "1" ]; then
            echo ">>> Step 3 (name extraction)  season=${SEASON}"
            STEP3_ARGS=(
                --data_dir      "${STEP2_OUT}"
                --state_path    "${STATE_PATH}"
                --season_filter "${SEASON}"
            )
            [ "${UPDATE}" -eq 1 ] && STEP3_ARGS+=(--update_pool)
            STAGE="Step3 (name extraction) season=${SEASON} update_pool=${UPDATE}"
            python "audio_script/Speaker_Track/step3_speaker_name_extract.py" "${STEP3_ARGS[@]}"
        fi
    done

    conda deactivate
fi

echo ""
echo "Pipeline complete for ${DATASET_TAG} (${METHOD})."
