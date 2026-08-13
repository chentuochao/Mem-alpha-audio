#!/usr/bin/env bash
#
# End-to-end Mix_Mosaic audio pipeline driver:
#   Step1 (diar+ASR) -> Step2 (speaker matching) -> Step3 (speaker-name
#   extraction), in one script.
#
# Mirrors run_demo_pipeline_bazinga.sh (no Step0 — Mix_Mosaic is already mixed
# on disk with transcripts) and borrows run_demo_pipeline_perltqa.sh's BUNDLE
# mode for Step2/Step3. Differences:
#   - Step1 uses the Mix_Mosaic loader (audio_script.Multi_ASR.step1_mosaic).
#     It runs ONCE over every conversation (no cross-conversation state).
#   - Step2/Step3 run per BUNDLE from bundles.json (built by
#     audio_script/make_mix_mosaic_bundles.py). Each bundle becomes an
#     INDEPENDENT speaker pool + name state (bundle == "season"; the bundle's
#     pair-folder names are the Step2 --season_filter). With BUNDLE_MANIFEST
#     empty, Step2/Step3 fall back to a single global pool over everything.
#
# Override anything inline, e.g.:
#   METHOD=nemo-offline ./audio_script/run_demo_pipeline_mosaic.sh
#   BUNDLE_MANIFEST="" ./audio_script/run_demo_pipeline_mosaic.sh   # global pool
#
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
#  Configuration — edit to match your setup
# ──────────────────────────────────────────────────────────────────────

# Backend for Step1: nemo-streaming | nemo-offline | vibevoice
METHOD="${METHOD:-vibevoice}"

# Mix_Mosaic root (holds Pxxx_Pyyy/<conv>/mixed_conv.wav dirs).
RAW_DATA_PATH="${RAW_DATA_PATH:-/checkpoint/seamless/tuochao/data/Mix_Mosaic/naturalistic/test}"

# Bundle manifest (make_mix_mosaic_bundles.py output). Each bundle == one
# independent Step2/Step3 pool. Set empty for a single global pool.
BUNDLE_MANIFEST="${BUNDLE_MANIFEST:-${RAW_DATA_PATH}/bundles.json}"

# Per-phase enable switches (set to 0 to skip a phase).
RUN_STEP1="${RUN_STEP1:-1}"
RUN_STEP2="${RUN_STEP2:-1}"
RUN_STEP3="${RUN_STEP3:-1}"

# Wipe any existing Step2 pool / Step3 state before running so pools are built
# fresh in this run. Set to 0 to resume on top of existing state.
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
# Must match launch_vllm.sh's --served-model-name.
QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-qwen3-32b}"

# ──────────────────────────────────────────────────────────────────────
#  Derived paths (separate result tree per backend + dataset folder).
# ──────────────────────────────────────────────────────────────────────
DATASET_TAG="$(basename "${RAW_DATA_PATH}")"
STEP1_OUT="${RESULTS_ROOT}/${METHOD}/${DATASET_TAG}/step1"
STEP2_OUT="${RESULTS_ROOT}/${METHOD}/${DATASET_TAG}/step2"
POOL_PATH="${STEP2_OUT}/pool.npz"                       # global-mode Step2 state
STATE_PATH="${STEP2_OUT}/speakers_name_pool.json"       # global-mode Step3 state

# ──────────────────────────────────────────────────────────────────────
#  Logging — tee all output to a timestamped log file, and on any failure
#  report which stage/line died (set -e still aborts the run afterwards).
# ──────────────────────────────────────────────────────────────────────
LOG_DIR="${RESULTS_ROOT}/logs"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/pipeline_mosaic_$(date +%Y%m%d_%H%M%S).log"
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
# Stream Python prints live (stdout is a tee pipe here, so otherwise the slow,
# silent WeSpeaker load + embedding extraction looks "stuck").
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "  Mix_Mosaic pipeline   method=${METHOD}"
echo "  Data dir  : ${RAW_DATA_PATH}"
echo "  Manifest  : ${BUNDLE_MANIFEST:-<none: global pool>}"
echo "  Step1 out : ${STEP1_OUT}"
echo "  Step2 out : ${STEP2_OUT}"
echo "============================================================"

# ──────────────────────────────────────────────────────────────────────
#  Step 1 — diarization + ASR (once, all conversations)
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

    STAGE="Step1 (diar+ASR)"
    python -m audio_script.Multi_ASR.step1_mosaic "${STEP1_ARGS[@]}"
    conda deactivate
fi

# ──────────────────────────────────────────────────────────────────────
#  Optionally reset cross-run state so pools are built fresh this run.
# ──────────────────────────────────────────────────────────────────────
if [ "${RESET_STATE}" = "1" ]; then
    echo ""
    echo ">>> Resetting cross-run state (RESET_STATE=1)"
    if [ -n "${BUNDLE_MANIFEST}" ]; then
        rm -rf "${STEP2_OUT}"      # wipe all per-bundle pools/state
    else
        rm -f "${POOL_PATH}" "${STATE_PATH}"
    fi
fi

# ──────────────────────────────────────────────────────────────────────
#  Step 2 + Step 3
#    - BUNDLE_MANIFEST set   -> one independent pool per bundle
#                               (bundle == season; pair folders == filter)
#    - BUNDLE_MANIFEST empty  -> single global pool over everything
# ──────────────────────────────────────────────────────────────────────
run_step2() {  # $1=data_dir(step1 out)  $2=output_dir  $3=pool_path ; extra args = season_filter
    local data_dir="$1" out_dir="$2" pool="$3"; shift 3
    local args=(
        --data_dir             "${data_dir}"
        --output_dir           "${out_dir}"
        --embedding_model_dir  "${EMBEDDING_MODEL_DIR}"
        --similarity_threshold "${SIMILARITY_THRESHOLD}"
        --embedding_device     "${EMBEDDING_DEVICE}"
        --pool_path            "${pool}"
        --update_pool
    )
    [ "$#" -gt 0 ] && args+=(--season_filter "$@")
    python "audio_script/Speaker_Track/step2_speaker_match_v2.py" "${args[@]}"
}

run_step3() {  # $1=data_dir(step2 out)  $2=state_path
    python "audio_script/Speaker_Track/step3_speaker_name_extract.py" \
        --data_dir "$1" --state_path "$2" --update_pool
}

if [ "${RUN_STEP2}" = "1" ] || [ "${RUN_STEP3}" = "1" ]; then
    conda activate "${ENV_MEM}"
    export QWEN_URL="${QWEN_URL}"
    export QWEN_MODEL_NAME="${QWEN_MODEL_NAME}"
    mkdir -p "${STEP2_OUT}"

    if [ -n "${BUNDLE_MANIFEST}" ]; then
        echo ""
        echo ">>> Step 2/3 in BUNDLE mode: ${BUNDLE_MANIFEST}"
        [ -f "${BUNDLE_MANIFEST}" ] || { echo "ERROR: manifest not found: ${BUNDLE_MANIFEST}"; exit 1; }

        # emit "<bundle_id>\t<pair1> <pair2> ..." per bundle (Mix_Mosaic manifest
        # lists pair folders under b["folders"]).
        while IFS=$'\t' read -r BID PAIRS; do
            read -ra PARR <<< "${PAIRS}"
            B2_OUT="${STEP2_OUT}/bundle_${BID}"
            mkdir -p "${B2_OUT}"
            echo ""
            echo "──────── bundle ${BID}: ${#PARR[@]} pair-folder(s) ────────"

            if [ "${RUN_STEP2}" = "1" ]; then
                STAGE="Step2 bundle ${BID}"
                run_step2 "${STEP1_OUT}" "${B2_OUT}" "${B2_OUT}/pool.npz" "${PARR[@]}"
            fi
            if [ "${RUN_STEP3}" = "1" ]; then
                STAGE="Step3 bundle ${BID}"
                run_step3 "${B2_OUT}" "${B2_OUT}/state.json"
            fi
        done < <(python - "${BUNDLE_MANIFEST}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for b in m["bundles"]:
    print(f"{b['bundle_id']}\t{' '.join(b['folders'])}")
PY
)
    else
        echo ""
        echo ">>> Step 2/3 in GLOBAL mode (single pool)"
        [ "${RUN_STEP2}" = "1" ] && { STAGE="Step2 (global)"; run_step2 "${STEP1_OUT}" "${STEP2_OUT}" "${POOL_PATH}"; }
        [ "${RUN_STEP3}" = "1" ] && { STAGE="Step3 (global)"; run_step3 "${STEP2_OUT}" "${STATE_PATH}"; }
    fi

    conda deactivate
fi

echo ""
echo "Pipeline complete for ${DATASET_TAG} (${METHOD})."
