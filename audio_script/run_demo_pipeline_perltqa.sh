#!/usr/bin/env bash
#
# End-to-end PerLTQA audio pipeline driver:
#   Step0 (build GT annotations) -> Step1 (diar+ASR) -> Step2 (speaker matching)
#   -> Step3 (speaker-name extraction), in one script.
#
# Mirrors run_demo_pipeline_bazinga.sh, adapted for the synthesized PerLTQA
# dialogue-TTS data (ctbox_tts/perltqa_dialogue_tts.py output). Differences:
#   - Step0 builds per-speaker ground truth (turn text from channel_map.json +
#     Silero VAD) with ctbox_tts/generate_annotations.py. Step1 REQUIRES these,
#     so it runs first (skip with RUN_ANNOTATE=0 if annotations already exist).
#   - Step1 uses the PerLTQA loader (audio_script.Multi_ASR.step1_perltqa).
#   - PerLTQA has no "seasons": Step2/Step3 run ONCE over all dialogues and
#     build the pool/state in a single pass (--update_pool). There is no
#     season-by-season incremental loop.
#
# Override anything inline, e.g.:
#   METHOD=nemo-streaming ./audio_script/run_demo_pipeline_perltqa.sh
#   RAW_DATA_PATH=/.../PerLTQA/dialogue_tts_en_v2 ./audio_script/run_demo_pipeline_perltqa.sh
#

# # step1 once (on the 31 valid profiles) + reuse for both modes
# RUN_STEP2=0 RUN_STEP3=0 bash audio_script/run_demo_pipeline_perltqa.sh

# # per-profile pools (30 independent pools)
# BUNDLE_MANIFEST=/checkpoint/seamless/tuochao/data/PerLTQA/dialogue_tts_en_v2/bundles_per_profile.json \
#   RUN_STEP1=0 bash audio_script/run_demo_pipeline_perltqa.sh

# # multi pools (3 bundles)
# BUNDLE_MANIFEST=/checkpoint/seamless/tuochao/data/PerLTQA/dialogue_tts_en_v2/bundles_multi.json \
#   RUN_STEP1=0 bash audio_script/run_demo_pipeline_perltqa.sh

# # or the whole thing in one go
# BUNDLE_MANIFEST=.../bundles_multi.json bash audio_script/run_demo_pipeline_perltqa.sh

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
#  Configuration — edit to match your setup
# ──────────────────────────────────────────────────────────────────────

# Backend for Step1: nemo-streaming | nemo-offline | vibevoice
METHOD="${METHOD:-vibevoice}"

# PerLTQA dialogue-TTS output folder (holds <Profile>/<dialogue_id>/ dirs).
RAW_DATA_PATH="${RAW_DATA_PATH:-/checkpoint/seamless/tuochao/data/PerLTQA/dialogue_tts_en_v2}"

# Per-phase enable switches (set to 0 to skip a phase).
RUN_ANNOTATE="${RUN_ANNOTATE:-1}"   # Step0: build *_annotation.json (GT)
RUN_STEP1="${RUN_STEP1:-1}"
RUN_STEP2="${RUN_STEP2:-1}"
RUN_STEP3="${RUN_STEP3:-1}"

# Re-annotate even if *_annotation.json already exists (Step0 --overwrite).
ANNOTATE_OVERWRITE="${ANNOTATE_OVERWRITE:-0}"

# Wipe any existing Step2 pool / Step3 state before running so the pool is built
# fresh in this run. Set to 0 to resume on top of existing state.
RESET_STATE="${RESET_STATE:-1}"

# Repo root (added to PYTHONPATH for every phase).
PYTHONPATH_ROOT="/storage/home/tuochao/Mem-alpha-audio"
RESULTS_ROOT="${PYTHONPATH_ROOT}/Audio_Results"
DEVICE="cuda"

# ── Bundle manifests ─────────────────────────────────────────────────
# Step1 only transcribes profiles referenced by these manifests (the ones with
# QAs), not all ~141 profiles. Uses the union across both files so a single
# Step1 run covers whichever bundle mode you evaluate later.
PP_MANIFEST="${PP_MANIFEST:-${RAW_DATA_PATH}/bundles_per_profile.json}"
MULTI_MANIFEST="${MULTI_MANIFEST:-${RAW_DATA_PATH}/bundles_multi.json}"

# Which manifest drives Step2/Step3 grouping. Each bundle becomes an INDEPENDENT
# speaker pool + name state (one bundle == one "season"). Set to a manifest path
# to run per-bundle; leave empty for a single global pool over everything.
#   BUNDLE_MANIFEST="${PP_MANIFEST}"     -> per-profile pools (30)
#   BUNDLE_MANIFEST="${MULTI_MANIFEST}"  -> multi-profile pools (3)
BUNDLE_MANIFEST="${BUNDLE_MANIFEST:-}"

# ── Step0 (annotations) ──────────────────────────────────────────────
ENV_CTBOX="ctbox"

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
# Must match launch_vllm.sh's --served-model-name (the server only answers to
# that name, not the HF repo id "Qwen/Qwen3-32B").
QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-qwen3-32b}"

# ──────────────────────────────────────────────────────────────────────
#  Derived paths (separate result tree per backend + dataset folder).
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
RUN_LOG="${LOG_DIR}/pipeline_perltqa_$(date +%Y%m%d_%H%M%S).log"
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
# Stream Python prints live: stdout is a pipe (tee) here, so without this the
# per-step progress is block-buffered and the run looks "stuck" during the
# (slow, silent) WeSpeaker model load + embedding extraction.
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "  PerLTQA pipeline   method=${METHOD}"
echo "  Data dir  : ${RAW_DATA_PATH}"
echo "  Step1 out : ${STEP1_OUT}"
echo "  Step2 out : ${STEP2_OUT}"
echo "  Pool/state: ${POOL_PATH} | ${STATE_PATH}"
echo "============================================================"

# ──────────────────────────────────────────────────────────────────────
#  Step 0 — build ground-truth annotations (turn text + Silero VAD)
# ──────────────────────────────────────────────────────────────────────
if [ "${RUN_ANNOTATE}" = "1" ]; then
    echo ""
    echo ">>> Step 0 (annotations)  env=${ENV_CTBOX}"
    conda activate "${ENV_CTBOX}"
    ANNOTATE_ARGS=(
        --output-dir "${RAW_DATA_PATH}"
    )
    [ "${ANNOTATE_OVERWRITE}" = "1" ] && ANNOTATE_ARGS+=(--overwrite)
    STAGE="Step0 (annotations)"
    python "${PYTHONPATH_ROOT}/ctbox_tts/generate_annotations.py" "${ANNOTATE_ARGS[@]}"
    conda deactivate
fi

# ──────────────────────────────────────────────────────────────────────
#  Step 1 — diarization + ASR (PerLTQA loader)
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

    # Restrict Step1 to profiles referenced by the bundle manifests (skip the
    # ~110 QA-less profiles). Names are anchored with a trailing "_" so they
    # match conv_id "<Profile>_<dialogue>" exactly (avoids prefix collisions).
    mapfile -t VALID_FILTERS < <(python - "${PP_MANIFEST}" "${MULTI_MANIFEST}" <<'PY'
import json, sys
profs = set()
for f in sys.argv[1:]:
    try:
        m = json.load(open(f))
    except Exception:
        continue
    for b in m.get("bundles", []):
        for p in b.get("profiles", []):
            profs.add(p["profile"])
for p in sorted(profs):
    print(p + "_")
PY
)

    STEP1_ARGS=(
        --method     "${METHOD}"
        --data_dir   "${RAW_DATA_PATH}"
        --output_dir "${STEP1_OUT}"
        --device     "${DEVICE}"
    )
    if [ "${#VALID_FILTERS[@]}" -gt 0 ]; then
        echo "[step1] restricting to ${#VALID_FILTERS[@]} valid profile(s) from manifests"
        STEP1_ARGS+=(--season_filter "${VALID_FILTERS[@]}")
    else
        echo "[step1][WARN] no manifest profiles found; running Step1 on ALL dialogues"
    fi
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
    python -m audio_script.Multi_ASR.step1_perltqa "${STEP1_ARGS[@]}"
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
#    - BUNDLE_MANIFEST set  -> one independent pool per bundle
#                              (bundle == season; profiles == episodes)
#    - BUNDLE_MANIFEST empty -> single global pool over everything
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

        # emit "<bundle_id>\t<profile>_ <profile>_ ..." per bundle
        while IFS=$'\t' read -r BID PROFS; do
            read -ra PARR <<< "${PROFS}"
            B2_OUT="${STEP2_OUT}/bundle_${BID}"
            mkdir -p "${B2_OUT}"
            echo ""
            echo "──────── bundle ${BID}: ${#PARR[@]} profile(s) ────────"

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
    profs = " ".join(p["profile"] + "_" for p in b["profiles"])
    print(f"{b['bundle_id']}\t{profs}")
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
