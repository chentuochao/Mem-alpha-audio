#!/usr/bin/env bash
# Run the BEHAVIORAL (counterfactual) error probe for one constructed PERLTQA instance.
#
# Same machinery as run_probe_errors.sh (see that file for the G/C/T/S probe semantics);
# this wrapper just wires in the perltqa defaults and grading:
#
#   * perltqa QA gt_source.file is a profile/session ref ('Cao_Lili/25_0_0_0'); the
#     probe normalizes it to the on-disk / parquet chunk folder
#     ('Cao_Lili_25_0_0_0/CHUNK_0/parsed_dialog_gt.json') via --data_source perltqa,
#     so evidence localization stays DETERMINISTIC (parquet chunk_folders map, no
#     fuzzy matching), exactly as for the audio datasets.
#   * QA accuracy is NOT multiple-choice here. Each probe re-answer is graded with
#     the same rule evaluate_agent_results.py uses for perltqa: SCORER=keyword
#     (default, ';'-split containment, no API) or SCORER=llm_judge (LLM-as-judge via
#     the QWEN_URL server; needs QWEN_URL / QWEN_MODEL_NAME / OPENROUTER_API_KEY).
#
# Usage:
#   bash diagnostic/run_probe_errors_perltqa.sh [BASE_DIR] [DATA_ROOT] [QA_FILE] [PARQUET] [SERVER_URL]
#   SCORER=llm_judge bash diagnostic/run_probe_errors_perltqa.sh ...   # LLM-judge grading
#   RUN_GOLDEN=1     bash diagnostic/run_probe_errors_perltqa.sh ...   # also run the G-probe
#
# BASE_DIR is the run dir (holding 0/ and seed*/0/), not the .../0 subdir. The T/G
# probes are precomputed once per DATA_ROOT into <DATA_ROOT>/tg_probe_cache.json and
# reused across base_dirs / seeds sharing that DATA_ROOT.
set -euo pipefail

DATA_SOURCE="perltqa"
# perltqa grading of each probe re-answer: keyword (default, no API) | llm_judge.
SCORER="${SCORER:-keyword}"

# The llm_judge scorer grades through the OpenAI client, reading QWEN_URL /
# QWEN_MODEL_NAME / OPENROUTER_API_KEY (same as evaluate_agent_results.py). These are
# NOT exported into your interactive shell by launch_servers.sh, so an unset QWEN_URL
# makes the client fall back to api.openai.com and fail with a 401 "EMPTY key". Default
# them to the local vLLM backend (matching launch_servers.sh's VLLM_PORT/QWEN_MODEL_NAME)
# when unset; export your own (e.g. OpenRouter) to override.
if [ "$SCORER" = "llm_judge" ]; then
    export QWEN_URL="${QWEN_URL:-http://localhost:8002/v1}"
    export QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-qwen3-32b}"
    export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-EMPTY}"
    echo "[llm_judge] QWEN_URL=$QWEN_URL  QWEN_MODEL_NAME=$QWEN_MODEL_NAME"
fi

# Base run dir (NOT the .../0 subdir). Tolerate a trailing "/0".
BASE_DIR="${1:-agents/qwen3.6-27b_Qwen_Qwen3.6-27B_perltqa_dataset_pred_name_bundle_0_no_thinking_tokens_2048}"
BASE_DIR="${BASE_DIR%/}"
BASE_DIR="${BASE_DIR%/0}"
# Dialogue root: each chunk folder holds BOTH parsed_dialog_gt.json (gold evidence)
# and parsed_dialog_pred.json (ASR transcript for the T-probe).
DATA_ROOT="${2:-outputs/step3_perltqa/bundle_0}"
QA_FILE="${3:-outputs/perltqa_data/qa_multi/bundle_0_filterd/qa.jsonl}"
# Parquet whose chunk_folders list maps chunk_idx -> "{profile}_{session}/CHUNK_0".
# The agent was built from the pred-name variant, so map against that one.
PARQUET="${4:-outputs/step3_perltqa/bundle_0/dataset_gt_name_bundle_0.parquet}"
SERVER_URL="${5:-http://127.0.0.1:5005/batch_process}"

# G-probe (gold-dialogue re-answer, the ceiling) is off by default. Set RUN_GOLDEN=1.
RUN_GOLDEN="${RUN_GOLDEN:-}"
GOLDEN_ARGS=()
[ -n "$RUN_GOLDEN" ] && GOLDEN_ARGS=(--run_golden)

# T/G re-answers depend only on the DATA_ROOT dialogues + question, so every base_dir /
# seed sharing this DATA_ROOT reuses one cache (kept in DATA_ROOT for cross-run reuse).
TG_CACHE="$DATA_ROOT/tg_probe_cache.json"

# Enumerate the instance dirs to probe: <BASE_DIR>/0 plus any <BASE_DIR>/seedX/0.
INSTANCE_DIR="$BASE_DIR/0"
INSTANCE_DIRS=()
[ -f "$INSTANCE_DIR/results.json" ] && INSTANCE_DIRS+=("$INSTANCE_DIR")
for sd in "$BASE_DIR"/seed*/; do
    [ -d "$sd" ] || continue
    [ -f "${sd}0/results.json" ] && INSTANCE_DIRS+=("${sd}0")
done

if [ ${#INSTANCE_DIRS[@]} -eq 0 ]; then
    echo "ERROR: no instance dir with results.json found (looked at $INSTANCE_DIR and $BASE_DIR/seed*/0)" >&2
    exit 1
fi

echo "Probing ${#INSTANCE_DIRS[@]} instance dir(s)  [data_source=$DATA_SOURCE  scorer=$SCORER]:"
printf '  %s\n' "${INSTANCE_DIRS[@]}"

# Score each subfolder before probing (writes its own evaluation_metrics.json).
for idir in "${INSTANCE_DIRS[@]}"; do
    echo
    echo "================= evaluating $idir ================="
    python evaluate_agent_results.py --base_dir "$idir"
done

# Precompute the T (and, with RUN_GOLDEN, G) probe answers ONCE for this DATA_ROOT.
echo
echo "================= precomputing T/G probes -> $TG_CACHE ================="
PYTHONPATH=diagnostic python diagnostic/precompute_tg_probes.py \
    --qa_file "$QA_FILE" \
    --data_root "$DATA_ROOT" \
    --parquet "$PARQUET" \
    --server_url "$SERVER_URL" \
    --data_source "$DATA_SOURCE" \
    --cache "$TG_CACHE" \
    "${GOLDEN_ARGS[@]}"

for idir in "${INSTANCE_DIRS[@]}"; do
    echo
    # Skip if this instance dir was already probed (both --debug outputs present).
    if [ -f "$idir/error_probe.json" ] && [ -f "$idir/error_probe_debug.json" ]; then
        echo "================= skipping $idir (error_probe.json + error_probe_debug.json exist) ================="
        continue
    fi
    echo "================= probing $idir ================="
    # T/G stages load READ-ONLY from the shared cache above; only C and S hit the server.
    PYTHONPATH=diagnostic python diagnostic/probe_errors.py \
        --instance_dir "$idir" \
        --qa_file "$QA_FILE" \
        --data_root "$DATA_ROOT" \
        --parquet "$PARQUET" \
        --server_url "$SERVER_URL" \
        --data_source "$DATA_SOURCE" \
        --scorer "$SCORER" \
        --full_qa \
        --debug \
        --tg_probe_cache "$TG_CACHE" \
        "${GOLDEN_ARGS[@]}"
done

# Aggregate per-seed summaries when more than one seed was probed.
if [ ${#INSTANCE_DIRS[@]} -gt 1 ]; then
    echo
    echo "================= aggregating ${#INSTANCE_DIRS[@]} seed probes ================="
    PYTHONPATH=diagnostic python diagnostic/aggregate_probe_seeds.py --base_dir "$BASE_DIR"
fi
