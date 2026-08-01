#!/usr/bin/env bash
# Run the BEHAVIORAL (counterfactual) error probe for one constructed instance.
#
# Unlike trace_errors_clean.py (which decides attribution by string-MATCHING gold
# evidence against memory), this re-runs the real QA model on curated contexts and
# attributes by whether the ANSWER flips. Evidence is localized DETERMINISTICALLY via
# the parquet's chunk_folders map (QA gt_source folder -> chunk_idx -> memory ids),
# so no fuzzy matching. It needs the memory server up.
#
# Usage:
#   bash diagnostic/run_probe_errors.sh [BASE_DIR] [DATA_ROOT] [QA_FILE] [PARQUET] [SERVER_URL]
#   RUN_GOLDEN=1 bash diagnostic/run_probe_errors.sh ...   # also run the G-probe
#
# BASE_DIR is the run dir (holding 0/ and seed*/0/), not the .../0 subdir. The T/G
# probes are precomputed once per DATA_ROOT into <DATA_ROOT>/tg_probe_cache.json and
# reused by every base_dir / seed / compression variant sharing that DATA_ROOT; the
# per-instance probes then only run the C and S stages.
set -euo pipefail

# Transcript matcher for the T-probe stays lexical-only (no API keys needed); the
# embedding / LLM-judge layers auto-enable only if OPENAI_API_KEY / OPENROUTER_API_KEY
# are set. The QA re-answers themselves go through the memory server below.
unset OPENAI_API_KEY || true

# Base run dir (NOT the .../0 subdir). The agent subdir is always "0/", so the
# original instance is <BASE_DIR>/0 and per-seed instances are <BASE_DIR>/seedX/0.
BASE_DIR="${1:-./memory_result/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_Season1_vibevoice_gt_name_no_thinking_tokens_2048}"
# Tolerate a trailing "/0" (or "/0/") if the caller still passes the old form.
BASE_DIR="${BASE_DIR%/}"
BASE_DIR="${BASE_DIR%/0}"
# Single dialogue root: each chunk holds BOTH parsed_dialog_gt.json (gold evidence)
# and parsed_dialog_pred.json (ASR transcript for the T-probe).
DATA_ROOT="${2:-outputs/step3_anony/S01_S03_Clean_Anoy}"

QA_FILE="${3:-outputs/step3_anony/qas/merged_qa_anoy.jsonl}"
PARQUET="${4:-outputs/step3_anony/S01_S03_Clean_Anoy/dataset_gt_name_Season01_Clean_Anoy.parquet}"
SERVER_URL="${5:-http://127.0.0.1:5005/batch_process}"
# Parquet whose chunk_folders list maps chunk_idx -> "{episode}/CHUNK_N" (see
# prepare_parquet_from_step3.py). Empty -> auto-discover dataset_gt_name_*.parquet in DATA_ROOT.

# G-probe (gold-dialogue re-answer, the ceiling) is off by default. Set RUN_GOLDEN=1 to
# precompute it and have the probe load it. Default (unset) skips G everywhere.
RUN_GOLDEN="${RUN_GOLDEN:-}"
GOLDEN_ARGS=()
[ -n "$RUN_GOLDEN" ] && GOLDEN_ARGS=(--run_golden)

# The T/G re-answers depend only on the DATA_ROOT transcripts + question, not on the
# per-instance memory, so ALL base_dirs / seeds / compression variants sharing this
# DATA_ROOT reuse one cache. It lives in DATA_ROOT (not BASE_DIR) for cross-run reuse.
TG_CACHE="$DATA_ROOT/tg_probe_cache.json"


# Enumerate the instance dirs to probe: the original `<BASE_DIR>/0` plus any
# per-seed subfolders `<BASE_DIR>/seedX/0`. Runs without any seedX subfolder just
# probe the original `0/` exactly as before.
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

echo "Probing ${#INSTANCE_DIRS[@]} instance dir(s):"
printf '  %s\n' "${INSTANCE_DIRS[@]}"

# Score each subfolder (the original 0/ and every seedX/0/) individually before
# the error probing. evaluate_agent_results.py walks the given dir, so pointing
# --base_dir at each instance dir writes that folder's own evaluation_metrics.json.
for idir in "${INSTANCE_DIRS[@]}"; do
    echo
    echo "================= evaluating $idir ================="
    python evaluate_agent_results.py --base_dir "$idir"
done

# Precompute the T (and, with RUN_GOLDEN, G) probe answers ONCE for this DATA_ROOT into
# the shared cache. Driven straight from the qa_file (no memory needed), so it covers a
# superset of what the probes below request; idempotent (only misses hit the server).
echo
echo "================= precomputing T/G probes -> $TG_CACHE ================="
PYTHONPATH=diagnostic python diagnostic/precompute_tg_probes.py \
    --qa_file "$QA_FILE" \
    --data_root "$DATA_ROOT" \
    --parquet "$PARQUET" \
    --server_url "$SERVER_URL" \
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
    # T/G stages load READ-ONLY from the shared cache above (hard error on miss); only
    # C and S hit the server per instance.
    PYTHONPATH=diagnostic python diagnostic/probe_errors.py \
        --instance_dir "$idir" \
        --qa_file "$QA_FILE" \
        --data_root "$DATA_ROOT" \
        --parquet "$PARQUET" \
        --server_url "$SERVER_URL" \
        --full_qa \
        --debug \
        --tg_probe_cache "$TG_CACHE" \
        "${GOLDEN_ARGS[@]}"
done

# When more than one seed was probed, aggregate the per-instance error_probe.json
# summaries into a single mean ± std report across seeds (written to
# <base_dir>/error_probe_seed_summary.json).
if [ ${#INSTANCE_DIRS[@]} -gt 1 ]; then
    echo
    echo "================= aggregating ${#INSTANCE_DIRS[@]} seed probes ================="
    PYTHONPATH=diagnostic python diagnostic/aggregate_probe_seeds.py --base_dir "$BASE_DIR"
fi
