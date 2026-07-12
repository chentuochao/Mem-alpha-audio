#!/usr/bin/env bash
# Run the BEHAVIORAL (counterfactual) error probe for one constructed instance.
#
# Unlike trace_errors_clean.py (which decides attribution by string-MATCHING gold
# evidence against memory), this re-runs the real QA model on curated contexts and
# attributes by whether the ANSWER flips. It therefore needs the memory server up.
#
# Usage:
#   bash diagnostic/run_probe_errors.sh [INSTANCE_DIR] [SERVER_URL]
set -euo pipefail

# Transcript matcher for the T-probe stays lexical-only (no API keys needed); the
# embedding / LLM-judge layers auto-enable only if OPENAI_API_KEY / OPENROUTER_API_KEY
# are set. The QA re-answers themselves go through the memory server below.
unset OPENAI_API_KEY || true

INSTANCE_DIR="${1:-./memory_result/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_Season1_vibevoice_gt_name_no_thinking_tokens_2048/0}"
SERVER_URL="${2:-http://127.0.0.1:5005/batch_process}"

PYTHONPATH=diagnostic python diagnostic/probe_errors.py \
    --instance_dir "$INSTANCE_DIR" \
    --server_url "$SERVER_URL"
