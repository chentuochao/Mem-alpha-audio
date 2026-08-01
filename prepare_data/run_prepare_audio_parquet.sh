#!/usr/bin/env bash
#
# Build per-season audio parquets from the step2 sequence output for
# Season01-Season03, each BOTH ways:
#   - with ground-truth speaker names  (--use_gt_name)
#   - with predicted speaker names     (no flag)
#
# 3 seasons x 2 name modes = 6 runs.
#
# Override any path inline, e.g.:
#   DATA_DIR=./Audio_Results/vibevoice/TheBigBangTheory_SNR5/step2_sequence/ \
#     ./prepare_data/run_prepare_audio_parquet.sh
#
set -euo pipefail

# Repo root so `PYTHONPATH=.` and the relative default paths resolve no matter
# where the script is invoked from.
REPO_ROOT="/storage/home/tuochao/Mem-alpha-audio"
cd "${REPO_ROOT}"

# ── Config ────────────────────────────────────────────────────────────
FOLDER_NAME="TheBigBangTheory_interf_SNR10"
DATA_DIR="${DATA_DIR:-./Audio_Results/vibevoice/${FOLDER_NAME}/step2/}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/step3_new/${FOLDER_NAME}}"
SEASONS=("Season01" "Season02" "Season03")

echo "============================================================"
echo "  prepare_audio_parquet  (Season01-03, gt + pred names)"
echo "  data_dir    : ${DATA_DIR}"
echo "  output_root : ${OUTPUT_ROOT}"
echo "============================================================"

for season in "${SEASONS[@]}"; do
    for names in gt pred; do
        args=(
            --data_dir      "${DATA_DIR}"
            --output_root   "${OUTPUT_ROOT}"
            --season_filter "${season}"
        )
        [ "${names}" = "gt" ] && args+=(--use_gt_name)

        echo ""
        echo ">>> ${season} | names=${names}"
        echo "    python prepare_data/prepare_audio_parquet.py ${args[*]}"
        PYTHONPATH=. python prepare_data/prepare_audio_parquet.py "${args[@]}"
    done
done

echo ""
echo "All 6 runs complete. Parquets under: ${OUTPUT_ROOT}"
