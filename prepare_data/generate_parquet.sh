#!/usr/bin/env bash
#
# Build the step3 Parquet dataset for one dataset with prepare_audio_parquet.py.
#
# Usage:
#   bash run_prepare_parquet.sh <DATASETNAME>
#
#   DATASETNAME     Name under ./Audio_Results/vibevoice/<DATASETNAME>/step2
#
# Derived paths:
#   data_dir     = ./Audio_Results/vibevoice/<DATASETNAME>/step2
#   output_root  = outputs/step3_new/<DATASETNAME>
#
# Optional env-var overrides:
#   MODE=pred|gt|both   Which parquet(s) to build (default: pred)
#   SEASON_FILTER=Season01   Restrict to one season substring (default: none)
#   TIME_INFO_PATH=path/to/timeline.json   Season timeline (default: the
#                       prepare_audio_parquet.py built-in TBBT Season 1 timeline)
#
# Examples:
#   bash run_prepare_parquet.sh TheBigBangTheory
#   MODE=both SEASON_FILTER=Season01 bash run_prepare_parquet.sh TheBigBangTheory

set -euo pipefail

# ---------------------------------------------------------------------------- #
# Parse arguments
# ---------------------------------------------------------------------------- #
if [[ $# -lt 1 ]]; then
    echo "Usage: bash run_prepare_parquet.sh <DATASETNAME>"
    exit 1
fi

DATASETNAME="$1"
MODE="${MODE:-both}"

DATA_DIR="./Audio_Results/vibevoice/${DATASETNAME}/step2"
OUTPUT_ROOT="outputs/step3_new/${DATASETNAME}"

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: data_dir not found: $DATA_DIR"
    exit 1
fi

echo "==============================================================="
echo "Dataset      : $DATASETNAME"
echo "data_dir     : $DATA_DIR"
echo "output_root  : $OUTPUT_ROOT"
echo "MODE         : $MODE"

echo "==============================================================="

run_prepare() {
    # $1 = extra name-mode flag(s), e.g. "--use_gt_name" or empty for pred
    PYTHONPATH=. python prepare_data/prepare_audio_parquet.py \
        --data_dir "$DATA_DIR" \
        --output_root "$OUTPUT_ROOT"
}

case "$MODE" in
    pred)  run_prepare "" ;;
    gt)    run_prepare "--use_gt_name" ;;
    both)  run_prepare "";  run_prepare "--use_gt_name" ;;
    *)     echo "ERROR: MODE must be pred|gt|both (got '$MODE')"; exit 1 ;;
esac
