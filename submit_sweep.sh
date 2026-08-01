#!/usr/bin/env bash
# =============================================================================
# submit_sweep.sh -- submit one `submit_pipeline.slurm` job per compression
# ratio, with all arguments read from a single sourced config file.
#
# Usage:
#   bash submit_sweep.sh <config_file>
#
# The config is a plain bash file that is `source`d (see sweep_configs/
# example.conf). It must set PARQUET_PATH and a COMPRESSION_RATIOS array;
# DATASET / CUSTOM_QA_DIR and the VLLM_* / MEM_ENV overrides are optional.
#
# Each ratio becomes an independent 2-GPU job. Use the literal "none" (or an
# empty string) in COMPRESSION_RATIOS to run the baseline with no compression
# flag (no _comp_ postfix on the output folder).
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

CONFIG="${1:?Usage: bash submit_sweep.sh <config_file>}"
[[ -f "$CONFIG" ]] || { echo "ERROR: config file not found: $CONFIG" >&2; exit 1; }

# --- load config -----------------------------------------------------------
# shellcheck disable=SC1090
source "$CONFIG"

: "${PARQUET_PATH:?PARQUET_PATH not set in $CONFIG}"
DATASET="${DATASET:-seamlessinteraction_options}"
CUSTOM_QA_DIR="${CUSTOM_QA_DIR:-outputs/step3_anony/qas/}"

if [[ -z "${COMPRESSION_RATIOS+x}" || ${#COMPRESSION_RATIOS[@]} -eq 0 ]]; then
    echo "ERROR: COMPRESSION_RATIOS array is empty/unset in $CONFIG" >&2
    exit 1
fi

# Optional seed sweep. Empty -> a single "" entry meaning "no seed override".
if [[ -z "${SEEDS+x}" || ${#SEEDS[@]} -eq 0 ]]; then
    SEEDS=("")
fi
MEM_TEMPERATURE="${MEM_TEMPERATURE:-}"

# A real seed sweep needs temperature > 0, otherwise greedy decoding makes every
# seed produce identical memory (see run_memory_construction_new.py).
sweeping_seeds=0
for s in "${SEEDS[@]}"; do [[ -n "$s" ]] && sweeping_seeds=1; done
if [[ $sweeping_seeds -eq 1 ]]; then
    if [[ -z "$MEM_TEMPERATURE" ]] || awk "BEGIN{exit !($MEM_TEMPERATURE <= 0)}"; then
        echo "ERROR: SEEDS sweep requested but MEM_TEMPERATURE is not > 0 in $CONFIG." >&2
        echo "       Greedy decoding ignores the seed; set e.g. MEM_TEMPERATURE=\"0.7\"." >&2
        exit 1
    fi
fi

# Optional passthrough env vars -> forwarded to sbatch --export if set.
EXPORT_EXTRA=""
for v in VLLM_SCRIPT VLLM_ENV MEM_ENV; do
    if [[ -n "${!v:-}" ]]; then
        EXPORT_EXTRA+=",${v}=${!v}"
    fi
done

echo "================================================================"
echo "sweep config : $CONFIG"
echo "parquet      : $PARQUET_PATH"
echo "dataset      : $DATASET"
echo "qa_dir       : $CUSTOM_QA_DIR"
echo "ratios       : ${COMPRESSION_RATIOS[*]}"
echo "seeds        : ${SEEDS[*]:-<none>}"
[[ $sweeping_seeds -eq 1 ]] && echo "temperature  : $MEM_TEMPERATURE"
[[ -n "$EXPORT_EXTRA" ]] && echo "extra export :${EXPORT_EXTRA}"
echo "================================================================"

# --- fan out: one sbatch per (ratio, seed) ---------------------------------
n_jobs=0
for c in "${COMPRESSION_RATIOS[@]}"; do
    if [[ "$c" == "none" || "$c" == "None" || -z "$c" ]]; then
        STRAT=""
        clabel="baseline"
    else
        STRAT="$c"
        clabel="$c"
    fi

    for s in "${SEEDS[@]}"; do
        EXPORT="ALL,COMPRESSION_STRATEGY=${STRAT}${EXPORT_EXTRA}"
        if [[ -n "$s" ]]; then
            EXPORT+=",SEED=${s},MEM_TEMPERATURE=${MEM_TEMPERATURE},ROLLOUT_LABEL=seed${s}"
            echo ">> submitting compression=${clabel} seed=${s}"
        else
            echo ">> submitting compression=${clabel}"
        fi
        sbatch --export="$EXPORT" \
            submit_pipeline.slurm "$PARQUET_PATH" "$DATASET" "$CUSTOM_QA_DIR"
        n_jobs=$((n_jobs + 1))
    done
done

echo ">> submitted ${n_jobs} job(s). Track with: squeue --me"
