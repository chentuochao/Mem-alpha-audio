#!/usr/bin/env bash
# =============================================================================
# submit_sweep.sh -- submit one `submit_pipeline.slurm` job per compression
# ratio, with all arguments read from a single sourced config file.
#
# Usage:
#   bash submit_sweep.sh <config_file>
#
# The config is a plain bash file that is `source`d (see sweep_configs/
# example.conf). It can use either of these formats:
#
#   1. One input: set PARQUET_PATH and COMPRESSION_RATIOS.
#   2. Multiple inputs/SNRs: set SWEEP_CASES, with one pipe-delimited entry per
#      input in the form:
#        name|parquet_path|dataset|qa_dir|ratio1 ratio2 ...
#
# DATASET / CUSTOM_QA_DIR and the VLLM_* / MEM_ENV overrides are optional for
# the one-input format. Multi-input cases specify dataset and QA dir explicitly.
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

multi_input=0
if [[ -n "${SWEEP_CASES+x}" && ${#SWEEP_CASES[@]} -gt 0 ]]; then
    multi_input=1
else
    : "${PARQUET_PATH:?PARQUET_PATH not set in $CONFIG}"
    DATASET="${DATASET:-seamlessinteraction_options}"
    CUSTOM_QA_DIR="${CUSTOM_QA_DIR:-outputs/step3_anony/qas/}"

    if [[ -z "${COMPRESSION_RATIOS+x}" || ${#COMPRESSION_RATIOS[@]} -eq 0 ]]; then
        echo "ERROR: COMPRESSION_RATIOS array is empty/unset in $CONFIG" >&2
        exit 1
    fi
fi

# Optional seed sweep. Empty (or the literal "none"/"None") -> a single ""
# entry meaning "no seed override"; run_pipeline.sh then omits --seed, which
# argparse would reject as a non-integer.
if [[ -z "${SEEDS+x}" || ${#SEEDS[@]} -eq 0 ]]; then
    SEEDS=("")
fi
for i in "${!SEEDS[@]}"; do
    [[ "${SEEDS[$i]}" == "none" || "${SEEDS[$i]}" == "None" ]] && SEEDS[$i]=""
done
MEM_TEMPERATURE="${MEM_TEMPERATURE:-}"
DRY_RUN="${DRY_RUN:-0}"

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

# Optional passthrough env vars -> put into sbatch's own environment (see the
# note near the sbatch call: do NOT use `--export=ALL,VAR=...` on this cluster).
# ANON_SPEAKER=true selects the anonymized-speaker pipeline in submit_pipeline.slurm.
BASE_ENV=()
# A temperature set in the config applies even without a seed sweep.
[[ -n "$MEM_TEMPERATURE" ]] && BASE_ENV+=("MEM_TEMPERATURE=${MEM_TEMPERATURE}")
for v in VLLM_SCRIPT VLLM_ENV MEM_ENV ANON_SPEAKER FORCE_REANSWER; do
    if [[ -n "${!v:-}" ]]; then
        BASE_ENV+=("${v}=${!v}")
    fi
done

n_jobs=0

submit_case() {
    local case_name="$1"
    local parquet_path="$2"
    local dataset="$3"
    local qa_dir="$4"
    shift 4
    local ratios=("$@")

    if [[ -z "$case_name" || -z "$parquet_path" || -z "$dataset" || -z "$qa_dir" || ${#ratios[@]} -eq 0 ]]; then
        echo "ERROR: invalid sweep case '$case_name' in $CONFIG" >&2
        exit 1
    fi

    echo "================================================================"
    echo "case          : $case_name"
    echo "parquet       : $parquet_path"
    echo "dataset       : $dataset"
    echo "qa_dir        : $qa_dir"
    echo "ratios        : ${ratios[*]}"
    echo "seeds         : ${SEEDS[*]:-<none>}"
    [[ $sweeping_seeds -eq 1 ]] && echo "temperature   : $MEM_TEMPERATURE"
    [[ ${#BASE_ENV[@]} -gt 0 ]] && echo "extra env     : ${BASE_ENV[*]}"
    echo "================================================================"

    local c s strat clabel
    for c in "${ratios[@]}"; do
        if [[ "$c" == "none" || "$c" == "None" || -z "$c" ]]; then
            strat=""
            clabel="baseline"
        else
            strat="$c"
            clabel="$c"
        fi

        for s in "${SEEDS[@]}"; do
            JOB_ENV=("COMPRESSION_STRATEGY=${strat}" "${BASE_ENV[@]}")
            if [[ -n "$s" ]]; then
                JOB_ENV+=("SEED=${s}" "ROLLOUT_LABEL=seed${s}")
                echo ">> submitting case=${case_name} compression=${clabel} seed=${s}"
            else
                echo ">> submitting case=${case_name} compression=${clabel}"
            fi
            # NOTE: pass job vars through sbatch's OWN environment, which Slurm
            # propagates with the default --export=ALL. Using an explicit
            # `--export=ALL,VAR=value` makes slurmd try to reconstruct the login
            # environment on the compute node (`su - $USER`); that retrieval fails
            # on this cluster and the job is requeued and held with
            # "user_env_retrieval_failed_requeued_held".
            if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" ]]; then
                printf '   DRY RUN:'
                printf ' %q' env "${JOB_ENV[@]}" sbatch submit_pipeline.slurm \
                    "$parquet_path" "$dataset" "$qa_dir"
                printf '\n'
            else
                env "${JOB_ENV[@]}" \
                    sbatch submit_pipeline.slurm "$parquet_path" "$dataset" "$qa_dir"
            fi
            n_jobs=$((n_jobs + 1))
        done
    done
}

# --- fan out: one sbatch per (input/SNR, ratio, seed) ----------------------
echo "sweep config : $CONFIG"
if [[ $multi_input -eq 1 ]]; then
    for case_spec in "${SWEEP_CASES[@]}"; do
        IFS='|' read -r case_name case_parquet case_dataset case_qa case_ratios extra <<< "$case_spec"
        if [[ -n "${extra:-}" || -z "${case_ratios:-}" ]]; then
            echo "ERROR: malformed SWEEP_CASES entry (expected 5 pipe-delimited fields):" >&2
            echo "       $case_spec" >&2
            exit 1
        fi
        read -r -a case_ratio_array <<< "$case_ratios"
        submit_case "$case_name" "$case_parquet" "$case_dataset" "$case_qa" "${case_ratio_array[@]}"
    done
else
    submit_case "default" "$PARQUET_PATH" "$DATASET" "$CUSTOM_QA_DIR" "${COMPRESSION_RATIOS[@]}"
fi

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" ]]; then
    echo ">> dry run complete: ${n_jobs} job(s) would be submitted"
else
    echo ">> submitted ${n_jobs} job(s). Track with: squeue --me"
fi
