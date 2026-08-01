#!/usr/bin/env bash
#
# Multi-GPU launcher for PerLTQA dialogue TTS.
#
# Chatterbox loads its model on the first *visible* CUDA device at import time,
# so we parallelize by launching one process per GPU, each pinned with
# CUDA_VISIBLE_DEVICES and handed a disjoint shard of dialogue blocks
# (block_index % NUM_GPUS == shard_index). Blocks write to per-block output
# dirs, so shards never collide, and re-runs resume via the channel_map.json
# done-marker.
#
# The reference voice bank must be built ONCE before launching (otherwise all
# workers race on reference_voice_map.json). Do that with either:
#     python select_and_check_voices.py select          # embedding-aware
#     # or:  python perltqa_dialogue_tts.py --prepare-only
#
# Usage:
#     bash run_tts_multigpu.sh                 # 8 GPUs, all blocks
#     NUM_GPUS=4 bash run_tts_multigpu.sh      # 4 GPUs
#     LIMIT=10 bash run_tts_multigpu.sh        # first 10 blocks *per shard* (smoke test)
#     OVERWRITE=1 bash run_tts_multigpu.sh     # re-generate existing blocks
#
set -euo pipefail

# ---- config (override via env) --------------------------------------------
NUM_GPUS="${NUM_GPUS:-8}"
CONDA_ENV="${CONDA_ENV:-ctbox}"
LIMIT="${LIMIT:-0}"                 # <=0 = all blocks in the shard
OVERWRITE="${OVERWRITE:-0}"         # 1 = pass --overwrite
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-${HERE}/logs_tts_multigpu}"
# Keep CPU threads modest so N processes don't oversubscribe the node.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Chatterbox's generate() prints a per-turn tqdm "Sampling" bar to stderr, which
# floods the redirected logs (~1000 updates/turn). Silence it by default; the
# per-dialogue START/DONE lines give liveness + timing instead. Set SHOW_TQDM=1
# to keep the bar.
if [[ "${SHOW_TQDM:-0}" != "1" ]]; then
    export TQDM_DISABLE=1
fi

mkdir -p "${LOG_DIR}"

EXTRA_ARGS=()
[[ "${OVERWRITE}" == "1" ]] && EXTRA_ARGS+=(--overwrite)

echo "[launch] NUM_GPUS=${NUM_GPUS} env=${CONDA_ENV} limit=${LIMIT} "\
     "overwrite=${OVERWRITE} logs=${LOG_DIR}"

declare -a PIDS=()
for (( k=0; k<NUM_GPUS; k++ )); do
    log="${LOG_DIR}/shard_${k}.log"
    echo "[launch] GPU ${k} -> shard ${k}/${NUM_GPUS}  (log: ${log})"
    CUDA_VISIBLE_DEVICES="${k}" \
        conda run --no-capture-output -n "${CONDA_ENV}" \
        python "${HERE}/perltqa_dialogue_tts.py" \
            --skip-prepare \
            --num-shards "${NUM_GPUS}" \
            --shard-index "${k}" \
            --limit "${LIMIT}" \
            "${EXTRA_ARGS[@]}" \
        > "${log}" 2>&1 &
    PIDS+=("$!")
done

echo "[launch] ${#PIDS[@]} workers running. Tail a shard with:"
echo "    tail -f ${LOG_DIR}/shard_0.log"

# ---- wait for all workers, report failures --------------------------------
fail=0
for (( k=0; k<NUM_GPUS; k++ )); do
    if wait "${PIDS[$k]}"; then
        echo "[launch] shard ${k} finished OK"
    else
        rc=$?
        echo "[launch][ERROR] shard ${k} exited with code ${rc} "\
             "(see ${LOG_DIR}/shard_${k}.log)" >&2
        fail=1
    fi
done

if [[ "${fail}" == "1" ]]; then
    echo "[launch] one or more shards failed." >&2
    exit 1
fi
echo "[launch] all ${NUM_GPUS} shards completed."
