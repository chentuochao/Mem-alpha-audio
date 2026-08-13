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
#     DATA=/path/to/in.json OUTPUT_DIR=/path/to/out bash run_tts_multigpu.sh
#     REF_DIR=ref_voices/perltqa_name_replaced bash run_tts_multigpu.sh
#
set -euo pipefail

# ---- config (override via env) --------------------------------------------
NUM_GPUS="${NUM_GPUS:-8}"           # GPUs (worker processes) on THIS node
# Multi-node sharding: the driver partitions blocks by (block_index %
# NUM_SHARDS == shard_index). For a single node leave these at the defaults
# (NUM_SHARDS=NUM_GPUS, SHARD_OFFSET=0). For a multi-node Slurm run, set
# NUM_SHARDS = total GPUs across all nodes and SHARD_OFFSET = this node's first
# global GPU index (e.g. node_rank * NUM_GPUS). See submit_tts_multinode.slurm.
NUM_SHARDS="${NUM_SHARDS:-${NUM_GPUS}}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
CONDA_ENV="${CONDA_ENV:-ctbox}"
LIMIT="${LIMIT:-0}"                 # <=0 = all blocks in the shard
OVERWRITE="${OVERWRITE:-0}"         # 1 = pass --overwrite
DATA="${DATA:-}"                    # input JSON (empty = driver default)
OUTPUT_DIR="${OUTPUT_DIR:-}"        # TTS output root (empty = driver default)
REF_DIR="${REF_DIR:-}"             # reference-voice dir (empty = driver default)
REF_MAP="${REF_MAP:-}"             # reference_voice_map.json (empty = <ref-dir>/...)
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
[[ -n "${DATA}" ]]        && EXTRA_ARGS+=(--data "${DATA}")
[[ -n "${OUTPUT_DIR}" ]]  && EXTRA_ARGS+=(--output-dir "${OUTPUT_DIR}")
[[ -n "${REF_DIR}" ]]     && EXTRA_ARGS+=(--ref-dir "${REF_DIR}")
[[ -n "${REF_MAP}" ]]     && EXTRA_ARGS+=(--ref-map "${REF_MAP}")

# Sanity: this node's global shard range must fit inside the total shard count.
if (( SHARD_OFFSET + NUM_GPUS > NUM_SHARDS )); then
    echo "[launch][ERROR] SHARD_OFFSET(${SHARD_OFFSET}) + NUM_GPUS(${NUM_GPUS}) "\
         "> NUM_SHARDS(${NUM_SHARDS}); shard indices would overflow." >&2
    exit 1
fi

echo "[launch] NUM_GPUS=${NUM_GPUS} num_shards=${NUM_SHARDS} offset=${SHARD_OFFSET} "\
     "env=${CONDA_ENV} limit=${LIMIT} overwrite=${OVERWRITE} "\
     "data=${DATA:-<default>} out=${OUTPUT_DIR:-<default>} logs=${LOG_DIR}"

declare -a PIDS=()
for (( k=0; k<NUM_GPUS; k++ )); do
    gshard=$(( SHARD_OFFSET + k ))          # global shard index across all nodes
    log="${LOG_DIR}/shard_${gshard}.log"    # global index -> unique log per node
    echo "[launch] GPU ${k} -> shard ${gshard}/${NUM_SHARDS}  (log: ${log})"
    CUDA_VISIBLE_DEVICES="${k}" \
        conda run --no-capture-output -n "${CONDA_ENV}" \
        python "${HERE}/perltqa_dialogue_tts.py" \
            --skip-prepare \
            --num-shards "${NUM_SHARDS}" \
            --shard-index "${gshard}" \
            --limit "${LIMIT}" \
            "${EXTRA_ARGS[@]}" \
        > "${log}" 2>&1 &
    PIDS+=("$!")
done

echo "[launch] ${#PIDS[@]} workers running. Tail a shard with:"
echo "    tail -f ${LOG_DIR}/shard_${SHARD_OFFSET}.log"

# ---- wait for all workers, report failures --------------------------------
fail=0
for (( k=0; k<NUM_GPUS; k++ )); do
    gshard=$(( SHARD_OFFSET + k ))
    if wait "${PIDS[$k]}"; then
        echo "[launch] shard ${gshard} finished OK"
    else
        rc=$?
        echo "[launch][ERROR] shard ${gshard} exited with code ${rc} "\
             "(see ${LOG_DIR}/shard_${gshard}.log)" >&2
        fail=1
    fi
done

if [[ "${fail}" == "1" ]]; then
    echo "[launch] one or more shards failed." >&2
    exit 1
fi
echo "[launch] all ${NUM_GPUS} shards completed."
