#!/usr/bin/env bash
# =============================================================================
# Multi-node PerLTQA dialogue-TTS on 32 GPUs (32 nodes x 1 GPU).
#
# One Slurm task per node; each task runs run_tts_multigpu.sh, which forks
# GPUS_PER_NODE worker processes (here 1, on the node's single GPU). Global
# block sharding is coordinated across nodes via:
#     NUM_SHARDS   = SLURM_NNODES * GPUS_PER_NODE   (total workers, e.g. 32)
#     SHARD_OFFSET = SLURM_NODEID * GPUS_PER_NODE   (this node's first shard)
# so every one of the 32 workers gets a disjoint 1/32 slice of dialogue blocks.
# Blocks write to per-block dirs, so workers never collide; re-runs resume via
# each block's channel_map.json done-marker.
#
# PREREQUISITE (run ONCE before submitting): build the reference-voice map for
# the new names, otherwise every worker fails on a missing speaker:
#     python ctbox_tts/select_and_check_voices.py select \
#       --data    "$DATA" \
#       --ref-dir ctbox_tts/ref_voices/perltqa_name_replaced \
#       --cache   ctbox_tts/ref_voices/perltqa/libritts_wespeaker_cache.npz
#
# Submit (paths can be overridden via --export):
#     sbatch ctbox_tts/submit_tts_multinode.slurm
#     sbatch --export=ALL,DATA=/path/in.json,OUTPUT_DIR=/path/out \
#            ctbox_tts/submit_tts_multinode.slurm
# =============================================================================
#SBATCH --job-name=perltqa-tts
#SBATCH --account=seamless
#SBATCH --qos=h200_seamless_high
#SBATCH --partition=h200
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out

set -euo pipefail

REPO="/storage/home/tuochao/Mem-alpha-audio"
GPUS_PER_NODE=1

# ---- run parameters (override via --export=ALL,VAR=...) --------------------
export DATA="${DATA:-/checkpoint/seamless/tuochao/data/PerLTQA/new_name_replace_perltqa/new_perltmem_en_name_replaced.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-/checkpoint/seamless/tuochao/data/PerLTQA/dialogue_tts_en_name_replaced}"
export REF_DIR="${REF_DIR:-ref_voices/perltqa_name_replaced}"
export OVERWRITE="${OVERWRITE:-0}"
export LIMIT="${LIMIT:-0}"

# Total workers across all nodes; each node's launcher fans out GPUS_PER_NODE.
export NUM_GPUS="${GPUS_PER_NODE}"
TOTAL_SHARDS=$(( SLURM_NNODES * GPUS_PER_NODE ))

# Make conda available to the (non-interactive) launcher, which uses `conda run`.
export PATH="$HOME/miniconda3/bin:$HOME/miniconda3/condabin:$PATH"

mkdir -p "${REPO}/logs"
echo "[slurm] job=${SLURM_JOB_ID} nodes=${SLURM_NNODES} gpus/node=${GPUS_PER_NODE} "\
     "total_shards=${TOTAL_SHARDS}"
echo "[slurm] DATA=${DATA}"
echo "[slurm] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[slurm] nodelist=${SLURM_JOB_NODELIST}"

# One task per node. Each computes its own SHARD_OFFSET from SLURM_NODEID and
# runs the (unchanged-interface) multi-GPU launcher.
srun --ntasks-per-node=1 --export=ALL,TOTAL_SHARDS="${TOTAL_SHARDS}",GPUS_PER_NODE="${GPUS_PER_NODE}" \
    bash -c '
        set -euo pipefail
        export NUM_SHARDS="${TOTAL_SHARDS}"
        export SHARD_OFFSET=$(( SLURM_NODEID * GPUS_PER_NODE ))
        # Per-node log dir so shard logs from different nodes never overwrite.
        export LOG_DIR="'"${REPO}"'/ctbox_tts/logs_tts_multigpu/job_${SLURM_JOB_ID}"
        echo "[node ${SLURM_NODEID} @ $(hostname)] shards ${SHARD_OFFSET}..$(( SHARD_OFFSET + GPUS_PER_NODE - 1 )) of ${NUM_SHARDS}"
        cd "'"${REPO}"'/ctbox_tts"
        bash run_tts_multigpu.sh
    '

echo "[slurm] all nodes finished."
