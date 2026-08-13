#!/usr/bin/env bash
# =============================================================================
# SLURM submit script (fair-sc): bring up the reward-model vLLM server +
# memory_server, then run the full memory/QA pipeline, then tear everything
# down -- all inside ONE job, no second terminal.
#
# GPU layout (2-GPU allocation): server on GPU 1, memory agent on GPU 0.
#
# Submit:
#   sbatch submit_pipeline.slurm <parquet_path> [dataset] [custom_qa_dir] [compression]
#
# Compression may be given either as the 4th positional arg or via the
# COMPRESSION_STRATEGY env var (positional wins if both are set):
#   sbatch submit_pipeline.slurm outputs/bazinga_data/Season1_gt.parquet \
#          seamlessinteraction_options outputs/step3_anony/qas/ x3
#   sbatch --export=ALL,COMPRESSION_STRATEGY=x3 \
#          submit_pipeline.slurm outputs/bazinga_data/Season1_gt.parquet
#
# Override the reward model (default: Qwen3.6-27B launcher):
#   sbatch --export=ALL,VLLM_SCRIPT=./launch_vllm.sh submit_pipeline.slurm <parquet>
#
# Seed-variance run (forwarded to run_pipeline.sh via --export=ALL):
#   SEED            seed for step 1 memory construction
#   MEM_TEMPERATURE sampling temperature (must be > 0 for seeds to diverge;
#                   the default config decodes greedily and ignores the seed)
#   ROLLOUT_LABEL   output-folder postfix (_rollout_<label>) so seeds don't clash
#   sbatch --export=ALL,SEED=1,MEM_TEMPERATURE=0.7,ROLLOUT_LABEL=seed1 \
#          submit_pipeline.slurm outputs/bazinga_data/Season1_gt.parquet
# =============================================================================
#SBATCH --job-name=mem-pipeline
#SBATCH --account=seamless
#SBATCH --qos=h200_seamless_high
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# --- inputs ----------------------------------------------------------------
PARQUET_PATH="${1:?Usage: sbatch submit_pipeline.slurm <parquet_path> [dataset] [custom_qa_dir] [compression]}"
DATASET="${2:-seamlessinteraction_options}"
CUSTOM_QA_DIR="${3:-outputs/step3_anony/qas/}"
# Compression: 4th positional arg wins; otherwise fall back to the env var.
# Empty -> run_pipeline omits the flag (Python default 'default', no _comp_ postfix).
export COMPRESSION_STRATEGY="${4:-${COMPRESSION_STRATEGY:-}}"

# Which vLLM launcher hosts the reward model (Qwen3.6-27B by default here).
export VLLM_SCRIPT="${VLLM_SCRIPT:-./launch_vllm.sh}"
export VLLM_ENV="${VLLM_ENV:-vllm}"   # env with vllm + CUDA
export MEM_ENV="${MEM_ENV:-mem}"      # env with flask (memory_server + QA)

# --- per-job server ports/files (avoid collisions when jobs co-locate) ------
# Multiple sweep jobs can land on the SAME node. With fixed ports (8002/5005)
# and a shared .server_pids, the first job to finish tore down the servers the
# other job was still using (RemoteDisconnected at step 2). Give each job its
# own ports, PID file, and logs so co-located jobs are fully isolated.
find_free_port() {  # <start_port> -> first free TCP port >= start_port
    local p="$1"
    while lsof -ti "tcp:${p}" >/dev/null 2>&1; do
        p=$((p + 1))
    done
    echo "$p"
}
JOBTAG="${SLURM_JOB_ID:-$$}"
# IMPORTANT: do NOT `export` VLLM_PORT. It is a reserved vLLM env var (base for
# vLLM's internal engine/distributed port). If it lived in the job environment,
# step 1's in-process vLLM (run_memory_construction) would use it as its engine
# base and co-located jobs would collide. Keep these as plain shell vars and
# pass them to launch_servers.sh inline; the vLLM API port is set via --port.
VLLM_PORT="$(find_free_port $((8000 + JOBTAG % 900)))"
MEM_PORT="$(find_free_port $((5000 + JOBTAG % 900)))"
PID_FILE=".server_pids.${JOBTAG}"
VLLM_LOG="logs/vllm_server-${JOBTAG}.log"
MEM_LOG="logs/memory_server-${JOBTAG}.log"
# Point step 2 QA at THIS job's memory_server (overrides the config YAML port).
# Safe to export: not a vLLM-recognized variable.
export EXTERNAL_MODEL_URL="http://127.0.0.1:${MEM_PORT}/batch_process"

echo "================================================================"
echo "job=${SLURM_JOB_ID:-?}  node=$(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-?}"
echo "parquet=$PARQUET_PATH  dataset=$DATASET  qa_dir=$CUSTOM_QA_DIR"
echo "compression=${COMPRESSION_STRATEGY:-None}  vllm_script=$VLLM_SCRIPT"
echo "seed=${SEED:-None}  temperature=${MEM_TEMPERATURE:-None}  rollout_label=${ROLLOUT_LABEL:-None}"
echo "vllm_port=$VLLM_PORT  mem_port=$MEM_PORT  pid_file=$PID_FILE"
echo "================================================================"

# --- 1. start servers, guarantee teardown on any exit ----------------------
# Server on GPU 1; wait until both /v1/models and /health are live.
# Pass the per-job server vars INLINE (not exported) so they reach launch_servers
# but never leak VLLM_PORT into step 1's in-process vLLM.
SERVER_ENV=(VLLM_PORT="$VLLM_PORT" MEM_PORT="$MEM_PORT" PID_FILE="$PID_FILE"
            VLLM_LOG="$VLLM_LOG" MEM_LOG="$MEM_LOG")
env "${SERVER_ENV[@]}" SERVER_GPU=1 ./launch_servers.sh start

cleanup() {
    echo ">> tearing down servers"
    env "${SERVER_ENV[@]}" ./launch_servers.sh stop || true
}
trap cleanup EXIT

# --- 2. run the pipeline; memory agent pinned to GPU 0 ---------------------
# run_pipeline*.sh activates its own conda envs (vllm for step1, mem for step2)
# and points step2 at memory_server via the agent config's external_model_url.
# ANON_SPEAKER=true selects the anonymized-speaker variant (registry-in-core
# prompt + post-hoc name substitution); anything else uses the normal pipeline.
case "${ANON_SPEAKER:-}" in
    true|True|TRUE|1|yes) PIPELINE_SCRIPT="run_pipeline_speaker_name.sh" ;;
    *)                    PIPELINE_SCRIPT="run_pipeline.sh" ;;
esac
echo "pipeline=${PIPELINE_SCRIPT}  anon_speaker=${ANON_SPEAKER:-false}"
CUDA_VISIBLE_DEVICES=0 \
QWEN_URL="http://localhost:${VLLM_PORT}/v1" \
QWEN_MODEL_NAME="qwen3-32b" \
    bash "$PIPELINE_SCRIPT" "$PARQUET_PATH" "$DATASET" "$CUSTOM_QA_DIR"

echo "================================================================"
echo "[DONE] pipeline finished; servers will be stopped by the exit trap."
echo "================================================================"
