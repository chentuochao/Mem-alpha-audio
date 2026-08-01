#!/usr/bin/env bash
# One-terminal launcher for the QA/probe backend: reward-model vLLM server
# (port 8002) + memory_server.py (port 5005).
#
# Replaces the two-terminal dance (srun --overlap / ssh to a second shell just
# to run ./launch_vllm.sh in the foreground). Here both servers are started in
# the BACKGROUND (each in its own process group + conda env), health-checked,
# and then the terminal is handed back so you can run the pipeline / probe in
# the SAME shell.
#
# The two servers need DIFFERENT conda envs on fair-sc:
#   * vLLM API server  -> 'vllm' env (has vllm + CUDA; NO flask)
#   * memory_server.py -> 'mem'  env (has flask; vllm._C can't load CUDA)
#
# Usage:
#   ./launch_servers.sh              # start both, wait until healthy, return
#   ./launch_servers.sh stop         # kill both process groups (via .server_pids)
#   ./launch_servers.sh status       # curl both health endpoints
#
# Knobs (env vars):
#   VLLM_SCRIPT   which vLLM launcher to run   (default ./launch_vllm.sh)
#   SERVER_GPU    GPU for the reward model     (default 1) -- keep OFF the GPU
#                 running the memory agent (e.g. agent on GPU0 -> server GPU1)
#   VLLM_PORT     OpenAI server port           (default 8002)
#   MEM_PORT      memory_server port           (default 5005)
#   VLLM_ENV      conda env for the vLLM server   (default vllm; empty = current)
#   MEM_ENV       conda env for memory_server.py  (default mem;  empty = current)
#   QWEN_MODEL_NAME  model id memory_server sends (default qwen3-32b, matching
#                    the --served-model-name in the vLLM launchers)
#   STARTUP_TIMEOUT  seconds to wait for vLLM weights to load (default 900)
set -euo pipefail

cd "$(dirname "$0")"

VLLM_SCRIPT="${VLLM_SCRIPT:-./launch_vllm.sh}"
SERVER_GPU="${SERVER_GPU:-1}"
VLLM_PORT="${VLLM_PORT:-8002}"
MEM_PORT="${MEM_PORT:-5005}"
VLLM_ENV="${VLLM_ENV:-vllm}"
MEM_ENV="${MEM_ENV:-mem}"
QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-qwen3-32b}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"

QWEN_URL="http://localhost:${VLLM_PORT}/v1"
# PID_FILE and the server logs are overridable so concurrent jobs on one node
# (e.g. a seed/compression sweep) don't clobber each other's tracking/logs.
PID_FILE="${PID_FILE:-.server_pids}"
mkdir -p logs
VLLM_LOG="${VLLM_LOG:-logs/vllm_server.log}"
MEM_LOG="${MEM_LOG:-logs/memory_server.log}"

# Locate conda so we can run each server in its own env inside a batch job.
_source_conda() {
    if [[ -n "${CONDA_SH:-}" && -f "$CONDA_SH" ]]; then source "$CONDA_SH"; return; fi
    for c in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
        [[ -f "$c" ]] && { source "$c"; return; }
    done
    [[ -n "${CONDA_EXE:-}" ]] && source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
}

# Run <cmd...> in its own process group (setsid), optionally after activating a
# conda env. Echoes the leader PID (== PGID) so the caller can kill the group.
spawn_in_env() {  # <env> <logfile> <cmd...>
    local env="$1" log="$2"; shift 2
    local prelude=""
    if [[ -n "$env" ]]; then
        prelude="source \"$(_conda_sh_path)\"; conda activate \"$env\"; "
    fi
    setsid bash -c "${prelude}exec \"\$@\"" _ "$@" > "$log" 2>&1 &
    echo $!
}

_conda_sh_path() {
    for c in "${CONDA_SH:-}" "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
        [[ -n "$c" && -f "$c" ]] && { echo "$c"; return; }
    done
    [[ -n "${CONDA_EXE:-}" ]] && echo "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
}

# Free a TCP port by killing whoever listens on it. Targets by PORT (not by
# command-string match), so it can never self-match the killing shell and it
# catches orphans that are missing from $PID_FILE. TERM first, then KILL.
kill_port() {  # <port> <label>
    local port="$1" label="$2" pids
    pids=$(lsof -ti "tcp:${port}" 2>/dev/null || true)
    [[ -z "$pids" ]] && pids=$(fuser "${port}/tcp" 2>/dev/null || true)
    pids=$(echo $pids)  # normalize whitespace
    if [[ -z "$pids" ]]; then
        echo "  port $port ($label): already free"
        return
    fi
    echo "  port $port ($label): TERM $pids"
    kill -TERM $pids 2>/dev/null || true
    sleep 5
    local still; still=$(lsof -ti "tcp:${port}" 2>/dev/null || true)
    if [[ -n "$still" ]]; then
        echo "  port $port ($label): SIGKILL $still"
        kill -KILL $still 2>/dev/null || true
    fi
}

wait_for() {  # <url> <name> <timeout>
    local url="$1" name="$2" timeout="$3" waited=0
    printf 'Waiting for %s at %s ' "$name" "$url"
    while ! curl -sf "$url" >/dev/null 2>&1; do
        if (( waited >= timeout )); then
            printf '\n[FAIL] %s not ready after %ss. Check the log.\n' "$name" "$timeout" >&2
            return 1
        fi
        sleep 3; waited=$((waited + 3)); printf '.'
    done
    printf ' ready (%ss)\n' "$waited"
}

case "${1:-start}" in
  stop)
    # 1) kill tracked process groups (server + all children) with TERM->KILL.
    #    Runs only on THIS node; for a server on another node, run this on that
    #    node, e.g.  srun --overlap --jobid <JOBID> bash -c './launch_servers.sh stop'
    if [[ -f "$PID_FILE" ]]; then
        mapfile -t PGIDS < "$PID_FILE"
        for pgid in "${PGIDS[@]}"; do
            [[ -n "$pgid" ]] || continue
            kill -TERM "-$pgid" 2>/dev/null && echo "TERM group $pgid" || true
        done
        sleep 5
        for pgid in "${PGIDS[@]}"; do
            [[ -n "$pgid" ]] || continue
            kill -KILL "-$pgid" 2>/dev/null || true
        done
        rm -f "$PID_FILE"
    else
        echo "no $PID_FILE (stale/absent) -> port-based cleanup only"
    fi
    # 2) safety net: free the ports regardless, catching orphans not tracked in
    #    $PID_FILE (e.g. servers started before this script, or from a crash).
    kill_port "$VLLM_PORT" vLLM
    kill_port "$MEM_PORT" memory_server
    echo "stop complete."
    exit 0
    ;;
  status)
    echo "--- vLLM ($QWEN_URL/models) ---";        curl -s "$QWEN_URL/models" || echo "(down)"; echo
    echo "--- memory_server (:$MEM_PORT/health) ---"; curl -s "http://127.0.0.1:${MEM_PORT}/health" || echo "(down)"; echo
    exit 0
    ;;
  start) ;;
  *) echo "usage: $0 [start|stop|status]" >&2; exit 2 ;;
esac

# --- 1. reward-model vLLM server -------------------------------------------
echo ">> starting vLLM ($VLLM_SCRIPT) env=$VLLM_ENV GPU=$SERVER_GPU port=$VLLM_PORT -> $VLLM_LOG"
# CUDA_VISIBLE_DEVICES / PORT are honored by launch_vllm_qwen36.sh; the older
# launch_vllm.sh hardcodes GPU1/port 8002 (its documented default).
export CUDA_VISIBLE_DEVICES="$SERVER_GPU" PORT="$VLLM_PORT"
# VLLM_PORT is a RESERVED vLLM env var: vLLM uses it as the base for its INTERNAL
# EngineCore / torch.distributed rendezvous port. The API port is set via --port
# ($PORT); if vLLM also inherits VLLM_PORT it walks upward from there and, when
# several jobs are packed on one node with adjacent ports, collides on a
# neighbour's port (EADDRINUSE). Strip it so vLLM auto-picks a free engine port.
# (QWEN_URL was already built from the port above, so nothing else needs it.)
unset VLLM_PORT
VLLM_PGID=$(spawn_in_env "$VLLM_ENV" "$VLLM_LOG" bash "$VLLM_SCRIPT")
echo "$VLLM_PGID" > "$PID_FILE"
unset CUDA_VISIBLE_DEVICES PORT

wait_for "$QWEN_URL/models" "vLLM reward model" "$STARTUP_TIMEOUT"

# --- 2. memory_server (CPU-only Flask client) ------------------------------
echo ">> starting memory_server env=$MEM_ENV port=$MEM_PORT (QWEN_MODEL_NAME=$QWEN_MODEL_NAME) -> $MEM_LOG"
export QWEN_URL QWEN_MODEL_NAME CUDA_VISIBLE_DEVICES=""
MEM_PGID=$(spawn_in_env "$MEM_ENV" "$MEM_LOG" python memory_server.py --port "$MEM_PORT")
echo "$MEM_PGID" >> "$PID_FILE"
unset CUDA_VISIBLE_DEVICES

wait_for "http://127.0.0.1:${MEM_PORT}/health" "memory_server" 120

echo
echo "Both servers up.  Process groups (vLLM $VLLM_PGID, memory_server $MEM_PGID) in $PID_FILE"
echo "  logs:  tail -f $VLLM_LOG   |   tail -f $MEM_LOG"
echo "  stop:  ./launch_servers.sh stop"
echo "You can now run run_pipeline.sh / the probe in THIS terminal."
