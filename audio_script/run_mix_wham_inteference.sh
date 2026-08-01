#!/usr/bin/env bash
#
# Mix competing-speech INTERFERENCE into the Bazinga audio at SNR levels.
# Interference is drawn from the sdialog/voices-libritts HuggingFace voice bank:
# per episode, 1-3 interference tracks are built by concatenating randomly
# sampled clips (with a random [0, gap_max]s silence gap between them) and mixed
# into the target track at the requested SNR.
#
# Produces sibling folders next to DATA_PATH:
#   <DATA_PATH>_interf_SNR10 / _interf_SNR5 / _interf_SNR0   (noisy .en.wav + verbatim .txt)
#
# Then point run_demo_step1_bazinga.sh's DATA_PATH at one of those folders.
#
set -euo pipefail

DATA_PATH="/checkpoint/seamless/tuochao/data/bazinga/data/TheBigBangTheory"
SNRS=(5 10 15)
POOL_MINUTES=30
NUM_INTERF_MIN=1
NUM_INTERF_MAX=4
GAP_MAX=3.0
SEED=0
# Only mix episodes whose id contains one of these substrings (empty = all).
SEASON_FILTER=("Season01" "Season02" "Season03")
PYTHONPATH_ROOT="/storage/home/tuochao/Mem-alpha-audio"

# conda env that has datasets + soundfile + librosa (reuse the nemo env).
CONDA_ENV="${CONDA_ENV:-nemo}"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -n "${CONDA_EXE:-}" ]; then
    source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}" || true

export PYTHONPATH="${PYTHONPATH_ROOT}"

MIX_ARGS=(
    --data_dir            "${DATA_PATH}"
    --snr                 "${SNRS[@]}"
    --pool_minutes        "${POOL_MINUTES}"
    --num_interf_min      "${NUM_INTERF_MIN}"
    --num_interf_max      "${NUM_INTERF_MAX}"
    --gap_max             "${GAP_MAX}"
    --seed                "${SEED}"
)
if [ "${#SEASON_FILTER[@]}" -gt 0 ]; then
    MIX_ARGS+=(--season_filter "${SEASON_FILTER[@]}")
fi

python -m audio_script.datasets.mix_speech_interference "${MIX_ARGS[@]}"
