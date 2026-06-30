#!/usr/bin/env bash
#
# Mix WHAM noise into the Bazinga audio at SNR = 10 / 5 / 0 dB.
# Produces sibling folders next to DATA_PATH:
#   <DATA_PATH>_SNR10 / _SNR5 / _SNR0   (noisy .en.wav + verbatim .txt)
#
# Then point run_demo_step1_bazinga.sh's DATA_PATH at one of those folders.
#
set -euo pipefail

DATA_PATH="/checkpoint/seamless/tuochao/data/bazinga/data/TheBigBangTheory"
SNRS=(15 10 5)
NOISE_POOL_MINUTES=30
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
    --noise_pool_minutes  "${NOISE_POOL_MINUTES}"
    --seed                "${SEED}"
)
if [ "${#SEASON_FILTER[@]}" -gt 0 ]; then
    MIX_ARGS+=(--season_filter "${SEASON_FILTER[@]}")
fi

python -m audio_script.datasets.mix_wham_noise "${MIX_ARGS[@]}"
