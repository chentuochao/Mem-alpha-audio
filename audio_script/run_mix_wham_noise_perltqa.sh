#!/usr/bin/env bash
#
# Mix WHAM noise into the PerLTQA TTS audio at SNR levels.
# PerLTQA counterpart of run_mix_wham_noise.sh. WHAM noise is streamed from the
# philgzl/wham HuggingFace dataset into an in-memory pool; per dialogue one noise
# segment is drawn and mixed into dialogue_mono_TTS.wav at the requested SNR.
#
# Only the dialogue folders belonging to a *valid profile* (union of
# chunks[].rel_path in bundles_multi.json / bundles_per_profile.json) are mixed.
# Per dialogue: only dialogue_mono_TTS.wav is mixed; all non-wav sidecars are
# copied verbatim; other .wav files (e.g. dialogue_multichannel_TTS.wav) skipped.
#
# Produces sibling folders next to DATA_PATH:
#   <DATA_PATH>_SNR0 / _SNR-5 ...   (nested per-dialogue)
#
set -euo pipefail

DATA_PATH="/checkpoint/seamless/tuochao/data/PerLTQA/dialogue_tts_en_v2"
SNRS=(0 5)
NOISE_POOL_MINUTES=30
SEED=0
BUNDLE_FILES=("bundles_multi.json" "bundles_per_profile.json")
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
    --bundle_files        "${BUNDLE_FILES[@]}"
)

python -m audio_script.datasets.mix_wham_noise_perltqa "${MIX_ARGS[@]}"
