#!/usr/bin/env bash
#
# Mix competing-speech INTERFERENCE into the PerLTQA TTS audio at SNR levels.
# PerLTQA counterpart of run_mix_wham_inteference.sh. Interference is drawn from
# the sdialog/voices-libritts HuggingFace voice bank: per dialogue, 1-N tracks
# are built by concatenating randomly sampled clips (random [0, gap_max]s gap)
# and mixed into dialogue_mono_TTS.wav at the requested SNR.
#
# Only the dialogue folders belonging to a *valid profile* (union of
# chunks[].rel_path in bundles_multi.json / bundles_per_profile.json) are mixed.
# Per dialogue: only dialogue_mono_TTS.wav is mixed; all non-wav sidecars are
# copied verbatim; other .wav files (e.g. dialogue_multichannel_TTS.wav) skipped.
#
# Produces sibling folders next to DATA_PATH:
#   <DATA_PATH>_interf_SNR5 / _interf_SNR10 / _interf_SNR15   (nested per-dialogue)
#
set -euo pipefail

DATA_PATH="/checkpoint/seamless/tuochao/data/PerLTQA/dialogue_tts_en_name_replaced"
SNRS=(-5)
POOL_MINUTES=30
NUM_INTERF_MIN=1
NUM_INTERF_MAX=4
GAP_MAX=3.0
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
    --pool_minutes        "${POOL_MINUTES}"
    --num_interf_min      "${NUM_INTERF_MIN}"
    --num_interf_max      "${NUM_INTERF_MAX}"
    --gap_max             "${GAP_MAX}"
    --seed                "${SEED}"
    --bundle_files        "${BUNDLE_FILES[@]}"
)

python -m audio_script.datasets.mix_speech_interference_perltqa "${MIX_ARGS[@]}"
