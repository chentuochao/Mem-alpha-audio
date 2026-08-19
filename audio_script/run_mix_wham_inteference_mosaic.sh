#!/usr/bin/env bash
#
# Mix competing-speech INTERFERENCE into the Mix_Mosaic audio at SNR levels.
# Mix_Mosaic counterpart of run_mix_wham_inteference_perltqa.sh. Interference is
# drawn from the sdialog/voices-libritts HuggingFace voice bank: per
# conversation, 1-N tracks are built by concatenating randomly sampled clips
# (random [0, gap_max]s gap) and mixed into mixed_conv.wav at the requested SNR.
#
# Only the conversation folders listed in bundles.json (built by
# audio_script/make_mix_mosaic_bundles.py) are mixed. Per conversation: only
# mixed_conv.wav is mixed; all non-wav sidecars (transcript*.json, vad*.json)
# are copied verbatim; other .wav files are skipped.
#
# Produces sibling folders next to DATA_PATH:
#   <DATA_PATH>_interf_SNR-5 / _interf_SNR5 ...   (nested <pair>/<conv_id>)
#
# Feed one of them back in via:
#   RAW_DATA_PATH=<DATA_PATH>_interf_SNR-5 ./audio_script/run_demo_pipeline_mosaic.sh
#
set -euo pipefail

DATA_PATH="/checkpoint/seamless/tuochao/data/Mix_Mosaic/naturalistic/test"
SNRS=(0 5 10)
POOL_MINUTES=30
NUM_INTERF_MIN=1
NUM_INTERF_MAX=4
GAP_MAX=3.0
SEED=0
BUNDLE_FILES=("bundles.json")
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

python -m audio_script.datasets.mix_speech_interference_mosaic "${MIX_ARGS[@]}"
