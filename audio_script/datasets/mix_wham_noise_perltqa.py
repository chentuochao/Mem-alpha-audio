"""
Mix WHAM noise into the PerLTQA TTS audio at fixed SNR levels.

PerLTQA counterpart of :mod:`audio_script.datasets.mix_wham_noise`. The WHAM
noise model (one noise segment per item, reused across SNRs) is identical; only
the folder layout and file selection differ, matching
:mod:`audio_script.datasets.mix_speech_interference_perltqa`.

Unlike the flat Bazinga layout, PerLTQA is nested per dialogue::

    <data_dir>/<profile>/<chunk_id>/dialogue_mono_TTS.wav      <- mixed
                                    dialogue_multichannel_TTS.wav  (other .wav: skipped)
                                    *_TTS.npy, *_annotation.json,
                                    *_vad.npy, channel_map.json    (non-wav: copied verbatim)

Only the dialogue folders referenced by a *valid profile* in
``bundles_multi.json`` / ``bundles_per_profile.json`` are processed (the union of
their ``chunks[].rel_path`` entries). Everything else on disk is ignored.

For each processed dialogue folder:
  * ``dialogue_mono_TTS.wav`` is mixed with WHAM noise at each SNR.
  * every *non-wav* sidecar in the folder is copied verbatim (timestamps
    unchanged, so annotations stay aligned).
  * other ``.wav`` files (e.g. ``dialogue_multichannel_TTS.wav``) are NOT copied.

Top-level non-wav files in ``<data_dir>`` (the bundles/stats JSONs) are also
copied into each output folder so it stays a drop-in replacement.

Folder layout produced::

    <data_dir>_SNR10/<profile>/<chunk_id>/dialogue_mono_TTS.wav ...
    <data_dir>_SNR5/ ...

Example::

    python -m audio_script.datasets.mix_wham_noise_perltqa \\
        --data_dir /checkpoint/seamless/tuochao/data/PerLTQA/dialogue_tts_en_v2 \\
        --snr 10 5 0 \\
        --noise_pool_minutes 60 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    raise ImportError("librosa is required (used for loading/resampling audio)") from exc

# Reuse the WHAM noise model / mixing math from the Bazinga version.
from audio_script.datasets.mix_wham_noise import (
    SR,
    build_noise_pool,
    sample_noise,
    mix_at_snr,
    _peak_normalize,
    out_dir_for_snr,
)
# Reuse the PerLTQA folder traversal / file-selection helpers.
from audio_script.datasets.mix_speech_interference_perltqa import (
    MONO_WAV_NAME,
    DEFAULT_BUNDLE_FILES,
    load_valid_relpaths,
    copy_nonwav_sidecars,
)


def process_perltqa(
    data_dir: str,
    snrs: List[float],
    pool: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    overwrite: bool,
    bundle_files: List[str],
) -> None:
    relpaths = load_valid_relpaths(data_dir, bundle_files)
    if not relpaths:
        raise ValueError(
            f"No valid rel_paths found via {bundle_files} in {data_dir!r}")
    print(f"Processing {len(relpaths)} valid dialogue folder(s) from {data_dir}")

    out_dirs = {snr: out_dir_for_snr(data_dir, snr) for snr in snrs}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)
    print("Output folders:")
    for snr, d in out_dirs.items():
        print(f"  SNR {snr:g} dB -> {d}")

    # Copy the top-level non-wav files (bundles/stats JSONs) into each output
    # folder so it stays a self-contained drop-in replacement.
    for d in out_dirs.values():
        copy_nonwav_sidecars(data_dir, d, overwrite)

    n_done = n_missing = 0
    for rel in relpaths:
        src_dir = os.path.join(data_dir, rel)
        wav_path = os.path.join(src_dir, MONO_WAV_NAME)
        if not os.path.exists(wav_path):
            print(f"  [miss] {rel}: no {MONO_WAV_NAME}")
            n_missing += 1
            continue

        speech, _ = librosa.load(wav_path, sr=sr, mono=True)
        speech = speech.astype(np.float32)

        # One noise segment per dialogue, reused across all SNR levels.
        noise = sample_noise(pool, speech.shape[0], rng)

        for snr in snrs:
            out_sub = os.path.join(out_dirs[snr], rel)
            os.makedirs(out_sub, exist_ok=True)
            out_wav = os.path.join(out_sub, MONO_WAV_NAME)

            if os.path.exists(out_wav) and not overwrite:
                pass  # already mixed
            else:
                mix = _peak_normalize(mix_at_snr(speech, noise, snr))
                sf.write(out_wav, mix, sr)

            # Copy the per-dialogue non-wav sidecars verbatim (other .wav skipped).
            copy_nonwav_sidecars(src_dir, out_sub, overwrite)

        n_done += 1
        print(f"  [ok]   {rel} -> {len(snrs)} SNR(s)")

    print(f"\nDone. Mixed {n_done} dialogue folder(s) "
          f"({n_missing} missing {MONO_WAV_NAME}).")
    print("Point the PerLTQA step1 pipeline at one of the _SNRx folders above "
          "to run on noisy audio.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mix WHAM noise into PerLTQA audio at fixed SNRs.")
    p.add_argument("--data_dir", required=True,
                   help="PerLTQA raw folder (e.g. .../dialogue_tts_en_v2).")
    p.add_argument("--snr", type=float, nargs="+", default=[10.0, 5.0, 0.0],
                   help="SNR levels in dB (default: 10 5 0).")
    p.add_argument("--bundle_files", type=str, nargs="+", default=DEFAULT_BUNDLE_FILES,
                   help="Bundle JSON(s) defining valid profiles/rel_paths (relative "
                        f"to data_dir or absolute). Default: {DEFAULT_BUNDLE_FILES}.")
    p.add_argument("--noise_pool_minutes", type=float, default=60.0,
                   help="Minutes of WHAM noise to stream into the in-memory pool.")
    p.add_argument("--sr", type=int, default=SR, help="Target sample rate (Hz).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for noise sampling.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-mix and overwrite existing output wavs/sidecars.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    rng = np.random.default_rng(args.seed)
    pool = build_noise_pool(args.noise_pool_minutes, args.sr, args.seed)
    process_perltqa(
        data_dir=args.data_dir,
        snrs=args.snr,
        pool=pool,
        sr=args.sr,
        rng=rng,
        overwrite=args.overwrite,
        bundle_files=args.bundle_files,
    )


if __name__ == "__main__":
    main()
