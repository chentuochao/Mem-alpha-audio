"""
Mix competing-speech INTERFERENCE into the PerLTQA TTS audio at fixed SNRs.

PerLTQA counterpart of :mod:`audio_script.datasets.mix_speech_interference`. The
interference model (LibriTTS "babble" tracks, one signal per item reused across
SNRs) is identical; only the folder layout and file selection differ.

Unlike the flat Bazinga layout, PerLTQA is nested per dialogue::

    <data_dir>/<profile>/<chunk_id>/dialogue_mono_TTS.wav      <- mixed
                                    dialogue_multichannel_TTS.wav  (other .wav: skipped)
                                    *_TTS.npy, *_annotation.json,
                                    *_vad.npy, channel_map.json    (non-wav: copied verbatim)

Only the dialogue folders referenced by a *valid profile* in
``bundles_multi.json`` / ``bundles_per_profile.json`` are processed (the union of
their ``chunks[].rel_path`` entries). Everything else on disk is ignored.

For each processed dialogue folder:
  * ``dialogue_mono_TTS.wav`` is mixed with interference at each SNR.
  * every *non-wav* sidecar in the folder is copied verbatim (timestamps
    unchanged, so annotations stay aligned).
  * other ``.wav`` files (e.g. ``dialogue_multichannel_TTS.wav``) are NOT copied.

Top-level non-wav files in ``<data_dir>`` (the bundles/stats JSONs) are also
copied into each output folder so it stays a drop-in replacement.

Folder layout produced::

    <data_dir>_interf_SNR10/<profile>/<chunk_id>/dialogue_mono_TTS.wav ...
    <data_dir>_interf_SNR5/ ...

Example::

    python -m audio_script.datasets.mix_speech_interference_perltqa \\
        --data_dir /checkpoint/seamless/tuochao/data/PerLTQA/dialogue_tts_en_v2 \\
        --snr 10 5 0 \\
        --pool_minutes 30 \\
        --num_interf_min 1 --num_interf_max 4 \\
        --gap_max 3.0 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import List

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    raise ImportError("librosa is required (used for loading/resampling audio)") from exc

# Reuse the interference model / mixing math from the Bazinga version.
from audio_script.datasets.mix_speech_interference import (
    SR,
    DEFAULT_HF_DATASET,
    build_speech_pool,
    build_interference,
    mix_at_snr,
    _peak_normalize,
    out_dir_for_snr,
)

MONO_WAV_NAME = "dialogue_mono_TTS.wav"
DEFAULT_BUNDLE_FILES = ["bundles_multi.json", "bundles_per_profile.json"]


def load_valid_relpaths(data_dir: str, bundle_files: List[str]) -> List[str]:
    """Union of ``chunks[].rel_path`` across the given bundle JSON files.

    These rel_paths are the dialogue folders belonging to *valid profiles*; only
    they are mixed. Missing bundle files are skipped with a warning.
    """
    relpaths: set[str] = set()
    for fn in bundle_files:
        path = fn if os.path.isabs(fn) else os.path.join(data_dir, fn)
        if not os.path.exists(path):
            print(f"[warn] bundle file not found, skipping: {path}")
            continue
        with open(path, "r") as f:
            data = json.load(f)
        n_before = len(relpaths)
        for bundle in data.get("bundles", []):
            for prof in bundle.get("profiles", []):
                for chunk in prof.get("chunks", []):
                    rp = chunk.get("rel_path")
                    if rp:
                        relpaths.add(rp)
        print(f"[bundles] {os.path.basename(path)}: +{len(relpaths) - n_before} "
              f"new rel_path(s) (running total {len(relpaths)})")
    return sorted(relpaths)


def copy_nonwav_sidecars(src_dir: str, dst_dir: str, overwrite: bool) -> int:
    """Copy every non-wav *file* from ``src_dir`` to ``dst_dir``. Returns count."""
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(src_dir)):
        src = os.path.join(src_dir, name)
        if not os.path.isfile(src) or name.lower().endswith(".wav"):
            continue
        dst = os.path.join(dst_dir, name)
        if overwrite or not os.path.exists(dst):
            shutil.copy2(src, dst)
            n += 1
    return n


def process_perltqa(
    data_dir: str,
    snrs: List[float],
    clips: List[np.ndarray],
    sr: int,
    rng: np.random.Generator,
    overwrite: bool,
    num_interf_min: int,
    num_interf_max: int,
    gap_max: float,
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

        # One interference signal per dialogue, reused across all SNR levels.
        num_interf = int(rng.integers(num_interf_min, num_interf_max + 1))
        interf = build_interference(clips, speech.shape[0], sr, num_interf, gap_max, rng)

        for snr in snrs:
            out_sub = os.path.join(out_dirs[snr], rel)
            os.makedirs(out_sub, exist_ok=True)
            out_wav = os.path.join(out_sub, MONO_WAV_NAME)

            if os.path.exists(out_wav) and not overwrite:
                pass  # already mixed
            else:
                mix = _peak_normalize(mix_at_snr(speech, interf, snr))
                sf.write(out_wav, mix, sr)

            # Copy the per-dialogue non-wav sidecars verbatim (other .wav skipped).
            copy_nonwav_sidecars(src_dir, out_sub, overwrite)

        n_done += 1
        print(f"  [ok]   {rel}: {num_interf} interference track(s) -> "
              f"{len(snrs)} SNR(s)")

    print(f"\nDone. Mixed {n_done} dialogue folder(s) "
          f"({n_missing} missing {MONO_WAV_NAME}).")
    print("Point the PerLTQA step1 pipeline at one of the _interf_SNRx folders "
          "above to run on interfered audio.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mix competing-speech interference into PerLTQA audio at fixed SNRs.")
    p.add_argument("--data_dir", required=True,
                   help="PerLTQA raw folder (e.g. .../dialogue_tts_en_v2).")
    p.add_argument("--snr", type=float, nargs="+", default=[10.0, 5.0, 0.0],
                   help="SNR levels in dB (default: 10 5 0).")
    p.add_argument("--bundle_files", type=str, nargs="+", default=DEFAULT_BUNDLE_FILES,
                   help="Bundle JSON(s) defining valid profiles/rel_paths (relative "
                        f"to data_dir or absolute). Default: {DEFAULT_BUNDLE_FILES}.")
    p.add_argument("--pool_minutes", type=float, default=30.0,
                   help="Minutes of LibriTTS speech to stream into the in-memory pool.")
    p.add_argument("--hf_dataset", type=str, default=DEFAULT_HF_DATASET,
                   help=f"HuggingFace speech dataset (default: {DEFAULT_HF_DATASET}).")
    p.add_argument("--num_interf_min", type=int, default=1,
                   help="Minimum number of interference tracks per dialogue (default: 1).")
    p.add_argument("--num_interf_max", type=int, default=3,
                   help="Maximum number of interference tracks per dialogue (default: 3).")
    p.add_argument("--gap_max", type=float, default=2.0,
                   help="Max silence gap (seconds) between concatenated clips (default: 2.0).")
    p.add_argument("--sr", type=int, default=SR, help="Target sample rate (Hz).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for sampling.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-mix and overwrite existing output wavs/sidecars.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.num_interf_min < 1 or args.num_interf_max < args.num_interf_min:
        raise ValueError("Require 1 <= num_interf_min <= num_interf_max.")
    rng = np.random.default_rng(args.seed)
    clips = build_speech_pool(args.pool_minutes, args.sr, args.hf_dataset)
    process_perltqa(
        data_dir=args.data_dir,
        snrs=args.snr,
        clips=clips,
        sr=args.sr,
        rng=rng,
        overwrite=args.overwrite,
        num_interf_min=args.num_interf_min,
        num_interf_max=args.num_interf_max,
        gap_max=args.gap_max,
        bundle_files=args.bundle_files,
    )


if __name__ == "__main__":
    main()
