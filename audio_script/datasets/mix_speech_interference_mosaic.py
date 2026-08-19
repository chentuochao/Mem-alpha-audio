"""
Mix competing-speech INTERFERENCE into the Mix_Mosaic audio at fixed SNRs.

Mix_Mosaic counterpart of :mod:`audio_script.datasets.mix_speech_interference`
(and a sibling of :mod:`audio_script.datasets.mix_speech_interference_perltqa`).
The interference model (LibriTTS "babble" tracks, one signal per item reused
across SNRs) is identical; only the folder layout and file selection differ.

Mix_Mosaic is nested per speaker-pair / conversation::

    <data_dir>/<Pxxx_Pyyy>/<conv_id>/mixed_conv.wav              <- mixed
                                     transcript1.json  transcript2.json
                                     vad1.json         vad2.json   (non-wav: copied verbatim)

Only the conversation folders listed in ``bundles.json`` (built by
``audio_script/make_mix_mosaic_bundles.py``) are processed — the union of its
``bundles[].conversations[]`` ``<pair>/<conv_id>`` entries. With
``--bundle_files`` set to nothing (empty list) the whole tree is walked instead
(every ``<pair>/<conv_id>`` folder holding a ``mixed_conv.wav``).

For each processed conversation folder:
  * ``mixed_conv.wav`` is mixed with interference at each SNR.
  * every *non-wav* sidecar in the folder is copied verbatim (timestamps
    unchanged, so transcripts/VAD stay aligned).
  * other ``.wav`` files are NOT copied.

Top-level non-wav files in ``<data_dir>`` (``bundles.json`` etc.) are also
copied into each output folder so it stays a drop-in replacement. Note the
copied ``bundles.json`` keeps its original absolute ``path`` / ``data_dir``
fields (the pipeline only reads ``bundles[].folders``).

Folder layout produced::

    <data_dir>_interf_SNR10/<Pxxx_Pyyy>/<conv_id>/mixed_conv.wav ...
    <data_dir>_interf_SNR5/ ...

Example::

    python -m audio_script.datasets.mix_speech_interference_mosaic \\
        --data_dir /checkpoint/seamless/tuochao/data/Mix_Mosaic/naturalistic/test \\
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
# The sidecar-copy helper is layout agnostic — reuse the PerLTQA one.
from audio_script.datasets.mix_speech_interference_perltqa import copy_nonwav_sidecars

MONO_WAV_NAME = "mixed_conv.wav"
DEFAULT_BUNDLE_FILES = ["bundles.json"]


def _walk_conv_relpaths(data_dir: str) -> List[str]:
    """Every ``<pair>/<conv_id>`` folder under ``data_dir`` with a mixed wav."""
    rels: List[str] = []
    for pair in sorted(os.listdir(data_dir)):
        pair_dir = os.path.join(data_dir, pair)
        if not os.path.isdir(pair_dir):
            continue
        for conv in sorted(os.listdir(pair_dir)):
            conv_dir = os.path.join(pair_dir, conv)
            if os.path.isfile(os.path.join(conv_dir, MONO_WAV_NAME)):
                rels.append(os.path.join(pair, conv))
    return rels


def load_valid_relpaths(data_dir: str, bundle_files: List[str]) -> List[str]:
    """Union of ``<pair>/<conv_id>`` across the given Mix_Mosaic bundle JSONs.

    Falls back to walking the tree when no bundle file is given / found, so the
    script also works on an unbundled dump.
    """
    if not bundle_files:
        rels = _walk_conv_relpaths(data_dir)
        print(f"[bundles] no bundle file requested: walked {len(rels)} conversation folder(s)")
        return rels

    relpaths: set[str] = set()
    found_any = False
    for fn in bundle_files:
        path = fn if os.path.isabs(fn) else os.path.join(data_dir, fn)
        if not os.path.exists(path):
            print(f"[warn] bundle file not found, skipping: {path}")
            continue
        found_any = True
        with open(path, "r") as f:
            data = json.load(f)
        n_before = len(relpaths)
        for bundle in data.get("bundles", []):
            for conv in bundle.get("conversations", []):
                pair, conv_id = conv.get("pair"), conv.get("conv_id")
                if pair and conv_id:
                    relpaths.add(os.path.join(pair, conv_id))
        print(f"[bundles] {os.path.basename(path)}: +{len(relpaths) - n_before} "
              f"new conversation(s) (running total {len(relpaths)})")

    if not found_any:
        rels = _walk_conv_relpaths(data_dir)
        print(f"[bundles] no bundle file found: walked {len(rels)} conversation folder(s)")
        return rels
    return sorted(relpaths)


def process_mosaic(
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
            f"No conversation folders found via {bundle_files} in {data_dir!r}")
    print(f"Processing {len(relpaths)} conversation folder(s) from {data_dir}")

    out_dirs = {snr: out_dir_for_snr(data_dir, snr) for snr in snrs}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)
    print("Output folders:")
    for snr, d in out_dirs.items():
        print(f"  SNR {snr:g} dB -> {d}")

    # Copy the top-level non-wav files (bundles.json etc.) into each output
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

        # One interference signal per conversation, reused across all SNR levels.
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

            # Copy the per-conversation non-wav sidecars verbatim.
            copy_nonwav_sidecars(src_dir, out_sub, overwrite)

        n_done += 1
        print(f"  [ok]   {rel}: {num_interf} interference track(s) -> "
              f"{len(snrs)} SNR(s)")

    print(f"\nDone. Mixed {n_done} conversation folder(s) "
          f"({n_missing} missing {MONO_WAV_NAME}).")
    print("Point run_demo_pipeline_mosaic.sh's RAW_DATA_PATH at one of the "
          "_interf_SNRx folders above to run on interfered audio.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mix competing-speech interference into Mix_Mosaic audio at fixed SNRs.")
    p.add_argument("--data_dir", required=True,
                   help="Mix_Mosaic root (e.g. .../Mix_Mosaic/naturalistic/test).")
    p.add_argument("--snr", type=float, nargs="+", default=[10.0, 5.0, 0.0],
                   help="SNR levels in dB (default: 10 5 0).")
    p.add_argument("--bundle_files", type=str, nargs="*", default=DEFAULT_BUNDLE_FILES,
                   help="Bundle JSON(s) listing the conversations to mix (relative to "
                        f"data_dir or absolute). Default: {DEFAULT_BUNDLE_FILES}. Pass "
                        "no value to walk the whole tree instead.")
    p.add_argument("--pool_minutes", type=float, default=30.0,
                   help="Minutes of LibriTTS speech to stream into the in-memory pool.")
    p.add_argument("--hf_dataset", type=str, default=DEFAULT_HF_DATASET,
                   help=f"HuggingFace speech dataset (default: {DEFAULT_HF_DATASET}).")
    p.add_argument("--num_interf_min", type=int, default=1,
                   help="Minimum number of interference tracks per conversation (default: 1).")
    p.add_argument("--num_interf_max", type=int, default=3,
                   help="Maximum number of interference tracks per conversation (default: 3).")
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
    process_mosaic(
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
