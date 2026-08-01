"""
Synthetically mix *speech interference* into the Bazinga audio at fixed SNRs.

Same driver / folder layout as :mod:`audio_script.datasets.mix_wham_noise`, but
instead of stationary WHAM noise the interferer is competing speech drawn from
the ``sdialog/voices-libritts`` HuggingFace voice bank ("babble" style
interference).

For each TV-series episode:

  * 1-3 independent *interference tracks* are built (random per episode).
  * Each interference track is a concatenation of randomly sampled LibriTTS
    clips separated by a random silence gap of ``[0, gap_max]`` seconds. Clips
    are sampled *with replacement* and the track is grown until it is at least
    as long as the episode, then trimmed to match.
  * The 1-3 tracks are summed into a single interference signal, which is then
    scaled and mixed into the episode at each requested SNR.

The *same* interference signal is reused across SNR levels for a given episode,
so the only thing that changes between the ``_interf_SNRx`` folders is the
interference gain — a clean A/B for "how does interference level affect the
pipeline".

Folder layout produced (mirrors what ``Bazinga_loader`` expects)::

    <data_dir>/                       Friends.S01.E01.en.wav   Friends.S01.E01.txt ...
    <data_dir>_interf_SNR10/          Friends.S01.E01.en.wav   Friends.S01.E01.txt ...
    <data_dir>_interf_SNR5/           ...
    <data_dir>_interf_SNR0/           ...

Mixing model (per file, energy on the whole signal)::

    scale = sqrt( P_speech / (P_interf * 10^(snr/10)) )
    mix   = speech + scale * interference

Example::

    python -m audio_script.datasets.mix_speech_interference \\
        --data_dir /checkpoint/seamless/tuochao/data/bazinga/data/TheBigBangTheory \\
        --snr 10 5 0 \\
        --pool_minutes 30 \\
        --num_interf_min 1 --num_interf_max 3 \\
        --gap_max 2.0 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import shutil
from typing import List, Optional

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    raise ImportError("librosa is required (used for loading/resampling audio)") from exc


SR = 16000  # target sample rate (matches audio_script.Multi_ASR.constants.SR)
WAV_SUFFIX = ".en.wav"
DEFAULT_HF_DATASET = "sdialog/voices-libritts"


# ──────────────────────────────────────────────────────────────────────────────
# Interference speech pool
# ──────────────────────────────────────────────────────────────────────────────

def build_speech_pool(
    target_minutes: float,
    sr: int,
    hf_dataset: str = DEFAULT_HF_DATASET,
) -> List[np.ndarray]:
    """Stream LibriTTS voice clips until ``target_minutes`` of audio is collected.

    Returns a *list* of mono float32 clips (each resampled to ``sr``). Keeping
    the clips separate — rather than one concatenated pool — lets us build
    interference tracks by sampling whole clips with replacement.
    """
    from datasets import Audio, load_dataset

    target_samples = int(target_minutes * 60 * sr)
    print(f"[interf] Streaming '{hf_dataset}' speech until ~{target_minutes:.1f} min "
          f"({target_samples} samples @ {sr} Hz) collected ...")

    ds = load_dataset(hf_dataset, split="train", streaming=True)
    # Don't let `datasets` auto-decode audio (avoids torchcodec/FFmpeg issues);
    # we decode the embedded bytes ourselves with soundfile.
    ds = ds.cast_column("audio", Audio(decode=False))

    clips: List[np.ndarray] = []
    total = 0
    for ex in ds:
        audio_obj = ex["audio"]
        if audio_obj.get("bytes") is None:
            continue
        with io.BytesIO(audio_obj["bytes"]) as f:
            x, fs = sf.read(f, dtype="float32")
        x = np.asarray(x, dtype=np.float32)
        if x.ndim > 1:                       # stereo -> mono
            x = x.mean(axis=1)
        if fs != sr:
            x = librosa.resample(x, orig_sr=fs, target_sr=sr)
        if x.shape[0] == 0:
            continue
        clips.append(x.astype(np.float32))
        total += x.shape[0]
        if total >= target_samples:
            break

    if not clips:
        raise RuntimeError(f"No speech was loaded from {hf_dataset!r} — check dataset access.")

    print(f"[interf] Built speech pool: {len(clips)} clip(s), "
          f"{total} samples ({total / sr / 60:.1f} min).")
    return clips


def build_interference_track(
    clips: List[np.ndarray],
    length: int,
    sr: int,
    gap_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build one interference track of exactly ``length`` samples.

    Randomly samples clips (with replacement) and concatenates them, inserting a
    silence gap of ``Uniform[0, gap_max]`` seconds between successive clips, until
    the track reaches ``length`` samples; then trims to ``length``.
    """
    n_clips = len(clips)
    gap_max_samples = int(round(gap_max * sr))
    pieces: List[np.ndarray] = []
    total = 0
    first = True
    while total < length:
        if not first and gap_max_samples > 0:
            gap = int(rng.integers(0, gap_max_samples + 1))
            if gap > 0:
                pieces.append(np.zeros(gap, dtype=np.float32))
                total += gap
        clip = clips[int(rng.integers(0, n_clips))]
        pieces.append(clip)
        total += clip.shape[0]
        first = False

    track = np.concatenate(pieces)
    return track[:length].astype(np.float32)


def build_interference(
    clips: List[np.ndarray],
    length: int,
    sr: int,
    num_interf: int,
    gap_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sum ``num_interf`` independent interference tracks into one signal."""
    mix = np.zeros(length, dtype=np.float32)
    for _ in range(num_interf):
        mix += build_interference_track(clips, length, sr, gap_max, rng)
    return mix


# ──────────────────────────────────────────────────────────────────────────────
# Mixing
# ──────────────────────────────────────────────────────────────────────────────

def mix_at_snr(speech: np.ndarray, interf: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix ``interf`` into ``speech`` at ``snr_db`` (whole-signal energy)."""
    p_speech = float(np.mean(speech ** 2))
    p_interf = float(np.mean(interf ** 2))
    if p_interf <= 0.0 or p_speech <= 0.0:
        return speech.astype(np.float32)
    scale = np.sqrt(p_speech / (p_interf * (10.0 ** (snr_db / 10.0))))
    return (speech + scale * interf).astype(np.float32)


def _peak_normalize(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Scale ``x`` down so its max abs amplitude is <= ``peak`` (no-op if already)."""
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m > peak:
        x = x * (peak / m)
    return x.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────

def out_dir_for_snr(data_dir: str, snr: float) -> str:
    """``/.../TheBigBangTheory`` + snr 5 -> ``/.../TheBigBangTheory_interf_SNR5``."""
    base = data_dir.rstrip("/")
    snr_tag = f"{snr:g}"  # 10.0 -> "10", 0.0 -> "0", 2.5 -> "2.5"
    return f"{base}_interf_SNR{snr_tag}"


def process(
    data_dir: str,
    snrs: List[float],
    clips: List[np.ndarray],
    sr: int,
    rng: np.random.Generator,
    overwrite: bool,
    num_interf_min: int,
    num_interf_max: int,
    gap_max: float,
    season_filter: Optional[List[str]] = None,
) -> None:
    wav_files = sorted(glob.glob(os.path.join(data_dir, "*" + WAV_SUFFIX)))
    if not wav_files:
        raise ValueError(f"No '*{WAV_SUFFIX}' files found in {data_dir!r}")
    print(f"Found {len(wav_files)} episode(s) in {data_dir}")
    if season_filter:
        print(f"Season filter: {season_filter}")

    out_dirs = {snr: out_dir_for_snr(data_dir, snr) for snr in snrs}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)
    print("Output folders:")
    for snr, d in out_dirs.items():
        print(f"  SNR {snr:g} dB -> {d}")

    for wav_path in wav_files:
        episode_id = os.path.basename(wav_path).replace(WAV_SUFFIX, "")
        if season_filter and not any(s in episode_id for s in season_filter):
            print(f"  [skip] {episode_id} (no match in season_filter)")
            continue
        speech, _ = librosa.load(wav_path, sr=sr, mono=True)
        speech = speech.astype(np.float32)

        # One interference signal per episode, reused across all SNR levels.
        num_interf = int(rng.integers(num_interf_min, num_interf_max + 1))
        interf = build_interference(
            clips, speech.shape[0], sr, num_interf, gap_max, rng
        )
        print(f"  [interf] {episode_id}: {num_interf} interference track(s)")

        for snr in snrs:
            out_dir = out_dirs[snr]
            out_wav = os.path.join(out_dir, os.path.basename(wav_path))

            if os.path.exists(out_wav) and not overwrite:
                print(f"  [skip] {episode_id} @ SNR {snr:g} (exists)")
            else:
                mix = mix_at_snr(speech, interf, snr)
                mix = _peak_normalize(mix)
                sf.write(out_wav, mix, sr)
                print(f"  [ok]   {episode_id} @ SNR {snr:g} -> {out_wav}")

            # Copy the per-episode sidecars verbatim (timestamps unchanged, so
            # they stay aligned with the noisy audio): the annotation txt and
            # the ground-truth transcript / vad jsons.
            for pattern in (
                episode_id + "*.txt",
                episode_id + "*_gt_transcript.json",
                episode_id + "*_gt_vad.json",
            ):
                for src in glob.glob(os.path.join(data_dir, pattern)):
                    dst = os.path.join(out_dir, os.path.basename(src))
                    if overwrite or not os.path.exists(dst):
                        shutil.copy2(src, dst)

    print("\nDone. Point run_demo_step1_bazinga.sh's DATA_PATH at one of the "
          "_interf_SNRx folders above to run the pipeline on interfered audio.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mix competing-speech interference into Bazinga audio at fixed SNRs.")
    p.add_argument("--data_dir", required=True,
                   help="Original clean episode folder (e.g. .../TheBigBangTheory).")
    p.add_argument("--snr", type=float, nargs="+", default=[10.0, 5.0, 0.0],
                   help="SNR levels in dB (default: 10 5 0).")
    p.add_argument("--pool_minutes", type=float, default=30.0,
                   help="Minutes of LibriTTS speech to stream into the in-memory pool.")
    p.add_argument("--hf_dataset", type=str, default=DEFAULT_HF_DATASET,
                   help=f"HuggingFace speech dataset (default: {DEFAULT_HF_DATASET}).")
    p.add_argument("--num_interf_min", type=int, default=1,
                   help="Minimum number of interference tracks per episode (default: 1).")
    p.add_argument("--num_interf_max", type=int, default=3,
                   help="Maximum number of interference tracks per episode (default: 3).")
    p.add_argument("--gap_max", type=float, default=2.0,
                   help="Max silence gap (seconds) between concatenated clips (default: 2.0).")
    p.add_argument("--sr", type=int, default=SR, help="Target sample rate (Hz).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for sampling.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-mix and overwrite existing output wavs/txts.")
    p.add_argument("--season_filter", type=str, nargs="*", default=[],
                   help=("Optional substrings (e.g. 'Season01' 'Season02'). Only "
                         "episodes whose id contains at least one are processed; "
                         "empty list disables the filter."))
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.num_interf_min < 1 or args.num_interf_max < args.num_interf_min:
        raise ValueError("Require 1 <= num_interf_min <= num_interf_max.")
    rng = np.random.default_rng(args.seed)
    clips = build_speech_pool(args.pool_minutes, args.sr, args.hf_dataset)
    process(
        data_dir=args.data_dir,
        snrs=args.snr,
        clips=clips,
        sr=args.sr,
        rng=rng,
        overwrite=args.overwrite,
        num_interf_min=args.num_interf_min,
        num_interf_max=args.num_interf_max,
        gap_max=args.gap_max,
        season_filter=args.season_filter,
    )


if __name__ == "__main__":
    main()
