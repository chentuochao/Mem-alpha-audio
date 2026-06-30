"""
Synthetically mix WHAM noise into the Bazinga audio at fixed SNR levels.

For each requested SNR (e.g. 10 / 5 / 0 dB) this script clones an existing
Bazinga episode folder into a sibling ``<name>_SNR<snr>`` folder, mixing WHAM
noise into every ``*.en.wav`` while copying the matching ``*.txt`` annotation
verbatim (timestamps are unchanged, so the noisy audio stays aligned).

Folder layout produced (mirrors what ``Bazinga_loader`` expects)::

    <data_dir>/                       Friends.S01.E01.en.wav   Friends.S01.E01.txt ...
    <data_dir>_SNR10/                 Friends.S01.E01.en.wav   Friends.S01.E01.txt ...
    <data_dir>_SNR5/                  ...
    <data_dir>_SNR0/                  ...

Mixing model (per file, energy on the whole signal)::

    scale = sqrt( P_speech / (P_noise * 10^(snr/10)) )
    mix   = speech + scale * noise

The *same* noise segment is reused across SNR levels for a given episode, so
the only thing that changes between the _SNRx folders is the noise gain — a
clean A/B for "how does noise level affect the pipeline".

Noise source: the ``philgzl/wham`` HuggingFace dataset, streamed once into an
in-memory pool (resampled to the target SR). Episodes draw a random contiguous
segment from that pool, wrapping around if the episode is longer than the pool.

Example::

    python -m audio_script.datasets.mix_wham_noise \\
        --data_dir /checkpoint/seamless/tuochao/data/bazinga/data/TheBigBangTheory \\
        --snr 10 5 0 \\
        --noise_pool_minutes 60 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import shutil
from typing import List

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    raise ImportError("librosa is required (used for loading/resampling audio)") from exc


SR = 16000  # target sample rate (matches audio_script.Multi_ASR.constants.SR)
WAV_SUFFIX = ".en.wav"


# ──────────────────────────────────────────────────────────────────────────────
# WHAM noise pool
# ──────────────────────────────────────────────────────────────────────────────

def build_noise_pool(target_minutes: float, sr: int, seed: int) -> np.ndarray:
    """Stream WHAM noise clips until ``target_minutes`` of audio is collected.

    Returns a single mono float32 array (resampled to ``sr``).
    """
    from datasets import Features, Value, load_dataset

    target_samples = int(target_minutes * 60 * sr)
    print(f"[noise] Streaming WHAM noise until ~{target_minutes:.1f} min "
          f"({target_samples} samples @ {sr} Hz) collected ...")

    pieces: List[np.ndarray] = []
    total = 0
    ds = load_dataset(
        "philgzl/wham",
        split="train",
        streaming=True,
        features=Features({"audio": Value("binary"), "name": Value("string")}),
    )

    for item in ds:
        x, fs = sf.read(io.BytesIO(item["audio"]))
        x = np.asarray(x, dtype=np.float32)
        if x.ndim > 1:                       # stereo -> mono
            x = x.mean(axis=1)
        if fs != sr:
            x = librosa.resample(x, orig_sr=fs, target_sr=sr)
        pieces.append(x.astype(np.float32))
        total += x.shape[0]
        if total >= target_samples:
            break

    if not pieces:
        raise RuntimeError("No WHAM noise was loaded — check dataset access.")

    pool = np.concatenate(pieces)
    print(f"[noise] Built noise pool: {pool.shape[0]} samples "
          f"({pool.shape[0] / sr / 60:.1f} min) from {len(pieces)} clip(s).")
    return pool


def sample_noise(pool: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Return a noise segment of exactly ``length`` samples from ``pool``.

    Picks a random start offset; wraps (tiles) if the requested length exceeds
    the remaining pool.
    """
    n = pool.shape[0]
    if length <= n:
        start = int(rng.integers(0, n - length + 1))
        return pool[start:start + length]
    # Episode longer than the whole pool: tile from a random offset.
    start = int(rng.integers(0, n))
    reps = int(np.ceil((length + start) / n))
    tiled = np.tile(pool, reps)
    return tiled[start:start + length]


# ──────────────────────────────────────────────────────────────────────────────
# Mixing
# ──────────────────────────────────────────────────────────────────────────────

def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix ``noise`` into ``speech`` at ``snr_db`` (whole-signal energy)."""
    p_speech = float(np.mean(speech ** 2))
    p_noise = float(np.mean(noise ** 2))
    if p_noise <= 0.0 or p_speech <= 0.0:
        return speech.astype(np.float32)
    scale = np.sqrt(p_speech / (p_noise * (10.0 ** (snr_db / 10.0))))
    return (speech + scale * noise).astype(np.float32)


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
    """``/.../TheBigBangTheory`` + snr 5 -> ``/.../TheBigBangTheory_SNR5``."""
    base = data_dir.rstrip("/")
    snr_tag = f"{snr:g}"  # 10.0 -> "10", 0.0 -> "0", 2.5 -> "2.5"
    return f"{base}_SNR{snr_tag}"


def process(
    data_dir: str,
    snrs: List[float],
    pool: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    overwrite: bool,
) -> None:
    wav_files = sorted(glob.glob(os.path.join(data_dir, "*" + WAV_SUFFIX)))
    if not wav_files:
        raise ValueError(f"No '*{WAV_SUFFIX}' files found in {data_dir!r}")
    print(f"Found {len(wav_files)} episode(s) in {data_dir}")

    out_dirs = {snr: out_dir_for_snr(data_dir, snr) for snr in snrs}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)
    print("Output folders:")
    for snr, d in out_dirs.items():
        print(f"  SNR {snr:g} dB -> {d}")

    for wav_path in wav_files:
        episode_id = os.path.basename(wav_path).replace(WAV_SUFFIX, "")
        speech, _ = librosa.load(wav_path, sr=sr, mono=True)
        speech = speech.astype(np.float32)

        # One noise segment per episode, reused across all SNR levels.
        noise = sample_noise(pool, speech.shape[0], rng)

        for snr in snrs:
            out_dir = out_dirs[snr]
            out_wav = os.path.join(out_dir, os.path.basename(wav_path))

            if os.path.exists(out_wav) and not overwrite:
                print(f"  [skip] {episode_id} @ SNR {snr:g} (exists)")
            else:
                mix = mix_at_snr(speech, noise, snr)
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
          "_SNRx folders above to run the pipeline on noisy audio.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mix WHAM noise into Bazinga audio at fixed SNRs.")
    p.add_argument("--data_dir", required=True,
                   help="Original clean episode folder (e.g. .../TheBigBangTheory).")
    p.add_argument("--snr", type=float, nargs="+", default=[10.0, 5.0, 0.0],
                   help="SNR levels in dB (default: 10 5 0).")
    p.add_argument("--noise_pool_minutes", type=float, default=60.0,
                   help="Minutes of WHAM noise to stream into the in-memory pool.")
    p.add_argument("--sr", type=int, default=SR, help="Target sample rate (Hz).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for noise sampling.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-mix and overwrite existing output wavs/txts.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    rng = np.random.default_rng(args.seed)
    pool = build_noise_pool(args.noise_pool_minutes, args.sr, args.seed)
    process(
        data_dir=args.data_dir,
        snrs=args.snr,
        pool=pool,
        sr=args.sr,
        rng=rng,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
