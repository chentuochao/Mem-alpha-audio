"""
Test: How does speech duration affect speaker embeddings?

For each speaker, extract sub-clips of increasing duration (e.g. 1s, 2s, 4s,
8s, …) and compute embeddings.  Then measure:

  1. **Self-consistency** – cosine similarity between a sub-clip embedding and
     the full-length embedding *of the same speaker in the same conversation*.
  2. **Same-speaker cross-conv** – cosine similarity between embeddings of the
     same physical speaker across *different conversations* at each duration.
  3. **Different-speaker** – cosine similarity between embeddings of *different
     physical speakers* at each duration.

As duration grows, (1) and (2) should increase while (3) should stay low,
giving us the minimum duration needed for reliable speaker discrimination.

Usage
-----
    python -m audio_script.test_duration_embedding \
        --data_path /checkpoint/seamless/data/Mosaic \
        --embedding_model_dir /path/to/wespeaker_model \
        --output_dir ./duration_test_output
"""

import argparse
import itertools
import json
import os
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from audio_script.step2_speaker_match import WeSpeakerBackend
from audio_script.datasets.SeamlessInteraction_loader import InterActDataset


# ─── Helpers ──────────────────────────────────────────────────────────


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def apply_vad(audio: np.ndarray, sr: int, vad_segments: List[Dict]) -> np.ndarray:
    """
    Keep only the voiced portions of *audio* according to VAD annotations
    and return them concatenated.

    Args:
        audio: 1-D waveform array
        sr: sample rate
        vad_segments: list of {"start": float, "end": float} dicts (seconds)

    Returns:
        Concatenated speech-only audio (may be shorter than original).
    """
    chunks = []
    for seg in vad_segments:
        s = max(0, int(seg["start"] * sr))
        e = min(len(audio), int(seg["end"] * sr))
        if e > s:
            chunks.append(audio[s:e])
    if not chunks:
        return np.array([], dtype=audio.dtype)
    return np.concatenate(chunks)


def crop_audio(audio: np.ndarray, sr: int, duration_sec: float) -> Optional[np.ndarray]:
    """Return the first *duration_sec* seconds of *audio*, or None if too short."""
    n_samples = int(duration_sec * sr)
    if len(audio) < n_samples:
        return None
    return audio[:n_samples]


def save_tmp_wav(audio: np.ndarray, sr: int, tmp_dir: str, tag: str) -> str:
    path = os.path.join(tmp_dir, f"{tag}.wav")
    sf.write(path, audio, sr)
    return path


# ─── Core experiment ──────────────────────────────────────────────────


def load_vad_json(path: str) -> List[Dict]:
    """
    Load a VAD file that may be either plain JSON (single array) or
    JSONL (one JSON object per line).

    Returns: [{"start": float, "end": float}, ...]
    """
    with open(path, "r") as f:
        text = f.read().strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        pass

    entries = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def collect_speaker_audios(
    dataset: InterActDataset,
    max_conversations: int = -1,
) -> Dict[str, List[Tuple[str, np.ndarray, int]]]:
    """
    Walk the InterAct dataset and return per-speaker *speech-only* audio arrays
    (silence removed using VAD annotations).

    Returns:
        speaker_id -> [(conv_id, speech_only_audio, sample_rate), ...]
    """
    speaker_audios: Dict[str, List[Tuple[str, np.ndarray, int]]] = defaultdict(list)
    n = len(dataset) if max_conversations < 0 else min(max_conversations, len(dataset))

    for idx in range(n):
        conv_id, s1, s2 = dataset.valid_data[idx]
        audiof1, audiof2 = dataset.audio_files[idx]
        vadf1, vadf2 = dataset.vad_files[idx]
        spk1, spk2 = dataset.speaker_data[idx]

        try:
            import librosa
            audio1, sr = librosa.load(audiof1, sr=dataset.sample_rate, mono=True)
            audio2, sr = librosa.load(audiof2, sr=dataset.sample_rate, mono=True)
        except Exception as e:
            print(f"  [SKIP] conv {conv_id}: {e}")
            continue

        try:
            vad1 = load_vad_json(vadf1)
            vad2 = load_vad_json(vadf2)
        except Exception as e:
            print(f"  [SKIP] conv {conv_id} VAD load failed: {e}")
            continue

        speech1 = apply_vad(audio1, sr, vad1)
        speech2 = apply_vad(audio2, sr, vad2)

        raw_dur1, raw_dur2 = len(audio1) / sr, len(audio2) / sr
        spk_dur1, spk_dur2 = len(speech1) / sr, len(speech2) / sr

        if len(speech1) == 0 or len(speech2) == 0:
            print(f"  [SKIP] conv {conv_id}: VAD produced empty speech for a speaker")
            continue

        speaker_audios[spk1].append((conv_id, speech1, sr))
        speaker_audios[spk2].append((conv_id, speech2, sr))
        print(
            f"  Loaded conv {conv_id}: "
            f"{spk1} ({raw_dur1:.1f}s raw -> {spk_dur1:.1f}s speech), "
            f"{spk2} ({raw_dur2:.1f}s raw -> {spk_dur2:.1f}s speech)"
        )

    return dict(speaker_audios)


def extract_embeddings_at_durations(
    speaker_audios: Dict[str, List[Tuple[str, np.ndarray, int]]],
    durations: List[float],
    backend: WeSpeakerBackend,
    tmp_dir: str,
) -> Dict[str, Dict[str, Dict[float, np.ndarray]]]:
    """
    For every (speaker, conversation, duration) triple, extract an embedding.

    Returns:
        speaker_id -> conv_id -> duration_sec -> embedding
    """
    embeddings: Dict[str, Dict[str, Dict[float, np.ndarray]]] = {}

    for spk_id, clips in speaker_audios.items():
        embeddings[spk_id] = {}
        for conv_id, audio, sr in clips:
            dur_embs: Dict[float, np.ndarray] = {}
            full_dur = len(audio) / sr
            for dur in durations:
                if dur > full_dur:
                    continue
                cropped = crop_audio(audio, sr, dur)
                if cropped is None:
                    continue
                tag = f"{spk_id}_{conv_id}_{dur:.1f}s"
                wav_path = save_tmp_wav(cropped, sr, tmp_dir, tag)
                try:
                    emb = backend.extract(wav_path)
                    dur_embs[dur] = emb
                except Exception as e:
                    print(f"  [WARN] embedding failed for {tag}: {e}")
            if dur_embs:
                embeddings[spk_id][conv_id] = dur_embs
                print(f"  {spk_id}/{conv_id}: embeddings at {sorted(dur_embs.keys())} sec")

    return embeddings


# ─── Analysis ─────────────────────────────────────────────────────────


def analyse_self_consistency(
    embeddings: Dict[str, Dict[str, Dict[float, np.ndarray]]],
    durations: List[float],
) -> Dict[float, List[float]]:
    """
    For each speaker & conversation, compare the sub-clip embedding to
    the longest available embedding (proxy for "full" speaker embedding).

    Returns:  duration -> [sim_scores]
    """
    results: Dict[float, List[float]] = defaultdict(list)
    for spk_id, convs in embeddings.items():
        for conv_id, dur_embs in convs.items():
            available = sorted(dur_embs.keys())
            if len(available) < 2:
                continue
            ref_emb = dur_embs[available[-1]]
            for dur in available[:-1]:
                sim = cosine_similarity(dur_embs[dur], ref_emb)
                results[dur].append(sim)
    return dict(results)


def analyse_same_speaker_cross_conv(
    embeddings: Dict[str, Dict[str, Dict[float, np.ndarray]]],
    durations: List[float],
) -> Dict[float, List[float]]:
    """
    For the same physical speaker across different conversations,
    compare embeddings at each duration.

    Returns:  duration -> [sim_scores]
    """
    results: Dict[float, List[float]] = defaultdict(list)
    for spk_id, convs in embeddings.items():
        conv_ids = list(convs.keys())
        if len(conv_ids) < 2:
            continue
        for (c1, c2) in itertools.combinations(conv_ids, 2):
            common_durs = set(convs[c1].keys()) & set(convs[c2].keys())
            for dur in common_durs:
                sim = cosine_similarity(convs[c1][dur], convs[c2][dur])
                results[dur].append(sim)
    return dict(results)


def analyse_different_speakers(
    embeddings: Dict[str, Dict[str, Dict[float, np.ndarray]]],
    durations: List[float],
    max_pairs: int = 500,
) -> Dict[float, List[float]]:
    """
    Compare embeddings between *different* speakers at each duration.
    Uses the first conversation per speaker to keep it simple.

    Returns:  duration -> [sim_scores]
    """
    results: Dict[float, List[float]] = defaultdict(list)

    spk_first: Dict[str, Dict[float, np.ndarray]] = {}
    for spk_id, convs in embeddings.items():
        first_conv = next(iter(convs.values()))
        spk_first[spk_id] = first_conv

    spk_ids = list(spk_first.keys())
    pairs = list(itertools.combinations(spk_ids, 2))
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]

    for s1, s2 in pairs:
        common_durs = set(spk_first[s1].keys()) & set(spk_first[s2].keys())
        for dur in common_durs:
            sim = cosine_similarity(spk_first[s1][dur], spk_first[s2][dur])
            results[dur].append(sim)

    return dict(results)


def summarise(label: str, data: Dict[float, List[float]]) -> Dict[float, Dict[str, float]]:
    """Print & return {duration: {mean, std, n}} for one analysis."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    print(f"  {'Duration':>10s}  {'Mean':>8s}  {'Std':>8s}  {'N':>6s}")

    summary: Dict[float, Dict[str, float]] = {}
    for dur in sorted(data.keys()):
        vals = np.array(data[dur])
        m, s = vals.mean(), vals.std()
        summary[dur] = {"mean": float(m), "std": float(s), "n": len(vals)}
        print(f"  {dur:10.1f}  {m:8.4f}  {s:8.4f}  {len(vals):6d}")
    return summary


# ─── Plotting ─────────────────────────────────────────────────────────


def plot_results(
    self_cons: Dict[float, Dict[str, float]],
    same_spk: Dict[float, Dict[str, float]],
    diff_spk: Dict[float, Dict[str, float]],
    output_path: str,
):
    """Save a matplotlib figure showing all three curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    def _plot(summary, label, color, marker):
        durs = sorted(summary.keys())
        means = [summary[d]["mean"] for d in durs]
        stds = [summary[d]["std"] for d in durs]
        ax.errorbar(durs, means, yerr=stds, label=label,
                     color=color, marker=marker, capsize=3, linewidth=2)

    if self_cons:
        _plot(self_cons, "Self-consistency (sub-clip vs full)", "tab:blue", "o")
    if same_spk:
        _plot(same_spk, "Same speaker, cross-conversation", "tab:green", "s")
    if diff_spk:
        _plot(diff_spk, "Different speakers", "tab:red", "^")

    ax.set_xlabel("Speech Duration (seconds)", fontsize=13)
    ax.set_ylabel("Cosine Similarity", fontsize=13)
    ax.set_title("Speaker Embedding Similarity vs. Speech Duration", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Test how speech duration affects speaker embedding quality"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Root path to the Mosaic / InterAct dataset",
    )
    parser.add_argument(
        "--diag_format", type=str, default="naturalistic",
        help="Dialogue format sub-folder (default: naturalistic)",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--embedding_model_dir", type=str, required=True,
        help="Path to WeSpeaker model directory",
    )
    parser.add_argument(
        "--embedding_device", type=str, default="cuda:0",
        help="Device for embedding model (default: cuda:0)",
    )
    parser.add_argument(
        "--durations", type=float, nargs="+",
        default=[1.0, 2.0, 4.0, 8.0, 16.0, 30.0],
        help="Durations (seconds) to test",
    )
    parser.add_argument(
        "--max_conversations", type=int, default=-1,
        help="Cap the number of conversations to process (-1 = all)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./duration_test_output",
        help="Directory to save results & plots",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = os.path.join(args.output_dir, "tmp_clips")
    os.makedirs(tmp_dir, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────
    print("Loading InterAct dataset …")
    dataset = InterActDataset(
        data_path=args.data_path,
        diag_format=args.diag_format,
        split=args.split,
    )
    print(f"  {len(dataset)} conversations found\n")

    # ── Collect per-speaker single-channel audio ──────────────────────
    print("Collecting per-speaker audio …")
    speaker_audios = collect_speaker_audios(dataset, args.max_conversations)
    print(f"\n  {len(speaker_audios)} unique speakers collected")

    multi_conv_speakers = {s: cs for s, cs in speaker_audios.items() if len(cs) >= 2}
    print(f"  {len(multi_conv_speakers)} speakers appear in ≥ 2 conversations\n")

    # ── Load embedding model ──────────────────────────────────────────
    print("Loading WeSpeaker embedding model …")
    device_id = 0
    if ":" in args.embedding_device:
        device_id = int(args.embedding_device.split(":")[-1])
    backend = WeSpeakerBackend(model_dir=args.embedding_model_dir, device=device_id)

    # ── Extract embeddings at each duration ───────────────────────────
    print("\nExtracting embeddings at each duration …")
    durations = sorted(args.durations)
    embeddings = extract_embeddings_at_durations(
        speaker_audios, durations, backend, tmp_dir
    )

    # ── Analysis ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ANALYSIS")
    print("=" * 60)

    self_cons_raw = analyse_self_consistency(embeddings, durations)
    same_spk_raw = analyse_same_speaker_cross_conv(embeddings, durations)
    diff_spk_raw = analyse_different_speakers(embeddings, durations)

    self_cons = summarise("Self-consistency (sub-clip vs. longest clip)", self_cons_raw)
    same_spk = summarise("Same speaker, cross-conversation", same_spk_raw)
    diff_spk = summarise("Different speakers", diff_spk_raw)

    # ── Save JSON results ─────────────────────────────────────────────
    results = {
        "durations_tested": durations,
        "self_consistency": {str(k): v for k, v in self_cons.items()},
        "same_speaker_cross_conv": {str(k): v for k, v in same_spk.items()},
        "different_speakers": {str(k): v for k, v in diff_spk.items()},
    }
    json_path = os.path.join(args.output_dir, "duration_embedding_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved to {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────
    plot_path = os.path.join(args.output_dir, "duration_vs_similarity.png")
    plot_results(self_cons, same_spk, diff_spk, plot_path)


if __name__ == "__main__":
    main()
