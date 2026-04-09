"""
Step 1 Evaluation: Compute DER and cpWER from saved inference results.

Reads the step1_manifest.json produced by step1_diarize_asr.py and evaluates
each conversation against its ground-truth VAD and transcript files.

Metrics:
  - DER  (Diarization Error Rate): compares diar.npy vs vad1.json / vad2.json
  - cpWER (concatenated min-permutation WER): compares predicted per-speaker
    transcripts vs transcript1.json / transcript2.json
"""

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np

from audio_script.eval.multitalker_metrics import compute_der, calculate_session_cpWER, normalize_string


# ── Helpers ──────────────────────────────────────────────────────────────


def load_vad_json(path: str) -> List[Dict]:
    """Load a VAD file (plain JSON array or JSONL) → [{start, end}, ...]"""
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


def vad_segments_to_binary(vad_segments: List[Dict], total_frames: int,
                           frame_duration: float = 0.01) -> np.ndarray:
    """Convert a list of {start, end} VAD segments to a binary vector."""
    binary = np.zeros(total_frames, dtype=np.float32)
    for seg in vad_segments:
        s = int(seg["start"] / frame_duration)
        e = int(seg["end"] / frame_duration)
        e = min(e, total_frames)
        if s < total_frames:
            binary[s:e] = 1.0
    return binary


def extract_text_from_transcript(transcript_path: str) -> str:
    """Load a transcript JSON and return concatenated segment-level text."""
    with open(transcript_path, "r") as f:
        transcript = json.load(f)
    words = []
    for seg in transcript:
        words.append(seg["text"])
    return " ".join(words)


def build_speaker_transcripts(word_list: Dict[str, List[Dict]]) -> List[str]:
    """
    From a word_list dict {speaker_id: [{word, start, end, ...}, ...]},
    return a list of concatenated text strings for each non-empty speaker.
    """
    transcripts = {}

    for segment in word_list:
        print(segment)
        trans = segment["text"]
        speaker = segment["speaker"]
        if speaker not in transcripts:
            transcripts[speaker] = trans
        else:
            transcripts[speaker] += trans

    transcripts_plain = []
    for speaker, transcript in transcripts.items():
        if len(transcript) == 0:
            continue
        # transcript = normalize_string(transcript)
        transcripts_plain.append(transcript)
    return transcripts_plain

def discover_samples(data_dir: str) -> List[Dict]:
    """
    Walk the directory tree and find all sample folders containing
    sample_info.json produced by Step 1.

    Returns a list of entry dicts, each augmented with ``sample_dir``,
    ``diart_path``, and ``transcript_path``.
    """
    samples = []
    for info_path in sorted(glob.glob(os.path.join(data_dir, "*", "*", "sample_info.json"))):
        sample_dir = os.path.dirname(info_path)
        diar_path = os.path.join(sample_dir, "diart_pred.npy")
        transcript_path = os.path.join(sample_dir, "transcript_pred.json")
        if not os.path.exists(diar_path) or not os.path.exists(transcript_path):
            continue
        with open(info_path, "r") as f:
            info = json.load(f)
        info["sample_dir"] = sample_dir
        info["diart_path"] = diar_path
        info["transcript_path"] = transcript_path
        samples.append(info)
    return samples


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Step 1 Evaluation: DER and cpWER from saved inference results"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root output directory from Step 1 containing "
             "{spk_pair}/{conv_id}/sample_info.json sub-folders",
    )
    parser.add_argument(
        "--frame_duration",
        type=float,
        default=None,
        help="Override frame duration (seconds) for DER. "
             "If not set, uses feat_len_sec from sample_info.",
    )
    args = parser.parse_args()

    samples = discover_samples(args.data_dir)
    print(f"Found {len(samples)} samples under {args.data_dir}")
    if not samples:
        print("No samples found. Check your --data_dir path.")
        return

    all_ders = []
    all_cpwers = []

    for entry in samples:
        spk_pair = entry["spk_pair"]
        conv_id = entry["conv_id"]
        print(f"\n{'=' * 70}")
        print(f"Evaluating: {spk_pair} / {conv_id}")
        print(f"{'=' * 70}")

        # ── Load predictions ──────────────────────────────────────────
        diar_result = np.load(entry["diart_path"])
        with open(entry["transcript_path"], "r") as f:
            word_list = json.load(f)
        print(word_list)
        frame_duration = args.frame_duration or entry.get("feat_len_sec", 0.08)

        # ── Evaluate DER ──────────────────────────────────────────────
        total_frames = diar_result.shape[0]
        vad1 = load_vad_json(entry["vad1_path"])
        vad2 = load_vad_json(entry["vad2_path"])
        gt_spk1 = vad_segments_to_binary(vad1, total_frames, frame_duration)
        gt_spk2 = vad_segments_to_binary(vad2, total_frames, frame_duration)
        gt_matrix = np.stack([gt_spk1, gt_spk2], axis=0)  # (2, T)
        pred_matrix = diar_result.T  # (num_speakers, T)

        print(f"  pred shape: {pred_matrix.shape}, gt shape: {gt_matrix.shape}")
        der, der_details = compute_der(pred_matrix, gt_matrix,
                                       frame_duration=frame_duration)
        print(f"  DER: {der:.4f}  "
              f"(miss={der_details['miss']:.2f}s, fa={der_details['fa']:.2f}s, "
              f"conf={der_details['conf']:.2f}s, total={der_details['total']:.2f}s)")
        all_ders.append({
            "spk_pair": spk_pair, "conv_id": conv_id,
            "der": der, **der_details,
        })

        # ── Evaluate cpWER ────────────────────────────────────────────
        spk_hypothesis = build_speaker_transcripts(word_list)
        if len(spk_hypothesis) < 1:
            print("  [SKIP cpWER] no valid speaker transcripts")
            continue

        ref_text1 = extract_text_from_transcript(entry["transcript1_path"])
        ref_text2 = extract_text_from_transcript(entry["transcript2_path"])
        spk_reference = [ref_text1, ref_text2]

        print(f"  Hypothesis ({len(spk_hypothesis)} speakers): "
              f"{[t[:80] + '...' for t in spk_hypothesis]}")
        print(f"  Reference: {[t[:80] + '...' for t in spk_reference]}")

        cpwer, best_perm, _ = calculate_session_cpWER(spk_hypothesis, spk_reference)
        print(f"  cpWER: {cpwer:.4f}")
        all_cpwers.append({
            "spk_pair": spk_pair, "conv_id": conv_id, "cpwer": cpwer,
        })

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 70}")

    if all_ders:
        avg_der = np.mean([d["der"] for d in all_ders])
        avg_miss = np.mean([d["miss"] for d in all_ders])
        avg_fa = np.mean([d["fa"] for d in all_ders])
        avg_conf = np.mean([d["conf"] for d in all_ders])
        print(f"\nDER ({len(all_ders)} sessions)")
        print(f"  Avg DER:  {avg_der:.4f}")
        print(f"  Avg Miss: {avg_miss:.2f}s  |  Avg FA: {avg_fa:.2f}s  |  Avg Conf: {avg_conf:.2f}s")
        for d in all_ders:
            print(f"    {d['spk_pair']}/{d['conv_id']}: DER={d['der']:.4f}")

    if all_cpwers:
        avg_cpwer = np.mean([c["cpwer"] for c in all_cpwers])
        print(f"\ncpWER ({len(all_cpwers)} sessions)")
        print(f"  Avg cpWER: {avg_cpwer:.4f}")
        for c in all_cpwers:
            print(f"    {c['spk_pair']}/{c['conv_id']}: cpWER={c['cpwer']:.4f}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
