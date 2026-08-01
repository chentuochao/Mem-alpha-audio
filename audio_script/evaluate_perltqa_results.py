#!/usr/bin/env python3
"""
Evaluate PerLTQA Step-1 (diar + ASR) results: DER + cpWER.

Differences from the Bazinga-oriented ``evaluate_audio_results.py``:
  * No hardcoded ``SeasonXX`` path filter — PerLTQA conv_ids are
    ``<Profile>_<dialogue>`` with no season, so nothing is silently skipped.
    (Optional ``--profile_filter`` substrings if you want a subset.)
  * Handles PerLTQA's **turn-level** ground truth: ``transcript_gt.json`` entries
    carry a ``text`` field (a whole turn), whereas predictions carry ``word``.
    Both are concatenated per speaker for cpWER.
  * Metrics only (no plotting), so it runs without matplotlib.

Recursively finds every ``sample_info.json`` under the given folder and computes
DER (from ``diart_pred.npy`` vs the GT VAD) and cpWER (from
``transcript_pred.json`` vs ``transcript_gt.json``) when present.

Run in the ``mem`` env (has editdistance / numpy):
    PYTHONPATH=. python audio_script/evaluate_perltqa_results.py \
        ./Audio_Results/vibevoice/dialogue_tts_en_v2/step1/
    # optional: only some profiles, custom frame size, save a JSON summary
    PYTHONPATH=. python audio_script/evaluate_perltqa_results.py <dir> \
        --profile_filter Zhang_Xiaohong Mao_Gang --out summary.json
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from audio_script.eval.multitalker_metrics import (
    compute_der_bruteforce, calculate_session_cpWER,
)
from audio_script.eval.eval_utils import build_speaker_transcripts

FRAME_DEFAULT = 0.08


# ── IO helpers ────────────────────────────────────────────────────────────────
def load_vad_json(path: str) -> List[Dict]:
    """Load a VAD file (plain JSON array or JSONL) -> [{start, end}, ...]."""
    with open(path) as f:
        text = f.read().strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def vad_to_binary(segments: List[Dict], total_frames: int,
                  frame_duration: float) -> np.ndarray:
    binary = np.zeros(total_frames, dtype=np.float32)
    for seg in segments:
        s = int(seg["start"] / frame_duration)
        e = min(int(seg["end"] / frame_duration), total_frames)
        if s < total_frames:
            binary[s:e] = 1.0
    return binary


def _entry_text(w: Dict) -> str:
    """Text of a transcript entry: prediction uses 'word', PerLTQA GT uses 'text'."""
    return w.get("word", w.get("text", "")) or ""


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_der(pred_npy: str, vads: List[List[Dict]],
                frame_duration: float) -> Tuple[float, Dict]:
    pred_raw = np.load(pred_npy)               # (T, N_pred)
    if pred_raw.ndim == 1:
        pred_raw = pred_raw[:, np.newaxis]
    T, N_pred = pred_raw.shape
    pred_bin = (pred_raw.T >= 0.5).astype(np.float32)   # (N_pred, T)

    gt_mat = np.stack([vad_to_binary(v, T, frame_duration) for v in vads], axis=0)
    N_gt = gt_mat.shape[0]
    if N_gt > N_pred:                          # pad missing predicted speakers
        pad = np.zeros((N_gt, T), dtype=pred_bin.dtype)
        pad[:N_pred] = pred_bin
        pred_bin = pad

    der, details = compute_der_bruteforce(pred_bin, gt_mat, frame_duration)
    details["n_pred"] = N_pred
    details["n_gt"] = N_gt
    return der, details


def compute_cpwer(pred_word_list: Dict, gt_word_list: Dict) -> float:
    """cpWER with prediction ('word') vs PerLTQA turn GT ('text')."""
    hyp_texts, _ = build_speaker_transcripts(pred_word_list, pad_char=" ")
    gt_norm = {
        spk: [dict(w, word=_entry_text(w)) for w in words]
        for spk, words in gt_word_list.items()
    }
    ref_texts, _ = build_speaker_transcripts(gt_norm, pad_char=" ")
    if not hyp_texts or not ref_texts:
        return float("nan")
    cpwer, _, _, _ = calculate_session_cpWER(hyp_texts, ref_texts)
    return cpwer


# ── Driver ────────────────────────────────────────────────────────────────────
def evaluate_sample(si_path: str, frame_override: Optional[float]) -> Optional[Dict]:
    sample_folder = os.path.dirname(si_path)
    with open(si_path) as f:
        info = json.load(f)

    pred_npy = os.path.join(sample_folder, "diart_pred.npy")
    if not os.path.exists(pred_npy):
        print(f"  [SKIP] missing {pred_npy}")
        return None

    # GT VAD: PerLTQA writes a single vad_path -> {speaker: [segs]}.
    if info.get("vad_path"):
        with open(info["vad_path"]) as f:
            vads = list(json.load(f).values())
    elif info.get("vad1_path") and info.get("vad2_path"):
        vads = [load_vad_json(info["vad1_path"]), load_vad_json(info["vad2_path"])]
    else:
        print(f"  [SKIP] no VAD path in {si_path}")
        return None

    frame_dur = frame_override or info.get("feat_len_sec", FRAME_DEFAULT)

    der, details = compute_der(pred_npy, vads, frame_dur)

    cpwer = None
    pred_tp = info.get("pred_transcript_path")
    gt_tp = info.get("transcript_path")
    if pred_tp and os.path.exists(pred_tp) and gt_tp and os.path.exists(gt_tp):
        try:
            with open(pred_tp) as f:
                pred_wl = json.load(f)
            with open(gt_tp) as f:
                gt_wl = json.load(f)
            cpwer = compute_cpwer(pred_wl, gt_wl)
        except Exception as e:
            print(f"  [WARN] cpWER failed: {e}")

    # conv_id = the folder directly above CHUNK_i (i.e. "<Profile>_<dialogue>")
    conv_id = os.path.basename(os.path.dirname(sample_folder))
    line = f"  DER {der*100:6.2f}%"
    if cpwer is not None and not math.isnan(cpwer):
        line += f" | cpWER {cpwer*100:6.2f}%"
    print(line)
    return {
        "path": si_path, "conv_id": conv_id, "der": der,
        "miss": details["miss"], "fa": details["fa"], "conf": details["conf"],
        "total": details["total"], "n_pred": details["n_pred"], "n_gt": details["n_gt"],
        "cpwer": cpwer,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate PerLTQA Step-1 DER + cpWER over sample_info.json files.")
    ap.add_argument("folder", metavar="DIR",
                    help="Step-1 (or per-bundle Step-2) output dir to search recursively.")
    ap.add_argument("--frame_duration", type=float, default=None,
                    help="Frame size in seconds (default: sample_info.feat_len_sec or 0.08).")
    ap.add_argument("--profile_filter", nargs="*", default=[],
                    help="Only evaluate samples whose path contains one of these "
                         "substrings; empty = all.")
    ap.add_argument("--limit", type=int, default=0, help="cap #samples (<=0 = all).")
    ap.add_argument("--out", default=None, help="write a JSON summary here.")
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(f"ERROR: not a directory: {args.folder}")

    sample_infos = sorted(
        os.path.join(dp, "sample_info.json")
        for dp, _, fs in os.walk(args.folder) if "sample_info.json" in fs
    )
    if args.profile_filter:
        sample_infos = [p for p in sample_infos
                        if any(s in p for s in args.profile_filter)]
    if args.limit > 0:
        sample_infos = sample_infos[: args.limit]
    if not sample_infos:
        sys.exit(f"No sample_info.json found under {args.folder}")

    print(f"Found {len(sample_infos)} sample(s) under {args.folder}\n")

    results = []
    for idx, si in enumerate(sample_infos):
        print(f"[{idx + 1}/{len(sample_infos)}] {os.path.relpath(si, args.folder)}")
        try:
            r = evaluate_sample(si, args.frame_duration)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue
        if r:
            results.append(r)

    if not results:
        sys.exit("\nNo samples evaluated.")

    ders = [r["der"] for r in results]
    cpwers = [r["cpwer"] for r in results
              if r["cpwer"] is not None and not math.isnan(r["cpwer"])]
    print("\n" + "=" * 60)
    print(f"  PerLTQA Step-1 summary  ({len(results)}/{len(sample_infos)} evaluated)")
    print("-" * 60)
    print(f"  DER   — Mean {np.mean(ders)*100:6.2f}%  Median {np.median(ders)*100:6.2f}%  "
          f"Min {np.min(ders)*100:5.2f}%  Max {np.max(ders)*100:5.2f}%")
    if cpwers:
        print(f"  cpWER — Mean {np.mean(cpwers)*100:6.2f}%  Median {np.median(cpwers)*100:6.2f}%  "
              f"Min {np.min(cpwers)*100:5.2f}%  Max {np.max(cpwers)*100:5.2f}%  "
              f"({len(cpwers)} samples)")
    print("=" * 60)

    if args.out:
        summary = {
            "folder": args.folder,
            "num_samples": len(sample_infos),
            "num_evaluated": len(results),
            "der_mean": float(np.mean(ders)),
            "der_median": float(np.median(ders)),
            "cpwer_mean": float(np.mean(cpwers)) if cpwers else None,
            "cpwer_median": float(np.median(cpwers)) if cpwers else None,
            "num_cpwer": len(cpwers),
            "results": results,
        }
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
