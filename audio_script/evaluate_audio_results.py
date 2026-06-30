"""
Plot and evaluate diarization predictions against ground-truth.

Usage:
    python plot_diarization.py path/to/step1_largepad/T

    # optionally override frame duration:
    python plot_diarization.py path/to/folder --frame_duration 0.08

Recursively finds every sample_info.json under the given folder and, for each:
  - Computes DER (diarization error rate) and saves a diarization_plot.png
  - Computes cpWER (concatenated min-permutation WER) when transcript files exist

Output:
    - PNG figure per sample: GT vs Pred (aligned) diarization
    - Printed DER + cpWER breakdown per sample and batch summary
"""

import argparse
import json
import math
import os
import sys
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from audio_script.eval.multitalker_metrics import compute_der_bruteforce, calculate_session_cpWER, normalize_string
from audio_script.eval.eval_utils import build_speaker_transcripts, eval_cpwer_seamlessinteraction


# ── Inline helpers so the script is self-contained ────────────────────────────

def load_vad_json(path: str) -> List[Dict]:
    """Load a VAD file (plain JSON array or JSONL) → [{start, end}, ...]"""
    with open(path) as f:
        text = f.read().strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        entries = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        return entries


def vad_to_binary(segments: List[Dict], total_frames: int,
                  frame_duration: float = 0.08) -> np.ndarray:
    """Convert [{start, end}] VAD segments to a binary frame vector."""
    binary = np.zeros(total_frames, dtype=np.float32)
    for seg in segments:
        s = int(seg["start"] / frame_duration)
        e = int(seg["end"] / frame_duration)
        e = min(e, total_frames)
        if s < total_frames:
            binary[s:e] = 1.0
    return binary


def _der_for_perm(pred_aligned: np.ndarray, gt: np.ndarray,
                  frame_duration: float) -> Tuple[float, Dict]:
    """DER arithmetic given an already-permuted prediction matrix (N, T)."""
    ref_speech = gt.sum(axis=0) > 0
    sys_speech = pred_aligned.sum(axis=0) > 0
    miss = np.logical_and(ref_speech, ~sys_speech).sum()
    fa = np.logical_and(~ref_speech, sys_speech).sum()
    speech_both = np.logical_and(ref_speech, sys_speech)
    speaker_err = np.sum(gt != pred_aligned, axis=0)
    conf = np.sum(speech_both & (speaker_err > 0))
    acc_err = np.sum(gt != pred_aligned) / (gt.shape[0] * gt.shape[1])
    total_ref = ref_speech.sum()
    miss_s = miss * frame_duration
    fa_s = fa * frame_duration
    conf_s = conf * frame_duration
    total_s = total_ref * frame_duration
    der = (miss_s + fa_s + conf_s) / total_s if total_s > 0 else 0.0
    return der, {"miss": miss_s, "fa": fa_s, "conf": conf_s,
                 "total": total_s, "acc_err": acc_err}


# ── cpWER helper ──────────────────────────────────────────────────────────────


def compute_cpwer(pred_word_list: Dict, gt_word_list: Dict) -> Tuple[float, List[str]]:
    """
    Compute cpWER between predicted and GT word-list dicts.

    Both arguments use the format {speaker_id: [{word, start, end, ...}, ...]}.
    Returns (cpwer_float, best_perm_speaker_ids).
    """
    hyp_texts, pred_speakers = build_speaker_transcripts(pred_word_list, pad_char = " ")
    ref_texts, _gt_speakers = build_speaker_transcripts(gt_word_list, pad_char = " ")

    if not hyp_texts or not ref_texts:
        return float("nan"), []

    # print(hyp_texts)
    # print(ref_texts)
    cpwer, _, _, best_perm_idx = calculate_session_cpWER(hyp_texts, ref_texts)
    best_perm = [pred_speakers[i] for i in best_perm_idx]
    return cpwer, best_perm


# ── Plotting ──────────────────────────────────────────────────────────────────

_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
           "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]


def _raster_ax(ax: plt.Axes, mat: np.ndarray, title: str,
               frame_duration: float, speaker_labels: Optional[List[str]] = None):
    """Draw a diarization binary matrix as a raster plot on *ax*."""
    N, T = mat.shape
    ax.set_facecolor("#f8f8f8")
    time = np.arange(T) * frame_duration  # seconds

    for spk_idx in range(N):
        label = speaker_labels[spk_idx] if speaker_labels else f"Spk {spk_idx}"
        color = _COLORS[spk_idx % len(_COLORS)]
        active = mat[spk_idx].astype(bool)
        # draw filled bars for active segments
        starts = np.where(np.diff(np.concatenate([[False], active, [False]])
                                  .astype(int)) == 1)[0]
        ends = np.where(np.diff(np.concatenate([[False], active, [False]])
                                .astype(int)) == -1)[0]
        for s, e in zip(starts, ends):
            ax.barh(spk_idx, (e - s) * frame_duration,
                    left=s * frame_duration,
                    height=0.6, color=color, alpha=0.85)

    ax.set_yticks(range(N))
    ax.set_yticklabels(speaker_labels if speaker_labels else
                       [f"Spk {i}" for i in range(N)], fontsize=9)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_xlim(0, T * frame_duration)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.invert_yaxis()


def plot_and_eval(pred_npy: str,
                  vads: List,
                  frame_duration: float = 0.08,
                  output_path: str = "diarization_plot.png",
                  speaker_names: Optional[Tuple[str, str]] = None):
    """
    Load predictions + GT, compute DER, and save a 3-panel figure.

    Panel layout:
        [GT] | [Pred (raw)] | [Pred (aligned to GT)]
    """
    # ── Load pred ─────────────────────────────────────────────────────
    pred_raw = np.load(pred_npy)  # (T, N_pred)
    if pred_raw.ndim == 1:
        pred_raw = pred_raw[:, np.newaxis]
    T, N_pred = pred_raw.shape
    pred_mat = pred_raw.T  # (N_pred, T)

    # ── Build GT matrix ───────────────────────────────────────────────
    # gt_spk1 = vad_to_binary(vad1, T, frame_duration)
    # gt_spk2 = vad_to_binary(vad2, T, frame_duration)
    # gt_mat = np.stack([gt_spk1, gt_spk2], axis=0)  # (2, T)

    gt_mat = []
    for vad in vads:
        gt_spk = vad_to_binary(vad, T, frame_duration)
        gt_mat.append(gt_spk)
    gt_mat = np.stack(gt_mat, axis=0)  # (2, T)
    N_gt, T = gt_mat.shape
    # ── Compute DER ───────────────────────────────────────────────────
    # Binarize pred (threshold 0.5)
    pred_bin = (pred_mat >= 0.5).astype(np.float32)
    valid_speakers = int((pred_bin.sum(axis=1) > 0).sum())
    print(f"Valid speakers (at least 1 active frame): {valid_speakers}", N_pred, N_gt)


    if N_gt > N_pred:
        pred_padded = np.zeros((N_gt, T), dtype=pred_bin.dtype)
        pred_padded[:N_pred] = pred_bin
    else:
        pred_padded = pred_bin

    der, details = compute_der_bruteforce(pred_padded, gt_mat, frame_duration)
    col_ind = details["col_ind"]           # best permutation of pred rows
    # col_ind
    pred_aligned = pred_padded[list(col_ind)] # (N_gt, T) reordered

    # ── Print summary ─────────────────────────────────────────────────
    print("=" * 60)
    print(f"  Prediction : {pred_npy}")
    # print(f"  GT VAD 1   : {vad1_path}")
    # print(f"  GT VAD 2   : {vad2_path}")
    print(f"  Frames     : {T}  |  frame_duration: {frame_duration} s")
    print(f"  Duration   : {T * frame_duration:.1f} s")
    print("-" * 60)
    print(f"  DER        : {der * 100:.2f} %")
    print(f"  Miss       : {details['miss']:.2f} s")
    print(f"  False Alarm: {details['fa']:.2f} s")
    print(f"  Confusion  : {details['conf']:.2f} s")
    print(f"  Total Ref  : {details['total']:.2f} s")
    print(f"  Frame Acc  : {(1 - details['acc_err']) * 100:.2f} %")
    print(f"  Best perm  : pred rows {list(col_ind)}")
    print("=" * 60)

    # ── Plot ──────────────────────────────────────────────────────────
    if speaker_names:
        gt_labels = list(speaker_names)
        pred_raw_labels = [f"Pred Spk {i}" for i in range(N_pred)]
        pred_aligned_labels = [f"Pred→{speaker_names[i]}" for i in range(len(col_ind))]
    else:
        gt_labels = [f"GT Spk {i}" for i in range(gt_mat.shape[0])]
        pred_raw_labels = [f"Pred Spk {i}" for i in range(N_pred)]
        pred_aligned_labels = [f"Pred (aligned) Spk {i}" for i in range(len(col_ind))]

    n_rows = gt_mat.shape[0]
    fig_h = max(4, 1.2 * max(n_rows, N_pred))
    fig, axes = plt.subplots(2, 1, figsize=(20, fig_h),
                             constrained_layout=True)

    _raster_ax(axes[0], gt_mat, "Ground Truth", frame_duration, gt_labels)
    # _raster_ax(axes[1], pred_bin, "Prediction (raw)", frame_duration,
    #            pred_raw_labels)
    _raster_ax(axes[1], pred_aligned,
               f"Prediction (aligned to GT)  —  DER = {der * 100:.2f}%",
               frame_duration, pred_aligned_labels)

    # Match colors between GT and aligned-pred panels
    handles = []
    for i in range(gt_mat.shape[0]):
        lbl = gt_labels[i] if speaker_names else f"Speaker {i}"
        handles.append(mpatches.Patch(color=_COLORS[i % len(_COLORS)],
                                      label=lbl))
    fig.legend(handles=handles, loc="upper right", fontsize=9,
               title="Speaker", framealpha=0.8)

    plt.suptitle(
        f"Diarization  |  DER={der*100:.2f}%  "
        f"(Miss={details['miss']:.1f}s  FA={details['fa']:.1f}s  "
        f"Conf={details['conf']:.1f}s)",
        fontsize=11, fontweight="bold", y=1.02,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {output_path}")
    return der, details


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot and evaluate diarization predictions vs GT for all "
                    "sample_info.json files found under a folder."
    )
    parser.add_argument(
        "folder",
        metavar="DIR",
        help="Root folder to search recursively for sample_info.json files. "
             "A diarization_plot.png is saved next to each one found.",
    )
    parser.add_argument(
        "--frame_duration",
        type=float,
        default=None,
        help="Duration of each frame in seconds (default: read from "
             "sample_info.json or 0.08).",
    )
    args = parser.parse_args()

    root = args.folder
    if not os.path.isdir(root):
        print(f"ERROR: folder path is not a directory: {root}")
        sys.exit(1)

    sample_infos = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "sample_info.json" in filenames:
            sample_infos.append(os.path.join(dirpath, "sample_info.json"))
    sample_infos.sort()

    if not sample_infos:
        print(f"No sample_info.json files found under {root}")
        sys.exit(1)

    print(f"Found {len(sample_infos)} sample(s) under {root}\n")

    results = []
    for idx, si_path in enumerate(sample_infos):
        print(f"[{idx}/{len(sample_infos)}] {si_path}")
        # if "Season02" not in si_path:
        #     continue
        try:
            with open(si_path) as f:
                info = json.load(f)
        except Exception as e:
            print(f"  [SKIP] Cannot read: {e}")
            continue

        sample_folder = os.path.dirname(si_path)
        pred_npy = os.path.join(sample_folder, "diart_pred.npy")
        if not os.path.exists(pred_npy):
            print(f"  [SKIP] Missing pred file: {pred_npy}")
            continue

        if "vad_path" not in info:
            vad1_path = info.get("vad1_path")
            vad2_path = info.get("vad2_path")
            if not vad1_path or not vad2_path:
                print(f"  [SKIP] No VAD paths in {si_path}")
                continue
            vads = [load_vad_json(vad1_path), load_vad_json(vad2_path)]
        else:
            with open(info["vad_path"]) as f:
                vad_speaker = json.load(f)
            vads = list(vad_speaker.values())

        frame_dur = args.frame_duration or info.get("feat_len_sec", 0.08)

        spk_names: Optional[Tuple[str, str]] = None
        spk_pair = info.get("spk_pair", "")
        if spk_pair and "_" in spk_pair:
            parts = spk_pair.split("_", 1)
            spk_names = (parts[0], parts[1])

        out = os.path.join(sample_folder, "diarization_plot.png")

        der, cpwer = None, None
        # try:
        der, details = plot_and_eval(pred_npy, vads, frame_dur, out, spk_names)
        # except Exception as e:
        #     print(f"  [ERROR] DER: {e}")

        # ── cpWER ─────────────────────────────────────────────────────
        pred_trans_path = info.get("pred_transcript_path")
        gt_trans_path = info.get("transcript_path")  # bazinga: {spk: [words]}


        if pred_trans_path and os.path.exists(pred_trans_path):
            try:
                with open(pred_trans_path) as f:
                    pred_word_list = json.load(f)

                if gt_trans_path and os.path.exists(gt_trans_path):
                    with open(gt_trans_path) as f:
                        gt_word_list = json.load(f)

                    cpwer, best_perm = compute_cpwer(pred_word_list, gt_word_list)
                else:
                    print("  [SKIP cpWER] No GT transcript path in sample_info")

                if cpwer is not None:
                    if not math.isnan(cpwer):
                        print(f"  cpWER : {cpwer * 100:.2f} %  (best perm: {best_perm})")
            except Exception as e:
                print(f"  [ERROR] cpWER: {e}")
        else:
            print("  [SKIP cpWER] pred_transcript_path missing or not found")

        if der is not None:
            entry = {"path": si_path, "der": der, **details}
            if cpwer is not None:
                entry["cpwer"] = cpwer
            results.append(entry)

    if results:
        ders = [r["der"] for r in results]
        cpwers = [r["cpwer"] for r in results if "cpwer" in r and not math.isnan(r["cpwer"])]
        print("\n" + "=" * 60)
        print(f"  Batch summary  ({len(results)} / {len(sample_infos)} processed)")
        print(f"  DER  — Mean: {np.mean(ders) * 100:.2f}%  Median: {np.median(ders) * 100:.2f}%  "
              f"Min: {np.min(ders) * 100:.2f}%  Max: {np.max(ders) * 100:.2f}%")
        if cpwers:
            print(f"  cpWER— Mean: {np.mean(cpwers) * 100:.2f}%  Median: {np.median(cpwers) * 100:.2f}%  "
                  f"Min: {np.min(cpwers) * 100:.2f}%  Max: {np.max(cpwers) * 100:.2f}%  "
                  f"({len(cpwers)} samples)")
        print("=" * 60)


if __name__ == "__main__":
    main()
