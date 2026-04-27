"""
Plot and evaluate diarization predictions against ground-truth.

Usage (standalone):
    python plot_diarization.py \
        --pred_npy  path/to/diart_pred.npy \
        --vad1      path/to/vad1.json \
        --vad2      path/to/vad2.json \
        --frame_duration 0.08 \
        --output    diarization_plot.png

    # Or point at a sample_info.json produced by step1_diarize_asr.py:
    python plot_diarization.py --sample_info path/to/sample_info.json

Output:
    - PNG figure: 3-panel plot (GT | Pred raw | Pred aligned to GT)
    - Printed DER breakdown (miss / FA / confusion)
"""

import argparse
import json
import os
import sys
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


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


def compute_der_bruteforce(pred: np.ndarray, gt: np.ndarray,
                           frame_duration: float = 0.08
                           ) -> Tuple[float, Dict]:
    """
    Brute-force DER: try all permutations of pred rows, pick the best.

    Args:
        pred: (N_pred, T) binary prediction matrix
        gt:   (N_gt,   T) binary ground-truth matrix
    Returns:
        best_der, best_details  (details includes 'col_ind' for reordering)
    """
    N_pred = pred.shape[0]
    N_gt = gt.shape[0]
    best_der = float("inf")
    best_details: Optional[Dict] = None

    for perm in permutations(range(N_pred), N_gt):
        pred_aligned = pred[list(perm)]
        der, details = _der_for_perm(pred_aligned, gt, frame_duration)
        if der < best_der:
            best_der = der
            best_details = details
            best_details["col_ind"] = np.array(perm)

    return best_der, best_details  # type: ignore[return-value]


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
                  vad1_path: str,
                  vad2_path: str,
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
    vad1 = load_vad_json(vad1_path)
    vad2 = load_vad_json(vad2_path)
    gt_spk1 = vad_to_binary(vad1, T, frame_duration)
    gt_spk2 = vad_to_binary(vad2, T, frame_duration)
    gt_mat = np.stack([gt_spk1, gt_spk2], axis=0)  # (2, T)

    # ── Compute DER ───────────────────────────────────────────────────
    # Binarize pred (threshold 0.5)
    pred_bin = (pred_mat >= 0.5).astype(np.float32)
    der, details = compute_der_bruteforce(pred_bin, gt_mat, frame_duration)

    col_ind = details["col_ind"]           # best permutation of pred rows
    pred_aligned = pred_bin[list(col_ind)] # (N_gt, T) reordered

    # ── Print summary ─────────────────────────────────────────────────
    print("=" * 60)
    print(f"  Prediction : {pred_npy}")
    print(f"  GT VAD 1   : {vad1_path}")
    print(f"  GT VAD 2   : {vad2_path}")
    print(f"  Frames     : {T}  |  frame_duration: {frame_duration} s")
    print(f"  Duration   : {T * frame_duration:.1f} s")
    print("-" * 60)
    print(f"  DER        : {der * 100:.2f} %")
    print(f"  Miss       : {details['miss']:.2f} s")
    print(f"  False Alarm: {details['fa']:.2f} s")
    print(f"  Confusion  : {details['conf']:.2f} s")
    print(f"  Total Ref  : {details['total']:.2f} s")
    print(f"  Frame Acc  : {(1 - details['acc_err']) * 100:.2f} %")
    print(f"  Best perm  : pred rows {list(col_ind)} → gt rows [0,1]")
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
    fig, axes = plt.subplots(3, 1, figsize=(20, fig_h),
                             constrained_layout=True)

    _raster_ax(axes[0], gt_mat, "Ground Truth", frame_duration, gt_labels)
    _raster_ax(axes[1], pred_bin, "Prediction (raw)", frame_duration,
               pred_raw_labels)
    _raster_ax(axes[2], pred_aligned,
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
        description="Plot and evaluate diarization predictions vs GT."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sample_info",
        metavar="PATH",
        help="Path to sample_info.json produced by step1_diarize_asr.py. "
             "All other paths are inferred from it automatically.",
    )
    parser.add_argument(
        "--frame_duration",
        type=float,
        default=None,
        help="Duration of each frame in seconds (default: read from "
             "sample_info.json or 0.08).",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Output PNG path. Defaults to <pred_npy_dir>/diarization_plot.png.",
    )
    args = parser.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────
    with open(args.sample_info) as f:
        info = json.load(f)
    sample_folder = os.path.dirname(args.sample_info)
    pred_npy = info.get("diart_path") or os.path.join(
        sample_folder, "diart_pred.npy"
    )
    vad1 = info["vad1_path"]
    vad2 = info["vad2_path"]
    frame_dur = args.frame_duration or info.get("feat_len_sec", 0.08)
    spk_pair = info.get("spk_pair", "")
    spk_names: Optional[Tuple[str, str]] = None
    if spk_pair and "_" in spk_pair:
        parts = spk_pair.split("_", 1)
        spk_names = (parts[0], parts[1])
    out = args.output or os.path.join(
        sample_folder, "diarization_plot.png"
    )

    # ── Run ───────────────────────────────────────────────────────────
    plot_and_eval(pred_npy, vad1, vad2, frame_dur, out, spk_names)


if __name__ == "__main__":
    main()
