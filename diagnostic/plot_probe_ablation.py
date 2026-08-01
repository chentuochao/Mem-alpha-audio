#!/usr/bin/env python3
"""Probe figures for a PIPELINE-ABLATION comparison (no noise / compression sweep).

Sibling of plot_probe_figures.py, but instead of grouping runs by audio noise
family + compression strength, the ONLY category here is the named pipeline
variant given in FOLDERS (e.g. Full vs. dropping name-extraction vs. dropping
global tracking). Each FOLDERS entry becomes one x-axis category; there is no
SNR grouping and no compression-ratio axis, so the ratio-curve figures
(fig2/3/5 in plot_probe_figures.py) are intentionally omitted.

Reuses the same data loader as compare_probes.py (analyze / find_probe_files),
which reads the per-question error_probe.json probe_errors.py writes, pooling all
instance/seed subdirs of each folder.

Figures written to diagnostic/figures_ablation/:
  fig_probe_bars       grouped T / C / S / E2E accuracy, one cluster per variant
  fig_cascade          stacked stage-loss bar per variant (audio 100-T,
                         construction T-C, retrieval C-S, kept S; E2E overlaid)
  fig_memory_dynamics  self-correction (T✗→C✓) vs memory-loss (T✓→C✗) per variant
  fig_confusion_counts T×C confusion sample-count table per variant

Run with an env that has matplotlib (e.g. the `mem` env):
  ~/miniconda3/envs/mem/bin/python diagnostic/plot_probe_ablation.py
"""
import os
import sys
import json
import argparse
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_probes import analyze, find_probe_files  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_ablation")

# ---- input: named pipeline variants -> probed run folder --------------------
# Each key is a category on the x-axis (order preserved). Override at runtime
# with --folders '{"name": "path", ...}'.
FOLDERS = {
    "ASR+local_diarization+globaltracking+name_extraction(Full)": "/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_Season01_Clean_Anoy_no_thinking_tokens_2048",
    "ASR+local_diarization+globaltracking": "/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_anon_global_Season01_Clean_no_thinking_tokens_2048",
    "ASR+local_diarization": "/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_anon_local_Season01_Clean_no_thinking_tokens_2048",
}

# ---- presentation constants -------------------------------------------------
# One distinct colour per variant (cycled if more variants than colours).
METHOD_COLORS = ["#1f77b4", "#e08b1e", "#2a9d5c", "#b3121f", "#6a3d9a", "#17becf"]
# Probe-bar palette (T / C / S / E2E), shared across variants.
PROBE_COLORS = {"T": "#546069", "C": "#e08b1e", "S": "#2a9d5c", "E2E": "#111111"}


def wrap_label(s, width=16):
    """Break a long '+'-joined variant name onto multiple lines for the x-axis."""
    return "\n".join(textwrap.wrap(s.replace("+", " + "), width=width,
                                   break_long_words=False)) or s


def load(pairs):
    """(label, folder) list -> {label: metrics} in the given order.

    Only the fields that make sense without a noise/compression axis are kept;
    all come straight from compare_probes.analyze (pooled over instances/seeds)."""
    recs = {}
    for label, path in pairs:
        if not find_probe_files(path):
            print(f"[skip] no error_probe.json under {path}")
            continue
        m = analyze(path, label=label)
        ca = m["cross_all"]; appl = ca["applicable"] or 1
        recs[label] = {
            "folder": path, "n": m["total"], "appl": ca["applicable"],
            "T": m["t_all"][2] * 100, "C": m["c_all"][2] * 100,
            "S": m["s_all"][2] * 100, "E2E": m["acc"] * 100,
            "self": ca["self_correction"] / appl * 100,
            "loss": ca["memory_loss"] / appl * 100,
            "TcSc": ca["both_right"], "TwSc": ca["self_correction"],
            "TcSw": ca["memory_loss"], "TwSw": ca["both_wrong"],
        }
    return recs


def save(fig, name):
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_probe_bars(recs):
    """Grouped bars: T / C / S / E2E accuracy, one cluster per variant."""
    labels = list(recs)
    metrics = [("T", "T-probe (transcript)"), ("C", "C-probe (construction)"),
               ("S", "S-probe (retr. subset)"), ("E2E", "E2E (real run)")]
    n = len(labels); w = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(7, 2.4 * n), 5.0))
    for mi, (key, name) in enumerate(metrics):
        xs = [i + mi * w for i in range(n)]
        vals = [recs[k][key] for k in labels]
        bars = ax.bar(xs, vals, w, color=PROBE_COLORS[key], label=name,
                      edgecolor="white")
        for x, v in zip(xs, vals):
            ax.text(x, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks([i + 0.4 - w / 2 for i in range(n)])
    ax.set_xticklabels([wrap_label(k) for k in labels], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Probe accuracy by pipeline variant", fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.95)
    ax.margins(x=0.02)
    fig.tight_layout()
    save(fig, "fig_probe_bars")


def fig_cascade(recs):
    """Stacked stage-loss bar per variant: audio (100-T) -> construction (T-C)
    -> retrieval (C-S) -> kept (S), with E2E overlaid as a marker."""
    labels = list(recs)
    xs = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(7, 2.4 * len(labels)), 5.4))
    kept = [recs[k]["S"] for k in labels]
    retr = [recs[k]["C"] - recs[k]["S"] for k in labels]
    cons = [recs[k]["T"] - recs[k]["C"] for k in labels]
    audio = [100 - recs[k]["T"] for k in labels]
    b = [0] * len(labels)
    ax.bar(xs, kept, 0.6, bottom=b, color="#2a9d5c", label="Kept (S = retrieved)")
    b = [i + j for i, j in zip(b, kept)]
    ax.bar(xs, retr, 0.6, bottom=b, color="#cfd8dc", label="Retrieval loss (C−S)")
    b = [i + j for i, j in zip(b, retr)]
    ax.bar(xs, cons, 0.6, bottom=b, color="#e08b1e", label="Construction loss (T−C)")
    b = [i + j for i, j in zip(b, cons)]
    ax.bar(xs, audio, 0.6, bottom=b, color="#546069", hatch="//",
           edgecolor="white", label="Audio loss (100−T)")
    ax.scatter(xs, [recs[k]["E2E"] for k in labels], color="black", zorder=5,
               s=40, marker="D", label="E2E (real run)")
    for x, k in zip(xs, labels):
        ax.text(x, recs[k]["E2E"] + 1.5, f"{recs[k]['E2E']:.0f}", ha="center",
                va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([wrap_label(k) for k in labels], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("QA accuracy (%)")
    ax.set_title("Where accuracy is lost: audio → construction → retrieval",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.95)
    ax.margins(x=0.02)
    fig.tight_layout()
    save(fig, "fig_cascade")


def fig_memory_dynamics(recs):
    """Self-correction (T✗→C✓, up) vs memory-loss (T✓→C✗, down) per variant."""
    labels = list(recs)
    xs = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(9, 2.4 * len(labels) + 3), 4.8))
    ax.bar(xs, [recs[k]["self"] for k in labels], 0.6, color="#2a9d5c",
           label="Self-correction  T✗→C✓  (construction recovers)")
    ax.bar(xs, [-recs[k]["loss"] for k in labels], 0.6, color="#c1121f",
           label="Memory loss  T✓→C✗  (construction drops)")
    ax.axhline(0, color="black", lw=0.8)
    for x, k in zip(xs, labels):
        ax.text(x, recs[k]["self"] + 0.3, f"{recs[k]['self']:.0f}%", ha="center",
                va="bottom", fontsize=8)
        ax.text(x, -recs[k]["loss"] - 0.3, f"{recs[k]['loss']:.0f}%", ha="center",
                va="top", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([wrap_label(k) for k in labels], fontsize=8)
    ax.set_ylabel("% of QAs  (up = recovered, down = lost)")
    ax.set_title("Memory dynamics: construction recovery vs loss", fontweight="bold")
    # Headroom so the legend (above the plot) never overlaps bars or value labels.
    top = max(recs[k]["self"] for k in labels)
    bot = max(recs[k]["loss"] for k in labels)
    ax.set_ylim(-bot * 1.25, top * 1.25)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False)
    fig.tight_layout()
    save(fig, "fig_memory_dynamics")


def fig_confusion_counts(recs):
    """Table of raw T×C confusion counts per variant."""
    labels = list(recs)
    col_names = ["T✓ C✓\n(both right)", "T✗ C✓\n(self-corr)",
                 "T✓ C✗\n(mem-loss)", "T✗ C✗\n(both wrong)", "Total"]
    cell_text = [[f"{recs[k]['TcSc']}", f"{recs[k]['TwSc']}", f"{recs[k]['TcSw']}",
                  f"{recs[k]['TwSw']}", f"{recs[k]['appl']}"] for k in labels]
    n_rows, n_cols = len(labels), len(col_names)
    fig, ax = plt.subplots(figsize=(3.0 + 1.7 * n_cols, 1.0 + 0.55 * n_rows))
    ax.axis("off")
    ax.set_title("T×C confusion — sample counts (all QAs)", fontweight="bold", pad=12)
    tbl = ax.table(cellText=cell_text,
                   rowLabels=[wrap_label(k, 22) for k in labels],
                   colLabels=col_names, cellLoc="center", rowLoc="center",
                   loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#40466e"); cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif c == -1:
            cell.set_facecolor("#d9e1f2"); cell.get_text().set_fontweight("bold")
        elif c == 1:
            cell.set_facecolor("#e3f2e8")   # self-correction column
        elif c == 2:
            cell.set_facecolor("#fbe3e3")   # memory-loss column
        elif r % 2 == 0:
            cell.set_facecolor("#f2f2f2")
    fig.tight_layout()
    save(fig, "fig_confusion_counts")


def main():
    ap = argparse.ArgumentParser(description="pipeline-ablation probe figures")
    ap.add_argument("--folders", dest="folders_json", default=None,
                    help='JSON object {"name": "path", ...} to override FOLDERS')
    args = ap.parse_args()

    folders = json.loads(args.folders_json) if args.folders_json else FOLDERS
    pairs = list(folders.items())

    os.makedirs(OUT, exist_ok=True)
    recs = load(pairs)
    if not recs:
        sys.exit("no probe data found (no error_probe.json under any folder)")
    print(f"loaded {len(recs)} variant(s): {', '.join(recs)}")

    fig_probe_bars(recs)
    fig_cascade(recs)
    fig_memory_dynamics(recs)
    fig_confusion_counts(recs)


if __name__ == "__main__":
    main()
