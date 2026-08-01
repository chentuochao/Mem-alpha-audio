#!/usr/bin/env python3
"""Publication figures for the audio-memory probe sweep.

Reuses the data loader in compare_probes.py (same FOLDERS dict) and renders:

  fig1_cascade           stacked stage-loss bars: where accuracy is lost
                         (audio 100-T, construction T-C, retrieval C-S, kept S;
                          E2E overlaid as a marker)
  fig2_double_dissoc     (a) transcript ceiling T by noise (compression-invariant)
                         (b) construction retention C/T vs measured ratio
                             -> lines collapse across noise (compression-driven)
  fig3_e2e_curve         E2E vs measured compression ratio, one line per noise
                         condition, with safe/cliff zones
  fig4_memory_dynamics   self-correction (T✗→C✓) vs memory-loss (T✓→C✗)
  fig5_c_probe_curve     C-probe accuracy (constructed memory) vs measured ratio

Run with an env that has matplotlib (e.g. the `mem` env):
  ~/miniconda3/envs/mem/bin/python diagnostic/plot_probe_figures.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_probes import FOLDERS, analyze, find_probe_files  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# ---- presentation constants -------------------------------------------------
NOISE_ORDER = ["Clean", "BG_SNR5", "BG_SNR0", "interf_SNR10", "interf_SNR5"]
NOISE_LABEL = {"Clean": "Clean", "BG_SNR5": "BG SNR5", "BG_SNR0": "BG SNR0",
               "interf_SNR10": "Interf SNR10", "interf_SNR5": "Interf SNR5"}
NOISE_COLOR = {"Clean": "#1f77b4", "BG_SNR5": "#f0b429", "BG_SNR0": "#9c6b0b",
               "interf_SNR10": "#e8735a", "interf_SNR5": "#b3121f"}
COMP_ORDER = ["base", "x3", "x4"]
COMP_MARK = {"base": "o", "x3": "s", "x4": "^", "x5": "D"}


def parse_label(label):
    """'BG_SNR5_x4' -> ('BG_SNR5', 'x4');  'Clean' -> ('Clean', 'base')."""
    m = re.search(r"_(x[\d.]+)$", label)
    if m:
        return label[: m.start()], m.group(1)
    return label, "base"


def load(pairs=None):
    """Return {label: metrics-dict} with derived %s, ordered by noise then comp.
    `pairs` is a list of (label, folder); defaults to the hardcoded FOLDERS."""
    if pairs is None:
        pairs = list(FOLDERS.items())
    recs = {}
    for label, path in pairs:
        if not find_probe_files(path):
            print(f"[skip] no error_probe.json under {path}")
            continue
        m = analyze(path, label=label)
        noise, comp = parse_label(label)
        ca = m["cross_all"]; appl = ca["applicable"] or 1
        recs[label] = {
            "noise": noise, "comp": comp,
            "ratio": m["comp_ratio"],
            "T": m["t_all"][2] * 100, "C": m["c_all"][2] * 100,
            "S": m["s_all"][2] * 100, "E2E": m["acc"] * 100,
            "self": ca["self_correction"] / appl * 100,
            "loss": ca["memory_loss"] / appl * 100,
            # raw T×S confusion counts
            "TcSc": ca["both_right"], "TwSc": ca["self_correction"],
            "TcSw": ca["memory_loss"], "TwSw": ca["both_wrong"], "n": appl,
        }
    # stable order: by noise family, then compression strength
    order = sorted(recs, key=lambda k: (NOISE_ORDER.index(recs[k]["noise"])
                                        if recs[k]["noise"] in NOISE_ORDER else 99,
                                        COMP_ORDER.index(recs[k]["comp"])
                                        if recs[k]["comp"] in COMP_ORDER else 99))
    return {k: recs[k] for k in order}


def save(fig, name):
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def _plot_e2e_vs_ratio(ax, recs, zones=True, title=None):
    """E2E (%) vs measured compression ratio, one line per noise condition."""
    groups = [n for n in NOISE_ORDER if any(recs[k]["noise"] == n for k in recs)]
    if zones:
        ax.axvspan(1.0, 2.5, color="#2a9d5c", alpha=0.07)
        ax.axvspan(3.0, 5.0, color="#c1121f", alpha=0.07)
        ax.text(1.75, 22, "safe\n(≤~2.5×)", ha="center", color="#2a9d5c", fontsize=8)
        ax.text(4.0, 22, "cliff\n(≥~3×)", ha="center", color="#c1121f", fontsize=8)
    for n in groups:
        ks = [k for k in recs if recs[k]["noise"] == n]
        ks.sort(key=lambda k: recs[k]["ratio"])
        ax.plot([recs[k]["ratio"] for k in ks], [recs[k]["E2E"] for k in ks],
                marker="o", color=NOISE_COLOR[n], label=NOISE_LABEL[n], lw=2)
    ax.set_xlabel("Measured compression ratio")
    ax.set_ylabel("End-to-end QA accuracy (%)")
    ax.set_ylim(20, 75)
    if title:
        ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)


def fig6_confusion_counts(recs):
    """Table of raw T×C confusion counts per condition:
    (1) T✓C✓  (2) T✗C✓ self-correction  (3) T✓C✗ memory loss  (4) T✗C✗."""
    import matplotlib.pyplot as plt
    labels = list(recs)
    col_names = ["T✓ C✓\n(both right)", "T✗ C✓\n(self-corr)",
                 "T✓ C✗\n(mem-loss)", "T✗ C✗\n(both wrong)", "Total"]
    cell_text = [[f"{recs[k]['TcSc']}", f"{recs[k]['TwSc']}",
                  f"{recs[k]['TcSw']}", f"{recs[k]['TwSw']}", f"{recs[k]['n']}"]
                 for k in labels]
    n_rows, n_cols = len(labels), len(col_names)
    fig, ax = plt.subplots(figsize=(1.6 + 1.7 * n_cols, 0.9 + 0.42 * n_rows))
    ax.axis("off")
    ax.set_title("T×C confusion — sample counts (all QAs)", fontweight="bold", pad=12)
    tbl = ax.table(cellText=cell_text, rowLabels=labels, colLabels=col_names,
                   cellLoc="center", rowLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    # tint the two asymmetric-transition columns (self-corr green, mem-loss red)
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
    save(fig, "fig6_confusion_counts")


def fig5_c_probe_curve(recs):
    """C-probe accuracy (constructed memory) vs measured compression ratio."""
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [n for n in NOISE_ORDER if any(recs[k]["noise"] == n for k in recs)]
    ax.axvspan(1.0, 2.5, color="#2a9d5c", alpha=0.07)
    ax.axvspan(3.0, 5.0, color="#c1121f", alpha=0.07)
    ax.text(1.75, 26, "safe\n(≤~2.5×)", ha="center", color="#2a9d5c", fontsize=8)
    ax.text(4.0, 26, "cliff\n(≥~3×)", ha="center", color="#c1121f", fontsize=8)
    for n in groups:
        ks = [k for k in recs if recs[k]["noise"] == n]
        ks.sort(key=lambda k: recs[k]["ratio"])
        ax.plot([recs[k]["ratio"] for k in ks], [recs[k]["C"] for k in ks],
                marker="o", color=NOISE_COLOR[n], label=NOISE_LABEL[n], lw=2)
    ax.set_xlabel("Measured compression ratio")
    ax.set_ylabel("C-probe accuracy — constructed memory (%)")
    ax.set_ylim(20, 90)
    ax.set_title("Construction quality vs compression ratio", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig5_c_probe_curve")


def fig1_cascade(recs):
    labels = list(recs)
    # x positions with a gap between noise families
    xs, x, prev = [], 0, None
    for k in labels:
        n = recs[k]["noise"]
        if prev is not None and n != prev:
            x += 0.8
        xs.append(x); x += 1; prev = n
    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=(max(13, 0.7 * len(labels) + 7), 5.2),
        gridspec_kw={"width_ratios": [2.4, 1]})
    kept = [recs[k]["S"] for k in labels]
    retr = [recs[k]["C"] - recs[k]["S"] for k in labels]
    cons = [recs[k]["T"] - recs[k]["C"] for k in labels]
    audio = [100 - recs[k]["T"] for k in labels]
    b = [0] * len(labels)
    ax.bar(xs, kept, 0.8, bottom=b, color="#2a9d5c", label="Kept (S = retrieved)")
    b = [i + j for i, j in zip(b, kept)]
    ax.bar(xs, retr, 0.8, bottom=b, color="#cfd8dc", label="Retrieval loss (C−S)")
    b = [i + j for i, j in zip(b, retr)]
    ax.bar(xs, cons, 0.8, bottom=b, color="#e08b1e", label="Construction loss (T−C)")
    b = [i + j for i, j in zip(b, cons)]
    ax.bar(xs, audio, 0.8, bottom=b, color="#546069", hatch="//",
           edgecolor="white", label="Audio loss (100−T)")
    # E2E as a black marker (real run tracks S)
    ax.scatter(xs, [recs[k]["E2E"] for k in labels], color="black", zorder=5,
               s=28, marker="D", label="E2E (real run)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{recs[k]['comp']}\n{recs[k]['ratio']:.1f}×" for k in labels],
                       fontsize=8)
    # family band labels
    for n in NOISE_ORDER:
        idx = [i for i, k in enumerate(labels) if recs[k]["noise"] == n]
        if idx:
            ax.text(sum(xs[i] for i in idx) / len(idx), -14, NOISE_LABEL[n],
                    ha="center", va="top", fontweight="bold", fontsize=9)
    ax.set_ylim(0, 100); ax.set_ylabel("QA accuracy (%)")
    ax.set_title("(a) Where accuracy is lost: audio → construction → retrieval",
                 fontweight="bold")
    # ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.95)
    ax.margins(x=0.01)
    # right panel: E2E vs measured compression ratio
    _plot_e2e_vs_ratio(axr, recs, title="(b) E2E vs compression ratio")
    fig.subplots_adjust(bottom=0.20)
    fig.tight_layout()
    save(fig, "fig1_cascade")


def fig2_double_dissoc(recs):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) T by noise family, bars per compression -> T flat within group
    groups = [n for n in NOISE_ORDER if any(recs[k]["noise"] == n for k in recs)]
    comps = COMP_ORDER
    w = 0.8 / len(comps)
    for gi, n in enumerate(groups):
        for ci, comp in enumerate(comps):
            k = next((k for k in recs if recs[k]["noise"] == n
                      and recs[k]["comp"] == comp), None)
            if k is None:
                continue
            axL.bar(gi + ci * w, recs[k]["T"], w, color=NOISE_COLOR[n],
                    alpha=0.45 + 0.18 * ci, edgecolor="white")
    axL.set_xticks([i + 0.3 for i in range(len(groups))])
    axL.set_xticklabels([NOISE_LABEL[n] for n in groups], fontsize=9)
    axL.set_ylim(0, 100); axL.set_ylabel("Transcript ceiling  T (%)")
    axL.set_title("(a) Audio noise sets T — flat across compression", fontweight="bold")
    axL.legend(handles=[Patch(fc="grey", alpha=a, label=c)
                        for c, a in zip(comps, [0.45, 0.63, 0.81, 0.99])],
               title="compression", fontsize=8, title_fontsize=8)

    # (b) C/T vs measured ratio, one line per noise -> lines collapse
    for n in groups:
        ks = [k for k in recs if recs[k]["noise"] == n]
        ks.sort(key=lambda k: recs[k]["ratio"])
        axR.plot([recs[k]["ratio"] for k in ks],
                 [recs[k]["C"] / recs[k]["T"] for k in ks],
                 marker="o", color=NOISE_COLOR[n], label=NOISE_LABEL[n], lw=2)
    axR.axhline(0.6, ls="--", color="grey", lw=1)
    axR.axvline(3.0, ls=":", color="red", lw=1)
    axR.text(3.02, 0.95, "knee ≈ 3×", color="red", fontsize=8)
    axR.set_xlabel("Measured compression ratio"); axR.set_ylabel("Construction retention  C/T")
    axR.set_ylim(0.3, 1.05)
    axR.set_title("(b) Compression sets C/T — noise-agnostic", fontweight="bold")
    axR.legend(fontsize=8)
    fig.suptitle("Double dissociation: noise → T, compression → C", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig2_double_dissoc")


def fig3_e2e_curve(recs):
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_e2e_vs_ratio(ax, recs,
                       title="Compression operating curve (interference sits lowest)")
    fig.tight_layout()
    save(fig, "fig3_e2e_curve")


def fig4_memory_dynamics(recs):
    # GT_Clean (ceiling, no ASR) and BG_SNR10 excluded from this view.
    exclude = {"GT_Clean", "BG_SNR10"}
    labels = [k for k in recs if k not in exclude]
    xs, x, prev = [], 0, None
    for k in labels:
        n = recs[k]["noise"]
        if prev is not None and n != prev:
            x += 0.8
        xs.append(x); x += 1; prev = n
    fig, ax = plt.subplots(figsize=(max(9, 0.7 * len(labels) + 3), 4.8))
    ax.bar(xs, [recs[k]["self"] for k in labels], 0.8, color="#2a9d5c",
           label="Self-correction  T✗→C✓  (construction recovers)")
    ax.bar(xs, [-recs[k]["loss"] for k in labels], 0.8, color="#c1121f",
           label="Memory loss  T✓→C✗  (construction drops)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{recs[k]['comp']}\n{recs[k]['ratio']:.1f}×" for k in labels],
                       fontsize=8)
    for n in NOISE_ORDER:
        idx = [i for i, k in enumerate(labels) if recs[k]["noise"] == n]
        if idx:
            ax.text(sum(xs[i] for i in idx) / len(idx), ax.get_ylim()[0],
                    NOISE_LABEL[n], ha="center", va="top", fontweight="bold", fontsize=9)
    ax.set_ylabel("% of QAs  (up = recovered, down = lost)")
    ax.set_title("Memory dynamics: denoiser at low compression, net-lossy overall",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    fig.subplots_adjust(bottom=0.20)
    save(fig, "fig4_memory_dynamics")


def main():
    import argparse
    from compare_probes import discover
    ap = argparse.ArgumentParser(description="probe figures")
    ap.add_argument("--scan", nargs="?", const="agents", default=None, metavar="ROOT",
                    help="use ALL folders with error_probe.json under ROOT "
                         "(default 'agents') instead of the hardcoded FOLDERS")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    recs = load(discover(args.scan) if args.scan else None)
    if not recs:
        sys.exit("no probe data found")
    print(f"loaded {len(recs)} conditions")
    fig1_cascade(recs)
    fig2_double_dissoc(recs)
    fig3_e2e_curve(recs)
    fig4_memory_dynamics(recs)
    fig5_c_probe_curve(recs)
    fig6_confusion_counts(recs)


if __name__ == "__main__":
    main()
