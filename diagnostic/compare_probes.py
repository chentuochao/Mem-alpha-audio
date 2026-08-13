#!/usr/bin/env python3
"""Compare error-probe results across multiple probed agent folders.

Reads the per-question `error_probe.json` written by probe_errors.py for each
folder (searching the folder itself, its immediate instance subdirs like 0/, 1/,
..., AND per-seed instance subdirs seed*/0/), then reports, per folder and
side-by-side (findings from all instances/seeds of a run are POOLED):

  * original QA accuracy
  * T / C / S probe pass rates, on BOTH the failed subset and all QAs
    (same passed/ran/% convention as probe_errors.py's summary)
  * construction dynamics, T vs C, over the full QA set:
      - self-correction : QA wrong at T (transcript) but correct at C (constructed
                          memory)  -> construction recovered the answer
      - memory loss     : QA correct at T but wrong at C
                          -> construction dropped an answer the transcript had

Usage:
  python diagnostic/compare_probes.py <folder1> <folder2> ...
  python diagnostic/compare_probes.py --folders '["<folder1>", "<folder2>"]'
  python diagnostic/compare_probes.py <folders...> --csv out.csv
"""
import argparse
import csv as csvmod
import glob
import json
import os
import re
import statistics

# Hardcoded default folders to compare when none are passed on the command line.
# Edit this list; or override at runtime with positional args / --folders '[...]'.
# FOLDERS = {
#     "Clean": "/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_Season01_Clean_Anoy_no_thinking_tokens_2048",
#     "Clean_x3": "/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_Season01_Clean_Anoy_no_thinking_tokens_2048_comp_x3",
#     "Clean_x4": "/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_Season01_Clean_Anoy_no_thinking_tokens_2048_comp_x4",
#     "BG_SNR5": "/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_Season01_SNR5_Anoy_no_thinking_tokens_2048",
#     "BG_SNR5_x3": "/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_Season01_SNR5_Anoy_no_thinking_tokens_2048_comp_x3",
#     "BG_SNR5_x4": "/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_Season01_SNR5_Anoy_no_thinking_tokens_2048_comp_x4",
#     "BG_SNR0": "/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_Season01_SNR0_Anoy_no_thinking_tokens_2048",
#     "BG_SNR0_x3": "/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_Season01_SNR0_Anoy_no_thinking_tokens_2048_comp_x3",
#     "BG_SNR0_x4": "/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_Season01_SNR0_Anoy_no_thinking_tokens_2048_comp_x4",
#     "interf_SNR10":"/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_Season01_interf_SNR10_no_thinking_tokens_2048",
#     "interf_SNR10_x3":"/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_Season01_interf_SNR10_no_thinking_tokens_2048_comp_x3",
#     "interf_SNR10_x4":"/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_Season01_interf_SNR10_no_thinking_tokens_2048_comp_x4",
#     "interf_SNR5":"/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_Season01_interf_SNR5_no_thinking_tokens_2048",
#     "interf_SNR5_x3":"/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_Season01_interf_SNR5_no_thinking_tokens_2048_comp_x3",
#     "interf_SNR5_x4":"/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_Season01_interf_SNR5_no_thinking_tokens_2048_comp_x4",
# }

## audio pipeline experiment:
# FOLDERS = {
#     "ASR+local_diarization+globaltracking+name_extraction(Full)": "/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_Season01_Clean_Anoy_no_thinking_tokens_2048",
#     "ASR+local_diarization+globaltracking": "/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_anon_global_Season01_Clean_no_thinking_tokens_2048",
#     "ASR+local_diarization+globaltracking (new prompt)": "/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_anon_global_Season01_Clean_no_thinking_tokens_2048_anonspk",
#     "ASR+local_diarization": "/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_anon_local_Season01_Clean_no_thinking_tokens_2048",
# }

# PerltQA dataset
FOLDERS = {
    "Perltqa clean": "/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_perltqa_dataset_pred_name_bundle_0_perltqa_no_thinking_tokens_2048",
    "Perltqa interf 5dB": "/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_perltqa_dataset_pred_name_bundle_0_perltqa_interf_SNR5_no_thinking_tokens_2048",
    "Perltqa interf 0dB": "/storage/home/tuochao/Mem-alpha-audio/agents/qwen3.6-27b_Qwen_Qwen3.6-27B_perltqa_dataset_pred_name_bundle_0_perltqa_interf_SNR0_no_thinking_tokens_2048",

}
def find_probe_files(folder):
    """Return error_probe.json paths for a run folder: the folder itself, its immediate
    instance subdirs (0/, 1/, ...), AND per-seed instance subdirs (seed*/0/). Findings
    from all of them are pooled into the folder's metrics (a run without seed* subdirs
    behaves exactly as before)."""
    cands = [os.path.join(folder, "error_probe.json")]
    cands += sorted(glob.glob(os.path.join(folder, "*", "error_probe.json")))
    cands += sorted(glob.glob(os.path.join(folder, "seed*", "*", "error_probe.json")))
    return [p for p in dict.fromkeys(cands) if os.path.isfile(p)]


def base_folder_for(probe_path):
    """Map an error_probe.json path back to its run (base) folder, collapsing the
    instance subdir (numeric batch idx like 0/) and a per-seed level (seed*/): both
    `<base>/0/error_probe.json` and `<base>/seedN/0/error_probe.json` -> `<base>`."""
    inst = os.path.dirname(probe_path)
    folder = os.path.dirname(inst) if os.path.basename(inst).isdigit() else inst
    if re.fullmatch(r"seed\d+", os.path.basename(folder)):
        folder = os.path.dirname(folder)
    return folder


def load_findings(folder):
    """Load findings across all instance/seed subdirs of one run folder.

    Returns (merged_findings, files, per_file_findings) where per_file_findings[i] is
    the findings list of files[i] — one probed instance/seed. The per-file split lets
    us report the std ACROSS seeds in addition to the pooled metrics."""
    files = find_probe_files(folder)
    per_file, findings = [], []
    for p in files:
        with open(p) as f:
            fnd = json.load(f).get("findings", [])
        per_file.append(fnd)
        findings.extend(fnd)
    return findings, files, per_file


def label_for(folder):
    """Short, distinguishing label: the compression tag if present, else basename."""
    base = os.path.basename(folder.rstrip("/"))
    m = re.search(r"comp_x[\d.]+", base)
    if m:
        return m.group(0)
    return "default" if base else base


def derive_label(folder):
    """Readable condition label from a full agent folder name, e.g.
    '..._dataset_pred_name_Season01_interf_SNR5_no_thinking_tokens_2048_comp_x3'
    -> 'interf_SNR5_x3';  background 'SNRk' -> 'BG_SNRk';  gt_name -> 'GT_' prefix."""
    b = os.path.basename(folder.rstrip("/"))
    comp = ""
    mc = re.search(r"_comp_(x[\d.]+)$", b)
    if mc:
        comp = "_" + mc.group(1)
    mcond = re.search(r"Season01_(.+?)_no_thinking", b)
    cond = mcond.group(1) if mcond else b
    cond = cond.replace("_Anoy", "")
    if re.match(r"^SNR\d", cond):          # background-noise SNR -> BG_ prefix
        cond = "BG_" + cond
    label = cond + comp
    if "dataset_gt_name" in b:
        label = "GT_" + label
    return label


def discover(root):
    """Find every agent folder under `root` that has an error_probe.json.
    Returns [(label, folder), ...] sorted, with duplicate labels disambiguated."""
    folders = {}
    for p in glob.glob(os.path.join(root, "**", "error_probe.json"), recursive=True):
        folders[base_folder_for(p)] = True
    pairs, seen = [], {}
    for folder in sorted(folders):
        lab = derive_label(folder)
        if lab in seen:
            seen[lab] += 1
            lab = f"{lab}#{seen[lab]}"
        else:
            seen[lab] = 1
        pairs.append((lab, folder))
    return pairs


def ratio(recs, flag):
    """(passed, ran, ratio) for a probe flag over recs (ran = flag present)."""
    ran = [r for r in recs if r.get("detail", {}).get(flag) is not None]
    passed = sum(1 for r in ran if r["detail"][flag])
    return passed, len(ran), (passed / len(ran) if ran else 0.0)


def cross(recs):
    """T-vs-C dynamics over recs where BOTH t_correct and c_correct ran.
    Isolates the CONSTRUCTION stage (transcript -> constructed memory):
      self_correction = T wrong & C right  (construction recovered the answer)
      memory_loss     = T right & C wrong  (construction dropped the answer)
    Returns the full 2x2 counts."""
    both = [r for r in recs
            if r.get("detail", {}).get("t_correct") is not None
            and r.get("detail", {}).get("c_correct") is not None]
    t = lambda r: r["detail"]["t_correct"]
    c = lambda r: r["detail"]["c_correct"]
    self_corr = sum(1 for r in both if (not t(r)) and c(r))   # T wrong -> C right
    mem_loss = sum(1 for r in both if t(r) and (not c(r)))    # T right -> C wrong
    return {"applicable": len(both), "self_correction": self_corr, "memory_loss": mem_loss,
            "both_right": sum(1 for r in both if t(r) and c(r)),
            "both_wrong": sum(1 for r in both if (not t(r)) and (not c(r)))}


def col(p, n, r):
    return f"{p:3d}/{n:<3d} ({100*r:3.0f}%)"


def compression_ratios(files):
    """Per-instance input/memory compression_ratio values, one per compression.json
    found next to an error_probe.json (files without one are skipped)."""
    ratios = []
    for p in files:
        cj = os.path.join(os.path.dirname(p), "compression.json")
        if os.path.isfile(cj):
            try:
                r = json.load(open(cj)).get("compression_ratio")
                if r is not None:
                    ratios.append(float(r))
            except (ValueError, OSError, json.JSONDecodeError):
                pass
    return ratios


def compression_ratio(files):
    """Mean compression_ratio across instances (None if no compression.json found)."""
    ratios = compression_ratios(files)
    return sum(ratios) / len(ratios) if ratios else None


def seed_std(per_file):
    """Population std across per-seed (per-instance-file) rates for the all-samples
    T/C/S probe pass-rates and E2E accuracy — the seed-to-seed spread that pooling
    hides. Matches aggregate_probe_seeds.py (pstdev). std is 0.0 for a single instance.
    Returns {n_seeds, t, c, s, e2e} with the std values as fractions."""
    def rates(fnd):
        tot = len(fnd)
        e2e = (sum(1 for r in fnd if r.get("real_correct")) / tot) if tot else 0.0
        return (ratio(fnd, "t_correct")[2], ratio(fnd, "c_correct")[2],
                ratio(fnd, "s_correct")[2], e2e)
    per = [rates(fnd) for fnd in per_file if fnd]
    out = {"n_seeds": len(per)}
    for i, k in enumerate(("t", "c", "s", "e2e")):
        vals = [p[i] for p in per]
        out[k] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return out


def analyze(folder, label=None):
    findings, files, per_file = load_findings(folder)
    total = len(findings)
    fails = [f for f in findings if not f.get("real_correct")]
    correct = total - len(fails)
    crs = compression_ratios(files)
    m = {"label": label or label_for(folder), "folder": folder, "files": files,
         "total": total, "correct": correct, "failed": len(fails),
         "acc": correct / total if total else 0.0,
         "comp_ratio": sum(crs) / len(crs) if crs else None,
         "comp_ratio_std": statistics.pstdev(crs) if len(crs) > 1 else None,
         "seed_std": seed_std(per_file)}
    for tag, flag in (("t", "t_correct"), ("c", "c_correct"), ("s", "s_correct")):
        m[f"{tag}_fail"] = ratio(fails, flag)
        m[f"{tag}_all"] = ratio(findings, flag)
    m["cross_all"] = cross(findings)
    m["cross_fail"] = cross(fails)
    return m


def print_folder(m):
    ss = m["seed_std"]; nseed = ss["n_seeds"]
    # Std across the per-seed all-samples rates (blank when there's a single instance).
    def sd(k):
        return f"±{100*ss[k]:.1f}%" if nseed > 1 else "—"
    print(f"\n### {m['label']}   ({len(m['files'])} instance(s), {m['total']} QAs, "
          f"{m['failed']} failed)")
    print(f"    path: {m['folder']}")
    cr = f"{m['comp_ratio']:.2f}x" if m['comp_ratio'] is not None else "n/a"
    cr_std = m.get("comp_ratio_std")
    cr_std_str = f"±{cr_std:.2f}x" if cr_std is not None else "—"
    seedhdr = f"seed std (n={nseed})" if nseed > 1 else "seed std"
    print(f"    {'':24s} {'failed samples':>16s}  {'all samples':>16s}  {seedhdr:>14s}")
    print(f"  {'compression ratio':22s} {'—':>16s}  {cr:>16s}  {cr_std_str:>14s}")
    for tag, name in (("t", "T-probe (transcript)"),
                      ("c", "C-probe (construction)"),
                      ("s", "S-probe (retr. subset)")):
        print(f"  {name:22s} {col(*m[f'{tag}_fail']):>16s}  {col(*m[f'{tag}_all']):>16s}  {sd(tag):>14s}")
    print(f"  {'E2E':22s} {'—':>16s}  "
          f"{col(m['correct'], m['total'], m['acc']):>16s}  {sd('e2e'):>14s}")
    ca, cf = m["cross_all"], m["cross_fail"]
    def pct(x, n): return f"{100*x/n:.0f}%" if n else "  —"
    print("  ---- construction dynamics, T vs C ----")
    print(f"  self-correction (T✗→C✓)   all {ca['self_correction']:3d}/{ca['applicable']:<3d} "
          f"({pct(ca['self_correction'], ca['applicable'])})   "
          f"failed {cf['self_correction']:3d}/{cf['applicable']:<3d} "
          f"({pct(cf['self_correction'], cf['applicable'])})")
    print(f"  memory loss     (T✓→C✗)   all {ca['memory_loss']:3d}/{ca['applicable']:<3d} "
          f"({pct(ca['memory_loss'], ca['applicable'])})   "
          f"failed {cf['memory_loss']:3d}/{cf['applicable']:<3d} "
          f"({pct(cf['memory_loss'], cf['applicable'])})")
    # Full 2x2 confusion over all QAs (the four cells sum to 'applicable').
    n = ca["applicable"]
    def cell(x): return f"{x:3d} ({100*x/n:4.1f}%)" if n else "—"
    print(f"  ---- T×C confusion, all QAs (n={n}) ----")
    print(f"    {'':10s}{'C correct':>16s}{'C wrong':>16s}")
    print(f"    {'T correct':10s}{cell(ca['both_right']):>16s}{cell(ca['memory_loss']):>16s}"
          f"   <- T✓C✗ = memory loss")
    print(f"    {'T wrong':10s}{cell(ca['self_correction']):>16s}{cell(ca['both_wrong']):>16s}"
          f"   <- T✗C✓ = self-correction")


def print_comparison(ms):
    labels = [m["label"] for m in ms]
    w = max(26, *(len(l) for l in labels)) if labels else 26
    def row(name, vals):
        print(f"  {name:30s} " + "  ".join(f"{v:>{w}s}" for v in vals))
    print("\n==================== SIDE-BY-SIDE (all samples) ====================")
    row("metric", labels)
    row("compression ratio",
        [f"{m['comp_ratio']:.2f}x" if m['comp_ratio'] is not None else "n/a" for m in ms])
    for tag, name in (("t", "T-probe"), ("c", "C-probe"), ("s", "S-probe")):
        row(name, [col(*m[f"{tag}_all"]) for m in ms])
    row("E2E", [col(*(m["correct"], m["total"], m["acc"])) for m in ms])
    def cr(m, k):
        c = m["cross_all"]; n = c["applicable"]
        return f"{c[k]:3d}/{n:<3d} ({100*c[k]/n:3.0f}%)" if n else "—"
    row("self-correction (T✗→C✓)", [cr(m, "self_correction") for m in ms])
    row("memory loss (T✓→C✗)", [cr(m, "memory_loss") for m in ms])
    print("====================================================================")


def _metric_cells(m):
    """Ordered (column_name, value_string) for one folder row of the flipped table.
    Values are percent-only, one decimal (no raw counts)."""
    cr = f"{m['comp_ratio']:.1f}x" if m['comp_ratio'] is not None else "n/a"
    def pct(passed, ran):
        return f"{100*passed/ran:.1f}%" if ran else "—"
    def xr(k):
        c = m["cross_all"]
        return pct(c[k], c["applicable"])
    return [
        ("Comp ratio", cr),
        ("T-probe", pct(m["t_all"][0], m["t_all"][1])),
        ("C-probe", pct(m["c_all"][0], m["c_all"][1])),
        ("S-probe", pct(m["s_all"][0], m["s_all"][1])),
        ("E2E", pct(m["correct"], m["total"])),
        ("Self-corr (T✗→C✓)", xr("self_correction")),
        ("Mem-loss (T✓→C✗)", xr("memory_loss")),
    ]


def save_fig(ms, out_base):
    """Render the flipped comparison table (folders = rows, metrics = columns)
    to <out_base>.png and <out_base>.pdf."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col_names = [c for c, _ in _metric_cells(ms[0])]
    row_labels = [m["label"] for m in ms]
    cell_text = [[v for _, v in _metric_cells(m)] for m in ms]

    n_rows, n_cols = len(row_labels), len(col_names)
    fig_w = 1.6 + 1.9 * n_cols
    fig_h = 0.9 + 0.42 * n_rows
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title("Probe comparison (all samples)", fontweight="bold", pad=12)

    tbl = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_names,
                   cellLoc="center", rowLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)

    # Style header row + row-label column; zebra-stripe body rows.
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:                      # column headers
            cell.set_facecolor("#40466e"); cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif c == -1:                   # row labels (folder names from FOLDERS keys)
            cell.set_facecolor("#d9e1f2"); cell.get_text().set_fontweight("bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f2f2f2")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = f"{out_base}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote figure -> {path}")
    plt.close(fig)


def write_csv(ms, path):
    cols = ["label", "folder", "instances", "total", "correct", "acc",
            "T_all_pass", "T_all_ran", "C_all_pass", "C_all_ran",
            "S_all_pass", "S_all_ran",
            "T_fail_pass", "T_fail_ran", "C_fail_pass", "C_fail_ran",
            "S_fail_pass", "S_fail_ran",
            "xcompare_applicable",
            "TcSc_both_right", "TwSc_self_correction",
            "TcSw_memory_loss", "TwSw_both_wrong"]
    with open(path, "w", newline="") as f:
        wr = csvmod.writer(f); wr.writerow(cols)
        for m in ms:
            ca = m["cross_all"]
            wr.writerow([m["label"], m["folder"], len(m["files"]), m["total"],
                         m["correct"], f"{m['acc']:.4f}",
                         m["t_all"][0], m["t_all"][1], m["c_all"][0], m["c_all"][1],
                         m["s_all"][0], m["s_all"][1],
                         m["t_fail"][0], m["t_fail"][1], m["c_fail"][0], m["c_fail"][1],
                         m["s_fail"][0], m["s_fail"][1],
                         ca["applicable"], ca["both_right"], ca["self_correction"],
                         ca["memory_loss"], ca["both_wrong"]])
    print(f"\nWrote CSV -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*", help="probe folders to compare")
    ap.add_argument("--folders", dest="folders_json", default=None,
                    help='JSON list of folders, e.g. \'["a","b"]\'')
    ap.add_argument("--csv", default=None, help="also write metrics to this CSV path")
    ap.add_argument("--fig", default="diagnostic/probe_comparison",
                    help="output basename for the .png/.pdf table (default: "
                         "diagnostic/probe_comparison; empty string to skip)")
    ap.add_argument("--scan", nargs="?", const="agents", default=None, metavar="ROOT",
                    help="discover and analyze EVERY folder with an error_probe.json "
                         "under ROOT (default 'agents'), instead of the hardcoded FOLDERS.")
    args = ap.parse_args()

    # Build (label, path) pairs. CLI args label themselves via label_for();
    # --scan auto-discovers; otherwise the hardcoded FOLDERS dict is used.
    pairs = [(label_for(p), p) for p in args.folders]
    if args.folders_json:
        pairs += [(label_for(p), p) for p in json.loads(args.folders_json)]
    if args.scan:
        pairs += discover(args.scan)
    if not pairs:
        pairs = list(FOLDERS.items())   # {label: path}
    if not pairs:
        ap.error("no folders: pass them positionally, via --folders '[...]', or set FOLDERS")

    ms = []
    for label, folder in pairs:
        if not find_probe_files(folder):
            print(f"[skip] no error_probe.json under {folder}")
            continue
        ms.append(analyze(folder, label=label))

    # for m in ms:
    #     if m["label"] == "BG_SNR5_x3":
    #         m['c_all'] = (202, 264, 0.7640909090909091)
    #         m['correct'] = 173
    # print(ms)
    # exit(0)
    if not ms:
        ap.error("no folders with error_probe.json found")

    for m in ms:
        print_folder(m)
    if len(ms) > 1:
        print_comparison(ms)
    if args.csv:
        write_csv(ms, args.csv)
    if args.fig:
        save_fig(ms, args.fig)


if __name__ == "__main__":
    main()
