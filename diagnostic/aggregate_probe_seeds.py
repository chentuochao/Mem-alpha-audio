#!/usr/bin/env python3
"""Aggregate per-seed error_probe.json summaries into a single mean ± std report.

run_probe_errors.sh probes the original `<base>/0/` instance plus every per-seed
`<base>/seedX/0/` instance, each of which writes its own error_probe.json. This
script collects those summaries and reports, across seeds:

  * original QA accuracy               (mean ± std)
  * probe pass rates on ALL QAs        (G/C/T/S, mean ± std of ratio)
  * failure attribution                (per-seed fraction of failures, mean ± std,
                                         plus pooled counts)
  * retrieval vs response split         (pooled)
  * self-correction                     (pooled)

It is a no-op (nothing to aggregate) when only the original `0/` was probed.
"""
import os
import glob
import json
import argparse
import statistics
from typing import Dict, List


def _find_probe_files(base_dir: str) -> Dict[str, str]:
    """Map seed label -> error_probe.json path for the original + seed runs."""
    found: Dict[str, str] = {}
    orig = os.path.join(base_dir, "0", "error_probe.json")
    if os.path.isfile(orig):
        found["original"] = orig
    for sd in sorted(glob.glob(os.path.join(base_dir, "seed*", "0", "error_probe.json"))):
        # .../<base>/seedX/0/error_probe.json  ->  seedX
        seed = os.path.basename(os.path.dirname(os.path.dirname(sd)))
        found[seed] = sd
    return found


def _mean_std(vals: List[float]):
    if not vals:
        return 0.0, 0.0
    mean = statistics.fmean(vals)
    # Population std (matches np.std default), 0 for a single seed.
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return float(mean), float(std)


def main():
    ap = argparse.ArgumentParser(description="Aggregate per-seed error_probe.json summaries.")
    ap.add_argument("--base_dir", required=True,
                    help="Base run dir containing 0/ and (optionally) seedX/ subfolders.")
    ap.add_argument("--out", default=None,
                    help="Output json (default: <base_dir>/error_probe_seed_summary.json).")
    args = ap.parse_args()

    probe_files = _find_probe_files(args.base_dir)
    if len(probe_files) < 2:
        print(f"Nothing to aggregate: found {len(probe_files)} probe file(s) under "
              f"{args.base_dir} (need >=2 seeds).")
        return

    seeds = sorted(probe_files)
    summaries: Dict[str, dict] = {}
    for seed in seeds:
        with open(probe_files[seed]) as f:
            summaries[seed] = json.load(f).get("summary", {})

    # --- original QA accuracy across seeds ---
    accs = [summaries[s].get("original_qa_accuracy", 0.0) for s in seeds]
    acc_mean, acc_std = _mean_std(accs)

    # --- probe pass rates (on ALL QAs) across seeds ---
    probe_keys = ["g_probe", "c_probe", "t_probe", "s_probe"]
    probe_stats: Dict[str, dict] = {}
    for pk in probe_keys:
        ratios = [summaries[s].get("probe_pass_rates_all", {}).get(pk, {}).get("ratio", 0.0)
                  for s in seeds]
        m, sd = _mean_std(ratios)
        probe_stats[pk] = {"mean": m, "std": sd, "per_seed": dict(zip(seeds, ratios))}

    # --- failure attribution: per-seed fraction of that seed's failures ---
    all_buckets = sorted({k for s in seeds for k in summaries[s].get("attribution", {})})
    attribution: Dict[str, dict] = {}
    for b in all_buckets:
        fracs, pooled = [], 0
        for s in seeds:
            attr = summaries[s].get("attribution", {})
            failed = max(summaries[s].get("failed", 0), 1)
            cnt = attr.get(b, 0)
            pooled += attr.get(b, 0)
            fracs.append(cnt / failed)
        m, sd = _mean_std(fracs)
        attribution[b] = {"frac_mean": m, "frac_std": sd, "pooled_count": pooled}

    # --- pooled retrieval split & self-correction ---
    pooled_response = sum(summaries[s].get("retrieval_split", {}).get("response", 0) for s in seeds)
    pooled_retrieval = sum(summaries[s].get("retrieval_split", {}).get("retrieval", 0) for s in seeds)
    pooled_selfcorr_failed = sum(summaries[s].get("self_correction", {}).get("failed", 0) for s in seeds)
    pooled_selfcorr_all = sum(summaries[s].get("self_correction", {}).get("all", 0) for s in seeds)
    pooled_total = sum(summaries[s].get("total", 0) for s in seeds)
    pooled_failed = sum(summaries[s].get("failed", 0) for s in seeds)

    # --- print report ---
    print("\n============ CROSS-SEED ERROR PROBE SUMMARY ============")
    print(f"base_dir       : {args.base_dir}")
    print(f"seeds ({len(seeds)})      : {', '.join(seeds)}")
    print(f"pooled QAs     : {pooled_total}  (failed {pooled_failed})")
    print(f"original QA accuracy   : {acc_mean:.3f} ± {acc_std:.3f}")
    print("  per seed             : " + ", ".join(f"{s}={a:.3f}" for s, a in zip(seeds, accs)))
    print("---- probe pass rates (all QAs, mean ± std across seeds) ----")
    for pk in probe_keys:
        st = probe_stats[pk]
        print(f"  {pk:8s}  {st['mean']*100:5.1f}% ± {st['std']*100:4.1f}%")
    print("---- failure attribution (frac of failures, mean ± std) ----")
    for b in all_buckets:
        st = attribution[b]
        print(f"  {b:24s} {st['frac_mean']*100:5.1f}% ± {st['frac_std']*100:4.1f}%  "
              f"(pooled {st['pooled_count']})")
    print("---- pooled ----")
    print(f"  retrieval split        response {pooled_response} / retrieval {pooled_retrieval}")
    print(f"  self-correction        failed {pooled_selfcorr_failed} / all {pooled_selfcorr_all}")
    print("========================================================\n")

    out = args.out or os.path.join(args.base_dir, "error_probe_seed_summary.json")
    with open(out, "w") as f:
        json.dump({
            "base_dir": args.base_dir,
            "seeds": seeds,
            "probe_files": probe_files,
            "original_qa_accuracy": {"mean": acc_mean, "std": acc_std,
                                     "per_seed": dict(zip(seeds, accs))},
            "probe_pass_rates_all": probe_stats,
            "attribution": attribution,
            "pooled": {
                "total": pooled_total, "failed": pooled_failed,
                "retrieval_split": {"response": pooled_response, "retrieval": pooled_retrieval},
                "self_correction": {"failed": pooled_selfcorr_failed, "all": pooled_selfcorr_all},
            },
        }, f, indent=2)
    print(f"Wrote cross-seed summary -> {out}")


if __name__ == "__main__":
    main()
