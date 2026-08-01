#!/usr/bin/env python3
"""
Pre-divide PerLTQA profiles into fixed, replicable bundles for the
memory-vs-full-context evaluation.

Each bundle groups several profiles so that the concatenated transcript hits a
target token budget (stressing the full-context baseline while per-profile
memory stays cheap). Assignment is deterministic (no RNG) so the split is
reproducible across runs.

Speaker-name collisions
-----------------------
PerLTQA profiles are independent: the same common name (e.g. "Li Ming", in 52
profiles) refers to *different* people, but ``perltqa_dialogue_tts`` canonicalizes
names globally so they share one synthesized voice. Two profiles with a shared
speaker name in the same bundle => same name + same voice for different people =>
ambiguous attribution. We therefore **minimize** intra-bundle name collisions.

Note: zero collisions is impossible at large bundle sizes — a name in K profiles
needs >= K bundles to stay collision-free. This script minimizes collisions and
reports the unavoidable residual.

Inputs
------
  --data-dir : dialogue-TTS root (<Profile>/<dialogue_id>/channel_map.json)
  --stats    : annotation_token_stats.json (per-profile Qwen token counts);
               if absent, token counts are read but the script errors out.

Output
------
  bundles.json : {meta, bundles:[{bundle_id, profiles, total_tokens,
                  num_profiles, collisions:{name:[profiles]}}]}

Example
-------
    python audio_script/make_perltqa_bundles.py \
        --data-dir /checkpoint/seamless/tuochao/data/PerLTQA/dialogue_tts_en_v2 \
        --target-tokens 80000 --tolerance 0.15
"""

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Set


def name_key(n: str) -> str:
    """Case/separator-insensitive key (mirrors perltqa_dialogue_tts canonicalization)."""
    return re.sub(r"[\s_\-]+", " ", str(n).strip().lower()).strip()


def load_profile_speakers(data_dir: str) -> Dict[str, Set[str]]:
    """profile -> set of canonical speaker names (from every channel_map.json)."""
    prof_speakers: Dict[str, Set[str]] = defaultdict(set)
    for cm in glob.glob(os.path.join(data_dir, "**", "channel_map.json"), recursive=True):
        profile = os.path.relpath(cm, data_dir).split(os.sep)[0]
        try:
            m = json.load(open(cm))
        except Exception:
            continue
        for nm in (m.get("channel_map", {}) or {}).keys():
            prof_speakers[profile].add(name_key(nm))
    return dict(prof_speakers)


def load_profile_tokens(stats_path: str) -> Dict[str, int]:
    """profile -> Qwen token count, from annotation_token_stats.json."""
    stats = json.load(open(stats_path))
    return {r["profile"]: int(r["tokens"]) for r in stats["per_profile"]}


def bundle_profiles(
    tokens: Dict[str, int],
    speakers: Dict[str, Set[str]],
    target: int,
    tolerance: float,
) -> List[dict]:
    """
    Deterministic collision-minimizing balanced assignment.

    Number of bundles = round(total_tokens / target). Profiles are placed largest
    first (first-fit-decreasing, deterministic ties by name); each goes to the
    bundle that (1) stays <= max budget, (2) adds the fewest speaker-name
    collisions, (3) has the smallest running token sum (balance).
    """
    profiles = sorted(tokens.keys())  # deterministic base order
    total = sum(tokens[p] for p in profiles)
    n_bundles = max(1, round(total / target))
    max_tok = int(target * (1 + tolerance))

    bundles = [{"tokens": 0, "profiles": [], "names": defaultdict(int)}
               for _ in range(n_bundles)]

    # largest-first so big profiles are placed while bins are empty; ties by name
    order = sorted(profiles, key=lambda p: (-tokens[p], p))
    for p in order:
        p_names = speakers.get(p, set())
        best = None
        for bi, b in enumerate(bundles):
            over = b["tokens"] + tokens[p] > max_tok
            collisions = sum(1 for nm in p_names if b["names"].get(nm))
            # rank: prefer under-budget, then fewest collisions, then lightest,
            # then lowest index (determinism)
            cand = (0 if not over else 1, collisions, b["tokens"], bi)
            if best is None or cand < best[0]:
                best = (cand, bi)
        bi = best[1]
        b = bundles[bi]
        b["tokens"] += tokens[p]
        b["profiles"].append(p)
        for nm in p_names:
            b["names"][nm] += 1

    out = []
    for bi, b in enumerate(bundles):
        collisions = {}
        for nm, cnt in b["names"].items():
            if cnt >= 2:
                holders = sorted(p for p in b["profiles"] if nm in speakers.get(p, set()))
                collisions[nm] = holders
        out.append({
            "bundle_id": bi,
            "num_profiles": len(b["profiles"]),
            "total_tokens": b["tokens"],
            "profiles": sorted(b["profiles"]),
            "collisions": dict(sorted(collisions.items(), key=lambda kv: -len(kv[1]))),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="/checkpoint/seamless/tuochao/data/PerLTQA/dialogue_tts_en_v2")
    ap.add_argument("--stats", default=None,
                    help="annotation_token_stats.json (default: <data-dir>/annotation_token_stats.json)")
    ap.add_argument("--target-tokens", type=int, default=80000)
    ap.add_argument("--tolerance", type=float, default=0.15)
    ap.add_argument("--out", default=None,
                    help="output JSON (default: <data-dir>/bundles.json)")
    args = ap.parse_args()

    stats_path = args.stats or os.path.join(args.data_dir, "annotation_token_stats.json")
    out_path = args.out or os.path.join(args.data_dir, "bundles.json")

    tokens = load_profile_tokens(stats_path)
    speakers = load_profile_speakers(args.data_dir)
    # keep only profiles present in both
    profiles = sorted(set(tokens) & set(speakers))
    tokens = {p: tokens[p] for p in profiles}
    speakers = {p: speakers[p] for p in profiles}

    bundles = bundle_profiles(tokens, speakers, args.target_tokens, args.tolerance)

    # ── report ──
    tks = [b["total_tokens"] for b in bundles]
    nps = [b["num_profiles"] for b in bundles]
    lo, hi = int(args.target_tokens * (1 - args.tolerance)), int(args.target_tokens * (1 + args.tolerance))
    in_band = sum(1 for t in tks if lo <= t <= hi)
    total_collisions = sum(len(b["collisions"]) for b in bundles)
    collision_free = sum(1 for b in bundles if not b["collisions"])

    print("=" * 68)
    print(f"BUNDLES  (target {args.target_tokens:,} ±{int(args.tolerance*100)}%  ->  {lo:,}-{hi:,})")
    print("=" * 68)
    print(f"  profiles bundled   : {len(profiles)}")
    print(f"  bundles            : {len(bundles)}")
    print(f"  tokens/bundle      : min={min(tks):,} mean={sum(tks)//len(tks):,} max={max(tks):,}")
    print(f"  in target band     : {in_band}/{len(bundles)}")
    print(f"  profiles/bundle    : min={min(nps)} max={max(nps)}")
    print(f"  collision-free bnd : {collision_free}/{len(bundles)}")
    print(f"  residual collided names (summed over bundles): {total_collisions}")
    print()
    print(f"  {'bundle':>6}{'profs':>7}{'tokens':>10}{'collided_names':>16}")
    for b in bundles:
        print(f"  {b['bundle_id']:>6}{b['num_profiles']:>7}{b['total_tokens']:>10,}{len(b['collisions']):>16}")

    meta = {
        "data_dir": args.data_dir,
        "target_tokens": args.target_tokens,
        "tolerance": args.tolerance,
        "num_bundles": len(bundles),
        "num_profiles": len(profiles),
        "collision_policy": "minimize",
        "tokenizer": json.load(open(stats_path)).get("tokenizer", "unknown"),
    }
    json.dump({"meta": meta, "bundles": bundles}, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
