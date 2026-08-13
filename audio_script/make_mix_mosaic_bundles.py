#!/usr/bin/env python3
"""
Divide the Mix_Mosaic (mixed Seamless-Interaction dyads) conversations into a
small number of fixed, replicable bundles for the memory-vs-full-context
evaluation.

Data layout (produced by ``audio_script/datasets/mix_interact.py``)::

    <data-dir>/
        Pxxx_Pyyy/                     # one folder per speaker *pair*
            V00_Sxxxx_Ixxxxxxxx/       # one folder per conversation
                mixed_conv.wav
                transcript1.json  transcript2.json   # per-speaker, word-level
                vad1.json         vad2.json
            ...
        ...

``Pxxx`` is a stable, unique speaker id.  We want every conversation involving a
given speaker to land in the **same** bundle (the opposite of the perltqa
bundler, which *spreads* shared names apart).

How grouping works
------------------
1. Each ``Pxxx_Pyyy`` folder is an atomic unit — all its conversations share the
   same two speakers, so they always stay together.
2. Folders that share a speaker are merged into **connected components**
   (union-find over the speaker graph).  A component is the true atomic bundle
   unit; because it contains *every* folder each of its speakers appears in, no
   speaker can ever be split across bundles once components are kept whole.
3. Components are packed into ``--num-bundles`` bundles by balanced
   longest-processing-time (largest component -> currently-lightest bundle).
   Since components never share a speaker, this pure token-balancing pass keeps
   every speaker inside exactly one bundle by construction.

Token counting mirrors the perltqa flow: the concatenated word-level transcript
of each folder is encoded with a Qwen tokenizer and the count is recorded.

Output
------
  bundles.json : {meta, bundles:[{bundle_id, num_folders, num_convs,
                  num_speakers, total_tokens, speakers, folders, conversations}]}

Example
-------
    python audio_script/make_mix_mosaic_bundles.py \\
        --data-dir /checkpoint/seamless/tuochao/data/Mix_Mosaic/naturalistic/test \\
        --num-bundles 4
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Set

from transformers import AutoTokenizer

DEFAULT_DATA_DIR = "/checkpoint/seamless/tuochao/data/Mix_Mosaic/naturalistic/test"
DEFAULT_TOKENIZER = (
    "/checkpoint/seamless/tuochao/Models/huggingface/"
    "models--Qwen--Qwen3-1.7B/snapshots/"
    "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)
TRANSCRIPT_FILES = ("transcript1.json", "transcript2.json")


# ──────────────────────────────────────────────────────────────────────────────
# Disk discovery + token counting
# ──────────────────────────────────────────────────────────────────────────────

def conversation_text(conv_dir: str) -> str:
    """Concatenate every spoken word (both speakers) in one conversation."""
    words: List[str] = []
    for tf in TRANSCRIPT_FILES:
        p = os.path.join(conv_dir, tf)
        if not os.path.exists(p):
            continue
        try:
            segs = json.load(open(p))
        except Exception:
            continue
        for seg in segs:
            for w in seg.get("wfeats", []):
                words.append(str(w.get("word", "")))
    return " ".join(words)


def discover_pairs(data_dir: str, tok) -> Dict[str, dict]:
    """
    Scan *data_dir* for ``Pxxx_Pyyy`` folders.

    Returns ``{pair_folder: {speakers, tokens, conversations}}`` where
    ``conversations`` is a sorted list of ``{conv_id, path}`` dicts.
    """
    pairs: Dict[str, dict] = {}
    for pf in sorted(os.listdir(data_dir)):
        pfp = os.path.join(data_dir, pf)
        if not os.path.isdir(pfp):
            continue
        speakers = set(pf.split("_"))
        convs: List[dict] = []
        texts: List[str] = []
        for cd in sorted(os.listdir(pfp)):
            cdp = os.path.join(pfp, cd)
            if not os.path.isdir(cdp):
                continue
            convs.append({"conv_id": cd, "path": cdp})
            texts.append(conversation_text(cdp))
        if not convs:
            continue
        tokens = len(tok.encode("\n".join(texts)))
        pairs[pf] = {"speakers": speakers, "tokens": tokens, "conversations": convs}
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Shared-speaker connected components (union-find)
# ──────────────────────────────────────────────────────────────────────────────

def connected_components(pairs: Dict[str, dict]) -> List[List[str]]:
    """Merge pair-folders that share a speaker into components (sorted, det.)."""
    parent = {pf: pf for pf in pairs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    spk2pf: Dict[str, List[str]] = defaultdict(list)
    for pf, info in pairs.items():
        for s in info["speakers"]:
            spk2pf[s].append(pf)
    for pfs in spk2pf.values():
        for other in pfs[1:]:
            union(pfs[0], other)

    comps: Dict[str, List[str]] = defaultdict(list)
    for pf in pairs:
        comps[find(pf)].append(pf)
    # deterministic: heaviest component first, ties by member folders
    out = [sorted(members) for members in comps.values()]
    out.sort(key=lambda members: (-sum(pairs[m]["tokens"] for m in members), members))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Balanced packing (longest-processing-time)
# ──────────────────────────────────────────────────────────────────────────────

def pack_components(
    components: List[List[str]],
    pairs: Dict[str, dict],
    n_bundles: int,
) -> List[List[str]]:
    """
    Greedy LPT: assign each component (heaviest first) to the currently-lightest
    bundle.  Ties broken by lowest bundle index for determinism.  Returns a list
    of ``n_bundles`` bundles, each a list of pair-folder names.
    """
    n_bundles = max(1, min(n_bundles, len(components)))
    bundles: List[List[str]] = [[] for _ in range(n_bundles)]
    loads = [0] * n_bundles
    for comp in components:  # already heaviest-first
        comp_tokens = sum(pairs[m]["tokens"] for m in comp)
        bi = min(range(n_bundles), key=lambda i: (loads[i], i))
        bundles[bi].extend(comp)
        loads[bi] += comp_tokens
    return bundles


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER,
                    help="Path/name of the Qwen tokenizer used for token counts.")
    ap.add_argument("--num-bundles", type=int, default=4,
                    help="Number of bundles to produce (primary control).")
    ap.add_argument("--target-tokens", type=int, default=None,
                    help="If set, overrides --num-bundles via round(total/target).")
    ap.add_argument("--out", default=None,
                    help="Output JSON (default: <data-dir>/bundles.json)")
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.data_dir, "bundles.json")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    pairs = discover_pairs(args.data_dir, tok)
    if not pairs:
        raise SystemExit(f"No Pxxx_Pyyy folders found under {args.data_dir}")
    total_tokens = sum(p["tokens"] for p in pairs.values())

    components = connected_components(pairs)

    if args.target_tokens:
        n_bundles = max(1, round(total_tokens / args.target_tokens))
    else:
        n_bundles = args.num_bundles
    n_bundles = max(1, min(n_bundles, len(components)))

    packed = pack_components(components, pairs, n_bundles)

    # ── assemble output ──
    bundles = []
    for bi, folders in enumerate(packed):
        folders = sorted(folders)
        speakers: Set[str] = set()
        conversations: List[dict] = []
        tok_sum = 0
        n_conv = 0
        for pf in folders:
            info = pairs[pf]
            speakers |= info["speakers"]
            tok_sum += info["tokens"]
            n_conv += len(info["conversations"])
            for c in info["conversations"]:
                conversations.append({"pair": pf, "conv_id": c["conv_id"], "path": c["path"]})
        bundles.append({
            "bundle_id": bi,
            "num_folders": len(folders),
            "num_convs": n_conv,
            "num_speakers": len(speakers),
            "total_tokens": tok_sum,
            "speakers": sorted(speakers),
            "folders": folders,
            "conversations": conversations,
        })

    # ── invariant: every speaker in exactly one bundle ──
    spk_bundles: Dict[str, Set[int]] = defaultdict(set)
    for b in bundles:
        for s in b["speakers"]:
            spk_bundles[s].add(b["bundle_id"])
    split = {s: sorted(bs) for s, bs in spk_bundles.items() if len(bs) > 1}
    assert not split, f"BUG: speakers split across bundles: {split}"

    meta = {
        "data_dir": args.data_dir,
        "tokenizer": args.tokenizer_dir,
        "num_bundles": len(bundles),
        "num_pair_folders": len(pairs),
        "num_conversations": sum(len(p["conversations"]) for p in pairs.values()),
        "num_speakers": len({s for p in pairs.values() for s in p["speakers"]}),
        "total_tokens": total_tokens,
        "num_components": len(components),
        "grouping": "shared-speaker components, LPT-balanced by tokens",
        "target_tokens": args.target_tokens,
    }
    json.dump({"meta": meta, "bundles": bundles}, open(out_path, "w"), indent=2)

    # ── report ──
    tks = [b["total_tokens"] for b in bundles]
    print("=" * 72)
    print(f"MIX_MOSAIC BUNDLES  ({len(bundles)} bundles, {len(pairs)} folders, "
          f"{meta['num_conversations']} convs, {meta['num_speakers']} speakers)")
    print("=" * 72)
    print(f"  total tokens       : {total_tokens:,}")
    print(f"  components         : {len(components)}")
    print(f"  tokens/bundle      : min={min(tks):,} mean={sum(tks)//len(tks):,} max={max(tks):,}")
    print(f"  speakers split     : {len(split)} (must be 0)")
    print()
    print(f"  {'bundle':>6}{'folders':>9}{'convs':>7}{'spk':>6}{'tokens':>12}")
    for b in bundles:
        print(f"  {b['bundle_id']:>6}{b['num_folders']:>9}{b['num_convs']:>7}"
              f"{b['num_speakers']:>6}{b['total_tokens']:>12,}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
