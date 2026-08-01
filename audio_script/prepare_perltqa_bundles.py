#!/usr/bin/env python3
"""
Pre-divide PerLTQA profiles into replicable evaluation bundles + attach the QA
list, producing a manifest that is consistent with the Step-1 hierarchy.

Two modes:
  --mode per_profile   one bundle == one profile (Mem-alpha's native unit).
                       Keeps only profiles with >= --min-qa dialogue QAs
                       (default 20, mirroring Mem-alpha's answerable-QA filter).
  --mode multi         drop profiles with < --min-qa QAs (default 1, i.e. drop
                       QA-less), then pack the remaining profiles up to
                       --target-tokens, minimizing speaker-name collisions.
                       No QA-less distractors, so every bundle has QAs.

Step-1 / Bazinga correspondence (kept consistent on purpose):
  bundle_id + profile           <->  Season + Episode   (the "episode" key)
  dialogue sub-folder under it  <->  chunk id           (CHUNK_i)
So each bundle groups "episodes" (profiles); each profile is an ordered stream of
"chunks" (its dialogue folders, same folders Step-1 reads via channel_map.json).

QA:
  Extracted from perltqa_en_v2.json, **only the "dialogues" memory type** (the
  synthesized audio contains dialogues only). Evidence is reported at CHUNK level
  ("<profile>/<dialogue_folder>"), not turn level.

Output:
  <out>.json   full manifest {meta, bundles:[...]}
  <out>.jsonl  one bundle per line (same bundle dicts)

Example:
  python audio_script/prepare_perltqa_bundles.py --mode multi --target-tokens 80000
  python audio_script/prepare_perltqa_bundles.py --mode per_profile
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Set

# ── defaults ────────────────────────────────────────────────────────────────
DATA_DIR = "/checkpoint/seamless/tuochao/data/PerLTQA/dialogue_tts_en_v2"
PERLTMEM = "/checkpoint/seamless/tuochao/data/PerLTQA/Dataset/en_v2/perltmem_en_v2.json"
PERLTQA = "/checkpoint/seamless/tuochao/data/PerLTQA/Dataset/en_v2/perltqa_en_v2.json"

_EVID_RE = re.compile(r"\d+_\d+_\d+#\d+")           # dialogue evidence key form
_NUMS_RE = re.compile(r"\d+")


def safe_filename(name: str) -> str:
    """Mirror perltqa_dialogue_tts.safe_filename (folder name for a raw name/key)."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    return name.strip("_") or "unnamed"


def name_key(n: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(n).strip().lower()).strip()


def _sort_key(dialogue_key: str):
    return tuple(int(x) for x in _NUMS_RE.findall(dialogue_key)) or (0,)


# ── load helpers ─────────────────────────────────────────────────────────────
def load_profile_tokens(stats_path: str) -> Dict[str, int]:
    stats = json.load(open(stats_path))
    return {r["profile"]: int(r["tokens"]) for r in stats["per_profile"]}, \
           stats.get("tokenizer", "unknown")


def load_existing_chunks(data_dir: str) -> Dict[str, Set[str]]:
    """profile_folder -> set of dialogue sub-folders that were actually generated."""
    chunks: Dict[str, Set[str]] = defaultdict(set)
    for cm in glob.glob(os.path.join(data_dir, "**", "channel_map.json"), recursive=True):
        rel = os.path.relpath(os.path.dirname(cm), data_dir).split(os.sep)
        if len(rel) == 2:
            chunks[rel[0]].add(rel[1])
    return dict(chunks)


def load_profile_speakers(data_dir: str) -> Dict[str, Set[str]]:
    spk: Dict[str, Set[str]] = defaultdict(set)
    for cm in glob.glob(os.path.join(data_dir, "**", "channel_map.json"), recursive=True):
        profile = os.path.relpath(cm, data_dir).split(os.sep)[0]
        try:
            m = json.load(open(cm))
        except Exception:
            continue
        for nm in (m.get("channel_map", {}) or {}).keys():
            spk[profile].add(name_key(nm))
    return dict(spk)


def load_ordered_dialogue_keys(perltmem_path: str) -> Dict[str, List[str]]:
    """profile_folder -> ordered raw dialogue keys (chronological, from perltmem)."""
    d = json.load(open(perltmem_path))
    out: Dict[str, List[str]] = {}
    for prof, pv in d.items():
        folder = safe_filename(prof)
        keys = list((pv.get("dialogues", {}) or {}).keys())
        keys.sort(key=_sort_key)
        out[folder] = keys
    return out


def load_dialogue_qas(perltqa_path: str) -> Dict[str, list]:
    """profile_folder -> list of raw 'dialogues' QA items ({dialogue_key:[qa,...]})."""
    q = json.load(open(perltqa_path))
    out: Dict[str, list] = defaultdict(list)
    for entry in q:
        for char, mem in entry.items():
            out[safe_filename(char)].extend((mem.get("dialogues", []) or []))
    return dict(out)


def build_chunks(folder: str, ordered_keys: List[str], existing: Set[str]) -> List[dict]:
    """Ordered chunk list for a profile: only dialogue keys whose folder exists."""
    chunks = []
    for k in ordered_keys:
        f = safe_filename(k)
        if f in existing:
            chunks.append({"chunk_id": f, "dialogue_key": k,
                           "rel_path": f"{folder}/{f}"})
    # include any generated folder not present in perltmem order (defensive), sorted
    known = {c["chunk_id"] for c in chunks}
    for f in sorted(existing - known):
        chunks.append({"chunk_id": f, "dialogue_key": None, "rel_path": f"{folder}/{f}"})
    return chunks


def extract_qas(folder: str, dialogue_items: list, existing: Set[str]) -> List[dict]:
    """Flatten dialogues QAs -> {profile, question, answer, evidence_chunks, ...}."""
    out = []
    for item in dialogue_items:
        if not isinstance(item, dict):
            continue
        for dkey, qalist in item.items():
            if not isinstance(qalist, list):
                continue
            for qa in qalist:
                if not isinstance(qa, dict):
                    continue
                question = qa.get("Question")
                answer = qa.get("Answer")
                if not question or not answer:
                    continue
                # evidence keys: Reference Memory list + the outer key
                keys = set()
                rm = qa.get("Reference Memory")
                if isinstance(rm, str):
                    keys.update(_EVID_RE.findall(rm))
                elif isinstance(rm, list):
                    for x in rm:
                        keys.update(_EVID_RE.findall(str(x)))
                keys.add(dkey)
                ev = sorted({f"{folder}/{safe_filename(k)}"
                             for k in keys if safe_filename(k) in existing})
                out.append({
                    "profile": folder,
                    "question": question.strip(),
                    "answer": answer.strip() if isinstance(answer, str) else answer,
                    "evidence_chunks": ev,          # chunk-level, not turn-level
                    "memory_type": "dialogues",
                    "data_source": "perltqa",
                })
    return out


# ── bundling ─────────────────────────────────────────────────────────────────
def bundle_multi(profiles, tokens, speakers, target, tol):
    """Deterministic collision-minimizing balanced packing to ~target tokens."""
    total = sum(tokens[p] for p in profiles)
    n_bundles = max(1, round(total / target))
    max_tok = int(target * (1 + tol))
    bundles = [{"tokens": 0, "profiles": [], "names": defaultdict(int)} for _ in range(n_bundles)]
    for p in sorted(profiles, key=lambda x: (-tokens[x], x)):   # largest-first, det. ties
        pn = speakers.get(p, set())
        best = None
        for bi, b in enumerate(bundles):
            over = b["tokens"] + tokens[p] > max_tok
            coll = sum(1 for nm in pn if b["names"].get(nm))
            cand = ((0 if not over else 1, coll, b["tokens"], bi), bi)
            if best is None or cand[0] < best[0]:
                best = cand
        b = bundles[best[1]]
        b["tokens"] += tokens[p]; b["profiles"].append(p)
        for nm in pn:
            b["names"][nm] += 1
    return [sorted(b["profiles"]) for b in bundles]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--perltmem", default=PERLTMEM)
    ap.add_argument("--perltqa", default=PERLTQA)
    ap.add_argument("--stats", default=None, help="annotation_token_stats.json (default: <data-dir>/...)")
    ap.add_argument("--mode", choices=["per_profile", "multi"], default="multi")
    ap.add_argument("--target-tokens", type=int, default=80000)
    ap.add_argument("--tolerance", type=float, default=0.15)
    ap.add_argument("--min-qa", type=int, default=None,
                    help="drop profiles with fewer than this many dialogue QAs "
                         "(default: 20 for per_profile [Mem-alpha rule], 1 for multi)")
    ap.add_argument("--out", default=None, help="output path stem (default: <data-dir>/bundles_<mode>)")
    args = ap.parse_args()

    stats_path = args.stats or os.path.join(args.data_dir, "annotation_token_stats.json")
    out_stem = args.out or os.path.join(args.data_dir, f"bundles_{args.mode}")
    if args.min_qa is None:
        args.min_qa = 20 if args.mode == "per_profile" else 1

    tokens, tokenizer = load_profile_tokens(stats_path)
    existing = load_existing_chunks(args.data_dir)
    speakers = load_profile_speakers(args.data_dir)
    ordered_keys = load_ordered_dialogue_keys(args.perltmem)
    dialogue_qas = load_dialogue_qas(args.perltqa)

    # profiles present on disk with token counts
    all_profiles = sorted(set(tokens) & set(existing))
    # per-profile QA (dialogues only)
    qa_by_profile = {p: extract_qas(p, dialogue_qas.get(p, []), existing.get(p, set()))
                     for p in all_profiles}
    qa_profiles = sorted(p for p in all_profiles if qa_by_profile[p])

    # keep only profiles with enough dialogue QAs (drops QA-less / low-QA profiles)
    include = sorted(p for p in all_profiles if len(qa_by_profile[p]) >= args.min_qa)
    dropped = len(all_profiles) - len(include)
    if not include:
        raise SystemExit(f"no profiles with >= {args.min_qa} dialogue QAs")

    # ── form bundles ──
    if args.mode == "per_profile":
        groups = [[p] for p in include]
    else:
        groups = bundle_multi(include, tokens, speakers, args.target_tokens, args.tolerance)
        groups = [g for g in groups if g]

    # ── assemble manifest ──
    bundles = []
    for bid, profs in enumerate(groups):
        prof_entries, bundle_qa = [], []
        for p in profs:
            chunks = build_chunks(p, ordered_keys.get(p, []), existing.get(p, set()))
            prof_entries.append({
                "profile": p,
                "episode_id": f"bundle{bid:02d}_{p}",   # <- Season+Episode analog
                "tokens": tokens[p],
                "num_chunks": len(chunks),
                "chunks": chunks,                         # <- chunk == dialogue folder
            })
            bundle_qa.extend(qa_by_profile.get(p, []))
        # collision report (names shared by >=2 profiles in the bundle)
        nm_holder = defaultdict(list)
        for p in profs:
            for nm in speakers.get(p, set()):
                nm_holder[nm].append(p)
        collisions = {nm: sorted(ps) for nm, ps in nm_holder.items() if len(ps) >= 2}
        bundles.append({
            "bundle_id": bid,
            "conv_prefix": f"bundle{bid:02d}",           # <- Season analog
            "num_profiles": len(profs),
            "total_tokens": sum(tokens[p] for p in profs),
            "num_qa": len(bundle_qa),
            "profiles": prof_entries,
            "collisions": dict(sorted(collisions.items(), key=lambda kv: -len(kv[1]))),
            "qa": bundle_qa,
        })

    meta = {
        "mode": args.mode,
        "min_qa": args.min_qa,
        "profiles_with_any_qa": len(qa_profiles),
        "profiles_dropped": dropped,
        "target_tokens": args.target_tokens if args.mode == "multi" else None,
        "tolerance": args.tolerance if args.mode == "multi" else None,
        "tokenizer": tokenizer,
        "num_bundles": len(bundles),
        "num_profiles": sum(b["num_profiles"] for b in bundles),
        "num_qa": sum(b["num_qa"] for b in bundles),
        "data_dir": args.data_dir,
        "evidence_granularity": "chunk",
        "qa_memory_types": ["dialogues"],
    }
    manifest = {"meta": meta, "bundles": bundles}

    with open(out_stem + ".json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(out_stem + ".jsonl", "w") as f:
        for b in bundles:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")

    # ── report ──
    tks = [b["total_tokens"] for b in bundles]
    qas = [b["num_qa"] for b in bundles]
    print("=" * 64)
    print(f"mode={args.mode}  min_qa={args.min_qa}  tokenizer={tokenizer}")
    print("=" * 64)
    print(f"  profiles with any QA: {len(qa_profiles)}")
    print(f"  profiles kept (>= {args.min_qa} QA): {meta['num_profiles']}  (dropped {dropped})")
    print(f"  bundles            : {len(bundles)}")
    print(f"  total QA (dialogues): {meta['num_qa']}")
    print(f"  QA/bundle          : min={min(qas)} mean={sum(qas)//len(qas)} max={max(qas)}")
    print(f"  bundles with 0 QA  : {sum(1 for q in qas if q == 0)}/{len(bundles)}")
    print(f"  tokens/bundle      : min={min(tks):,} mean={sum(tks)//len(tks):,} max={max(tks):,}")
    if args.mode == "multi":
        coll = sum(len(b["collisions"]) for b in bundles)
        print(f"  collided names sum : {coll} | collision-free bundles: {sum(1 for b in bundles if not b['collisions'])}/{len(bundles)}")
    print(f"\n  -> {out_stem}.json / .jsonl")


if __name__ == "__main__":
    main()
