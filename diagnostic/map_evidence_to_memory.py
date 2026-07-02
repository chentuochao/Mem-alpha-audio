#!/usr/bin/env python3
"""
map_evidence_to_memory.py — Provenance map from QA evidence turns to memory ids.

Unlike the substring matcher in probe_errors.py (which greps the gold evidence
text against every raw_chunk), this uses EXACT folder provenance:

    evidence turn  --(gt_source.file = "<episode>/CHUNK_N/...")-->  source folder
    source folder  --(chunk_folders column in the parquet)------->  chunk_idx
    chunk_idx      --(new_memory_insert ids in chunks_and_function_calls)->  memory ids

The parquet's `chunk_folders` list is index-aligned with the `chunks` list the
memory agent consumed, so chunk_folders[chunk_idx] is exactly the origin folder of
`chunk_idx` in chunks_and_function_calls.json. (Add that column by running the
patched prepare_data/prepare_audio_parquet.py; or pass --data_dir to reconstruct
the same sorted-glob ordering on the fly.)

Granularity: every memory unit is inserted from a whole chunk's content, so a chunk
has no per-turn attribution — every evidence turn in a chunk maps to ALL units
inserted for that chunk (hence "one or more memory ids" per turn).

Usage:
    python diagnostic/map_evidence_to_memory.py \
        --results_dir agents/new_qa_results \
        --qa_file outputs/tmp_folder_for_95_qs_chunked/merged_95_chunked.jsonl \
        --parquet outputs/step3_new/dataset_gt_name_Season01.parquet

    # if the parquet predates the chunk_folders column, reconstruct it from disk:
    python diagnostic/map_evidence_to_memory.py \
        --results_dir agents/new_qa_results \
        --qa_file .../merged_95_chunked.jsonl \
        --data_dir outputs/step3/vibevoice_TheBigBangTheory_predname

    (--out defaults to <results_dir>/evidence_to_memory.json)
"""

import os
import re
import ast
import glob
import json
import argparse
from collections import OrderedDict


# --------------------------------------------------------------------------- #
# chunks_and_function_calls.json  ->  {chunk_idx: {uid: (mtype, text)}}
# --------------------------------------------------------------------------- #
_MTYPE = {"core_memory": "core", "episodic_memory": "episodic",
          "semantic_memory": "semantic"}


def _balanced_dict(s, start):
    """Substring of `s` spanning the balanced {...} that opens at `start`."""
    depth = 0
    for j in range(start, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[start:j + 1]
    return None


def _parse_inserted(tool_result):
    """Pull the {id: content} dict a new_memory_insert returned from its result str.

    tool_result looks like:
      "[tool new_memory_insert executed successfully] -> {'status': 'ok',
       'new_memory': {'cd61': '...'}}"
    The id ('cd61') is exactly the id stored in agent_state.json.
    """
    if not tool_result:
        return {}
    key = tool_result.find("'new_memory'")
    if key == -1:
        return {}
    brace = tool_result.find("{", key)
    blob = _balanced_dict(tool_result, brace) if brace != -1 else None
    if not blob:
        return {}
    try:
        d = ast.literal_eval(blob)
    except Exception:
        return {}
    return d if isinstance(d, dict) else {}


def build_provenance(chunks):
    """{chunk_idx: {uid: (mtype, text)}} — units INSERTED while processing a chunk."""
    prov = {}
    for ch in chunks:
        idx = ch.get("chunk_idx")
        units = {}
        for fc in ch.get("function_calls", []):
            call = fc.get("function_call", {})
            if call.get("name") != "new_memory_insert":
                continue
            try:
                args = json.loads(call.get("arguments", "{}"))
            except Exception:
                args = {}
            mtype = _MTYPE.get(args.get("memory_type", ""), "episodic")
            for uid, text in _parse_inserted(fc.get("tool_result", "")).items():
                units[uid] = (mtype, text)
        prov[idx] = units
    return prov


# --------------------------------------------------------------------------- #
# Final memory (agent_state.json) -> set of surviving unit ids
# --------------------------------------------------------------------------- #
def final_unit_ids(state):
    ids = set()
    for mtype in ("episodic", "semantic"):
        for d in state.get(mtype) or []:
            if isinstance(d, dict) and d:
                ids.add(next(iter(d)))
    return ids


# --------------------------------------------------------------------------- #
# chunk_idx -> source folder ("<episode>/CHUNK_N")
# --------------------------------------------------------------------------- #
def chunk_folders_from_parquet(parquet_path, row):
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    if "chunk_folders" not in df.columns:
        raise KeyError(
            f"{parquet_path} has no 'chunk_folders' column — rebuild it with the "
            "patched prepare_audio_parquet.py, or pass --data_dir to reconstruct.")
    if row >= len(df):
        raise IndexError(f"parquet row {row} out of range (rows={len(df)})")
    return json.loads(df.iloc[row]["chunk_folders"])


def chunk_folders_from_data_dir(data_dir, season_filter=None):
    """Recompute the SAME sorted-glob ordering prepare_audio_parquet.py uses so the
    positions line up with chunk_idx even without the parquet column."""
    folders = []
    for name in ("parsed_dialog_gt.json", "parsed_dialog_pred.json"):
        paths = sorted(glob.glob(os.path.join(data_dir, "*", "*", name)))
        if season_filter:
            paths = [p for p in paths if season_filter in p]
        if paths:
            return [os.path.relpath(os.path.dirname(p), data_dir) for p in paths]
    return folders


# --------------------------------------------------------------------------- #
# QA loading
# --------------------------------------------------------------------------- #
def load_qa(qa_file):
    items = []
    with open(qa_file) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def qa_sources(qa):
    """Normalize gt_source to a list of {file, evidence_turns} (both schemas)."""
    gt = qa.get("gt_source", {}) or {}
    sources = gt.get("sources")
    if sources is None and "evidence_turns" in gt:
        sources = [gt]
    return sources or []


_TURN_RE = re.compile(r"_T(\d+)$")


def _turn_index(turn_id):
    """'S01E01_C003_T000' -> 0 (positional index of the turn within its chunk)."""
    m = _TURN_RE.search(str(turn_id))
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Per-instance mapping
# --------------------------------------------------------------------------- #
def map_instance(instance_dir, chunk_folders, qa_items, dialog_root):
    with open(os.path.join(instance_dir, "chunks_and_function_calls.json")) as f:
        chunks = json.load(f)
    prov = build_provenance(chunks)

    final_ids = set()
    state_path = os.path.join(instance_dir, "agent_state.json")
    if os.path.isfile(state_path):
        with open(state_path) as f:
            final_ids = final_unit_ids(json.load(f))

    folder2idx = {f: i for i, f in enumerate(chunk_folders)}

    # memory_units: uid -> {mtype, chunk_idx, in_final, text}
    memory_units = {}
    chunk_to_memory = {}
    for idx, units in prov.items():
        chunk_to_memory[idx] = list(units.keys())
        for uid, (mtype, text) in units.items():
            memory_units[uid] = {"mtype": mtype, "chunk_idx": idx,
                                 "in_final": uid in final_ids, "text": text}

    turn_to_memory = OrderedDict()
    question_map = []
    unmatched = set()

    for qa in qa_items:
        q_sources = []
        for src in qa_sources(qa):
            f_field = src.get("file", "")
            folder = os.path.dirname(f_field)              # "<episode>/CHUNK_N"
            idx = folder2idx.get(folder)
            if idx is None:
                unmatched.add(folder)
                continue                                   # not in this instance
            mem_ids = chunk_to_memory.get(idx, [])
            turns = src.get("evidence_turns", [])
            turn_entries = []
            for tid in turns:
                turn_to_memory.setdefault(tid, {
                    "folder": folder, "chunk_idx": idx, "memory_ids": mem_ids})
                turn_entries.append({
                    "turn_id": tid,
                    "turn_index": _turn_index(tid),
                    "text": _turn_text(dialog_root, f_field, _turn_index(tid)),
                })
            q_sources.append({
                "file": f_field, "folder": folder, "chunk_idx": idx,
                "memory_ids": mem_ids, "evidence_turns": turn_entries,
            })
        if q_sources:
            question_map.append({
                "question": (qa.get("question", "").split("\n")[0]),
                "category": qa.get("category"),
                "answer": qa.get("answer"),
                "sources": q_sources,
            })

    return {
        "n_chunks": len(chunk_folders),
        "n_memory_units": len(memory_units),
        "n_evidence_turns_mapped": len(turn_to_memory),
        "unmatched_evidence_folders": sorted(unmatched),
        "memory_units": memory_units,
        "chunk_to_memory": {str(k): v for k, v in chunk_to_memory.items()},
        "turn_to_memory": turn_to_memory,
        "question_map": question_map,
    }


_dialog_cache = {}


def _turn_text(dialog_root, file_field, turn_index):
    """Best-effort speaker+text for a turn (None if the dialog file isn't found)."""
    if not dialog_root or turn_index is None:
        return None
    path = os.path.join(dialog_root, file_field)
    if path not in _dialog_cache:
        try:
            with open(path) as f:
                _dialog_cache[path] = json.load(f)
        except Exception:
            _dialog_cache[path] = None
    turns = _dialog_cache[path]
    if not turns or not (0 <= turn_index < len(turns)):
        return None
    t = turns[turn_index]
    return f"{t.get('speaker', '?')}: {t.get('text', '')}"


# --------------------------------------------------------------------------- #
# Instance discovery
# --------------------------------------------------------------------------- #
def find_instances(results_dir):
    """[(name, dir)] for every subdir that holds chunks_and_function_calls.json.

    Accepts either a results root (with numeric instance subdirs like '0') or a
    single instance dir passed directly.
    """
    if os.path.isfile(os.path.join(results_dir, "chunks_and_function_calls.json")):
        return [(os.path.basename(results_dir.rstrip("/")) or "0", results_dir)]
    out = []
    for name in sorted(os.listdir(results_dir),
                       key=lambda n: (0, int(n)) if n.isdigit() else (1, n)):
        d = os.path.join(results_dir, name)
        if os.path.isdir(d) and os.path.isfile(
                os.path.join(d, "chunks_and_function_calls.json")):
            out.append((name, d))
    return out


def main():
    p = argparse.ArgumentParser(
        description="Map QA evidence turns -> memory ids via chunk-folder provenance.")
    p.add_argument("--results_dir", required=True,
                   help="Results folder (e.g. agents/new_qa_results) or a single "
                        "instance dir containing chunks_and_function_calls.json.")
    p.add_argument("--qa_file", required=True,
                   help="Chunked QA jsonl (gt_source.file = '<episode>/CHUNK_N/...').")
    p.add_argument("--parquet", default=None,
                   help="Parquet with the chunk_folders column (chunk provenance).")
    p.add_argument("--data_dir", default=None,
                   help="Fallback: reconstruct chunk_folders from this dialogue root "
                        "using the same sorted-glob order as prepare_audio_parquet.py.")
    p.add_argument("--season_filter", default=None,
                   help="Season substring for --data_dir reconstruction (e.g. Season01).")
    p.add_argument("--dialog_root", default=None,
                   help="Root to resolve gt_source.file for turn text (default: --data_dir).")
    p.add_argument("--out", default=None,
                   help="Output json (default: <results_dir>/evidence_to_memory.json).")
    args = p.parse_args()

    if not args.parquet and not args.data_dir:
        p.error("provide --parquet (with chunk_folders) or --data_dir to reconstruct it")

    qa_items = load_qa(args.qa_file)
    dialog_root = args.dialog_root or args.data_dir
    instances = find_instances(args.results_dir)
    if not instances:
        p.error(f"no instances with chunks_and_function_calls.json under {args.results_dir}")

    out = {"results_dir": args.results_dir, "qa_file": args.qa_file,
           "parquet": args.parquet, "data_dir": args.data_dir, "instances": {}}

    for name, inst_dir in instances:
        row = int(name) if name.isdigit() else 0
        if args.parquet:
            try:
                folders = chunk_folders_from_parquet(args.parquet, row)
            except KeyError as e:
                if not args.data_dir:
                    raise
                print(f"[map] {e}; falling back to --data_dir")
                folders = chunk_folders_from_data_dir(args.data_dir, args.season_filter)
        else:
            folders = chunk_folders_from_data_dir(args.data_dir, args.season_filter)

        res = map_instance(inst_dir, folders, qa_items, dialog_root)
        out["instances"][name] = res
        print(f"[map] instance {name}: chunks={res['n_chunks']} "
              f"units={res['n_memory_units']} "
              f"evidence_turns_mapped={res['n_evidence_turns_mapped']} "
              f"unmatched_folders={len(res['unmatched_evidence_folders'])}")

    out_path = args.out or os.path.join(args.results_dir, "evidence_to_memory.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote evidence->memory map -> {out_path}")


if __name__ == "__main__":
    main()
