#!/usr/bin/env python3
"""Precompute the T-probe (transcript) and, optionally, G-probe (gold dialogue)
re-answers ONCE per DATA_ROOT and store them in a shared JSON cache.

Why this exists
---------------
probe_errors.py runs four counterfactual re-answer stages per question: C
(constructed memory), S (retrieved subset), T (raw ASR transcript), G (gold
dialogue). Only C and S depend on the per-instance memory. The T and G contexts are
built purely from the dialogue files under DATA_ROOT plus the question, and the server
re-answer is deterministic (temperature 0) — so the SAME transcript/gold answer is
valid for every instance dir, every seed, and every compression variant that shares
that DATA_ROOT (e.g. `..._Clean_Anoy...` and its `_comp_x{3,4,5}` siblings).

This script computes those instance-independent answers once and writes them to
`<DATA_ROOT>/tg_probe_cache.json` (override with --cache). probe_errors.py then loads
T/G from that cache (`--tg_probe_cache`) and only runs C/S per instance. The cache key
is a content hash of (kind, prompt, dialogue ctx, max_tokens, server_url, data_source)
computed by the SAME `_tg_cache_key` probe_errors uses, so keys match exactly.

It is fully DATA_ROOT-scoped: it iterates the qa_file directly (no memory / no agent
run dir needed) and reconstructs each full question exactly as run_qa_evaluation.py
does, so it covers every question any instance could ask — a guaranteed superset of
what probe_errors will request (which is why probe_errors can hard-error on a miss).
Idempotent: only misses hit the server.

Usage
-----
    PYTHONPATH=diagnostic python diagnostic/precompute_tg_probes.py \
        --qa_file   outputs/step3_anony/qas/merged_qa_anoy.jsonl \
        --data_root outputs/step3_anony/S01_S03_Clean_Anoy \
        --parquet   outputs/step3_anony/S01_S03_Clean_Anoy/dataset_pred_name_Season01_Clean_Anoy.parquet \
        --server_url http://127.0.0.1:5005/batch_process \
        [--run_golden]
"""
import os
import argparse

from data_utils import load_qa, DialogResolver
from probe_errors import (
    load_query_prompt, build_qa_prompt,
    load_chunk_folder_map, evidence_chunk_idxs,
    chunk_dialog, chunk_ctx, qa_batch,
    normalize_qa_sources,
    _load_tg_cache, _save_tg_cache, _tg_cache_key,
    _CTX_KEY,
)


def full_question(qa):
    """Reconstruct the full question text exactly as run_qa_evaluation.py builds it
    (run_qa_evaluation.py:70-74): bare question + each option ('A. ...') + a trailing
    'C. not sure'. This is what ends up in results.json and what probe_errors.py hashes
    into the T/G cache key, so reconstructing it here makes the keys match byte-for-byte.
    Datasets without an `options` field keep the bare question."""
    q = qa.get("question", "")
    options = qa.get("options")
    if options:
        for k, v in options.items():
            q += f"\n{k}. {v}"
        q += "\nC. not sure"
    return q


def main():
    ap = argparse.ArgumentParser(description="Precompute T/G-probe answers per DATA_ROOT.")
    ap.add_argument("--qa_file", default="outputs/step3_anony/qas/merged_qa_anoy.jsonl")
    ap.add_argument("--data_root", default="outputs/step3_anony/S01_S03_Clean_Anoy",
                    help="Dialogue root with per-chunk parsed_dialog_{gt,pred}.json.")
    ap.add_argument("--parquet", default=None,
                    help="step3 parquet with the chunk_folders map (auto-discovered in "
                         "--data_root if omitted). Used only for evidence gating.")
    ap.add_argument("--server_url", default="http://127.0.0.1:5005/batch_process")
    ap.add_argument("--data_source", default="seamlessinteraction_options")
    ap.add_argument("--prompts_yaml", default="config/prompts_wrt_datasource.yaml")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=64,
                    help="Jobs per server POST. Smaller batches keep each request under "
                         "--timeout and let the cache be saved incrementally (so a "
                         "timeout/crash never loses finished answers; just rerun to "
                         "resume). Default 64.")
    ap.add_argument("--timeout", type=int, default=1200,
                    help="Per-POST read timeout in seconds (default 1200). One giant "
                         "batch can blow past this and lose everything — hence batching.")
    ap.add_argument("--run_golden", action="store_true",
                    help="Also precompute the G-probe (gold-dialogue) answers.")
    ap.add_argument("--cache", default=None,
                    help="Output cache json (default: <data_root>/tg_probe_cache.json).")
    args = ap.parse_args()

    cache_path = args.cache or os.path.join(args.data_root, "tg_probe_cache.json")
    kinds = ["t", "g"] if args.run_golden else ["t"]

    qa_items = load_qa(args.qa_file)
    normalize_qa_sources(qa_items, args.data_source)   # perltqa: profile/session -> chunk folder
    resolver = DialogResolver([args.data_root])
    query_prompt = load_query_prompt(args.data_source, args.prompts_yaml)
    folder2idx, parquet_used = load_chunk_folder_map(args.parquet, args.data_root)

    if not qa_items:
        raise SystemExit(f"No questions found in {args.qa_file}")
    print(f"[precompute] qa_file={args.qa_file}  questions={len(qa_items)}")
    print(f"[precompute] data_root={args.data_root}  parquet={parquet_used}  "
          f"kinds={kinds}  server={args.server_url}")

    # Build the (kind, ctx, prompt) job list straight from the qa_file, mirroring
    # probe_errors' gating under --full_qa (skip questions with no gold evidence).
    jobs = []            # (key, ctx_dict, qa_prompt)
    gated = {"no_gold_evidence": 0, "ok": 0}
    for qa in qa_items:
        idxs, _, _ = evidence_chunk_idxs(qa, folder2idx)
        if not idxs:
            gated["no_gold_evidence"] += 1
            continue
        gated["ok"] += 1
        rec = {"_qa_prompt": build_qa_prompt(full_question(qa), query_prompt)}
        for kind in kinds:
            texts, _ = chunk_dialog(qa, resolver, pred=(kind == "t"))
            rec[_CTX_KEY[kind]] = chunk_ctx(texts, kind)
            key = _tg_cache_key(args, rec, kind)
            jobs.append((key, rec[_CTX_KEY[kind]], rec["_qa_prompt"]))
    print(f"[precompute] gated: ok={gated['ok']}  "
          f"no_gold_evidence={gated['no_gold_evidence']}  -> {len(jobs)} probe jobs")

    # Only compute misses (idempotent). Dedup keys within this run too.
    cache = _load_tg_cache(cache_path)
    unique_keys = {key for key, _, _ in jobs}
    reused = sum(1 for key in unique_keys if key in cache)

    misses, seen = [], set()
    for key, ctx, prompt in jobs:
        if key in cache or key in seen:
            continue
        seen.add(key)
        misses.append((key, ctx, prompt))

    if misses:
        bs = max(1, args.batch_size)
        print(f"[precompute] querying server for {len(misses)} new answers "
              f"({reused} already cached) in batches of {bs} (timeout {args.timeout}s) ...")
        done = 0
        for i in range(0, len(misses), bs):
            chunk = misses[i:i + bs]
            responses = qa_batch(args.server_url,
                                 [(ctx, prompt) for _, ctx, prompt in chunk],
                                 args.max_tokens, timeout=args.timeout)
            for (key, _, _), resp in zip(chunk, responses):
                cache[key] = {"response": resp}
            # Persist after every batch so a later timeout/crash keeps finished answers
            # (rerun resumes from here — only the still-missing keys hit the server).
            _save_tg_cache(cache_path, cache)
            done += len(chunk)
            print(f"[precompute]   {done}/{len(misses)} computed  -> saved {cache_path}")
        print(f"[precompute] computed {len(misses)} / reused {reused}  -> wrote {cache_path}")
    else:
        print(f"[precompute] computed 0 / reused {reused}  -> cache already complete: {cache_path}")


if __name__ == "__main__":
    main()
