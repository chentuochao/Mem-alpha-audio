#!/usr/bin/env python3
"""eval_tg_cache.py — grade the precomputed T/G-probe cache and print stage accuracy.

`diagnostic/precompute_tg_probes.py` stores ONLY the raw server re-answers in
`tg_probe_cache.json`, keyed by a content hash (no gold answers, no question text).
This script grades those cached answers WITHOUT touching the server:

  1. Rebuild the exact jobs precompute built — iterate the qa_file, normalize the
     gt_source refs (perltqa profile/session -> chunk folder), build the T (and,
     with --run_golden, G) dialogue context and QA prompt for each question.
  2. Recompute each cache key with the SAME probe_errors._tg_cache_key, look up the
     cached response, and grade it against the QA's gold answer using the SAME grader
     probe_errors uses (--scorer keyword | llm_judge).
  3. Print per-stage accuracy for T (transcript) and G (gold).

Because the cache stores only responses (scorer-independent), you can grade the same
cache with either scorer for free. Every KEY-AFFECTING argument
(--data_root/--parquet/--server_url/--data_source/--max_tokens) MUST match the values
used at precompute time, or the keys won't be found (reported as "missing").

Usage:
    PYTHONPATH=diagnostic python diagnostic/eval_tg_cache.py \
        --qa_file    outputs/perltqa_data/GPT-5.6_perltqa_bundles/hundle0/bundle1_qa_memalpha_shortened_unified.jsonl \
        --data_root  outputs/step3_perltqa/bundle_0 \
        --parquet    outputs/step3_perltqa/bundle_0/dataset_pred_name_bundle_0.parquet \
        --data_source perltqa \
        --scorer     keyword \
        [--run_golden] [--out <path.json>]
"""
import os
import json
import argparse
from collections import Counter

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from data_utils import load_qa, DialogResolver
import probe_errors as pe
from probe_errors import (
    load_query_prompt, build_qa_prompt,
    load_chunk_folder_map, evidence_chunk_idxs,
    chunk_dialog, chunk_ctx, normalize_qa_sources,
    _load_tg_cache, _tg_cache_key, _CTX_KEY,
)
from precompute_tg_probes import full_question


_KIND_LABEL = {"t": "T-probe (transcript)", "g": "G-probe (gold, ceiling)"}


def make_score_fn(args):
    """Return score(response, gold, question) -> float in [0, 1], matching how
    probe_errors grades each stage. A stage is 'correct' iff the score is 1.0
    (== evaluate_agent_results.py's judgment=='correct'). perltqa keyword can return
    a partial fraction, so we report both strict accuracy and mean score."""
    ds = args.data_source or ""
    if "perltqa" in ds:
        if args.scorer == "llm_judge":
            judge = pe._PerltqaJudge()
            return lambda resp, gold, q: judge.score(q, gold, resp)
        return lambda resp, gold, q: pe._perltqa_score(resp, gold)
    return lambda resp, gold, q: 1.0 if pe.is_correct(resp, gold) else 0.0


def main():
    ap = argparse.ArgumentParser(description="Grade the precomputed T/G-probe cache.")
    ap.add_argument("--qa_file", default="outputs/step3_anony/qas/merged_qa_anoy.jsonl")
    ap.add_argument("--data_root", default="outputs/step3_anony/S01_S03_Clean_Anoy",
                    help="Dialogue root with per-chunk parsed_dialog_{gt,pred}.json.")
    ap.add_argument("--parquet", default=None,
                    help="step3 parquet with the chunk_folders map (auto-discovered in "
                         "--data_root if omitted). Used only for evidence gating.")
    # server_url / max_tokens / data_source are part of the cache key: they must match
    # the precompute run even though no server is contacted here.
    ap.add_argument("--server_url", default="http://127.0.0.1:5005/batch_process",
                    help="MUST match the precompute run (it is part of the cache key); "
                         "no request is sent.")
    ap.add_argument("--data_source", default="seamlessinteraction_options")
    ap.add_argument("--prompts_yaml", default="config/prompts_wrt_datasource.yaml")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--scorer", choices=["keyword", "llm_judge"], default="keyword",
                    help="perltqa grading: keyword containment (default, no API) or "
                         "LLM-as-judge (needs QWEN_URL / QWEN_MODEL_NAME / "
                         "OPENROUTER_API_KEY). Ignored for multiple-choice datasources.")
    ap.add_argument("--run_golden", action="store_true",
                    help="Also grade the G-probe (gold-dialogue) answers.")
    ap.add_argument("--cache", default=None,
                    help="T/G cache json to grade (default: <data_root>/tg_probe_cache.json).")
    ap.add_argument("--out", default=None,
                    help="Optional path to dump per-question grades as json.")
    args = ap.parse_args()

    cache_path = args.cache or os.path.join(args.data_root, "tg_probe_cache.json")
    cache = _load_tg_cache(cache_path)
    if not cache:
        raise SystemExit(f"No cache entries in {cache_path}; run "
                         f"diagnostic/precompute_tg_probes.py first.")

    kinds = ["t", "g"] if args.run_golden else ["t"]

    qa_items = load_qa(args.qa_file)
    normalize_qa_sources(qa_items, args.data_source)
    resolver = DialogResolver([args.data_root])
    query_prompt = load_query_prompt(args.data_source, args.prompts_yaml)
    folder2idx, parquet_used = load_chunk_folder_map(args.parquet, args.data_root)
    score_fn = make_score_fn(args)

    print(f"[eval] qa_file={args.qa_file}  questions={len(qa_items)}")
    print(f"[eval] cache={cache_path} ({len(cache)} entries)  parquet={parquet_used}")
    print(f"[eval] data_source={args.data_source}  scorer="
          f"{args.scorer if 'perltqa' in (args.data_source or '') else 'choice'}  "
          f"kinds={kinds}")

    stats = {k: {"graded": 0, "missing": 0, "correct": 0, "score_sum": 0.0}
             for k in kinds}
    gated_out = 0
    per_q = []      # optional per-question dump

    for qa in qa_items:
        idxs, _, _ = evidence_chunk_idxs(qa, folder2idx)
        if not idxs:                                # mirror precompute's gating
            gated_out += 1
            continue
        gold = qa.get("answer", "") or ""
        question = (qa.get("question", "") or "").split("\n")[0]
        rec = {"_qa_prompt": build_qa_prompt(full_question(qa), query_prompt)}
        # gt_source is echoed for debugging; note its .file has been rewritten by
        # normalize_qa_sources to the resolved chunk-folder form, so you can see the
        # exact evidence folder each grade was localized to.
        row = {"question": question, "gold_answer": gold,
               "gt_source": qa.get("gt_source")}
        for kind in kinds:
            texts, _ = chunk_dialog(qa, resolver, pred=(kind == "t"))
            rec[_CTX_KEY[kind]] = chunk_ctx(texts, kind)
            key = _tg_cache_key(args, rec, kind)
            entry = cache.get(key)
            if entry is None:
                stats[kind]["missing"] += 1
                row[kind] = {"status": "missing"}
                continue
            resp = entry.get("response", "")
            s = float(score_fn(resp, gold, question))
            stats[kind]["graded"] += 1
            stats[kind]["score_sum"] += s
            if s >= 1.0:
                stats[kind]["correct"] += 1
            row[kind] = {"score": s, "correct": s >= 1.0, "response": resp[:200]}
        per_q.append(row)

    _print_summary(args, stats, kinds, gated_out, len(per_q))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"cache": cache_path, "data_source": args.data_source,
                       "scorer": args.scorer, "stats": stats,
                       "gated_out_no_evidence": gated_out,
                       "per_question": per_q}, f, indent=2, ensure_ascii=False)
        print(f"\nWrote per-question grades -> {args.out}  ({len(per_q)} questions)")


def _print_summary(args, stats, kinds, gated_out, n_probed):
    print("\n================ T/G CACHE EVALUATION ================")
    print(f"questions with evidence : {n_probed}")
    print(f"gated out (no evidence) : {gated_out}")
    print("---- stage accuracy ----")
    print(f"  {'stage':<26}{'graded':>7}  {'missing':>7}  "
          f"{'correct':>16}  {'mean score':>10}")
    for k in kinds:
        s = stats[k]
        g, miss = s["graded"], s["missing"]
        acc = s["correct"] / g if g else 0.0
        mean = s["score_sum"] / g if g else 0.0
        print(f"  {_KIND_LABEL[k]:<26}{g:>7}  {miss:>7}  "
              f"{s['correct']:>6}/{g:<4} ({100*acc:5.1f}%)  {mean:>9.3f}")
    if any(stats[k]["missing"] for k in kinds):
        print("  NOTE: 'missing' entries are questions whose cache key was not found — "
              "rerun precompute_tg_probes.py with the SAME "
              "data_root/parquet/server_url/data_source/max_tokens (and --run_golden "
              "for G).")
    print("======================================================\n")


if __name__ == "__main__":
    main()
