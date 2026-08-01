#!/usr/bin/env python3
"""Filter a perltqa QA list down to the questions the G-probe answered correctly.

Pipeline: probe/precompute -> eval_tg_cache.py (with --run_golden --out ...) -> THIS.

eval_tg_cache.py's --out json holds a `per_question` list; each row carries the
question, gold answer, the (normalized) gt_source, and the per-stage grades
(`t`/`g` = {score, correct, response} or {status:"missing"}). Only questions that
were localized to an evidence chunk appear there at all (eval gates the rest out).

This selects the rows where:
  * the chosen stage (default G = gold-dialogue ceiling) is CORRECT, and
  * the question actually has an evidence ref,
then writes the MATCHING ORIGINAL QA records (schema untouched — evidence_chunks /
gt_source and all) to <out_dir>/<same filename as --qa_file>, so the filtered set
is a drop-in replacement for the input QA list.

Matching the eval rows back to the original records is by (question, answer), the
same fields eval_tg_cache echoes, so both QA schemas (gt_source and qa_multi's
evidence_chunks) join identically.

Usage:
  python diagnostic/filter_qa_by_gprobe.py \
      --eval_json outputs/step3_perltqa/bundle_0/tg_cache_eval_multi.json \
      --qa_file   outputs/perltqa_data/qa_multi/bundle_0/qa.jsonl \
      --out_dir   outputs/perltqa_data/qa_multi/bundle_0_filterd
"""
import os
import json
import argparse


def load_qa(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def qa_key(question, answer):
    """Join key shared by eval rows and original records: first question line +
    answer, both stripped (perltqa questions are single-line; this stays robust
    if options were ever appended)."""
    q = (question or "").split("\n")[0].strip()
    return (q, (answer or "").strip())


def has_evidence(qa):
    """True if the original QA carries a usable evidence ref, under either schema:
    gt_source.sources / evidence_turns, or the flat qa_multi evidence_chunks list."""
    gt = qa.get("gt_source") or {}
    if gt.get("sources") or "evidence_turns" in gt:
        return True
    return bool(qa.get("evidence_chunks"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_json", required=True,
                    help="eval_tg_cache.py --out json (run it with --run_golden so a "
                         "'g' grade exists).")
    ap.add_argument("--qa_file", required=True,
                    help="Original QA jsonl to pull full records from (schema preserved).")
    ap.add_argument("--out_dir", required=True,
                    help="Directory to write the filtered QA list into (created if "
                         "needed); the file keeps --qa_file's basename.")
    ap.add_argument("--stage", default="g", choices=["g", "t"],
                    help="Which probe stage must be correct (default: g = gold ceiling).")
    args = ap.parse_args()

    with open(args.eval_json) as f:
        ev = json.load(f)
    rows = ev.get("per_question", [])
    if not rows:
        raise SystemExit(f"No 'per_question' rows in {args.eval_json}.")

    stage = args.stage
    passed = set()
    n_graded = n_missing = n_correct = 0
    for r in rows:
        sd = r.get(stage)
        if not isinstance(sd, dict):
            continue                      # stage absent (e.g. --run_golden not used)
        if sd.get("status") == "missing":
            n_missing += 1
            continue
        n_graded += 1
        if sd.get("correct") and r.get("gt_source"):   # G-correct AND has evidence ref
            n_correct += 1
            passed.add(qa_key(r.get("question"), r.get("gold_answer")))

    if n_graded == 0:
        raise SystemExit(
            f"No '{stage}'-stage grades in {args.eval_json} "
            f"({n_missing} missing). Re-run eval_tg_cache.py with --run_golden "
            f"(and matching --scorer) first.")

    qa = load_qa(args.qa_file)
    kept = [q for q in qa
            if has_evidence(q) and qa_key(q.get("question"), q.get("answer")) in passed]

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, os.path.basename(args.qa_file))
    with open(out_path, "w") as f:
        for q in kept:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"eval rows            : {len(rows)}")
    print(f"{stage}-stage graded  : {n_graded}  (missing {n_missing})")
    print(f"{stage}-correct + evid: {n_correct}")
    print(f"kept original records : {len(kept)} / {len(qa)}")
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
