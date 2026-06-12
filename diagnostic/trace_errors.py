#!/usr/bin/env python3
"""
trace_errors.py — Cascade error attribution for the Mem-alpha memory QA pipeline.

Given one constructed instance (an output dir with results.json / agent_state.json
/ chunks_and_function_calls.json) plus the QA file with gold source-evidence, this
attributes every *failed* question to the earliest stage where the gold evidence
disappears:

    GATE         -> annotation / judge problems (excluded from system blame)
    CONSTRUCTION -> evidence in raw transcript but NOT in the stored memory
                    (extraction vs update/deletion)
    RETRIEVAL    -> evidence in the store but NOT in retrieved_memory
    RESPONSE     -> evidence was retrieved but the model still answered wrong

This mirrors the MemTrace taxonomy (trace_err.pdf): the decisive error is the
*earliest* operation whose correct output would have rescued the answer.

Layout:
    matching.py    — evidence/memory similarity (lexical + embedding + LLM judge)
    data_utils.py  — loading & parsing (QA, dialog, memory flattening, scoring)
    trace_errors.py — this file: the cascade orchestrator + CLI

Usage:
    python diagnostic/trace_errors.py \
        --instance_dir data/<run_name>/0 \
        --qa_file outputs/tmp_folder_for_95_qs/merged_95.jsonl \
        --dialog_root outputs/bazinga/TheBigBangTheory/Season1

    (--out defaults to <instance_dir>/error_trace.json)
"""

import os
import sys
import json
import argparse
from collections import Counter

# Make sibling modules importable whether run as a script or as a module.
# Optional: load .env so OPENAI/OPENROUTER keys are picked up automatically.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from matching import (
    EmbeddingMatcher, LLMJudge, lexical_score,
    present, evidence_rank, LEX_TAU, EMB_TAU,
)
from data_utils import (
    extract_choice, gold_letter, load_qa, DialogResolver,
    evidence_texts, memory_records, retrieved_records, fix_space_in_text,
)


# --------------------------------------------------------------------------- #
# Construction sub-classification
# --------------------------------------------------------------------------- #
def construction_subtype(missing_turns, chunks_fc):
    """Decide extraction vs update/deletion for the turns missing from the store.

    For each missing turn:
      - if it appears in some chunk's raw_chunk (so the agent saw it) but in no
        function_call write -> extraction (never captured).
      - if it appears in a function_call's arguments (it WAS written) yet is gone
        from the final store -> update/deletion (later overwritten/removed).
    """
    written, in_raw = 0, 0
    for t in missing_turns:
        saw_raw = any(lexical_score(t, ch.get("raw_chunk", "")) >= LEX_TAU for ch in chunks_fc)
        wrote = False
        for ch in chunks_fc:
            for fc in ch.get("function_calls", []):
                args = fc.get("function_call", {}).get("arguments", "")
                content = args if isinstance(args, str) else json.dumps(args)
                if lexical_score(t, content) >= LEX_TAU:
                    wrote = True
                    break
            if wrote:
                break
        in_raw += int(saw_raw)
        written += int(wrote)
    subtype = "update/deletion" if written > 0 else "extraction"
    return subtype, {"missing_turns": len(missing_turns),
                     "seen_in_raw": in_raw, "written_then_lost": written}


# --------------------------------------------------------------------------- #
# Main cascade
# --------------------------------------------------------------------------- #
def trace(args):
    qa_items = load_qa(args.qa_file) # load QA items from the QA file
    resolver = DialogResolver(args.dialog_root) # load dialog resolver from the dialog root

    with open(os.path.join(args.instance_dir, "results.json")) as f: # load results from the results file
        results = json.load(f)
    with open(os.path.join(args.instance_dir, "agent_state.json")) as f: # load agent state from the agent state file
        state = json.load(f)
    chunks_fc = []
    cf_path = os.path.join(args.instance_dir, "chunks_and_function_calls.json")
    if os.path.exists(cf_path):
        with open(cf_path) as f:
            chunks_fc = json.load(f)
        # Normalize the construction trace at load so it tokenizes like the
        # (already-normalized) evidence turns and memory units.
        for ch in chunks_fc:
            ch["raw_chunk"] = fix_space_in_text(ch.get("raw_chunk", ""))
            for fc in ch.get("function_calls", []):
                call = fc.get("function_call", {})
                if isinstance(call.get("arguments"), str):
                    call["arguments"] = fix_space_in_text(call["arguments"])
                    
    # global memory records from agent_states.json
    store_records, episodic_records = memory_records(state)

    emb = EmbeddingMatcher()
    judge = LLMJudge()
    print(f"[matcher] lexical=on  embedding={'on' if emb.enabled else 'off'}  "
          f"llm_judge={'on' if judge.enabled else 'off'}")

    # Map result -> QA by question prefix (results question = qa question + options).
    qa_by_q = {qa["question"].strip(): qa for qa in qa_items}

    def find_qa(result_q):
        rq = result_q.strip()
        if rq in qa_by_q:
            return qa_by_q[rq]
        for q, qa in qa_by_q.items():
            if rq.startswith(q):
                return qa
        return None

    findings = []

    #iterate over the results, each QA item 

    for res in results:
        qa = find_qa(res["question"])
        pred = extract_choice(res.get("response", ""))
        gold = gold_letter(res.get("answer", ""))
        correct = (pred is not None and gold is not None and pred == gold)

        rec = {
            "question": res["question"].split("\n")[0],
            "options": (qa.get("options") if qa else None),
            "gold": gold, "pred": pred, "correct": correct,
        }

        if correct:
            rec["stage"] = "correct"
            findings.append(rec)
            continue

        # --- GATE: judge / parse ---
        if pred is None:
            rec["stage"] = "gate:no_parseable_answer"
            findings.append(rec)
            continue

        # --- GATE: annotation (no usable evidence) ---
        if qa is None:
            rec["stage"] = "gate:qa_not_found"
            findings.append(rec)
            continue
        ev_list, unresolved = evidence_texts(qa, resolver, args.min_turn_words)
        ev_blob = " ".join(ev_list)
        if unresolved and not ev_list:
            rec["stage"] = "gate:evidence_file_unavailable"
            rec["detail"] = {"unresolved_files": unresolved}
            findings.append(rec)
            continue
        if not ev_blob.strip():
            rec["stage"] = "gate:no_gold_evidence"
            findings.append(rec)
            continue
        rec["evidence"] = ev_list
        if unresolved:
            rec["partial_evidence_unresolved"] = unresolved

        # --- STAGE: construction (is evidence in the stored memory?) ---
        in_store, store_info = present(ev_list, store_records, emb, judge)
        if not in_store:
            subtype, sub_info = construction_subtype(store_info["missing"], chunks_fc)
            rec["stage"] = f"construction:{subtype}"
            rec["detail"] = {"store_coverage": store_info, **sub_info}
            findings.append(rec)
            continue

        # --- STAGE: retrieval (was evidence shown to the QA model?) ---
        retr_records, _ = retrieved_records(res.get("retrieved_memory"))
        in_retrieved, retr_info = present(ev_list, retr_records, emb, judge)
        if not in_retrieved:
            best_rank, ranks = evidence_rank(ev_list, episodic_records)
            rec["stage"] = "retrieval"
            rec["detail"] = {
                "store_coverage": store_info,
                "retrieved_coverage": retr_info,
                "best_episodic_rank": best_rank,
                "evidence_episodic_ranks": ranks,
                "retrieved_episodic_count": len(res.get("retrieved_memory", {}).get("episodic", []) or []),
            }
            findings.append(rec)
            continue

        # --- STAGE: response (evidence retrieved, answer still wrong) ---
        rec["stage"] = "response"
        rec["detail"] = {"store_coverage": store_info, "retrieved_coverage": retr_info}
        findings.append(rec)

    _summarize_and_save(args, findings, emb, judge)


def _summarize_and_save(args, findings, emb, judge):
    total = len(findings)
    wrong = [f for f in findings if not f["correct"]]
    buckets = Counter(f["stage"] for f in wrong)

    print("\n================ ERROR TRACE SUMMARY ================")
    print(f"instance      : {args.instance_dir}")
    print(f"questions      : {total}")
    print(f"correct        : {total - len(wrong)}")
    print(f"failed         : {len(wrong)}")
    print("---- failure attribution ----")
    order = ["gate:no_parseable_answer", "gate:qa_not_found", "gate:no_gold_evidence",
             "gate:evidence_file_unavailable",
             "construction:extraction", "construction:update/deletion",
             "retrieval", "response"]
    for k in order:
        if buckets.get(k):
            print(f"  {k:32s} {buckets[k]:3d}  ({100*buckets[k]/max(len(wrong),1):.0f}%)")
    for k, v in buckets.items():
        if k not in order:
            print(f"  {k:32s} {v:3d}")
    print("=====================================================\n")

    out = {
        "instance_dir": args.instance_dir,
        "matcher": {"lexical": True, "embedding": emb.enabled, "llm_judge": judge.enabled,
                    "thresholds": {"lexical": LEX_TAU, "embedding": EMB_TAU}},
        "summary": {"total": total, "correct": total - len(wrong),
                    "failed": len(wrong), "attribution": dict(buckets)},
        "findings": findings,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote per-question trace -> {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Cascade error attribution for Mem-alpha QA.")
    p.add_argument("--instance_dir",
                   default="memory_result/minimal_memory_agent_qwen_converted_YuWangX_Memalpha-4B_seamlessinteraction_options_Season1_vibevoice_gt_name_no_thinking_tokens_2048/0",
                   help="Dir with results.json / agent_state.json / chunks_and_function_calls.json")
    p.add_argument("--qa_file", default="outputs/tmp_folder_for_95_qs/merged_95.jsonl",
                   help="QA jsonl with gt_source.evidence_turns")
    p.add_argument("--dialog_root", nargs="+",
                   default=["outputs/bazinga/TheBigBangTheory/Season1"],
                   help="Dir(s) searched recursively for per-question evidence dialog files (by basename). "
                        "Must use REAL speaker names to match the memory store (not the anonymized P0001 set). "
                        "When multiple roots share a basename, the first listed wins.")
    p.add_argument("--min_turn_words", type=int, default=3,
                   help="Drop evidence turns whose utterance has fewer than this many words "
                        "(too short to match reliably). Set 0 to disable.")
    p.add_argument("--out", default=None, help="Output json (default: <instance_dir>/error_trace.json)")
    args = p.parse_args()
    if args.out is None:
        args.out = os.path.join(args.instance_dir, "error_trace.json")
    return args


if __name__ == "__main__":
    trace(parse_args())
