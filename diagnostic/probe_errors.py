#!/usr/bin/env python3
"""
probe_errors.py — Behavioral (counterfactual) error attribution for Mem-alpha QA.

A standalone alternative to the matching cascade in `trace_errors_clean.py`.

Instead of asking "does the gold evidence string still APPEAR in the memory /
transcript?" (a matching question, sensitive to LEX_TAU / EMB_TAU / COVERAGE_TAU
and blind to paraphrase), this asks "can the QA model still ANSWER correctly when
fed exactly the memory / transcript derived from the evidence turns?" — a
behavioral question answered by re-running the real QA model on curated contexts.

Provenance, not fuzzy matching, locates the relevant memory: every memory unit is
created by a `new_memory_insert` recorded in chunks_and_function_calls.json, and
the returned unit id (e.g. 'cd61') is the SAME id stored in agent_state.json. So
"the memory constructed from turn T" = the FINAL stored units whose insert lives
in the chunk that contained T. (We use FINAL memory only — no extraction vs
update/deletion split — so a single `construction` bucket.)

Per FAILED question (the real run got it wrong), three oracle QA re-answers:

    C-probe  : final memory units traced to the evidence chunk(s)
                 wrong  -> construction  (if even transcript can't rescue it)
                 right  -> store has it; failure is downstream
    T-probe  : the matched transcript turns (raw ASR + speaker naming)
                 (only run when C-probe failed; right => construction, not ASR)
    rescue   : actual retrieved_memory  UNION  evidence units, re-answered
                 (only run when C-probe passed)
                 right  -> retrieval   (store had it, retriever dropped it)
                 wrong  -> response    (it was shown, model still wrong)

The QA re-answers hit the SAME memory server the real run used (/batch_process),
so the context-serialization, retriever, and model are identical.

Usage:
    PYTHONPATH=. python diagnostic/probe_errors.py \
        --instance_dir memory_result/<run_name>/0 \
        --qa_file outputs/QA.../merged_95.jsonl \
        --dialog_root outputs/bazinga/TheBigBangTheory/Season1 \
        --server_url http://127.0.0.1:5005/batch_process

    (--out defaults to <instance_dir>/error_probe.json)
"""

import os
import ast
import json
import argparse
from collections import Counter

import requests

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from matching import EmbeddingMatcher, LLMJudge, present, _utterance
from data_utils import (
    extract_choice, gold_letter, load_qa, DialogResolver,
    evidence_texts, evidence_episodes, TranscriptLoader, fix_space_in_text,
)


# --------------------------------------------------------------------------- #
# Provenance: chunks_and_function_calls.json -> {chunk_idx: inserted units}
# --------------------------------------------------------------------------- #
_MTYPE = {"core_memory": "core", "episodic_memory": "episodic",
          "semantic_memory": "semantic"}


def _balanced_dict(s, start):
    """Return the substring of `s` spanning the balanced {...} starting at `start`."""
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
    """Pull the {id: content} dict an insert returned, from its tool_result string.

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
    """[{chunk_idx, raw_chunk, norm, units:{id:(mtype,text)}}] per chunk.

    `units` are the memory units INSERTED while processing that chunk (id -> type
    + text). `norm` is the normalized raw_chunk text for substring turn lookup.
    """
    prov = []
    for ch in chunks:
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
                units[uid] = (mtype, fix_space_in_text(text))
        raw = ch.get("raw_chunk", "") or ""
        prov.append({"chunk_idx": ch.get("chunk_idx"), "raw_chunk": raw,
                     "norm": _norm(raw), "units": units})
    return prov


# --------------------------------------------------------------------------- #
# Final memory (agent_state.json) -> {id: (mtype, text)} + core string
# --------------------------------------------------------------------------- #
def final_units(state):
    units, core = {}, None
    c = state.get("core")
    if isinstance(c, list):
        c = " ".join(str(x) for x in c)
    if c:
        core = fix_space_in_text(c)
    for mtype in ("episodic", "semantic"):
        for d in state.get(mtype) or []:
            if isinstance(d, dict) and d:
                uid, text = next(iter(d.items()))
                units[uid] = (mtype, fix_space_in_text(text))
    return units, core


# --------------------------------------------------------------------------- #
# Locate the chunk(s) a question's evidence turns were fed in
# --------------------------------------------------------------------------- #
_MAX_CHUNKS_PER_TURN = 6   # a turn matching more chunks than this is non-distinctive


def _norm(text):
    """Lowercase, strip to alnum tokens, collapse whitespace — for substring
    matching the evidence utterance against the raw chunk text (which uses
    different spacing / <speaker> tags)."""
    return " ".join(ch if ch.isalnum() else " " for ch in (text or "").lower()).split()


def _norm_str(text):
    return " ".join(_norm(text))


def evidence_chunks(ev_texts, prov):
    """Indices into `prov` of the chunk(s) whose raw text contains the evidence.

    Chunks are non-overlapping, so a turn normally lives in exactly one chunk. A
    turn matches a chunk if its normalized utterance is a substring of the chunk;
    for long turns (which a small spacing/ASR diff can break) we fall back to a
    single contiguous MIDDLE window of the utterance (the middle avoids generic
    openers like the "[Dialogue ... on <date>]" header / timestamps that recur in
    every chunk). Returns (chunk_indices, per_turn_located: list[bool]).
    """
    hits, located = set(), []
    for ev in ev_texts:
        utt = _norm(_utterance(ev))
        if not utt:
            located.append(False)
            continue
        probes = [" ".join(utt)]
        if len(utt) > 12:                       # robust middle window for long turns
            mid = len(utt) // 2
            probes.append(" ".join(utt[mid - 5:mid + 5]))
        matched = [i for i, c in enumerate(prov)
                   if any(p in " ".join(c["norm"]) for p in probes)]
        located.append(bool(matched))
        # A turn that matches many chunks is non-distinctive (a short generic line
        # like "Yeah." / "What?"); it can't localize provenance, so don't let it
        # drag dozens of chunks in. Distinctive turns match 1-2 chunks.
        if 0 < len(matched) <= _MAX_CHUNKS_PER_TURN:
            hits.update(matched)
    return sorted(hits), located


# --------------------------------------------------------------------------- #
# Context builders (-> memory dict the server expects: core/episodic/semantic)
# --------------------------------------------------------------------------- #
def _empty_ctx():
    return {"episodic": [], "semantic": []}


def construction_ctx(chunk_idxs, prov, units_final, include_core, core_final):
    """Final memory units whose insert lives in any evidence chunk (and still
    exist in the final store), grouped back into episodic/semantic. Optionally
    add the final core string (global, not turn-scoped)."""
    ctx = _empty_ctx()
    seen, kept = set(), []
    for i in chunk_idxs:
        for uid in prov[i]["units"]:
            if uid in seen or uid not in units_final:
                continue
            seen.add(uid)
            mtype, text = units_final[uid]
            if mtype in ("episodic", "semantic"):
                ctx[mtype].append({uid: text})
                kept.append(uid)
    if include_core and core_final:
        ctx["core"] = core_final
    return ctx, kept


def transcript_ctx(qa, transcript, ev_texts, emb, judge):
    """The matched transcript turns (raw ASR + speaker naming) for the evidence,
    fed as pseudo-episodic units. Empty if none match (== ASR dropped it)."""
    ctx = _empty_ctx()
    if not transcript.available:
        return ctx, {"note": "transcript_unavailable"}
    recs = transcript.records_for_episodes(evidence_episodes(qa))
    if not recs:
        return ctx, {"note": "no_transcript_for_episode"}
    _, info = present(ev_texts, recs, emb, judge, match_speaker=True,
                      use_emb=False, use_judge=False)
    by_id = {r["id"]: r["text"] for r in recs}
    n = 0
    for m in info["matches"]:
        if m["found"] and m["memory_id"] in by_id:
            ctx["episodic"].append({f"t{n}": by_id[m["memory_id"]]})
            n += 1
    return ctx, {"coverage": info["coverage"], "matched_turns": n}


def rescue_ctx(retrieved, constr_ctx):
    """Actual retrieved_memory UNION the evidence construction units. The server
    re-retrieves top-k over this set; the evidence units are query-relevant so
    they survive, giving "real context + the missing evidence"."""
    ctx = {"episodic": list((retrieved or {}).get("episodic") or []),
           "semantic": list((retrieved or {}).get("semantic") or [])}
    core = (retrieved or {}).get("core")
    if core:
        ctx["core"] = core
    have = {next(iter(d)) for grp in (ctx["episodic"], ctx["semantic"]) for d in grp if d}
    for mtype in ("episodic", "semantic"):
        for d in constr_ctx.get(mtype, []):
            if d and next(iter(d)) not in have:
                ctx[mtype].append(d)
    return ctx


# --------------------------------------------------------------------------- #
# QA server + grading
# --------------------------------------------------------------------------- #
def qa_batch(server_url, jobs, max_tokens=2048):
    """jobs = [(memory_dict, question), ...] -> [response_text, ...]."""
    if not jobs:
        return []
    payload = {"memories": [m for m, _ in jobs],
               "questions": [[q] for _, q in jobs],
               "max_tokens": max_tokens, "temperature": 0}
    r = requests.post(server_url, json=payload, timeout=1200)
    if r.status_code != 200:
        raise RuntimeError(f"server {r.status_code}: {r.text[:300]}")
    out = r.json().get("result", [])
    return [(grp[0] if grp else "") for grp in out]


def is_correct(response, gold_answer):
    """Multiple-choice letter match (mirrors evaluate_agent_results.py). 'C' is the
    appended not-sure option -> not correct."""
    pred = extract_choice(response)
    gold = gold_letter(gold_answer)
    return pred is not None and gold is not None and pred == gold


# --------------------------------------------------------------------------- #
# Main probe
# --------------------------------------------------------------------------- #
def probe(args):
    qa_items = load_qa(args.qa_file)
    resolver = DialogResolver(args.dialog_root)
    transcript = TranscriptLoader(args.transcript_root)
    emb, judge = EmbeddingMatcher(), LLMJudge()

    with open(os.path.join(args.instance_dir, "results.json")) as f:
        results = json.load(f)
    with open(os.path.join(args.instance_dir, "agent_state.json")) as f:
        state = json.load(f)
    with open(os.path.join(args.instance_dir, "chunks_and_function_calls.json")) as f:
        chunks = json.load(f)

    prov = build_provenance(chunks)
    units_final, core_final = final_units(state)
    print(f"[probe] chunks={len(prov)}  final_units={len(units_final)}  "
          f"transcript={'on' if transcript.available else 'off'}  server={args.server_url}")

    qa_by_q = {qa["question"].strip(): qa for qa in qa_items}

    def find_qa(rq):
        rq = rq.strip()
        if rq in qa_by_q:
            return qa_by_q[rq]
        for q, qa in qa_by_q.items():
            if rq.startswith(q):
                return qa
        return None

    findings = []
    pending = []   # (rec, kind) jobs to send to the server in a batch

    # ---- pass 1: gate + locate evidence + queue the C-probe ----
    for res in results:
        question = res["question"]
        gold_ans = res.get("answer", "")
        real_correct = (res.get("score") == 1.0) if "score" in res \
            else is_correct(res.get("response", ""), gold_ans)

        rec = {"question": question.split("\n")[0], "gold_answer": gold_ans,
               "real_correct": real_correct, "real_response": res.get("response", "")[:200]}

        if real_correct:
            rec["stage"] = "correct"
            findings.append(rec)
            continue

        qa = find_qa(question)
        if qa is None:
            rec["stage"] = "gate:qa_not_found"
            findings.append(rec)
            continue
        ev_texts, unresolved = evidence_texts(qa, resolver, args.min_turn_words)
        if not ev_texts:
            rec["stage"] = "gate:no_gold_evidence"
            rec["detail"] = {"unresolved_files": unresolved}
            findings.append(rec)
            continue

        idxs, located = evidence_chunks(ev_texts, prov)
        c_ctx, kept = construction_ctx(idxs, prov, units_final,
                                       not args.no_core, core_final)

        rec["_qa"] = qa
        rec["_ev"] = ev_texts
        rec["_retrieved"] = res.get("retrieved_memory")
        rec["_c_ctx"] = c_ctx
        rec["detail"] = {
            "evidence_turns": len(ev_texts),
            "turns_located_in_chunks": sum(located),
            "evidence_chunks": idxs,
            "construction_units_kept": kept,
            "construction_unit_count": len(kept) + (1 if c_ctx.get("core") else 0),
        }
        findings.append(rec)
        pending.append((rec, "c"))

    # ---- send all C-probes ----
    _run(args, pending, "c", "C-probe (construction)")

    # ---- pass 2: branch on C-probe result; queue T-probe or rescue ----
    pending2 = []
    for rec, _ in pending:
        if rec["detail"]["c_correct"]:
            rec["_rescue_ctx"] = rescue_ctx(rec["_retrieved"], rec["_c_ctx"])
            pending2.append((rec, "rescue"))
        else:
            t_ctx, t_info = transcript_ctx(rec["_qa"], transcript, rec["_ev"], emb, judge)
            rec["_t_ctx"] = t_ctx
            rec["detail"]["transcript_probe"] = t_info
            pending2.append((rec, "t"))

    _run(args, pending2, "rescue", "rescue (retrieval/response)")
    _run(args, pending2, "t", "T-probe (transcription)")

    # ---- resolve stages ----
    for rec, kind in pending2:
        if kind == "rescue":
            rec["stage"] = "retrieval" if rec["detail"]["rescue_correct"] else "response"
        else:
            rec["stage"] = "construction" if not rec["detail"]["t_correct"] else "transcription"

    for rec in findings:
        for k in ("_qa", "_ev", "_retrieved", "_c_ctx", "_rescue_ctx", "_t_ctx"):
            rec.pop(k, None)

    _summarize_and_save(args, findings)


_CTX_KEY = {"c": "_c_ctx", "rescue": "_rescue_ctx", "t": "_t_ctx"}
_FLAG = {"c": "c_correct", "rescue": "rescue_correct", "t": "t_correct"}
_RKEY = {"c": "c_response", "rescue": "rescue_response", "t": "t_response"}


def _run(args, pending, kind, label):
    """Send the queued jobs of one `kind` to the QA server and record correctness."""
    recs = [rec for rec, k in pending if k == kind]
    if not recs:
        return
    print(f"[probe] {label}: {len(recs)} questions")
    payload = [(rec[_CTX_KEY[kind]], rec["question"]) for rec in recs]
    responses = qa_batch(args.server_url, payload, args.max_tokens)
    for rec, resp in zip(recs, responses):
        rec["detail"][_FLAG[kind]] = is_correct(resp, rec["gold_answer"])
        rec["detail"][_RKEY[kind]] = resp[:200]


def _summarize_and_save(args, findings):
    wrong = [f for f in findings if not f["real_correct"]]
    buckets = Counter(f["stage"] for f in wrong)
    print("\n================ ERROR PROBE SUMMARY ================")
    print(f"instance       : {args.instance_dir}")
    print(f"questions      : {len(findings)}")
    print(f"correct        : {len(findings) - len(wrong)}")
    print(f"failed         : {len(wrong)}")
    print("---- failure attribution (behavioral) ----")
    order = ["gate:qa_not_found", "gate:no_gold_evidence",
             "transcription", "construction", "retrieval", "response"]
    for k in order:
        if buckets.get(k):
            print(f"  {k:24s} {buckets[k]:3d}  ({100*buckets[k]/max(len(wrong),1):.0f}%)")
    for k, v in buckets.items():
        if k not in order:
            print(f"  {k:24s} {v:3d}")
    print("====================================================\n")

    out = {"instance_dir": args.instance_dir,
           "method": "behavioral_counterfactual_probe",
           "summary": {"total": len(findings), "failed": len(wrong),
                       "attribution": dict(buckets)},
           "findings": findings}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote per-question probe -> {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Behavioral error attribution for Mem-alpha QA.")
    p.add_argument("--instance_dir", required=True,
                   help="Dir with results.json / agent_state.json / chunks_and_function_calls.json")
    p.add_argument("--qa_file", default="outputs/tmp_folder_for_95_qs/merged_95.jsonl")
    p.add_argument("--dialog_root", nargs="+",
                   default=["outputs/bazinga/TheBigBangTheory/Season1"],
                   help="Dir(s) searched for evidence dialog files (REAL speaker names).")
    p.add_argument("--transcript_root",
                   default="outputs/step3/vibevoice_TheBigBangTheory_predname",
                   help="Transcribed dialogue root for the T-probe. Set '' to disable.")
    p.add_argument("--server_url", default="http://127.0.0.1:5005/batch_process",
                   help="Memory server /batch_process endpoint (same one the run used).")
    p.add_argument("--no_core", action="store_true",
                   help="Exclude the (global) final core string from the construction context.")
    p.add_argument("--min_turn_words", type=int, default=0)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--out", default=None, help="Output json (default: <instance_dir>/error_probe.json)")
    args = p.parse_args()
    if args.out is None:
        args.out = os.path.join(args.instance_dir, "error_probe.json")
    return args


if __name__ == "__main__":
    probe(parse_args())
