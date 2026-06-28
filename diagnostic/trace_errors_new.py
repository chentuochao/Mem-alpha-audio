#!/usr/bin/env python3
"""
trace_errors_new.py — Cascade error attribution for the *chunked* QA schema.

Same MemTrace cascade as trace_errors_clean.py, but adapted to the NEW QA format
(e.g. outputs/tmp_folder_for_key_phrases_qa/merged_qa/merged_qa.jsonl) whose gold
evidence is anchored to the CHUNKED dialogue rather than the whole-episode files.

What changed vs the old schema
------------------------------
old gt_source.sources[]:
    file          : "TheBigBangTheory.Season01.Episode01.json"   (whole episode)
    evidence_turns: [58, 59, 61]                                  (integer indices)

new gt_source.sources[]:
    file          : "TheBigBangTheory.Season01.Episode02/CHUNK_1/parsed_dialog_gt.json"
                    (a per-chunk path, RELATIVE TO --transcript_root)
    evidence_turns: ["S01E02_C001_T003"]                          (string turn IDs;
                    the trailing T<NNN> is the 0-based turn index within that chunk)

Both dialogue views come from the same source: the whole-episode files under
outputs/bazinga/... were chunked into <root>/<episode>/CHUNK_*/parsed_dialog_*.json.
The new evidence points INTO those chunk files, which already live under the
transcript root — so the separate --dialog_root argument is gone. The gold
evidence (parsed_dialog_gt.json) and the transcribed dialogue (parsed_dialog_pred.json)
are read from the same root, just different filenames in each chunk dir.

Stages (unchanged taxonomy):
    GATE         -> annotation / judge problems (excluded from system blame)
    TRANSCRIPTION-> evidence in gold chunk but lost by ASR + speaker naming (pred)
    CONSTRUCTION -> evidence in transcript but NOT in the stored memory
    RETRIEVAL    -> evidence in the store but NOT in retrieved_memory
    RESPONSE     -> evidence was retrieved but the model still answered wrong

Usage:
    python diagnostic/trace_errors_new.py \
        --instance_dir data/<run_name>/0 \
        --qa_file outputs/tmp_folder_for_key_phrases_qa/merged_qa/merged_qa.jsonl \
        --transcript_root outputs/step3/vibevoice_TheBigBangTheory_predname

    (--out defaults to <instance_dir>/error_trace.json)
"""

import os
import re
import json
import argparse
from collections import Counter

# Optional: load .env so OPENAI/OPENROUTER keys are picked up automatically.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from matching import (
    EmbeddingMatcher, LLMJudge,
    present, present_sentencewise, evidence_rank, _utterance, LEX_TAU, EMB_TAU, COVERAGE_TAU,
)
from data_utils import (
    fix_space_in_text, extract_choice, gold_letter, load_qa,
    TranscriptLoader, memory_records, retrieved_records,
)


# --------------------------------------------------------------------------- #
# NEW-schema evidence loading (chunked dialogue)
# --------------------------------------------------------------------------- #
class ChunkDialogResolver:
    """Resolve a new-schema gt_source `file` to its turn list.

    `file` is a path RELATIVE to the transcript root, e.g.
    'TheBigBangTheory.Season01.Episode02/CHUNK_1/parsed_dialog_gt.json'. Unlike the
    old basename index, we resolve the FULL relative path so each (episode, chunk)
    loads its own dialog — the basenames ('parsed_dialog_gt.json') are not unique.
    """

    def __init__(self, root):
        self.root = root
        self.cache = {}      # abspath -> turns

    def turns_for(self, file_field):
        if not file_field:
            return None
        path = os.path.join(self.root, file_field)
        if path not in self.cache:
            try:
                with open(path) as f:
                    self.cache[path] = json.load(f)
            except Exception:
                self.cache[path] = None
        return self.cache[path]


def _turn_index(turn_id):
    """New string turn ID -> 0-based index into its chunk dialog.

    'S01E02_C001_T003' -> 3. Also accepts a bare int (old schema) for robustness.
    Returns None if no T<NNN> suffix is present.
    """
    if isinstance(turn_id, int):
        return turn_id
    m = re.search(r"[Tt](\d+)\s*$", str(turn_id))
    return int(m.group(1)) if m else None


def evidence_texts(qa, resolver, min_turn_words=0):
    """Return (texts, unresolved_files) for a QA item's gold evidence turns.

    texts: list of "speaker: text" strings pulled from each source's own CHUNK file.
    Each evidence_turns entry is a string ID whose trailing T<NNN> gives the turn's
    0-based index within that chunk. Turns shorter than `min_turn_words` words are
    dropped (too short to match reliably); if that drops ALL turns, the unfiltered
    list is kept so the question is not lost. unresolved_files: files not on disk.
    """
    out, unresolved = [], []
    gt = qa.get("gt_source", {})
    sources = gt.get("sources")
    if sources is None and "evidence_turns" in gt:          # single-source schema
        sources = [gt]
    for src in (sources or []):
        turns = resolver.turns_for(src.get("file", ""))
        if not turns:
            unresolved.append(src.get("file", ""))
            continue
        for tid in src.get("evidence_turns", []):
            ti = _turn_index(tid)
            if ti is not None and 0 <= ti < len(turns):
                t = turns[ti]
                out.append(f"{t.get('speaker','?')}: {fix_space_in_text(t.get('text',''))}")

    filtered = [t for t in out if len(_utterance(t).split()) >= min_turn_words]
    return (filtered if filtered else out), unresolved


def evidence_episodes(qa):
    """Episode name(s) a question's gold evidence comes from.

    The new `file` is '<episode>/CHUNK_n/parsed_dialog_gt.json', so the leading
    path component IS the episode dir, e.g. 'TheBigBangTheory.Season01.Episode02'.
    This matches the transcript layout (<root>/<episode>/CHUNK_*/), so the
    transcription stage can be scoped to the evidence's OWN episode.
    """
    gt = qa.get("gt_source", {})
    sources = gt.get("sources")
    if sources is None and "evidence_turns" in gt:
        sources = [gt]
    episodes = set()
    for src in (sources or []):
        f = (src.get("file", "") or "").replace("\\", "/")
        if f:
            episodes.add(f.split("/")[0])
    return episodes


def evidence_chunks(qa):
    """(episode, chunk) pairs a question's gold evidence comes from.

    The new `file` is '<episode>/CHUNK_n/parsed_dialog_gt.json', so its first two
    path components are the episode dir and the chunk dir. Used to scope the
    transcription match to the evidence's OWN chunk (not the whole episode)."""
    gt = qa.get("gt_source", {})
    sources = gt.get("sources")
    if sources is None and "evidence_turns" in gt:
        sources = [gt]
    chunks = set()
    for src in (sources or []):
        parts = (src.get("file", "") or "").replace("\\", "/").split("/")
        if len(parts) >= 2 and parts[0] and parts[1]:
            chunks.add((parts[0], parts[1]))
    return chunks


# --------------------------------------------------------------------------- #
# Main cascade
# --------------------------------------------------------------------------- #
def trace(args):
    qa_items = load_qa(args.qa_file)                       # QA items (chunked schema)
    resolver = ChunkDialogResolver(args.transcript_root)  # gold evidence chunk files
    transcript = TranscriptLoader(args.transcript_root)   # ASR + speaker naming, per episode

    with open(os.path.join(args.instance_dir, "results.json")) as f:
        results = json.load(f)
    with open(os.path.join(args.instance_dir, "agent_state.json")) as f:
        state = json.load(f)

    # global memory records from agent_state.json
    store_records, episodic_records = memory_records(state)

    emb = EmbeddingMatcher()
    judge = LLMJudge()
    print(f"[matcher] lexical=on  embedding={'on' if emb.enabled else 'off'}  "
          f"llm_judge={'on' if judge.enabled else 'off'}  "
          f"transcript={'on' if transcript.available else 'off'}")

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

    for res in results:
        qa = find_qa(res["question"])
        # Unparseable / abstained responses ("not sure", no \boxed{X}, ...) are NOT
        # gated out: they count as failures and are traced through the full cascade
        # to attribute where the evidence stood. The prediction is recorded as the
        # sentinel "not_parse" (never equals a gold letter, so correct=False).
        pred = extract_choice(res.get("response", "")) or "not_parse"
        gold = gold_letter(res.get("answer", ""))
        correct = (gold is not None and pred == gold)

        rec = {
            "question": res["question"].split("\n")[0],
            "options": (qa.get("options") if qa else None),
            "gold": gold, "pred": pred, "correct": correct,
        }

        if correct:
            rec["stage"] = "correct"
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

        # --- STAGE: transcription (did ASR + speaker-naming preserve the evidence?) ---
        # Matched only within the evidence's OWN chunk(s); the transcribed dialogue
        # is the input to memory construction, so a drop here is upstream of the agent.
        trans_info = None
        if transcript.available:
            chunks = evidence_chunks(qa)
            trans_records = transcript.records_for_chunks(chunks)
            chunk_names = sorted("/".join(c) for c in chunks)
            if trans_records:
                # Require BOTH content and speaker name to be preserved. Gold turns are
                # speaker-merged blobs while the pred transcript is finely segmented, so
                # we decompose each gold turn into sentences and match each against the
                # single transcript turn that contains it (see present_sentencewise).
                in_transcript, trans_info = present_sentencewise(ev_list, trans_records,
                                                                 match_speaker=True)
                if not in_transcript:
                    rec["stage"] = "transcription"
                    rec["detail"] = {"transcript_coverage": trans_info,
                                     "chunks": chunk_names}
                    findings.append(rec)
                    continue
            else:
                trans_info = {"note": "transcript_unavailable_for_chunk",
                              "chunks": chunk_names}

        # --- STAGE: construction (is evidence in the stored memory?) ---
        in_store, store_info = present(ev_list, store_records, emb, judge, use_judge=True)
        if not in_store:
            rec["stage"] = "construction"
            rec["detail"] = {"transcript_coverage": trans_info,
                             "store_coverage": store_info}
            findings.append(rec)
            continue

        # --- STAGE: retrieval (was evidence shown to the QA model?) ---
        # The retrieved memory is a SUBSET of the full store, sharing the same unit
        # ids. So we don't re-run the matcher (and its LLM-judge calls) here: we
        # reuse the construction matches (evidence turn -> matched memory id, or
        # None) and just check whether that id is among the retrieved unit ids.
        retr_records, _ = retrieved_records(res.get("retrieved_memory"))
        retrieved_ids = {r.get("id") for r in retr_records}
        retr_matches, hits = [], 0
        for m in store_info["matches"]:
            mid = m.get("memory_id")
            in_retr = bool(m.get("found")) and mid is not None and mid in retrieved_ids
            retr_matches.append({"turn": m["turn"], "memory_id": mid,
                                 "found": m.get("found"), "retrieved": in_retr})
            hits += int(in_retr)
        total = len(retr_matches)
        cov = hits / total if total else 0.0
        in_retrieved = cov >= COVERAGE_TAU
        retr_info = {"coverage": round(cov, 3), "matched": hits, "total": total,
                     "matches": retr_matches}
        if not in_retrieved:
            best_rank, ranks = evidence_rank(ev_list, episodic_records)
            rec["stage"] = "retrieval"
            rec["detail"] = {
                "transcript_coverage": trans_info,
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
        rec["detail"] = {"transcript_coverage": trans_info,
                         "store_coverage": store_info, "retrieved_coverage": retr_info}
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
             "transcription",
             "construction",
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
    p = argparse.ArgumentParser(description="Cascade error attribution for Mem-alpha QA (chunked schema).")
    p.add_argument("--instance_dir",
                   default=None,
                   help="Dir with results.json / agent_state.json / chunks_and_function_calls.json")
    p.add_argument("--qa_file",
                   default="outputs/tmp_folder_for_key_phrases_qa/merged_qa/merged_qa.jsonl",
                   help="QA jsonl (chunked schema) with gt_source.sources[].evidence_turns as string IDs")
    p.add_argument("--transcript_root",
                   default="outputs/step3/vibevoice_TheBigBangTheory_predname",
                   help="Root holding BOTH the gold evidence chunks and the transcribed dialogue, laid "
                        "out as <root>/<episode>/CHUNK_*/parsed_dialog_gt.json (gold evidence, indexed "
                        "by the new QA `file` paths) and .../parsed_dialog_pred.json (ASR + speaker "
                        "naming, used for the 'transcription' stage).")
    p.add_argument("--min_turn_words", type=int, default=0,
                   help="Drop evidence turns whose utterance has fewer than this many words "
                        "(too short to match reliably). Set 0 to disable.")
    p.add_argument("--out", default=None, help="Output json (default: <instance_dir>/error_trace.json)")
    args = p.parse_args()
    if args.out is None:
        args.out = os.path.join(args.instance_dir, "error_trace.json")
    return args


if __name__ == "__main__":
    trace(parse_args())
