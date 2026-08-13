#!/usr/bin/env python3
"""
probe_errors.py — Behavioral (counterfactual) error attribution for Mem-alpha QA.

A standalone alternative to the matching cascade in `trace_errors_clean.py`.

Instead of asking "does the gold evidence string still APPEAR in the memory /
transcript?" (a matching question, sensitive to LEX_TAU / EMB_TAU / COVERAGE_TAU
and blind to paraphrase), this asks "can the QA model still ANSWER correctly when
fed exactly the memory / transcript derived from the evidence turns?" — a
behavioral question answered by re-running the real QA model on curated contexts.

Provenance + a deterministic chunk map (no fuzzy matching) locate the relevant memory.
Every memory unit is created by a `new_memory_insert` (and possibly edited by later
`memory_update`s) recorded in chunks_and_function_calls.json under a `chunk_idx`, and the
unit id (e.g. 'cd61') is the SAME id stored in agent_state.json. The QA's gt_source names
the source chunk folder ('{episode}/CHUNK_N'), and the step3 parquet's `chunk_folders`
list maps that folder to its chunk_idx (its position in the list IS the chunk_idx — see
prepare_parquet_from_step3.py). So "the memory constructed from the evidence chunk" = the
FINAL stored units whose insert OR a later update lives in that chunk_idx; ids removed by
`memory_delete` are excluded. We use the FINAL memory content (from agent_state.json), so a
single `construction` bucket.

Per FAILED question (the real run got it wrong), oracle QA re-answers ALL four probes
(G/C/T/S) — we no longer branch the cascade on the C-probe result. Running every stage
on every failure exposes cases the old C-gated cascade hid, notably SELF-CORRECTION:
the raw transcript alone loses the answer (T wrong) yet the constructed memory recovers
it (C right) — which the old code never saw because it only ran T when C failed.

    G-probe  : the WHOLE gold dialogue CHUNK (all turns of parsed_dialog_gt.json for
                 the evidence chunk named by gt_source.file) -- the UPPER BOUND. Does
                 not drive attribution; it caps what any downstream stage could recover
                 (wrong here => not the pipeline's fault: insufficient evidence or a
                 model-reasoning limit). We feed the whole chunk, not individual turns,
                 because the per-turn evidence annotations are unreliable.
    C-probe  : final memory units traced to the evidence chunk(s). Decides the PRIMARY
                 stage: wrong -> transcription/construction (split by T); right ->
                 retrieval/response (split by S).
    S-probe  : re-answer on (evidence units ∩ retrieved_memory) — exactly the evidence
                 the real run's retriever surfaced (id membership, one QA call).
                 right  -> response   (retrieved evidence sufficed; run was distracted)
                 wrong  -> retrieval  (retriever didn't surface enough of the evidence)
    T-probe  : the WHOLE predicted dialogue CHUNK (all turns of parsed_dialog_pred.json,
                 raw ASR + speaker naming, for the same chunk).
                 right  -> construction  (transcript had it, construction lost it)
                 wrong  -> transcription (ASR/naming already lost it upstream)

All four now run for every gated-in failure; when C passes but T fails we flag
detail.self_correction=True (memory recovered what the transcript alone lost).

The QA re-answers hit the SAME memory server the real run used (/batch_process),
so the context-serialization, retriever, and model are identical.

Usage:
    PYTHONPATH=. python diagnostic/probe_errors.py \
        --instance_dir memory_result/<run_name>/0 \
        --qa_file outputs/step3_anony/qas/merged_qa_anoy.jsonl \
        --data_root outputs/step3_anony/S01_S03_Clean_Anoy \
        --server_url http://127.0.0.1:5005/batch_process

    (--out defaults to <instance_dir>/error_probe.json)
"""

import os
import ast
import glob
import json
import time
import hashlib
import argparse
from collections import Counter

import yaml
import requests

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from data_utils import (
    extract_choice, gold_letter, load_qa, DialogResolver, fix_space_in_text,
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
    """[{chunk_idx, raw_chunk, units:{id:(mtype,text)}}] per chunk.

    `units` are the memory units a chunk CONTRIBUTED to — inserted via
    `new_memory_insert` OR edited via a later `memory_update` (so a unit appears under
    every chunk that contributed to it). Units removed by `memory_delete` are dropped
    from all chunks. The text here is best-effort (insert result / update new_content);
    construction_ctx re-reads the FINAL text from agent_state.json, so it is not relied
    on. Provenance is keyed by chunk_idx, which the parquet's chunk_folders list maps
    deterministically to the source chunk folder (no text matching needed).
    """
    prov = []
    deleted = set()
    for ch in chunks:
        units = {}
        for fc in ch.get("function_calls", []):
            call = fc.get("function_call", {})
            name = call.get("name")
            if name not in ("new_memory_insert", "memory_update", "memory_delete"):
                continue
            try:
                args = json.loads(call.get("arguments", "{}"))
            except Exception:
                args = {}
            mtype = _MTYPE.get(args.get("memory_type", ""), "episodic")
            if name == "new_memory_insert":
                for uid, text in _parse_inserted(fc.get("tool_result", "")).items():
                    units[uid] = (mtype, fix_space_in_text(text))
            elif name == "memory_update":
                uid = args.get("memory_id")   # None for core -> skip
                if uid:
                    units[uid] = (mtype, fix_space_in_text(args.get("new_content", "")))
            else:  # memory_delete
                uid = args.get("memory_id")
                if uid:
                    deleted.add(uid)
        raw = ch.get("raw_chunk", "") or ""
        prov.append({"chunk_idx": ch.get("chunk_idx"), "raw_chunk": raw,
                     "units": units})
    # a delete anywhere removes the id from every chunk it was attributed to
    if deleted:
        for c in prov:
            for uid in [u for u in c["units"] if u in deleted]:
                del c["units"][uid]
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
# Locate the chunk(s) a question's evidence came from — DETERMINISTIC.
# gt_source names the source chunk folder ("{episode}/CHUNK_N"); the parquet's
# chunk_folders list maps folder -> chunk_idx (its position IS the chunk_idx used
# in chunks_and_function_calls.json). No text matching.
# --------------------------------------------------------------------------- #
def load_chunk_folder_map(parquet_path, data_root):
    """Return ({folder: chunk_idx}, resolved_parquet_path).

    Reads `chunk_folders` from the step3 parquet (see prepare_parquet_from_step3.py):
    a json list whose i-th entry ("{episode}/CHUNK_N") is the source folder of agent
    chunk_idx i. If parquet_path is None, auto-discover a dataset_*_*.parquet in
    data_root (preferring gt_name).
    """
    import pandas as pd  # lazy: only needed for the deterministic map
    if parquet_path is None:
        cands = (sorted(glob.glob(os.path.join(data_root, "dataset_gt_name_*.parquet")))
                 or sorted(glob.glob(os.path.join(data_root, "dataset_*.parquet"))))
        if not cands:
            raise FileNotFoundError(
                f"no dataset parquet with chunk_folders found in {data_root}; "
                f"pass --parquet explicitly")
        parquet_path = cands[0]
    df = pd.read_parquet(parquet_path)
    folders = json.loads(df.iloc[0]["chunk_folders"])
    return {f: i for i, f in enumerate(folders)}, parquet_path


def evidence_chunk_idxs(qa, folder2idx):
    """Deterministic evidence chunk indices from the QA's gt_source folders.

    Returns (chunk_idxs, evidence_folders, unmapped_folders). Each gt_source file
    ("{episode}/CHUNK_N/parsed_dialog_gt.json") -> its folder -> chunk_idx via
    folder2idx. Folders absent from the map (e.g. a different season than the run)
    are reported in unmapped_folders.
    """
    idxs, folders, unmapped = [], [], []
    for f in _source_files(qa, pred=False):
        folder = os.path.dirname(f)
        if folder not in folders:
            folders.append(folder)
        i = folder2idx.get(folder)
        if i is None:
            unmapped.append(folder)
        elif i not in idxs:
            idxs.append(i)
    return sorted(idxs), folders, unmapped


# --------------------------------------------------------------------------- #
# Context builders (-> memory dict the server expects: core/episodic/semantic)
# --------------------------------------------------------------------------- #
def _empty_ctx():
    return {"episodic": [], "semantic": []}


def construction_ctx(chunk_idxs, prov_by_idx, units_final, include_core, core_final):
    """Final memory units whose insert lives in any evidence chunk (and still
    exist in the final store), grouped back into episodic/semantic. Optionally
    add the final core string (global, not turn-scoped). `prov_by_idx` is keyed by
    chunk_idx (== the parquet chunk_folders position)."""
    ctx = _empty_ctx()
    seen, kept = set(), []
    for i in chunk_idxs:
        p = prov_by_idx.get(i)
        if not p:
            continue
        for uid in p["units"]:
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


def evidence_mapping(qa, chunk_idxs, prov_by_idx, units_final, ev_folders, unmapped):
    """Explicit QA -> chunk -> memory-id provenance for the output json.

    Deterministic: the QA's gt_source names the source chunk folder(s), the parquet
    maps each to a chunk_idx, and provenance gives the memory unit ids that chunk
    contributed (kept only if they survive in the final store).
    """
    per_chunk = []
    all_ids = []
    for i in chunk_idxs:
        p = prov_by_idx.get(i)
        ids = [uid for uid in (p["units"] if p else {}) if uid in units_final]
        per_chunk.append({
            "chunk_idx": i,
            "chunk_folder": (p.get("chunk_folder") if p else None),
            "memory_ids": ids,
        })
        for uid in ids:
            if uid not in all_ids:
                all_ids.append(uid)
    return {
        "question": (qa.get("question", "") or "").split("\n")[0],
        "gt_source_files": _source_files(qa, pred=False),
        "gt_source_chunk_folders": ev_folders,
        "unmapped_folders": unmapped,
        "chunks": per_chunk,
        "memory_ids": all_ids,
    }


def normalize_qa_sources(qa_items, data_source):
    """Rewrite gt_source file refs into the canonical chunked-schema form
    '<folder>/CHUNK_0/parsed_dialog_gt.json' that every downstream loader expects.

    Only perltqa needs this. Its QA `file` is a profile/session ref like
    'Cao_Lili/25_0_0_0', whereas the on-disk chunk folder AND the parquet's
    chunk_folders entry are 'Cao_Lili_25_0_0_0/CHUNK_0'. We map
    'A/B' -> 'A_B/CHUNK_0/parsed_dialog_gt.json' (replace '/' with '_', append the
    single CHUNK_0) so _source_files / evidence_chunk_idxs / DialogResolver resolve
    it exactly as for the audio datasets, whose files already carry this shape.
    No-op for any other datasource (mutates in place; also returns qa_items)."""
    if "perltqa" not in (data_source or ""):
        return qa_items
    for qa in qa_items:
        gt = qa.get("gt_source", {}) or {}
        srcs = gt.get("sources")
        if srcs is None and "evidence_turns" in gt:
            srcs = [gt]
        # qa_multi schema: no gt_source; evidence is a flat `evidence_chunks` list of
        # the same profile/session refs ('Cao_Lili/25_0_0_0'). Synthesize a gt_source
        # so the rest of the pipeline (localization, dialog loading, grading) is
        # identical to the gt_source schema.
        if not srcs and qa.get("evidence_chunks"):
            srcs = [{"file": c} for c in qa["evidence_chunks"] if c]
            qa["gt_source"] = {**gt, "sources": srcs}
        for src in (srcs or []):
            f = src.get("file", "") or ""
            if not f or os.path.basename(f) == "parsed_dialog_gt.json":
                continue  # empty or already canonical
            src["file"] = f"{f.replace('/', '_')}/CHUNK_0/parsed_dialog_gt.json"
    return qa_items


def _source_files(qa, pred=False):
    """Evidence-chunk dialog file paths from gt_source — turn-id INDEPENDENT.

    Uses only gt_source.sources[].file ('<episode>/CHUNK_n/parsed_dialog_gt.json'),
    which names the evidence chunk directly. pred=True swaps the basename to
    parsed_dialog_pred.json (the ASR transcript) for the T-probe.
    """
    gt = qa.get("gt_source", {}) or {}
    sources = gt.get("sources")
    if sources is None and "evidence_turns" in gt:
        sources = [gt]
    files = []
    for src in (sources or []):
        f = src.get("file", "") or ""
        if not f:
            continue
        if pred:
            f = os.path.join(os.path.dirname(f), "parsed_dialog_pred.json")
        if f not in files:
            files.append(f)
    return files


def chunk_dialog(qa, resolver, pred=False):
    """All turns of the evidence CHUNK(s) as "speaker: text" strings, turn-id free.

    The per-turn evidence annotations are unreliable, so instead of selecting single
    turns we take the WHOLE chunk named by gt_source.file. Returns (texts, info)."""
    texts, info = [], {"chunks": [], "turns": 0, "missing": []}
    for f in _source_files(qa, pred=pred):
        turns = resolver.turns_for(f)
        if not turns:
            info["missing"].append(f)
            continue
        info["chunks"].append(os.path.dirname(f))
        for t in turns:
            texts.append(fix_space_in_text(f"{t.get('speaker','?')}: {t.get('text','')}"))
    info["turns"] = len(texts)
    return texts, info


def chunk_ctx(texts, prefix):
    """Feed a list of dialogue turns as pseudo-episodic units for a QA re-answer."""
    ctx = _empty_ctx()
    for n, t in enumerate(texts):
        ctx["episodic"].append({f"{prefix}{n}": t})
    return ctx


def retrieved_index(retrieved):
    """{uid: (mtype, rank)} for the units the real run actually surfaced to the QA
    model. List order == retrieval rank (0 = top). First occurrence wins."""
    idx = {}
    for mtype in ("episodic", "semantic"):
        for i, d in enumerate((retrieved or {}).get(mtype) or []):
            if isinstance(d, dict) and d:
                idx.setdefault(next(iter(d)), (mtype, i))
    return idx


def retrieval_diag(rec):
    """Record which evidence units the real run actually retrieved (exact id membership
    against results.json retrieved_memory), plus per-unit rank ("why dropped"). Returns
    the set of shown ids. Does NOT decide the stage — the S-probe below does that."""
    evidence_ids = rec["detail"].get("construction_units_kept", [])
    ridx = retrieved_index(rec.get("_retrieved"))
    shown = [uid for uid in evidence_ids if uid in ridx]
    rec["detail"]["retrieval_check"] = {
        "method": "shown_subset_behavioral",
        "evidence_units": evidence_ids,
        "shown_units": shown,
        "coverage": (len(shown) / len(evidence_ids)) if evidence_ids else 1.0,
        "evidence_retrieved_ranks": {
            uid: (f"{ridx[uid][0]}:{ridx[uid][1]}" if uid in ridx else None)
            for uid in evidence_ids},
        "retrieved_counts": {
            "episodic": len((rec.get("_retrieved") or {}).get("episodic") or []),
            "semantic": len((rec.get("_retrieved") or {}).get("semantic") or [])},
    }
    return set(shown)


def shown_ctx(c_ctx, shown_ids):
    """The evidence units that WERE retrieved (evidence ∩ retrieved), + the always-in-
    prompt core. This is exactly the evidence the real run's retriever surfaced, so
    QA on it asks: 'was what the retriever actually delivered sufficient?'
      correct  -> response   (retrieved evidence sufficed; real run was distracted)
      wrong    -> retrieval  (retriever didn't surface enough of the evidence)."""
    ctx = _empty_ctx()
    for mtype in ("episodic", "semantic"):
        for d in c_ctx.get(mtype, []):
            if d and next(iter(d)) in shown_ids:
                ctx[mtype].append(d)
    core = c_ctx.get("core")
    if core:
        ctx["core"] = core
    return ctx


# --------------------------------------------------------------------------- #
# QA server + grading
# --------------------------------------------------------------------------- #
def load_query_prompt(data_source, path="config/prompts_wrt_datasource.yaml"):
    """The per-datasource query prompt run_qa_evaluation.py prepends to each question
    (e.g. the multiple-choice \\boxed{X} instruction for seamlessinteraction_options).
    Returns None if the file/key is missing or the prompt is null."""
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        print(f"[probe] WARNING: could not read {path}; sending questions without a query prompt")
        return None
    return (cfg.get(data_source) or {}).get("query_prompt")


def build_qa_prompt(question, query_prompt):
    """Reproduce run_qa_evaluation.py's QA prompt: the full question (options already
    appended in results.json) with the datasource query prompt prepended."""
    if query_prompt:
        return f"{query_prompt}\n\n{question}"
    return question


def find_qa(qa_by_q, rq):
    """Match a results.json question (full text, options appended) back to its qa_file
    entry. `qa_by_q` maps a BARE question -> qa; the results question starts with it.
    Shared by probe() and precompute_tg_probes.py so both localize identically."""
    rq = rq.strip()
    if rq in qa_by_q:
        return qa_by_q[rq]
    for q, qa in qa_by_q.items():
        if rq.startswith(q):
            return qa
    return None


def qa_batch(server_url, jobs, max_tokens=2048, timeout=1200):
    """jobs = [(memory_dict, qa_prompt), ...] -> [response_text, ...].

    Mirrors run_qa_evaluation.py's server payload (temperature 0, thinking off) so the
    probe's re-answers use the same decoding as the real run. `timeout` is the per-POST
    read timeout in seconds; keep batches small enough to finish within it.
    """
    if not jobs:
        return []
    payload = {"memories": [m for m, _ in jobs],
               "questions": [[q] for _, q in jobs],
               "max_tokens": max_tokens, "temperature": 0,
               "enable_thinking": False, "qwen_batch_size": 1}
    r = requests.post(server_url, json=payload, timeout=timeout)
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


def _perltqa_score(predicted_answer, gold_answer):
    """Replicate evaluate_agent_results.py's perltqa KEYWORD score exactly so the
    probe's per-stage 'correct' lines up with the headline accuracy: a ';'-separated
    gold answer scores the fraction of its keywords contained in the prediction,
    otherwise plain containment. (Faithfully mirrors the evaluator's quirk of NOT
    lower-casing the prediction in the ';' branch — keep them in sync.)"""
    gold_answer = gold_answer or ""
    predicted_answer = predicted_answer or ""
    if ";" in gold_answer:
        parts = gold_answer.split(";")
        hit = sum(1 for a in parts if a.lower().strip() in predicted_answer)
        return hit / len(parts) if parts else 0.0
    return 1.0 if gold_answer.lower() in predicted_answer.lower() else 0.0


class _PerltqaJudge:
    """LLM-as-judge scorer for perltqa, mirroring evaluate_agent_results.py's
    seamless/perltqa judge (correct / not_sure / wrong). Talks to the QWEN_URL server
    through the OpenAI client; the probe treats only 'correct' as a pass."""

    _TEMPLATE = (
        "You are an evaluation judge. Given a question, a reference answer, and a model's response, "
        "classify the model's response into exactly one of three categories:\n\n"
        "1. **correct**: The model's response contains the key information from the reference answer. "
        "Paraphrasing, elaboration, or additional details are acceptable as long as the core answer is correct.\n"
        "2. **not_sure**: The model does not give a specific answer, hedges, express it is not sure, "
        " states that the information is not available in its memory/context, or require more information to answer the question.\n"
        "3. **wrong**: The model gives a specific, concrete answer, but it is factually incorrect "
        "compared to the reference answer.\n\n"
        "Question: {question}\n\n"
        "Reference Answer: {gold_answer}\n\n"
        "Model Response: {predicted_answer}\n\n"
        "Respond with ONLY one word: correct, not_sure, or wrong."
    )

    def __init__(self):
        from openai import OpenAI  # lazy: only the llm_judge path needs it
        base_url = os.getenv("QWEN_URL")
        model = os.getenv("QWEN_MODEL_NAME")
        if not base_url or not model:
            # Without QWEN_URL the OpenAI client silently targets api.openai.com and
            # dies with a confusing 401 "EMPTY key"; fail legibly instead.
            raise RuntimeError(
                "scorer=llm_judge needs QWEN_URL and QWEN_MODEL_NAME set (and "
                "OPENROUTER_API_KEY for a real endpoint). For the local vLLM: "
                "export QWEN_URL=http://localhost:8002/v1 QWEN_MODEL_NAME=qwen3-32b "
                "OPENROUTER_API_KEY=EMPTY  — or use --scorer keyword.")
        self.client = OpenAI(base_url=base_url,
                             api_key=os.getenv("OPENROUTER_API_KEY", "EMPTY"))
        self.model = model

    def score(self, question, gold_answer, predicted_answer):
        prompt = self._TEMPLATE.format(question=question, gold_answer=gold_answer,
                                       predicted_answer=predicted_answer)
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=8,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        j = (r.choices[0].message.content or "").strip().lower()
        if "correct" in j and "not_sure" not in j and "wrong" not in j:
            return 1.0
        if "not_sure" in j:
            return 0.5
        return 0.0


def make_scorer(args):
    """Return a correct(response, gold, question) -> bool grader for this datasource.

    perltqa: keyword containment (default, no API) or LLM-as-judge (--scorer llm_judge,
    needs QWEN_URL / QWEN_MODEL_NAME / OPENROUTER_API_KEY); a stage counts as correct
    only when its score is 1.0 (matches evaluate_agent_results.py's judgment=='correct').
    Everything else keeps the multiple-choice \\boxed{X} letter match."""
    ds = args.data_source or ""
    if "perltqa" in ds:
        if getattr(args, "scorer", "keyword") == "llm_judge":
            judge = _PerltqaJudge()
            return lambda resp, gold, q: judge.score(q, gold, resp) == 1.0
        return lambda resp, gold, q: _perltqa_score(resp, gold) == 1.0
    return lambda resp, gold, q: is_correct(resp, gold)


# --------------------------------------------------------------------------- #
# Main probe
# --------------------------------------------------------------------------- #
def probe(args):
    qa_items = load_qa(args.qa_file)
    normalize_qa_sources(qa_items, args.data_source)   # perltqa: profile/session -> chunk folder
    resolver = DialogResolver([args.data_root])
    query_prompt = load_query_prompt(args.data_source, args.prompts_yaml)
    args._scorer = make_scorer(args)                   # datasource-aware per-stage grader
    print(f"[probe] data_source={args.data_source}  "
          f"query_prompt={'set' if query_prompt else 'none'}  "
          f"scorer={getattr(args, 'scorer', 'keyword') if 'perltqa' in (args.data_source or '') else 'choice'}")

    with open(os.path.join(args.instance_dir, "results.json")) as f:
        results = json.load(f)
    with open(os.path.join(args.instance_dir, "agent_state.json")) as f:
        state = json.load(f)
    with open(os.path.join(args.instance_dir, "chunks_and_function_calls.json")) as f:
        chunks = json.load(f)

    prov = build_provenance(chunks)
    units_final, core_final = final_units(state)

    # Deterministic chunk_idx <-> source-folder map from the parquet (no text matching).
    folder2idx, parquet_used = load_chunk_folder_map(args.parquet, args.data_root)
    idx2folder = {i: f for f, i in folder2idx.items()}
    for p in prov:
        p["chunk_folder"] = idx2folder.get(p["chunk_idx"])
    prov_by_idx = {p["chunk_idx"]: p for p in prov}
    if len(folder2idx) != len(prov):
        print(f"[probe] WARNING: parquet chunk_folders ({len(folder2idx)}) != agent "
              f"chunks ({len(prov)}); the parquet may be for a different season/run.")

    print(f"[probe] chunks={len(prov)}  final_units={len(units_final)}  "
          f"parquet={parquet_used}  server={args.server_url}")

    qa_by_q = {qa["question"].strip(): qa for qa in qa_items}

    findings = []
    pending = []   # (rec, kind) jobs to send to the server in a batch

    # ---- pass 1: gate + locate evidence + queue the C-probe ----
    for res in results:
        question = res["question"]                 # full text incl. options (A/B/C. not sure)
        gold_ans = res.get("answer", "")
        real_correct = (res.get("score") == 1.0) if "score" in res \
            else args._scorer(res.get("response", ""), gold_ans, question.split("\n")[0])

        rec = {"question": question.split("\n")[0], "gold_answer": gold_ans,
               "real_correct": real_correct, "real_response": res.get("response", "")[:200],
               # exact prompt the QA server is re-answered with (matches run_qa_evaluation.py)
               "_qa_prompt": build_qa_prompt(question, query_prompt)}

        if real_correct and not args.full_qa:
            rec["stage"] = "correct"
            findings.append(rec)
            continue

        qa = find_qa(qa_by_q, question)
        if qa is None:
            rec["stage"] = "gate:qa_not_found"
            findings.append(rec)
            continue
        # Deterministic localization: gt_source names the source chunk folder(s), the
        # parquet maps each folder -> chunk_idx (no text matching). The gold/pred
        # dialogue for the G/T probes is still loaded straight from those folders.
        idxs, ev_folders, unmapped = evidence_chunk_idxs(qa, folder2idx)
        if not idxs:
            rec["stage"] = "gate:no_gold_evidence"
            rec["detail"] = {"evidence_folders": ev_folders, "unmapped_folders": unmapped}
            findings.append(rec)
            continue

        c_ctx, kept = construction_ctx(idxs, prov_by_idx, units_final,
                                       not args.no_core, core_final)

        rec["_qa"] = qa
        rec["_retrieved"] = res.get("retrieved_memory")
        rec["_c_ctx"] = c_ctx
        rec["detail"] = {
            "evidence_chunk_folders": ev_folders,
            "unmapped_folders": unmapped,
            "evidence_chunks": idxs,
            "construction_units_kept": kept,
            "construction_unit_count": len(kept) + (1 if c_ctx.get("core") else 0),
            # explicit QA -> chunk -> memory-id provenance (deterministic)
            "evidence_map": evidence_mapping(qa, idxs, prov_by_idx, units_final,
                                             ev_folders, unmapped),
        }
        findings.append(rec)
        pending.append((rec, "c"))
        if args.run_golden:
            gt_texts, _ = chunk_dialog(qa, resolver, pred=False)   # whole gold chunk dialogue
            rec["_g_ctx"] = chunk_ctx(gt_texts, "g")
            pending.append((rec, "g"))

    # ---- send all C-probes and G-probes (both use pass-1 contexts) ----
    _run(args, pending, "c", "C-probe (construction)")
    _run(args, pending, "g", "G-probe (whole gold chunk)")

    # ---- pass 2: run BOTH the S-probe and T-probe on EVERY gated-in failure ----
    # Previously we branched on the C-probe result (C pass -> S only, C fail -> T only).
    # That hid self-correction: when the raw transcript alone loses the answer (T wrong)
    # but the constructed memory recovers it (C right), the old code never ran T. Now we
    # run all stages on every failure so every probe (G/C/T/S) has a value.
    # (iterate only the "c" entries — pending also holds the "g" G-probe jobs)
    pending2 = []
    for rec, kind in pending:
        if kind != "c":
            continue
        # S-probe: (evidence ∩ retrieved) — retrieval_diag records retrieval_check first.
        shown = retrieval_diag(rec)
        rec["_s_ctx"] = shown_ctx(rec["_c_ctx"], shown)
        pending2.append((rec, "s"))
        # T-probe: whole predicted (ASR) chunk.
        pred_texts, t_info = chunk_dialog(rec["_qa"], resolver, pred=True)
        rec["_t_ctx"] = chunk_ctx(pred_texts, "t")
        rec["detail"]["transcript_probe"] = t_info
        pending2.append((rec, "t"))

    _run(args, pending2, "s", "S-probe (retrieved evidence subset)")
    _run(args, pending2, "t", "T-probe (transcription)")

    # ---- resolve stages (every failure now has c/t/s probes) ----
    # C-probe still decides the PRIMARY stage; T and S provide the split. When C passes
    # but the raw transcript alone would have failed, flag self-correction.
    for rec, kind in pending:
        if kind != "c":
            continue
        d = rec["detail"]
        if d.get("c_correct"):
            split_stage = "response" if d.get("s_correct") else "retrieval"
        else:
            split_stage = "transcription" if not d.get("t_correct") else "construction"
        # real-correct records (only present under --full_qa) keep the "correct" bucket;
        # the failure-attribution split is still recorded in detail for analysis.
        rec["stage"] = "correct" if rec.get("real_correct") else split_stage
        d["attribution_split"] = split_stage
        d["self_correction"] = bool(d.get("c_correct") and not d.get("t_correct"))

    if getattr(args, "debug", False):
        _write_debug(args, findings)

    for rec in findings:
        for k in ("_qa", "_qa_prompt", "_retrieved", "_c_ctx", "_g_ctx", "_s_ctx",
                  "_t_ctx", "_debug"):
            rec.pop(k, None)

    _summarize_and_save(args, findings)


def _write_debug(args, findings):
    """Dump per-stage QA/answer/evidence for every probed question to the instance
    folder (only questions that actually ran a probe, i.e. the failures)."""
    out = []
    for rec in findings:
        dbg = rec.get("_debug")
        if not dbg:
            continue
        out.append({
            "question": rec["question"],
            "gold_answer": rec["gold_answer"],
            "real_correct": rec["real_correct"],
            "real_response": rec.get("real_response"),
            "stage": rec.get("stage"),
            # per-probe (g/c/t): {qa_prompt, gt_answer, agent_answer, correct, evidence}
            "probes": dbg,
        })
    path = os.path.join(args.instance_dir, "error_probe_debug.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote debug (per-stage QA/answer/evidence) -> {path}  ({len(out)} questions)")


_CTX_KEY = {"c": "_c_ctx", "g": "_g_ctx", "s": "_s_ctx", "t": "_t_ctx"}
_FLAG = {"c": "c_correct", "g": "g_correct", "s": "s_correct", "t": "t_correct"}
_RKEY = {"c": "c_response", "g": "g_response", "s": "s_response", "t": "t_response"}


def _load_tg_cache(path):
    """Load the shared T/G-probe answer cache ({content_hash: {response}}). Missing or
    unreadable -> empty dict."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_tg_cache(path, cache):
    """Persist the T/G-probe cache atomically (tmp + replace) so a concurrent reader
    never sees a half-written file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, path)


def _tg_cache_key(args, rec, kind):
    """Content hash for a T- or G-probe answer. Both re-answers depend ONLY on the
    dialogue context (ASR transcript for T / gold dialogue for G, both from DATA_ROOT),
    the QA prompt, and the decoding/server — never on the per-instance memory. So the
    same key maps to the same answer for every instance/seed sharing that DATA_ROOT.
    `kind` is part of the key so T and G never collide in one cache file. Keying on
    content makes the cache self-invalidating (change transcript/prompt/server -> new key)."""
    blob = json.dumps({
        "kind": kind,
        "prompt": rec["_qa_prompt"],
        "ctx": rec[_CTX_KEY[kind]],
        "max_tokens": args.max_tokens,
        "server_url": args.server_url,
        "data_source": args.data_source,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _record_answer(args, rec, kind, resp):
    """Assign a probe response onto a rec (identical for cached and fresh answers)."""
    rec["detail"][_FLAG[kind]] = args._scorer(resp, rec["gold_answer"], rec["question"])
    rec["detail"][_RKEY[kind]] = resp[:200]
    if getattr(args, "debug", False):
        # per-stage record: the QA prompt actually sent, the agent's full answer, the
        # gold answer, and the exact evidence/context dict fed to the QA model.
        rec.setdefault("_debug", {})[kind] = {
            "qa_prompt": rec["_qa_prompt"],
            "gt_answer": rec["gold_answer"],
            "agent_answer": resp,
            "correct": rec["detail"][_FLAG[kind]],
            "evidence": rec[_CTX_KEY[kind]],
        }


def _run(args, pending, kind, label):
    """Send the queued jobs of one `kind` to the QA server and record correctness.

    For the transcript/gold probes (kind in {'t','g'}), if --tg_probe_cache is set the
    answers are loaded READ-ONLY from the shared per-DATA_ROOT cache populated by
    diagnostic/precompute_tg_probes.py; a missing entry is a hard error (the precompute
    must run first). C/S always hit the server. Cached answers are recorded via the same
    path as fresh ones, so the output is identical to running with no cache."""
    recs = [rec for rec, k in pending if k == kind]
    if not recs:
        return

    cache_path = getattr(args, "tg_probe_cache", None) if kind in ("t", "g") else None
    if cache_path:
        cache = _load_tg_cache(cache_path)
        keys = [_tg_cache_key(args, rec, kind) for rec in recs]
        missing = [key for key in keys if key not in cache]
        if missing:
            raise RuntimeError(
                f"{len(missing)}/{len(recs)} {kind}-probe answers missing from cache "
                f"{cache_path}; run diagnostic/precompute_tg_probes.py (with the same "
                f"--data_root/--qa_file/--parquet/--server_url"
                f"{' --run_golden' if kind == 'g' else ''}) first.")
        print(f"[probe] {label}: {len(recs)} questions (all from cache)")
        for rec, key in zip(recs, keys):
            _record_answer(args, rec, kind, cache[key]["response"])
        return

    bs = max(1, getattr(args, "batch_size", 64))
    timeout = getattr(args, "timeout", 1200)
    n = len(recs)
    n_batches = (n + bs - 1) // bs
    print(f"[probe] {label}: {n} questions in {n_batches} batch(es) of {bs} "
          f"(timeout {timeout}s)", flush=True)
    done = 0
    for bi in range(n_batches):
        chunk = recs[bi * bs:(bi + 1) * bs]
        t0 = time.time()
        payload = [(rec[_CTX_KEY[kind]], rec["_qa_prompt"]) for rec in chunk]
        responses = qa_batch(args.server_url, payload, args.max_tokens, timeout=timeout)
        for rec, resp in zip(chunk, responses):
            _record_answer(args, rec, kind, resp)
        done += len(chunk)
        print(f"[probe] {label}: batch {bi + 1}/{n_batches} done "
              f"({done}/{n} questions) in {time.time() - t0:.1f}s", flush=True)


def _summarize_and_save(args, findings):
    total = len(findings)
    wrong = [f for f in findings if not f["real_correct"]]
    correct = total - len(wrong)
    buckets = Counter(f["stage"] for f in wrong)

    def _ratio(recs, flag):
        """(#passed, #ran, pass_ratio) for a probe flag over the given records."""
        ran = [f for f in recs if flag in f.get("detail", {})]
        passed = sum(1 for f in ran if f["detail"][flag])
        return passed, len(ran), (passed / len(ran) if ran else 0.0)

    c_pass, c_n, c_r = _ratio(wrong, "c_correct")          # constructed memory answers?
    t_pass, t_n, t_r = _ratio(wrong, "t_correct")          # raw transcript answers?
    g_pass, g_n, g_r = _ratio(wrong, "g_correct")          # GOLD chunk answers? (ceiling)
    s_pass, s_n, s_r = _ratio(wrong, "s_correct")          # retrieved evidence subset answers?
    # same probes over ALL probed questions (== failed set unless --full_qa)
    c_pass_a, c_n_a, c_r_a = _ratio(findings, "c_correct")
    t_pass_a, t_n_a, t_r_a = _ratio(findings, "t_correct")
    g_pass_a, g_n_a, g_r_a = _ratio(findings, "g_correct")
    s_pass_a, s_n_a, s_r_a = _ratio(findings, "s_correct")
    ret_n = buckets.get("retrieval", 0)
    resp_n = buckets.get("response", 0)
    c_answerable = ret_n + resp_n                           # == C-probe passers
    # memory recovered an answer the raw transcript alone lost (C right, T wrong)
    self_corr_n = sum(1 for f in wrong if f.get("detail", {}).get("self_correction"))
    self_corr_all = sum(1 for f in findings if f.get("detail", {}).get("self_correction"))

    def _col(p, n, r):
        return f"{p:3d}/{n:<3d} ({100*r:3.0f}%)"

    print("\n================ ERROR PROBE SUMMARY ================")
    print(f"instance       : {args.instance_dir}")
    print(f"questions      : {total}")
    print(f"correct        : {correct}")
    print(f"failed         : {len(wrong)}")
    print("---- probe pass rates ----          "
          f"{'failed QAs':>16s}  {'all QAs':>16s}")
    print(f"  original QA accuracy      "
          f"{'—':>16s}  {_col(correct, total, correct/max(total,1)):>16s}")
    if args.run_golden:
        print(f"  G-probe (gold, ceiling)  {_col(g_pass, g_n, g_r):>16s}  {_col(g_pass_a, g_n_a, g_r_a):>16s}")
    else:
        print(f"  G-probe (gold, ceiling)  skipped (pass --run_golden to enable)")
    print(f"  T-probe (transcript)     {_col(t_pass, t_n, t_r):>16s}  {_col(t_pass_a, t_n_a, t_r_a):>16s}")
    print(f"  C-probe (construction)   {_col(c_pass, c_n, c_r):>16s}  {_col(c_pass_a, c_n_a, c_r_a):>16s}")
    print(f"  S-probe (retr. subset)   {_col(s_pass, s_n, s_r):>16s}  {_col(s_pass_a, s_n_a, s_r_a):>16s}")
    print(f"  retrieval split          response {resp_n} / retrieval {ret_n}  "
          f"(of {c_answerable} C-answerable; shown-subset behavioral)")
    print(f"  self-correction          failed {self_corr_n} / all {self_corr_all}  "
          f"(memory answered where raw transcript alone failed: C right, T wrong)")
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
           "summary": {"total": total, "correct": correct, "failed": len(wrong),
                       "original_qa_accuracy": correct / max(total, 1),
                       "probe_pass_rates": {
                           "g_probe": {"passed": g_pass, "ran": g_n, "ratio": g_r},
                           "c_probe": {"passed": c_pass, "ran": c_n, "ratio": c_r},
                           "t_probe": {"passed": t_pass, "ran": t_n, "ratio": t_r},
                           "s_probe": {"passed": s_pass, "ran": s_n, "ratio": s_r}},
                       "probe_pass_rates_all": {
                           "g_probe": {"passed": g_pass_a, "ran": g_n_a, "ratio": g_r_a},
                           "c_probe": {"passed": c_pass_a, "ran": c_n_a, "ratio": c_r_a},
                           "t_probe": {"passed": t_pass_a, "ran": t_n_a, "ratio": t_r_a},
                           "s_probe": {"passed": s_pass_a, "ran": s_n_a, "ratio": s_r_a}},
                       "retrieval_split": {"response": resp_n, "retrieval": ret_n,
                                           "c_answerable": c_answerable,
                                           "method": "shown_subset_behavioral"},
                       "self_correction": {"failed": self_corr_n, "all": self_corr_all},
                       "attribution": dict(buckets)},
           # consolidated QA -> chunk -> memory-id provenance for every probed question
           "qa_evidence_map": [f["detail"]["evidence_map"]
                               for f in findings
                               if isinstance(f.get("detail"), dict)
                               and "evidence_map" in f["detail"]],
           "findings": findings}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote per-question probe -> {args.out}")

    # also dump the QA -> chunk -> memory-id mapping on its own for easy consumption
    map_path = os.path.join(os.path.dirname(args.out) or ".", "qa_evidence_map.json")
    with open(map_path, "w") as f:
        json.dump(out["qa_evidence_map"], f, indent=2, ensure_ascii=False)
    print(f"Wrote QA->chunk->memory-id mapping -> {map_path}  "
          f"({len(out['qa_evidence_map'])} questions)")


def parse_args():
    p = argparse.ArgumentParser(description="Behavioral error attribution for Mem-alpha QA.")
    p.add_argument("--instance_dir", required=True,
                   help="Dir with results.json / agent_state.json / chunks_and_function_calls.json")
    p.add_argument("--qa_file", default="outputs/step3_anony/qas/merged_qa_anoy.jsonl")
    p.add_argument("--data_root", default="outputs/step3_anony/S01_S03_Clean_Anoy",
                   help="Single dialogue root holding, per chunk, BOTH parsed_dialog_gt.json "
                        "(gold evidence, real names) and parsed_dialog_pred.json (ASR "
                        "transcript for the T-probe). Layout: <root>/<episode>/CHUNK_*/.")
    p.add_argument("--parquet", default=None,
                   help="step3 parquet whose `chunk_folders` list maps chunk_idx -> source "
                        "folder ('{episode}/CHUNK_N'), the deterministic QA->chunk->memory-id "
                        "link (see prepare_parquet_from_step3.py). Default: auto-discover "
                        "dataset_gt_name_*.parquet in --data_root.")
    p.add_argument("--server_url", default="http://127.0.0.1:5005/batch_process",
                   help="Memory server /batch_process endpoint (same one the run used).")
    p.add_argument("--data_source", default="seamlessinteraction_options",
                   help="Datasource key in the prompts yaml; selects the query prompt "
                        "prepended to each question (matches run_qa_evaluation.py).")
    p.add_argument("--prompts_yaml", default="config/prompts_wrt_datasource.yaml",
                   help="Path to prompts_wrt_datasource.yaml holding the per-datasource query prompt.")
    p.add_argument("--scorer", choices=["keyword", "llm_judge"], default="keyword",
                   help="perltqa grading of each probe re-answer: 'keyword' (default, "
                        "containment; no API) or 'llm_judge' (needs QWEN_URL / "
                        "QWEN_MODEL_NAME / OPENROUTER_API_KEY). Ignored for the "
                        "multiple-choice datasources (they always use the letter match).")
    p.add_argument("--no_core", action="store_true",
                   help="Exclude the (global) final core string from the construction context.")
    p.add_argument("--full_qa", action="store_true",
                   help="Probe EVERY question (including the ones the real run got right), "
                        "instead of only the failures. Correct records stay in the 'correct' "
                        "bucket but still get G/C/T/S probes recorded for analysis.")
    p.add_argument("--run_golden", action="store_true",
                   help="Run the G-probe (re-answer on the whole GOLD dialogue chunk, the "
                        "upper-bound ceiling). Skipped by default to save QA server calls.")
    p.add_argument("--min_turn_words", type=int, default=0)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=64,
                   help="Questions per server POST for the C/S probes. Smaller batches "
                        "print progress more often and keep each POST under --timeout.")
    p.add_argument("--timeout", type=int, default=1200,
                   help="Per-POST read timeout in seconds for the C/S probes (default 1200).")
    p.add_argument("--debug", action="store_true",
                   help="Dump per-stage (G/C/T) qa_prompt, agent answer, gold "
                        "answer, and the evidence fed to the QA model -> "
                        "<instance_dir>/error_probe_debug.json.")
    p.add_argument("--out", default=None, help="Output json (default: <instance_dir>/error_probe.json)")
    p.add_argument("--tg_probe_cache", default=None,
                   help="Shared per-DATA_ROOT JSON cache of T-probe (transcript) and "
                        "G-probe (gold) answers, keyed by a content hash of (kind, "
                        "prompt, dialogue ctx, decoding/server). These re-answers are "
                        "instance/seed-independent, so one cache serves every base_dir / "
                        "seed / compression variant sharing the DATA_ROOT. When set, the "
                        "T/G stages load READ-ONLY from this cache and a missing entry is "
                        "a hard error — populate it first with "
                        "diagnostic/precompute_tg_probes.py. Default: off (T/G hit the "
                        "server directly, as before).")
    args = p.parse_args()
    if args.out is None:
        args.out = os.path.join(args.instance_dir, "error_probe.json")
    return args


if __name__ == "__main__":
    probe(parse_args())
