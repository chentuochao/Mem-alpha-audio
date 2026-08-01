#!/usr/bin/env python3
"""
data_utils.py — loading & parsing for the Mem-alpha error tracer.

Handles everything that touches disk or raw JSON shapes:
    - answer parsing      : extract_choice, gold_letter
    - QA + dialog loading : load_qa, DialogResolver, evidence_texts, evidence_episodes
    - transcribed dialog  : TranscriptLoader (ASR + speaker naming, per episode)
    - memory flattening   : memory_records, retrieved_records

Each loader returns plain dicts/lists so the cascade and matcher stay decoupled
from the on-disk schema.
"""

import os
import re
import json
import string

from matching import _utterance


# --------------------------------------------------------------------------- #
# Text normalization (applied at load time)
# --------------------------------------------------------------------------- #
def fix_space_in_text(text):
    """Collapse Penn-Treebank spacing so the dialog source matches the memory store.

    Removes the space before punctuation ("Hi ." -> "Hi.") and before contraction
    tails ("ca n't" -> "can't", "they 're" -> "they're"). The apostrophe/quote
    characters are intentionally EXCLUDED from the generic punctuation pass so that
    opening quotes ("said 'soft") are not glued onto the previous word; only the
    explicit contraction suffixes below are joined.
    """
    if not text:
        return text
    patterns = [" " + c for c in string.punctuation if c not in "'\""]
    patterns += [" n't", " 'm", " 's", " 've", " 're", " 'll", " 'd",
                 " 't", " 'y", " 'z"]
    for p in patterns:
        text = text.replace(p, p.strip())
    return text


# --------------------------------------------------------------------------- #
# Correctness scoring (reuses the \boxed{X} convention from evaluate_agent_results.py)
# --------------------------------------------------------------------------- #
def extract_choice(text):
    """Extract a single multiple-choice letter from a model response."""
    if not text:
        return None
    m = re.search(r'\\boxed\{([A-Za-z])\}', text)
    if m:
        return m.group(1).upper()
    # Fallbacks: "answer is A", "(A)", leading "A."
    m = re.search(r'answer\s*(?:is|:)\s*\(?([A-Za-z])\)?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'\(([A-Za-z])\)', text)
    if m:
        return m.group(1).upper()
    m = re.match(r'\s*([A-Za-z])[.)]', text)
    if m:
        return m.group(1).upper()
    return None


def gold_letter(answer):
    """'A. Invite Penny over for lunch' -> 'A'."""
    if not answer:
        return None
    m = re.match(r'\s*([A-Za-z])[.)]', answer.strip())
    return m.group(1).upper() if m else None


# --------------------------------------------------------------------------- #
# QA + dialog loading
# --------------------------------------------------------------------------- #
def load_qa(qa_file):
    """Read a QA jsonl (one object per line) into a list of dicts."""
    items = []
    with open(qa_file) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


class DialogResolver:
    """Resolve a gt_source `file` to its turn list, for BOTH QA schemas.

    - Old whole-episode schema: `file` is a bare '<episode>.json' (unique basename),
      resolved via a basename index over the roots.
    - Chunked schema: `file` is a path RELATIVE to a root, e.g.
      '<episode>/CHUNK_n/parsed_dialog_gt.json'. Basenames ('parsed_dialog_gt.json')
      are NOT unique, so this is resolved by joining the relative path onto each root.

    Each question's evidence_turns index into ITS OWN file, so we load the right
    dialog per source rather than assuming a single transcript. Loaded dialogs are
    cached by resolved path.
    """

    def __init__(self, roots):
        self.roots = [r for r in (roots or []) if r and os.path.isdir(r)]
        self.index = {}      # basename -> path (old whole-episode schema)
        self.cache = {}      # path -> turns
        for root in self.roots:
            for dirpath, _, files in os.walk(root):
                for fn in files:
                    if fn.endswith(".json"):
                        self.index.setdefault(fn, os.path.join(dirpath, fn))

    def _load(self, path):
        if not path:
            return None
        if path not in self.cache:
            try:
                with open(path) as f:
                    self.cache[path] = json.load(f)
            except Exception:
                self.cache[path] = None
        return self.cache[path]

    def turns_for(self, file_field):
        if not file_field:
            return None
        # chunked schema: file carries a directory -> resolve the full relative path
        if os.path.dirname(file_field):
            for root in self.roots:
                p = os.path.join(root, file_field)
                if os.path.isfile(p):
                    return self._load(p)
        # old whole-episode schema: resolve by (unique) basename
        return self._load(self.index.get(os.path.basename(file_field)))


def _turn_index(turn_id):
    """Turn ID -> 0-based index into its dialog file, for BOTH schemas.

    'S01E02_C001_T003' -> 3 (chunked string ID); a bare int is returned as-is (old
    schema). Returns None if a string carries no trailing T<NNN> suffix.
    """
    if isinstance(turn_id, int):
        return turn_id
    m = re.search(r"[Tt](\d+)\s*$", str(turn_id))
    return int(m.group(1)) if m else None


def evidence_texts(qa, resolver, min_turn_words=3):
    """Return (texts, unresolved_files) for a QA item's gold evidence turns.

    texts: list of "speaker: text" strings pulled from each source's own dialog.
    Handles both the old (integer turn index) and chunked (string 'T<NNN>' turn ID)
    schemas via `_turn_index`. Turns whose utterance has fewer than `min_turn_words`
    words are dropped (too short to match reliably). If that would drop ALL turns, the
    unfiltered list is kept so the question is not lost. unresolved_files: source
    files that could not be located on disk.
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
    """Episode name(s) a question's gold evidence comes from, for BOTH schemas.

    - Old schema: `file` = '<episode>.json' -> strip the '.json' extension.
    - Chunked schema: `file` = '<episode>/CHUNK_n/parsed_dialog_gt.json' -> the leading
      path component IS the episode dir.
    Used to scope transcription matching to the evidence's OWN episode."""
    gt = qa.get("gt_source", {})
    sources = gt.get("sources")
    if sources is None and "evidence_turns" in gt:
        sources = [gt]
    episodes = set()
    for src in (sources or []):
        f = src.get("file", "") or ""
        if os.path.dirname(f):                              # chunked: leading dir
            ep = f.replace(os.sep, "/").split("/")[0]
        else:                                              # old: bare '<episode>.json'
            ep = os.path.basename(f)
            if ep.endswith(".json"):
                ep = ep[:-len(".json")]
        if ep:
            episodes.add(ep)
    return episodes


# --------------------------------------------------------------------------- #
# Transcribed dialogue (ASR + speaker-naming step, BEFORE memory construction)
# --------------------------------------------------------------------------- #
class TranscriptLoader:
    """Loads the transcribed dialogue produced upstream of memory construction.

    Layout: <root>/<episode>/CHUNK_*/parsed_dialog_pred.json, each a list of turns
    {speaker, start, end, text}. `episode_records` concatenates all of an episode's
    chunks into matcher records {id, mtype='transcript', text}, cached per episode.
    Matching is scoped per episode (the transcript carries the episode), so we never
    match evidence across episodes here.
    """

    def __init__(self, root, pred_filename="parsed_dialog_pred.json"):
        self.root = root
        self.pred_filename = pred_filename
        self.cache = {}      # episode -> records
        self.available = bool(root) and os.path.isdir(root)

    @staticmethod
    def _chunk_key(name):
        m = re.search(r"(\d+)", name)
        return (0, int(m.group(1))) if m else (1, name)

    def chunk_records(self, episode, chunk):
        """Transcript records for ONE chunk (<root>/<episode>/<chunk>/pred file).
        Cached per (episode, chunk)."""
        key = (episode, chunk)
        if key in self.cache:
            return self.cache[key]
        records = []
        p = os.path.join(self.root, episode, chunk, self.pred_filename) if self.root else ""
        if p and os.path.isfile(p):
            try:
                with open(p) as f:
                    turns = json.load(f)
            except Exception:
                turns = []
            for i, t in enumerate(turns):
                speaker = t.get("speaker", "?")
                text = fix_space_in_text(f"{speaker}: {t.get('text','')}")
                records.append({"id": f"{episode}/{chunk}:{i}", "mtype": "transcript",
                                "speaker": speaker, "text": text})
        self.cache[key] = records
        return records

    def episode_records(self, episode):
        if episode in self.cache:
            return self.cache[episode]
        records = []
        epdir = os.path.join(self.root, episode) if self.root else ""
        if epdir and os.path.isdir(epdir):
            for ch in sorted(os.listdir(epdir), key=self._chunk_key):
                records.extend(self.chunk_records(episode, ch))
        self.cache[episode] = records
        return records

    def records_for_episodes(self, episodes):
        """Concatenated transcript records for a set of episodes (union)."""
        out = []
        for ep in episodes:
            out.extend(self.episode_records(ep))
        return out

    def records_for_chunks(self, chunks):
        """Concatenated transcript records for a set of (episode, chunk) pairs."""
        out = []
        for ep, ch in chunks:
            out.extend(self.chunk_records(ep, ch))
        return out


# --------------------------------------------------------------------------- #
# Memory flattening
# --------------------------------------------------------------------------- #
def _records_from_blob(blob):
    """Turn a memory blob (core str + episodic/semantic lists of {hash:text}) into
    a flat list of records {id, mtype, text}, preserving the unit hash id and type."""
    records, episodic = [], []
    if not blob:
        return records, episodic
    core = blob.get("core")
    if isinstance(core, list):
        core = " ".join(str(x) for x in core)
    if core:
        records.append({"id": "core", "mtype": "core", "text": fix_space_in_text(core)})
    for mtype in ("episodic", "semantic"):
        for d in blob.get(mtype) or []:
            if isinstance(d, dict) and d:
                uid, text = next(iter(d.items()))
            else:
                uid, text = None, str(d)
            rec = {"id": uid, "mtype": mtype, "text": fix_space_in_text(text)}
            records.append(rec)
            if mtype == "episodic":
                episodic.append(rec)
    return records, episodic


def memory_records(state):
    """Stored agent_state -> (all_records, episodic_records)."""
    return _records_from_blob(state)


def retrieved_records(retrieved):
    """results.json retrieved_memory -> (all_records, episodic_records)."""
    return _records_from_blob(retrieved)
