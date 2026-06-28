#!/usr/bin/env python3
"""
matching.py — evidence/memory similarity for the Mem-alpha error tracer.

A layered matcher answers one question: "is this evidence turn present in this
memory text?"

    lexical  (always on) : phrase / n-gram containment, locality-enforcing
    embedding (optional) : OpenAI text-embedding-3-small cosine
    llm judge (optional) : OpenRouter entailment check

The embedding / LLM layers auto-enable when API keys are present in the
environment; otherwise the matcher runs on the lexical signal alone.

Public API:
    lexical_score(evidence, candidate) -> float
    EmbeddingMatcher, LLMJudge
    present(evidence_turns, records, emb, judge) -> (bool, info)
    evidence_rank(evidence_turns, episodic_records) -> (best_rank, ranks)
    LEX_TAU, EMB_TAU, COVERAGE_TAU
"""

import os
import re
from functools import lru_cache

import numpy as np


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #
LEX_TAU = 0.6        # per-turn: fraction of a turn's n-grams present in a unit
EMB_TAU = 0.55       # per-turn cosine for text-embedding-3-small
COVERAGE_TAU = 0.5   # evidence is "present" if >= this fraction of its turns are found
JUDGE_MIN_CHARS = 15 # skip the LLM judge for utterances this short (too little
                     # content to entail reliably; would only add noise/cost)


# --------------------------------------------------------------------------- #
# Tokenization + lexical (phrase-containment) matcher
# --------------------------------------------------------------------------- #
_STOP = set("""a an the of to in on at for and or but if is are was were be been being
this that these those with as by from it its his her their he she they we you i me my your
he's she's said say says was were do does did has have had not no yes will would can could
should about over after before then there here what who whom whose which when where why how""".split())


def _tokens(text):
    """Tokenize and drop apostrophes per token, so "can't" -> "cant" and an
    opening-quote "'soft" -> "soft" on both sides.

    Spaced contractions in the dialog source ("ca n't", "they 're") are already
    reconciled with the joined memory forms ("can't", "they're") upstream by
    data_utils.fix_space_in_text at load time, so no contraction merging is needed
    here."""
    raw = re.findall(r"[a-z0-9']+", (text or "").lower())
    return [t for t in (tok.replace("'", "") for tok in raw) if t]


@lru_cache(maxsize=300000)
def _content_tokens(text):
    """Content words (stopwords + <=2 char tokens removed). Cached: memory units
    are matched against many turns, so the candidate side repeats heavily."""
    return tuple(t for t in _tokens(text) if t not in _STOP and len(t) > 2)


def _utterance(turn):
    """Strip the 'speaker: ' prefix so matching uses the spoken text. The speaker
    prefix would otherwise shift n-grams and break short verbatim quotes."""
    return turn.split(": ", 1)[1] if ": " in turn else turn


def _speaker(turn):
    """The 'speaker' part before the first ': ' (complement of _utterance)."""
    return turn.split(": ", 1)[0] if ": " in turn else ""


def speaker_match(gt_name, pred_name):
    """Fuzzy speaker-name match (gold vs predicted/transcription name).

    Tolerates first-name-only and partial predictions, e.g. predicted 'Sheldon'
    matches gold 'sheldon_cooper'. Not exact: a predicted name matches if it is a
    substring of the gold name, or the gold name starts with the predicted name or
    its first token.
    """
    if not gt_name or not pred_name:
        return False
    g = gt_name.strip().lower()
    p = pred_name.strip().lower()
    first_p = p.split()[0] if p.split() else p
    return (p in g) or g.startswith(p) or g.startswith(first_p)


def _ngrams(tokens, n):
    return set(zip(*(tokens[i:] for i in range(n)))) if len(tokens) >= n else set()


@lru_cache(maxsize=300000)
def _ngram_index(text):
    """Cache a candidate's content unigrams/bigrams/trigrams (it's matched against
    many turns)."""
    toks = _content_tokens(text)
    return set(toks), _ngrams(toks, 2), _ngrams(toks, 3)


def lexical_score(evidence, candidate):
    """Phrase-containment match of an evidence turn against a candidate memory unit.

    score = fraction of the turn's content-word n-grams that appear in the
    candidate, using the longest n that the turn supports (trigrams, else bigrams,
    else a single content word). Computed on the UTTERANCE only.

    This is both length-symmetric (works for a 3-word or a 30-word turn) and
    locality-enforcing (n-grams require contiguous co-occurrence), so a long memory
    unit can no longer match on a few scattered shared words.
    """
    ev_tokens = _content_tokens(_utterance(evidence))
    if not ev_tokens:
        return 0.0
    cand_uni, cand_bi, cand_tri = _ngram_index(candidate)
    if not cand_uni:
        return 0.0

    ev_tri = _ngrams(ev_tokens, 3)
    if ev_tri:
        return len(ev_tri & cand_tri) / len(ev_tri)
    ev_bi = _ngrams(ev_tokens, 2)
    if ev_bi:
        return len(ev_bi & cand_bi) / len(ev_bi)
    return 1.0 if ev_tokens[0] in cand_uni else 0.0


# --------------------------------------------------------------------------- #
# Optional embedding matcher (OpenAI text-embedding-3-small, same as memory.py)
# --------------------------------------------------------------------------- #
class EmbeddingMatcher:
    def __init__(self):
        self.enabled = False
        self.client = None
        self.cache = {}
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            try:
                from openai import OpenAI
                self.client = OpenAI()
                self.enabled = True
            except Exception as e:
                print(f"[embed] disabled ({e})")

    def embed(self, texts):
        out = []
        todo = [t for t in texts if t not in self.cache]
        if todo:
            resp = self.client.embeddings.create(model="text-embedding-3-small", input=todo)
            for t, d in zip(todo, resp.data):
                self.cache[t] = np.array(d.embedding, dtype=np.float32)
        for t in texts:
            out.append(self.cache[t])
        return np.vstack(out)

    def best_score_idx(self, evidence, candidates):
        """Return (max_cosine, argmax_index) of evidence vs candidate texts."""
        if not self.enabled or not candidates:
            return 0.0, -1
        ev = self.embed([evidence])
        ca = self.embed(candidates)
        ev = ev / (np.linalg.norm(ev, axis=1, keepdims=True) + 1e-9)
        ca = ca / (np.linalg.norm(ca, axis=1, keepdims=True) + 1e-9)
        sims = ca @ ev[0]
        i = int(sims.argmax())
        return float(sims[i]), i

    def best_score(self, evidence, candidates):
        return self.best_score_idx(evidence, candidates)[0]


# --------------------------------------------------------------------------- #
# Optional LLM entailment judge (OpenRouter, used to confirm borderline matches)
# --------------------------------------------------------------------------- #
class LLMJudge:
    def __init__(self):
        self.enabled = False
        self.client = None
        self.model = os.environ.get("QWEN_MODEL_NAME", "qwen/qwen3-32b")
        key = os.environ.get("OPENROUTER_API_KEY")
        url = os.environ.get("QWEN_URL", "https://openrouter.ai/api/v1")
        if key:
            try:
                from openai import OpenAI
                self.client = OpenAI(base_url=url, api_key=key)
                self.enabled = True
            except Exception as e:
                print(f"[judge] disabled ({e})")

    def entails(self, evidence, candidate):
        if not self.enabled:
            return None
        prompt = (
            "Does TEXT contain or imply the FACT? Answer only 'yes' or 'no'.\n\n"
            f"FACT: {evidence}\n\nTEXT: {candidate}\n\nAnswer:"
        )
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=8,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            print(prompt)
            print(r.choices[0].message.content )
            print()
            print()
            return "yes" in (r.choices[0].message.content or "").strip().lower()
        except Exception as e:
            print(f"[judge] error ({e})")
            return None


# --------------------------------------------------------------------------- #
# Layered presence test
# --------------------------------------------------------------------------- #
def _turn_present(turn, records, emb, judge, match_speaker=False,
                  use_emb=True, use_judge=True):
    """Is a single evidence turn present in any memory record? Returns (bool, info).

    `records` are dicts {id, mtype, text} (and {speaker} when match_speaker is used).
    The returned info reports which memory unit matched (memory_id, memory_type),
    how (method), and the score.

    If match_speaker is True, only records whose speaker fuzzily matches the
    evidence turn's speaker are considered — so a turn counts as present only when
    BOTH its content and its speaker are preserved.
    """
    if match_speaker:
        spk = _speaker(turn)
        records = [r for r in records if speaker_match(spk, r.get("speaker", ""))]
    if not records:
        return False, {"method": "none", "score": 0.0, "memory_id": None,
                       "memory_type": None, "top3_lex": []}

    def info(method, score, i):
        r = records[i]
        return {"method": method, "score": score,
                "memory_id": r.get("id"), "memory_type": r.get("mtype"),
                "unit": r.get("text", "")[:160]}

    lex = sorted(((lexical_score(turn, r["text"]), i) for i, r in enumerate(records)),
                 key=lambda x: -x[0])
    # Top-3 lexical candidates, recorded on every result (hit OR miss) so the trace
    # always shows which memory units were the closest lexical neighbours.
    top3 = [{"memory_id": records[i].get("id"), "memory_type": records[i].get("mtype"),
             "score": round(s, 3)} for s, i in lex[:3]]
    best_lex, best_i = lex[0]
    if best_lex >= LEX_TAU:
        out = info("lex", round(best_lex, 3), best_i)
        out["top3_lex"] = top3
        return True, out
    if use_emb and emb.enabled:
        es, ei = emb.best_score_idx(turn, [r["text"] for r in records])
        if es >= EMB_TAU:
            out = info("emb", round(es, 3), ei)
            out["top3_lex"] = top3
            return True, out
    if use_judge and judge.enabled and len(_utterance(turn).strip()) > JUDGE_MIN_CHARS:
        for wi, i in lex[:3]:
            if judge.entails(turn, records[i]["text"]):
                out = info("llm", None, i)
                out["top3_lex"] = top3
                return True, out
    out = info("miss", round(best_lex, 3), best_i)   # best_unit kept for debugging the miss
    out["matched"] = False
    out["top3_lex"] = top3
    return False, out


def present(evidence_turns, records, emb, judge, match_speaker=False,
            use_emb=False, use_judge=False):
    """Coverage-based presence over a SET of evidence turns.

    Returns (matched: bool, info) where matched means coverage >= COVERAGE_TAU.
    info["matches"] lists, per evidence turn, whether it was found and in which
    memory unit (memory_id / memory_type), so attribution is fully auditable.

    match_speaker=True additionally requires the matched record's speaker to
    fuzzily match the evidence turn's speaker (used for the transcription stage).

    use_emb / use_judge let a caller disable the embedding or LLM-judge tiers for
    THIS call even when emb.enabled / judge.enabled are globally on (e.g. run the
    construction stage lexical-only). Lexical matching is always on.
    """
    if not evidence_turns:
        return False, {"coverage": 0.0, "matched": 0, "total": 0, "missing": [], "matches": []}
    hits, missing, matches = 0, [], []
    for t in evidence_turns:
        ok, tinfo = _turn_present(t, records, emb, judge, match_speaker=match_speaker,
                                  use_emb=use_emb, use_judge=use_judge)
        matches.append({
            "turn": t[:120],
            "found": ok,
            "memory_id": tinfo.get("memory_id") if ok else None,
            "memory_type": tinfo.get("memory_type") if ok else None,
            "method": tinfo.get("method"),
            "score": tinfo.get("score"),
            # Top-3 lexical neighbours, recorded even on a miss for debugging.
            "top3_lex": tinfo.get("top3_lex", []),
        })
        if ok:
            hits += 1
        else:
            missing.append(t[:80])
    cov = hits / len(evidence_turns)
    return cov >= COVERAGE_TAU, {
        "coverage": round(cov, 3), "matched": hits, "total": len(evidence_turns),
        "missing": missing[:6], "matches": matches,
    }


SENT_MIN_CONTENT_TOKENS = 2   # sentences with fewer content words are too short to match reliably


def _split_sentences(utterance):
    """Split an utterance into sentences on . ! ? boundaries (empties dropped)."""
    return [s.strip() for s in re.split(r"[.!?]+", utterance) if s.strip()]


def present_sentencewise(evidence_turns, records, match_speaker=True):
    """Coverage test that decomposes each evidence turn into sentences before matching.

    A gold evidence turn is often a speaker-MERGE of several utterances, which both
    (a) spreads its content across several finely-segmented transcript turns and
    (b) invents spurious cross-sentence n-grams that match nothing — so scoring the
    whole turn against a single candidate under-counts. Splitting on sentence
    boundaries removes the phantom n-grams and lets each sentence land in the single
    transcript turn that actually contains it.

    Per turn: drop trivially short sentences (< SENT_MIN_CONTENT_TOKENS content
    words); each remaining sentence is "found" if its best lexical score over the
    (same-speaker) candidates clears LEX_TAU; the turn is preserved if the
    content-token-weighted fraction of found sentences clears COVERAGE_TAU (the
    fact-bearing sentence thus weighs more than filler). Overall presence = fraction
    of preserved turns >= COVERAGE_TAU. Lexical-only, matching the transcription
    stage (which runs without the embedding / judge tiers).
    """
    if not evidence_turns:
        return False, {"coverage": 0.0, "matched": 0, "total": 0, "missing": [], "matches": []}

    hits, missing, matches = 0, [], []
    for t in evidence_turns:
        recs = records
        if match_speaker:
            spk = _speaker(t)
            recs = [r for r in records if speaker_match(spk, r.get("speaker", ""))]

        sub = [s for s in _split_sentences(_utterance(t))
               if len(_content_tokens(s)) >= SENT_MIN_CONTENT_TOKENS]
        if not sub:                         # all sentences trivial -> fall back to whole utterance
            whole = _utterance(t)
            sub = [whole] if _content_tokens(whole) else []

        sent_info, found_w, total_w = [], 0.0, 0.0
        for s in sub:
            w = len(_content_tokens(s)) or 1
            best = max((lexical_score(s, r["text"]) for r in recs), default=0.0)
            ok = best >= LEX_TAU
            total_w += w
            found_w += w if ok else 0
            sent_info.append({"sentence": s[:80], "found": ok, "score": round(best, 3), "weight": w})

        cov = (found_w / total_w) if total_w else 0.0
        turn_ok = bool(recs) and cov >= COVERAGE_TAU
        matches.append({"turn": t[:120], "found": turn_ok, "method": "sentencewise",
                        "score": round(cov, 3), "speaker_records": len(recs),
                        "sentences": sent_info})
        if turn_ok:
            hits += 1
        else:
            missing.append(t[:80])

    cov = hits / len(evidence_turns)
    return cov >= COVERAGE_TAU, {
        "coverage": round(cov, 3), "matched": hits, "total": len(evidence_turns),
        "missing": missing[:6], "matches": matches,
    }


def evidence_rank(evidence_turns, episodic_records):
    """For each evidence turn, its best episodic match's rank (1-based) and id.

    Returns (best_rank, ranks) where ranks is a list of
    {turn, rank, memory_id} for turns whose best lexical match clears LEX_TAU.
    """
    best = None
    ranks = []
    for t in evidence_turns:
        scored = sorted(((lexical_score(t, r["text"]), i) for i, r in enumerate(episodic_records)),
                        key=lambda x: -x[0])
        if scored and scored[0][0] >= LEX_TAU:
            i = scored[0][1]
            r = i + 1
            ranks.append({"turn": t[:80], "rank": r, "memory_id": episodic_records[i].get("id")})
            best = r if best is None else min(best, r)
    ranks.sort(key=lambda x: x["rank"])
    return best, ranks
