# speaker_registry.py — Qwen3 edition
# Qwen3 is accessed via its OpenAI-compatible API.
# Set base_url to your provider:
#   - Local vLLM:      http://localhost:8000/v1
#   - Together AI:     https://api.together.xyz/v1
#   - DashScope:       https://dashscope.aliyuncs.com/compatible-mode/v1
# Set QWEN_API_KEY in your environment accordingly.

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Literal, Optional

from openai import OpenAI

# ── Client setup ────────────────────────────────────────────────────────────

client = OpenAI(
    api_key=os.environ.get("QWEN_API_KEY", "EMPTY"),  # "EMPTY" works for local vLLM
    base_url=os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1"),
)

QWEN3_MODEL = os.environ.get("QWEN3_MODEL", "Qwen/Qwen3-32B")

# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Evidence:
    dialogue_id: str
    cue_type: str        # "vocative" | "self_intro" | "third_person"
    raw_text: str
    confidence: float

@dataclass
class SpeakerRecord:
    speaker_id: str
    name: Optional[str] = None
    status: Literal["unknown", "candidate", "confirmed"] = "unknown"
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        return self.name if self.name else self.speaker_id


# ── Qwen3 helpers ────────────────────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    """
    Qwen3 thinking-mode responses wrap internal reasoning in <think>…</think>.
    Strip that block so only the final answer reaches the JSON parser.
    The block may span multiple lines, so we use re.DOTALL.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def qwen3_chat(
    system: str,
    user: str,
    *,
    enable_thinking: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """
    Single-turn chat with a Qwen3 model.

    Qwen3 thinking mode is controlled by two mechanisms:
      1. Append /think or /no_think to the *last user message* — the model
         respects this at inference time regardless of sampling params.
      2. Some deployments expose an extra_body flag; we support both.

    We default to /no_think for extraction calls because we only need the
    JSON output, not the chain-of-thought — this saves tokens and latency.
    Change enable_thinking=True for debugging tricky dialogues.
    """
    suffix = " /think" if enable_thinking else " /no_think"
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user + suffix},
    ]

    response = client.chat.completions.create(
        model=QWEN3_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        # Qwen3 on vLLM / DashScope also accepts this flag in extra_body:
        extra_body={
            "enable_thinking": enable_thinking,
        },
    )

    raw = response.choices[0].message.content or ""
    return strip_thinking(raw)


# ── Prompt templates ─────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """
You are a speaker identity analyst. You receive a diarized conversation transcript
where speakers are labeled Speaker_0, Speaker_1, etc. Your job is to detect any
linguistic cues that reveal the real name of a speaker.

## Cue types to detect (in order of reliability)

1. VOCATIVE — one speaker directly addresses another by name.
   Example: <Speaker_1> "Leonard, I don't think I can do this."
   → Speaker_1 is addressing Speaker_0 as "Leonard", so Speaker_0 = Leonard.
   ⚠️  The speaker of the utterance is NOT the named person. The name belongs to
       whoever is being spoken TO — infer from context.

2. SELF_INTRO — a speaker introduces themselves.
   Example: <Speaker_2> "Hi, I'm Penny."
   → Speaker_2 = Penny.

3. THIRD_PERSON_REF — a speaker refers to another by name in the third person,
   and context makes the referent unambiguous.
   Example: <Speaker_1> "Sheldon, this was your idea."
   → Speaker_1 is addressing Speaker_0 and uses the name Sheldon, so Speaker_0 = Sheldon.

## Output format — JSON ONLY, no prose, no markdown fences

{
  "extractions": [
    {
      "speaker_id": "Speaker_X",
      "name": "ActualName",
      "cue_type": "vocative" | "self_intro" | "third_person",
      "evidence_utterance": "exact quote from transcript",
      "confidence": 0.0–1.0,
      "reasoning": "one sentence"
    }
  ]
}

If no cues are found, return exactly: {"extractions": []}

## Confidence guide
- 1.0  Direct self-introduction
- 0.9  Clear vocative, unambiguous referent
- 0.7  Vocative but referent could be ambiguous
- 0.5  Third-person reference with some ambiguity
- 0.3  Indirect / speculative — do NOT include below 0.5

## Hard rules
- Only output names that appear literally in the transcript text.
- Never invent, guess, or hallucinate a name.
- If the same speaker appears in multiple cues, include each as a separate entry.
- If a speaker is named differently in two cues, include both and set confidence ≤ 0.6.
""".strip()


def build_extraction_prompt(dialogue: str, registry: dict[str, SpeakerRecord]) -> str:
    known = {sid: rec.name for sid, rec in registry.items() if rec.status != "unknown"}
    prior = f"Already identified speakers: {known}" if known else "No speakers identified yet."
    return f"""{prior}

## New dialogue to analyze

{dialogue}

Analyze the dialogue. Return only valid JSON with no markdown fences."""


# ── Fuzzy name grouping ──────────────────────────────────────────────────────

def _name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _group_name_variants(
    names: list[str], threshold: float = 0.80
) -> dict[str, str]:
    """Map each name variant to a canonical form (most frequent spelling)."""
    if not names:
        return {}
    counts = Counter(n.lower() for n in names)
    sorted_names = sorted(set(n.lower() for n in names), key=lambda n: -counts[n])
    canonical_map: dict[str, str] = {}
    groups: list[list[str]] = []

    for name in sorted_names:
        merged = False
        for group in groups:
            if _name_similarity(name, group[0]) >= threshold:
                group.append(name)
                merged = True
                break
        if not merged:
            groups.append([name])

    for group in groups:
        canon = max(group, key=lambda n: counts[n])
        for variant in group:
            canonical_map[variant] = canon
    return canonical_map


# ── Registry update logic (buffer evidence, defer name assignment) ───────────

def buffer_evidence(
    registry: dict[str, SpeakerRecord],
    extractions: list[dict],
    dialogue_id: str,
    min_confidence: float = 0.5,
) -> dict[str, SpeakerRecord]:
    """Append evidence to the registry without resolving names yet."""
    for ext in extractions:
        sid  = ext["speaker_id"]
        name = ext["name"].strip().title()
        conf = float(ext["confidence"])

        if conf < min_confidence:
            continue

        if sid not in registry:
            registry[sid] = SpeakerRecord(speaker_id=sid)

        rec = registry[sid]
        ev = Evidence(
            dialogue_id=dialogue_id,
            cue_type=ext["cue_type"],
            raw_text=ext["evidence_utterance"],
            confidence=conf,
        )
        ev._candidate_name = name  # type: ignore[attr-defined]
        rec.evidence.append(ev)

    return registry


def resolve_registry(
    registry: dict[str, SpeakerRecord],
    confirm_threshold: int = 2,
    fuzzy_threshold: float = 0.80,
) -> dict[str, SpeakerRecord]:
    """
    Resolve speaker names from buffered evidence using weighted voting.

    For each speaker:
      1. Collect all candidate names from evidence.
      2. Group spelling variants via fuzzy matching.
      3. Score each canonical name = sum(confidence) across all matching evidence.
      4. Pick the highest-scoring candidate.
      5. Set status based on evidence strength.

    Then cross-validate: if two speakers got the same name, keep only the
    one with the stronger score and demote the other.
    """
    name_scores: dict[str, dict[str, float]] = {}

    for sid, rec in registry.items():
        if not rec.evidence:
            continue

        raw_names = [_extract_name_from_evidence(e) for e in rec.evidence]
        raw_names = [n for n in raw_names if n]

        if not raw_names:
            continue

        canon_map = _group_name_variants(raw_names, threshold=fuzzy_threshold)
        scores: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for ev, raw_name in zip(rec.evidence, [_extract_name_from_evidence(e) for e in rec.evidence]):
            if not raw_name:
                continue
            canon = canon_map.get(raw_name.lower(), raw_name.lower())
            scores[canon] += ev.confidence
            counts[canon] += 1

        if not scores:
            continue

        best_name = max(scores, key=lambda n: scores[n])
        best_score = scores[best_name]
        best_count = counts[best_name]

        rec.name = best_name.title()
        name_scores[sid] = {best_name: best_score}

        high_conf_count = sum(
            1 for ev in rec.evidence
            if ev.confidence >= 0.7
            and canon_map.get(_extract_name_from_evidence(ev, "").lower(), "") == best_name
        )
        if high_conf_count >= confirm_threshold or best_score >= 1.8:
            rec.status = "confirmed"
        elif best_count >= 1:
            rec.status = "candidate"

    _cross_validate(registry, name_scores)
    return registry


def _extract_name_from_evidence(ev: Evidence, default: str | None = None) -> str | None:
    """Extract the candidate name stored alongside evidence.

    We piggyback on the cue_type field naming: the evidence raw_text
    contains the utterance, but we need the candidate name.  Since
    buffer_evidence stores Evidence per extraction and the name is set
    on the record, we attach the name to Evidence via a lightweight
    attribute.  If not present, return default.
    """
    return getattr(ev, "_candidate_name", default)


def _cross_validate(
    registry: dict[str, SpeakerRecord],
    name_scores: dict[str, dict[str, float]],
) -> None:
    """If two speakers resolved to the same name, keep only the stronger one."""
    name_to_sids: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for sid, scores in name_scores.items():
        for name, score in scores.items():
            name_to_sids[name].append((sid, score))

    for name, sid_list in name_to_sids.items():
        if len(sid_list) <= 1:
            continue
        sid_list.sort(key=lambda x: -x[1])
        for sid, _ in sid_list[1:]:
            rec = registry[sid]
            rec.name = None
            rec.status = "unknown"


# Backward-compatible alias
update_registry = buffer_evidence


# ── Main pipeline ────────────────────────────────────────────────────────────

def identify_speakers(
    dialogues: list[str],
    enable_thinking: bool = False,
    confirm_threshold: int = 2,
    fuzzy_threshold: float = 0.80,
) -> dict[str, SpeakerRecord]:
    """
    Two-phase speaker identification robust to ASR/diarization noise.

    Phase 1: Process each dialogue, extract name cues, and buffer all
             evidence without committing to any name assignment.
    Phase 2: Resolve names via weighted voting across all accumulated
             evidence, with fuzzy matching to merge spelling variants
             and cross-validation to prevent duplicate assignments.
    """
    registry: dict[str, SpeakerRecord] = {}

    # Phase 1: buffer evidence from all dialogues
    for i, dialogue in enumerate(dialogues):
        dialogue_id  = f"dialogue_{i + 1}"
        user_prompt  = build_extraction_prompt(dialogue, registry)
        raw_response = qwen3_chat(
            system=EXTRACTION_SYSTEM_PROMPT,
            user=user_prompt,
            enable_thinking=enable_thinking,
            temperature=0.0,
            max_tokens=2048,
        )
        raw_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```[a-z]*\n?|```$", "", raw_response.strip(), flags=re.MULTILINE).strip()

        try:
            result      = json.loads(cleaned)
            extractions = result.get("extractions", [])
        except json.JSONDecodeError as e:
            print(f"[{dialogue_id}] JSON parse error: {e}\nRaw output:\n{raw_response}")
            extractions = []

        registry = buffer_evidence(registry, extractions, dialogue_id)

        print(f"\n[{dialogue_id}] Evidence buffered: "
              f"{sum(len(r.evidence) for r in registry.values())} total cues")

    # Phase 2: resolve names from all buffered evidence
    registry = resolve_registry(
        registry,
        confirm_threshold=confirm_threshold,
        fuzzy_threshold=fuzzy_threshold,
    )

    print("\n── Final resolved registry ──")
    for sid, rec in sorted(registry.items()):
        ev_summary = []
        for ev in rec.evidence:
            cand = getattr(ev, "_candidate_name", "?")
            ev_summary.append(f"{cand}({ev.confidence:.1f})")
        print(f"  {sid:12s} → {rec.display_name:20s} [{rec.status}]  evidence: {', '.join(ev_summary)}")

    return registry


def resolve_transcript(dialogue: str, registry: dict[str, SpeakerRecord]) -> str:
    """Substitute Speaker_X tags with resolved names in a transcript."""
    for sid, rec in registry.items():
        dialogue = dialogue.replace(f"<{sid}>", f"<{rec.display_name}>")
    return dialogue


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DIALOGUE_1 = """[Dialogue1 between multiple people on 2023-05-01]
<Speaker_0> So if a photon is directed through a plane with two slits in it...
<Speaker_1> Agreed, what's your point?
<Speaker_2> Can I help you?
<Speaker_1> Yes. Um, is this the High IQ sperm bank?
<Speaker_2> If you have to ask, maybe you shouldn't be here."""

    DIALOGUE_2 = """[Dialogue2 between multiple people on 2023-05-01]
<Speaker_0> Leonard, I don't think I can do this.
<Speaker_1> What, are you kidding? You're a semi-pro.
<Speaker_1> Sheldon, this was your idea.
<Speaker_0> I know, and I do yearn for faster downloads."""

    dialogues = [DIALOGUE_1, DIALOGUE_2]
    registry  = identify_speakers(dialogues)

    print("\n── Final resolved transcript (Dialogue 2) ──")
    print(resolve_transcript(DIALOGUE_2, registry))
