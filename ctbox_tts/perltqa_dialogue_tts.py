#!/usr/bin/env python3
"""
Generate two-party dialogue TTS for the PerLTQA dataset.

Pipeline
--------
1. Parse `perltmem_en_v2.json` -> walk every profile's `dialogues` ->
   each `contents` block is a list of "Speaker Name: text" lines.
2. Count all unique speaker names (written to a stats JSON).
3. Assign every unique speaker a *unique* reference audio by streaming the
   HuggingFace `sdialog/voices-libritts` voice bank (best-effort gender match),
   caching one distinct wav per speaker under --ref-dir, and saving a
   name -> wav reference map JSON.
4. For each (mostly two-party) dialogue block, synthesise a stereo dialogue
   with `chatterbox_dialogue_tts.generate_dialogue_tts`, writing the result to
       <output-dir>/<ProfileOwner>/<dialogue_id>/
   (output is organised per profile owner / top-level dataset key).

The heavy bits (chatterbox model, datasets/librosa) are imported lazily so that
`--count-only` and `--prepare-only` do not pull in modules they don't need.

Examples
--------
    # Just count speakers and dump stats
    python perltqa_dialogue_tts.py --count-only

    # Build the reference-voice bank only (no TTS)
    python perltqa_dialogue_tts.py --prepare-only

    # Full flow but only synthesise the first 5 dialogue blocks (test subset)
    python perltqa_dialogue_tts.py --limit 5

    # Generate everything (heavy!)
    python perltqa_dialogue_tts.py --limit 0
"""

import argparse
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
# DEFAULT_DATA = HERE.parent / "perlTQA" / "perltmem_en_v2.json"
# DEFAULT_OUTPUT_DIR = HERE / "tts_outputs" / "perltqa"
DEFAULT_REF_DIR = HERE / "ref_voices" / "perltqa"
DEFAULT_HF_DATASET = "sdialog/voices-libritts"
DEFAULT_TARGET_SR = 24000
DEFAULT_REF_DURATION_SEC = 10.0

# "Speaker Name: the rest of the utterance"
LINE_RE = re.compile(r"^\s*([^:]+?)\s*:\s*(.*)$", re.DOTALL)

# Relationship word -> gender hint
FEMALE_REL = {
    "sister", "elder sister", "younger sister", "girlfriend", "wife",
    "mother", "mom", "daughter", "aunt", "grandmother", "niece",
    "mother-in-law", "wife's", "fiancee",
}
MALE_REL = {
    "brother", "elder brother", "younger brother", "boyfriend", "husband",
    "father", "dad", "son", "uncle", "grandfather", "nephew",
    "father-in-law", "fiance",
}


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return name.strip("_") or "unnamed"


def guess_gender_from_text(text: str) -> Optional[str]:
    """Best-effort gender from pronoun frequency in a free-text description."""
    if not text:
        return None
    t = " " + text.lower() + " "
    female = len(re.findall(r"\b(she|her|hers|herself)\b", t))
    male = len(re.findall(r"\b(he|him|his|himself)\b", t))
    if female > male:
        return "F"
    if male > female:
        return "M"
    return None


def guess_gender_from_relationship(rel: str) -> Optional[str]:
    if not rel:
        return None
    r = rel.strip().lower()
    if r in FEMALE_REL:
        return "F"
    if r in MALE_REL:
        return "M"
    for w in r.split():
        if w in FEMALE_REL:
            return "F"
        if w in MALE_REL:
            return "M"
    return None


# ----------------------------------------------------------------------------
# Dataset parsing
# ----------------------------------------------------------------------------
class DialogueBlock:
    """One `contents` timestamped block: an ordered multi-party conversation."""

    __slots__ = ("profile", "dialogue_id", "timestamp", "turns", "parties")

    def __init__(self, profile, dialogue_id, timestamp, turns, parties):
        self.profile = profile
        self.dialogue_id = dialogue_id
        self.timestamp = timestamp
        self.turns = turns                      # list[(speaker, text)] in order
        self.parties = parties                  # distinct speakers, first-seen order


def parse_line(line: str) -> Tuple[Optional[str], str]:
    """Return (speaker_or_None, text). Malformed lines have speaker=None."""
    m = LINE_RE.match(line)
    if not m:
        return None, line.strip()
    name = m.group(1).strip()
    text = m.group(2).strip()
    # Guard against absurdly long "names" (a colon mid-sentence, not a speaker).
    if not name or len(name) > 40 or "." in name and len(name.split()) > 4:
        return None, line.strip()
    return name, text


def resolve_block(profile: str, dialogue_id: str, timestamp: str,
                  lines: List[str]) -> Optional[DialogueBlock]:
    """
    Turn a raw list of lines into an ordered multi-party block with >= 2
    speakers. Name-less ("...: text") lines are only recoverable in the
    two-party case (via strict alternation); a block that has name-less lines
    but is not exactly two-party can't attribute them and is skipped.
    Returns None for blocks that can't be turned into a clean >=2-speaker
    conversation.
    """
    parsed = [parse_line(ln) for ln in lines if ln and ln.strip()]
    if not parsed:
        return None

    # Distinct named speakers in order of first appearance.
    order: List[str] = []
    for name, _ in parsed:
        if name and name not in order:
            order.append(name)

    has_nameless = any(name is None for name, _ in parsed)

    if has_nameless:
        # Can only repair name-less lines when exactly two parties are named.
        if len(order) != 2:
            return None
        party1, party2 = order[0], order[1]
        turns: List[Tuple[str, str]] = []
        last = None
        for name, text in parsed:
            if name is None:
                name = party2 if last == party1 else party1
            turns.append((name, text))
            last = name
        return DialogueBlock(profile, dialogue_id, timestamp, turns, order)

    # Fully-named: support any number of speakers, but require a real dialogue.
    if len(order) < 2:
        return None  # monologue / single speaker -> skip
    turns = [(name, text) for name, text in parsed]
    return DialogueBlock(profile, dialogue_id, timestamp, turns, order)


def parse_dataset(data: dict) -> Tuple[List[DialogueBlock], Counter, dict, Counter]:
    """
    Returns (blocks, speaker_counter, skip_stats, party_count_dist).
    `speaker_counter` counts utterances per speaker across ALL blocks
    (including skipped ones) so the "count all speakers" step is complete.
    `party_count_dist` maps #speakers -> #accepted blocks.
    """
    blocks: List[DialogueBlock] = []
    speaker_counter: Counter = Counter()
    skip = Counter()
    party_dist: Counter = Counter()

    for profile, pv in data.items():
        dialogues = pv.get("dialogues", {}) or {}
        for dialogue_id, dv in dialogues.items():
            contents = dv.get("contents", {}) or {}
            for timestamp, lines in contents.items():
                if not isinstance(lines, list):
                    skip["non_list"] += 1
                    continue
                # Count every named speaker for the global tally.
                for ln in lines:
                    name, _ = parse_line(ln)
                    if name:
                        speaker_counter[name] += 1
                block = resolve_block(profile, dialogue_id, timestamp, lines)
                if block is None:
                    skip["unusable"] += 1
                    continue
                blocks.append(block)
                party_dist[len(block.parties)] += 1
    return blocks, speaker_counter, skip, party_dist


def build_gender_map(data: dict) -> Dict[str, str]:
    """name -> 'M'/'F' best-effort, from protagonist gender + relationships."""
    gender: Dict[str, str] = {}

    def set_if_absent(name, g):
        if name and g and name not in gender:
            gender[name] = g

    for _, pv in data.items():
        prof = pv.get("profile", {}) or {}
        protagonist = prof.get("Protagonist")
        g = (prof.get("Gender") or "").strip().lower()
        if protagonist:
            if g.startswith("m"):
                set_if_absent(protagonist, "M")
            elif g.startswith("f"):
                set_if_absent(protagonist, "F")

        rels = pv.get("social_relationship", {}) or {}
        if not isinstance(rels, dict):
            continue
        for _, rv in rels.items():
            if not isinstance(rv, dict):
                continue
            name = rv.get("Supporting Characters")
            if not name:
                continue
            g2 = (guess_gender_from_relationship(rv.get("Relationship", ""))
                  or guess_gender_from_text(rv.get("Description", "")))
            set_if_absent(name, g2)

    return gender


# ----------------------------------------------------------------------------
# Reference voice bank
# ----------------------------------------------------------------------------
def load_ref_map(ref_map_path: Path) -> dict:
    if ref_map_path.exists():
        with ref_map_path.open(encoding="utf-8") as f:
            return json.load(f)
    return {"reference_voice_map": {}, "selected_metadata": []}


def save_ref_map(ref_map_path: Path, payload: dict) -> None:
    ref_map_path.parent.mkdir(parents=True, exist_ok=True)
    with ref_map_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def prepare_reference_voices(
    speakers: List[str],
    gender_map: Dict[str, str],
    ref_dir: Path,
    ref_map_path: Path,
    hf_dataset: str,
    target_sr: int,
    ref_duration: float,
) -> dict:
    """
    Ensure every speaker in `speakers` has a unique cached reference wav.
    Resumable: speakers already present (with an existing wav) are skipped.
    Unique folder per speaker:  <ref-dir>/<safe_name>/reference.wav
    """
    import io
    import librosa
    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    ref_dir.mkdir(parents=True, exist_ok=True)
    payload = load_ref_map(ref_map_path)
    ref_map: dict = payload.setdefault("reference_voice_map", {})
    metadata: list = payload.setdefault("selected_metadata", [])
    used_identifiers = {m.get("libritts_identifier") for m in metadata}

    # What still needs a voice?
    pending = []
    for name in speakers:
        existing = ref_map.get(name)
        if existing and Path(existing).exists():
            continue
        # round-robin fallback for unknown gender so we don't drain one pool
        g = gender_map.get(name)
        if g not in ("M", "F"):
            g = "M" if (len(pending) % 2 == 0) else "F"
        pending.append({"name": name, "gender": g})

    if not pending:
        print(f"[prepare] all {len(speakers)} speakers already have a "
              f"reference voice. Nothing to pull.")
        save_ref_map(ref_map_path, payload)
        return payload

    print(f"[prepare] {len(pending)} / {len(speakers)} speakers need a voice. "
          f"Streaming {hf_dataset} ...")

    ds = load_dataset(hf_dataset, split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    target_len = int(target_sr * ref_duration)
    remaining = list(pending)
    done = 0

    for ex in ds:
        if not remaining:
            break
        gender = ex.get("gender")
        identifier = str(ex.get("identifier"))
        if identifier in used_identifiers:
            continue

        # first pending speaker that needs this gender
        match_idx = next((i for i, it in enumerate(remaining)
                          if it["gender"] == gender), None)
        if match_idx is None:
            continue

        audio_obj = ex.get("audio") or {}
        if audio_obj.get("bytes") is None:
            continue
        with io.BytesIO(audio_obj["bytes"]) as fh:
            wav, sr = sf.read(fh, dtype="float32")
        if getattr(wav, "ndim", 1) == 2:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        # trim / pad to a clean fixed length
        if len(wav) >= target_len:
            wav = wav[:target_len]
        else:
            padded = np.zeros(target_len, dtype="float32")
            padded[: len(wav)] = wav
            wav = padded
        max_abs = float(np.max(np.abs(wav))) if len(wav) else 0.0
        if max_abs > 1e-8:
            wav = (wav / max_abs * 0.95).astype("float32")

        item = remaining.pop(match_idx)
        name = item["name"]
        spk_dir = ref_dir / safe_filename(name)
        spk_dir.mkdir(parents=True, exist_ok=True)
        out_path = spk_dir / "reference.wav"
        sf.write(str(out_path), wav, sr)

        used_identifiers.add(identifier)
        ref_map[name] = str(out_path)
        metadata.append({
            "speaker_name": name,
            "assigned_gender": gender,
            "reference_audio": str(out_path),
            "libritts_identifier": identifier,
            "libritts_speaker_name": ex.get("name"),
            "libritts_subset": ex.get("subset"),
            "saved_duration_s": ref_duration,
            "sample_rate": sr,
        })
        done += 1
        if done % 25 == 0 or not remaining:
            print(f"[prepare]   {done}/{len(pending)} assigned "
                  f"(last: {name} <- {gender}/{identifier})")
            save_ref_map(ref_map_path, payload)  # periodic checkpoint

    save_ref_map(ref_map_path, payload)

    if remaining:
        missing = [it["name"] for it in remaining]
        print(f"[prepare][WARN] voice bank exhausted before matching "
              f"{len(missing)} speakers (gender mismatch / not enough clips): "
              f"{missing[:10]}{' ...' if len(missing) > 10 else ''}",
              file=sys.stderr)

    print(f"[prepare] done. reference map: {ref_map_path}")
    return payload


# ----------------------------------------------------------------------------
# TTS generation
# ----------------------------------------------------------------------------
def generate_all(
    blocks: List[DialogueBlock],
    ref_map: Dict[str, str],
    output_dir: Path,
    limit: int,
    overwrite: bool,
) -> None:
    # Heavy import: loads the chatterbox model once.
    import chatterbox_dialogue_tts as cdt

    # Make every pulled reference voice visible to the generator.
    cdt.REFERENCE_VOICE_MAP.update(ref_map)

    selected = blocks if limit <= 0 else blocks[:limit]
    print(f"[generate] synthesising {len(selected)} / {len(blocks)} "
          f"dialogue blocks -> {output_dir}")

    ok, skipped, failed = 0, 0, 0
    for idx, block in enumerate(selected):
        out = (output_dir / safe_filename(block.profile)
               / safe_filename(block.dialogue_id))
        # the multispeaker function writes channel_map.json; use it as the marker
        done_marker = out / "channel_map.json"
        if done_marker.exists() and not overwrite:
            skipped += 1
            continue

        # every party must have a reference voice
        missing = [p for p in block.parties
                   if p not in cdt.REFERENCE_VOICE_MAP]
        if missing:
            print(f"[generate][skip] {block.profile}/{block.dialogue_id}: "
                  f"no reference voice for {missing}", file=sys.stderr)
            skipped += 1
            continue

        # ordered list of {speaker_name: text} turns
        speaker_texts = [{name: text} for name, text in block.turns]
        try:
            cdt.generate_multispeaker_dialogue_tts(
                speaker_names=block.parties,
                speaker_texts=speaker_texts,
                output_dir=str(out),
            )
            ok += 1
            print(f"[generate] ({idx + 1}/{len(selected)}) "
                  f"{block.profile}/{block.dialogue_id}: "
                  f"{len(block.parties)} speakers "
                  f"[{', '.join(block.parties)}] "
                  f"({len(block.turns)} turns)")
        except Exception as e:  # keep going on a single bad block
            failed += 1
            print(f"[generate][FAIL] {block.profile}/{block.dialogue_id}: {e}",
                  file=sys.stderr)

    print(f"[generate] done. ok={ok} skipped={skipped} failed={failed}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    # ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--ref-dir", type=Path, default=DEFAULT_REF_DIR)
    ap.add_argument("--ref-map", type=Path, default=None,
                    help="reference map JSON (default: <ref-dir>/reference_voice_map.json)")
    ap.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    ap.add_argument("--target-sr", type=int, default=DEFAULT_TARGET_SR)
    ap.add_argument("--ref-duration", type=float, default=DEFAULT_REF_DURATION_SEC)
    ap.add_argument("--limit", type=int, default=5,
                    help="number of dialogue blocks to synthesise; <=0 means all")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-generate blocks whose output already exists")
    ap.add_argument("--count-only", action="store_true",
                    help="only count speakers + write stats, then exit")
    ap.add_argument("--prepare-only", action="store_true",
                    help="count + build reference bank, then exit (no TTS)")
    ap.add_argument("--skip-prepare", action="store_true",
                    help="skip voice-bank pulling (use existing reference map)")
    args = ap.parse_args()

    ref_map_path = args.ref_map or (args.ref_dir / "reference_voice_map.json")
    args.data = Path("/checkpoint/seamless/tuochao/data/PerLTQA/Dataset/en_v2/perltmem_en_v2.json")
    args.output_dir = Path("/checkpoint/seamless/tuochao/data/PerLTQA/audio")
    # 1) parse + count -----------------------------------------------------
    with args.data.open(encoding="utf-8") as f:
        data = json.load(f)

    blocks, speaker_counter, skip, party_dist = parse_dataset(data)
    gender_map = build_gender_map(data)
    speakers = sorted(speaker_counter.keys())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_dir / "speaker_stats.json"
    stats = {
        "num_unique_speakers": len(speakers),
        "num_dialogue_blocks": len(blocks),
        "party_count_distribution": {str(k): v for k, v in sorted(party_dist.items())},
        "skipped_blocks": dict(skip),
        "speakers": [
            {
                "name": name,
                "utterances": speaker_counter[name],
                "gender": gender_map.get(name, "unknown"),
            }
            for name, _ in speaker_counter.most_common()
        ],
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[count] unique speakers: {len(speakers)}")
    print(f"[count] dialogue blocks: {len(blocks)} "
          f"(by #speakers: {dict(sorted(party_dist.items()))})")
    print(f"[count] skipped blocks: {dict(skip)}")
    print(f"[count] gender resolved for {sum(1 for s in speakers if s in gender_map)} "
          f"/ {len(speakers)} speakers")
    print(f"[count] stats written to {stats_path}")

    if args.count_only:
        return

    # 2) reference voices --------------------------------------------------
    if args.skip_prepare:
        payload = load_ref_map(ref_map_path)
    else:
        payload = prepare_reference_voices(
            speakers=speakers,
            gender_map=gender_map,
            ref_dir=args.ref_dir,
            ref_map_path=ref_map_path,
            hf_dataset=args.hf_dataset,
            target_sr=args.target_sr,
            ref_duration=args.ref_duration,
        )

    ref_map = payload.get("reference_voice_map", {})
    if args.prepare_only:
        print(f"[prepare-only] {len(ref_map)} speakers have reference voices.")
        return

    # 3) generate ----------------------------------------------------------
    generate_all(
        blocks=blocks,
        ref_map=ref_map,
        output_dir=args.output_dir,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
