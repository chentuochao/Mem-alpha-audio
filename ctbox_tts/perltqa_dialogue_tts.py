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
import time
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
# Local copy of the voice-bank dataset (download it here to avoid streaming over
# HTTP, which can fail with "[Errno 9] Bad file descriptor"). See --help.
# DEFAULT_LOCAL_DATASET_DIR = HERE / "voices-libritts"
DEFAULT_LOCAL_DATASET_DIR = Path("/checkpoint/seamless/tuochao/data/ctbox_tts/voices-libritts/")
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


# Name folding: "liu rui" / "Liu Rui" / "liu_rui" / "LIU-RUI" -> one speaker.
# Populated by build_canonical_map(); maps a normalized key -> canonical spelling.
_CANON: Dict[str, str] = {}


def _name_key(name: str) -> str:
    """Case- and separator-insensitive key: spaces, '_' and '-' are equivalent."""
    return re.sub(r"[\s_\-]+", " ", name.strip().lower()).strip()


def canonicalize(name: str) -> str:
    """Fold a speaker name to its canonical spelling (case/separator-insensitive)."""
    if not name:
        return name
    return _CANON.get(_name_key(name), name.strip())


def build_canonical_map(data: dict) -> Tuple[Dict[str, str], int]:
    """
    Scan every speaker name (dialogue lines + profile protagonist + supporting
    characters) and pick one canonical spelling per case-insensitive key.
    Representative = the spelling that occurs most often; ties prefer a
    capitalized first letter, then the longer string, then alphabetical.

    Returns (canon_map, num_merged) where num_merged is the number of distinct
    spellings that were folded away (i.e. total variants - unique speakers).
    """
    variants: Counter = Counter()
    for _, pv in data.items():
        for _, dv in (pv.get("dialogues", {}) or {}).items():
            for _, lines in (dv.get("contents", {}) or {}).items():
                if not isinstance(lines, list):
                    continue
                for ln in lines:
                    name, _ = parse_line(ln)
                    if name:
                        variants[name.strip()] += 1
        prof = pv.get("profile", {}) or {}
        if isinstance(prof, dict) and prof.get("Protagonist"):
            variants[str(prof["Protagonist"]).strip()] += 1
        rels = pv.get("social_relationship", {}) or {}
        if isinstance(rels, dict):
            for rv in rels.values():
                if isinstance(rv, dict) and rv.get("Supporting Characters"):
                    variants[str(rv["Supporting Characters"]).strip()] += 1

    groups: Dict[str, List[Tuple[str, int]]] = {}
    for name, cnt in variants.items():
        groups.setdefault(_name_key(name), []).append((name, cnt))

    canon: Dict[str, str] = {}
    num_merged = 0
    for key, items in groups.items():
        # representative: most frequent; ties prefer no '_'/'-' separators,
        # then a capitalized first letter, then the longer string, then alpha.
        items.sort(key=lambda x: (-x[1],
                                  1 if re.search(r"[_\-]", x[0]) else 0,
                                  0 if x[0][:1].isupper() else 1,
                                  -len(x[0]), x[0]))
        canon[key] = items[0][0]
        num_merged += len(items) - 1  # extra spellings folded into the canon
    return canon, num_merged


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
    # Fold case variants of the same name to one canonical speaker.
    parsed = [(canonicalize(n) if n else None, t) for n, t in parsed]
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


def _profile_gender(pv: dict) -> str:
    """Owner (protagonist) gender from profile.Gender -> 'M'/'F'/'unknown'."""
    g = str((pv.get("profile", {}) or {}).get("Gender", "")).strip().lower()
    if g.startswith("m"):
        return "M"
    if g.startswith("f"):
        return "F"
    return "unknown"


def parse_dataset(data: dict) -> Tuple[List[dict], Counter, dict, Counter]:
    """
    Returns (profile_groups, speaker_counter, skip_stats, party_count_dist).

    profile_groups is a list grouped by profile owner, in dataset order:
        [{"profile": <owner name>,
          "gender":  "M" | "F" | "unknown",   # the owner's gender
          "blocks":  [DialogueBlock, ...]},    # that owner's usable blocks
         ...]
    Only profiles with at least one usable block are included.

    `speaker_counter` counts utterances per (canonical) speaker across ALL
    blocks so the "count all speakers" step is complete.
    `party_count_dist` maps #speakers -> #accepted blocks.
    """
    profile_groups: List[dict] = []
    speaker_counter: Counter = Counter()
    skip = Counter()
    party_dist: Counter = Counter()

    for profile, pv in data.items():
        group_blocks: List[DialogueBlock] = []
        dialogues = pv.get("dialogues", {}) or {}
        for dialogue_id, dv in dialogues.items():
            contents = dv.get("contents", {}) or {}
            for timestamp, lines in contents.items():
                if not isinstance(lines, list):
                    skip["non_list"] += 1
                    continue
                # Count every named speaker for the global tally (case-folded).
                for ln in lines:
                    name, _ = parse_line(ln)
                    if name:
                        speaker_counter[canonicalize(name)] += 1
                block = resolve_block(profile, dialogue_id, timestamp, lines)
                if block is None:
                    skip["unusable"] += 1
                    continue
                group_blocks.append(block)
                party_dist[len(block.parties)] += 1

        if group_blocks:
            profile_groups.append({
                "profile": profile,
                "gender": _profile_gender(pv),
                "blocks": group_blocks,
            })

    return profile_groups, speaker_counter, skip, party_dist


def build_gender_map(data: dict) -> Dict[str, str]:
    """name -> 'M'/'F' best-effort, from protagonist gender + relationships."""
    gender: Dict[str, str] = {}

    def set_if_absent(name, g):
        name = canonicalize(name) if name else name
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
    local_dataset_dir: Optional[Path] = None,
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

    print(f"[prepare] {len(pending)} / {len(speakers)} speakers need a voice.")

    # Prefer a locally-downloaded copy (reading parquet from disk avoids the
    # flaky streaming-over-HTTP path that throws "[Errno 9] Bad file descriptor").
    local_dir = Path(local_dataset_dir) if local_dataset_dir else None
    if local_dir and local_dir.exists():
        files = sorted(str(p) for p in local_dir.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"--local-dataset-dir {local_dir} has no .parquet files. "
                f"Download the dataset there first, e.g.:\n"
                f"    huggingface-cli download {hf_dataset} --repo-type dataset "
                f"--local-dir {local_dir} --include 'data/*.parquet'"
            )
        print(f"[prepare] loading {len(files)} local parquet shard(s) from "
              f"{local_dir}")
        ds = load_dataset("parquet", data_files=files, split="train",
                          streaming=True)
    else:
        print(f"[prepare] no local copy at {local_dir}; "
              f"streaming {hf_dataset} from the HuggingFace hub ...")
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
    profile_groups: List[dict],
    ref_map: Dict[str, str],
    output_dir: Path,
    limit: int,
    overwrite: bool,
    num_shards: int = 1,
    shard_index: int = 0,
) -> None:
    # Heavy import: loads the chatterbox model once (on the first visible CUDA
    # device, i.e. whatever CUDA_VISIBLE_DEVICES pins for this process).
    import chatterbox_dialogue_tts as cdt

    # Make every pulled reference voice visible to the generator.
    cdt.REFERENCE_VOICE_MAP.update(ref_map)

    # Flatten to a deterministic global list of (profile, block) so we can shard
    # it across independent workers. Round-robin (idx % num_shards) balances
    # variable-length blocks better than contiguous chunks. Blocks write to
    # disjoint per-block dirs, so shards never collide.
    all_pairs = [(g, b) for g in profile_groups for b in g["blocks"]]
    if num_shards > 1:
        shard_pairs = [p for i, p in enumerate(all_pairs)
                       if i % num_shards == shard_index]
    else:
        shard_pairs = all_pairs

    budget = len(shard_pairs) if limit <= 0 else min(limit, len(shard_pairs))
    print(f"[generate][shard {shard_index}/{num_shards}] synthesising up to "
          f"{budget} / {len(shard_pairs)} dialogue blocks "
          f"(of {len(all_pairs)} total) -> {output_dir}")

    ok, skipped, failed, seen = 0, 0, 0, 0
    for group, block in shard_pairs:
        if seen >= budget:
            break
        seen += 1
        profile = group["profile"]
        profile_dir = output_dir / safe_filename(profile)

        out = profile_dir / safe_filename(block.dialogue_id)
        # the multispeaker function writes channel_map.json; use it as marker
        done_marker = out / "channel_map.json"
        if done_marker.exists() and not overwrite:
            skipped += 1
            continue

        # every party must have a reference voice
        missing = [p for p in block.parties
                   if p not in cdt.REFERENCE_VOICE_MAP]
        if missing:
            print(f"[generate][skip] {profile}/{block.dialogue_id}: "
                  f"no reference voice for {missing}", file=sys.stderr)
            skipped += 1
            continue

        # ordered list of {speaker_name: text} turns
        speaker_texts = [{name: text} for name, text in block.turns]
        print(f"[generate][shard {shard_index}] ({seen}/{budget}) START "
              f"{profile}/{block.dialogue_id}: "
              f"{len(block.parties)} speakers "
              f"[{', '.join(block.parties)}] ({len(block.turns)} turns)",
              flush=True)
        t0 = time.time()
        try:
            cdt.generate_multispeaker_dialogue_tts(
                speaker_names=block.parties,
                speaker_texts=speaker_texts,
                output_dir=str(out),
            )
            ok += 1
            elapsed = time.time() - t0
            print(f"[generate][shard {shard_index}] ({seen}/{budget}) DONE  "
                  f"{profile}/{block.dialogue_id}: "
                  f"{len(block.turns)} turns in {elapsed:.1f}s "
                  f"({elapsed / max(1, len(block.turns)):.1f}s/turn)",
                  flush=True)
        except Exception as e:  # keep going on a single bad block
            failed += 1
            elapsed = time.time() - t0
            print(f"[generate][FAIL] {profile}/{block.dialogue_id} "
                  f"after {elapsed:.1f}s: {e}", file=sys.stderr, flush=True)
    print(f"[generate][shard {shard_index}/{num_shards}] done. "
          f"ok={ok} skipped={skipped} failed={failed}")


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
    ap.add_argument(
        "--local-dataset-dir", type=Path, default=DEFAULT_LOCAL_DATASET_DIR,
        help="local dir holding the downloaded voice-bank parquet files. "
             "If it exists, voices are read from disk instead of streamed. "
             "Download with: huggingface-cli download "
             "sdialog/voices-libritts --repo-type dataset "
             "--local-dir <dir> --include 'data/*.parquet'")
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
    ap.add_argument("--num-shards", type=int, default=1,
                    help="total number of parallel worker shards (e.g. #GPUs)")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="this worker's shard index in [0, num-shards)")
    args = ap.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        ap.error(f"--shard-index must be in [0, {args.num_shards}); "
                 f"got {args.shard_index}")

    ref_map_path = args.ref_map or (args.ref_dir / "reference_voice_map.json")
    args.data = Path("/checkpoint/seamless/tuochao/data/PerLTQA/Dataset/en_v2/perltmem_en_v2.json")
    args.output_dir = Path("/checkpoint/seamless/tuochao/data/PerLTQA/dialogue_tts_en_v2")
    # 1) parse + count -----------------------------------------------------
    with args.data.open(encoding="utf-8") as f:
        data = json.load(f)

    # Build the case-insensitive name map first; both parse_dataset and
    # build_gender_map rely on canonicalize().
    global _CANON
    _CANON, num_merged = build_canonical_map(data)
    if num_merged:
        print(f"[count] case-folded {num_merged} case-variant name(s) "
              f"into their canonical speakers")

    profile_groups, speaker_counter, skip, party_dist = parse_dataset(data)
    gender_map = build_gender_map(data)
    speakers = sorted(speaker_counter.keys())
    total_blocks = sum(len(g["blocks"]) for g in profile_groups)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_dir / "speaker_stats.json"
    stats = {
        "num_unique_speakers": len(speakers),
        "num_profiles": len(profile_groups),
        "num_dialogue_blocks": total_blocks,
        "party_count_distribution": {str(k): v for k, v in sorted(party_dist.items())},
        "skipped_blocks": dict(skip),
        "profiles": [
            {
                "profile": g["profile"],
                "gender": g["gender"],
                "num_blocks": len(g["blocks"]),
            }
            for g in profile_groups
        ],
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
    print(f"[count] profiles: {len(profile_groups)}")
    print(f"[count] dialogue blocks: {total_blocks} "
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
            local_dataset_dir=args.local_dataset_dir,
        )

    ref_map = payload.get("reference_voice_map", {})
    if args.prepare_only:
        print(f"[prepare-only] {len(ref_map)} speakers have reference voices.")
        return

    # 3) generate ----------------------------------------------------------
    generate_all(
        profile_groups=profile_groups,
        ref_map=ref_map,
        output_dir=args.output_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )


if __name__ == "__main__":
    main()
