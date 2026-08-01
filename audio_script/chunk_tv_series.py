"""
Process TV-series word-level annotation txt files into turn-level conversations
using AlignedProcess_Morespeakers.

Input format (one word per line, space-separated columns):
    file_id  speaker  start  end  word  score  listener  scene_context  [extra...]

Output: for each input .txt file, writes a .json file with the parsed turns.

Usage:
    python process_tv_series.py                          # process all .txt in ./tv_series
    python process_tv_series.py --input_dir ./my_dir     # custom input directory
    python process_tv_series.py --output_dir ./out       # custom output directory
"""

import json
import sys
import os
import argparse
from collections import defaultdict
from typing import Dict, List
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_script.datasets.turn_annotation import AlignedProcess_Morespeakers

TURN_GAP_TH = 1.5

# Chunk duration constraints (seconds).
CHUNK_MIN_DURATION = 60    # 1 minute
CHUNK_MAX_DURATION = 300   # 10 minutes
# Silence gap threshold: gaps larger than this are preferred split points.
CHUNK_GAP_THRESHOLD = 3.0


# ---------------------------------------------------------------------------
# Step 1: Parse the txt file into per-speaker word lists
# ---------------------------------------------------------------------------

def parse_txt(txt_path: str) -> Dict[str, List[Dict]]:
    """Parse a word-level annotation txt file into per-speaker word dicts.

    Each line has space-separated columns:
        file_id  speaker  start  end  word  score  listener  scene_context  [extra...]

    Args:
        txt_path: path to the .txt annotation file.

    Returns:
        dict mapping speaker name -> list of word dicts, sorted by start time.
        Each word dict has keys: word, start, end, score, listener, scene_context.
    """
    speaker_words: Dict[str, List[Dict]] = defaultdict(list)
    raw_transcript = []

    with open(txt_path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 6:
                print(f"Warning: Skipping malformed line {line_num} in {txt_path}: {line}")
                continue

            # Column layout: file_id(0) speaker(1) start(2) end(3) word(4) score(5)
            #                listener(6) scene_context(7) [extra...]
            speaker = parts[1]
            try:
                start = float(parts[2])
                end = float(parts[3])
            except ValueError:
                print(f"Warning: Bad timestamps at line {line_num}, skipping: {line}")
                continue

            word = parts[4]
            try:
                score = float(parts[5])
            except ValueError:
                score = 1.0

            listener = parts[6] if len(parts) > 6 else "_"
            scene_context = parts[7] if len(parts) > 7 else "_"

            speaker_words[speaker].append({
                "word": word,
                "start": start,
                "end": end,
                "score": score,
                "listener": listener,
                "scene_context": scene_context,
            })

            raw_transcript.append({
                "speaker": speaker,
                "word": word,
                "start": start,
                "end": end,
                "score": score,
                "listener": listener,
                "scene_context": scene_context,
            })

    # Sort each speaker's words chronologically.
    for spk in speaker_words:
        speaker_words[spk].sort(key=lambda w: w["start"])

    return dict(speaker_words), raw_transcript


# ---------------------------------------------------------------------------
# Step 2: Group flat word lists into segments (required by AlignedProcess)
# ---------------------------------------------------------------------------

def words_to_segments(word_list: List[Dict], turn_gap: float = 1.5) -> List[Dict]:
    """Group a sorted word list into utterance segments split by silence gaps.

    AlignedProcess_Morespeakers expects each speaker's transcript as a list
    of segment dicts, where each segment has a 'words' key containing the
    individual word dicts.  This function creates those segments by splitting
    wherever the gap between consecutive words exceeds turn_gap seconds.

    Args:
        word_list: sorted list of word dicts for one speaker.
        turn_gap: silence threshold (seconds) to split segments.

    Returns:
        list of segment dicts: [{start, end, text, words}, ...].
    """
    if not word_list:
        return []

    segments = []
    current_words = [word_list[0]]

    for w in word_list[1:]:
        gap = w["start"] - current_words[-1]["end"]
        if gap > turn_gap:
            segments.append(_build_segment(current_words))
            current_words = [w]
        else:
            current_words.append(w)

    if current_words:
        segments.append(_build_segment(current_words))

    return segments


def _build_segment(words: List[Dict]) -> Dict:
    """Build a single segment dict from a list of consecutive word dicts."""
    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": " ".join(w["word"] for w in words),
        "words": words,
    }


# ---------------------------------------------------------------------------
# Step 3: Run AlignedProcess_Morespeakers and collect results
# ---------------------------------------------------------------------------

def process_file(txt_path: str) -> List[Dict]:
    """Parse a single txt file and return turn-level conversation as a list.

    Pipeline:
        1. parse_txt        -> {speaker: [word_dicts]}
        2. words_to_segments -> {speaker: [segment_dicts]}  (per-speaker)
        3. AlignedProcess_Morespeakers -> turn-level dialog with types

    Args:
        txt_path: path to the .txt annotation file.

    Returns:
        list of turn dicts sorted by time, each with keys:
            dialog_type, speaker, start, end, text, wfeats.
    """
    # Step 1: Parse txt -> per-speaker word lists.
    speaker_words = parse_txt(txt_path)

    if len(speaker_words) == 0:
        print(f"  No speakers found in {txt_path}, skipping.")
        return []

    # Step 2: Group words into segments per speaker.
    valid_speakers = []
    transcripts_list = []
    for speaker, words in speaker_words.items():
        if len(words) == 0:
            continue
        segments = words_to_segments(words, turn_gap=TURN_GAP_TH)
        valid_speakers.append(speaker)
        transcripts_list.append(segments)

    if len(valid_speakers) == 0:
        print(f"  All speakers empty in {txt_path}, skipping.")
        return []

    if len(valid_speakers) == 1:
        # Single speaker: just return one turn per segment.
        turns = []
        for seg in transcripts_list[0]:
            turns.append({
                "dialog_type": "dialog",
                "speaker": valid_speakers[0],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            })
        return turns

    # Step 3: Run AlignedProcess_Morespeakers for 2+ speakers.
    aligned_process = AlignedProcess_Morespeakers(
        transcripts=transcripts_list,
        speaker_names=valid_speakers,
        interval_character=' ',
        turn_gap_threshold=TURN_GAP_TH,
    )

    turns = aligned_process.get_parsed_dialog_combined()

    # Strip 'wfeats' from output to keep the JSON clean (optional).
    for turn in turns:
        turn.pop("wfeats", None)

    return turns


# ---------------------------------------------------------------------------
# Step 4: Split words into time-based chunks
# ---------------------------------------------------------------------------

def find_split_points(all_words: List[Dict], gap_threshold: float = CHUNK_GAP_THRESHOLD):
    """Find candidate split points where there are large silence gaps.

    Scans the globally-sorted word list and records every position where
    the gap between consecutive words exceeds gap_threshold.

    Args:
        all_words: list of word dicts sorted by start time (all speakers mixed).
        gap_threshold: minimum silence gap (seconds) to be a candidate split.

    Returns:
        list of (index, gap_size) tuples.  index is the position *after* which
        the split would occur (i.e., chunk boundary falls between
        all_words[index] and all_words[index+1]).
    """
    splits = []
    for i in range(1, len(all_words)):
        gap = all_words[i]["start"] - all_words[i - 1]["end"]
        if gap >= gap_threshold:
            splits.append((i, gap))
    return splits


def chunk_words(
    all_words: List[Dict],
    min_dur: float = CHUNK_MIN_DURATION,
    max_dur: float = CHUNK_MAX_DURATION,
    gap_threshold: float = CHUNK_GAP_THRESHOLD,
) -> List[List[Dict]]:
    """Split a sorted word list into chunks respecting duration constraints.

    Algorithm:
        1. Accumulate words into the current chunk.
        2. At each candidate split point (gap >= gap_threshold):
           - If the current chunk duration >= min_dur, finalize it.
        3. If the current chunk exceeds max_dur without a gap split point,
           force-split at the largest gap seen so far within the chunk.
        4. At the end, if the last chunk is too short (< min_dur), merge it
           into the previous chunk.

    Args:
        all_words: globally sorted word list (all speakers).
        min_dur: minimum chunk duration in seconds.
        max_dur: maximum chunk duration in seconds.
        gap_threshold: silence gap (seconds) that marks a natural boundary.

    Returns:
        list of chunks, where each chunk is a list of word dicts.
    """
    if not all_words:
        return []

    chunks = []
    chunk_start_idx = 0

    # Track the best (largest) gap within the current chunk for forced splits.
    best_gap_idx = None
    best_gap_size = -1.0

    for i in range(1, len(all_words)):
        gap = all_words[i]["start"] - all_words[i - 1]["end"]
        chunk_duration = all_words[i - 1]["end"] - all_words[chunk_start_idx]["start"]

        # Track the largest gap inside the current chunk.
        if gap > best_gap_size:
            best_gap_size = gap
            best_gap_idx = i

        # Natural split: large gap AND chunk is long enough.
        if gap >= gap_threshold and chunk_duration >= min_dur:
            chunks.append(all_words[chunk_start_idx:i])
            chunk_start_idx = i
            best_gap_idx = None
            best_gap_size = -1.0
            continue

        # Forced split: chunk exceeds max duration.
        if chunk_duration >= max_dur:
            if best_gap_idx is not None and best_gap_idx > chunk_start_idx:
                # Split at the largest gap we've seen in this chunk.
                chunks.append(all_words[chunk_start_idx:best_gap_idx])
                chunk_start_idx = best_gap_idx
            else:
                # No good gap found — hard-cut at current position.
                chunks.append(all_words[chunk_start_idx:i])
                chunk_start_idx = i
            best_gap_idx = None
            best_gap_size = -1.0

    # Flush remaining words as the last chunk.
    if chunk_start_idx < len(all_words):
        last_chunk = all_words[chunk_start_idx:]
        last_dur = last_chunk[-1]["end"] - last_chunk[0]["start"]

        # If the last chunk is too short, merge it into the previous one.
        if last_dur < min_dur and len(chunks) > 0:
            chunks[-1].extend(last_chunk)
        else:
            chunks.append(last_chunk)

    return chunks


def chunk_file(txt_path: str, min_dur=CHUNK_MIN_DURATION, max_dur=CHUNK_MAX_DURATION,
               gap_threshold=CHUNK_GAP_THRESHOLD) -> List[Dict]:
    """Parse a txt file and split it into time-based chunks.

    Each chunk is returned as a dict with metadata and per-speaker word lists
    ready for AlignedProcess_Morespeakers.

    Args:
        txt_path: path to the annotation txt file.
        min_dur / max_dur: chunk duration bounds in seconds.
        gap_threshold: silence gap threshold for natural splits.

    Returns:
        list of chunk dicts, each with keys:
            chunk_id, start, end, duration, speakers,
            speaker_words (per-speaker word lists),
            transcripts (per-speaker segment lists for AlignedProcess).
    """
    # Parse all words, sorted globally by time.
    speaker_words, all_words = parse_txt(txt_path)
    if not all_words:
        return []

    # Split into chunks.
    word_chunks = chunk_words(all_words, min_dur, max_dur, gap_threshold)

    # Build per-speaker structures for each chunk.
    result = []
    for idx, chunk_words_list in enumerate(word_chunks):
        # Group words by speaker within this chunk.
        per_speaker: Dict[str, List[Dict]] = defaultdict(list)
        for w in chunk_words_list:
            per_speaker[w["speaker"]].append(w)

        # Build segment lists per speaker (for AlignedProcess_Morespeakers).
        speakers = []
        transcripts = []
        for spk in sorted(per_speaker.keys()):
            words = per_speaker[spk]
            segs = words_to_segments(words, turn_gap=TURN_GAP_TH)
            speakers.append(spk)
            transcripts.append(segs)

        chunk_start = chunk_words_list[0]["start"]
        chunk_end = chunk_words_list[-1]["end"]

        result.append({
            "chunk_id": idx,
            "start": chunk_start,
            "end": chunk_end,
            "duration": chunk_end - chunk_start,
            "num_words": len(chunk_words_list),
            "speakers": speakers,
            "transcripts": transcripts,
        })

    return result


def process_file_chunked(txt_path: str, min_dur=CHUNK_MIN_DURATION,
                         max_dur=CHUNK_MAX_DURATION) -> List[Dict]:
    """Process a txt file into chunked, turn-level conversations.

    Pipeline:
        1. Parse txt and split into time-based chunks.
        2. For each chunk, run AlignedProcess_Morespeakers.
        3. Collect all chunks with their turn-level output.

    Args:
        txt_path: path to the .txt annotation file.
        min_dur / max_dur: chunk duration bounds in seconds.

    Returns:
        list of chunk result dicts, each containing chunk metadata and
        a 'turns' key with the parsed dialog turns.
    """
    chunks = chunk_file(txt_path, min_dur, max_dur)
    results = []

    for chunk in chunks:
        chunk_result = {
            "chunk_id": chunk["chunk_id"],
            "start": chunk["start"],
            "end": chunk["end"],
            "duration": chunk["duration"],
            "num_words": chunk["num_words"],
            "speakers": chunk["speakers"],
        }

        if len(chunk["speakers"]) == 0:
            chunk_result["turns"] = []

        elif len(chunk["speakers"]) == 1:
            # Single speaker: return one turn per segment.
            turns = []
            for seg in chunk["transcripts"][0]:
                turns.append({
                    "dialog_type": "dialog",
                    "speaker": chunk["speakers"][0],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                })
            chunk_result["turns"] = turns

        else:
            # Multi-speaker: run AlignedProcess_Morespeakers.
            try:
                proc = AlignedProcess_Morespeakers(
                    transcripts=chunk["transcripts"],
                    speaker_names=chunk["speakers"],
                    interval_character=' ',
                    turn_gap_threshold=TURN_GAP_TH,
                )
                turns = proc.get_parsed_dialog_combined()
                for t in turns:
                    t.pop("wfeats", None)
                chunk_result["turns"] = turns
            except Exception as e:
                print(f"  Warning: chunk {chunk['chunk_id']} failed: {e}")
                chunk_result["turns"] = []
                chunk_result["error"] = str(e)

        results.append(chunk_result)

    return results


# ---------------------------------------------------------------------------
# Step 5: Batch-process all txt files in a directory
# ---------------------------------------------------------------------------

def process_directory(input_dir: str, output_dir: str,
                      min_dur: float = CHUNK_MIN_DURATION,
                      max_dur: float = CHUNK_MAX_DURATION,
                      chunked: bool = True):
    """Process all .txt files under input_dir and write .json outputs.

    For each input file like:
        input_dir/TheBigBangTheory.Season01.Episode01.txt

    writes the parsed turns to:
        output_dir/TheBigBangTheory.Season01.Episode01.json

    When chunked=True (default), the output JSON is a list of chunk objects,
    each with metadata (start, end, duration, speakers) and a 'turns' list.

    Args:
        input_dir: directory containing .txt annotation files.
        output_dir: directory to write .json output files.
        min_dur / max_dur: chunk duration bounds in seconds.
        chunked: if True, split into chunks; if False, process whole file.
    """
    input_path = Path(input_dir)
    # output_path = Path(output_dir)
    # output_path.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_path.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    print(f"Found {len(txt_files)} txt file(s) in {input_dir}")
    # print(f"Output directory: {output_dir}")
    if chunked:
        print(f"Chunk duration: {min_dur/60:.0f}min – {max_dur/60:.0f}min")
    print()
    txt_files = txt_files[:10]
    for txt_file in txt_files:
        print(f"Processing: {txt_file.name}")

        if chunked:
            chunks = process_file_chunked(str(txt_file), min_dur, max_dur)
            total_turns = sum(len(c["turns"]) for c in chunks)
            all_speakers = set()
            for c in chunks:
                all_speakers.update(c["speakers"])

            print(f"  -> {len(chunks)} chunks, {total_turns} total turns")
            print(f"  -> Speakers: {sorted(all_speakers)}")

            # Print chunk summary.
            for c in chunks:
                n_turns = len(c["turns"])
                print(f"     Chunk {c['chunk_id']:2d}: "
                      f"{c['start']:7.1f}s – {c['end']:7.1f}s  "
                      f"({c['duration']/60:4.1f}min, "
                      f"{c['num_words']:4d} words, "
                      f"{n_turns:3d} turns, "
                      f"{len(c['speakers'])} spk)")

            # out_file = output_path / txt_file.with_suffix(".json").name
            # with open(out_file, "w", encoding="utf-8") as f:
            #     json.dump(chunks, f, indent=2, ensure_ascii=False)

        else:
            turns = process_file(str(txt_file))
            print(f"  -> {len(turns)} turns extracted")
            speakers = set(t["speaker"] for t in turns)
            print(f"  -> Speakers: {sorted(speakers)}")

            # out_file = output_path / txt_file.with_suffix(".json").name
            # with open(out_file, "w", encoding="utf-8") as f:
            #     json.dump(turns, f, indent=2, ensure_ascii=False)

        # print(f"  -> Saved to: {out_file}")
        # print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process TV-series txt annotations into turn-level JSON.")
    parser.add_argument(
        "--input_dir", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tv_series"),
        help="Directory containing .txt annotation files (default: ./tv_series)")
    # parser.add_argument(
    #     "--output_dir", type=str,
    #     default=None,
    #     help="Directory to write .json output files (default: same as input_dir)")
    parser.add_argument(
        "--min_dur", type=float, default=CHUNK_MIN_DURATION,
        help=f"Minimum chunk duration in seconds (default: {CHUNK_MIN_DURATION})")
    parser.add_argument(
        "--max_dur", type=float, default=CHUNK_MAX_DURATION,
        help=f"Maximum chunk duration in seconds (default: {CHUNK_MAX_DURATION})")
    parser.add_argument(
        "--no_chunk", action="store_true",
        help="Process whole file without chunking")

    args = parser.parse_args()
    # if args.output_dir is None:
    #     args.output_dir = args.input_dir
    args.output_dir = None

    process_directory(args.input_dir, args.output_dir,
                      min_dur=args.min_dur, max_dur=args.max_dur,
                      chunked=not args.no_chunk)
