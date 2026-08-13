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
            if speaker == "all" or "#unknown#" in speaker:
                continue

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

    # Sort each speaker's words chronologically.
    for spk in speaker_words:
        speaker_words[spk].sort(key=lambda w: w["start"])
    print("speakers = ", speaker_words.keys(), "\n")
    return dict(speaker_words)


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
# Step 4: Batch-process all txt files in a directory
# ---------------------------------------------------------------------------

def process_directory(input_dir: str, output_dir: str):
    """Process all .txt files under input_dir and write .json outputs.

    For each input file like:
        input_dir/TheBigBangTheory.Season01.Episode01.txt

    writes the parsed turns to:
        output_dir/TheBigBangTheory.Season01.Episode01.json

    Args:
        input_dir: directory containing .txt annotation files.
        output_dir: directory to write .json output files.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_path.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    print(f"Found {len(txt_files)} txt file(s) in {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    for txt_file in txt_files:
        print(f"Processing: {txt_file.name}")

        # Parse and get turns.
        turns = process_file(str(txt_file))
        print(f"  -> {len(turns)} turns extracted")

        # Show a quick summary of speakers found.
        speakers = set(t["speaker"] for t in turns)
        print(f"  -> Speakers: {sorted(speakers)}")

        # Write output JSON.
        out_file = output_path / txt_file.with_suffix(".json").name
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(turns, f, indent=2, ensure_ascii=False)
        print(f"  -> Saved to: {out_file}")

        # Print first few turns as a preview.
        print(f"  -> Preview (first 5 turns):")
        for turn in turns[:5]:
            print(f"     [{turn['dialog_type']:12s}] "
                  f"{turn['start']:7.2f}s - {turn['end']:7.2f}s  "
                  f"| {turn['speaker']:20s} | {turn['text'][:60]}")
        print()


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
    parser.add_argument(
        "--output_dir", type=str,
        default=None,
        help="Directory to write .json output files (default: same as input_dir)")

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir

    process_directory(args.input_dir, args.output_dir)
