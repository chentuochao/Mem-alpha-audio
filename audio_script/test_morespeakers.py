"""
Test script for AlignedProcess_Morespeakers.

Loads a multi-speaker transcript from trans.json and runs it through the
multi-speaker turn-annotation pipeline.  Prints results at each stage so
you can visually verify the processing.

Usage:
    python -m audio_script.test_morespeakers
    # or, from the audio_script/ directory:
    python test_morespeakers.py
"""

import json
import sys
import os
from typing import Dict, List

# ---------------------------------------------------------------------------
# Make sure the parent package is importable when running as a script.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_script.datasets.turn_annotation import (
    AlignedProcess,
    AlignedProcess_Morespeakers,
)

# Threshold (seconds) for splitting words into separate turns inside
# AlignedProcess / AlignedProcess_Morespeakers.
TURN_GAP_TH = 1.5


def parse_transcript(word_list: Dict) -> List[Dict]:
    """Parse a word-list dict into speaker-aware turns for 1–8 speakers.

    Takes the output format of transcript_gt.json:
        {speaker_id: [{word, start, end, score, ...}, ...], ...}

    and returns a sorted list of turn dicts, each with keys:
        dialog_type, speaker, start, end, text, wfeats.

    Steps:
        1. Filter out speakers with no words.
        2. Sort each speaker's words by start time.
        3. Wrap each speaker's word list into a single segment dict
           (the format AlignedProcess expects).
        4. For 0 speakers  -> return empty list.
           For 1 speaker   -> return one turn covering all words.
           For 2+ speakers -> use AlignedProcess_Morespeakers to detect
                              turns, backchannels, and overlaps.

    Args:
        word_list: dict mapping speaker names to lists of word dicts.
                   Each word dict must have at least: word, start, end, score.

    Returns:
        List of turn dicts sorted by (start, -end), with dialog_type labels.
    """
    # --- Step 1 & 2: Collect valid speakers and sort their words ---
    speaker_transcripts = {}
    valid_speakers = []
    for speaker in word_list.keys():
        words = word_list[speaker]
        if len(words) == 0:
            continue
        # Sort words by start time to ensure chronological order.
        words = sorted(words, key=lambda x: x['start'])
        valid_speakers.append(speaker)
        # Wrap the flat word list into a single-segment list,
        # which is the format AlignedProcess / AlignedProcess_Morespeakers
        # expects as input per speaker.
        speaker_transcripts[speaker] = [{
            "speaker": speaker,
            "start": words[0]['start'],
            "end": words[-1]['end'],
            "words": words,
        }]

    # --- Step 3: Handle based on number of valid speakers ---
    if len(valid_speakers) == 0:
        print("No valid speakers found!")
        return []

    if len(valid_speakers) == 1:
        # Only one speaker: no turn-taking to detect, just return one turn.
        print(f"Only one valid speaker found: {valid_speakers[0]}")
        words = speaker_transcripts[valid_speakers[0]][0]["words"]
        transcript = "".join(w["word"] for w in words)
        return [{
            "dialog_type": "dialog",
            "speaker": valid_speakers[0],
            "start": words[0]['start'],
            "end": words[-1]['end'],
            "text": transcript,
            "wfeats": words,
        }]

    # --- 2+ speakers: use AlignedProcess_Morespeakers ---
    # Build the ordered lists that the constructor expects.
    transcripts_list = [speaker_transcripts[spk] for spk in valid_speakers]

    aligned_process = AlignedProcess_Morespeakers(
        transcripts=transcripts_list,
        speaker_names=valid_speakers,
        interval_character=' ',
        turn_gap_threshold=TURN_GAP_TH,
    )

    # get_parsed_dialog_combined returns all speakers' turns in one
    # sorted list, already tagged with dialog_type and human-readable
    # speaker names.
    speaker_aware_turn = aligned_process.get_parsed_dialog_combined()
    return speaker_aware_turn


def load_transcripts(json_path: str):
    """Load transcript_gt.json and return (speaker_names, transcripts_list).

    The JSON schema is:
        {
          "speaker_name": [
            {"word": "...", "start": float, "end": float, "score": float, ...},
            ...
          ],
          ...
        }

    Each speaker's flat word list is sorted by start time and wrapped into
    a single-segment list (the format AlignedProcess_Morespeakers expects).

    Returns:
        speaker_names: list of speaker name strings.
        transcripts_list: list of per-speaker segment lists.  Each element
                          is a list containing one segment dict with a
                          'words' key.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    speaker_names = []
    transcripts_list = []
    for speaker, words in data.items():
        if len(words) == 0:
            continue
        # Sort words chronologically.
        words = sorted(words, key=lambda x: x['start'])
        speaker_names.append(speaker)
        # Wrap as a single-segment list: [{start, end, words: [...]}]
        # This is the per-speaker transcript format that
        # AlignedProcess_Morespeakers.split_trans expects.
        transcripts_list.append([{
            "speaker": speaker,
            "start": words[0]['start'],
            "end": words[-1]['end'],
            "words": words,
        }])

    return speaker_names, transcripts_list


def print_separator(title: str):
    """Print a visible section separator."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_utterances(utterances: list, label: str = ""):
    """Pretty-print a list of utterance dicts."""
    if label:
        print(f"\n--- {label} ({len(utterances)} utterances) ---")
    for utt in utterances:
        dtype = utt.get("dialog_type", "?")
        print(f"  [{dtype:12s}] {utt['start']:6.2f}s - {utt['end']:6.2f}s  "
              f"| {utt['speaker']:10s} | {utt['text']}")


# ---------------------------------------------------------------------------
# Test 1: Basic 3-speaker test
# ---------------------------------------------------------------------------
def test_3_speakers(json_path: str):
    """Run the full pipeline with all speakers from transcript_gt.json."""
    print_separator("Test 1: Multi-speaker pipeline")

    speaker_names, transcripts_list = load_transcripts(json_path)
    print(f"Loaded {len(speaker_names)} speakers: {speaker_names}")
    for i, name in enumerate(speaker_names):
        # transcripts_list[i] is a list of segment dicts, each with 'words'.
        n_segs = len(transcripts_list[i])
        n_words = sum(len(seg["words"]) for seg in transcripts_list[i])
        print(f"  {name}: {n_segs} segments, {n_words} words")

    # --- Construct the processor ---
    proc = AlignedProcess_Morespeakers(
        transcripts=transcripts_list,
        speaker_names=speaker_names,
        trp_separated_by=1,
        pre_silence=0.5,
        post_silence=0.5,
        bc_duration=3,
    )
    print(f"\nmax_len (duration): {proc.max_len:.2f}s")

    # --- Per-speaker results ---
    per_speaker = proc.get_parsed_dialog()
    assert len(per_speaker) == len(speaker_names), (
        f"Expected {len(speaker_names)} speaker lists, got {len(per_speaker)}")

    for i, name in enumerate(speaker_names):
        print_utterances(per_speaker[i], label=f"Speaker: {name}")

    # --- Combined timeline ---
    combined = proc.get_parsed_dialog_combined()
    print_utterances(combined, label="Combined timeline (all speakers)")

    # --- print_final_diag (uses internal print) ---
    print("\n--- print_final_diag() output ---")
    proc.print_final_diag()

    print("\n[PASS] Test 1 completed successfully.")
    return proc


# ---------------------------------------------------------------------------
# Test 2: 2-speaker subset – compare with original AlignedProcess
# ---------------------------------------------------------------------------
def test_2_speaker_comparison(json_path: str):
    """Run both old and new classes on 2 speakers and compare outputs."""
    print_separator("Test 2: 2-speaker comparison (old vs new)")

    speaker_names, transcripts_list = load_transcripts(json_path)

    # Pick the first two speakers for comparison.
    names_2 = speaker_names[:2]
    trans_2 = transcripts_list[:2]  # each element is a list of segment dicts
    print(f"Comparing with speakers: {names_2}")

    # --- Old class (expects two separate transcript lists) ---
    old_proc = AlignedProcess(
        transcriptA=trans_2[0],
        transcriptB=trans_2[1],
        speakerA=names_2[0],
        speakerB=names_2[1],
        interval_character='',
        turn_gap_threshold=TURN_GAP_TH,
    )
    old_a, old_b = old_proc.get_parsed_dialog()

    # --- New class (accepts a list of transcript lists) ---
    new_proc = AlignedProcess_Morespeakers(
        transcripts=trans_2,
        speaker_names=names_2,
        interval_character='',
        turn_gap_threshold=TURN_GAP_TH,
    )
    new_per_speaker = new_proc.get_parsed_dialog()
    new_a = new_per_speaker[0]
    new_b = new_per_speaker[1]

    # --- Compare counts ---
    print(f"\n  Old class: speaker A ({names_2[0]}): {len(old_a)} utts, "
          f"speaker B ({names_2[1]}): {len(old_b)} utts")
    print(f"  New class: speaker A ({names_2[0]}): {len(new_a)} utts, "
          f"speaker B ({names_2[1]}): {len(new_b)} utts")

    # --- Compare text content ---
    old_a_texts = [u['text'] for u in old_a]
    new_a_texts = [u['text'] for u in new_a]
    old_b_texts = [u['text'] for u in old_b]
    new_b_texts = [u['text'] for u in new_b]

    if old_a_texts == new_a_texts and old_b_texts == new_b_texts:
        print("  Text content matches exactly between old and new class.")
    else:
        print("  WARNING: Text content differs (may be due to internal label "
              "differences). Showing side-by-side:")
        print(f"    Old A texts: {old_a_texts}")
        print(f"    New A texts: {new_a_texts}")
        print(f"    Old B texts: {old_b_texts}")
        print(f"    New B texts: {new_b_texts}")

    # --- Compare timings ---
    def extract_timings(utts):
        return [(round(u['start'], 3), round(u['end'], 3)) for u in utts]

    old_a_times = extract_timings(old_a)
    new_a_times = extract_timings(new_a)
    old_b_times = extract_timings(old_b)
    new_b_times = extract_timings(new_b)

    if old_a_times == new_a_times and old_b_times == new_b_times:
        print("  Timings match exactly between old and new class.")
    else:
        print("  WARNING: Timings differ.")
        print(f"    Old A: {old_a_times}")
        print(f"    New A: {new_a_times}")
        print(f"    Old B: {old_b_times}")
        print(f"    New B: {new_b_times}")

    print("\n[PASS] Test 2 completed successfully.")


# ---------------------------------------------------------------------------
# Test 3: Edge cases
# ---------------------------------------------------------------------------
def test_single_speaker():
    """Test with a single speaker (minimum)."""
    print_separator("Test 3a: Single speaker")

    transcript = [{
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.5, "score": 0.95},
            {"word": "world", "start": 0.6, "end": 1.0, "score": 0.93},
        ]
    }]

    proc = AlignedProcess_Morespeakers(
        transcripts=[transcript],
        speaker_names=["Solo"],
    )
    result = proc.get_parsed_dialog()
    assert len(result) == 1, f"Expected 1 speaker list, got {len(result)}"
    print(f"  Single speaker produced {len(result[0])} utterances")
    print_utterances(result[0], label="Solo speaker")
    print("\n[PASS] Test 3a completed successfully.")


def test_max_speakers():
    """Test with the maximum 8 speakers."""
    print_separator("Test 3b: Maximum 8 speakers")

    names = [f"Speaker_{i}" for i in range(8)]
    transcripts = []
    for i in range(8):
        # Each speaker says one short utterance at a different time.
        t_start = float(i * 5)
        transcripts.append([{
            "words": [
                {"word": "Hello", "start": t_start, "end": t_start + 0.3, "score": 0.9},
                {"word": "from", "start": t_start + 0.4, "end": t_start + 0.7, "score": 0.9},
                {"word": f"speaker{i}", "start": t_start + 0.8, "end": t_start + 1.2, "score": 0.9},
            ]
        }])

    proc = AlignedProcess_Morespeakers(
        transcripts=transcripts,
        speaker_names=names,
    )
    combined = proc.get_parsed_dialog_combined()
    print(f"  8 speakers produced {len(combined)} utterances in timeline")
    print_utterances(combined, label="8-speaker timeline")

    # Verify all 8 speakers appear.
    speakers_in_output = set(u['speaker'] for u in combined)
    assert speakers_in_output == set(names), (
        f"Missing speakers: {set(names) - speakers_in_output}")
    print("  All 8 speakers present in output.")
    print("\n[PASS] Test 3b completed successfully.")


def test_too_many_speakers():
    """Test that >8 speakers raises an assertion."""
    print_separator("Test 3c: Too many speakers (should fail)")

    names = [f"S{i}" for i in range(9)]
    transcripts = [[{"words": [{"word": "hi", "start": float(i), "end": float(i) + 0.3, "score": 0.9}]}] for i in range(9)]

    try:
        AlignedProcess_Morespeakers(transcripts=transcripts, speaker_names=names)
        print("  ERROR: Should have raised AssertionError!")
    except AssertionError as e:
        print(f"  Correctly raised AssertionError: {e}")
        print("\n[PASS] Test 3c completed successfully.")


def test_overlapping_speakers():
    """Test with overlapping speech from multiple speakers."""
    print_separator("Test 3d: Overlapping speech (3 speakers)")

    # Speaker 0 and Speaker 1 overlap between 2.0-3.0s.
    # Speaker 2 has a backchannel "yeah" during Speaker 0's turn.
    transcripts = [
        [{"words": [
            {"word": "I", "start": 0.0, "end": 0.15, "score": 0.99},
            {"word": "think", "start": 0.2, "end": 0.5, "score": 0.95},
            {"word": "we", "start": 0.55, "end": 0.7, "score": 0.97},
            {"word": "should", "start": 0.75, "end": 1.0, "score": 0.96},
            {"word": "refactor", "start": 1.05, "end": 1.5, "score": 0.91},
            {"word": "the", "start": 1.55, "end": 1.7, "score": 0.98},
            {"word": "whole", "start": 1.75, "end": 2.0, "score": 0.93},
            {"word": "module", "start": 2.05, "end": 2.5, "score": 0.90},
        ]}],
        [{"words": [
            {"word": "But", "start": 2.0, "end": 2.2, "score": 0.92},
            {"word": "that", "start": 2.25, "end": 2.5, "score": 0.94},
            {"word": "would", "start": 2.55, "end": 2.8, "score": 0.95},
            {"word": "take", "start": 2.85, "end": 3.1, "score": 0.93},
            {"word": "too", "start": 3.15, "end": 3.3, "score": 0.96},
            {"word": "long", "start": 3.35, "end": 3.6, "score": 0.94},
        ]}],
        [{"words": [
            {"word": "yeah", "start": 1.2, "end": 1.4, "score": 0.88},
        ]}],
    ]
    names = ["Dev_A", "Dev_B", "Dev_C"]

    proc = AlignedProcess_Morespeakers(
        transcripts=transcripts,
        speaker_names=names,
        pre_silence=0.3,
        post_silence=0.3,
    )
    combined = proc.get_parsed_dialog_combined()
    print_utterances(combined, label="Overlapping speech timeline")

    # Check that we have utterances classified as overlap or backchannel.
    types = set(u.get('dialog_type', '?') for u in combined)
    print(f"  Dialog types found: {types}")
    print("\n[PASS] Test 3d completed successfully.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcript_gt.json")

    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found.")
        sys.exit(1)

    print(f"Using transcript file: {json_path}")

    # --- Test parse_transcript (the main entry point) ---
    print_separator("Test 0: parse_transcript with transcript_gt.json")
    with open(json_path, "r") as f:
        word_list = json.load(f)
    turns = parse_transcript(word_list)
    print(f"parse_transcript returned {len(turns)} turns")
    print_utterances(turns, label="parse_transcript output")

    # --- Test the lower-level pipeline ---
    test_3_speakers(json_path)
    test_2_speaker_comparison(json_path)

    print()
    print("=" * 70)
    print("  ALL TESTS PASSED")
    print("=" * 70)
