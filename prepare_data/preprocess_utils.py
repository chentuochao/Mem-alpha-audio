import pandas as pd
import json
import os
import glob
import jsonlines
import string
import jsonlines
from typing import Dict, List, Optional, Tuple


def transcription_to_vad(transcripts: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    return {
        spk: [{"start": seg["start"], "end": seg["end"]} for seg in segs]
        for spk, segs in transcripts.items()
    }



def chunk_dialog(dialog, min_dur=60.0, max_dur=600.0, gap_threshold=4.0):
    """
    Split a flat list of utterance dicts into smaller conversation chunks.

    Strategy:
      1. A natural boundary is detected when the silence gap between two
         consecutive utterances exceeds `gap_threshold` seconds.
      2. A chunk is cut at the earliest natural boundary AFTER `min_dur` seconds
         have accumulated, or hard-cut at `max_dur` regardless.
      3. If no boundary is found before `max_dur`, the chunk is cut between the
         two utterances whose gap is the largest within the window.

    Args:
        dialog: list of utterance dicts with keys 'start', 'end', 'text', 'speaker'.
        min_dur: minimum chunk duration in seconds.
        max_dur: maximum chunk duration in seconds.
        gap_threshold: silence gap (seconds) that marks a natural boundary.

    Returns:
        List of lists, each inner list is a group of utterance dicts.
    """
    if not dialog:
        return []

    chunks = []
    chunk_start = 0  # index into dialog

    while chunk_start < len(dialog):
        chunk_begin_time = dialog[chunk_start]["start"]
        best_gap_idx = None   # index of utterance *after* which to cut (largest gap)
        best_gap_val = -1.0

        i = chunk_start
        max_end_so_far = dialog[chunk_start]["end"]
        while i < len(dialog):
            max_end_so_far = max(max_end_so_far, dialog[i]["end"])
            elapsed = max_end_so_far - chunk_begin_time

            # Compute gap to next utterance (if any)
            if i + 1 < len(dialog):
                gap = dialog[i + 1]["start"] - max_end_so_far
            else:
                gap = 0.0

            # Track largest gap seen so far (fallback hard-cut)
            if gap > best_gap_val:
                best_gap_val = gap
                best_gap_idx = i

            # Natural boundary after min_dur
            if elapsed >= min_dur and gap >= gap_threshold:
                chunks.append(dialog[chunk_start: i + 1])
                chunk_start = i + 1
                break

            # Hard cut at max_dur
            if elapsed >= max_dur:
                # Cut at the largest gap seen within the window
                cut_at = best_gap_idx if best_gap_idx is not None else i
                chunks.append(dialog[chunk_start: cut_at + 1])
                chunk_start = cut_at + 1
                break

            i += 1
        else:
            # Reached end of dialog — append remaining utterances to the last chunk
            if chunks:
                chunks[-1] = chunks[-1] + dialog[chunk_start:]
            else:
                chunks.append(dialog[chunk_start:])
            break

    return chunks



def fix_space_in_text(text):
    punctuation_pattern = [" " + c for c in string.punctuation]
    punctuation_pattern.append(" n't")
    # punctuation_pattern.extend([" 'm", " 's", " 've", " 're", " 'll", " 'd", " 'n", " 't", " 'y", " 'z"])
    for pattern in punctuation_pattern:
        text = text.replace(pattern, pattern.strip())
    return text



def load_episode_chunks(
    json_paths: list,
    time_maps: dict,
    min_dur: float = 60.0,
    max_dur: float = 300.0,
    gap_threshold: float = 5.0,
    use_gt_speaker: bool = False,
):
    """Load and merge chunks from one or more JSON episode files.

    Speaker IDs are assigned globally so the same real speaker name maps
    to the same Speaker_X label across all input files.
    """
    speakers_pool = {}
    chunks = []

    if isinstance(json_paths, str):
        json_paths = [json_paths]

    for json_path in json_paths:
        session_name = os.path.basename(json_path)
        timestamp = time_maps[session_name]
        with open(json_path, "r") as f:
            dialog = json.load(f)

        sub_chunks = chunk_dialog(dialog, min_dur=min_dur, max_dur=max_dur, gap_threshold=gap_threshold)

        for sub in sub_chunks:
            dialog_chunk = f"[Dialogue between multiple people on {timestamp}]\n"
            for turn in sub:
                speaker = turn["speaker"]
                if speaker not in speakers_pool:
                    speakers_pool[speaker] = len(speakers_pool)
                anon_speaker = "Speaker_" + str(speakers_pool[speaker])
                turn_text = fix_space_in_text(turn["text"])

                if use_gt_speaker:
                    dialog_chunk += f"<{speaker}> {turn_text}\n"
                else:
                    dialog_chunk += f"<{anon_speaker}> {turn_text}\n"
            chunks.append(dialog_chunk)

    return chunks, speakers_pool
