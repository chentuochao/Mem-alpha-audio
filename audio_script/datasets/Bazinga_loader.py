"""
Loader for the Bazinga (Friends TV show) dataset.

Dataset layout:
    data_dir/
        Friends.Season01.Episode01.en.wav
        Friends.Season01.Episode01.txt
        Friends.Season01.Episode02.en.wav
        Friends.Season01.Episode02.txt
        ...

TXT format (space-separated, 9 columns):
    file_id  speaker  start_time  end_time  word  confidence  listener  scene_context  misc

Provides a discover_conversations() function compatible with the
step1_diarize_asr.py pipeline, plus an in-memory BazingaDataset class.
"""

import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_bazinga_txt(txt_path: str) -> Dict[str, List[Dict]]:
    """
    Parse a Bazinga annotation txt file and return per-speaker word lists.

    Each word entry is a dict compatible with AlignedProcess.split_trans:
        { "word": str, "start": float, "end": float, "score": float,
          "listener": str, "scene_context": str }

    Returns:
        { speaker_name: [word_dict, ...] }  (words sorted by start time)
    """
    speaker_words: Dict[str, List[Dict]] = defaultdict(list)
    raw_transcript = []

    with open(txt_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                print("Warning: Malformed line in Bazinga txt file: ", line)
                continue  # malformed line

            # file_id speaker start end word confidence [listener] [scene] [misc]
            speaker = parts[1]
            try:
                start = float(parts[2])
                end = float(parts[3])
            except ValueError:
                continue
            word = parts[4]
            try:
                score = float(parts[5])
            except ValueError:
                score = 1.0

            listener = parts[6] if len(parts) > 6 else "_"
            scene_context = parts[7] if len(parts) > 7 else "_"
            raw_transcript.append({
                "word": word,
                "start": start,
                "end": end,
                "score": score,
                "listener": listener,
                "scene_context": scene_context,
            })
            speaker_words[speaker].append({
                "word": word,
                "start": start,
                "end": end,
                "score": score,
                "listener": listener,
                "scene_context": scene_context,
            })

    # Sort each speaker's words by start time
    for spk in speaker_words:
        speaker_words[spk].sort(key=lambda w: w["start"])

    return dict(speaker_words), raw_transcript


def words_to_segments(
    word_list: List[Dict],
    turn_gap: float = 1.5,
) -> List[Dict]:
    """
    Group a flat list of word dicts (sorted by start time) into utterance
    segments separated by silences longer than `turn_gap` seconds.

    Each returned segment has the format expected by AlignedProcess:
        { "start": float, "end": float, "text": str, "words": [word_dict, ...] }
    """
    if not word_list:
        return []

    segments: List[Dict] = []
    current_words: List[Dict] = [word_list[0]]

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
    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": " ".join(w["word"] for w in words),
        "words": words,
    }


def speaker_words_to_transcript(
    speaker_words: Dict[str, List[Dict]],
    turn_gap: float = 1.5,
) -> Dict[str, List[Dict]]:
    """
    Convert per-speaker word dicts into per-speaker segment lists
    (the format accepted by AlignedProcess as transcript1 / transcript2).
    """
    return {
        spk: words_to_segments(words, turn_gap=turn_gap)
        for spk, words in speaker_words.items()
    }


def transcription_to_vad(transcripts: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """
    Convert per-speaker segment dicts into per-speaker VAD dicts
    (the format accepted by AlignedProcess as vad1 / vad2).
    """
    return {spk: [{"start": seg["start"], "end": seg["end"]} for seg in segs] for spk, segs in transcripts.items()}


# ──────────────────────────────────────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────────────────────────────────────

def _find_episodes(data_dir: str) -> List[Tuple[str, str, str]]:
    """
    Scan *data_dir* for matched (episode_id, wav_path, txt_path) triples.

    Wav files are expected to have the suffix `.en.wav`; txt files share the
    same stem without the `.en` part.

    Returns list of (episode_id, wav_path, txt_path).
    """
    wav_files = sorted(glob.glob(os.path.join(data_dir, "*.en.wav")))
    episodes: List[Tuple[str, str, str]] = []

    for wav_path in wav_files:
        basename = os.path.basename(wav_path)           # e.g. Friends.S01.E01.en.wav
        episode_id = basename.replace(".en.wav", "")    # e.g. Friends.S01.E01
        txt_path = os.path.join(data_dir, episode_id + ".txt")

        if not os.path.exists(txt_path):
            # Fall back: look for any .txt whose stem starts with episode_id
            candidates = glob.glob(os.path.join(data_dir, episode_id + "*.txt"))
            if not candidates:
                continue
            txt_path = candidates[0]

        episodes.append((episode_id, wav_path, txt_path))

    return episodes


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline-compatible discover_conversations
# ──────────────────────────────────────────────────────────────────────────────



# ──────────────────────────────────────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────────────────────────────────────

class BazingaDataset:
    """
    Dataset class for the Bazinga (Friends) corpus.

    Provides episode-level access to:
        - mixed audio (loaded as float32 numpy array)
        - per-speaker segment transcripts (compatible with AlignedProcess)
        - raw per-speaker word lists

    Example usage
    -------------
    >>> dataset = BazingaDataset("/path/to/bazinga")
    >>> sample = dataset.load_sample(0)
    >>> print(sample["conv_id"], sample["speakers"])
    >>> print(sample["transcripts"]["monica_geller"][:2])
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 16000,
    ):
        self.data_dir = data_dir
        self.sample_rate = sample_rate

        self.conversations = _find_episodes(data_dir)

        if not self.conversations:
            raise ValueError(
                f"No matching .en.wav / .txt episode pairs found in '{data_dir}'"
            )

    def __len__(self) -> int:
        return len(self.conversations)

    def __repr__(self) -> str:
        return (
            f"BazingaDataset(data_dir={self.data_dir!r}, "
            f"num_episodes={len(self)})"
        )

    # ------------------------------------------------------------------
    # Core access
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load one episode by index.

        Returns a dict with:
            conv_id     – episode identifier string
            audio       – float32 numpy array, shape (T,)
            sr          – sample rate (int)
            speakers    – sorted list of speaker name strings
            transcripts – { speaker: [ segment_dict, ... ] }
                          where each segment_dict has keys:
                            start (float), end (float), text (str),
                            words (list of word dicts)
            word_lists  – { speaker: [ word_dict, ... ] }
                          flat per-speaker word list (start/end/word/score)
            audio_path  – path to the wav file
            txt_path    – path to the annotation txt file
        """
        conv = self.conversations[idx]
        episode_id, audio_path, txt_path = conv

        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        audio = audio.astype(np.float32)

        transcripts, raw_transcript = parse_bazinga_txt(txt_path)
        speakers = list(transcripts.keys())

        # transcript_gt_path = os.path.join(self.data_dir, episode_id + "_gt_transcript.json")
        # with open(transcript_gt_path, "w", encoding="utf-8") as fh:
        #     json.dump(transcripts, fh, indent=2)
        # ## convert transcription to VAD array for diarization gt
        # vad_gt = transcription_to_vad(transcripts)
        # # save vad_gt to json file
        # vad_gt_path = os.path.join(self.data_dir, episode_id + "_gt_vad.json")
        # with open(vad_gt_path, "w", encoding="utf-8") as fh:
        #     json.dump(vad_gt, fh, indent=2)


        # Re-build flat word lists from the cached segment data
        return {
            "conv_id": episode_id,
            "audio": audio,
            "raw_transcript": raw_transcript,
            "audio_path": audio_path,
            "txt_path": txt_path,
            "sr": sr,
            "speakers": speakers
        }
