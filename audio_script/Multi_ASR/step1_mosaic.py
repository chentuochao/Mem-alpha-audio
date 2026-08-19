"""
Step 1 (Mix_Mosaic): Multi-talker diarization + ASR on the mixed Seamless-
Interaction dyads produced by ``audio_script/datasets/mix_interact.py``.

This mirrors :pymod:`audio_script.Multi_ASR.step1_bazinga` and
:pymod:`audio_script.Multi_ASR.step1_perltqa` exactly — same chunking, ground-
truth building, per-chunk output layout, and pluggable inference backends — but
sources its "episodes" from the pre-mixed Mix_Mosaic folders.

Data layout (one folder per speaker *pair*, one sub-folder per conversation)::

    <data_dir>/
        Pxxx_Pyyy/                     # Pxxx / Pyyy are stable speaker ids
            V00_Sxxxx_Ixxxxxxxx/
                mixed_conv.wav         # mixed mono audio (backend input)
                transcript1.json       # speaker Pxxx: turn segments (+ wfeats)
                transcript2.json       # speaker Pyyy: turn segments (+ wfeats)
                vad1.json  vad2.json   # per-speaker VAD (not needed here — the
                                       # pipeline rebuilds vad_gt from the turns)

``transcript1.json`` always corresponds to the first id in the folder name and
``transcript2.json`` to the second (mix_interact.py sorts the pair before
writing). Each transcript is a list of turn segments ``{start, end, text,
speaker, wfeats, ...}``; we take the turn-level ``text`` + timing (word-level
timing is not needed: cpWER uses per-speaker text, DER uses per-turn
``{start,end}``, and chunking splits on per-unit ``start``/``end``), relabel the
speaker to the real Pxxx id, and merge both speakers into one time-sorted
``raw_transcript`` — exactly the shape the shared pipeline consumes.

``conv_id`` embeds the pair folder (``"<Pxxx_Pyyy>_<conv>"``) so Step 2's
``--season_filter`` can select a whole bundle by its pair-folder names.

Invocation (works as module or script when the repo root is on PYTHONPATH)::

    python -m audio_script.Multi_ASR.step1_mosaic --method vibevoice \\
        --data_dir /path/to/Mix_Mosaic/naturalistic/test --output_dir /path/out \\
        --model_path /path/to/vibevoice-checkpoint

    python -m audio_script.Multi_ASR.step1_mosaic --method nemo-offline \\
        --data_dir /path/to/Mix_Mosaic/naturalistic/test --output_dir /path/out \\
        --diar_model_path diar.nemo --asr_model_path asr.nemo --max_num_of_spks 4

Per-chunk output layout (identical to Bazinga / PerLTQA), under
``<output_dir>/<conv_id>/CHUNK_<i>/``:
    diart_pred.npy / transcript_pred.json / transcript_gt.json /
    vad_gt.json / sample_info.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np

# Absolute imports so this runs as ``python -m ...`` or as a direct script.
from audio_script.Multi_ASR.backends import add_backend_args, build_backend
from audio_script.Multi_ASR.constants import SR
# Reuse the *entire* dataset-agnostic pipeline from the Bazinga step1.
from audio_script.Multi_ASR.step1_bazinga import add_common_args, run_dataset


DATASET_NAME = "mosaic"

MIXED_AUDIO_NAME = "mixed_conv.wav"
# (transcript file, index of the matching speaker id in the pair folder name)
TRANSCRIPT_FILES: Tuple[Tuple[str, int], ...] = (
    ("transcript1.json", 0),
    ("transcript2.json", 1),
)


# ─────────────────────────────────────────────────────────────────────────────
# Transcript -> flat turn list
# ─────────────────────────────────────────────────────────────────────────────

def _turns_from_transcript(segments: List[Dict], speaker: str) -> List[Dict]:
    """Read one speaker's turn segments into the GT units the pipeline expects.

    Each returned unit is ``{speaker, text, start, end}`` with the speaker
    relabelled to the real Pxxx id (the on-disk ``wfeats`` speaker is just
    "A"/"B"). ``text`` falls back to joining the word-level ``wfeats`` when a
    segment has no ``text`` field.
    """
    out: List[Dict] = []
    for seg in segments or []:
        if seg.get("start") is None or seg.get("end") is None:
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            text = " ".join(w.get("word", "") for w in seg.get("wfeats", [])).strip()
        if not text:
            continue
        out.append({
            "speaker": speaker,
            "text": text,
            "start": float(seg["start"]),
            "end": float(seg["end"]),
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MixMosaicDataset:
    """Episode-level access to the pre-mixed Mix_Mosaic conversation folders.

    Yields the same ``sample`` dict contract as ``BazingaDataset`` /
    ``PerLTQADataset`` so it can be fed straight into ``run_dataset`` /
    ``process_episode``:
        conv_id, audio (float32 @ SR), raw_transcript (flat turn dicts),
        audio_path, txt_path, sr, speakers.
    """

    def __init__(self, data_dir: str, sample_rate: int = SR):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.conversations = self._discover(data_dir)
        if not self.conversations:
            raise ValueError(
                f"No Pxxx_Pyyy/<conv>/{MIXED_AUDIO_NAME} conversations found "
                f"under {data_dir!r}."
            )

    @staticmethod
    def _discover(data_dir: str) -> List[Tuple[str, str]]:
        """Return sorted ``(pair_folder, conv_id)`` tuples that have mixed audio."""
        found: List[Tuple[str, str]] = []
        for pair in sorted(os.listdir(data_dir)):
            pair_dir = os.path.join(data_dir, pair)
            if not os.path.isdir(pair_dir):
                continue
            for conv in sorted(os.listdir(pair_dir)):
                conv_dir = os.path.join(pair_dir, conv)
                if os.path.isfile(os.path.join(conv_dir, MIXED_AUDIO_NAME)):
                    found.append((pair, conv))
        return found

    def __len__(self) -> int:
        return len(self.conversations)

    def __repr__(self) -> str:
        return (f"MixMosaicDataset(data_dir={self.data_dir!r}, "
                f"num_episodes={len(self)})")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair, conv = self.conversations[idx]
        conv_dir = os.path.join(self.data_dir, pair, conv)
        # conv_id embeds the pair folder so Step2 --season_filter can select a
        # whole bundle by its pair-folder names.
        conv_id = f"{pair}_{conv}"
        speaker_ids = pair.split("_")

        audio_path = os.path.join(conv_dir, MIXED_AUDIO_NAME)
        audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        audio = audio.astype(np.float32)

        raw_transcript: List[Dict] = []
        speakers: List[str] = []
        for fname, spk_idx in TRANSCRIPT_FILES:
            path = os.path.join(conv_dir, fname)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as fh:
                segments = json.load(fh)
            speaker = (speaker_ids[spk_idx] if spk_idx < len(speaker_ids)
                       else f"spk{spk_idx}")
            turns = _turns_from_transcript(segments, speaker)
            if not turns:
                continue
            if speaker not in speakers:
                speakers.append(speaker)
            raw_transcript.extend(turns)

        # The shared pipeline assumes a globally time-sorted unit stream.
        raw_transcript.sort(key=lambda w: (w["start"], w["end"]))

        return {
            "conv_id": conv_id,
            "audio": audio,
            "raw_transcript": raw_transcript,
            "audio_path": audio_path,
            "txt_path": None,
            "sr": self.sample_rate,
            "speakers": speakers,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI / entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 1 (Mix_Mosaic): multi-talker diarization + ASR with "
                    "pluggable inference backends."
    )
    add_common_args(parser)   # --data_dir --output_dir --device --season_filter
    add_backend_args(parser)  # --method + backend-specific flags
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split the conversation list into this many shards "
                             "(for data-parallel runs across GPUs).")
    parser.add_argument("--shard_index", type=int, default=0,
                        help="Which shard this process handles (0-based). Shards "
                             "are round-robin over the sorted conversation list, "
                             "so every shard gets a mix of long/short files. All "
                             "shards can safely share one --output_dir: results "
                             "are written per conv_id.")
    return parser


def main():
    args = build_argparser().parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards.")

    dataset = MixMosaicDataset(args.data_dir, sample_rate=SR)
    print(f"Found {len(dataset)} conversation(s) under {args.data_dir}")
    if args.num_shards > 1:
        dataset.conversations = dataset.conversations[args.shard_index::args.num_shards]
        print(f"Shard {args.shard_index}/{args.num_shards}: "
              f"{len(dataset)} conversation(s) to process")

    backend = build_backend(args)
    print(f"Using inference backend: {backend.name}")

    run_dataset(
        dataset=dataset,
        backend=backend,
        output_dir=args.output_dir,
        dataset_name=DATASET_NAME,
        season_filter=args.season_filter,
    )


if __name__ == "__main__":
    main()
