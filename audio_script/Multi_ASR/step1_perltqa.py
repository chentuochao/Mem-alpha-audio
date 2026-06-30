"""
Step 1 (PerLTQA): Multi-talker diarization + ASR on the PerLTQA dialogue-TTS audio.

This mirrors :pymod:`audio_script.Multi_ASR.step1_bazinga` exactly — same
chunking, ground-truth building, per-chunk output layout, and pluggable
inference backends — but sources its "episodes" from the dialogue-TTS folders
produced by ``ctbox_tts/perltqa_dialogue_tts.py`` and annotated by
``ctbox_tts/generate_annotations.py``.

Each PerLTQA episode is one dialogue folder::

    <data_dir>/<Profile>/<dialogue_id>/
        dialogue_mono_TTS.wav        # mixed mono audio (backend input)
        dialogue_multichannel_TTS.wav
        <Speaker>_TTS.npy            # per-speaker isolated tracks
        <Speaker>_annotation.json    # GT: WhisperX words + Silero VAD
        channel_map.json

The per-speaker GT word alignments written by ``generate_annotations.py``
(``transcript.words``: ``{speaker, word, start, end, score}`` with timestamps
absolute within the dialogue) are merged into a single flat ``raw_transcript`` —
exactly the shape the shared pipeline (chunk_dialog / build_speaker_transcripts /
process_episode) consumes.

IMPORTANT: run ``generate_annotations.py`` FIRST so every dialogue folder has its
``*_annotation.json`` files; folders without annotations are skipped.

Invocation (works as module or script when the repo root is on PYTHONPATH)::

    python -m audio_script.Multi_ASR.step1_perltqa --method vibevoice \\
        --data_dir /path/to/perltqa/audio --output_dir /path/to/out \\
        --model_path /path/to/vibevoice-checkpoint

    python -m audio_script.Multi_ASR.step1_perltqa --method nemo-offline \\
        --data_dir /path/to/perltqa/audio --output_dir /path/to/out \\
        --diar_model_path diar.nemo --asr_model_path asr.nemo

Per-chunk output layout (identical to Bazinga), under
``<output_dir>/<Profile>/<dialogue_id>/CHUNK_<i>/``:
    diart_pred.npy / transcript_pred.json / transcript_gt.json /
    vad_gt.json / sample_info.json
"""

from __future__ import annotations

import argparse
import glob
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


DATASET_NAME = "perltqa"

# Mixed-audio candidates, in preference order (librosa down-mixes to mono).

# ─────────────────────────────────────────────────────────────────────────────
# Annotation -> flat word list
# ─────────────────────────────────────────────────────────────────────────────

def _words_from_annotation(ann: Dict, speaker_fallback: str) -> List[Dict]:
    """Flatten one speaker's annotation into word dicts the pipeline expects."""
    speaker = ann.get("speaker") or speaker_fallback
    tr = ann.get("transcript", {}) or {}

    out: List[Dict] = []

    # Preferred: the flat `words` list emitted by generate_annotations.py.
    for w in tr.get("words", []) or []:
        if w.get("start") is None or w.get("end") is None:
            continue
        out.append({
            "speaker": w.get("speaker", speaker),
            "word": w.get("word", ""),
            "start": float(w["start"]),
            "end": float(w["end"]),
            "score": float(w.get("score", 1.0)),
        })
    if out:
        return out

    # Fallback: flatten WhisperX `segments[].words` (older annotation files).
    for seg in tr.get("segments", []) or []:
        for w in seg.get("words", []) or []:
            if w.get("start") is None or w.get("end") is None:
                continue
            out.append({
                "speaker": speaker,
                "word": w.get("word", ""),
                "start": float(w["start"]),
                "end": float(w["end"]),
                "score": float(w.get("score", 1.0)),
            })
    return out


def _load_mixed_audio(folder: str, sample_rate: int) -> Tuple[np.ndarray, str]:
    """Load the dialogue's mixed mono audio (resampled to ``sample_rate``)."""
    path = os.path.join(folder, "dialogue_mono_TTS.wav")
    print("*"*10, path)
    if os.path.exists(path):
        audio, _ = librosa.load(path, sr=sample_rate, mono=True)
        return audio.astype(np.float32), path
    raise FileNotFoundError(
        f"No mixed-audio wav ({' / '.join(_AUDIO_CANDIDATES)}) found in {folder}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PerLTQADataset:
    """Episode-level access to PerLTQA dialogue-TTS folders.

    Yields the same ``sample`` dict contract as ``BazingaDataset`` so it can be
    fed straight into ``run_dataset`` / ``process_episode``:
        conv_id, audio (float32 @ SR), raw_transcript (flat word dicts),
        audio_path, txt_path, sr, speakers.

    Only folders that have at least one ``*_annotation.json`` (i.e. that have
    been through ``generate_annotations.py``) are included.
    """

    def __init__(self, data_dir: str, sample_rate: int = SR):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.conversations = self._discover(data_dir)
        if not self.conversations:
            raise ValueError(
                f"No annotated dialogue folders found under {data_dir!r}. "
                f"Run generate_annotations.py first so each folder has its "
                f"*_annotation.json files."
            )

    @staticmethod
    def _discover(data_dir: str) -> List[str]:
        folders = sorted({
            os.path.dirname(p)
            for p in glob.glob(os.path.join(data_dir, "**", "channel_map.json"),
                               recursive=True)
        })
        return [
            f for f in folders
            if glob.glob(os.path.join(f, "*_annotation.json"))
        ]

    def __len__(self) -> int:
        return len(self.conversations)

    def __repr__(self) -> str:
        return (f"PerLTQADataset(data_dir={self.data_dir!r}, "
                f"num_episodes={len(self)})")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        folder = self.conversations[idx]
        conv_id = os.path.relpath(folder, self.data_dir).replace(os.sep, "_")

        audio, audio_path = _load_mixed_audio(folder, self.sample_rate)

        raw_transcript: List[Dict] = []
        speakers: List[str] = []
        for ann_path in sorted(glob.glob(os.path.join(folder, "*_annotation.json"))):
            with open(ann_path, "r", encoding="utf-8") as fh:
                ann = json.load(fh)
            fallback = os.path.basename(ann_path)[: -len("_annotation.json")]
            words = _words_from_annotation(ann, fallback)
            if not words:
                continue
            spk = words[0]["speaker"]
            if spk not in speakers:
                speakers.append(spk)
            raw_transcript.extend(words)

        # The shared pipeline assumes a globally time-sorted word stream.
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
        description="Step 1 (PerLTQA): multi-talker diarization + ASR with "
                    "pluggable inference backends."
    )
    add_common_args(parser)   # --data_dir --output_dir --device --season_filter
    add_backend_args(parser)  # --method + backend-specific flags
    return parser


def main():
    args = build_argparser().parse_args()

    dataset = PerLTQADataset(args.data_dir, sample_rate=SR)
    print(f"Found {len(dataset)} annotated dialogue(s) under {args.data_dir}")

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
