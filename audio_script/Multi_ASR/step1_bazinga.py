"""
Step 1 (Bazinga): Multi-talker diarization + ASR on the Bazinga (Friends TV
show) dataset.

This file contains the entire dataset-side pipeline (chunking, GT building,
output writing, episode/dataset driver, CLI).  Inference backends remain in
:py:mod:`audio_script.Multi_ASR.backends` — pick one via ``--method``:

  - nemo-streaming  NeMo streaming SortFormer + cache-aware ASR     (env1)
  - nemo-offline    NeMo offline  SortFormer + offline ASR (merged) (env1)
  - vibevoice       VibeVoice end-to-end multi-talker ASR           (env2)

Invocation — works either as a module or as a direct script as long as the
repo root is on ``PYTHONPATH`` (the ``run_demo_step1*.sh`` scripts already
export it)::

    # module form
    python -m audio_script.Multi_ASR.step1_bazinga --method vibevoice ...

    # script form
    python audio_script/Multi_ASR/step1_bazinga.py --method vibevoice ...

Example arg sets:

    # NeMo streaming
    --method nemo-streaming \\
    --data_dir /path/to/bazinga --output_dir /path/to/out \\
    --diar_model_path /path/to/diar.nemo --asr_model_path /path/to/asr.nemo

    # NeMo offline
    --method nemo-offline \\
    --data_dir /path/to/bazinga --output_dir /path/to/out \\
    --diar_model_path /path/to/diar.nemo --asr_model_path /path/to/asr.nemo \\
    --max_num_of_spks 4

    # VibeVoice
    --method vibevoice \\
    --data_dir /path/to/bazinga --output_dir /path/to/out \\
    --model_path /path/to/vibevoice-checkpoint

Per-chunk output layout (one folder per chunk under <output_dir>/<conv_id>/):
    diart_pred.npy        binary diarization matrix (num_frames, num_speakers)
    transcript_pred.json  per-speaker word-level predictions
    transcript_gt.json    per-speaker ground-truth words (chunk-relative time)
    vad_gt.json           per-speaker VAD intervals
    sample_info.json      manifest entry for Step 2 / evaluation
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Absolute imports so this works whether run as ``python -m ...`` or as a
# direct script.
from audio_script.datasets.Bazinga_loader import BazingaDataset
from audio_script.Multi_ASR.backends import add_backend_args, build_backend
from audio_script.Multi_ASR.backends.base import BaseBackend
from audio_script.Multi_ASR.constants import FRAME_LEN_SEC, SR, CHUNK_MIN_DURATION, CHUNK_MAX_DURATION, CHUNK_GAP_THRESHOLD
from prepare_data.preprocess_utils import chunk_dialog, transcription_to_vad


DATASET_NAME = "bazinga"


def build_speaker_transcripts(
    chunk: List[Dict],
    chunk_start_sec: float,
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """Group a chunk's words by speaker, shifting timestamps to be chunk-relative.

    Returns ``(speaker_transcripts, vad_gt)``.
    """
    speaker_transcripts: Dict[str, List[Dict]] = defaultdict(list)
    for w in chunk:
        w_shifted = dict(w)
        w_shifted["start"] = w["start"] - chunk_start_sec
        w_shifted["end"] = w["end"] - chunk_start_sec
        speaker_transcripts[w["speaker"]].append(w_shifted)
    speaker_transcripts = dict(speaker_transcripts)
    return speaker_transcripts, transcription_to_vad(speaker_transcripts)


# ─────────────────────────────────────────────────────────────────────────────
# Per-chunk save
# ─────────────────────────────────────────────────────────────────────────────

def save_chunk_outputs(
    chunk_dir: str,
    diar_binary: np.ndarray,
    word_list: Dict[str, List[Dict]],
    speaker_transcripts_gt: Dict[str, List[Dict]],
    vad_gt: Dict[str, List[Dict]],
    sample_info: Dict,
) -> Tuple[str, str, str, str, str]:
    """Write the five canonical per-chunk artifacts and return their paths."""
    os.makedirs(chunk_dir, exist_ok=True)
    diar_path = os.path.join(chunk_dir, "diart_pred.npy")
    word_list_path = os.path.join(chunk_dir, "transcript_pred.json")
    word_list_gt_path = os.path.join(chunk_dir, "transcript_gt.json")
    vad_gt_path = os.path.join(chunk_dir, "vad_gt.json")
    info_path = os.path.join(chunk_dir, "sample_info.json")

    np.save(diar_path, diar_binary)
    with open(word_list_path, "w", encoding="utf-8") as fh:
        json.dump(word_list, fh, indent=2)
    with open(word_list_gt_path, "w", encoding="utf-8") as fh:
        json.dump(speaker_transcripts_gt, fh, indent=2)
    with open(vad_gt_path, "w", encoding="utf-8") as fh:
        json.dump(vad_gt, fh, indent=2)
    with open(info_path, "w", encoding="utf-8") as fh:
        json.dump(sample_info, fh, indent=2)

    return diar_path, word_list_path, word_list_gt_path, vad_gt_path, info_path


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode + per-dataset drivers
# ─────────────────────────────────────────────────────────────────────────────

def process_episode(
    sample: Dict,
    backend: BaseBackend,
    output_dir: str,
    dataset_name: str = DATASET_NAME,
) -> Tuple[int, int, int]:
    """Run the full chunk → infer → save loop for one episode.

    Sample dict keys consumed:
      ``conv_id``, ``audio`` (np.ndarray float32 @ SR), ``audio_path``,
      ``raw_transcript`` (List[{start, end, speaker, word, ...}]),
      optional ``txt_path`` / ``speakers`` (only used for the manifest / log).

    Returns ``(num_processed, num_skipped, num_fail)`` chunk counts.
    """
    conv_id = sample["conv_id"]
    save_dir = os.path.join(output_dir, conv_id)
    os.makedirs(save_dir, exist_ok=True)

    raw_audio: np.ndarray = sample["audio"]
    T = raw_audio.shape[0]
    raw_transcript: List[Dict] = sample["raw_transcript"]
    # CHUNK_MIN_DURATION, CHUNK_MAX_DURATION, CHUNK_GAP_THRESHOLD
    transcript_chunks = chunk_dialog(raw_transcript, min_dur=CHUNK_MIN_DURATION, max_dur=CHUNK_MAX_DURATION, gap_threshold=CHUNK_GAP_THRESHOLD)
    print(f"  Split into {len(transcript_chunks)} chunk(s)")

    num_processed = num_skipped = num_fail = 0

    for chunk_id, chunk in enumerate(transcript_chunks):
        start_sample = int(SR * chunk[0]["start"])
        end_sample = min(int(SR * chunk[-1]["end"]), T)
        chunk_audio = raw_audio[start_sample:end_sample]
        chunk_start_sec = float(start_sample) / SR

        chunk_dir = os.path.join(save_dir, f"CHUNK_{chunk_id}")
        diar_path = os.path.join(chunk_dir, "diart_pred.npy")
        word_list_path = os.path.join(chunk_dir, "transcript_pred.json")
        info_path = os.path.join(chunk_dir, "sample_info.json")

        if (
            os.path.exists(diar_path)
            and os.path.exists(word_list_path)
            and os.path.exists(info_path)
        ):
            print(f"  Skipping chunk {chunk_id} (already exists)")
            num_skipped += 1
            continue

        speaker_transcripts, vad_gt = build_speaker_transcripts(
            chunk, chunk_start_sec
        )

        try:
            word_list, diar_binary = backend.transcribe(
                chunk_audio, audio_file=sample.get("audio_path")
            )
        except Exception as exc:
            print(f"  Error processing chunk {chunk_id}: {exc}")
            num_fail += 1
            continue

        sample_info = {
            "dataset": dataset_name,
            "conv_id": conv_id,
            "chunk_id": chunk_id,
            "audio_file": sample.get("audio_path"),
            "txt_path": sample.get("txt_path"),
            "speakers": list(speaker_transcripts.keys()),
            "transcript_path": os.path.join(chunk_dir, "transcript_gt.json"),
            "vad_path": os.path.join(chunk_dir, "vad_gt.json"),
            "diart_path": diar_path,
            "pred_transcript_path": word_list_path,
            "feat_len_sec": FRAME_LEN_SEC,
            "time_stamp": [start_sample, end_sample],
            **backend.extra_manifest(),
        }

        save_chunk_outputs(
            chunk_dir=chunk_dir,
            diar_binary=diar_binary,
            word_list=word_list,
            speaker_transcripts_gt=speaker_transcripts,
            vad_gt=vad_gt,
            sample_info=sample_info,
        )

        print(f"  Saved: {diar_path}  shape={diar_binary.shape}")
        print(f"  Saved: {word_list_path}  ({len(word_list)} speaker(s))")
        num_processed += 1

    return num_processed, num_skipped, num_fail



def run_dataset(
    dataset: Iterable[Dict],
    backend: BaseBackend,
    output_dir: str,
    dataset_name: str = DATASET_NAME,
) -> Tuple[int, int, int]:
    """Iterate the dataset, run ``process_episode`` for each, and print a summary."""
    os.makedirs(output_dir, exist_ok=True)
    total_processed = total_skipped = total_fail = 0

    for sample in tqdm(dataset):
        conv_id = sample["conv_id"]
        if "Season01" not in conv_id:
            print("Skip!!!")
            break

        print(f"\n{'=' * 70}")
        print(f"Processing episode: {conv_id}")
        if "speakers" in sample:
            print(f"  Speakers : {sample['speakers']}")
        print(f"  Audio    : {sample.get('audio_path')}")
        print(f"{'=' * 70}")

        p, s, f = process_episode(
            sample=sample,
            backend=backend,
            output_dir=output_dir,
            dataset_name=dataset_name,
        )
        total_processed += p
        total_skipped += s
        total_fail += f

    print(
        f"\nStep 1 ({dataset_name}, {backend.name}) complete. "
        f"Processed {total_processed} chunks, "
        f"skipped {total_skipped}, "
        f"failed {total_fail}."
    )
    return total_processed, total_skipped, total_fail


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add I/O, device, chunking, and episode-filter arguments to ``parser``."""
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the dataset's audio + transcript files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory to save per-episode / per-chunk results.")

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        ),
        choices=["cuda", "cpu", "mps", "xpu", "auto"],
        help="Device to run inference on.",
    )
    return parser


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 1 (Bazinga): multi-talker diarization + ASR with "
                    "pluggable inference backends."
    )
    add_common_args(parser)
    add_backend_args(parser)
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = build_argparser().parse_args()

    dataset = BazingaDataset(args.data_dir, sample_rate=SR)
    print(f"Found {len(dataset)} episodes under {args.data_dir}")

    backend = build_backend(args)
    print(f"Using inference backend: {backend.name}")

    run_dataset(
        dataset=dataset,
        backend=backend,
        output_dir=args.output_dir,
        dataset_name=DATASET_NAME,
    )


if __name__ == "__main__":
    main()
