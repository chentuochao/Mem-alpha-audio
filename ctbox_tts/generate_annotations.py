#!/usr/bin/env python3
"""
Generate per-speaker ground-truth annotations for the dialogue-TTS output folders
produced by ``perltqa_dialogue_tts.py``.

Because this audio is *synthesized* from the PerLTQA dataset, the exact text and
the exact turn timing are already known — they live in each dialogue's
``channel_map.json`` (written by ``chatterbox_dialogue_tts.generate_multispeaker_
dialogue_tts``). So the **transcript** ground truth is taken directly from
channel_map.json instead of running ASR (WhisperX):

  * transcript = the PerLTQA source text of each turn (the exact words we asked
    Chatterbox to speak) — a cleaner reference than an ASR model's guess, with
    the exact per-turn ``start_sec`` / ``end_sec`` placement.

The **VAD** ground truth is still produced with Silero-VAD, run over each
speaker's isolated ``_TTS.npy`` track (this captures within-turn pauses that the
coarse turn windows do not).

Each dialogue folder looks like::

    <output>/<Profile>/<dialogue_id>/
        <SpeakerName>_TTS.npy            # isolated, timeline-aligned mono track
        dialogue_multichannel_TTS.wav
        dialogue_mono_TTS.wav
        channel_map.json                 # <-- turns + timing + text

For every speaker this script writes, in the same folder::

    <SpeakerName>_vad.npy          # int8 frame labels, one per 0.08s frame (Silero)
    <SpeakerName>_annotation.json  # turn-level transcript (channel_map) + Silero VAD

Examples
--------
    python generate_annotations.py --output-dir /path/to/perltqa/audio
    python generate_annotations.py --output-dir /path/to/audio --limit 5
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _safe_filename(name: str) -> str:
    """Mirror perltqa_dialogue_tts.safe_filename so we can rebuild npy names."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return name.strip("_") or "unnamed"


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
VAD_SR = 16000          # Silero-VAD operating sample rate
DEFAULT_FRAME_SEC = 0.08


# ----------------------------------------------------------------------------
# Silero-VAD (loaded once)
# ----------------------------------------------------------------------------
def load_silero_vad():
    import torch
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def vad_segments(
    wav_16k,                      # torch.Tensor, shape [T], float32, 16 kHz
    vad_model,
    get_speech_timestamps,
    threshold: float,
) -> List[Tuple[float, float]]:
    """Return list of (start_sec, end_sec) speech segments."""
    ts = get_speech_timestamps(
        wav_16k,
        vad_model,
        sampling_rate=VAD_SR,
        threshold=threshold,
    )
    return [(t["start"] / VAD_SR, t["end"] / VAD_SR) for t in ts]


def rasterize_frames(
    segments: List[Tuple[float, float]],
    duration_sec: float,
    frame_sec: float,
) -> np.ndarray:
    """
    Convert speech segments to a frame-level binary label array.
    Frame i covers [i*frame_sec, (i+1)*frame_sec); it is 1 if any speech
    overlaps it.
    """
    num_frames = max(1, int(np.ceil(duration_sec / frame_sec)))
    frames = np.zeros(num_frames, dtype=np.int8)
    for start, end in segments:
        i0 = int(np.floor(start / frame_sec))
        i1 = int(np.ceil(end / frame_sec))
        frames[max(0, i0):min(num_frames, i1)] = 1
    return frames


# ----------------------------------------------------------------------------
# channel_map.json -> per-speaker turn segments (transcript GT)
# ----------------------------------------------------------------------------
def turns_by_speaker(meta: Dict) -> Dict[str, List[Dict]]:
    """
    Group a dialogue's turns by speaker name.

    Returns {speaker_name: [{"speaker", "start", "end", "text"}, ...]} in
    chronological order, using the absolute ``start_sec``/``end_sec`` timeline
    placement recorded in channel_map.json.
    """
    by_spk: Dict[str, List[Dict]] = {}
    for t in meta.get("turns", []) or []:
        name = t.get("speaker_name")
        text = (t.get("text") or "").strip()
        if not name or not text:
            continue
        start = t.get("start_sec")
        end = t.get("end_sec")
        if start is None or end is None:
            # fall back to samples / sample_rate if *_sec is missing
            sr = int(meta.get("sample_rate", 24000)) or 24000
            start = (t.get("start_sample", 0)) / sr
            end = (t.get("end_sample", 0)) / sr
        by_spk.setdefault(name, []).append({
            "speaker": name,
            "start": float(start),
            "end": float(end),
            "text": text,
        })
    for name in by_spk:
        by_spk[name].sort(key=lambda s: (s["start"], s["end"]))
    return by_spk


# ----------------------------------------------------------------------------
# Per-speaker annotation
# ----------------------------------------------------------------------------
def annotate_speaker(
    npy_path: Path,
    speaker_name: str,
    transcript_segments: List[Dict],
    sample_rate: int,
    frame_sec: float,
    vad_model,
    get_speech_timestamps,
    vad_threshold: float,
    vad_frames_file: str,
) -> Tuple[dict, np.ndarray]:
    """Build one speaker's annotation: Silero VAD (from audio) + channel_map text."""
    import torch
    import torchaudio

    arr = np.load(npy_path).astype(np.float32)
    arr = np.squeeze(arr)
    wav = torch.from_numpy(arr).unsqueeze(0)  # [1, T]
    sr = int(sample_rate)
    duration_sec = arr.shape[-1] / sr

    # ---- VAD ground truth (Silero over the isolated track) ----
    vad_wav = wav
    if sr != VAD_SR:
        vad_wav = torchaudio.functional.resample(vad_wav, sr, VAD_SR)
    segments = vad_segments(
        vad_wav.squeeze(0), vad_model, get_speech_timestamps, vad_threshold
    )
    frames = rasterize_frames(segments, duration_sec, frame_sec)

    # ---- Transcript ground truth (turn text/timing from channel_map) ----
    full_text = " ".join(s["text"] for s in transcript_segments).strip()

    ann = {
        "speaker": speaker_name,
        "source_npy": npy_path.name,
        "sample_rate": sr,
        "duration_sec": float(duration_sec),
        "vad": {
            "frame_sec": frame_sec,
            "num_frames": int(frames.shape[0]),
            "speech_frame_ratio": float(frames.mean()) if frames.size else 0.0,
            "segments": [[float(s), float(e)] for s, e in segments],
            "frames_file": vad_frames_file,
            "threshold": vad_threshold,
            "source": "silero_vad",
        },
        "transcript": {
            "text": full_text,
            # turn-level GT segments (each carries `text`); no word-level
            # timing / forced alignment — see module docstring.
            "segments": transcript_segments,
            "source": "perltqa_channel_map",
        },
    }
    return ann, frames


# ----------------------------------------------------------------------------
# Folder discovery
# ----------------------------------------------------------------------------
def find_dialogue_folders(output_dir: Path) -> List[Path]:
    """All folders that contain a channel_map.json (a generated dialogue)."""
    return sorted({p.parent for p in output_dir.rglob("channel_map.json")})


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="root dir holding <Profile>/<dialogue_id>/ folders")
    ap.add_argument("--frame-size", type=float, default=DEFAULT_FRAME_SEC,
                    help="VAD frame size in seconds (default 0.08)")
    ap.add_argument("--vad-threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0,
                    help="max dialogue folders to process; <=0 means all")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-annotate speakers whose annotation already exists")
    args = ap.parse_args()

    folders = find_dialogue_folders(args.output_dir)
    if args.limit > 0:
        folders = folders[: args.limit]
    if not folders:
        print(f"[annotate] no dialogue folders (channel_map.json) under "
              f"{args.output_dir}", file=sys.stderr)
        return
    print(f"[annotate] {len(folders)} dialogue folder(s) under {args.output_dir}")

    # load Silero-VAD once
    print("[annotate] loading Silero-VAD ...")
    vad_model, get_speech_timestamps = load_silero_vad()

    ok, skipped, failed = 0, 0, 0
    for fi, folder in enumerate(folders):
        cmap = folder / "channel_map.json"
        try:
            meta = json.loads(cmap.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[annotate][FAIL] {folder}: bad channel_map.json: {e}",
                  file=sys.stderr)
            failed += 1
            continue

        sr = int(meta.get("sample_rate", 24000)) or 24000
        by_spk = turns_by_speaker(meta)
        if not by_spk:
            continue

        rel = folder.relative_to(args.output_dir)
        print(f"[annotate] ({fi + 1}/{len(folders)}) {rel} "
              f"-> {len(by_spk)} speaker(s)")

        for name, segments in by_spk.items():
            safe = _safe_filename(name)
            npy_path = folder / f"{safe}_TTS.npy"
            ann_path = folder / f"{safe}_annotation.json"
            vad_npy_path = folder / f"{safe}_vad.npy"
            if ann_path.exists() and not args.overwrite:
                skipped += 1
                continue
            if not npy_path.exists():
                print(f"[annotate][skip] {rel} / {name}: missing {npy_path.name}",
                      file=sys.stderr)
                skipped += 1
                continue
            try:
                ann, frames = annotate_speaker(
                    npy_path=npy_path,
                    speaker_name=name,
                    transcript_segments=segments,
                    sample_rate=sr,
                    frame_sec=args.frame_size,
                    vad_model=vad_model,
                    get_speech_timestamps=get_speech_timestamps,
                    vad_threshold=args.vad_threshold,
                    vad_frames_file=vad_npy_path.name,
                )
                np.save(vad_npy_path, frames)
                ann_path.write_text(
                    json.dumps(ann, ensure_ascii=False, indent=2),
                    encoding="utf-8")
                ok += 1
                print(f"[annotate]    {name}: "
                      f"{len(ann['vad']['segments'])} vad seg, "
                      f"{ann['vad']['num_frames']} frames, "
                      f"{len(ann['transcript']['segments'])} turn(s)")
            except Exception as e:
                failed += 1
                print(f"[annotate][FAIL] {rel} / {name}: {e}", file=sys.stderr)

    print(f"[annotate] done. ok={ok} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
