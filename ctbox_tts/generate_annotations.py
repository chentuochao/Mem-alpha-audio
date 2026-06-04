#!/usr/bin/env python3
"""
Generate per-speaker annotations (VAD ground-truth + ASR transcript) for the
dialogue-TTS output folders produced by perltqa_dialogue_tts.py.

Each dialogue folder looks like:
    <output>/<Profile>/<dialogue_id>/
        <SpeakerName>_TTS.npy            # isolated, timeline-aligned mono track
        dialogue_multichannel_TTS.wav
        dialogue_mono_TTS.wav
        channel_map.json

For every <SpeakerName>_TTS.npy this script writes, in the same folder:
    <SpeakerName>_vad.npy          # int8 frame labels, one per 0.08s frame
    <SpeakerName>_annotation.json  # vad segments + frames meta + ASR transcript

VAD
---
Silero-VAD (torch.hub `snakers4/silero-vad`) detects speech timestamps; those
are rasterised to a fixed **0.08 s** frame grid (configurable via --frame-size).

ASR
---
WhisperX transcribes + word-aligns each speaker's isolated track.

Examples
--------
    python generate_annotations.py --output-dir tts_outputs/perltqa
    python generate_annotations.py --output-dir /path/to/audio --limit 5
    python generate_annotations.py --output-dir tts_outputs/perltqa \
        --whisper-model large-v2 --language en --device cuda
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
ASR_SR = 16000          # WhisperX expects 16 kHz mono
DEFAULT_FRAME_SEC = 0.08


# ----------------------------------------------------------------------------
# Model loading (lazy / once)
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


def load_whisperx(device: str, compute_type: str, model_name: str, language: str):
    import whisperx
    asr_model = whisperx.load_model(
        model_name, device, compute_type=compute_type, language=language
    )
    align_model, align_meta = whisperx.load_align_model(
        language_code=language, device=device
    )
    return whisperx, asr_model, align_model, align_meta


# ----------------------------------------------------------------------------
# VAD
# ----------------------------------------------------------------------------
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
# Per-speaker annotation
# ----------------------------------------------------------------------------
def annotate_speaker(
    npy_path: Path,
    speaker_name: str,
    sample_rate: int,
    frame_sec: float,
    vad_model,
    get_speech_timestamps,
    vad_threshold: float,
    whisperx,
    whisperx_model,
    whisperx_align_model,
    whisperx_align_metadata,
    device: str,
    batch_size: int,
) -> dict:
    import torch
    import torchaudio

    arr = np.load(npy_path).astype(np.float32)
    arr = np.squeeze(arr)
    wav = torch.from_numpy(arr).unsqueeze(0)  # [1, T]
    sr = int(sample_rate)
    duration_sec = arr.shape[-1] / sr

    # ---- VAD ground truth ----
    vad_wav = wav
    if sr != VAD_SR:
        vad_wav = torchaudio.functional.resample(vad_wav, sr, VAD_SR)
    segments = vad_segments(
        vad_wav.squeeze(0), vad_model, get_speech_timestamps, vad_threshold
    )
    frames = rasterize_frames(segments, duration_sec, frame_sec)

    # ---- Transcript ----
    asr_wav = wav.clone()
    if sr != ASR_SR:
        asr_wav = torchaudio.functional.resample(asr_wav, sr, ASR_SR)
    asr_wav = asr_wav.squeeze(0).numpy()
    transcript = whisperx_model.transcribe(asr_wav, batch_size=batch_size)
    transcript = whisperx.align(
        transcript["segments"], whisperx_align_model, whisperx_align_metadata,
        asr_wav, device, return_char_alignments=False)
    transcript = transcript["segments"]

    # save the frame labels alongside the json
    vad_npy_path = npy_path.with_name(npy_path.stem.replace("_TTS", "") + "_vad.npy")
    np.save(vad_npy_path, frames)

    full_text = " ".join(
        seg.get("text", "").strip() for seg in transcript
    ).strip()

    # Flat, word-level list (timestamps absolute within this dialogue audio).
    # This is the compatibility hook consumed by step1_perltqa.py's loader:
    # it mirrors the per-word entries the shared Multi_ASR pipeline expects.
    words: List[dict] = []
    for seg in transcript:
        for w in seg.get("words", []) or []:
            if w.get("start") is None or w.get("end") is None:
                continue  # whisperx occasionally drops alignment for a token
            words.append({
                "speaker": speaker_name,
                "word": w.get("word", ""),
                "start": float(w["start"]),
                "end": float(w["end"]),
                "score": float(w.get("score", 1.0)),
            })

    return {
        "speaker": speaker_name,
        "source_npy": npy_path.name,
        "sample_rate": sr,
        "duration_sec": float(duration_sec),
        "vad": {
            "frame_sec": frame_sec,
            "num_frames": int(frames.shape[0]),
            "speech_frame_ratio": float(frames.mean()) if frames.size else 0.0,
            "segments": [[float(s), float(e)] for s, e in segments],
            "frames_file": vad_npy_path.name,
            "threshold": vad_threshold,
        },
        "transcript": {
            "text": full_text,
            "words": words,
            "segments": transcript,
        },
    }


# ----------------------------------------------------------------------------
# Folder discovery
# ----------------------------------------------------------------------------
def find_dialogue_folders(output_dir: Path) -> List[Path]:
    """All folders that contain a channel_map.json (a generated dialogue)."""
    return sorted({p.parent for p in output_dir.rglob("channel_map.json")})


def speakers_in_folder(folder: Path) -> Dict[str, Path]:
    """
    Map real speaker name -> its <Speaker>_TTS.npy path.
    Prefers channel_map.json (keeps original casing); falls back to globbing.
    """
    cmap = folder / "channel_map.json"
    result: Dict[str, Path] = {}
    if cmap.exists():
        try:
            meta = json.loads(cmap.read_text(encoding="utf-8"))
            for name, info in (meta.get("channel_map", {}) or {}).items():
                npy = info.get("npy_file")
                path = folder / Path(npy).name if npy else None
                if path is None or not path.exists():
                    # reconstruct the expected filename from the speaker name
                    path = folder / f"{_safe_filename(name)}_TTS.npy"
                if path.exists():
                    result[name] = path
            if result:
                return result
        except Exception:
            pass
    # fallback: glob, speaker name derived from filename (safe-name only)
    for p in sorted(folder.glob("*_TTS.npy")):
        name = p.stem[: -len("_TTS")] if p.stem.endswith("_TTS") else p.stem
        result[name] = p
    return result


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
    ap.add_argument("--whisper-model", default="large-v2")
    ap.add_argument("--language", default="en")
    ap.add_argument("--device", default=None,
                    help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--compute-type", default=None,
                    help="whisperx compute type (default: float16 on cuda, int8 on cpu)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help="max dialogue folders to process; <=0 means all")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-annotate speakers whose annotation already exists")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = args.compute_type or ("float16" if device == "cuda" else "int8")

    folders = find_dialogue_folders(args.output_dir)
    if args.limit > 0:
        folders = folders[: args.limit]
    if not folders:
        print(f"[annotate] no dialogue folders (channel_map.json) under "
              f"{args.output_dir}", file=sys.stderr)
        return
    print(f"[annotate] {len(folders)} dialogue folder(s) under {args.output_dir}")

    # load models once
    print("[annotate] loading Silero-VAD ...")
    vad_model, get_speech_timestamps = load_silero_vad()
    print(f"[annotate] loading WhisperX ({args.whisper_model}, {device}, "
          f"{compute_type}) ...")
    whisperx, whisperx_model, align_model, align_meta = load_whisperx(
        device, compute_type, args.whisper_model, args.language
    )

    ok, skipped, failed = 0, 0, 0
    for fi, folder in enumerate(folders):
        # sample_rate from channel_map.json (fallback 24000)
        sr = 24000
        cmap = folder / "channel_map.json"
        if cmap.exists():
            try:
                sr = int(json.loads(cmap.read_text(encoding="utf-8"))
                         .get("sample_rate", sr))
            except Exception:
                pass

        speakers = speakers_in_folder(folder)
        if not speakers:
            continue
        rel = folder.relative_to(args.output_dir)
        print(f"[annotate] ({fi + 1}/{len(folders)}) {rel} "
              f"-> {len(speakers)} speaker(s)")

        for name, npy_path in speakers.items():
            ann_path = npy_path.with_name(
                npy_path.stem.replace("_TTS", "") + "_annotation.json")
            if ann_path.exists() and not args.overwrite:
                skipped += 1
                continue
            try:
                ann = annotate_speaker(
                    npy_path=npy_path,
                    speaker_name=name,
                    sample_rate=sr,
                    frame_sec=args.frame_size,
                    vad_model=vad_model,
                    get_speech_timestamps=get_speech_timestamps,
                    vad_threshold=args.vad_threshold,
                    whisperx=whisperx,
                    whisperx_model=whisperx_model,
                    whisperx_align_model=align_model,
                    whisperx_align_metadata=align_meta,
                    device=device,
                    batch_size=args.batch_size,
                )
                ann_path.write_text(
                    json.dumps(ann, ensure_ascii=False, indent=2),
                    encoding="utf-8")
                ok += 1
                print(f"[annotate]    {name}: "
                      f"{len(ann['vad']['segments'])} vad seg, "
                      f"{ann['vad']['num_frames']} frames, "
                      f"{len(ann['transcript']['segments'])} asr seg")
            except Exception as e:
                failed += 1
                print(f"[annotate][FAIL] {rel} / {name}: {e}", file=sys.stderr)

    print(f"[annotate] done. ok={ok} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
