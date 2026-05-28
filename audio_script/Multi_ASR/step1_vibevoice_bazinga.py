"""
Step 1 (VibeVoice + Bazinga): Multi-talker diarization + ASR using the VibeVoice
model on the Bazinga (Friends TV show) dataset.

Input: Bazinga data directory with the flat structure:
  data_dir/
    Friends.Season01.Episode01.en.wav
    Friends.Season01.Episode01.txt
    Friends.Season01.Episode02.en.wav
    Friends.Season01.Episode02.txt
    ...

Each episode is split into time-based chunks (1–10 min) at natural silence gaps.
For each chunk, VibeVoice ASR inference is run and the following are saved:
  - diart_pred.npy        (binary diarization matrix, num_frames x num_speakers)
  - transcript_pred.json  (per-speaker word-level predictions)
  - transcript_gt.json    (ground-truth per-speaker word lists for this chunk)
  - vad_gt.json           (ground-truth VAD intervals per speaker)
  - sample_info.json      (manifest entry for Step 2 / evaluation)

Output layout mirrors step1_diarize_asr_bazinga.py so that downstream evaluation
code can be used without modification.
"""

import argparse
import json
import os
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from audio_script.datasets.Bazinga_loader import BazingaDataset
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from prepare_data.preprocess_utils import chunk_dialog, transcription_to_vad


SR = 16000
FRAME_DURATION = 0.08  # seconds per diarization frame

# ─── VibeVoice model wrapper (from step1_vibevoice.py) ────────────────────────

class VibeVoiceInference:
    """Thin wrapper around VibeVoice ASR for single-file inference."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
    ):
        print(f"Loading VibeVoice ASR model from {model_path}")
        self.processor = VibeVoiceASRProcessor.from_pretrained(
            model_path,
            language_model_pretrained_name="Qwen/Qwen2.5-7B",
        )
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device if device == "auto" else None,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        if device != "auto":
            self.model = self.model.to(device)
        self.device = device if device != "auto" else next(self.model.parameters()).device
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def transcribe(
        self,
        audio_path: str,
        max_new_tokens: int = 32768,
        temperature: float = 0.0,
        top_p: float = 1.0,
        num_beams: int = 1,
    ) -> List[Dict]:
        """
        Transcribe a single audio file.

        Returns:
            List of segment dicts with keys:
                start_time (float), end_time (float),
                speaker_id (str), text (str)
        """
        do_sample = temperature > 0

        inputs = self.processor(
            audio=[audio_path],
            sampling_rate=None,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        )
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        gen_cfg: Dict = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.processor.pad_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        if num_beams > 1:
            gen_cfg["num_beams"] = num_beams
            gen_cfg["do_sample"] = False
        else:
            gen_cfg["do_sample"] = do_sample
            if do_sample:
                gen_cfg["temperature"] = temperature
                gen_cfg["top_p"] = top_p

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_cfg)

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_length:]

        eos_positions = (
            generated_ids == self.processor.tokenizer.eos_token_id
        ).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            generated_ids = generated_ids[: eos_positions[0] + 1]

        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        try:
            segments = self.processor.post_process_transcription(generated_text)
        except Exception as exc:
            print(f"  Warning: failed to parse structured output: {exc}")
            segments = []

        return segments


# ─── Output conversion (from step1_vibevoice.py) ──────────────────────────────

def segments_to_word_list(segments: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Convert VibeVoice segment list to the word_list format expected by step2.

    Output format:
        { "speaker_0": [{"word": str, "start": float, "end": float}, ...], ... }

    Segments with speaker_id == "N/A" are skipped.
    """
    word_list: Dict[str, List[Dict]] = {}
    for seg in segments:
        spk_id = seg.get("speaker_id", "N/A")
        if spk_id == "N/A":
            continue
        key = f"speaker_{spk_id}"
        entry = {
            "word": seg.get("text", "").strip(),
            "start": float(seg.get("start_time", 0.0)),
            "end": float(seg.get("end_time", 0.0)),
            "score": 1.0,
        }
        word_list.setdefault(key, []).append(entry)
    return word_list


def segments_to_diar_matrix(
    segments: List[Dict],
    frame_duration: float = FRAME_DURATION,
) -> np.ndarray:
    """
    Convert VibeVoice segment list to a binary diarization matrix.

    Returns:
        np.ndarray of shape (num_frames, num_speakers), dtype bool
        Speakers are assigned sequential integer indices in order of first
        appearance (N/A segments are ignored).
    """
    speaker_order: List[str] = []
    seen: set = set()
    for seg in segments:
        spk_id = seg.get("speaker_id", "N/A")
        if spk_id == "N/A":
            continue
        if spk_id not in seen:
            speaker_order.append(spk_id)
            seen.add(spk_id)

    if not speaker_order:
        return np.zeros((1, 1), dtype=bool)

    total_end = max(
        float(seg.get("end_time", 0.0))
        for seg in segments
        if seg.get("speaker_id", "N/A") != "N/A"
    )
    num_frames = max(1, int(np.ceil(total_end / frame_duration)))
    num_speakers = len(speaker_order)
    spk_to_col = {spk: i for i, spk in enumerate(speaker_order)}

    diar = np.zeros((num_frames, num_speakers), dtype=bool)
    for seg in segments:
        spk_id = seg.get("speaker_id", "N/A")
        if spk_id == "N/A":
            continue
        col = spk_to_col[spk_id]
        start_frame = int(float(seg.get("start_time", 0.0)) / frame_duration)
        end_frame = int(np.ceil(float(seg.get("end_time", 0.0)) / frame_duration))
        end_frame = min(end_frame, num_frames)
        diar[start_frame:end_frame, col] = True

    return diar


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 1 (VibeVoice + Bazinga): Run multi-talker diarization "
                    "+ ASR with VibeVoice on the Bazinga/Friends dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing Friends *.en.wav and *.txt episode files",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the VibeVoice model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory to save per-episode / per-chunk results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        ),
        choices=["cuda", "cpu", "mps", "xpu", "auto"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="Maximum number of tokens to generate per audio chunk",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy decoding)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p for nucleus sampling",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for beam search (1 = greedy/sampling)",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["flash_attention_2", "sdpa", "eager", "auto"],
        help="Attention implementation ('auto' selects best for device)",
    )
    args = parser.parse_args()

    # Auto-select attention implementation
    if args.attn_implementation == "auto":
        if args.device == "cuda" and torch.cuda.is_available():
            try:
                import flash_attn  # noqa: F401
                args.attn_implementation = "flash_attention_2"
            except ImportError:
                args.attn_implementation = "sdpa"
        else:
            args.attn_implementation = "sdpa"
        print(f"Attention implementation: {args.attn_implementation}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────
    dataset = BazingaDataset(args.data_dir, sample_rate=SR)
    print(f"Found {len(dataset)} episodes under {args.data_dir}")

    # ── Determine dtype ───────────────────────────────────────────────
    dtype = torch.float32 if args.device in ("mps", "xpu", "cpu") else torch.bfloat16

    # ── Load model ────────────────────────────────────────────────────
    model = VibeVoiceInference(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )

    # ── Process each episode ──────────────────────────────────────────
    num_processed = 0
    num_skipped = 0
    num_fail = 0

    for sample in tqdm(dataset):
        conv_id = sample["conv_id"]
        print(f"\n{'=' * 70}")
        print(f"Processing episode: {conv_id}")
        print(f"  Speakers : {sample['speakers']}")
        print(f"  Audio    : {sample['audio_path']}")
        print(f"{'=' * 70}")
        if "Season01" not in conv_id:
            print("Skip!!!")
            break
        save_dir = os.path.join(args.output_dir, conv_id)
        os.makedirs(save_dir, exist_ok=True)

        raw_audio: np.ndarray = sample["audio"]   # shape (T,), float32 @ SR
        T = raw_audio.shape[0]
        raw_transcript: List[Dict] = sample["raw_transcript"]

        transcript_chunks = chunk_dialog(raw_transcript, min_dur=60.0, max_dur=300.0, gap_threshold=3.0)
        print(f"  Split into {len(transcript_chunks)} chunk(s)")

        for chunk_id, chunk in enumerate(transcript_chunks):
            start_sample = int(SR * chunk[0]["start"])
            end_sample = int(SR * chunk[-1]["end"])
            end_sample = min(end_sample, T)
            chunk_audio = raw_audio[start_sample:end_sample]

            chunk_start_sec = float(start_sample) / SR
            chunk_end_sec = float(end_sample) / SR

            chunk_dir = os.path.join(save_dir, f"CHUNK_{chunk_id}")
            diar_path = os.path.join(chunk_dir, "diart_pred.npy")
            word_list_path = os.path.join(chunk_dir, "transcript_pred.json")
            word_list_gt_path = os.path.join(chunk_dir, "transcript_gt.json")
            vad_gt_path = os.path.join(chunk_dir, "vad_gt.json")
            info_path = os.path.join(chunk_dir, "sample_info.json")

            if (
                os.path.exists(diar_path)
                and os.path.exists(word_list_path)
                and os.path.exists(info_path)
            ):
                print(f"  Skipping chunk {chunk_id} (already exists)")
                num_skipped += 1
                continue

            # Build ground-truth speaker transcripts for this chunk
            # (time-shift word timestamps to be chunk-relative)
            speaker_transcripts: Dict[str, List[Dict]] = defaultdict(list)
            for w in chunk:
                w_shifted = dict(w)
                w_shifted["start"] = w["start"] - chunk_start_sec
                w_shifted["end"] = w["end"] - chunk_start_sec
                speaker_transcripts[w["speaker"]].append(w_shifted)
            vad_gt = transcription_to_vad(speaker_transcripts)

            # Write chunk audio to a temp file so VibeVoice can read it
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_audio_path = tmp.name
            try:
                sf.write(tmp_audio_path, chunk_audio, SR)

                segments = model.transcribe(
                    tmp_audio_path,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                )
            except Exception as exc:
                print(f"  Error processing chunk {chunk_id}: {exc}")
                num_fail += 1
                continue
            finally:
                if os.path.exists(tmp_audio_path):
                    os.remove(tmp_audio_path)

            # Print preview
            print(f"  Chunk {chunk_id}: got {len(segments)} segment(s) from VibeVoice")
            for seg in segments[:3]:
                spk = seg.get("speaker_id", "N/A")
                start = seg.get("start_time", 0.0)
                end = seg.get("end_time", 0.0)
                text = seg.get("text", "")[:60]
                print(f"    [{start:.1f} - {end:.1f}] Speaker {spk}: {text}...")
            if len(segments) > 3:
                print(f"    ... and {len(segments) - 3} more segment(s)")

            # Convert to step1-compatible outputs
            word_list = segments_to_word_list(segments)
            diar_result = segments_to_diar_matrix(segments, frame_duration=FRAME_DURATION)

            os.makedirs(chunk_dir, exist_ok=True)
            np.save(diar_path, diar_result)

            with open(word_list_path, "w", encoding="utf-8") as fh:
                json.dump(word_list, fh, indent=2)
            with open(word_list_gt_path, "w", encoding="utf-8") as fh:
                json.dump(dict(speaker_transcripts), fh, indent=2)
            with open(vad_gt_path, "w", encoding="utf-8") as fh:
                json.dump(vad_gt, fh, indent=2)

            sample_info = {
                "dataset": "bazinga",
                "conv_id": conv_id,
                "chunk_id": chunk_id,
                "audio_file": sample["audio_path"],
                "txt_path": sample["txt_path"],
                "speakers": list(speaker_transcripts.keys()),
                "transcript_path": word_list_gt_path,
                "vad_path": vad_gt_path,
                "diart_path": diar_path,
                "pred_transcript_path": word_list_path,
                "feat_len_sec": FRAME_DURATION,
                "time_stamp": [start_sample, end_sample],
            }
            with open(info_path, "w", encoding="utf-8") as fh:
                json.dump(sample_info, fh, indent=2)

            print(f"  Saved: {diar_path}  shape={diar_result.shape}")
            print(f"  Saved: {word_list_path}  ({len(word_list)} speaker(s))")
            print(f"  Saved: {info_path}")
            num_processed += 1

    print(
        f"\nStep 1 (VibeVoice + Bazinga) complete. "
        f"Processed {num_processed} chunks, "
        f"skipped {num_skipped}, "
        f"failed {num_fail}."
    )


if __name__ == "__main__":
    main()
