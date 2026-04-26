"""
Step 1 (VibeVoice): Multi-talker diarization + ASR using the VibeVoice model.

Input: data directory produced by mix_interact.py with the structure:
  data_dir/
    {spk1}_{spk2}/
      {conv_id}/
        mixed_conv.wav
        transcript1.json  transcript2.json
        vad1.json         vad2.json

For each conversation, run VibeVoice ASR inference and save:
  - diart_pred.npy       (binary diarization matrix, num_frames x num_speakers)
  - transcript_pred.json (per-speaker word-level predictions)
  - sample_info.json     (metadata; compatible with step2_speaker_match.py)

The output format is identical to step1_diarize_asr.py so that step2 and
evaluation code can be used without modification.
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


FRAME_DURATION = 0.08  # seconds per diarization frame (matches step1_diarize_asr.py)


# ─── Model wrapper ────────────────────────────────────────────────────

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


# ─── Output conversion ────────────────────────────────────────────────

def segments_to_word_list(
    segments: List[Dict],
) -> Dict[str, List[Dict]]:
    """
    Convert VibeVoice segment list to the word_list format expected by step2.

    VibeVoice segment keys: start_time, end_time, speaker_id, text
    Output format:
        {
            "speaker_0": [{"word": "...", "start": 0.0, "end": 1.0}, ...],
            ...
        }

    Segments with speaker_id == "N/A" (environmental sounds / noise) are
    skipped because they have no real speaker assignment.
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
            "score": 1.0,  # placeholder
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
    # Collect unique speaker ids (excluding N/A) in order of appearance
    speaker_order: List[str] = []
    seen = set()
    for seg in segments:
        spk_id = seg.get("speaker_id", "N/A")
        if spk_id == "N/A":
            continue
        if spk_id not in seen:
            speaker_order.append(spk_id)
            seen.add(spk_id)

    if not speaker_order:
        # No speakers found → return a single silent frame
        return np.zeros((1, 1), dtype=bool)

    # Determine total duration
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


# ─── Conversation discovery (identical to step1_diarize_asr.py) ───────

def discover_conversations(data_dir: str) -> List[Dict]:
    """
    Walk the directory tree produced by mix_interact.py and return a list of
    conversation dicts.
    """
    conversations = []
    for spk_pair_dir in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not os.path.isdir(spk_pair_dir):
            continue
        spk_pair = os.path.basename(spk_pair_dir)
        for conv_dir in sorted(glob.glob(os.path.join(spk_pair_dir, "*"))):
            if not os.path.isdir(conv_dir):
                continue
            audio_path = os.path.join(conv_dir, "mixed_conv.wav")
            if not os.path.exists(audio_path):
                continue
            conv_id = os.path.basename(conv_dir)
            conversations.append({
                "spk_pair": spk_pair,
                "conv_id": conv_id,
                "conv_dir": conv_dir,
                "audio_path": audio_path,
                "transcript1_path": os.path.join(conv_dir, "transcript1.json"),
                "transcript2_path": os.path.join(conv_dir, "transcript2.json"),
                "vad1_path": os.path.join(conv_dir, "vad1.json"),
                "vad2_path": os.path.join(conv_dir, "vad2.json"),
            })
    return conversations


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 1 (VibeVoice): Run multi-talker diarization + ASR"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory produced by mix_interact.py "
             "(contains {spk_pair}/{conv_id}/ sub-folders)",
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
        default=None,
        help="Directory to save results. Defaults to writing inside each "
             "conversation folder under data_dir.",
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
        help="Maximum number of tokens to generate per audio file",
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

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # ── Discover conversations ────────────────────────────────────────
    conversations = discover_conversations(args.data_dir)
    print(f"Found {len(conversations)} conversations under {args.data_dir}")
    if not conversations:
        print("No conversations found. Check your --data_dir path.")
        return

    # ── Determine dtype ───────────────────────────────────────────────
    dtype = torch.float32 if args.device in ("mps", "xpu", "cpu") else torch.bfloat16

    # ── Load model ────────────────────────────────────────────────────
    model = VibeVoiceInference(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )

    # ── Process each conversation ─────────────────────────────────────
    num_processed = 0
    num_skipped = 0

    for conv in tqdm(conversations):
        print(f"\n{'=' * 70}")
        print(f"Processing: {conv['spk_pair']} / {conv['conv_id']}")
        print(f"  Audio: {conv['audio_path']}")
        print(f"{'=' * 70}")

        if num_processed > 10:
            exit(0)

        if args.output_dir is not None:
            save_dir = os.path.join(
                args.output_dir, conv["spk_pair"], conv["conv_id"]
            )
        else:
            save_dir = conv["conv_dir"]

        diar_path = os.path.join(save_dir, "diart_pred.npy")
        word_list_path = os.path.join(save_dir, "transcript_pred.json")
        info_path = os.path.join(save_dir, "sample_info.json")

        if (
            os.path.exists(diar_path)
            and os.path.exists(word_list_path)
            and os.path.exists(info_path)
        ):
            print(f"  Skipping (already exists): {save_dir}")
            num_skipped += 1
            continue

        try:
            segments = model.transcribe(
                conv["audio_path"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
            )
        except Exception as exc:
            print(f"  Error during transcription: {exc}")
            continue

        # Print preview
        print(f"  Got {len(segments)} segment(s) from VibeVoice:")
        for seg in segments[:5]:
            spk = seg.get("speaker_id", "N/A")
            start = seg.get("start_time", 0.0)
            end = seg.get("end_time", 0.0)
            text = seg.get("text", "")[:80]
            print(f"    [{start} - {end}] Speaker {spk}: {text}...")
        if len(segments) > 5:
            print(f"    ... and {len(segments) - 5} more segment(s)")
        # Convert to step1-compatible outputs
        word_list = segments_to_word_list(segments)
        diar_result = segments_to_diar_matrix(segments, frame_duration=FRAME_DURATION)

        os.makedirs(save_dir, exist_ok=True)
        np.save(diar_path, diar_result)
        with open(word_list_path, "w") as f:
            json.dump(word_list, f, indent=2)

        sample_info = {
            "spk_pair": conv["spk_pair"],
            "conv_id": conv["conv_id"],
            "audio_file": conv["audio_path"],
            "transcript1_path": conv["transcript1_path"],
            "transcript2_path": conv["transcript2_path"],
            "vad1_path": conv["vad1_path"],
            "vad2_path": conv["vad2_path"],
            "diart_path": diar_path,
            "pred_transcript_path": word_list_path,
            "feat_len_sec": FRAME_DURATION,
        }
        with open(info_path, "w") as f:
            json.dump(sample_info, f, indent=2)

        print(f"  Saved: {diar_path}  shape={diar_result.shape}")
        print(f"  Saved: {word_list_path}  ({len(word_list)} speaker(s))")
        print(f"  Saved: {info_path}")
        num_processed += 1
    print(
        f"\nStep 1 (VibeVoice) complete. "
        f"Processed {num_processed}, skipped {num_skipped} conversations."
    )


if __name__ == "__main__":
    main()
