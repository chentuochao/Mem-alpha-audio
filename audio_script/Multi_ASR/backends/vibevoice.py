"""VibeVoice end-to-end multi-talker ASR backend.

The model directly produces speaker-tagged segments — we convert those to the
unified ``word_list`` dict and a binary diarization matrix.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch

from ..constants import FRAME_LEN_SEC, SR
from .base import BaseBackend


class VibeVoiceBackend(BaseBackend):
    """End-to-end multi-talker diarization + ASR via the VibeVoice model."""

    name = "vibevoice"

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 32768,
        temperature: float = 0.0,
        top_p: float = 1.0,
        num_beams: int = 1,
    ):
        from vibevoice.modular.modeling_vibevoice_asr import (
            VibeVoiceASRForConditionalGeneration,
        )
        from vibevoice.processor.vibevoice_asr_processor import (
            VibeVoiceASRProcessor,
        )

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
        self.device = (
            device if device != "auto" else next(self.model.parameters()).device
        )
        self.model.eval()
        print(f"Model loaded on {self.device}")

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams

    def _generate_segments(self, audio_path: str) -> List[Dict]:
        do_sample = self.temperature > 0

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
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.processor.pad_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        if self.num_beams > 1:
            gen_cfg["num_beams"] = self.num_beams
            gen_cfg["do_sample"] = False
        else:
            gen_cfg["do_sample"] = do_sample
            if do_sample:
                gen_cfg["temperature"] = self.temperature
                gen_cfg["top_p"] = self.top_p

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_cfg)

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_length:]

        eos_positions = (
            generated_ids == self.processor.tokenizer.eos_token_id
        ).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            generated_ids = generated_ids[: eos_positions[0] + 1]

        generated_text = self.processor.decode(
            generated_ids, skip_special_tokens=True
        )

        try:
            return self.processor.post_process_transcription(generated_text)
        except Exception as exc:
            print(f"  Warning: failed to parse VibeVoice structured output: {exc}")
            return []

    @staticmethod
    def _segments_to_word_list(segments: List[Dict]) -> Dict[str, List[Dict]]:
        word_list: Dict[str, List[Dict]] = {}
        for seg in segments:
            spk_id = seg.get("speaker_id", "N/A")
            if spk_id == "N/A":
                continue
            key = f"speaker_{spk_id}"
            word_list.setdefault(key, []).append({
                "word": seg.get("text", "").strip(),
                "start": float(seg.get("start_time", 0.0)),
                "end": float(seg.get("end_time", 0.0)),
                "score": 1.0,
            })
        return word_list

    @staticmethod
    def _segments_to_diar_matrix(
        segments: List[Dict],
        frame_duration: float = FRAME_LEN_SEC,
    ) -> np.ndarray:
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

    def transcribe(self, audio, audio_file=None):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, audio, SR)
            segments = self._generate_segments(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        word_list = self._segments_to_word_list(segments)
        diar_matrix = self._segments_to_diar_matrix(
            segments, frame_duration=FRAME_LEN_SEC
        )
        return word_list, diar_matrix
