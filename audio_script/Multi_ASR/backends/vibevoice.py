"""VibeVoice end-to-end multi-talker ASR backend.

The model directly produces speaker-tagged segments — we convert those to the
unified ``word_list`` dict and a binary diarization matrix.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import torch

from ..constants import FRAME_LEN_SEC, SR
from .base import BaseBackend

# Maps the model's raw JSON keys to the internal segment schema that
# _segments_to_word_list / _segments_to_diar_matrix consume. Mirrors the mapping
# in VibeVoiceASRProcessor.post_process_transcription so the salvage parser
# produces identical output.
_SEGMENT_KEY_MAPPING = {
    "Start time": "start_time",
    "Start": "start_time",
    "End time": "end_time",
    "End": "end_time",
    "Speaker ID": "speaker_id",
    "Speaker": "speaker_id",
    "Content": "text",
}


class VibeVoiceBackend(BaseBackend):
    """End-to-end multi-talker diarization + ASR via the VibeVoice model."""

    name = "vibevoice"

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 10240,
        temperature: float = 0.0,
        top_p: float = 1.0,
        num_beams: int = 1,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 0,
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
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size

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
        # Anti-degeneration: greedy decoding on noisy audio collapses into
        # repetition loops that never emit EOS (run to max_new_tokens and yield
        # empty transcripts). A repetition penalty / n-gram block breaks the loop
        # so the model terminates and closes its JSON. Only set when enabled so
        # default greedy behaviour is unchanged when the knobs are off.
        if self.repetition_penalty and self.repetition_penalty != 1.0:
            gen_cfg["repetition_penalty"] = self.repetition_penalty
        if self.no_repeat_ngram_size and self.no_repeat_ngram_size > 0:
            gen_cfg["no_repeat_ngram_size"] = self.no_repeat_ngram_size
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

        # Debug: expose what the model actually produced so empty/malformed-JSON
        # chunks can be diagnosed. `hit_eos` distinguishes a clean stop from a
        # runaway generation that ran to max_new_tokens (the slow + empty case).
        hit_eos = len(eos_positions) > 0
        print(
            f"  generated: {len(generated_ids)} tokens | hit_eos={hit_eos} | "
            f"text_len={len(generated_text)}"
        )
        print(f"  generated_text[head]: {generated_text[:300]!r}")
        print(f"  generated_text[tail]: {generated_text[-300:]!r}")

        try:
            segments = self.processor.post_process_transcription(generated_text)
        except Exception as exc:
            print(f"  Warning: failed to parse VibeVoice structured output: {exc}")
            segments = []

        # The library parser is all-or-nothing: if the outer JSON array never
        # closes (runaway generation truncated mid-object), it returns nothing and
        # discards every complete segment that WAS produced. Salvage those by
        # parsing the balanced top-level objects individually.
        if not segments:
            salvaged = self._salvage_segments(generated_text)
            if salvaged:
                print(
                    f"  Salvaged {len(salvaged)} segment(s) from unparseable/"
                    f"truncated output (hit_eos={hit_eos})"
                )
            segments = salvaged
        return segments

    @staticmethod
    def _salvage_segments(text: str) -> List[Dict[str, Any]]:
        """Best-effort extraction of segment objects from malformed model output.

        Scans from the first ``[`` and collects every balanced top-level
        ``{...}`` object (respecting string literals and escapes so braces/quotes
        inside ``Content`` don't corrupt the count), json-decoding each one and
        stopping at the first incomplete object. Recovers the valid prefix of a
        truncated array; returns ``[]`` if nothing parseable is found.
        """
        arr_start = text.find("[")
        if arr_start == -1:
            return []

        raw_objs: List[Dict] = []
        i, n = arr_start + 1, len(text)
        while i < n:
            if text[i] != "{":
                i += 1
                continue
            depth = 0
            in_str = esc = False
            complete = False
            j = i
            while j < n:
                c = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                elif c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        complete = True
                        break
                j += 1
            if not complete:
                break  # trailing truncated object -> stop
            try:
                obj = json.loads(text[i:j + 1])
                if isinstance(obj, dict):
                    raw_objs.append(obj)
            except Exception:
                pass  # skip a single corrupt object, keep going
            i = j + 1

        cleaned: List[Dict[str, Any]] = []
        for item in raw_objs:
            mapped = {
                mapped_key: item[key]
                for key, mapped_key in _SEGMENT_KEY_MAPPING.items()
                if key in item
            }
            if mapped:
                cleaned.append(mapped)
        return cleaned

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
