"""NeMo offline backend: offline SortFormer + offline ASR, post-hoc merge.

The diarization model uses a high-latency config (chunk_len=340,
right_context=40) for better DER since it isn't constrained by an ASR
streaming buffer.  The ASR model is re-instantiated per chunk to avoid
CUDA-graph state accumulation across variable-length inputs.
"""

from __future__ import annotations

import gc
import os
import tempfile
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch

from ..constants import FRAME_LEN_SEC, SR
from .base import BaseBackend
from .nemo_config import MultitalkerTranscriptionConfig


class NemoOfflineBackend(BaseBackend):
    """Offline NeMo diarization + offline NeMo ASR + post-hoc word merge."""

    name = "nemo-offline"

    def __init__(
        self,
        diar_model_path: str,
        asr_model_path: str,
        max_num_of_spks: int = 4,
        device: str = "cuda",
    ):
        from nemo.collections.asr.models import SortformerEncLabelModel
        from omegaconf import OmegaConf

        self.device = torch.device(device)
        self.max_num_of_spks = max_num_of_spks
        # ASR model is re-instantiated per chunk to avoid CUDA-graph state
        # accumulation across variable-length inputs.
        self.asr_model_path = asr_model_path

        print("Loading diarization model (offline)...")
        self.diar_model = (
            SortformerEncLabelModel.restore_from(diar_model_path)
            .eval()
            .to(self.device)
        )

        cfg = OmegaConf.structured(MultitalkerTranscriptionConfig())
        cfg.att_context_size = [70, 13]
        cfg.max_num_of_spks = max_num_of_spks
        self.diar_model._cfg.max_num_of_spks = max_num_of_spks
        for key in cfg:
            if cfg[key] == "None":
                cfg[key] = None
        self.cfg = cfg

        self.diar_model.streaming_mode = cfg.streaming_mode
        self.diar_model.sortformer_modules.log = cfg.log

        # High-latency offline diarization config for best DER.
        self.diar_model.sortformer_modules.chunk_len = 340
        self.diar_model.sortformer_modules.spkcache_len = 188
        self.diar_model.sortformer_modules.chunk_left_context = 0
        self.diar_model.sortformer_modules.chunk_right_context = 40
        self.diar_model.sortformer_modules.fifo_len = 40
        self.diar_model.sortformer_modules.spkcache_update_period = 300

        self._tmp_dir = tempfile.mkdtemp(prefix="multi_asr_offline_")

    def __del__(self):
        try:
            os.rmdir(self._tmp_dir)
        except Exception:
            pass

    @staticmethod
    def _run_offline_diarization(
        audio: np.ndarray,
        diar_model,
        max_num_of_spks: int,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        predicted_segments, pred_tensors = diar_model.diarize(
            audio=[audio],
            batch_size=1,
            sample_rate=SR,
            include_tensor_outputs=True,
        )
        diar_probs = pred_tensors[0].squeeze(0).clone()  # (T, S)
        diar_probs[:, max_num_of_spks:] = 0.0
        diar_binary = (diar_probs > 0.5).cpu().numpy()
        diar_probs = diar_probs.cpu()
        # Free GPU tensors before ASR runs to avoid CUDA-graph capture issues.
        del pred_tensors, predicted_segments
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return diar_binary, diar_probs

    def _run_offline_asr(self, audio: np.ndarray, tag: str):
        from nemo.collections.asr.models import ASRModel

        asr_model = ASRModel.restore_from(self.asr_model_path).eval().to(self.device)
        tmp_wav = os.path.join(self._tmp_dir, f"tmp_{tag}.wav")
        sf.write(tmp_wav, audio, SR)
        try:
            transcriptions = asr_model.transcribe(
                audio=[tmp_wav],
                batch_size=1,
                timestamps=True,
            )
            hyps = transcriptions[0] if isinstance(transcriptions, tuple) else transcriptions
            return hyps[0]
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
            del asr_model
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _merge_diar_and_asr(
        diar_probs: torch.Tensor,
        asr_hyp,
        max_num_of_spks: int,
    ) -> Dict[str, List[Dict]]:
        words_by_speaker: Dict[str, List[Dict]] = defaultdict(list)

        if not hasattr(asr_hyp, "timestamp") or asr_hyp.timestamp is None:
            return dict(words_by_speaker)
        word_timestamps = asr_hyp.timestamp.get("word", [])
        if not word_timestamps:
            return dict(words_by_speaker)

        n_frames = diar_probs.shape[0]
        for w in word_timestamps:
            word_text = w.get("word", w.get("char", ""))
            if not word_text.strip():
                continue

            frame_stt = w["start_offset"]
            frame_end = w["end_offset"]
            if frame_stt == frame_end:
                if frame_stt >= n_frames - 1:
                    frame_stt, frame_end = n_frames - 1, n_frames
                else:
                    frame_end = frame_stt + 1

            stt_p = max(frame_stt - 1, 0)
            end_p = frame_end
            speaker_sigmoid = diar_probs[stt_p:end_p, :].mean(dim=0)
            speaker_sigmoid[max_num_of_spks:] = 0.0
            spk_id = speaker_sigmoid.argmax().item()

            words_by_speaker[f"speaker_{spk_id}"].append({
                "word": word_text,
                "start": round(frame_stt * FRAME_LEN_SEC, 3),
                "end": round(frame_end * FRAME_LEN_SEC, 3),
                "speaker": f"speaker_{spk_id}",
                "score": round(speaker_sigmoid[spk_id].item(), 4),
            })

        return dict(words_by_speaker)

    def transcribe(self, audio, audio_file=None):
        diar_binary, diar_probs = self._run_offline_diarization(
            audio, self.diar_model, self.max_num_of_spks
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tag = os.path.basename(audio_file or "chunk")
        asr_hyp = self._run_offline_asr(audio, tag)

        word_list = self._merge_diar_and_asr(
            diar_probs, asr_hyp, self.max_num_of_spks
        )
        return word_list, diar_binary

    def extra_manifest(self):
        return {
            "mode": self.name,
            "diar_config": {
                "chunk_len": int(self.diar_model.sortformer_modules.chunk_len),
                "chunk_right_context": int(
                    self.diar_model.sortformer_modules.chunk_right_context
                ),
                "fifo_len": int(self.diar_model.sortformer_modules.fifo_len),
                "spkcache_len": int(self.diar_model.sortformer_modules.spkcache_len),
            },
        }
