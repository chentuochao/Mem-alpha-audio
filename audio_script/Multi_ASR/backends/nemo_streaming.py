"""NeMo streaming backend: cache-aware streaming ASR + streaming SortFormer."""

from __future__ import annotations

import torch

from .base import BaseBackend
from .nemo_config import MultitalkerTranscriptionConfig


class NemoStreamingBackend(BaseBackend):
    """Streaming multi-talker diarization + ASR via NeMo's ``SpeakerTaggedASR``."""

    name = "nemo-streaming"

    def __init__(
        self,
        diar_model_path: str,
        asr_model_path: str,
        max_num_of_spks: int = 6,
        device: str = "cuda",
    ):
        from nemo.collections.asr.models import (
            ASRModel,
            SortformerEncLabelModel,
        )
        from omegaconf import OmegaConf

        self.device = torch.device(device)

        print("Loading diarization model (streaming)...")
        self.diar_model = (
            SortformerEncLabelModel.restore_from(diar_model_path)
            .eval()
            .to(self.device)
        )

        print("Loading ASR model (streaming)...")
        self.asr_model = (
            ASRModel.restore_from(asr_model_path).eval().to(self.device)
        )

        self.max_num_of_spks = max_num_of_spks
        cfg = OmegaConf.structured(MultitalkerTranscriptionConfig())
        cfg.att_context_size = [70, 13]
        cfg.max_num_of_spks = max_num_of_spks
        self.diar_model._cfg.max_num_of_spks = max_num_of_spks

        for key in cfg:
            if cfg[key] == "None":
                cfg[key] = None
        self.cfg = cfg

        # Low-latency streaming SortFormer config.
        self.diar_model.streaming_mode = cfg.streaming_mode
        self.diar_model.sortformer_modules.chunk_len = (
            cfg.chunk_len if cfg.chunk_len > 0 else 6
        )
        self.diar_model.sortformer_modules.spkcache_len = cfg.spkcache_len
        self.diar_model.sortformer_modules.chunk_left_context = cfg.chunk_left_context
        self.diar_model.sortformer_modules.chunk_right_context = (
            cfg.chunk_right_context if cfg.chunk_right_context > 0 else 7
        )
        self.diar_model.sortformer_modules.fifo_len = cfg.fifo_len
        self.diar_model.sortformer_modules.log = cfg.log
        self.diar_model.sortformer_modules.spkcache_refresh_rate = (
            cfg.spkcache_refresh_rate
        )

    def transcribe(self, audio, audio_file=None):
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
            SpeakerTaggedASR,
        )
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer,
        )

        cfg = self.cfg
        # cfg.audio_file is metadata only — pass the episode path if available.
        cfg.audio_file = audio_file
        samples = [{"audio_filepath": audio_file}] if audio_file else [{}]

        streaming_buffer = CacheAwareStreamingAudioBuffer(
            model=self.asr_model,
            online_normalization=cfg.online_normalization,
            pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
        )
        streaming_buffer.append_audio(audio=audio, stream_id=-1)
        multispk_asr_streamer = SpeakerTaggedASR(cfg, self.asr_model, self.diar_model)

        for step_num, (chunk_audio, chunk_lengths) in enumerate(iter(streaming_buffer)):
            drop_extra_pre_encoded = (
                0
                if step_num == 0 and not cfg.pad_and_drop_preencoded
                else self.asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
            )
            with torch.inference_mode():
                with torch.amp.autocast(self.diar_model.device.type, enabled=True):
                    with torch.no_grad():
                        multispk_asr_streamer.perform_parallel_streaming_stt_spk(
                            step_num=step_num,
                            chunk_audio=chunk_audio,
                            chunk_lengths=chunk_lengths,
                            is_buffer_empty=streaming_buffer.is_buffer_empty(),
                            drop_extra_pre_encoded=drop_extra_pre_encoded,
                        )

        multispk_asr_streamer.generate_seglst_dicts_from_parallel_streaming(samples=samples)
        multispk_asr_streamer.generate_words_list_from_parallel_streaming(samples=samples)

        word_list = multispk_asr_streamer.instance_manager.words_list
        diar_result = (
            (multispk_asr_streamer.instance_manager.diar_states.diar_pred_out_stream > 0.5)
            .cpu()
            .numpy()
        ).squeeze(0)
        return word_list, diar_result
