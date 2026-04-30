"""
Step 1 (Bazinga variant): Multi-talker diarization + ASR on the Bazinga
(Friends TV show) dataset.  Runs in env1 — NeMo environment.

Input: Bazinga data directory with the flat structure:
  data_dir/
    Friends.Season01.Episode01.en.wav
    Friends.Season01.Episode01.txt
    Friends.Season01.Episode02.en.wav
    Friends.Season01.Episode02.txt
    ...

For each episode, run streaming diarization + ASR and save:
  - diart_pred.npy        (binary diarization matrix, num_frames x num_speakers)
  - transcript_pred.json  (per-speaker word-level predictions)
  - sample_info.json      (manifest entry for Step 2 / evaluation)

The manifest records the path to the Bazinga ground-truth transcript JSON so
that downstream evaluation can compare predictions against reference labels.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from nemo.collections.asr.models import ASRModel, SortformerEncLabelModel
from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR
from nemo.collections.asr.parts.utils.streaming_utils import (
    CacheAwareStreamingAudioBuffer,
)
from omegaconf import OmegaConf
from tqdm import tqdm

from audio_script.datasets.Bazinga_loader import BazingaDataset


# ──────────────────────────────────────────────────────────────────────────────
# Model config  (identical to the original step1_diarize_asr.py)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MultitalkerTranscriptionConfig:
    """Configuration for multi-talker transcription with NeMo ASR + diarization."""

    diar_model: Optional[str] = None
    diar_pretrained_name: Optional[str] = None
    max_num_of_spks: Optional[int] = 4
    parallel_speaker_strategy: bool = True
    masked_asr: bool = True
    mask_preencode: bool = False
    cache_gating: bool = True
    cache_gating_buffer_size: int = 2
    single_speaker_mode: bool = False
    feat_len_sec: float = 0.01

    session_len_sec: float = -1
    num_workers: int = 8
    random_seed: Optional[int] = None
    log: bool = False

    streaming_mode: bool = True
    spkcache_len: int = 188
    spkcache_refresh_rate: int = 0
    fifo_len: int = 188
    chunk_len: int = 0
    chunk_left_context: int = 0
    chunk_right_context: int = 0

    cuda: Optional[int] = None
    allow_mps: bool = False
    matmul_precision: str = "highest"

    asr_model: Optional[str] = None
    device: str = "cuda"
    audio_file: Optional[str] = None
    manifest_file: Optional[str] = None
    att_context_size: Optional[List[int]] = field(default_factory=lambda: [70, 13])
    use_amp: bool = True
    debug_mode: bool = False
    deploy_mode: bool = False
    batch_size: int = 32
    chunk_size: int = -1
    shift_size: int = -1
    left_chunks: int = 2
    online_normalization: bool = False
    output_path: Optional[str] = None
    pad_and_drop_preencoded: bool = False
    set_decoder: Optional[str] = None
    generate_realtime_scripts: bool = False
    spk_supervision: str = "diar"
    binary_diar_preds: bool = False

    verbose: bool = False
    word_window: int = 50
    sent_break_sec: float = 30.0
    fix_prev_words_count: int = 5
    update_prev_words_sentence: int = 5
    left_frame_shift: int = -1
    right_frame_shift: int = 0
    min_sigmoid_val: float = 1e-2
    discarded_frames: int = 8
    print_time: bool = True

    print_sample_indices: List[int] = field(default_factory=lambda: [0])
    colored_text: bool = True
    real_time_mode: bool = False
    print_path: Optional[str] = None
    ignored_initial_frame_steps: int = 5
    finetune_realtime_ratio: float = 0.01


# ──────────────────────────────────────────────────────────────────────────────
# Core inference  (identical to the original step1_diarize_asr.py)
# ──────────────────────────────────────────────────────────────────────────────

def run_diarization_asr(
    audio: np.ndarray,
    asr_model,
    diar_model,
    cfg,
) -> Tuple[List[Dict], List[Dict], np.ndarray]:
    """
    Run streaming multi-talker diarization + ASR on one audio file.

    Returns:
        seglst_dict_list  – per-segment dicts {speaker, start_time, end_time, words}
        word_list         – per-speaker word-level predictions
        diar_result       – binary numpy array (num_frames, num_speakers)
    """
    # cfg.audio_file = audio_file
    samples = [{"audio_filepath": audio_file}]

    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=cfg.online_normalization,
        pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
    )
    # streaming_buffer.append_audio_file(audio_filepath=audio_file, stream_id=-1)
    streaming_buffer.append_audio(audio=audio, stream_id=-1)
    streaming_buffer_iter = iter(streaming_buffer)
    multispk_asr_streamer = SpeakerTaggedASR(cfg, asr_model, diar_model)

    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        drop_extra_pre_encoded = (
            0
            if step_num == 0 and not cfg.pad_and_drop_preencoded
            else asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
        )
        with torch.inference_mode():
            with torch.amp.autocast(diar_model.device.type, enabled=True):
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

    seglst_dict_list = multispk_asr_streamer.instance_manager.seglst_dict_list
    word_list = multispk_asr_streamer.instance_manager.words_list
    diar_result = (
        (multispk_asr_streamer.instance_manager.diar_states.diar_pred_out_stream > 0.5)
        .cpu()
        .numpy()
    )

    return seglst_dict_list, word_list, diar_result.squeeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 1 (Bazinga): Run multi-talker diarization + ASR on "
                    "the Bazinga/Friends dataset (NeMo env)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing Friends *.en.wav and *.txt episode files",
    )
    parser.add_argument(
        "--diar_model_path",
        type=str,
        required=True,
        help="Path to NeMo diarization model (.nemo)",
    )
    parser.add_argument(
        "--asr_model_path",
        type=str,
        required=True,
        help="Path to NeMo ASR model (.nemo)",
    )
    parser.add_argument(
        "--max_num_of_spks",
        type=int,
        default=6,
        help="Maximum number of speakers per episode (default: 6 for Friends)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results. Defaults to a 'bazinga/' sub-folder "
             "inside each episode's data directory.",
    )
    parser.add_argument(
        "--turn_gap",
        type=float,
        default=1.5,
        help="Silence gap (seconds) used to split ground-truth turns (default: 1.5)",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # ── Discover episodes via Bazinga loader ─────────────────────────
    dataset = BazingaDataset(
        args.data_dir,
        sample_rate=16000,
    )
    print(f"Found {len(dataset)} episodes under {args.data_dir}")

    # ── Load models ──────────────────────────────────────────────────
    print("Loading diarization model...")
    diar_model = (
        SortformerEncLabelModel.restore_from(args.diar_model_path)
        .eval()
        .to(torch.device("cuda"))
    )

    print("Loading ASR model...")
    asr_model = (
        ASRModel.restore_from(args.asr_model_path).eval().to(torch.device("cuda"))
    )

    # ── Configure ────────────────────────────────────────────────────
    cfg = OmegaConf.structured(MultitalkerTranscriptionConfig())
    cfg.att_context_size = [70, 13]
    cfg.max_num_of_spks = args.max_num_of_spks
    diar_model._cfg.max_num_of_spks = args.max_num_of_spks

    for key in cfg:
        cfg[key] = None if cfg[key] == "None" else cfg[key]

    diar_model.streaming_mode = cfg.streaming_mode
    diar_model.sortformer_modules.chunk_len = cfg.chunk_len if cfg.chunk_len > 0 else 6
    diar_model.sortformer_modules.spkcache_len = cfg.spkcache_len
    diar_model.sortformer_modules.chunk_left_context = cfg.chunk_left_context
    diar_model.sortformer_modules.chunk_right_context = (
        cfg.chunk_right_context if cfg.chunk_right_context > 0 else 7
    )
    diar_model.sortformer_modules.fifo_len = cfg.fifo_len
    diar_model.sortformer_modules.log = cfg.log
    diar_model.sortformer_modules.spkcache_refresh_rate = cfg.spkcache_refresh_rate

    print("Configuration complete:", cfg)

    # ── Process each episode ─────────────────────────────────────────
    num_processed = 0
    num_skipped = 0

    for sample in tqdm(dataset):
        print(f"\n{'=' * 70}")
        print(f"Processing episode: {sample['conv_id']}")
        print(f"  Speakers : {sample['speakers']}")
        print(f"  Audio    : {sample['audio_path']}")
        print(f"{'=' * 70}")

        conv_id = sample["conv_id"]
        # Where to save outputs
        save_dir = os.path.join(args.output_dir, conv_id)
        diar_path = os.path.join(save_dir, "diart_pred.npy")
        word_list_path = os.path.join(save_dir, "transcript_pred.json")
        info_path = os.path.join(save_dir, "sample_info.json")

        if os.path.exists(diar_path) and os.path.exists(word_list_path) and os.path.exists(info_path):
            print(f"  Skipping (already exists): {save_dir}")
            num_skipped += 1
            continue
        print(f"  Running diarization + ASR for {conv_id}")
        print(f"  Audio shape: {sample['audio'].shape}")
        exit(0)
        try:
            seglst_dict_list, word_list, diar_result = run_diarization_asr(
                sample["audio"], asr_model, diar_model, cfg
            )
        except Exception as e:
            print(f"  Error processing {conv_id}: {e}")
            continue
        

        os.makedirs(save_dir, exist_ok=True)
        np.save(diar_path, diar_result)
        with open(word_list_path, "w") as fh:
            json.dump(word_list, fh, indent=2)

        # Manifest entry — uses Bazinga-specific fields instead of
        # transcript1/2_path + vad1/2_path from the InterAct pipeline
        sample_info = {
            "dataset": "bazinga",
            "conv_id": conv["conv_id"],
            "audio_file": conv["audio_path"],
            "txt_path": conv["txt_path"],
            "speakers": conv["speakers"],
            "gt_transcript_path": conv["gt_transcript_path"],
            "diart_path": diar_path,
            "pred_transcript_path": word_list_path,
            "feat_len_sec": 0.08,
        }
        with open(info_path, "w") as fh:
            json.dump(sample_info, fh, indent=2)

        print(f"  Saved: {diar_path}  shape={diar_result.shape}")
        print(f"  Saved: {word_list_path}")
        print(f"  Saved: {info_path}")
        num_processed += 1

    print(
        f"\nStep 1 (Bazinga) complete. "
        f"Processed {num_processed}, skipped {num_skipped} episodes."
    )


if __name__ == "__main__":
    main()
