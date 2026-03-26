"""
Step 1: Multi-talker diarization + ASR  (runs in env1 — NeMo environment)

Input: data directory produced by mix_interact.py with the structure:
  data_dir/
    {spk1}_{spk2}/
      {conv_id}/
        mixed_conv.wav
        transcript1.json  transcript2.json
        vad1.json         vad2.json

For each conversation, run streaming diarization + ASR and save:
  - seglst.json   (speaker-tagged transcript segments)
  - diar.npy      (binary diarization matrix, num_frames x num_speakers)

A manifest file (step1_manifest.json) is written so Step 2 knows what to load.
"""

import argparse
import glob
import json
import os
from pprint import pprint
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from nemo.collections.asr.models import ASRModel, SortformerEncLabelModel
from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR
from nemo.collections.asr.parts.utils.streaming_utils import (
    CacheAwareStreamingAudioBuffer,
)
from omegaconf import OmegaConf

from dataclasses import dataclass, field, is_dataclass


# Use the pre-defined dataclass template `MultitalkerTranscriptionConfig` from `multitalker_transcript_config.py`.
# Configure the diarization model using streaming parameters:
# from multitalker_transcript_config import MultitalkerTranscriptionConfig
@dataclass
class MultitalkerTranscriptionConfig:
    """
    Configuration for Multi-talker transcription with an ASR model and a diarization model.
    """

    # Required configs
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

    # General configs
    session_len_sec: float = -1
    num_workers: int = 8
    random_seed: Optional[int] = None
    log: bool = True

    # Streaming diarization configs
    streaming_mode: bool = True
    spkcache_len: int = 188
    spkcache_refresh_rate: int = 0
    fifo_len: int = 188
    chunk_len: int = 0
    chunk_left_context: int = 0
    chunk_right_context: int = 0

    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = False
    matmul_precision: str = "highest"  # Literal["highest", "high", "medium"]

    # ASR Configs
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
    set_decoder: Optional[str] = None  # ["ctc", "rnnt"]
    generate_realtime_scripts: bool = False
    spk_supervision: str = "diar"  # ["diar", "rttm"]
    binary_diar_preds: bool = False

    # Multitalker transcription configs
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


def run_diarization_asr(
    audio_file: str,
    asr_model,
    diar_model,
    cfg,
) -> Tuple[List[Dict], np.ndarray]:
    """
    Run streaming multi-talker diarization + ASR on one audio file.

    Returns:
        seglst_dict_list: per-segment dicts {speaker, start_time, end_time, words}
        diar_result:      binary (num_frames, num_speakers)
    """
    cfg.audio_file = audio_file
    samples = [{"audio_filepath": audio_file}]

    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=cfg.online_normalization,
        pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
    )
    streaming_buffer.append_audio_file(audio_filepath=audio_file, stream_id=-1)
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

    multispk_asr_streamer.generate_seglst_dicts_from_parallel_streaming(
        samples=samples
    )
    seglst_dict_list = multispk_asr_streamer.instance_manager.seglst_dict_list

    diar_result = (
        (
            multispk_asr_streamer.instance_manager.diar_states.diar_pred_out_stream
            > 0.5
        )
        .cpu()
        .numpy()
    )

    return seglst_dict_list, diar_result

def discover_conversations(data_dir: str) -> List[Dict]:
    """
    Walk the directory tree produced by mix_interact.py and return a list of
    conversation dicts, each containing paths to the wav / transcript / vad files.

    Expected layout:
        data_dir/{spk_pair}/{conv_id}/mixed_conv.wav
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


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Run multi-talker diarization + ASR (NeMo env)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory produced by mix_interact.py "
             "(contains {spk_pair}/{conv_id}/ sub-folders)",
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
        default=4,
        help="Maximum number of speakers per audio file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results. Defaults to writing inside each "
             "conversation folder under data_dir.",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # ── Discover conversations ────────────────────────────────────────
    conversations = discover_conversations(args.data_dir)
    print(f"Found {len(conversations)} conversations under {args.data_dir}")
    if not conversations:
        print("No conversations found. Check your --data_dir path.")
        return

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
    diar_model.sortformer_modules.chunk_len = (
        cfg.chunk_len if cfg.chunk_len > 0 else 6
    )
    diar_model.sortformer_modules.spkcache_len = cfg.spkcache_len
    diar_model.sortformer_modules.chunk_left_context = cfg.chunk_left_context
    diar_model.sortformer_modules.chunk_right_context = (
        cfg.chunk_right_context if cfg.chunk_right_context > 0 else 7
    )
    diar_model.sortformer_modules.fifo_len = cfg.fifo_len
    diar_model.sortformer_modules.log = cfg.log
    diar_model.sortformer_modules.spkcache_refresh_rate = cfg.spkcache_refresh_rate

    print("Configuration complete.")

    # ── Process each conversation ─────────────────────────────────────
    manifest_entries = []

    for conv in conversations:
        print(f"\n{'=' * 70}")
        print(f"Processing: {conv['spk_pair']} / {conv['conv_id']}")
        print(f"  Audio: {conv['audio_path']}")
        print(f"{'=' * 70}")

        seglst_dict_list, diar_result = run_diarization_asr(
            conv["audio_path"], asr_model, diar_model, cfg
        )

        for seg in seglst_dict_list:
            speaker = seg.get("speaker", "Unknown")
            st = seg.get("start_time", 0.0)
            et = seg.get("end_time", 0.0)
            words = seg.get("words", "")
            print(f"  [{speaker}] ({st:.2f}s - {et:.2f}s): {words}")

        # Decide where to save: per-conversation folder or central output_dir
        if args.output_dir is not None:
            save_dir = os.path.join(
                args.output_dir, conv["spk_pair"], conv["conv_id"]
            )
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = conv["conv_dir"]

        seglst_path = os.path.join(save_dir, "seglst.json")
        diar_path = os.path.join(save_dir, "diar.npy")

        with open(seglst_path, "w") as f:
            json.dump(seglst_dict_list, f, indent=2)
        np.save(diar_path, diar_result)

        print(f"  Saved: {seglst_path}")
        print(f"  Saved: {diar_path}  shape={diar_result.shape}")

        manifest_entries.append({
            "spk_pair": conv["spk_pair"],
            "conv_id": conv["conv_id"],
            "conv_dir": save_dir,
            "audio_file": conv["audio_path"],
            "transcript1_path": conv["transcript1_path"],
            "transcript2_path": conv["transcript2_path"],
            "vad1_path": conv["vad1_path"],
            "vad2_path": conv["vad2_path"],
            "seglst_path": seglst_path,
            "diar_path": diar_path,
            "feat_len_sec": cfg.feat_len_sec,
        })

    # ── Write manifest for Step 2 ────────────────────────────────────
    manifest_dir = args.output_dir if args.output_dir else args.data_dir
    manifest_path = os.path.join(manifest_dir, "step1_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_entries, f, indent=2)
    print(f"\nStep 1 complete. Processed {len(manifest_entries)} conversations.")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
