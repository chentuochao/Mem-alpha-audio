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
from collections import defaultdict

from audio_script.datasets.Bazinga_loader import BazingaDataset

SR = 16000
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
    audio_file: str,
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
    cfg.audio_file = audio_file
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



def transcription_to_vad(transcripts: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """
    Convert per-speaker segment dicts into per-speaker VAD dicts
    (the format accepted by AlignedProcess as vad1 / vad2).
    """
    return {spk: [{"start": seg["start"], "end": seg["end"]} for seg in segs] for spk, segs in transcripts.items()}


def get_chunked_transcript(transcript: List[Dict], start_time: int, end_time: int) -> List[Dict]:
    """
    Get the chunked transcript for a given time range.
    """

    speaker_words: Dict[str, List[Dict]] = defaultdict(list)
    chunked_transcript = []

    for seg in transcript:
        if seg["start"] >= start_time and seg["end"] <= end_time:
            seg["start"] -= start_time
            seg["end"] -= start_time
            chunked_transcript.append(seg)
            speaker_words[seg["speaker"]].append(seg)

    vad = transcription_to_vad(speaker_words)
    return speaker_words, chunked_transcript, vad



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
        sample_rate=SR,
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

    # print("Configuration complete:", cfg)

    # ── Process each episode ─────────────────────────────────────────
    num_processed = 0
    num_skipped = 0
    num_fail = 0
    audio_chunk = SR*120
    for sample in tqdm(dataset):
        print(f"\n{'=' * 70}")
        print(f"Processing episode: {sample['conv_id']}")
        print(f"  Speakers : {sample['speakers']}")
        print(f"  Audio    : {sample['audio_path']}")
        print(f"{'=' * 70}")

        conv_id = sample["conv_id"]
        # Where to save outputs
        save_dir = os.path.join(args.output_dir, conv_id)
        os.makedirs(save_dir, exist_ok=True)
        raw_audio = sample['audio']
        T = raw_audio.shape[0]
        raw_transcript = sample['raw_transcript']
        chunk_id = 0

        for t in range(0, T, audio_chunk):
            start_time = t
            end_time = t + audio_chunk
            if end_time > T:
                end_time = T
            audio = raw_audio[start_time:end_time]

            speaker_transcripts, chunked_transcript, vad_gt = get_chunked_transcript(raw_transcript, start_time/SR, end_time/SR)

            diar_path = os.path.join(save_dir, f"CHUNK_{chunk_id}", "diart_pred.npy")
            word_list_path = os.path.join(save_dir, f"CHUNK_{chunk_id}", "transcript_pred.json")
            word_list_path_gt = os.path.join(save_dir, f"CHUNK_{chunk_id}", "transcript_gt.json")
            info_path = os.path.join(save_dir, f"CHUNK_{chunk_id}", "sample_info.json")
            vad_gt_path = os.path.join(save_dir, f"CHUNK_{chunk_id}", "vad_gt.json")


            if os.path.exists(diar_path) and os.path.exists(word_list_path) and os.path.exists(info_path):
                print(f"  Skipping (already exists): {diar_path}")
                num_skipped += 1
                chunk_id += 1
                continue


            try:
                seglst_dict_list, word_list, diar_result = run_diarization_asr(
                    audio, sample["audio_path"], asr_model, diar_model, cfg
                )
            except Exception as e:
                print(f"  Error processing {conv_id}: {e}")
                num_fail += 1
                chunk_id += 1
                continue

            os.makedirs(os.path.join(save_dir, f"CHUNK_{chunk_id}"), exist_ok=True)
            with open(word_list_path_gt, "w", encoding="utf-8") as fh:
                json.dump(speaker_transcripts, fh, indent=2)
            ## convert transcription to VAD array for diarization gt

            with open(vad_gt_path, "w", encoding="utf-8") as fh:
                json.dump(vad_gt, fh, indent=2)
            np.save(diar_path, diar_result)

            with open(word_list_path, "w") as fh:
                json.dump(word_list, fh, indent=2)

            # Manifest entry — uses Bazinga-specific fields instead of
            # transcript1/2_path + vad1/2_path from the InterAct pipeline
            sample_info = {
                "dataset": "bazinga",
                "conv_id": sample["conv_id"],
                "chunk_id": chunk_id,
                "audio_file": sample["audio_path"],
                "txt_path": sample["txt_path"],
                "speakers": list(speaker_transcripts.keys()),
                "transcript_path": word_list_path_gt,
                "vad_path": vad_gt_path,
                "diart_path": diar_path,
                "pred_transcript_path": word_list_path,
                "feat_len_sec": 0.08,
                "time_stamp": [start_time, end_time],
            }
            with open(info_path, "w") as fh:
                json.dump(sample_info, fh, indent=2)

            print(f"  Saved: {diar_path}  shape={diar_result.shape}")
            print(f"  Saved: {word_list_path}")
            print(f"  Saved: {info_path}")
            num_processed += 1
            chunk_id += 1
        break

    print(
        f"\nStep 1 (Bazinga) complete. "
        f"Processed {num_processed}, skipped {num_skipped} episodes, failes chunk {num_fail}."
    )


if __name__ == "__main__":
    main()
