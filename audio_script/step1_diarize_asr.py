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
  - seglst.json     (speaker-tagged transcript segments)
  - diar.npy        (binary diarization matrix, num_frames x num_speakers)
  - word_list.json  (per-speaker word-level predictions)

A manifest file (step1_manifest.json) is written so Step 2 / evaluation knows what to load.
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
from audio_script.datasets.turn_annotation import AlignedProcess
from audio_script.step1_eval import load_vad_json, vad_segments_to_binary
from audio_script.eval.multitalker_metrics import compute_der, calculate_session_cpWER, compute_der_bruteforce
from tqdm import tqdm


TURN_GAP_TH = 1.5
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
    log: bool = False

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
    # here first run melspect, each featuers size is 128, each features represents 10ms, sampling rate is 16000Hz
    streaming_buffer.append_audio_file(audio_filepath=audio_file, stream_id=-1)
    streaming_buffer_iter = iter(streaming_buffer)
    multispk_asr_streamer = SpeakerTaggedASR(cfg, asr_model, diar_model)

    # here each step the CacheAwareStreamingAudioBuffer will return audio with step size 112 mel frames, i.e., 1.12s, pre-encode cache is 0.09s, so. the chunk size is 1.21s
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        drop_extra_pre_encoded = (
            0
            if step_num == 0 and not cfg.pad_and_drop_preencoded
            else asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
        )
        # print(f"------------ step {step_num}", chunk_audio.shape, chunk_lengths)
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
    multispk_asr_streamer.generate_words_list_from_parallel_streaming(
        samples=samples
    )
    seglst_dict_list = multispk_asr_streamer.instance_manager.seglst_dict_list
    word_list = multispk_asr_streamer.instance_manager.words_list

    diar_result = (
        (
            multispk_asr_streamer.instance_manager.diar_states.diar_pred_out_stream
            > 0.5
        )
        .cpu()
        .numpy()
    )

    return seglst_dict_list, word_list, diar_result.squeeze(0)


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



def parse_transcript(word_list: Dict) -> List[Dict]:
    # parse the output of words list
    # check the speaker number
    speaker_transcripts = {}
    valid_speakers = []
    transcripts = []
    for speaker in word_list.keys():
        words = word_list[speaker]
        if len(words) == 0:
            continue
        # sorted the words by "start" time
        words = sorted(words, key=lambda x: x['start'])
        transcript = ""
        for word in words:
            transcript += word["word"]
        transcripts.append(transcript)
        valid_speakers.append(speaker)
        speaker_transcripts[speaker] = [{
            "speaker": speaker,
            "start": words[0]['start'],
            "end": words[-1]['end'],
            "words": words
        }]

    speaker_aware_turn = []
    transA, transB = None, None
    if len(valid_speakers) == 0:
        print(f"No valid speakers found for!")
        return []

    elif len(valid_speakers) == 1:
        print(f"Only one valid speaker found for")
        words = speaker_transcripts[valid_speakers[0]]["words"]
        transcript = transcripts[0]
        speaker_aware_turn = [{
            "dialog_type": "dialog",
            "speaker": valid_speakers[0],
            "start": words[0]['start'],
            "end": words[-1]['end'],
            "text": transcript,
            "wfeats": words
        }]
    elif len(valid_speakers) == 2:
        speaker0 = valid_speakers[0]
        speaker1 = valid_speakers[1]
        aligned_process = AlignedProcess(speaker_transcripts[speaker0], speaker_transcripts[speaker1], speaker0, speaker1, interval_character='', turn_gap_threshold = TURN_GAP_TH)
        transA, transB = aligned_process.get_parsed_dialog()
        speaker_aware_turn = transA + transB
        speaker_aware_turn.sort(key=lambda key: (key['start'], -key['end']))

    else:
        # find the top2 speaker with longest transcript
        # sort the valid_speakers by the length of transcripts
        # print(transcripts)
        # print(valid_speakers)
        lengths = [len(t) for t in transcripts]
        sorted_pairs = sorted(zip(lengths, valid_speakers), reverse=True)  # longer first
        valid_speakers = [speaker for length, speaker in sorted_pairs]
        valid_speakers = valid_speakers[:2]
        speaker0 = valid_speakers[0]
        speaker1 = valid_speakers[1]
        # print(valid_speakers)
        # exit(0)
        aligned_process = AlignedProcess(speaker_transcripts[speaker0], speaker_transcripts[speaker1], speaker0, speaker1, interval_character='', turn_gap_threshold = TURN_GAP_TH)
        transA, transB = aligned_process.get_parsed_dialog()
        speaker_aware_turn = transA + transB
        speaker_aware_turn.sort(key=lambda key: (key['start'], -key['end']))

    # print(speaker_aware_turn)
    for utt in speaker_aware_turn:
        print(utt["dialog_type"], utt["speaker"], utt["start"], utt["end"], utt["text"] )

    return speaker_aware_turn

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
    cfg.att_context_size = [70, 13] # 80ms frames
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

    print("Configuration complete:", cfg)

    # ── Process each conversation ─────────────────────────────────────
    num_processed = 0
    num_skipped = 0

    for conv in tqdm(conversations):
        print(f"\n{'=' * 70}")
        print(f"Processing: {conv['spk_pair']} / {conv['conv_id']}")
        print(f"  Audio: {conv['audio_path']}")
        print(f"{'=' * 70}")

        # Decide where to save
        if args.output_dir is not None:
            save_dir = os.path.join(
                args.output_dir, conv["spk_pair"], conv["conv_id"]
            )
        else:
            save_dir = conv["conv_dir"]

        diar_path = os.path.join(save_dir, "diart_pred.npy")
        word_list_path = os.path.join(save_dir, "transcript_pred.json")
        info_path = os.path.join(save_dir, "sample_info.json")

        if os.path.exists(diar_path) and os.path.exists(word_list_path) and os.path.exists(info_path):
            print(f"  Skipping (already exists): {save_dir}")
            num_skipped += 1
            continue

        try:
            seglst_dict_list, word_list, diar_result = run_diarization_asr(
                conv["audio_path"], asr_model, diar_model, cfg
            )
            speaker_aware_turn = parse_transcript(word_list)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        # ── Evaluate DER ──────────────────────────────────────────────
        frame_duration = 0.08  # 0.01s per frame
        total_frames = diar_result.shape[0]
        vad1 = load_vad_json(conv["vad1_path"])
        vad2 = load_vad_json(conv["vad2_path"])
        gt_spk1 = vad_segments_to_binary(vad1, total_frames, frame_duration)
        gt_spk2 = vad_segments_to_binary(vad2, total_frames, frame_duration)
        gt_matrix = np.stack([gt_spk1, gt_spk2], axis=0)  # (2, T)
        pred_matrix = diar_result.T  # (num_speakers, T)
        print(pred_matrix.shape, gt_matrix.shape)
        der, der_details = compute_der_bruteforce(pred_matrix, gt_matrix, frame_duration=frame_duration)
        print(f"  DER: {der:.4f}  "
              f"(miss={der_details['miss']:.2f}s, fa={der_details['fa']:.2f}s, "
              f"conf={der_details['conf']:.2f}s, total={der_details['total']:.2f}s)")

        os.makedirs(save_dir, exist_ok=True)
        np.save(diar_path, diar_result)
        with open(word_list_path, "w") as f:
            json.dump(speaker_aware_turn, f, indent=2)

        sample_info = {
            "spk_pair": conv["spk_pair"],
            "conv_id": conv["conv_id"],
            "audio_file": conv["audio_path"],
            "transcript1_path": conv["transcript1_path"],
            "transcript2_path": conv["transcript2_path"],
            "vad1_path": conv["vad1_path"],
            "vad2_path": conv["vad2_path"],
            "diart_path": diar_path,
            "transcript_path": word_list_path,
            "feat_len_sec": 0.08,
        }
        with open(info_path, "w") as f:
            json.dump(sample_info, f, indent=2)

        print(f"  Saved: {diar_path}  shape={diar_result.shape}")
        print(f"  Saved: {word_list_path}")
        print(f"  Saved: {info_path}")
        num_processed += 1

    print(f"\nStep 1 complete. Processed {num_processed}, skipped {num_skipped} conversations.")


if __name__ == "__main__":
    main()
