"""
Step 1 (Bazinga variant — OFFLINE): Multi-talker diarization + ASR on the
Bazinga (Friends TV show) dataset.  Runs in env1 — NeMo environment.

Unlike the streaming version (step1_diarize_asr_bazinga.py), this script runs
the diarization model and ASR model **independently in offline / batch mode**,
then merges their outputs.  This allows the diarization model to use the
high-latency configuration (chunk_len=340, right_context=40, etc.) for better
DER, since it is no longer constrained by the ASR streaming buffer's chunk size.

Input: Bazinga data directory with the flat structure:
  data_dir/
    Friends.Season01.Episode01.en.wav
    Friends.Season01.Episode01.txt
    ...

For each episode chunk, saves:
  - diart_pred.npy        (binary diarization matrix, num_frames x num_speakers)
  - transcript_pred.json  (per-speaker word-level predictions)
  - transcript_gt.json    (ground-truth per-speaker words)
  - vad_gt.json           (ground-truth VAD)
  - sample_info.json      (manifest entry for Step 2 / evaluation)
"""

import argparse
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gc

import numpy as np
import soundfile as sf
import torch
from nemo.collections.asr.models import ASRModel, SortformerEncLabelModel
from omegaconf import OmegaConf
from tqdm import tqdm

from audio_script.datasets.Bazinga_loader import BazingaDataset
from prepare_data.preprocess_utils import chunk_dialog, transcription_to_vad
SR = 16000
FRAME_LEN_SEC = 0.08

# ──────────────────────────────────────────────────────────────────────────────
# Chunk splitting (same logic as streaming version)
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
# Offline diarization + ASR
# ──────────────────────────────────────────────────────────────────────────────

def run_offline_diarization(
    audio: np.ndarray,
    diar_model: SortformerEncLabelModel,
    max_num_of_spks: int,
) -> np.ndarray:
    """Run offline diarization on a numpy audio array.

    Returns:
        diar_preds: binary numpy array (num_frames, num_speakers)
    """
    predicted_segments, pred_tensors = diar_model.diarize(
        audio=[audio],
        batch_size=1,
        sample_rate=SR,
        include_tensor_outputs=True,
    )
    diar_probs = pred_tensors[0].squeeze(0).clone()  # (T, S)
    diar_probs[:, max_num_of_spks:] = 0.0
    diar_binary = (diar_probs > 0.5).cpu().numpy()
    diar_probs = diar_probs.cpu()  # move off GPU before ASR runs
    # Explicitly free GPU tensors so Python GC doesn't trigger during CUDA Graph capture
    del pred_tensors, predicted_segments
    torch.cuda.synchronize()
    return diar_binary, diar_probs


def run_offline_asr(
    audio: np.ndarray,
    asr_model: ASRModel,
    tmp_dir: str,
    chunk_id_tag: str,
) -> list:
    """Run offline ASR on a numpy audio array.

    Writes a temporary WAV, transcribes it, and returns word-level hypotheses.

    Returns:
        word_timestamps: list of {word, start_offset, end_offset} dicts
    """
    tmp_wav = os.path.join(tmp_dir, f"tmp_{chunk_id_tag}.wav")
    sf.write(tmp_wav, audio, SR)
    try:
        transcriptions = asr_model.transcribe(
            audio=[tmp_wav],
            batch_size=1,
            timestamps=True,
        )
        hyps = transcriptions[0] if isinstance(transcriptions, tuple) else transcriptions
        hyp = hyps[0]
        return hyp
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def merge_diar_and_asr(
    diar_probs: torch.Tensor,
    asr_hyp,
    max_num_of_spks: int,
) -> Dict[str, List[Dict]]:
    """Merge diarization predictions with ASR word timestamps.

    For each ASR word, determine the speaker by averaging the diarization
    probability over the word's time span and picking the argmax speaker.

    Returns:
        per-speaker word list: {speaker_0: [{word, start, end, score}, ...], ...}
    """
    words_by_speaker: Dict[str, List[Dict]] = defaultdict(list)

    if not hasattr(asr_hyp, 'timestamp') or asr_hyp.timestamp is None:
        return dict(words_by_speaker)

    word_timestamps = asr_hyp.timestamp.get('word', [])
    if not word_timestamps:
        return dict(words_by_speaker)

    n_frames = diar_probs.shape[0]

    for w in word_timestamps:
        word_text = w.get('word', w.get('char', ''))
        if not word_text.strip():
            continue

        frame_stt = w['start_offset']
        frame_end = w['end_offset']

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

        stt_sec = frame_stt * FRAME_LEN_SEC
        end_sec = frame_end * FRAME_LEN_SEC

        words_by_speaker[f"speaker_{spk_id}"].append({
            "word": word_text,
            "start": round(stt_sec, 3),
            "end": round(end_sec, 3),
            "speaker": f"speaker_{spk_id}",
            "score": round(speaker_sigmoid[spk_id].item(), 4),
        })

    return dict(words_by_speaker)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _disable_cuda_graphs(model):
    """Disable CUDA graph optimization in NeMo ASR models.

    NeMo does 2 warm-up runs then captures a CUDA graph on run 3.  With
    variable-length audio chunks the graph replay accesses out-of-bounds
    memory, producing cudaErrorIllegalAddress.  Flip every knob we know.
    """
    from omegaconf import open_dict

    # 1. Module-level flags used by NeMo encoders
    for module in model.modules():
        for attr in ("use_graph", "_use_cuda_graph", "use_cuda_graph",
                     "cuda_graph_mode", "_cuda_graph_mode"):
            if hasattr(module, attr):
                setattr(module, attr, False)

    # 2. Decoding config: fused_batch_size=-1 disables CTC/RNNT CUDA graphs
    try:
        if hasattr(model, "cfg") and hasattr(model.cfg, "decoding"):
            with open_dict(model.cfg.decoding):
                if hasattr(model.cfg.decoding, "fused_batch_size"):
                    model.cfg.decoding.fused_batch_size = -1
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Step 1 (Bazinga — OFFLINE): Run multi-talker diarization + ASR "
                    "on the Bazinga/Friends dataset in offline mode"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing Friends *.en.wav and *.txt files")
    parser.add_argument("--diar_model_path", type=str, required=True,
                        help="Path to NeMo diarization model (.nemo)")
    parser.add_argument("--asr_model_path", type=str, required=True,
                        help="Path to NeMo ASR model (.nemo)")
    parser.add_argument("--max_num_of_spks", type=int, default=4,
                        help="Maximum number of speakers (default: 4)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results")
    parser.add_argument("--turn_gap", type=float, default=1.5,
                        help="Silence gap (s) for ground-truth turn splitting")
    # Diarization streaming config (high-latency defaults)
    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # ── Dataset ─────────────────────────────────────────────────────
    dataset = BazingaDataset(args.data_dir, sample_rate=SR)
    print(f"Found {len(dataset)} episodes under {args.data_dir}")

    # ── Load models ─────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading diarization model...")
    diar_model = (
        SortformerEncLabelModel.restore_from(args.diar_model_path)
        .eval()
        .to(device)
    )

    # ── Configure ────────────────────────────────────────────────────
    cfg = OmegaConf.structured(MultitalkerTranscriptionConfig())
    cfg.att_context_size = [70, 13]
    cfg.max_num_of_spks = args.max_num_of_spks
    diar_model._cfg.max_num_of_spks = args.max_num_of_spks

    for key in cfg:
        cfg[key] = None if cfg[key] == "None" else cfg[key]

    diar_model.streaming_mode = cfg.streaming_mode
    diar_model.sortformer_modules.log = cfg.log

    """
    Low latency mode
    """
    # diar_model.sortformer_modules.chunk_len = cfg.chunk_len if cfg.chunk_len > 0 else 6
    # diar_model.sortformer_modules.spkcache_len = cfg.spkcache_len
    # diar_model.sortformer_modules.chunk_left_context = cfg.chunk_left_context
    # diar_model.sortformer_modules.chunk_right_context = (
    #     cfg.chunk_right_context if cfg.chunk_right_context > 0 else 7
    # )
    # diar_model.sortformer_modules.fifo_len = cfg.fifo_len
    # diar_model.sortformer_modules.spkcache_refresh_rate = cfg.spkcache_refresh_rate

    """
    High latency mode
    """
    diar_model.sortformer_modules.chunk_len = 340
    diar_model.sortformer_modules.spkcache_len = 188
    diar_model.sortformer_modules.chunk_left_context = 0
    diar_model.sortformer_modules.chunk_right_context = 40
    diar_model.sortformer_modules.fifo_len = 40
    diar_model.sortformer_modules.spkcache_update_period = 300

    # ASR model is re-created per chunk to avoid CUDA graph state accumulation
    # across variable-length inputs (causes cudaErrorIllegalAddress on chunk 3+).
    asr_model_path = args.asr_model_path

    # ── Process each episode ────────────────────────────────────────
    num_processed = 0
    num_skipped = 0
    num_fail = 0
    tmp_dir = tempfile.mkdtemp(prefix="bazinga_offline_")

    for sample in tqdm(dataset):
        print(f"\n{'=' * 70}")
        print(f"Processing episode: {sample['conv_id']}")
        print(f"  Speakers : {sample['speakers']}")
        print(f"  Audio    : {sample['audio_path']}")
        print(f"{'=' * 70}")

        conv_id = sample["conv_id"]
        if "Season02" in conv_id:
            break

        save_dir = os.path.join(args.output_dir, conv_id)
        os.makedirs(save_dir, exist_ok=True)
        raw_audio = sample["audio"]
        T = raw_audio.shape[0]
        raw_transcript = sample["raw_transcript"]

        # transcript_chunks = chunk_words(raw_transcript)
        transcript_chunks = chunk_dialog(raw_transcript, min_dur=60.0, max_dur=300.0, gap_threshold=3.0)
        print(len(transcript_chunks))
        exit(0)
        # print("chunk_words with old")
        # for i in range(0, len(transcript_chunks)-1):
        #     print(f"gap{i}", transcript_chunks[i][-1]['end'], transcript_chunks[i+1][0]['start'])

        # print("chunk_words with new")
        # for i in range(0, len(transcript_chunks2)-1):
        #     print(f"gap{i}", transcript_chunks2[i][-1]['end'], transcript_chunks2[i+1][0]['start'])
        # exit(0)
        for chunk_id, chunk in enumerate(transcript_chunks):
            start_time = int(SR * chunk[0]["start"])
            end_time = int(SR * chunk[-1]["end"])
            print("chunkid", chunk_id, (end_time - start_time)/16000.0)
            if end_time > T:
                end_time = T
            audio = raw_audio[start_time:end_time]

            # Build ground-truth
            speaker_transcripts: Dict[str, List[Dict]] = defaultdict(list)
            for w in chunk:
                w["start"] -= float(start_time) / SR
                w["end"] -= float(start_time) / SR
                speaker_transcripts[w["speaker"]].append(w)
            vad_gt = transcription_to_vad(speaker_transcripts)

            chunk_dir = os.path.join(save_dir, f"CHUNK_{chunk_id}")
            diar_path = os.path.join(chunk_dir, "diart_pred.npy")
            word_list_path = os.path.join(chunk_dir, "transcript_pred.json")
            word_list_path_gt = os.path.join(chunk_dir, "transcript_gt.json")
            info_path = os.path.join(chunk_dir, "sample_info.json")
            vad_gt_path = os.path.join(chunk_dir, "vad_gt.json")

            if (os.path.exists(diar_path) and os.path.exists(word_list_path)
                    and os.path.exists(info_path)):
                print(f"  Skipping (already exists): {diar_path}")
                num_skipped += 1
                continue

            # try:
            # Step 1: Offline diarization
            diar_binary, diar_probs = run_offline_diarization(
                audio, diar_model, args.max_num_of_spks,
            )

            # Flush GPU cache so ASR model gets clean memory
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

            # Step 2: Offline ASR — fresh model each chunk to avoid CUDA graph state
            asr_model = ASRModel.restore_from(asr_model_path).eval().to(device)
            chunk_tag = f"{conv_id}_chunk{chunk_id}"
            asr_hyp = run_offline_asr(audio, asr_model, tmp_dir, chunk_tag)
            del asr_model
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

            # Step 3: Merge diarization + ASR
            word_list = merge_diar_and_asr(
                diar_probs, asr_hyp, args.max_num_of_spks,
            )

            # except Exception as e:
            #     print(f"  Error processing {conv_id} chunk {chunk_id}: {e}")
            #     import traceback
            #     traceback.print_exc()
            #     num_fail += 1
            #     continue

            # ── Save outputs ────────────────────────────────────────
            os.makedirs(chunk_dir, exist_ok=True)

            np.save(diar_path, diar_binary)

            with open(word_list_path, "w", encoding="utf-8") as fh:
                json.dump(word_list, fh, indent=2)

            with open(word_list_path_gt, "w", encoding="utf-8") as fh:
                json.dump(speaker_transcripts, fh, indent=2)

            with open(vad_gt_path, "w", encoding="utf-8") as fh:
                json.dump(vad_gt, fh, indent=2)

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
                "feat_len_sec": FRAME_LEN_SEC,
                "time_stamp": [start_time, end_time],
                "mode": "offline",
                "diar_config": {
                    "chunk_len": cfg.chunk_len,
                    "chunk_right_context": cfg.chunk_right_context,
                    "fifo_len": cfg.fifo_len,
                    "spkcache_len": cfg.spkcache_len,
                },
            }
            with open(info_path, "w", encoding="utf-8") as fh:
                json.dump(sample_info, fh, indent=2)

            print(f"  Saved: {diar_path}  shape={diar_binary.shape}")
            print(f"  Saved: {word_list_path}")
            print(f"  Saved: {info_path}")
            num_processed += 1
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    print(
        f"\nStep 1 (Bazinga — OFFLINE) complete. "
        f"Processed {num_processed}, skipped {num_skipped}, failed {num_fail}."
    )


if __name__ == "__main__":
    main()
