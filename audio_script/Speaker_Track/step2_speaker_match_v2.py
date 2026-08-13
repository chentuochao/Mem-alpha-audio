"""
Step 2: Speaker embedding extraction + cross-file matching  (runs in env2)

Walks the output directory produced by Step 1 and loads per-sample results
(sample_info.json + diart_pred.npy + transcript_pred.json).
For each audio file:
  1. Segments audio by diarization result
  2. Extracts speaker embedding per local speaker
  3. Registers into a global speaker pool (cosine-similarity matching,
     weighted-average embedding update)

Produces:
  - global_speaker_results.json  (final cross-file speaker mapping + transcripts)
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from audio_script.eval.eval_utils import (
    eval_der_seamlessinteraction,
    eval_cpwer_seamlessinteraction,
    best_match_tp_fp_fn,
    parse_turn,
    print_turns,
    vad_segments_to_binary,
    build_speaker_transcripts,
    parse_transcript_morespeakers,
)
from audio_script.eval.multitalker_metrics import (
    calculate_session_cpWER,
    normalize_string,
    compute_der_bruteforce,
)
from audio_script.Speaker_Track.speaker_pool import (
    EmbeddingBackend,
    WeSpeakerBackend,
    GlobalSpeaker,
    GlobalSpeakerPool,
    ASNormSpeakerPool,
    TwoPassSpeakerCluster,
    build_linker,
)

NEW_FORMAT_DATASETS = ["bazinga", "perltqa", "mosaic"]


def merge_adjacent_same_speaker(turns: List[Dict]) -> List[Dict]:
    """Fold consecutive same-speaker turns (in a time-sorted stream) into one.

    Mosaic GT is built by flattening two per-speaker turn lists and sorting by
    start time; a speaker's back-to-back segments (with no other speaker turn
    between them) then appear as adjacent same-speaker turns. Merge each such
    run into a single turn: ``start`` = first turn's start, ``end`` = max end
    (segments may overlap), ``text`` = space-joined. Non-text keys are taken
    from the first turn of the run.
    """
    merged: List[Dict] = []
    for t in turns:
        if merged and merged[-1]["speaker"] == t["speaker"]:
            prev = merged[-1]
            prev["end"] = max(prev["end"], t["end"])
            prev_text, cur_text = prev.get("text", "").strip(), t.get("text", "").strip()
            prev["text"] = " ".join(x for x in (prev_text, cur_text) if x)
        else:
            merged.append(dict(t))
    return merged

# ─── Audio segmentation helpers ──────────────────────────────────────
def segment_audio_by_diarization(
    diar_result: np.ndarray,
    frame_duration: float = 0.01,
    min_segment_duration: float = 0.3,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Convert a binary diarization matrix into per-speaker time segments.

    Args:
        diar_result: (num_frames, num_speakers) binary activity matrix
        frame_duration: seconds per frame (default 10 ms)
        min_segment_duration: drop segments shorter than this

    Returns:
        speaker_index -> [(start_sec, end_sec), ...]
    """
    num_frames, num_speakers = diar_result.shape
    speaker_segments: Dict[int, List[Tuple[float, float]]] = {}

    for spk_idx in range(num_speakers):
        activity = diar_result[:, spk_idx]
        if not activity.any():
            continue

        segments: List[Tuple[float, float]] = []
        in_seg = False
        start = 0

        for i in range(len(activity)):
            if activity[i] and not in_seg:
                start = i
                in_seg = True
            elif not activity[i] and in_seg:
                seg_s = start * frame_duration
                seg_e = i * frame_duration
                if seg_e - seg_s >= min_segment_duration:
                    segments.append((seg_s, seg_e))
                in_seg = False

        if in_seg:
            seg_s = start * frame_duration
            seg_e = len(activity) * frame_duration
            if seg_e - seg_s >= min_segment_duration:
                segments.append((seg_s, seg_e))

        if segments:
            speaker_segments[spk_idx] = segments

    return speaker_segments


def extract_speaker_audio(
    audio_file: str,
    segments: List[Tuple[float, float]],
    output_path: str,
    bias: float = 0.0,
) -> Optional[str]:
    """Concatenate the speaker's active segments and write to *output_path*."""
    audio, sr = sf.read(audio_file)
    if audio.ndim > 1:
        audio = audio[:, 0]

    chunks = []
    for start_sec, end_sec in segments:
        s = max(0, int(start_sec * sr + bias))
        e = min(len(audio), int(end_sec * sr + bias))
        if e > s:
            chunks.append(audio[s:e])

    if not chunks:
        return None

    sf.write(output_path, np.concatenate(chunks), sr)
    return output_path


# ─── Per-audio processing ────────────────────────────────────────────
def segment_duration(segments: List[Tuple[float, float]]) -> float:
    return sum(e - s for s, e in segments)


def process_single_audio(
    audio_file: str,
    word_list: Dict[str, List[Dict]],
    diar_result: np.ndarray,
    embedding_backend: EmbeddingBackend,
    temp_dir: str,
    unique_id: str = "",
    frame_duration: float = 0.08,
    bias: float = 0.0,
) -> Dict:
    """
    Given pre-computed diarization + ASR results for one audio file:
      1. Segment audio by diarization result
      2. Extract speaker embedding per local speaker

    Args:
        word_list: per-speaker word-level predictions from step1, e.g.
            {"speaker_0": [{"word": "hi", "start": 0.1, "end": 0.3}, ...]}
        unique_id: used to disambiguate temp files when multiple conversations
                   share the same audio filename (e.g. mixed_conv.wav).

    Returns:
        local_speaker_id -> {
            "embedding": np.ndarray,
            "text": str,
            "segments": [(start, end, words), ...],
        }
    """
    print(f"***** Processing embeddings: {audio_file}")

    speaker_texts: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    for speaker, segments in word_list.items():
        for segment in segments:
            speaker_texts[speaker].append(
                (segment["start"], segment["end"], segment["word"])
            )

    speaker_segments = segment_audio_by_diarization(
        diar_result, frame_duration=frame_duration
    )
    # Sort speakers by total active duration, longest first.
    speaker_segments = sorted(
        speaker_segments.items(),
        key=lambda x: segment_duration(x[1]),
        reverse=True,
    )

    local_speakers: Dict[str, Dict] = {}
    prefix = unique_id if unique_id else os.path.splitext(os.path.basename(audio_file))[0]

    for spk_idx, segments in speaker_segments:
        local_id = f"speaker_{spk_idx}"

        total_dur = segment_duration(segments)
        if total_dur < 4.0:
            print("  Skip short segment: ", total_dur, "s")
            continue
        if local_id not in speaker_texts:
            print(
                f"  {local_id} appear in diarization but not in ASR (word_list), "
                f"maybe it is too short!!!"
            )
            continue

        spk_audio_path = os.path.join(temp_dir, f"{prefix}_{local_id}.wav")
        result_path = extract_speaker_audio(
            audio_file, segments, spk_audio_path, bias=bias
        )
        if result_path is None:
            continue

        embedding = embedding_backend.extract(result_path)

        text_segs = speaker_texts[local_id]
        full_text = " ".join(w for _, _, w in text_segs)

        local_speakers[local_id] = {
            "embedding": embedding,
            "text": full_text,
            "segments": text_segs,
            "duration": total_dur,
        }

        print(
            f"  {local_id}: {len(segments)} segment(s), "
            f"{total_dur:.1f}s total, embedding extracted"
        )

    return local_speakers


def merge_speakers_by_global(
    diar_result: np.ndarray,
    word_list: Dict[str, List],
    local_to_global: Dict[str, str],
) -> Tuple[np.ndarray, Dict[str, List], Dict[str, str]]:
    """
    diar_result - shape (num_frames, num_speakers)
    word_list - {
        "speaker_0": [{"word": "hello", "start": 0.0, "end": 1.0}, ...],
        "speaker_1": [{"word": "world", "start": 1.0, "end": 2.0}, ...],
    }
    local_to_global - {
        "speaker_0": "global_speaker_0",
        "speaker_1": "global_speaker_1",
    }
    Merge local speakers that map to the same global speaker.

    *** Important ***: the index of diar_result is the local speaker id, not the
    global speaker id. First column is speaker_0, second column is speaker_1, etc.

    OR-merges diarization columns and reassigns speaker labels in word_list.
    For merged groups, the speaker ID with the longest diarization duration
    is kept as the representative label.  Unmerged speakers keep their
    original label.

    Returns:
        (merged_diar, merged_word_list, old_to_new_speaker_map)
    """
    global_to_local_cols: Dict[str, List[int]] = defaultdict(list)
    for local_id, global_name in local_to_global.items():
        spk_idx = int(local_id.split("_")[-1])
        global_to_local_cols[global_name].append(spk_idx)
    # global_to_local_cols: {global_name: [local_speaker_indices]},
    # e.g. {"global_speaker_0": [0, 2, 4], "global_speaker_1": [1, 3, 5]}
    col_durations = diar_result.sum(axis=0)

    merged_columns = []
    merged_word_list: Dict[str, List] = defaultdict(list)
    local_to_global_new: Dict[str, str] = {}

    for global_name in sorted(global_to_local_cols):
        col_indices = global_to_local_cols[global_name]
        best_col = max(col_indices, key=lambda c: col_durations[c])
        representative = f"speaker_{best_col}"

        # Merge diart prediction columns sharing the same global speaker.
        if len(col_indices) > 1:
            merged_col = diar_result[:, col_indices].max(axis=1)
        else:
            merged_col = diar_result[:, col_indices[0]]
        merged_columns.append(merged_col)

        # Merge transcripts sharing the same global speaker.
        for col_idx in col_indices:
            local_id = f"speaker_{col_idx}"
            merged_word_list[representative].extend(word_list[local_id])

        local_to_global_new[representative] = global_name

    merged_diar = np.column_stack(merged_columns)
    for spk, segments in merged_word_list.items():
        segments.sort(key=lambda x: x["start"])
        merged_word_list[spk] = segments

    return merged_diar, merged_word_list, local_to_global_new


def discover_samples(data_dir: str, season_filter: Optional[List[str]] = None) -> List[Dict]:
    """
    Walk the directory tree and return all sample entries (in sorted path
    order) matching ``data_dir/*/*/sample_info.json`` produced by Step 1.

    No speaker clustering is performed - samples are returned as a single flat
    list so they are processed strictly in path order (important for the
    order-dependent online/greedy linker).

    Each entry is the parsed ``sample_info.json`` dict augmented with
    ``sample_dir``, ``diart_path`` and ``pred_transcript_path``.
    """

    samples: List[Dict] = []
    # print(glob.glob(os.path.join(data_dir, "*", "*", "sample_info.json")))
    for info_path in sorted(glob.glob(os.path.join(data_dir, "*", "*", "sample_info.json"))):
        sample_dir = os.path.dirname(info_path)
        diar_path = os.path.join(sample_dir, "diart_pred.npy")
        transcript_path = os.path.join(sample_dir, "transcript_pred.json")
        if not os.path.exists(diar_path) or not os.path.exists(transcript_path):
            continue

        # Optional season filter: only keep samples whose path contains one of
        # the given substrings (e.g. ["Season01"]). Empty/None = keep all.
        if season_filter and not any(s in sample_dir for s in season_filter):
            continue

        with open(info_path, "r") as f:
            info = json.load(f)
        info["sample_dir"] = sample_dir
        info["diart_path"] = diar_path
        info["pred_transcript_path"] = transcript_path
        samples.append(info)

    return samples


def eval_der_cpwer(entry, word_list, diar_result, local_speakers):
    """
        entry - sample_info.json
        word_list - transcript_pred.json
        diar_result - diart_pred.npy

    Supports two dataset formats:
      - "bazinga": entry has "speakers" (list), "vad_path" (combined
        {speaker: [{start,end}]} JSON), "transcript_path" (combined
        {speaker: [{word,start,end,...}]} JSON).
      - default (InterAct / SeamlessInteraction): entry has "spk_pair",
        "vad1_path"/"vad2_path", "transcript1_path"/"transcript2_path".
    """
    frame_duration = entry.get("feat_len_sec", 0.08)

    if entry.get("dataset") in NEW_FORMAT_DATASETS:
        # ── Bazinga format ────────────────────────────────────────────────s
        speakers = entry["speakers"]

        with open(entry["vad_path"]) as f:
            vad_data = json.load(f)   # {speaker: [{start, end}, ...]}
        with open(entry["transcript_path"]) as f:
            trans_data = json.load(f)  # {speaker: [{word, start, end, ...}, ...]}

        # DER — build gt_matrix directly from in-memory VAD segments
        total_frames = diar_result.shape[0]
        gt_matrix = []
        speaker_gt_der = []
        print("existing speaker ", speakers, vad_data.keys())
        for spk in speakers:
            if spk not in vad_data:
                continue
            gt_array = vad_segments_to_binary(vad_data[spk], total_frames, frame_duration)
            gt_matrix.append(gt_array)
            speaker_gt_der.append(spk)
        gt_matrix = np.stack(gt_matrix, axis=0)
        print(diar_result.T.shape, gt_matrix.shape)
        der, der_details = compute_der_bruteforce(
            diar_result.T, gt_matrix, frame_duration=frame_duration
        )
        der_details["speaker_gt"] = speaker_gt_der

        best_perm_der_new = []
        for _i in der_details["col_ind"].tolist():
            if _i >= len(local_speakers):
                break
            best_perm_der_new.append(local_speakers[_i])
        best_perm_der = best_perm_der_new

        # cpWER — build references from in-memory GT transcripts. GT entries are
        # word-level ({"word": ...}, Bazinga) or turn-level ({"text": ...},
        # PerLTQA channel_map GT); take whichever is present. (Empty references
        # here make calculate_session_cpWER hang, so this must not be empty.)
        spk_hypothesis, speakers_pred = build_speaker_transcripts(
            word_list, pad_char=" "
        )
        spk_reference = []
        for spk in speakers:
            if spk not in trans_data:
                continue
            ref_text = normalize_string(
                " ".join((w.get("word") or w.get("text") or "")
                         for w in trans_data[spk])
            )
            spk_reference.append(ref_text)
        cpwer, _, _, best_perm_idx = calculate_session_cpWER(
            spk_hypothesis, spk_reference, limit_hypo_number=True
        )
        best_perm_cpwer = [speakers_pred[i] for i in best_perm_idx]

    else:
        # ── InterAct / SeamlessInteraction format ─────────────────────────
        spk_pair = entry.get("spk_pair", "")
        speaker0, speaker1 = spk_pair.split("_", 1)

        gt_vad_files = {
            speaker0: entry["vad1_path"],
            speaker1: entry["vad2_path"],
        }
        der, best_perm_der, der_details = eval_der_seamlessinteraction(
            diar_result, gt_vad_files, frame_duration
        )
        best_perm_der = best_perm_der.tolist()
        best_perm_der_new = []
        for _i in best_perm_der:
            if _i >= len(local_speakers):
                break
            best_perm_der_new.append(local_speakers[_i])
        best_perm_der = best_perm_der_new

        gt_trans_files = {
            speaker0: entry["transcript1_path"],
            speaker1: entry["transcript2_path"],
        }
        cpwer, best_perm_cpwer = eval_cpwer_seamlessinteraction(
            word_list, gt_trans_files, limit_hypo_number=True
        )

    print(
        f"  DER: {der:.4f}  "
        f"(miss={der_details['miss']:.2f}s, fa={der_details['fa']:.2f}s, "
        f"conf={der_details['conf']:.2f}s, total={der_details['total']:.2f}s)"
    )
    print(f"  cpWER: {cpwer:.4f}")
    print(f"  Best permutation: DER = {best_perm_der}, cpWER = {best_perm_cpwer}")
    return der, cpwer, best_perm_der, best_perm_cpwer


# ─── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Speaker embedding extraction + cross-file matching (env2)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root output directory from Step 1 containing "
             "{spk_pair}/{conv_id}/sample_info.json sub-folders",
    )
    parser.add_argument(
        "--embedding_model_dir",
        type=str,
        required=True,
        help="Path to WeSpeaker model directory for speaker embeddings",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.7,
        help="Cosine similarity threshold for cross-file speaker matching",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save final results. Defaults to --data_dir.",
    )
    parser.add_argument(
        "--embedding_device",
        type=str,
        default="cuda:0",
        help="Device for speaker embedding extraction",
    )
    parser.add_argument(
        "--pool_path",
        type=str,
        default=None,
        help="Single .npz file holding the cross-run global speaker pool. "
             "If it exists it is loaded before processing (so new data is "
             "matched against / appended to existing speakers); the updated "
             "pool is written back here afterwards. Omit for a one-shot run.",
    )
    parser.add_argument(
        "--update_pool",
        action="store_true",
        help="whether update the pool file or not",
    )

    parser.add_argument(
        "--season_filter",
        type=str,
        nargs="*",
        default=[],
        help="Optional substrings (e.g. Season01 Season02). Only samples whose "
             "path contains one of these are processed; empty = all.",
    )
    parser.add_argument(
        "--linker",
        type=str,
        default="greedy",
        choices=["greedy", "asnorm", "twopass"],
        help="Cross-file speaker linker: 'greedy' (online cosine, original), "
             "'asnorm' (online AS-norm + robust centroids), or "
             "'twopass' (batch agglomerative/spectral clustering).",
    )
    parser.add_argument(
        "--cluster_method",
        type=str,
        default="ahc",
        choices=["ahc", "spectral"],
        help="Clustering algorithm for '--linker twopass'.",
    )
    parser.add_argument(
        "--centroid_mode",
        type=str,
        default="weighted_mean",
        choices=["weighted_mean", "trimmed_mean", "medoid"],
        help="Centroid update strategy for '--linker asnorm'.",
    )
    parser.add_argument(
        "--use_asnorm_affinity",
        action="store_true",
        help="AS-norm the affinity matrix before clustering ('--linker twopass').",
    )
    parser.add_argument(
        "--debug_dir",
        type=str,
        default="./debug",
        help="If set (with '--linker twopass'), save clustering figures "
             "(PCA scatter + affinity heatmap) to this folder. "
             "Defaults to '<output_dir>/debug' when only the flag is given.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "speaker_segments")
    os.makedirs(temp_dir, exist_ok=True)

    # ── Discover samples from Step 1 output ──────────────────────────
    # args.data_dir format -> args.data_dir/*/*/sample_info.json
    samples = discover_samples(args.data_dir, season_filter=args.season_filter)
    print(f"Found {len(samples)} sample(s) under {args.data_dir}")
    if not samples:
        print("No samples found. Check your --data_dir path.")
        return

    # ── Load embedding model ─────────────────────────────────────────
    print("Loading speaker embedding model...")
    embedding_backend = WeSpeakerBackend(
        model_dir=args.embedding_model_dir, device=args.embedding_device
    )

    all_results: Dict[str, Dict] = {}

    # ── Build the cross-file speaker linker ──────────────────────────
    if args.linker == "asnorm":
        linker = build_linker(
            "asnorm",
            similarity_threshold=args.similarity_threshold,
            centroid_mode=args.centroid_mode,
        )
    elif args.linker == "twopass":
        linker = build_linker(
            "twopass",
            similarity_threshold=args.similarity_threshold,
            method=args.cluster_method,
            use_asnorm=args.use_asnorm_affinity,
        )
    else:
        linker = build_linker(
            "greedy", similarity_threshold=args.similarity_threshold
        )
    print(f"Using speaker linker: {args.linker}")

    # ── Restore a persistent pool for incremental (season-by-season) runs ──
    if args.pool_path and os.path.exists(args.pool_path):
        linker.load(args.pool_path)
    elif args.pool_path:
        print(f"[pool] no existing pool at {args.pool_path}; starting fresh")

    speaker_cluster_gt: Dict[str, List[str]] = {}
    speaker_cluster_pred: Dict[str, List[str]] = {}
    all_ders: List[float] = []
    all_cpwers: List[float] = []
    err_info: List[str] = []

    # ── Phase 1: per-file embedding extraction + registration ────────
    # Runs uniformly for online (greedy/asnorm) and batch (twopass) linkers:
    # every local speaker is registered with the linker, and the per-entry
    # context needed for phase 2 is stashed.  The authoritative global mapping
    # is read back from linker.finalize() afterwards.
    entry_contexts: List[Dict] = []
    for entry in samples:
        if entry.get("dataset") in NEW_FORMAT_DATASETS:
            speaker_gt = entry["speakers"]
            bias = float(entry["time_stamp"][0])  # unit in sample
            spk_pair = entry.get("conv_id", "")
            chunk_id = entry.get("chunk_id", "")
            conv_id = f"CHUNK_{chunk_id}"
        else:
            conv_id = entry.get("conv_id", "")
            spk_pair = entry.get("spk_pair", "")
            speaker_gt = spk_pair.split("_")
            bias = 0

        audio_file = entry["audio_file"]
        diar_path = entry["diart_path"]
        pred_transcript_path = entry["pred_transcript_path"]
        frame_duration = entry.get("feat_len_sec", 0.08)
        result_key = (
            f"{spk_pair}/{conv_id}" if spk_pair and conv_id else audio_file
        )
        unique_id = f"{spk_pair}_{conv_id}" if spk_pair and conv_id else ""
        print(f"\n{'=' * 60}")
        print(f"\nProcessing entry: {result_key}, {pred_transcript_path}")
        print(f"{'=' * 60}")

        output_sample_folder = os.path.join(output_dir, spk_pair, conv_id)
        os.makedirs(output_sample_folder, exist_ok=True)

        # Annotate ground-truth speaker ids.
        for spk in speaker_gt:
            speaker_cluster_gt.setdefault(spk, []).append(spk)

        with open(pred_transcript_path, "r") as f:
            word_list = json.load(f)
        diar_result = np.load(diar_path)

        # ── Evaluate DER & cpWER ─────────────────────────────────────
        der, cpwer, best_perm_der, best_perm_cpwer = eval_der_cpwer(
            entry, word_list, diar_result, list(word_list.keys())
        )
        all_ders.append(der)
        all_cpwers.append(cpwer)

        local_speakers = process_single_audio(
            audio_file,
            word_list,
            diar_result,
            embedding_backend,
            temp_dir,
            unique_id=unique_id,
            frame_duration=frame_duration,
            bias=bias,
        )

        # Register with the linker.  Online linkers resolve the mapping
        # immediately; batch linkers buffer and resolve in finalize().
        linker.add_audio_speakers(result_key, local_speakers)

        entry_contexts.append(
            {
                "entry": entry,
                "result_key": result_key,
                "spk_pair": spk_pair,
                "conv_id": conv_id,
                "audio_file": audio_file,
                "speaker_gt": speaker_gt,
                "best_perm_cpwer": best_perm_cpwer,
                "word_list": word_list,
                "output_sample_folder": output_sample_folder,
            }
        )

    # ── Resolve the authoritative global mapping ─────────────────────
    all_mappings = linker.finalize()

    # ── Persist the updated pool for the next incremental run ─────────
    if args.pool_path is not None and args.update_pool:
        print(f"saving to {args.pool_path}....")
        linker.save(args.pool_path)

    # ── Visualise the two-pass clustering for debugging ──────────────
    if isinstance(linker, TwoPassSpeakerCluster):
        debug_dir = args.debug_dir or os.path.join(output_dir, "debug")
        linker.visualize(debug_dir, prefix=f"twopass_{args.cluster_method}")

    # ── Phase 2: apply global mapping, parse + write transcripts ─────
    for ctx in entry_contexts:
        entry = ctx["entry"]
        result_key = ctx["result_key"]
        spk_pair = ctx["spk_pair"]
        conv_id = ctx["conv_id"]
        audio_file = ctx["audio_file"]
        speaker_gt = ctx["speaker_gt"]
        best_perm_cpwer = ctx["best_perm_cpwer"]
        word_list = ctx["word_list"]
        output_sample_folder = ctx["output_sample_folder"]

        local_to_global = all_mappings.get(result_key, {})
        # print("local_to_global", local_to_global, list(word_list.keys()))

        for local_id, global_id in local_to_global.items():
            if local_id in best_perm_cpwer:
                gt_id = best_perm_cpwer.index(local_id)
                spk_label_gt = speaker_gt[gt_id]
            else:
                # Local speaker not matched with GT -> mark as false positive.
                spk_label_gt = f"FP_{spk_pair}_{conv_id}"
            speaker_cluster_pred.setdefault(global_id, []).append(spk_label_gt)

        # ── Parse conversation-level transcripts ─────────────────────
        if entry.get("dataset") in NEW_FORMAT_DATASETS:
            dialog = parse_transcript_morespeakers(word_list, interval_character=" ")
        else:
            dialog = parse_transcript_morespeakers(word_list)

        dialog_pred = []
        for sent in dialog:
            if sent["speaker"] in local_to_global:
                sent["speaker"] = local_to_global[sent["speaker"]]
                dialog_pred.append(sent)

        dataset = entry.get("dataset")
        if dataset in ("perltqa", "mosaic"):
            # PerLTQA / Mix_Mosaic GT is ALREADY turn-annotated
            # (transcript_gt.json = {speaker: [{speaker, start, end, text}, ...]}).
            # Each entry is a turn, so we build dialog_gt directly — no
            # word->turn merge (parse_transcript_morespeakers).
            with open(entry["transcript_path"], "r") as f:
                trans_data = json.load(f)
            dialog_gt = [
                {"speaker": spk,
                 "start": t["start"], "end": t["end"],
                 "text": t.get("text", t.get("word", ""))}
                for spk, turns in trans_data.items() for t in turns
            ]
        elif dataset == "bazinga":
            # Bazinga GT is word-level ({speaker: [{word, start, end, ...}]});
            # merge words into turns via the annotator.
            with open(entry["transcript_path"], "r") as f:
                trans_data = json.load(f)
            dialog_gt = parse_transcript_morespeakers(
                trans_data, interval_character=" "
            )
        else:
            # InterAct / SeamlessInteraction: two per-speaker turn files.
            dialog_gt = []
            with open(entry["transcript1_path"], "r") as f:
                dialog_gt.extend(json.load(f))
            with open(entry["transcript2_path"], "r") as f:
                dialog_gt.extend(json.load(f))
        dialog_gt.sort(key=lambda x: x["start"])
        if dataset == "mosaic":
            # After the flatten+sort, a speaker's consecutive segments show up
            # as adjacent same-speaker turns — fold them back into single turns.
            dialog_gt = merge_adjacent_same_speaker(dialog_gt)

        dialog_gt_json = parse_turn(dialog_gt)
        # print_turns(dialog_gt_json)

        dialog_pred_json = parse_turn(dialog_pred)
        # print_turns(dialog_pred_json)

        all_results[result_key] = {
            "spk_pair": spk_pair,
            "conv_id": conv_id,
            "audio_file": audio_file,
            "dialog": dialog,
            "dialog_gt": dialog_gt,
            "local_to_global_mapping": local_to_global,
        }
        with open(os.path.join(output_sample_folder, "parsed_dialog_pred.json"), "w") as f:
            json.dump(dialog_pred_json, f, indent=2)
        with open(os.path.join(output_sample_folder, "parsed_dialog_gt.json"), "w") as f:
            json.dump(dialog_gt_json, f, indent=2)


    # ── Build speaker_map from speaker_cluster_pred ──────────────────
    speaker_map: Dict[str, str] = {}
    for global_spk, local_spk_list in speaker_cluster_pred.items():
        counter = Counter(local_spk_list)
        most_speaker_id = counter.most_common(1)[0][0]
        speaker_map[global_spk] = most_speaker_id
    print(speaker_map)

    with open(os.path.join(output_dir, "speaker_map.json"), "w") as f:
        json.dump(speaker_map, f, indent=2)

    with open(os.path.join(output_dir, "raw_speaker_tracking.json"), "w") as f:
        json.dump(
            {
                "speaker_cluster_pred": speaker_cluster_pred,
                "speaker_cluster_gt": speaker_cluster_gt,
            },
            f,
            indent=2,
        )

    # ── Accuracy vs. ground-truth ────────────────────────────────────
    speaker_cluster_pred_list = list(speaker_cluster_pred.values())
    speaker_cluster_gt_list = list(speaker_cluster_gt.values())
    tp_total, fp_total, fn_total = best_match_tp_fp_fn(
        speaker_cluster_pred_list, speaker_cluster_gt_list
    )
    print(f"TP: {tp_total}, FP: {fp_total}, FN: {fn_total}")
    print(f"Accuracy: {tp_total / (tp_total + fp_total + fn_total)}")

    avg_der = float(np.mean(all_ders))
    median_der = float(np.median(all_ders))
    avg_cpwer = float(np.mean(all_cpwers))
    median_cpwer = float(np.median(all_cpwers))

    print(f"  Avg DER before merge = {avg_der:.4f}")
    print(f"  Median DER before merge = {median_der:.4f}")
    print(f"  Avg cpWER before merge = {avg_cpwer:.4f}")
    print(f"  Median cpWER before merge = {median_cpwer:.4f}")

    print("err_info: ", err_info)


if __name__ == "__main__":
    main()
