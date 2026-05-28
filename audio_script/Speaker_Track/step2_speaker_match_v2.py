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
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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


# ─── Embedding backends ──────────────────────────────────────────────

class EmbeddingBackend(ABC):
    @abstractmethod
    def extract(self, audio_file: str) -> np.ndarray:
        """Return a 1-D numpy embedding for the given audio file."""
        ...


class WeSpeakerBackend(EmbeddingBackend):
    def __init__(self, model_dir: str, device: int = 0):
        import wespeaker
        self.model = wespeaker.load_model(model_dir)
        self.model.set_device(device)

    def extract(self, audio_file: str) -> np.ndarray:
        embedding = self.model.extract_embedding(audio_file)
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        return embedding.flatten()


# ─── Global speaker data ─────────────────────────────────────────────


@dataclass
class GlobalSpeaker:
    """A speaker in the global pool, aggregated across multiple audio files."""

    global_id: int
    name: str
    embedding: np.ndarray
    weight: int = 1
    transcriptions: List[Dict] = field(default_factory=list)


# ─── Global speaker pool ─────────────────────────────────────────────


class GlobalSpeakerPool:
    """
    Maintains a pool of globally-unique speakers.  Local speakers from
    each audio file are matched against the pool one-by-one.
    """

    def __init__(self, similarity_threshold: float = 0.65):
        self.similarity_threshold = similarity_threshold
        self.speakers: List[GlobalSpeaker] = []
        self._next_id = 0

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _create_speaker(
        self, embedding: np.ndarray, transcription: Dict
    ) -> GlobalSpeaker:
        spk = GlobalSpeaker(
            global_id=self._next_id,
            name=f"GLOBAL_SPK_{self._next_id}",
            embedding=embedding.copy(),
            weight=1,
            transcriptions=[transcription],
        )
        self.speakers.append(spk)
        self._next_id += 1
        return spk

    def _find_closest(
        self, embedding: np.ndarray
    ) -> Tuple[Optional[GlobalSpeaker], float]:
        if not self.speakers:
            return None, -1.0
        best_spk, best_sim = None, -1.0
        for spk in self.speakers:
            sim = self.cosine_similarity(embedding, spk.embedding)
            if sim > best_sim:
                best_spk, best_sim = spk, sim
        return best_spk, best_sim

    def register_speaker(
        self, embedding: np.ndarray, transcription: Dict
    ) -> GlobalSpeaker:
        """
        Match a local speaker embedding against the global pool.
        Merges into existing speaker (weighted-average embedding update) or
        creates a new one.
        """
        best_spk, best_sim = self._find_closest(embedding)

        if best_spk is not None and best_sim >= self.similarity_threshold:
            old_w = best_spk.weight
            new_w = old_w + 1
            best_spk.embedding = (best_spk.embedding * old_w + embedding) / new_w
            best_spk.weight = new_w
            best_spk.transcriptions.append(transcription)
            print(
                f"  -> Matched {best_spk.name} (sim={best_sim:.4f}, weight={new_w})"
            )
            return best_spk

        new_spk = self._create_speaker(embedding, transcription)
        print(f"  -> New {new_spk.name} (best_sim={best_sim:.4f})")
        return new_spk

    def register_audio_speakers(
        self, audio_file: str, local_speakers: Dict
    ) -> Dict[str, str]:
        """
        Register all local speakers from one audio file into the global pool.

        Returns:
            Mapping from local speaker id to global speaker name.
        """
        print(f"\nRegistering speakers from: {audio_file}")
        mapping = {}
        for local_id, info in local_speakers.items():
            print(f"  Local speaker '{local_id}':")
            transcription = {
                "audio_file": audio_file,
                "local_speaker_id": local_id,
                "text": info["text"],
                "segments": [
                    {"start": s, "end": e, "words": w}
                    for s, e, w in info["segments"]
                ],
            }
            global_spk = self.register_speaker(info["embedding"], transcription)
            mapping[local_id] = global_spk.name
        return mapping

    def summary(self):
        print(f"\n{'=' * 70}")
        print(f"Global Speaker Pool: {len(self.speakers)} unique speaker(s)")
        print(f"{'=' * 70}")
        for spk in self.speakers:
            print(f"\n  {spk.name}  (weight={spk.weight})")
            for t in spk.transcriptions:
                print(f"    [{t['audio_file']}] local_id={t['local_speaker_id']}")
                text_preview = t["text"][:120]
                if len(t["text"]) > 120:
                    text_preview += "..."
                print(f"      Text: {text_preview}")


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


def discover_samples(data_dir: str) -> List[Dict]:
    """
    Walk the directory tree and find all sample folders containing
    sample_info.json produced by Step 1.

    First-level sub-folder names are expected to be speaker-pair IDs joined
    by ``_`` (e.g. ``P0043_P0108``).  Sub-folders that share any speaker ID
    are transitively clustered together.

    Returns a list of cluster dicts, each with:
      - ``speaker_ids``: sorted list of all unique speaker IDs in the cluster
      - ``samples``: list of entry dicts (augmented with ``sample_dir``,
        ``diart_path``, ``transcript_path``)
    """
    # ── 1. Collect samples grouped by first-level sub-folder ─────────
    subfolder_samples: Dict[str, List[Dict]] = defaultdict(list)
    for info_path in sorted(glob.glob(os.path.join(data_dir, "*", "*", "sample_info.json"))):
        sample_dir = os.path.dirname(info_path)
        diar_path = os.path.join(sample_dir, "diart_pred.npy")
        transcript_path = os.path.join(sample_dir, "transcript_pred.json")
        if not os.path.exists(diar_path) or not os.path.exists(transcript_path):
            continue
        with open(info_path, "r") as f:
            info = json.load(f)
        info["sample_dir"] = sample_dir
        info["diart_path"] = diar_path
        info["pred_transcript_path"] = transcript_path
        group_key = os.path.basename(os.path.dirname(sample_dir))
        subfolder_samples[group_key].append(info)

    # ── 2. Union-Find to cluster sub-folders sharing a speaker ID ────
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    subfolder_speakers: Dict[str, List[str]] = {}
    for folder_name in subfolder_samples:
        spk_ids = folder_name.split("_")
        subfolder_speakers[folder_name] = spk_ids
        for sid in spk_ids:
            parent.setdefault(sid, sid)
        for i in range(1, len(spk_ids)):
            union(spk_ids[0], spk_ids[i])

    # ── 3. Group sub-folders by their cluster root ───────────────────
    cluster_map: Dict[str, Dict] = defaultdict(
        lambda: {"speaker_ids": set(), "samples": []}
    )
    for folder_name, entries in subfolder_samples.items():
        root = find(subfolder_speakers[folder_name][0])
        cluster_map[root]["speaker_ids"].update(subfolder_speakers[folder_name])
        cluster_map[root]["samples"].extend(entries)

    return [
        {"speaker_ids": sorted(v["speaker_ids"]), "samples": v["samples"]}
        for v in cluster_map.values()
    ]


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

    if entry.get("dataset") == "bazinga":
        # ── Bazinga format ────────────────────────────────────────────────
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

        # cpWER — build references from in-memory GT transcripts (Bazinga
        # word-level dicts use "word" key instead of "text")
        spk_hypothesis, speakers_pred = build_speaker_transcripts(
            word_list, pad_char=" "
        )
        spk_reference = []
        for spk in speakers:
            if spk not in trans_data:
                continue
            ref_text = normalize_string(
                " ".join(w["word"] for w in trans_data[spk] if w.get("word"))
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
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "speaker_segments")
    os.makedirs(temp_dir, exist_ok=True)

    # ── Discover samples from Step 1 output ──────────────────────────
    # args.data_dir format -> args.data_dir/*/*/sample_info.json
    clusters = discover_samples(args.data_dir)
    for cluster in clusters:
        print(
            f"Processing cluster: {cluster['speaker_ids']} "
            f"with {len(cluster['samples'])} sample(s)"
        )

    total_samples = sum(len(c["samples"]) for c in clusters)
    print(
        f"Found {total_samples} sample(s) in {len(clusters)} "
        f"speaker cluster(s) under {args.data_dir}"
    )
    if not clusters:
        print("No samples found. Check your --data_dir path.")
        return

    # ── Load embedding model ─────────────────────────────────────────
    print("Loading speaker embedding model...")
    embedding_backend = WeSpeakerBackend(
        model_dir=args.embedding_model_dir, device=args.embedding_device
    )

    all_results: Dict[str, Dict] = {}
    global_pool = GlobalSpeakerPool(similarity_threshold=args.similarity_threshold)
    speaker_cluster_gt: Dict[str, List[str]] = {}
    speaker_cluster_pred: Dict[str, List[str]] = {}
    all_ders: List[float] = []
    all_cpwers: List[float] = []

    err_info: List[str] = []
    for cluster in clusters:
        # Each cluster uses one global speaker pool.
        speaker_ids = cluster["speaker_ids"]
        samples = cluster["samples"]
        print(f"{'=' * 60}")
        print(f"Cluster speakers: {speaker_ids}  ({len(samples)} sample(s))")
        print(f"{'=' * 60}")

        for entry in samples:
            spk_pair = entry.get("spk_pair", "")
            print(spk_pair)
            if entry.get("dataset") == "bazinga":
                speaker_gt = entry["speakers"]
                bias = float(entry["time_stamp"][0])  # unit: samples
            else:
                speaker_gt = spk_pair.split("_")
                bias = 0
            conv_id = entry.get("conv_id", "")
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

            # Annotate ground-truth speaker ids per cluster.
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

            local_to_global = global_pool.register_audio_speakers(
                result_key, local_speakers
            )
            print("local_to_global", local_to_global, list(word_list.keys()))

            for local_id, global_id in local_to_global.items():
                if local_id in best_perm_cpwer:
                    gt_id = best_perm_cpwer.index(local_id)
                    spk_label_gt = speaker_gt[gt_id]
                else:
                    # Local speaker not matched with GT -> mark as false positive.
                    spk_label_gt = f"FP_{spk_pair}_{conv_id}"
                speaker_cluster_pred.setdefault(global_id, []).append(spk_label_gt)

            # ── Parse conversation-level transcripts ─────────────────────
            if entry.get("dataset") == "bazinga":
                dialog = parse_transcript_morespeakers(word_list, interval_character=" ")
            else:
                dialog = parse_transcript_morespeakers(word_list)

            dialog_pred = []
            for sent in dialog:
                if sent["speaker"] in local_to_global:
                    sent["speaker"] = local_to_global[sent["speaker"]]
                    dialog_pred.append(sent)

            if entry.get("dataset") == "bazinga":
                # transcript_path: {speaker: [{word, start, end, ...}]}
                with open(entry["transcript_path"], "r") as f:
                    trans_data = json.load(f)
                dialog_gt = parse_transcript_morespeakers(
                    trans_data, interval_character=" "
                )
            else:
                dialog_gt = []
                with open(entry["transcript1_path"], "r") as f:
                    dialog_gt.extend(json.load(f))
                with open(entry["transcript2_path"], "r") as f:
                    dialog_gt.extend(json.load(f))
            dialog_gt.sort(key=lambda x: x["start"])

            dialog_gt_json = parse_turn(dialog_gt)
            print_turns(dialog_gt_json)

            dialog_pred_json = parse_turn(dialog_pred)
            print_turns(dialog_pred_json)

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
