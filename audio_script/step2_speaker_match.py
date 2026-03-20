"""
Step 2: Speaker embedding extraction + cross-file matching  (runs in env2)

Reads intermediate outputs from Step 1 (seglst JSON + diarization .npy).
For each audio file:
  1. Segments audio by diarization result
  2. Extracts speaker embedding per local speaker
  3. Registers into a global speaker pool (cosine-similarity matching,
     weighted-average embedding update)

Produces:
  - global_speaker_results.json  (final cross-file speaker mapping + transcripts)
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
import soundfile as sf


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
            embedding=embedding.clone(),
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
) -> Optional[str]:
    """Concatenate the speaker's active segments and write to *output_path*."""
    audio, sr = sf.read(audio_file)
    if audio.ndim > 1:
        audio = audio[:, 0]

    chunks = []
    for start_sec, end_sec in segments:
        s = max(0, int(start_sec * sr))
        e = min(len(audio), int(end_sec * sr))
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
    seglst_dict_list: List[Dict],
    diar_result: np.ndarray,
    embedding_backend: EmbeddingBackend,
    temp_dir: str,
) -> Dict:
    """
    Given pre-computed diarization + ASR results for one audio file:
      1. Segment audio by diarization result
      2. Extract speaker embedding per local speaker

    Returns:
        local_speaker_id -> {
            "embedding": np.ndarray,
            "text": str,
            "segments": [(start, end, words), ...],
        }
    """
    print(f"\n{'=' * 70}")
    print(f"Processing embeddings: {audio_file}")
    print(f"{'=' * 70}")

    # Collect per-speaker transcript segments from ASR output
    speaker_texts: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    for seg in seglst_dict_list:
        speaker = seg.get("speaker", "Unknown")
        start_time = seg.get("start_time", 0.0)
        end_time = seg.get("end_time", 0.0)
        words = seg.get("words", "")
        speaker_texts[speaker].append((start_time, end_time, words))

    # Segment audio using binary diarization output
    speaker_segments = segment_audio_by_diarization(diar_result)

    # Extract embedding per local speaker
    local_speakers: Dict[str, Dict] = {}
    basename = os.path.splitext(os.path.basename(audio_file))[0]

    for spk_idx, segments in speaker_segments.items():
        local_id = f"speaker_{spk_idx}"

        # Skip if the segment total duraion is less than 4 second
        total_dur = segment_duration(segments)
        if total_dur < 4:
            print("  Skip short segment: ", total_dur, "s")
            continue

        spk_audio_path = os.path.join(temp_dir, f"{basename}_{local_id}.wav")
        result_path = extract_speaker_audio(audio_file, segments, spk_audio_path)
        if result_path is None:
            continue

        embedding = embedding_backend.extract(result_path)

        text_segs = speaker_texts.get(local_id, [])
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


# ─── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Speaker embedding extraction + cross-file matching (env2)"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to step1_manifest.json produced by Step 1",
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
        default="./demo_output",
        help="Directory to save final results",
    )
    parser.add_argument(
        "--embedding_device",
        type=str,
        default="cuda:0",
        help="Device for speaker embedding extraction",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, "speaker_segments")
    os.makedirs(temp_dir, exist_ok=True)

    # ── Load manifest from Step 1 ────────────────────────────────────
    with open(args.manifest, "r") as f:
        manifest = json.load(f)
    print(f"Loaded manifest with {len(manifest)} audio file(s)")

    # ── Load embedding model ─────────────────────────────────────────
    print("Loading speaker embedding model...")
    embedding_backend = WeSpeakerBackend(
        model_dir=args.embedding_model_dir, device=args.embedding_device
    )
    # ── Initialize global speaker pool ───────────────────────────────
    global_pool = GlobalSpeakerPool(
        similarity_threshold=args.similarity_threshold,
    )

    # ── Process each audio file sequentially ─────────────────────────
    all_results: Dict[str, Dict] = {}

    for entry in manifest:
        print("processing entry.....")
        audio_file = entry["audio_file"]
        seglst_path = entry["seglst_path"]
        diar_path = entry["diar_path"]

        # Load Step 1 outputs
        with open(seglst_path, "r") as f:
            seglst_dict_list = json.load(f)
        diar_result = np.load(diar_path)
        print("diar_result = ", diar_result.shape)
        # Segment audio + extract embeddings
        local_speakers = process_single_audio(
            audio_file,
            seglst_dict_list,
            diar_result[0],
            embedding_backend,
            temp_dir,
        )

        # Register into global pool
        local_to_global = global_pool.register_audio_speakers(
            audio_file, local_speakers
        )

        all_results[audio_file] = {
            "local_speakers": {
                k: {"text": v["text"], "segments": v["segments"]}
                for k, v in local_speakers.items()
            },
            "local_to_global_mapping": local_to_global,
        }

    # ── Summary ──────────────────────────────────────────────────────
    global_pool.summary()

    # ── Save JSON results ────────────────────────────────────────────
    output = {"per_audio_results": {}, "global_speakers": {}}

    for audio_file, result in all_results.items():
        output["per_audio_results"][audio_file] = {
            "local_to_global": result["local_to_global_mapping"],
            "local_speakers": {
                k: {
                    "text": v["text"],
                    "segments": [
                        {"start": s, "end": e, "words": w}
                        for s, e, w in v["segments"]
                    ],
                }
                for k, v in result["local_speakers"].items()
            },
        }

    for spk in global_pool.speakers:
        output["global_speakers"][spk.name] = {
            "global_id": spk.global_id,
            "weight": spk.weight,
            "transcriptions": spk.transcriptions,
        }

    output_path = os.path.join(args.output_dir, "global_speaker_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
