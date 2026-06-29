"""
Speaker linking back-ends for cross-file (global) speaker tracking.

This module holds the embedding back-ends and the global-speaker linkers used
by ``step2_speaker_match_v2.py``.  Three interchangeable linkers are provided,
all sharing the same :class:`BaseGlobalSpeakerLinker` interface so they can be
swapped from a single ``--linker`` flag:

  * :class:`GlobalSpeakerPool`    - original online greedy nearest-centroid
                                    linker (cosine + running-mean update).
  * :class:`ASNormSpeakerPool`    - online greedy linker with adaptive score
                                    normalisation (AS-norm) and robust
                                    duration-weighted / medoid centroids.
  * :class:`TwoPassSpeakerCluster`- batch linker: collect every local-speaker
                                    embedding first, then cluster them in one
                                    shot (agglomerative or spectral).

Common interface
----------------
    linker.add_audio_speakers(audio_key, local_speakers)   # one call per file
    mappings = linker.finalize()                           # {audio_key: {local_id: global_name}}

``local_speakers`` is the dict produced by ``process_single_audio`` in step2::

    {
        "speaker_0": {
            "embedding": np.ndarray,   # 1-D speaker embedding
            "text": str,
            "segments": [(start, end, word), ...],
            "duration": float,         # total active seconds (optional, used as weight)
        },
        ...
    }
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


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
        return np.asarray(embedding).flatten()


# ─── Helpers ─────────────────────────────────────────────────────────

def _as_vec(embedding) -> np.ndarray:
    """Return a contiguous 1-D float64 copy of an embedding (numpy or torch)."""
    arr = np.asarray(embedding, dtype=np.float64)
    return arr.flatten()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation of a 2-D matrix."""
    mat = np.asarray(mat, dtype=np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + 1e-8)


# ─── Global speaker data ─────────────────────────────────────────────

@dataclass
class GlobalSpeaker:
    """A speaker in the global pool, aggregated across multiple audio files."""

    global_id: int
    name: str
    embedding: np.ndarray
    weight: float = 1.0
    transcriptions: List[Dict] = field(default_factory=list)
    # Raw member embeddings + per-member weights, kept for robust centroids
    # (medoid / trimmed mean) and for AS-norm impostor cohorts.
    members: List[np.ndarray] = field(default_factory=list)
    member_weights: List[float] = field(default_factory=list)


# ─── Common interface ────────────────────────────────────────────────

class BaseGlobalSpeakerLinker(ABC):
    """
    Common interface shared by every linker.

    Online linkers compute the mapping incrementally inside
    ``add_audio_speakers`` and simply return the accumulated dict from
    ``finalize``.  Batch linkers buffer everything in ``add_audio_speakers``
    and do the real work in ``finalize``.
    """

    def __init__(self):
        self.speakers: List[GlobalSpeaker] = []
        self._next_id = 0
        self._mappings: Dict[str, Dict[str, str]] = {}

    @abstractmethod
    def add_audio_speakers(
        self, audio_key: str, local_speakers: Dict[str, Dict]
    ) -> Dict[str, str]:
        """Register all local speakers from one audio file. Returns the
        local_id -> global_name mapping known *so far* (may be provisional for
        batch linkers; the authoritative mapping comes from ``finalize``)."""
        ...

    def finalize(self) -> Dict[str, Dict[str, str]]:
        """Return the authoritative {audio_key: {local_id: global_name}} map."""
        return self._mappings

    # -- persistence (single-file pool state) -----------------------------
    def save(self, path: str) -> None:
        """Serialize the global speaker pool to a single ``.npz`` file.

        Stores only what greedy/online linking needs to continue across runs:
        per-speaker centroid + weight + id + name, plus ``next_id``.  The
        running per-chunk transcripts are intentionally *not* stored here -
        they already live in the per-chunk ``parsed_dialog_pred.json`` files.
        """
        import os as _os

        out_dir = _os.path.dirname(_os.path.abspath(path))
        _os.makedirs(out_dir, exist_ok=True)

        if self.speakers:
            embeddings = np.stack(
                [_as_vec(s.embedding) for s in self.speakers], axis=0
            ).astype(np.float32)
            weights = np.asarray([s.weight for s in self.speakers], dtype=np.float32)
            global_ids = np.asarray([s.global_id for s in self.speakers], dtype=np.int64)
            names = np.asarray([s.name for s in self.speakers], dtype=object)
        else:
            embeddings = np.zeros((0, 0), dtype=np.float32)
            weights = np.zeros((0,), dtype=np.float32)
            global_ids = np.zeros((0,), dtype=np.int64)
            names = np.asarray([], dtype=object)

        np.savez(
            path,
            embeddings=embeddings,
            weights=weights,
            global_ids=global_ids,
            names=names,
            next_id=np.int64(self._next_id),
        )
        print(f"[pool] saved {len(self.speakers)} global speaker(s) -> {path}")

    def load(self, path: str) -> None:
        """Restore a pool previously written by :meth:`save` (in place)."""
        data = np.load(path, allow_pickle=True)
        embeddings = data["embeddings"]
        weights = data["weights"]
        global_ids = data["global_ids"]
        names = data["names"]

        self.speakers = []
        for i in range(len(global_ids)):
            emb = _as_vec(embeddings[i])
            w = float(weights[i])
            self.speakers.append(
                GlobalSpeaker(
                    global_id=int(global_ids[i]),
                    name=str(names[i]),
                    embedding=emb.copy(),
                    weight=w,
                    transcriptions=[],
                    members=[emb.copy()],
                    member_weights=[w],
                )
            )
        self._next_id = int(data["next_id"])
        print(f"[pool] loaded {len(self.speakers)} global speaker(s) <- {path}")

    # -- shared utilities -------------------------------------------------
    @staticmethod
    def _transcription(audio_key: str, local_id: str, info: Dict) -> Dict:
        return {
            "audio_file": audio_key,
            "local_speaker_id": local_id,
            "text": info.get("text", ""),
            "segments": [
                {"start": s, "end": e, "words": w}
                for s, e, w in info.get("segments", [])
            ],
        }

    def summary(self):
        print(f"\n{'=' * 70}")
        print(f"Global Speaker Pool: {len(self.speakers)} unique speaker(s)")
        print(f"{'=' * 70}")
        for spk in self.speakers:
            print(f"\n  {spk.name}  (weight={spk.weight:.1f})")
            for t in spk.transcriptions:
                print(f"    [{t['audio_file']}] local_id={t['local_speaker_id']}")
                text_preview = t["text"][:120]
                if len(t["text"]) > 120:
                    text_preview += "..."
                print(f"      Text: {text_preview}")


# ─── 1. Original online greedy pool ──────────────────────────────────

class GlobalSpeakerPool(BaseGlobalSpeakerLinker):
    """
    Online greedy nearest-centroid linker.  Local speakers from each audio
    file are matched against the pool one-by-one using cosine similarity and a
    fixed threshold; matched speakers update the centroid with a weighted
    running average.
    """

    def __init__(self, similarity_threshold: float = 0.65):
        super().__init__()
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return cosine_similarity(a, b)

    def _create_speaker(
        self, embedding: np.ndarray, transcription: Dict
    ) -> GlobalSpeaker:
        emb = _as_vec(embedding)
        spk = GlobalSpeaker(
            global_id=self._next_id,
            name=f"GLOBAL_SPK_{self._next_id}",
            embedding=emb.copy(),
            weight=1.0,
            transcriptions=[transcription],
            members=[emb.copy()],
            member_weights=[1.0],
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
            sim = cosine_similarity(embedding, spk.embedding)
            if sim > best_sim:
                best_spk, best_sim = spk, sim
        return best_spk, best_sim

    def register_speaker(
        self, embedding: np.ndarray, transcription: Dict
    ) -> GlobalSpeaker:
        embedding = _as_vec(embedding)
        best_spk, best_sim = self._find_closest(embedding)

        if best_spk is not None and best_sim >= self.similarity_threshold:
            old_w = best_spk.weight
            new_w = old_w + 1.0
            best_spk.embedding = (best_spk.embedding * old_w + embedding) / new_w
            best_spk.weight = new_w
            best_spk.transcriptions.append(transcription)
            best_spk.members.append(embedding.copy())
            best_spk.member_weights.append(1.0)
            print(
                f"  -> Matched {best_spk.name} (sim={best_sim:.4f}, weight={new_w:.0f})"
            )
            return best_spk

        new_spk = self._create_speaker(embedding, transcription)
        print(f"  -> New {new_spk.name} (best_sim={best_sim:.4f})")
        return new_spk

    def add_audio_speakers(
        self, audio_key: str, local_speakers: Dict[str, Dict]
    ) -> Dict[str, str]:
        print(f"\nRegistering speakers from: {audio_key}")
        mapping: Dict[str, str] = {}
        for local_id, info in local_speakers.items():
            print(f"  Local speaker '{local_id}':")
            transcription = self._transcription(audio_key, local_id, info)
            global_spk = self.register_speaker(info["embedding"], transcription)
            mapping[local_id] = global_spk.name
        self._mappings[audio_key] = mapping
        return mapping

    # Backwards-compatible alias for the old call site.
    def register_audio_speakers(
        self, audio_file: str, local_speakers: Dict
    ) -> Dict[str, str]:
        return self.add_audio_speakers(audio_file, local_speakers)


# ─── 2. Online AS-norm pool with robust centroids ────────────────────

class ASNormSpeakerPool(BaseGlobalSpeakerLinker):
    """
    Online greedy linker that improves robustness over :class:`GlobalSpeakerPool`
    in two ways:

    1. **Adaptive score normalisation (AS-norm).**  Rather than comparing the
       raw cosine score against a fixed threshold, each score is normalised
       against an impostor *cohort* (the embeddings of all other global
       speakers seen so far).  This makes the decision threshold far less
       sensitive to gender / language / channel conditions.  When the cohort
       is too small to estimate statistics reliably, it falls back to raw
       cosine against ``raw_threshold``.

    2. **Robust centroids.**  Instead of an unweighted running mean, each
       global speaker keeps its member embeddings and recomputes its centroid
       as a duration-weighted mean, a trimmed mean, or a medoid - reducing the
       impact of a single mis-linked or noisy segment.

    Parameters
    ----------
    norm_threshold : float
        Decision threshold on the *normalised* score (roughly a z-score).
        Typical useful range is ~0.0-1.5; tune on a dev set.
    raw_threshold : float
        Fallback cosine threshold used while the cohort is smaller than
        ``min_cohort``.
    cohort_size : int
        Top-K cohort scores used to estimate mean/std for AS-norm.
    min_cohort : int
        Minimum number of cohort embeddings required before AS-norm kicks in.
    centroid_mode : {"weighted_mean", "trimmed_mean", "medoid"}
        How a global speaker's representative embedding is computed.
    trim_ratio : float
        Fraction trimmed from each tail when ``centroid_mode="trimmed_mean"``.
    """

    def __init__(
        self,
        norm_threshold: float = 0.5,
        raw_threshold: float = 0.65,
        cohort_size: int = 10,
        min_cohort: int = 5,
        centroid_mode: str = "weighted_mean",
        trim_ratio: float = 0.1,
    ):
        super().__init__()
        self.norm_threshold = norm_threshold
        self.raw_threshold = raw_threshold
        self.cohort_size = cohort_size
        self.min_cohort = min_cohort
        if centroid_mode not in ("weighted_mean", "trimmed_mean", "medoid"):
            raise ValueError(f"Unknown centroid_mode: {centroid_mode}")
        self.centroid_mode = centroid_mode
        self.trim_ratio = trim_ratio

    # -- cohort & AS-norm -------------------------------------------------
    def _cohort_embeddings(self, exclude: Optional[GlobalSpeaker]) -> np.ndarray:
        """All member embeddings except those of *exclude* (leave-one-out)."""
        rows = []
        for spk in self.speakers:
            if spk is exclude:
                continue
            rows.extend(spk.members)
        if not rows:
            return np.empty((0,))
        return _l2_normalize(np.stack(rows, axis=0))

    def _cohort_stats(self, vec: np.ndarray, cohort: np.ndarray) -> Tuple[float, float]:
        """Mean/std of the top-K cohort cosine scores for *vec*."""
        v = vec / (np.linalg.norm(vec) + 1e-8)
        scores = cohort @ v
        if scores.size > self.cohort_size:
            scores = np.sort(scores)[-self.cohort_size:]
        mu = float(np.mean(scores))
        sigma = float(np.std(scores) + 1e-6)
        return mu, sigma

    def _asnorm_score(
        self, query: np.ndarray, candidate: GlobalSpeaker
    ) -> Tuple[float, bool]:
        """
        Return (score, used_asnorm).  ``used_asnorm=False`` means the cohort
        was too small and the score is a raw cosine similarity.
        """
        raw = cosine_similarity(query, candidate.embedding)
        cohort = self._cohort_embeddings(exclude=candidate)
        if cohort.shape[0] < self.min_cohort:
            return raw, False
        mu_e, sig_e = self._cohort_stats(query, cohort)
        mu_c, sig_c = self._cohort_stats(candidate.embedding, cohort)
        norm = 0.5 * ((raw - mu_e) / sig_e + (raw - mu_c) / sig_c)
        return norm, True

    # -- robust centroid --------------------------------------------------
    def _recompute_centroid(self, spk: GlobalSpeaker) -> None:
        members = np.stack(spk.members, axis=0)
        weights = np.asarray(spk.member_weights, dtype=np.float64)

        if self.centroid_mode == "medoid" and len(members) >= 3:
            normed = _l2_normalize(members)
            sims = normed @ normed.T
            spk.embedding = members[int(np.argmax(sims.sum(axis=1)))].copy()
        elif self.centroid_mode == "trimmed_mean" and len(members) >= 5:
            # Trim members farthest from the weighted mean, then re-average.
            mean = np.average(members, axis=0, weights=weights)
            d = np.linalg.norm(members - mean, axis=1)
            k = max(1, int(len(members) * (1.0 - self.trim_ratio)))
            keep = np.argsort(d)[:k]
            spk.embedding = np.average(
                members[keep], axis=0, weights=weights[keep]
            )
        else:  # weighted_mean (default / fallback for tiny clusters)
            spk.embedding = np.average(members, axis=0, weights=weights)
        spk.weight = float(weights.sum())

    def _create_speaker(
        self, embedding: np.ndarray, weight: float, transcription: Dict
    ) -> GlobalSpeaker:
        emb = _as_vec(embedding)
        spk = GlobalSpeaker(
            global_id=self._next_id,
            name=f"GLOBAL_SPK_{self._next_id}",
            embedding=emb.copy(),
            weight=weight,
            transcriptions=[transcription],
            members=[emb.copy()],
            member_weights=[weight],
        )
        self.speakers.append(spk)
        self._next_id += 1
        return spk

    def register_speaker(
        self, embedding: np.ndarray, weight: float, transcription: Dict
    ) -> GlobalSpeaker:
        embedding = _as_vec(embedding)

        best_spk, best_score, best_used = None, -1e9, False
        for spk in self.speakers:
            score, used = self._asnorm_score(embedding, spk)
            if score > best_score:
                best_spk, best_score, best_used = spk, score, used

        threshold = self.norm_threshold if best_used else self.raw_threshold
        if best_spk is not None and best_score >= threshold:
            best_spk.members.append(embedding.copy())
            best_spk.member_weights.append(weight)
            best_spk.transcriptions.append(transcription)
            self._recompute_centroid(best_spk)
            mode = "asnorm" if best_used else "cosine"
            print(
                f"  -> Matched {best_spk.name} "
                f"({mode}={best_score:.4f}>= {threshold:.3f}, "
                f"weight={best_spk.weight:.1f})"
            )
            return best_spk

        new_spk = self._create_speaker(embedding, weight, transcription)
        mode = "asnorm" if best_used else "cosine"
        print(f"  -> New {new_spk.name} (best {mode}={best_score:.4f})")
        return new_spk

    def add_audio_speakers(
        self, audio_key: str, local_speakers: Dict[str, Dict]
    ) -> Dict[str, str]:
        print(f"\n[AS-norm] Registering speakers from: {audio_key}")
        mapping: Dict[str, str] = {}
        for local_id, info in local_speakers.items():
            weight = float(info.get("duration", 1.0)) or 1.0
            print(f"  Local speaker '{local_id}' (weight={weight:.1f}):")
            transcription = self._transcription(audio_key, local_id, info)
            global_spk = self.register_speaker(
                info["embedding"], weight, transcription
            )
            mapping[local_id] = global_spk.name
        self._mappings[audio_key] = mapping
        return mapping


# ─── 3. Two-pass batch clustering ────────────────────────────────────

class TwoPassSpeakerCluster(BaseGlobalSpeakerLinker):
    """
    Batch (two-pass) linker.  Pass 1 buffers every local-speaker embedding;
    ``finalize`` (pass 2) builds a full affinity matrix and clusters all
    embeddings at once.  Because the global decision sees every embedding
    together, it is free of the order-dependence and centroid-poisoning that
    affect greedy online linking.

    Parameters
    ----------
    method : {"ahc", "spectral"}
        Clustering algorithm.  "ahc" (agglomerative, average linkage) uses a
        cosine *distance* threshold and needs no known speaker count.
        "spectral" estimates the number of speakers via the eigengap heuristic.
    distance_threshold : float
        For AHC: merge clusters whose cosine distance (= 1 - cosine sim) is
        below this.  ``distance_threshold = 1 - similarity_threshold``.
    use_asnorm : bool
        If True, AS-norm the cosine affinity matrix before clustering, making
        the affinities more comparable across conditions.
    cohort_size : int
        Top-K cohort size for AS-norm of the affinity matrix.
    max_speakers : int
        Upper bound on the eigengap search for ``method="spectral"``.
    """

    def __init__(
        self,
        method: str = "ahc",
        distance_threshold: float = 0.35,
        use_asnorm: bool = False,
        cohort_size: int = 10,
        max_speakers: int = 50,
    ):
        super().__init__()
        if method not in ("ahc", "spectral"):
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.distance_threshold = distance_threshold
        self.use_asnorm = use_asnorm
        self.cohort_size = cohort_size
        self.max_speakers = max_speakers
        # Buffered per-local-speaker records.
        self._buffer: List[Dict] = []
        # Populated by finalize(); kept for debugging / visualisation.
        self._embeddings: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None

    def add_audio_speakers(
        self, audio_key: str, local_speakers: Dict[str, Dict]
    ) -> Dict[str, str]:
        for local_id, info in local_speakers.items():
            self._buffer.append(
                {
                    "audio_key": audio_key,
                    "local_id": local_id,
                    "embedding": _as_vec(info["embedding"]),
                    "weight": float(info.get("duration", 1.0)) or 1.0,
                    "transcription": self._transcription(audio_key, local_id, info),
                }
            )
        # Provisional (unclustered) mapping; authoritative map is from finalize.
        return {lid: "PENDING" for lid in local_speakers}

    # -- affinity & clustering -------------------------------------------
    def _affinity(self, embeddings: np.ndarray) -> np.ndarray:
        normed = _l2_normalize(embeddings)
        sim = normed @ normed.T  # cosine in [-1, 1]
        if self.use_asnorm:
            sim = self._asnorm_matrix(sim)
        return sim

    def _asnorm_matrix(self, sim: np.ndarray) -> np.ndarray:
        """Symmetric AS-norm of a cosine score matrix using each row as its
        own impostor cohort (top-K, leave-self-out)."""
        n = sim.shape[0]
        mu = np.zeros(n)
        sigma = np.zeros(n)
        for i in range(n):
            row = np.delete(sim[i], i)
            if row.size > self.cohort_size:
                row = np.sort(row)[-self.cohort_size:]
            mu[i] = row.mean()
            sigma[i] = row.std() + 1e-6
        norm = 0.5 * (
            (sim - mu[:, None]) / sigma[:, None]
            + (sim - mu[None, :]) / sigma[None, :]
        )
        return norm

    def _estimate_num_clusters(self, affinity: np.ndarray) -> int:
        """Eigengap heuristic on the normalised Laplacian."""
        a = np.clip(affinity, 0.0, None)
        a = 0.5 * (a + a.T)
        np.fill_diagonal(a, 0.0)
        deg = a.sum(axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(deg + 1e-8)
        lap = np.eye(a.shape[0]) - (a * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
        eigvals = np.sort(np.linalg.eigvalsh(lap))
        upper = min(self.max_speakers, len(eigvals) - 1)
        if upper < 1:
            return 1
        gaps = np.diff(eigvals[: upper + 1])
        return int(np.argmax(gaps) + 1)

    def _cluster(self, embeddings: np.ndarray) -> np.ndarray:
        n = embeddings.shape[0]
        if n == 1:
            return np.zeros(1, dtype=int)

        if self.method == "spectral":
            affinity = self._affinity(embeddings)
            affinity = np.clip(affinity, 0.0, None)
            affinity = 0.5 * (affinity + affinity.T)
            k = max(1, min(self._estimate_num_clusters(affinity), n))
            if k == 1:
                return np.zeros(n, dtype=int)
            return self._spectral(affinity, k)

        # Agglomerative (average-linkage) on cosine distance.
        sim = self._affinity(embeddings)
        if self.use_asnorm:
            # Normalised scores aren't bounded; convert to a distance by
            # shifting into [0, +inf).
            dist = sim.max() - sim
        else:
            dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)
        dist = np.clip(0.5 * (dist + dist.T), 0.0, None)
        return self._agglomerative(dist, self.distance_threshold)

    # -- clustering primitives (pure numpy; sklearn used only if present) --
    @staticmethod
    def _agglomerative(dist: np.ndarray, threshold: float) -> np.ndarray:
        """Average-linkage agglomerative clustering with a distance cutoff.

        Uses sklearn when available, otherwise a pure-numpy fallback.
        """
        try:
            from sklearn.cluster import AgglomerativeClustering

            model = AgglomerativeClustering(
                n_clusters=None,
                metric="precomputed",
                linkage="average",
                distance_threshold=threshold,
            )
            return model.fit_predict(dist)
        except Exception:
            pass

        n = dist.shape[0]
        clusters: List[List[int]] = [[i] for i in range(n)]
        while len(clusters) > 1:
            best_d, best_pair = np.inf, None
            for a in range(len(clusters)):
                for b in range(a + 1, len(clusters)):
                    d = float(
                        np.mean(dist[np.ix_(clusters[a], clusters[b])])
                    )  # average linkage
                    if d < best_d:
                        best_d, best_pair = d, (a, b)
            if best_pair is None or best_d > threshold:
                break
            a, b = best_pair
            clusters[a].extend(clusters[b])
            clusters.pop(b)

        labels = np.zeros(n, dtype=int)
        for lab, members in enumerate(clusters):
            for idx in members:
                labels[idx] = lab
        return labels

    def _spectral(self, affinity: np.ndarray, k: int) -> np.ndarray:
        """Normalised-cut spectral clustering.

        Uses sklearn when available, otherwise a numpy Laplacian-eigenmap +
        k-means fallback.
        """
        try:
            from sklearn.cluster import SpectralClustering

            model = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=0,
            )
            return model.fit_predict(affinity)
        except Exception:
            pass

        a = 0.5 * (affinity + affinity.T)
        np.fill_diagonal(a, 0.0)
        deg = a.sum(axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(deg + 1e-8)
        lap = np.eye(a.shape[0]) - (a * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
        eigvals, eigvecs = np.linalg.eigh(lap)
        feats = eigvecs[:, :k]
        feats = _l2_normalize(feats)
        return self._kmeans(feats, k)

    @staticmethod
    def _kmeans(x: np.ndarray, k: int, iters: int = 50) -> np.ndarray:
        """Deterministic k-means (k-means++ style seeding, fixed order)."""
        n = x.shape[0]
        # Deterministic farthest-point seeding starting from index 0.
        centers = [0]
        d2 = np.sum((x - x[0]) ** 2, axis=1)
        for _ in range(1, k):
            nxt = int(np.argmax(d2))
            centers.append(nxt)
            d2 = np.minimum(d2, np.sum((x - x[nxt]) ** 2, axis=1))
        cen = x[centers].copy()

        labels = np.zeros(n, dtype=int)
        for _ in range(iters):
            dists = np.linalg.norm(x[:, None, :] - cen[None, :, :], axis=2)
            new_labels = dists.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                pts = x[labels == j]
                if len(pts):
                    cen[j] = pts.mean(axis=0)
        return labels

    def finalize(self) -> Dict[str, Dict[str, str]]:
        if not self._buffer:
            return {}

        embeddings = np.stack([r["embedding"] for r in self._buffer], axis=0)
        print("embeddings shape", embeddings.shape)
        labels = self._cluster(embeddings)

        # Keep around for debugging / visualisation.
        self._embeddings = embeddings
        self._labels = labels

        # Build GlobalSpeaker objects (centroid = weighted mean of members).
        clusters: Dict[int, List[int]] = defaultdict(list)
        for idx, lab in enumerate(labels):
            clusters[int(lab)].append(idx)

        label_to_name: Dict[int, str] = {}
        for lab in sorted(clusters):
            idxs = clusters[lab]
            members = [self._buffer[i]["embedding"] for i in idxs]
            weights = np.asarray(
                [self._buffer[i]["weight"] for i in idxs], dtype=np.float64
            )
            centroid = np.average(np.stack(members, axis=0), axis=0, weights=weights)
            name = f"GLOBAL_SPK_{self._next_id}"
            label_to_name[lab] = name
            self.speakers.append(
                GlobalSpeaker(
                    global_id=self._next_id,
                    name=name,
                    embedding=centroid,
                    weight=float(weights.sum()),
                    transcriptions=[self._buffer[i]["transcription"] for i in idxs],
                    members=[m.copy() for m in members],
                    member_weights=weights.tolist(),
                )
            )
            self._next_id += 1

        # Assemble {audio_key: {local_id: global_name}}.
        mappings: Dict[str, Dict[str, str]] = defaultdict(dict)
        for idx, rec in enumerate(self._buffer):
            mappings[rec["audio_key"]][rec["local_id"]] = label_to_name[int(labels[idx])]
        self._mappings = dict(mappings)

        print(
            f"\n[TwoPass:{self.method}] Clustered {len(self._buffer)} local "
            f"speaker(s) into {len(self.speakers)} global speaker(s)."
        )
        return self._mappings

    def visualize(self, debug_dir: str, prefix: str = "twopass") -> List[str]:
        """
        Render the clustering for inspection and save figures to *debug_dir*.

        Must be called after :meth:`finalize`.  Produces:
          * ``{prefix}_scatter.png``  - 2-D PCA projection of all local-speaker
            embeddings, coloured by predicted global cluster.
          * ``{prefix}_affinity.png`` - cosine-affinity matrix reordered by
            cluster, with cluster block boundaries drawn.

        Returns the list of written file paths (empty if matplotlib is missing
        or there is nothing to plot).
        """
        if self._embeddings is None or self._labels is None:
            print("[TwoPass.visualize] Nothing to plot - call finalize() first.")
            return []

        names = [
            f"{r['audio_key']}#{r['local_id']}" for r in self._buffer
        ]
        cluster_names = [
            self._mappings[r["audio_key"]][r["local_id"]] for r in self._buffer
        ]
        affinity = self._affinity(self._embeddings)
        return plot_clustering(
            embeddings=self._embeddings,
            labels=np.asarray(self._labels),
            affinity=affinity,
            item_names=names,
            cluster_names=cluster_names,
            debug_dir=debug_dir,
            prefix=prefix,
            title=f"TwoPass ({self.method})",
        )


# ─── Visualisation ───────────────────────────────────────────────────

def _pca_2d(embeddings: np.ndarray) -> np.ndarray:
    """Project rows to 2-D via PCA (numpy SVD, no sklearn dependency)."""
    x = np.asarray(embeddings, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    # Economy SVD: principal axes are the rows of Vt.
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    comps = vt[:2] if vt.shape[0] >= 2 else np.vstack([vt, np.zeros_like(vt[:1])])
    return x @ comps.T


def plot_clustering(
    embeddings: np.ndarray,
    labels: np.ndarray,
    affinity: np.ndarray,
    item_names: List[str],
    cluster_names: List[str],
    debug_dir: str,
    prefix: str = "cluster",
    title: str = "Speaker clustering",
) -> List[str]:
    """
    Save a 2-D scatter and an affinity heatmap visualising a clustering.

    Imports matplotlib lazily with the non-interactive 'Agg' backend so it is
    safe on headless machines and adds no hard dependency.
    """
    import os

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on runtime env
        print(f"[plot_clustering] matplotlib unavailable ({exc}); skipping plots.")
        return []

    os.makedirs(debug_dir, exist_ok=True)
    labels = np.asarray(labels)
    n = embeddings.shape[0]
    uniq = sorted(set(labels.tolist()))
    # Map each label to a colour (version-robust across matplotlib releases).
    n_colors = max(len(uniq), 1)
    try:
        cmap = plt.colormaps["tab20"].resampled(n_colors)
    except Exception:
        from matplotlib import cm
        cmap = cm.get_cmap("tab20", n_colors)
    lab_to_color = {lab: cmap(i) for i, lab in enumerate(uniq)}
    written: List[str] = []

    # ── 1. 2-D PCA scatter ───────────────────────────────────────────
    proj = _pca_2d(embeddings)
    fig, ax = plt.subplots(figsize=(9, 7))
    for lab in uniq:
        m = labels == lab
        # Use the human-readable global name for the legend if available.
        gname = next(
            (cluster_names[i] for i in range(n) if labels[i] == lab), str(lab)
        )
        ax.scatter(
            proj[m, 0], proj[m, 1],
            s=80, alpha=0.8, color=lab_to_color[lab],
            edgecolors="k", linewidths=0.5,
            label=f"{gname} (n={int(m.sum())})",
        )
    # Annotate individual points when the plot is not too crowded.
    if n <= 40:
        for i in range(n):
            ax.annotate(
                item_names[i], (proj[i, 0], proj[i, 1]),
                fontsize=6, alpha=0.7,
                xytext=(3, 3), textcoords="offset points",
            )
    ax.set_title(f"{title} - {len(uniq)} cluster(s), {n} local speakers (PCA 2-D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    # ax.legend(fontsize=7, loc="best", framealpha=0.8)
    fig.tight_layout()
    scatter_path = os.path.join(debug_dir, f"{prefix}_scatter.png")
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    written.append(scatter_path)

    # ── 2. Affinity heatmap reordered by cluster ─────────────────────
    order = np.argsort(labels, kind="stable")
    aff = np.asarray(affinity)[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(aff, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="affinity")
    # Draw block boundaries between clusters.
    sorted_labels = labels[order]
    boundaries = np.where(np.diff(sorted_labels) != 0)[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color="red", linewidth=0.8)
        ax.axvline(b, color="red", linewidth=0.8)
    ax.set_title(f"{title} - affinity matrix (reordered by cluster)")
    ax.set_xlabel("local speaker (sorted by cluster)")
    ax.set_ylabel("local speaker (sorted by cluster)")
    if n <= 40:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([item_names[i] for i in order], rotation=90, fontsize=5)
        ax.set_yticklabels([item_names[i] for i in order], fontsize=5)
    fig.tight_layout()
    heatmap_path = os.path.join(debug_dir, f"{prefix}_affinity.png")
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    written.append(heatmap_path)

    print(f"[plot_clustering] wrote {len(written)} figure(s) to {debug_dir}")
    return written


# ─── Factory ─────────────────────────────────────────────────────────

def build_linker(name: str, similarity_threshold: float = 0.7, **kwargs):
    """
    Construct a linker by name.  ``similarity_threshold`` is the shared cosine
    threshold; for the two-pass AHC linker it maps to a distance threshold
    ``1 - similarity_threshold``.
    """
    name = name.lower()
    if name in ("greedy", "pool", "global"):
        return GlobalSpeakerPool(similarity_threshold=similarity_threshold)
    if name == "asnorm":
        return ASNormSpeakerPool(raw_threshold=similarity_threshold, **kwargs)
    if name in ("twopass", "cluster"):
        kwargs.setdefault("distance_threshold", 1.0 - similarity_threshold)
        return TwoPassSpeakerCluster(**kwargs)
    raise ValueError(
        f"Unknown linker '{name}'. Choose from: greedy, asnorm, twopass."
    )
