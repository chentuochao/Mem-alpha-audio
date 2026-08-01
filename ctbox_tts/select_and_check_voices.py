#!/usr/bin/env python3
"""
Embedding-aware reference-voice selection + post-TTS separation check.

This complements ``perltqa_dialogue_tts.py``, which by default assigns the
*first* gender-matching LibriTTS clip to each speaker with no regard for how
close two speakers end up in embedding space.  Here we make the assignment
speaker-embedding aware and add a sanity check on the synthesized audio.

Two independent stages (sub-commands):

    select   Pre-extract WeSpeaker embeddings for the whole LibriTTS voice bank
             once (cached to disk), then pick one distinct voice per PerLTQA
             speaker via *greedy farthest-point* selection inside each gender
             pool: a candidate is only accepted if its max cosine similarity to
             every already-accepted (same-gender) pick stays below ``--t-ref``.
             Writes a ``reference_voice_map.json`` that is a drop-in replacement
             for the one ``perltqa_dialogue_tts.py`` consumes.

    check    After TTS, walk the generated dialogue folders, embed each
             speaker's *synthesized* speech (sliced from its isolated track via
             ``channel_map.json``) and report pairwise cosine similarity, both
             within each dialogue block and globally across speaker names.
             Flags any different-speaker pair that lands above ``--flag-sim``
             (the space that actually matters downstream, where the global
             speaker pool merges at 0.65).

Note (by design): ``select`` only guarantees separation of the *reference*
audio.  Chatterbox is a lossy transform and can compress voices closer
together, so ``check`` is what verifies the property in TTS-output space.

Examples
--------
    # 1) build the embedding cache + select references (no TTS)
    python select_and_check_voices.py select

    # rebuild the cache from scratch (e.g. after changing the voice bank)
    python select_and_check_voices.py select --rebuild-cache

    # 2) after running perltqa_dialogue_tts.py, audit the synthesized voices
    python select_and_check_voices.py check
"""

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Sibling modules (perltqa_dialogue_tts / chatterbox_dialogue_tts) are imported
# without a package prefix, matching how the existing scripts are run.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# ----------------------------------------------------------------------------
# Defaults (mirror the hard-coded paths in perltqa_dialogue_tts.py)
# ----------------------------------------------------------------------------
DEFAULT_DATA = Path(
    "/checkpoint/seamless/tuochao/data/PerLTQA/Dataset/en_v2/perltmem_en_v2.json"
)
DEFAULT_OUTPUT_DIR = Path("/checkpoint/seamless/tuochao/data/PerLTQA/audio")
DEFAULT_REF_DIR = HERE / "ref_voices" / "perltqa"
DEFAULT_LOCAL_DATASET_DIR = Path(
    "/checkpoint/seamless/tuochao/data/ctbox_tts/voices-libritts/"
)
DEFAULT_HF_DATASET = "sdialog/voices-libritts"
DEFAULT_EMBED_MODEL_DIR = (
    "/checkpoint/seamless/tuochao/Models/huggingface/wespeaker-voxceleb-resnet293-LM"
)

DEFAULT_TARGET_SR = 24000          # sample rate of the saved reference wavs
DEFAULT_REF_DURATION_SEC = 10.0    # length of the saved reference wavs
EMBED_SR = 16000                   # WeSpeaker resnet models expect 16 kHz
EMBED_MAX_SEC = 20.0               # cap per-clip audio fed to the embedder

DEFAULT_T_REF = 0.50               # accept a voice only if max sim < this
DEFAULT_FLAG_SIM = 0.50            # flag synthesized different-speaker pairs >= this
DOWNSTREAM_MERGE_SIM = 0.65        # GlobalSpeakerPool.similarity_threshold


# ----------------------------------------------------------------------------
# WeSpeaker backend (minimal, self-contained)
# ----------------------------------------------------------------------------
class SpeakerEmbedder:
    """Thin wrapper around a WeSpeaker model (same API as step2's backend)."""

    def __init__(self, model_dir: str, device: int = 0):
        import torch
        import wespeaker

        self.model = wespeaker.load_model(model_dir)

        # Resolve a concrete device string; fall back to CPU if CUDA is absent.
        if torch.cuda.is_available() and device is not None and device >= 0:
            device_str = f"cuda:{device}"
        else:
            device_str = "cpu"
        self.model.set_device(device_str)

        # Some WeSpeaker builds (e.g. the resnet293 one here) compute fbank
        # features on CPU but never move them to the model's device before the
        # forward pass, which crashes on GPU with a
        # "Input type (torch.FloatTensor) and weight type (torch.cuda.*)"
        # mismatch. Patch the underlying nn.Module so its forward always
        # relocates the input to the weight device. No-op on CPU.
        net = getattr(self.model, "model", None)
        if net is not None:
            try:
                dev = next(net.parameters()).device
            except StopIteration:
                dev = torch.device(device_str)
            _orig_forward = net.forward

            def _relocating_forward(feats, *a, **k):
                if isinstance(feats, torch.Tensor) and feats.device != dev:
                    feats = feats.to(dev)
                return _orig_forward(feats, *a, **k)

            net.forward = _relocating_forward

    def extract(self, wav_path: str) -> np.ndarray:
        emb = self.model.extract_embedding(wav_path)
        if isinstance(emb, list):
            emb = np.array(emb)
        return np.asarray(emb, dtype=np.float32).flatten()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + 1e-8)


# ----------------------------------------------------------------------------
# Audio helpers
# ----------------------------------------------------------------------------
def _decode_audio_bytes(raw: bytes) -> Tuple[np.ndarray, int]:
    """Decode raw audio bytes -> (mono float32 waveform, sample_rate)."""
    import io
    import soundfile as sf

    with io.BytesIO(raw) as fh:
        wav, sr = sf.read(fh, dtype="float32")
    if getattr(wav, "ndim", 1) == 2:
        wav = wav.mean(axis=1)
    return np.asarray(wav, dtype=np.float32), int(sr)


def _resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav
    import librosa

    return librosa.resample(wav, orig_sr=sr, target_sr=target_sr)


def _write_tmp_wav(wav: np.ndarray, sr: int, tmp_dir: str, tag: str) -> str:
    import soundfile as sf

    path = os.path.join(tmp_dir, f"{tag}.wav")
    sf.write(path, wav, sr)
    return path


def _prep_for_embedding(wav: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    """Mono, 16 kHz, capped to EMBED_MAX_SEC — what we feed the embedder."""
    wav = _resample(wav, sr, EMBED_SR)
    max_len = int(EMBED_SR * EMBED_MAX_SEC)
    if len(wav) > max_len:
        wav = wav[:max_len]
    return wav, EMBED_SR


def _prep_reference_wav(
    wav: np.ndarray, sr: int, target_sr: int, ref_duration: float
) -> Tuple[np.ndarray, int]:
    """Resample, trim/pad to a fixed length and peak-normalize (as in
    perltqa_dialogue_tts.prepare_reference_voices)."""
    wav = _resample(wav, sr, target_sr)
    target_len = int(target_sr * ref_duration)
    if len(wav) >= target_len:
        wav = wav[:target_len]
    else:
        padded = np.zeros(target_len, dtype="float32")
        padded[: len(wav)] = wav
        wav = padded
    max_abs = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if max_abs > 1e-8:
        wav = (wav / max_abs * 0.95).astype("float32")
    return wav, target_sr


# ----------------------------------------------------------------------------
# LibriTTS embedding cache
# ----------------------------------------------------------------------------
def _iter_voice_bank(local_dataset_dir: Optional[Path], hf_dataset: str):
    """Yield LibriTTS examples (audio bytes not decoded) from local parquet or
    the HuggingFace hub — same loading path as prepare_reference_voices."""
    from datasets import Audio, load_dataset

    local_dir = Path(local_dataset_dir) if local_dataset_dir else None
    if local_dir and local_dir.exists():
        files = sorted(str(p) for p in local_dir.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"--local-dataset-dir {local_dir} has no .parquet files."
            )
        print(f"[cache] loading {len(files)} local parquet shard(s) from {local_dir}")
        ds = load_dataset("parquet", data_files=files, split="train", streaming=True)
    else:
        print(f"[cache] streaming {hf_dataset} from the HuggingFace hub ...")
        ds = load_dataset(hf_dataset, split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    return ds


def build_or_load_cache(
    cache_path: Path,
    embedder: SpeakerEmbedder,
    local_dataset_dir: Optional[Path],
    hf_dataset: str,
    rebuild: bool = False,
) -> dict:
    """
    Return the LibriTTS embedding cache as a dict of parallel arrays:
        {"identifiers": [...], "genders": [...], "names": [...],
         "subsets": [...], "embeddings": np.ndarray [N, D]}

    Resumable: identifiers already present in an existing cache are skipped.
    """
    existing: Dict[str, dict] = {}
    if cache_path.exists() and not rebuild:
        data = np.load(cache_path, allow_pickle=True)
        ids = list(data["identifiers"])
        embs = data["embeddings"]
        gens = list(data["genders"])
        names = list(data["names"])
        subs = list(data["subsets"])
        for i, ident in enumerate(ids):
            existing[str(ident)] = {
                "embedding": embs[i],
                "gender": str(gens[i]),
                "name": str(names[i]),
                "subset": str(subs[i]),
            }
        print(f"[cache] loaded {len(existing)} cached embeddings from {cache_path}")

    ds = _iter_voice_bank(local_dataset_dir, hf_dataset)
    tmp_dir = tempfile.mkdtemp(prefix="wespk_cache_")
    added = 0
    scanned = 0
    try:
        for ex in ds:
            scanned += 1
            identifier = str(ex.get("identifier"))
            if identifier in existing:
                continue
            audio_obj = ex.get("audio") or {}
            if audio_obj.get("bytes") is None:
                continue
            try:
                wav, sr = _decode_audio_bytes(audio_obj["bytes"])
                wav, sr = _prep_for_embedding(wav, sr)
                wav_path = _write_tmp_wav(wav, sr, tmp_dir, identifier.replace("/", "_"))
                emb = embedder.extract(wav_path)
                os.remove(wav_path)
            except Exception as e:  # keep going on a single bad clip
                print(f"[cache][WARN] {identifier}: {e}", file=sys.stderr)
                continue
            existing[identifier] = {
                "embedding": emb.astype(np.float32),
                "gender": str(ex.get("gender")),
                "name": str(ex.get("name")),
                "subset": str(ex.get("subset")),
            }
            added += 1
            if added % 100 == 0:
                print(f"[cache]   embedded {added} new clips "
                      f"({len(existing)} total, {scanned} scanned)")
                _save_cache(cache_path, existing)  # periodic checkpoint
    finally:
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    _save_cache(cache_path, existing)
    print(f"[cache] done. {len(existing)} embeddings ({added} newly added) -> {cache_path}")
    return _cache_to_arrays(existing)


def _save_cache(cache_path: Path, store: Dict[str, dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    arrs = _cache_to_arrays(store)
    np.savez_compressed(
        cache_path,
        identifiers=np.array(arrs["identifiers"], dtype=object),
        genders=np.array(arrs["genders"], dtype=object),
        names=np.array(arrs["names"], dtype=object),
        subsets=np.array(arrs["subsets"], dtype=object),
        embeddings=arrs["embeddings"],
    )


def _cache_to_arrays(store: Dict[str, dict]) -> dict:
    idents = list(store.keys())
    embs = np.stack([store[i]["embedding"] for i in idents]) if idents else np.zeros((0, 0))
    return {
        "identifiers": idents,
        "genders": [store[i]["gender"] for i in idents],
        "names": [store[i]["name"] for i in idents],
        "subsets": [store[i]["subset"] for i in idents],
        "embeddings": embs.astype(np.float32),
    }


# ----------------------------------------------------------------------------
# Greedy farthest-point selection
# ----------------------------------------------------------------------------
def farthest_point_select(
    emb: np.ndarray, k: int, t_ref: float
) -> Tuple[List[int], List[float]]:
    """
    Pick ``k`` rows of ``emb`` that are maximally spread in cosine space.

    Seeded at row 0 (rows are pre-sorted by identifier for determinism), each
    subsequent pick is the candidate *farthest* from the already-selected set
    (i.e. minimizing max cosine similarity to it).  ``t_ref`` is a soft gate:
    picks whose realized max-similarity exceeds it are counted as violations,
    but still taken (graceful fallback when the pool is too tight).

    Returns (selected_row_indices, realized_max_sim_per_pick).  For pick j,
    realized[j] is its max cosine sim to picks 0..j-1 (seed = -1.0), so
    ``max(realized)`` equals the closest (worst) pair among all selected rows.
    """
    n = emb.shape[0]
    if k <= 0:
        return [], []
    if k > n:
        raise ValueError(f"requested {k} voices but only {n} candidates available")

    en = _normalize_rows(emb)
    sim = en @ en.T  # [n, n] cosine similarity matrix

    selected = [0]
    realized = [-1.0]
    max_sim = sim[0].copy()
    max_sim[0] = np.inf  # exclude selected from future picks

    for _ in range(k - 1):
        nxt = int(np.argmin(max_sim))
        realized.append(float(max_sim[nxt]))
        selected.append(nxt)
        max_sim = np.maximum(max_sim, sim[nxt])
        max_sim[selected] = np.inf

    return selected, realized


def _speaker_genders(speakers: List[str], gender_map: Dict[str, str]) -> Dict[str, str]:
    """Resolve each speaker to 'M'/'F'; unknown genders round-robin (as in
    prepare_reference_voices) so we don't drain one pool."""
    resolved: Dict[str, str] = {}
    unknown_i = 0
    for name in speakers:
        g = gender_map.get(name)
        if g not in ("M", "F"):
            g = "M" if (unknown_i % 2 == 0) else "F"
            unknown_i += 1
        resolved[name] = g
    return resolved


def select_references(
    cache: dict,
    speakers: List[str],
    gender_map: Dict[str, str],
    t_ref: float,
) -> Tuple[Dict[str, str], dict]:
    """
    Assign each speaker a distinct LibriTTS identifier via per-gender FPS.

    Returns (assignment {speaker_name: identifier}, report dict).
    """
    idents = cache["identifiers"]
    genders = cache["genders"]
    embs = cache["embeddings"]

    # Per-gender candidate pools, sorted by identifier for determinism.
    pools: Dict[str, List[int]] = defaultdict(list)
    for i, g in enumerate(genders):
        pools[g].append(i)
    for g in pools:
        pools[g].sort(key=lambda i: idents[i])

    spk_gender = _speaker_genders(speakers, gender_map)
    speakers_by_gender: Dict[str, List[str]] = defaultdict(list)
    for name in sorted(speakers):
        speakers_by_gender[spk_gender[name]].append(name)

    assignment: Dict[str, str] = {}
    report = {"t_ref": t_ref, "per_gender": {}}

    for g, spk_list in speakers_by_gender.items():
        pool_idx = pools.get(g, [])
        k = len(spk_list)
        if k > len(pool_idx):
            raise RuntimeError(
                f"gender {g!r}: need {k} voices but pool only has {len(pool_idx)}"
            )
        sub_emb = embs[pool_idx]
        sel_local, realized = farthest_point_select(sub_emb, k, t_ref)

        worst = max(realized[1:]) if len(realized) > 1 else -1.0
        n_violations = sum(1 for r in realized[1:] if r >= t_ref)

        for name, loc in zip(spk_list, sel_local):
            assignment[name] = idents[pool_idx[loc]]

        report["per_gender"][g] = {
            "num_speakers": k,
            "pool_size": len(pool_idx),
            "worst_pair_cosine": worst,
            "num_violations_over_t_ref": n_violations,
        }
        print(f"[select] gender {g}: selected {k}/{len(pool_idx)} voices | "
              f"worst pair cosine = {worst:.4f} | "
              f"{n_violations} pair(s) >= t_ref={t_ref}")

    return assignment, report


def materialize_references(
    assignment: Dict[str, str],
    ref_dir: Path,
    ref_map_path: Path,
    local_dataset_dir: Optional[Path],
    hf_dataset: str,
    target_sr: int,
    ref_duration: float,
    cache: dict,
) -> dict:
    """
    Write one reference wav per assigned speaker and a reference_voice_map.json
    compatible with perltqa_dialogue_tts.generate_all.
    """
    import soundfile as sf

    ref_dir.mkdir(parents=True, exist_ok=True)
    # identifier -> [speaker names] (normally 1:1, but be defensive)
    want: Dict[str, List[str]] = defaultdict(list)
    for name, ident in assignment.items():
        want[ident].append(name)

    # metadata lookup from the cache
    id2meta = {
        cache["identifiers"][i]: {
            "gender": cache["genders"][i],
            "name": cache["names"][i],
            "subset": cache["subsets"][i],
        }
        for i in range(len(cache["identifiers"]))
    }

    from perltqa_dialogue_tts import safe_filename

    ref_map: Dict[str, str] = {}
    metadata: List[dict] = []
    remaining = set(want.keys())

    ds = _iter_voice_bank(local_dataset_dir, hf_dataset)
    for ex in ds:
        if not remaining:
            break
        identifier = str(ex.get("identifier"))
        if identifier not in remaining:
            continue
        audio_obj = ex.get("audio") or {}
        if audio_obj.get("bytes") is None:
            continue
        wav, sr = _decode_audio_bytes(audio_obj["bytes"])
        wav, sr = _prep_reference_wav(wav, sr, target_sr, ref_duration)

        for name in want[identifier]:
            spk_dir = ref_dir / safe_filename(name)
            spk_dir.mkdir(parents=True, exist_ok=True)
            out_path = spk_dir / "reference.wav"
            sf.write(str(out_path), wav, sr)
            ref_map[name] = str(out_path)
            meta = id2meta.get(identifier, {})
            metadata.append({
                "speaker_name": name,
                "assigned_gender": meta.get("gender"),
                "reference_audio": str(out_path),
                "libritts_identifier": identifier,
                "libritts_speaker_name": meta.get("name"),
                "libritts_subset": meta.get("subset"),
                "saved_duration_s": ref_duration,
                "sample_rate": sr,
            })
        remaining.discard(identifier)

    payload = {"reference_voice_map": ref_map, "selected_metadata": metadata}
    ref_map_path.parent.mkdir(parents=True, exist_ok=True)
    with ref_map_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[select] wrote {len(ref_map)} references -> {ref_map_path}")

    if remaining:
        missing = {i: want[i] for i in remaining}
        print(f"[select][WARN] {len(remaining)} identifier(s) not found in the "
              f"voice bank: {missing}", file=sys.stderr)
    return payload


# ----------------------------------------------------------------------------
# Stage 1: select
# ----------------------------------------------------------------------------
def _load_perltqa_speakers(data_path: Path) -> Tuple[List[str], Dict[str, str]]:
    """Reuse perltqa_dialogue_tts parsing to get canonical speakers + genders."""
    import perltqa_dialogue_tts as P

    with data_path.open(encoding="utf-8") as f:
        data = json.load(f)

    P._CANON, num_merged = P.build_canonical_map(data)
    _, speaker_counter, _, _ = P.parse_dataset(data)
    gender_map = P.build_gender_map(data)
    speakers = sorted(speaker_counter.keys())
    print(f"[select] {len(speakers)} unique PerLTQA speakers "
          f"({num_merged} case-variants folded)")
    return speakers, gender_map


def run_select(args: argparse.Namespace) -> None:
    device_id = 0
    if ":" in args.embedding_device:
        device_id = int(args.embedding_device.split(":")[-1])

    speakers, gender_map = _load_perltqa_speakers(args.data)

    print("[select] loading WeSpeaker embedding model ...")
    embedder = SpeakerEmbedder(args.embedding_model_dir, device=device_id)

    cache = build_or_load_cache(
        cache_path=args.cache,
        embedder=embedder,
        local_dataset_dir=args.local_dataset_dir,
        hf_dataset=args.hf_dataset,
        rebuild=args.rebuild_cache,
    )

    assignment, report = select_references(cache, speakers, gender_map, args.t_ref)

    if args.select_only:
        report_path = args.ref_dir / "selection_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[select] --select-only: wrote assignment report -> {report_path} "
              f"(no wav files written)")
        return

    payload = materialize_references(
        assignment=assignment,
        ref_dir=args.ref_dir,
        ref_map_path=args.ref_map or (args.ref_dir / "reference_voice_map.json"),
        local_dataset_dir=args.local_dataset_dir,
        hf_dataset=args.hf_dataset,
        target_sr=args.target_sr,
        ref_duration=args.ref_duration,
        cache=cache,
    )
    report["num_references_written"] = len(payload["reference_voice_map"])
    report_path = args.ref_dir / "selection_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[select] selection report -> {report_path}")


# ----------------------------------------------------------------------------
# Stage 2: check (post-TTS separation audit)
# ----------------------------------------------------------------------------
def _speaker_audio_from_block(meta: dict) -> Dict[str, Tuple[np.ndarray, int]]:
    """
    Reconstruct each speaker's *speech-only* audio for one dialogue block by
    slicing that speaker's isolated track (npy) at its own turn boundaries.

    Returns {speaker_name: (waveform, sample_rate)}.
    """
    sr = int(meta["sample_rate"])
    channel_map = meta.get("channel_map", {})
    turns = meta.get("turns", [])

    # Load each speaker's isolated npy track once.
    tracks: Dict[str, np.ndarray] = {}
    for name, info in channel_map.items():
        npy = info.get("npy_file") or info.get("file")
        if npy and Path(npy).exists():
            tracks[name] = np.load(npy)

    seg_by_speaker: Dict[str, List[np.ndarray]] = defaultdict(list)
    for t in turns:
        name = t.get("speaker_name")
        if name not in tracks:
            continue
        s, e = int(t["start_sample"]), int(t["end_sample"])
        seg = tracks[name][s:e]
        if len(seg):
            seg_by_speaker[name].append(seg)

    out: Dict[str, Tuple[np.ndarray, int]] = {}
    for name, segs in seg_by_speaker.items():
        if segs:
            out[name] = (np.concatenate(segs).astype(np.float32), sr)
    return out


def run_check(args: argparse.Namespace) -> None:
    device_id = 0
    if ":" in args.embedding_device:
        device_id = int(args.embedding_device.split(":")[-1])

    print("[check] loading WeSpeaker embedding model ...")
    embedder = SpeakerEmbedder(args.embedding_model_dir, device=device_id)

    block_files = sorted(Path(args.output_dir).rglob("channel_map.json"))
    if args.limit > 0:
        block_files = block_files[: args.limit]
    print(f"[check] found {len(block_files)} dialogue block(s) under {args.output_dir}")

    tmp_dir = tempfile.mkdtemp(prefix="wespk_check_")
    # Per-name embeddings across all blocks (for the global cross-block view).
    name_embs: Dict[str, List[np.ndarray]] = defaultdict(list)
    block_reports: List[dict] = []
    worst_offenders: List[dict] = []

    try:
        for bf in block_files:
            try:
                meta = json.loads(bf.read_text())
            except Exception as e:
                print(f"[check][WARN] {bf}: {e}", file=sys.stderr)
                continue

            spk_audio = _speaker_audio_from_block(meta)
            if len(spk_audio) < 2:
                continue

            embs: Dict[str, np.ndarray] = {}
            durs: Dict[str, float] = {}
            for name, (wav, sr) in spk_audio.items():
                dur = len(wav) / sr
                durs[name] = dur
                if dur < args.min_seconds:
                    continue  # too short to embed reliably
                wav16, sr16 = _prep_for_embedding(wav, sr)
                path = _write_tmp_wav(wav16, sr16, tmp_dir, "seg")
                try:
                    e = embedder.extract(path)
                finally:
                    os.remove(path)
                embs[name] = e
                name_embs[name].append(e)

            names = sorted(embs.keys())
            pairs = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    s = cosine_similarity(embs[names[i]], embs[names[j]])
                    pairs.append((names[i], names[j], s))
            if not pairs:
                continue
            worst = max(pairs, key=lambda p: p[2])
            rel = str(bf.parent.relative_to(args.output_dir))
            block_reports.append({
                "block": rel,
                "num_speakers_embedded": len(names),
                "worst_pair": [worst[0], worst[1]],
                "worst_pair_cosine": worst[2],
                "min_speech_sec": min(durs.values()) if durs else 0.0,
            })
            for a, b, s in pairs:
                if s >= args.flag_sim:
                    worst_offenders.append({
                        "block": rel, "speaker_a": a, "speaker_b": b, "cosine": s,
                    })
    finally:
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    # ── Global cross-block view ───────────────────────────────────────
    # Mean embedding per speaker name, then different-speaker similarities.
    name_mean = {n: np.mean(np.stack(es), axis=0) for n, es in name_embs.items()
                 if len(es) >= 1}
    same_name_consistency = {}
    for n, es in name_embs.items():
        if len(es) >= 2:
            sims = [cosine_similarity(es[i], es[j])
                    for i in range(len(es)) for j in range(i + 1, len(es))]
            same_name_consistency[n] = float(np.mean(sims))

    diff_names = sorted(name_mean.keys())
    diff_sims: List[float] = []
    global_collisions: List[dict] = []
    rng = np.random.default_rng(42)
    idx_pairs = [(i, j) for i in range(len(diff_names)) for j in range(i + 1, len(diff_names))]
    if len(idx_pairs) > args.max_global_pairs:
        sel = rng.choice(len(idx_pairs), size=args.max_global_pairs, replace=False)
        idx_pairs = [idx_pairs[i] for i in sel]
    for i, j in idx_pairs:
        a, b = diff_names[i], diff_names[j]
        s = cosine_similarity(name_mean[a], name_mean[b])
        diff_sims.append(s)
        if s >= DOWNSTREAM_MERGE_SIM:
            global_collisions.append({"speaker_a": a, "speaker_b": b, "cosine": s})

    worst_offenders.sort(key=lambda d: -d["cosine"])
    global_collisions.sort(key=lambda d: -d["cosine"])

    summary = {
        "num_blocks_checked": len(block_reports),
        "flag_sim": args.flag_sim,
        "downstream_merge_sim": DOWNSTREAM_MERGE_SIM,
        "within_block": {
            "worst_pair_cosine": max((b["worst_pair_cosine"] for b in block_reports),
                                     default=None),
            "mean_worst_pair_cosine": float(np.mean(
                [b["worst_pair_cosine"] for b in block_reports])) if block_reports else None,
            "num_flagged_pairs": len(worst_offenders),
        },
        "global": {
            "num_speaker_names": len(name_mean),
            "different_speaker_cosine_mean": float(np.mean(diff_sims)) if diff_sims else None,
            "different_speaker_cosine_max": float(np.max(diff_sims)) if diff_sims else None,
            "num_collisions_over_merge_sim": len(global_collisions),
            "same_name_consistency_mean": float(np.mean(
                list(same_name_consistency.values()))) if same_name_consistency else None,
        },
        "flagged_within_block_pairs": worst_offenders[:50],
        "global_collisions": global_collisions[:50],
        "block_reports": block_reports,
    }

    out_path = Path(args.report) if args.report else (Path(args.output_dir) / "voice_check_report.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 64)
    print("  POST-TTS SPEAKER SEPARATION CHECK")
    print("=" * 64)
    wb = summary["within_block"]
    gl = summary["global"]
    print(f"  blocks checked            : {summary['num_blocks_checked']}")
    print(f"  within-block worst cosine : {wb['worst_pair_cosine']}")
    print(f"  within-block flagged pairs (>= {args.flag_sim}): {wb['num_flagged_pairs']}")
    print(f"  global speaker names      : {gl['num_speaker_names']}")
    print(f"  different-speaker cos mean: {gl['different_speaker_cosine_mean']}")
    print(f"  different-speaker cos max : {gl['different_speaker_cosine_max']}")
    print(f"  collisions >= merge {DOWNSTREAM_MERGE_SIM}   : {gl['num_collisions_over_merge_sim']}")
    print(f"  same-name consistency mean: {gl['same_name_consistency_mean']}")
    if worst_offenders[:10]:
        print("\n  Top within-block collisions:")
        for o in worst_offenders[:10]:
            print(f"    {o['cosine']:.4f}  {o['block']}  [{o['speaker_a']} ~ {o['speaker_b']}]")
    print(f"\n[check] report -> {out_path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # shared embedding args
    def add_embed_args(p):
        p.add_argument("--embedding-model-dir", default=DEFAULT_EMBED_MODEL_DIR)
        p.add_argument("--embedding-device", default="cuda:0")

    # ── select ──
    ps = sub.add_parser("select", help="build cache + FPS reference selection")
    add_embed_args(ps)
    ps.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ps.add_argument("--ref-dir", type=Path, default=DEFAULT_REF_DIR)
    ps.add_argument("--ref-map", type=Path, default=None,
                    help="output map JSON (default: <ref-dir>/reference_voice_map.json)")
    ps.add_argument("--cache", type=Path, default=None,
                    help="embedding cache .npz (default: <ref-dir>/libritts_wespeaker_cache.npz)")
    ps.add_argument("--local-dataset-dir", type=Path, default=DEFAULT_LOCAL_DATASET_DIR)
    ps.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    ps.add_argument("--t-ref", type=float, default=DEFAULT_T_REF,
                    help="max allowed cosine sim to already-accepted same-gender picks")
    ps.add_argument("--target-sr", type=int, default=DEFAULT_TARGET_SR)
    ps.add_argument("--ref-duration", type=float, default=DEFAULT_REF_DURATION_SEC)
    ps.add_argument("--rebuild-cache", action="store_true",
                    help="ignore any existing cache and re-embed the whole bank")
    ps.add_argument("--select-only", action="store_true",
                    help="only run selection + write report, do not write wavs")

    # ── check ──
    pc = sub.add_parser("check", help="audit synthesized speaker separation")
    add_embed_args(pc)
    pc.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help="root dir containing generated dialogue blocks")
    pc.add_argument("--report", type=Path, default=None,
                    help="output JSON (default: <output-dir>/voice_check_report.json)")
    pc.add_argument("--flag-sim", type=float, default=DEFAULT_FLAG_SIM,
                    help="flag within-block different-speaker pairs at/above this cosine")
    pc.add_argument("--min-seconds", type=float, default=3.0,
                    help="skip a speaker whose synthesized speech is shorter than this")
    pc.add_argument("--max-global-pairs", type=int, default=200000)
    pc.add_argument("--limit", type=int, default=0,
                    help="cap number of blocks to check (<=0 = all)")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "select":
        if args.cache is None:
            args.cache = args.ref_dir / "libritts_wespeaker_cache.npz"
        run_select(args)
    elif args.cmd == "check":
        run_check(args)


if __name__ == "__main__":
    main()
