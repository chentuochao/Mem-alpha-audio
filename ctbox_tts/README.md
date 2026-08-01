# ctbox_tts — PerLTQA Dialogue TTS

Synthesize multi-party spoken dialogues for the **PerLTQA** dataset using
[Chatterbox TTS](https://github.com/resemble-ai/chatterbox), with reference
voices drawn from the `sdialog/voices-libritts` bank. Each conversation is
rendered as isolated per-speaker tracks plus mixed multichannel/mono wavs, so
downstream diarization / ASR has clean ground truth.

---

## Pipeline at a glance

```
PerLTQA JSON                LibriTTS voice bank            WeSpeaker model
     │                             │                             │
     ▼                             ▼                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 1. SELECT   select_and_check_voices.py select                        │
│    embed the voice bank (cached) → greedy farthest-point pick one     │
│    distinct, well-separated reference voice per speaker               │
│    → ref_voices/perltqa/reference_voice_map.json                      │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. GENERATE   run_tts_multigpu.sh  →  perltqa_dialogue_tts.py         │
│    parse dialogues → Chatterbox synthesis, sharded across N GPUs      │
│    → <output>/<Profile>/<dialogue_id>/…                               │
└─────────────────────────────────────────────────────────────────────┘
     │
     ├───────────────────────────────┐
     ▼                               ▼
┌──────────────────────────┐   ┌──────────────────────────────────────┐
│ 3. ANNOTATE              │   │ 4. CHECK (optional QA)                 │
│ generate_annotations.py  │   │ select_and_check_voices.py check       │
│ Silero-VAD + WhisperX    │   │ verify synthesized speakers stay        │
│ → *_vad.npy / *.json     │   │ separable in embedding space            │
└──────────────────────────┘   └──────────────────────────────────────┘
```

Steps 3 and 4 are independent; run either, both, or neither.

---

## Environment

```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate ctbox                      # env used throughout this README
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

Chatterbox loads its model onto the first **visible** CUDA device at import
time (`chatterbox_dialogue_tts.py`), which is what makes the one-process-per-GPU
sharding in step 2 work.

---

## Key paths (currently hard-coded)

These are set in the code, not via flags — edit the constants to relocate:

| What | Where it's set | Value |
|------|----------------|-------|
| PerLTQA input JSON | `perltqa_dialogue_tts.py` `main()` | `/checkpoint/seamless/tuochao/data/PerLTQA/Dataset/en_v2/perltmem_en_v2.json` |
| TTS output dir | `perltqa_dialogue_tts.py` `main()` | `/checkpoint/seamless/tuochao/data/PerLTQA/dialogue_tts_en_v2` |
| LibriTTS voice bank (local parquet) | `DEFAULT_LOCAL_DATASET_DIR` | `/checkpoint/seamless/tuochao/data/ctbox_tts/voices-libritts/` |
| Reference-voice dir | `DEFAULT_REF_DIR` | `ctbox_tts/ref_voices/perltqa/` |
| WeSpeaker model | `select_and_check_voices.py` `DEFAULT_EMBED_MODEL_DIR` | `/checkpoint/seamless/tuochao/Models/huggingface/wespeaker-voxceleb-resnet293-LM` |

---

## Step 1 — Select reference voices

Assigns every PerLTQA speaker a distinct LibriTTS reference voice, chosen to be
maximally separated in speaker-embedding space (so different speakers don't
sound alike). Details in the module docstring.

```bash
python select_and_check_voices.py select                 # embedding-aware pick
python select_and_check_voices.py select --rebuild-cache # re-embed the bank
python select_and_check_voices.py select --t-ref 0.45    # stricter separation
```

Produces, under `ref_voices/perltqa/`:

- `reference_voice_map.json` — `{speaker_name: wav_path}` (consumed by step 2)
- `<Speaker>/reference.wav` — the assigned 10 s reference clip per speaker
- `libritts_wespeaker_cache.npz` — cached embeddings of the whole voice bank
  (2455 × 256), so re-runs are near-instant. Build once; reuse forever.
- `selection_report.json` — per-gender `worst_pair_cosine` separation guarantee

> Alternative (legacy): `perltqa_dialogue_tts.py --prepare-only` builds a map by
> taking the *first* gender-matching voice, with **no** embedding-separation
> guarantee. Prefer `select` for the PerLTQA runs.

---

## Step 2 — Generate dialogue TTS

### Multi-GPU (recommended)

`run_tts_multigpu.sh` launches one process per GPU, each pinned with
`CUDA_VISIBLE_DEVICES` and handed a disjoint 1/N shard of dialogue blocks
(`block_index % N == shard_index`, round-robin for load balance).

```bash
bash run_tts_multigpu.sh                 # 8 GPUs, all dialogues
NUM_GPUS=4 bash run_tts_multigpu.sh      # 4 GPUs
LIMIT=5 bash run_tts_multigpu.sh         # smoke test: 5 blocks PER shard
OVERWRITE=1 bash run_tts_multigpu.sh     # re-render existing blocks
SHOW_TQDM=1 bash run_tts_multigpu.sh     # keep Chatterbox's sampling bar
```

Per-shard logs go to `logs_tts_multigpu/shard_<k>.log` (each block prints a
`START … / DONE … in Xs` line). Tail one with `tail -f logs_tts_multigpu/shard_0.log`.

> **Important:** the launcher runs workers with `--skip-prepare`, so step 1 must
> have produced `reference_voice_map.json` first — otherwise all workers race on
> building it.

### Single process

```bash
python perltqa_dialogue_tts.py --count-only     # just parse + write stats
python perltqa_dialogue_tts.py --skip-prepare --limit 5   # first 5 blocks
python perltqa_dialogue_tts.py --skip-prepare --limit 0   # everything (heavy)
```

Useful flags: `--skip-prepare` (reuse the map), `--overwrite`,
`--num-shards`/`--shard-index` (manual sharding), `--ref-dir`, `--ref-map`.

> `--limit` applies **per shard**: `LIMIT=5` on 8 GPUs synthesizes ~40 blocks.

---

## Step 3 — Annotations (ground truth)

Per-speaker ground truth. The **transcript** is taken **directly from
`channel_map.json`** (exact PerLTQA turn text + exact turn timing — no ASR/
WhisperX). The **VAD** is produced with **Silero-VAD** over each speaker's
isolated `_TTS.npy` track (captures within-turn pauses). Turn-level transcript is
all Step-1 / eval need (cpWER uses per-speaker text, DER uses per-turn timing).

```bash
python generate_annotations.py --output-dir /checkpoint/seamless/tuochao/data/PerLTQA/dialogue_tts_en_v2
python generate_annotations.py --output-dir <dir> --limit 5 --overwrite --vad-threshold 0.5
```

Writes next to each track:
- `<Speaker>_annotation.json` — `transcript.segments` = turn list
  `{speaker, start, end, text}` (`source: perltqa_channel_map`);
  `vad.segments` = Silero speech segments (`source: silero_vad`).
- `<Speaker>_vad.npy` — int8 frame labels @ 0.08 s (rasterized from Silero VAD).

---

## Step 4 — Separation check (optional QA)

Reference-space separation (step 1) does not guarantee TTS-*output* separation —
Chatterbox can compress voices together. This audits the actual synthesized
audio: it reconstructs each speaker's speech from its isolated track, embeds it
with WeSpeaker, and flags different-speaker pairs that land too close.

```bash
python select_and_check_voices.py check     # defaults to the step-2 output dir
python select_and_check_voices.py check --flag-sim 0.5
```

Writes `voice_check_report.json` (within-block worst pairs + global collisions
vs. the downstream merge threshold 0.65).

---

## Output structure

```
<output_dir>/
├── speaker_stats.json                      # written at parse time
├── voice_check_report.json                 # from step 4
├── <ProfileOwner>/                         # e.g. Li_Hua/
│   └── <dialogue_id>/                       # one conversation block
│       ├── <Speaker1>_TTS.npy               # isolated per-speaker track
│       ├── <Speaker2>_TTS.npy
│       ├── dialogue_multichannel_TTS.wav    # N channels, one per speaker
│       ├── dialogue_mono_TTS.wav            # mono mix-down
│       ├── channel_map.json                 # metadata + turn timing + DONE-marker
│       ├── <Speaker1>_vad.npy               # (step 3) frame labels
│       └── <Speaker1>_annotation.json       # (step 3) VAD + ASR
└── …
```

**Resume / re-run:** the only generation-resume state is the per-block
`channel_map.json`. A block whose `channel_map.json` exists is skipped unless
`--overwrite` (`OVERWRITE=1`). Deleting the output dir forces a full re-render;
nothing under `ref_voices/` affects generation resume.

---

## Files

| File | Role |
|------|------|
| `perltqa_dialogue_tts.py` | Main driver: parse PerLTQA → prepare/reuse refs → sharded generation |
| `chatterbox_dialogue_tts.py` | Chatterbox wrapper: `generate_dialogue_tts` (2-party) + `generate_multispeaker_dialogue_tts` (N-party) |
| `select_and_check_voices.py` | `select` (embedding-aware ref picking) + `check` (post-TTS separation audit) |
| `run_tts_multigpu.sh` | One-process-per-GPU launcher with block-level sharding |
| `generate_annotations.py` | Silero-VAD + WhisperX annotations on the outputs |
| `audio_prepare.py` | Legacy: builds refs for the old fixed 10-speaker demo map |
| `ref_voices/` | Cached reference wavs + `reference_voice_map.json` |

---

## Low-level API (direct import)

`chatterbox_dialogue_tts` can be called directly, independent of the PerLTQA flow.
Every speaker name used must be present in `REFERENCE_VOICE_MAP` (populate it, or
pass through `perltqa_dialogue_tts.generate_all`, which updates it from the map).

### Two-party

```python
from chatterbox_dialogue_tts import generate_dialogue_tts

result = generate_dialogue_tts(
    speaker1_name="Wang Xiaoming",
    speaker2_name="Wang Xiaohong",
    speaker1_text=["Hello Xiaohong.", "Left channel is my voice."],
    speaker2_text=["Hi Xiaoming.",   "Right channel is mine."],
    output_dir="tts_outputs/test",
)
# result: {"speaker1_tts": np.ndarray, "speaker2_tts": np.ndarray, "metadata": dict}
```

Guarantees: both tracks are equal length; speaker1→left, speaker2→right; the
non-speaking channel is zero-padded; the mapping is saved to `channel_map.json`.
Speaker 1 always starts. Lists = one turn per element, strictly alternating.

### N-party

```python
from chatterbox_dialogue_tts import generate_multispeaker_dialogue_tts

generate_multispeaker_dialogue_tts(
    speaker_names=["Alice", "Bob", "Carol"],      # each gets its own channel
    speaker_texts=[{"Alice": "Hi both."},          # ordered list of turns
                   {"Bob":   "Hey Alice."},
                   {"Carol": "Good to see you."}],
    output_dir="tts_outputs/multi",
)
```

Turns are placed on a shared timeline with randomized gaps/overlaps
(`get_interval`), then mixed per-speaker (`+=`, so overlaps blend) and clipped.

---

## Notes & gotchas

- **First turn per process is slow** — Chatterbox `torch.compile`s the model on
  the first `generate()` call; steady-state is much faster.
- **tqdm "Sampling" bar** comes from Chatterbox's T3 decode, not this code; the
  launcher silences it via `TQDM_DISABLE=1` (override with `SHOW_TQDM=1`).
- **WeSpeaker GPU fix:** `select_and_check_voices.py` patches a WeSpeaker build
  bug that leaves fbank features on CPU (crashes on GPU). No action needed.
- **CPU threads:** the launcher sets `OMP_NUM_THREADS=4` so N processes don't
  oversubscribe the node.
```
