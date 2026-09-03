# Bazinga (The Big Bang Theory) Audio Memory Pipeline

End-to-end guide for running the **Bazinga** dataset through the audio memory
agent: from raw episode audio → diarized/ASR'd dialogue → Parquet → memory
construction + QA evaluation → error probing / visualization.

- **Dataset:** The Big Bang Theory (TBBT), organized by season/episode. Every
  other show under `/checkpoint/seamless/tuochao/data/bazinga/data/` (Friends,
  TheOffice, …) uses the identical layout and runs through the same scripts —
  see [Running a different show](#running-a-different-show).
- **Raw data path:** `/checkpoint/seamless/tuochao/data/bazinga/data/TheBigBangTheory`
  - Noisy variants live in sibling `*_SNRx` / `*_interf_SNRx` folders, generated
    in Stage 0.
- **Repo root:** `/storage/home/tuochao/Mem-alpha-audio`

The pipeline has five stages:

| Stage | What it does | Entry point |
|-------|--------------|-------------|
| 0. Audio prep | noise/interference mixing, session timeline | `audio_script/run_mix_wham_*.sh`, `prepare_data/make_bazinga_timeline.py` |
| 1. Audio pipeline | diarization + ASR → speaker matching → speaker-name extraction | `audio_script/run_demo_pipeline_bazinga.sh` |
| 2. Parquet build | pack dialogue chunks into a Parquet dataset | `prepare_data/prepare_parquet_from_step3*.py` |
| 3. Memory + QA | build memory, then answer/score QA | `run_pipeline.sh` / `submit_sweep.sh` |
| 4. Error probing | attribute QA errors to a pipeline stage + plot | `diagnostic/run_probe_errors.sh` |

Stage 0 is only needed for a new show or a new noise condition; the clean TBBT
folders and `outputs/bazinga_data/TBBT_all_seasons_session_timeline.json` already
exist.

---

## Prerequisites

**Conda environments** (the scripts activate these by name):

| Env | Used for |
|-----|----------|
| `vibevoice` | Step1 ASR with the VibeVoice backend (default `METHOD`) |
| `nemo` | Step1 ASR with the NeMo backends (`nemo-streaming` / `nemo-offline`) |
| `mem` | Step2/Step3, QA evaluation, plotting (flask + matplotlib) |
| `vllm` | Memory construction (step 1 of `run_pipeline.sh`), reward-model server |

**A Qwen reward/LLM server** must be reachable:
- Step3 name extraction expects `QWEN_URL=http://localhost:8002/v1` and
  `QWEN_MODEL_NAME` equal to the server's `--served-model-name` (`qwen3-32b` for
  both `launch_vllm.sh` and `launch_vllm_qwen36.sh`).
- Memory construction / QA expects a vLLM server; on SLURM this is launched
  automatically by `submit_pipeline.slurm` (see Stage 3).

**Models** (paths baked into `run_demo_pipeline_bazinga.sh`, override via env):
- Diarization: `diar_streaming_sortformer_4spk-v2.1` (NeMo backend only)
- ASR: `multitalker-parakeet-streaming-0.6b-v1` (NeMo) or `microsoft/VibeVoice-ASR` (VibeVoice)
- Speaker embedding: `wespeaker-voxceleb-resnet293-LM`

Everything below assumes you run from the repo root and have
`PYTHONPATH=/storage/home/tuochao/Mem-alpha-audio`.

---

## Stage 0 — Audio preparation

### Raw data layout

Every show folder under `/checkpoint/seamless/tuochao/data/bazinga/data/` is flat,
two files per episode:

```
<Show>/
  <Show>.SeasonNN.EpisodeMM.en.wav    # 16 kHz mono PCM_16, ~22 min
  <Show>.SeasonNN.EpisodeMM.txt       # word-level annotation, one word per line:
                                      #   file_id speaker start end word score ...
  episodes.txt / characters.txt / credits.txt / episodes.MISSING.txt   # metadata, ignored
```

`Bazinga_loader.BazingaDataset` reads this pair directly; nothing needs
pre-converting. The stem (`<Show>.SeasonNN.EpisodeMM`) is the `conv_id` used as
the episode folder name everywhere downstream, and as the session-timeline key.

### 0a. Noise / interference mixing (optional, for SNR ablations)

Both scripts write **sibling** folders next to `DATA_PATH` containing noisy
`.en.wav` plus a verbatim copy of the `.txt`, so the result is a drop-in
`RAW_DATA_PATH` for Stage 1. Edit `DATA_PATH`, `SNRS`, and `SEASON_FILTER` at the
top of each script (they are plain assignments, not env-overridable).

```bash
# WHAM background noise -> <DATA_PATH>_SNR10 / _SNR5 / _SNR0 / _SNR-5
bash audio_script/run_mix_wham_noise.sh

# Competing speech (sdialog/voices-libritts) -> <DATA_PATH>_interf_SNR15 / _SNR10 / _SNR5
bash audio_script/run_mix_wham_inteference.sh
```

Both run in the `nemo` env (needs `datasets` + `soundfile` + `librosa`) and call
`audio_script/datasets/mix_wham_noise.py` / `mix_speech_interference.py`, which
you can also invoke directly:

```bash
python -m audio_script.datasets.mix_wham_noise \
  --data_dir /checkpoint/seamless/tuochao/data/bazinga/data/Friends \
  --snr 10 5 0 --noise_pool_minutes 30 --seed 0 \
  --season_filter Season01 Season02 Season03
```

Only the noise gain differs between `_SNRx` folders at a given seed, so DER/WER
deltas across them are attributable to SNR alone.

### 0b. Session timeline

Bazinga has no real timestamps, so each episode is assigned a synthetic
*benchmark history date*: the first episode gets `--start_date` and every later
episode advances one week in canonical Season/Episode order, without resetting at
season boundaries. Stage 2 stamps chunks with these dates.

`prepare_data/make_bazinga_timeline.py` generates one for any show:

```bash
python prepare_data/make_bazinga_timeline.py \
  --data_dir /checkpoint/seamless/tuochao/data/bazinga/data/Friends
# -> outputs/bazinga_data/Friends_all_seasons_session_timeline.json
#    show: Friends | seasons 1-10 | 233 episodes | 2023-05-01 -> 2027-10-11
```

It accepts either a raw show dir or an existing Step1/Step2 tree of episode
folders, and skips episodes absent from the source (e.g. Friends S05E24, listed
in `episodes.MISSING.txt`) while keeping the rest consecutive. Pass the result to
Stage 2 with `--time_info_path`; the TBBT default is already checked in.

---

## Stage 1 — Audio pipeline (diar + ASR → speaker tracking → names)

Script: `audio_script/run_demo_pipeline_bazinga.sh`

This runs three sub-steps in one driver:

1. **Step1** (`Multi_ASR/step1_bazinga.py`) — diarization + ASR over **all**
   seasons at once. No cross-season state.
2. **Step2** (`Speaker_Track/step2_speaker_match_v2.py`) — matches speakers to a
   voice-embedding pool, **season by season**.
3. **Step3** (`Speaker_Track/step3_speaker_name_extract.py`) — extracts real
   speaker names via the Qwen server, **season by season**.

**Incremental pool semantics:** the *first* season in `SEASONS` runs with
`--update_pool` (builds the initial speaker pool `pool.npz` + name state
`speakers_name_pool.json`); every later season is matched against that **frozen**
pool and does not mutate it.

### Run it

```bash
cd /storage/home/tuochao/Mem-alpha-audio

# Default: VibeVoice backend, clean audio, Season01–Season03.
bash audio_script/run_demo_pipeline_bazinga.sh
```

### Common overrides (inline env vars)

```bash
# NeMo streaming backend instead of VibeVoice
METHOD=nemo-streaming bash audio_script/run_demo_pipeline_bazinga.sh

# Run on a noisy SNR variant (point at the *_SNRx raw folder)
RAW_DATA_PATH=/checkpoint/seamless/tuochao/data/bazinga/data/TheBigBangTheory_SNR10 \
  bash audio_script/run_demo_pipeline_bazinga.sh

# Only re-run a later phase (skip Step1); resume on existing pool/state
RUN_STEP1=0 RESET_STATE=0 bash audio_script/run_demo_pipeline_bazinga.sh

# Step1 only, one season
SEASONS="Season01" RUN_STEP2=0 RUN_STEP3=0 bash audio_script/run_demo_pipeline_bazinga.sh

# Single episode (season_filter is a substring match on the episode id)
SEASONS="Season01.Episode01" RUN_STEP2=0 RUN_STEP3=0 \
  bash audio_script/run_demo_pipeline_bazinga.sh
```

Every knob below is `${VAR:-default}`, so inline env vars win:

| Var | Default | Notes |
|-----|---------|-------|
| `METHOD` | `vibevoice` | `vibevoice` \| `nemo-streaming` \| `nemo-offline` |
| `RAW_DATA_PATH` | `.../TheBigBangTheory` | basename becomes `DATASET_TAG` in the output tree |
| `SEASONS` | `"Season01 Season02 Season03"` | space-separated; substring-matched against the episode id |
| `RUN_STEP1/2/3` | `1` | set to `0` to skip a phase |
| `RESET_STATE` | `1` | wipes `pool.npz` + `speakers_name_pool.json`; see the caveat below |
| `SIMILARITY_THRESHOLD` | `0.5` | Step2 speaker-embedding match threshold |
| `EMBEDDING_MODEL_DIR`, `EMBEDDING_DEVICE` | wespeaker resnet293, `cuda:0` | Step2 |
| `QWEN_URL` | `http://localhost:8002/v1` | Step3 LLM server |
| `QWEN_MODEL_NAME` | `qwen3-32b` | **must equal the server's `--served-model-name`**, not the HF repo id |
| `MAX_NEW_TOKENS` | `8192` | VibeVoice generation cap |
| `REPETITION_PENALTY` | `1.0` | **leave at 1.0** — see below |
| `NO_REPEAT_NGRAM_SIZE` | `0` | preferred anti-degeneration knob |
| `TEMPERATURE`, `TOP_P`, `NUM_BEAMS`, `ATTN_IMPL` | `0.0`, `1.0`, `1`, `auto` | VibeVoice decoding |

> **Do not raise `REPETITION_PENALTY` above 1.0.** The VibeVoice output format
> requires `"Speaker"`, `"Content"` and the speaker-id digits to repeat once per
> segment; penalizing them pushes the model into emitting fewer, longer segments
> that merge several speakers' turns into one `Content` separated by `\n`, all
> tagged with a single speaker id. Measured on Friends S01E01 CHUNK_0 (6 GT
> speakers): `1.0` → 4 speakers / 15 segments; `1.2` → 1 speaker / 7 segments.
> It costs TBBT accuracy too (3 GT speakers: `1.0` → 3, `1.2` → 2). If a chunk
> degenerates into a repetition loop and never emits EOS, use
> `NO_REPEAT_NGRAM_SIZE` instead — it doesn't distort diarization.

> `RESET_STATE` runs **outside** the Step2/Step3 guard, so it deletes the pool
> and name state even with `RUN_STEP2=0 RUN_STEP3=0`. Pass `RESET_STATE=0` to
> keep them.

### Resuming an interrupted run

Step1 skips any chunk whose `diart_pred.npy` + `transcript_pred.json` +
`sample_info.json` all exist (`step1_bazinga.py:179`), so resume is chunk-level:
re-run the same command and it picks up where it stopped, including mid-episode.

The corollary: **stale output is never regenerated.** After changing a decoding
knob, delete the affected tree (`rm -rf Audio_Results/<METHOD>/<TAG>/step1`)
or the run will silently reuse the old results.

There is no watchdog — if a chunk blocks inside `generate()`, the run stalls
indefinitely with no error and no log output. Check liveness with the mtime of
the newest file in the output tree, not with `squeue`:

```bash
find Audio_Results/vibevoice/<TAG>/step1 -type f -printf '%T@ %TF %TR %p\n' | sort -n | tail -1
```

### Running a different show

Nothing is TBBT-specific — point `RAW_DATA_PATH` at any show folder. `DATASET_TAG`
is its basename, so results and state auto-separate.

```bash
RAW_DATA_PATH=/checkpoint/seamless/tuochao/data/bazinga/data/Friends \
SEASONS="Season01 Season02 Season03" \
  bash audio_script/run_demo_pipeline_bazinga.sh
# -> Audio_Results/vibevoice/Friends/{step1,step2}
```

Cost scales with episode count: ~10 s per chunk, ~13 chunks per episode, so
~2.5 min/episode (Friends Seasons 1–3 = 72 episodes ≈ 3 h; all 233 ≈ 10 h). Use
`sbatch`, not an interactive job.

Speaker load varies a lot by show, and Step2/Step3 get harder as it rises. Mean
unique speakers per episode in the shipped data:

| Show | Episodes | Speakers/episode (min / median / max) |
|------|----------|---------------------------------------|
| 24 | 204 | 1 / 1 / 24 |
| BreakingBad | 61 | 1 / 1 / 19 |
| ER | 330 | 1 / 2 / 52 |
| Homeland | 70 | 1 / 1 / 29 |
| SixFeetUnder | 63 | 1 / 1 / 26 |
| TheWalkingDead | 99 | 1 / 8 / 24 |
| **TheBigBangTheory** | **207** | **5 / 9 / 18** |
| **Friends** | **233** | **6 / 12 / 20** |
| BuffyTheVampireSlayer | 143 | 9 / 18 / 30 |
| TheOffice | 188 | 11 / 19 / 46 |
| Lost | 104 | 11 / 18 / 45 |
| BattlestarGalactica | 71 | 12 / 24 / 43 |
| GameOfThrones | 60 | 18 / 40 / 58 |

(`StarWars`, `HarryPotter`, `LordOfTheRings` are film sets — a handful of very
long, very crowded "episodes".)

### Evaluating Step1 quality

`audio_script/evaluate_audio_results.py` walks a Step1 tree and reports DER +
cpWER per chunk against the GT, and writes a GT-vs-pred diarization PNG each:

```bash
~/miniconda3/envs/mem/bin/python audio_script/evaluate_audio_results.py \
  Audio_Results/vibevoice/Friends/step1
```

For a quick speaker-count sanity check without running the full metric:

```bash
python3 -c "
import json,glob,collections
c=collections.Counter()
for p in glob.glob('Audio_Results/vibevoice/Friends/step1/*/CHUNK_*/transcript_pred.json'):
    c[len(json.load(open(p)))]+=1
print('pred speakers/chunk:',dict(sorted(c.items())))"
```

A spike at 1 speaker means the collapse described under `REPETITION_PENALTY`.

### Debug harnesses

Both run one chunk through VibeVoice directly, bypassing the pipeline, on a GPU
node (`srun --jobid=<JOBID> --overlap -n1 ... ` against an existing allocation;
set `HF_HUB_OFFLINE=1` since compute nodes have no proxy):

- `audio_script/debug/debug_vibevoice_chunk0.py` — ablation grid over streaming,
  clip length, prompt `context_info`, and gain, to isolate what drives a bad
  diarization.
- `audio_script/debug/debug_hang_chunk.py` — times `generate()` on a specific
  chunk and reports tokens, tok/s, and whether it hit `max_new_tokens`.

### Outputs

Results land in a per-backend / per-dataset tree:

```
Audio_Results/<METHOD>/<DATASET_TAG>/
  step1/                       # diarized + ASR dialogue
  step2/
    pool.npz                   # speaker-embedding pool (cross-season state)
    speakers_name_pool.json    # name state (cross-season)
    <show>/<episode>/CHUNK_N/
      parsed_dialog_pred.json  # ASR transcript, predicted names
      parsed_dialog_gt.json    # gold dialogue (if available)
  logs/pipeline_<timestamp>.log
```

`DATASET_TAG` = basename of `RAW_DATA_PATH` (so SNR variants auto-separate).

> Note: the pre-processed step3 chunk trees used downstream already live under
> `outputs/step3_anony/` (e.g. `S01_S03_Clean_Anoy`, `S01_S03_SNR5_Anoy`,
> `TheBigBangTheory_interf_SNR10`). Stage 2 below can consume either those or a
> fresh `Audio_Results/.../step2` tree — the layout is the same.

---

## Stage 2 — Build the Parquet dataset

A Parquet holds the dialogue chunks that the memory agent ingests. There are
three builders depending on the source and whether you want anonymized speakers.

All three share the same season timeline default:
`outputs/bazinga_data/TBBT_all_seasons_session_timeline.json` (override with
`--time_info_path`; generate one for another show with
[`make_bazinga_timeline.py`](#0b-session-timeline)). The lookup key is the
episode folder name — `source_file` minus `.json` must equal `path.split('/')[-3]`,
i.e. `<Show>.SeasonNN.EpisodeMM`; chunks whose episode is missing from the
timeline are dropped. Each chunk folder becomes `chunk_folders[i]` in the
Parquet, which is what the error probe later uses to map QA evidence → memory ids.

### 2a. From a fresh Step1/Step2/Step3 audio run

`prepare_data/prepare_audio_parquet.py` — reads
`{episode}/{CHUNK_N}/parsed_dialog_pred.json` plus the Step3 name map
`extracted_speaker_name_<Season>.json` (written into the `step2/` dir), and bakes
names in.

```bash
python -m prepare_data.prepare_audio_parquet \
  --data_dir Audio_Results/vibevoice/TheBigBangTheory/step2 \
  --output_root outputs/bazinga_data \
  --season_filter Season01
# -> outputs/bazinga_data/dataset_pred_name_Season01.parquet
# add --use_gt_name for the gold-name parquet
```

One Parquet per season — `--season_filter` both selects chunks and names the
output. With the flag it loads `extracted_speaker_name_<Season>.json`; without
it, `extracted_speaker_name.json` (all seasons).

#### For Friends

Two things must be overridden, or the run fails quietly:

- **`--time_info_path`** — the default is the TBBT timeline, whose keys are
  `TheBigBangTheory.*`. Point it at the Friends timeline from
  [Stage 0b](#0b-session-timeline) or *every* chunk is dropped with
  `WARNING: no timeline entry for 'Friends.Season01.Episode01'`.
- **`--output_root`** — the filename is `dataset_{pred,gt}_name_<Season>.parquet`
  with no show tag, so writing Friends into `outputs/bazinga_data` would
  overwrite the TBBT Parquet of the same season. `--output_root` also receives a
  per-episode dump of the named dialogue chunks, so give it its own folder.

```bash
cd /storage/home/tuochao/Mem-alpha-audio
export PYTHONPATH=/storage/home/tuochao/Mem-alpha-audio

# once, if not already built (Stage 0b)
python prepare_data/make_bazinga_timeline.py \
  --data_dir /checkpoint/seamless/tuochao/data/bazinga/data/Friends

for SEASON in Season01 Season02 Season03; do
  python -m prepare_data.prepare_audio_parquet \
    --data_dir Audio_Results/vibevoice/Friends/step2 \
    --output_root outputs/bazinga_data/Friends \
    --season_filter "${SEASON}" \
    --time_info_path outputs/bazinga_data/Friends_all_seasons_session_timeline.json
done
# -> outputs/bazinga_data/Friends/dataset_pred_name_Season01.parquet  (+ Season02, Season03)
```

Add `--use_gt_name` for the gold-name counterpart (`dataset_gt_name_<Season>.parquet`),
which Stage 4 needs as the gold-dialogue ceiling:

```bash
for SEASON in Season01 Season02 Season03; do
  python -m prepare_data.prepare_audio_parquet \
    --data_dir Audio_Results/vibevoice/Friends/step2 \
    --output_root outputs/bazinga_data/Friends \
    --season_filter "${SEASON}" \
    --time_info_path outputs/bazinga_data/Friends_all_seasons_session_timeline.json \
    --use_gt_name
done
```

Sanity-check that nothing was silently dropped — `Found N dialogue files` in the
output should match the chunk count on disk, and there should be no
`no timeline entry` warnings:

```bash
ls -d Audio_Results/vibevoice/Friends/step2/*Season01*/CHUNK_* | wc -l
# 341, matching "Found 341 dialogue files (season_filter=Season01)"
```

### 2b. From an already-named step3 folder (names baked in)

`prepare_data/prepare_parquet_from_step3.py` — no name lookup, just filters by
season and writes the Parquet. This is what the pre-built `outputs/step3_anony/*`
trees use.

```bash
python -m prepare_data.prepare_parquet_from_step3 \
  --data_dir outputs/step3_anony/S01_S03_Clean_Anoy \
  --season_filter Season01 --suffix Clean_Anoy
# -> .../dataset_pred_name_Season01_Clean_Anoy.parquet
# --use_gt_name -> dataset_gt_name_Season01_Clean_Anoy.parquet
```

### 2c. Anonymized speakers (ablation: recover identity from content)

`prepare_data/prepare_parquet_from_step3_anon.py` — replaces names with anonymous
labels. Writes to a new sibling folder `<data_dir>_anon_{global|local}`.

- `--anon_mode global` → one `Speaker{X}` map across all chunks (same person =
  same label everywhere): global differentiation, no names.
- `--anon_mode local` → `Conversation{Y}_Speaker{X}`, fresh per chunk: only
  local differentiation.

```bash
python -m prepare_data.prepare_parquet_from_step3_anon \
  --data_dir outputs/step3_anony/S01_S03_Clean_Anoy \
  --season_filter Season01 --suffix Clean --anon_mode global --dump_name_map
# -> outputs/step3_anony/S01_S03_Clean_Anoy_anon_global/
#      dataset_pred_name_anon_global_Season01_Clean.parquet
```

### 2d. Friends, pseudonymized names (`outputs/step3_anony/Friends`)

A ready-made Season01 tree (24 episodes, 341 chunks) whose
`parsed_dialog_pred.json` already carries **pseudonyms** rather than anonymous
labels — Marco Bellini, Adrian Mercer, Clara Bennett, Elena Mercer, Astrid
Larsen, Elias Foster, Victor Mercer, plus `#unknown`. Names are baked in, so this
is the **2b** path; `prepare_audio_parquet.py` (2a) is not involved.

Only `--time_info_path` needs overriding — the output goes into `--data_dir`, so
unlike 2a there is no cross-show filename collision.

```bash
cd /storage/home/tuochao/Mem-alpha-audio
export PYTHONPATH=/storage/home/tuochao/Mem-alpha-audio

python -m prepare_data.prepare_parquet_from_step3 \
  --data_dir outputs/step3_anony/Friends \
  --season_filter Season01 --suffix Anony \
  --time_info_path outputs/bazinga_data/Friends_all_seasons_session_timeline.json
# -> outputs/step3_anony/Friends/dataset_pred_name_Season01_Anony.parquet

# gold-name counterpart (see the caveat below before using it)
python -m prepare_data.prepare_parquet_from_step3 \
  --data_dir outputs/step3_anony/Friends \
  --season_filter Season01 --suffix Anony \
  --time_info_path outputs/bazinga_data/Friends_all_seasons_session_timeline.json \
  --use_gt_name
# -> outputs/step3_anony/Friends/dataset_gt_name_Season01_Anony.parquet
```

Both report `Found 341 dialogue files` with no `no timeline entry` warnings.

**Use the pred Parquet.** The matching QA set
(`outputs/step3_anony/Friends_Anony_QA/friends_s1_qa_anony.jsonl`, 276 items:
157 Content QA + 119 Named Attribution QA) phrases its answer options as
pseudonyms, and those appear **only** in `parsed_dialog_pred.json`.
`dataset_gt_name_*.parquet` carries the original names (`monica_geller`,
`rachel_green`, …) and would fail every Named Attribution question. It is usable
as a gold-dialogue ceiling only after mapping real names → pseudonyms.

The timeline generated in [Stage 0b](#0b-session-timeline) matches the dates the
QA was authored against — all 24 episodes agree (S01E01 = `2023-05-01`).

---

## Stage 3 — Memory construction + QA evaluation

### Option A — local, single run (`run_pipeline.sh`)

```bash
bash run_pipeline.sh <parquet_path> [dataset] [custom_qa_dir]
```

- `parquet_path` (required) — a Parquet from Stage 2.
- `dataset` (default `seamlessinteraction_options`; use `perltqa` for PerLTQA).
- `custom_qa_dir` (default `outputs/step3_anony/qas/`) — QA JSON/JSONL to answer.

```bash
bash run_pipeline.sh \
  outputs/step3_anony/S01_S03_Clean_Anoy/dataset_pred_name_Season01_Clean_Anoy.parquet \
  seamlessinteraction_options \
  outputs/step3_anony/qas/
```

Does **Step 1** memory construction (`vllm` env, `run_memory_construction_new.py`)
then **Step 2** QA evaluation (`mem` env, `run_qa_evaluation.py`). Requires a
Qwen vLLM server reachable via `QWEN_URL`.

Optional env knobs:
- `COMPRESSION_STRATEGY` — memory compression (`default`, `x1.5`, `x2`, `x3`,
  `x5`; keys from `config/prompts_wrt_datasource_compression.yaml`). Omitted =
  baseline (no `_comp_` postfix on the output folder).
- `SEED`, `MEM_TEMPERATURE`, `ROLLOUT_LABEL` — for seed-variance runs.
  **Temperature must be > 0** or every seed produces identical memory (the
  default config decodes greedily).

```bash
COMPRESSION_STRATEGY=x3 bash run_pipeline.sh <parquet> seamlessinteraction_options outputs/step3_anony/qas/
```

Output run dirs land under `agents/` (and/or `memory_result/`) named like:
`qwen3.6-27b_..._dataset_pred_name_Season01_Clean_Anoy_no_thinking_tokens_2048[_comp_x3]`.

### Option B — SLURM sweep (`submit_sweep.sh`)

Fans out one 2-GPU SLURM job per (compression ratio × seed). Each job brings up
its own vLLM + memory server, runs the pipeline, and tears them down — no second
terminal needed.

```bash
bash submit_sweep.sh <config_file>
```

The config is a bash file that is `source`d. See `sweep_configs/example.conf`;
real ones already exist, e.g. `sweep_configs/TBBT_clean_prediction.conf`:

```bash
PARQUET_PATH="./outputs/step3_anony/S01_S03_Clean_Anoy/dataset_pred_name_Season01_Clean_Anoy.parquet"
COMPRESSION_RATIOS=("x3" "x4")     # use "none" for baseline
SEEDS=(1 2 10)                     # cartesian product with ratios
MEM_TEMPERATURE="0.3"              # REQUIRED > 0 when sweeping seeds
DATASET="seamlessinteraction_options"
CUSTOM_QA_DIR="outputs/step3_anony/qas/"
```

```bash
bash submit_sweep.sh sweep_configs/TBBT_clean_prediction.conf
squeue --me          # track jobs
```

Extra passthrough env vars: `VLLM_SCRIPT`, `VLLM_ENV`, `MEM_ENV`, and
`ANON_SPEAKER=true` (selects the anonymized-speaker pipeline
`run_pipeline_speaker_name.sh` inside `submit_pipeline.slurm`).

Available sweep configs:
`TBBT_clean_prediction.conf`, `TBBT_clean_prediction_anon_global*.conf`,
`TBBT_clean_prediction_anon_local.conf`, `TBBT_SNR0/SNR5.conf`,
`TBBT_interf_SNR5/SNR10.conf`, `Friends_S01_anony.conf`.

### Running Friends Season01 (pseudonymized)

Uses the 2d Parquet and the Friends QA set. The only difference from a TBBT run
is `CUSTOM_QA_DIR` — the Friends QA lives in its own folder, not the shared
`outputs/step3_anony/qas/`.

```bash
# SLURM sweep (recommended) — sweep_configs/Friends_S01_anony.conf
bash submit_sweep.sh sweep_configs/Friends_S01_anony.conf
squeue --me
```

```bash
# or a single local run, against an already-running vLLM server
bash run_pipeline.sh \
  outputs/step3_anony/Friends/dataset_pred_name_Season01_Anony.parquet \
  seamlessinteraction_options \
  outputs/step3_anony/Friends_Anony_QA/
```

Results land in `agents/` as
`qwen3.6-27b_..._dataset_pred_name_Season01_Anony_no_thinking_tokens_2048`.

The shipped config runs a single baseline job (`COMPRESSION_RATIOS=("none")`);
add `"x3" "x4"` etc. to fan out. As everywhere else, `QWEN_MODEL_NAME` must match
the server's `--served-model-name` or the call 404s (see Troubleshooting).

---

## Stage 4 — Error probing + visualization

### 4a. Run the behavioral error probe

Script: `diagnostic/run_probe_errors.sh`. It re-runs the QA model on curated
contexts and attributes each error to a stage by whether the **answer flips**.
Requires the memory server up (`SERVER_URL`).

```bash
bash diagnostic/run_probe_errors.sh [BASE_DIR] [DATA_ROOT] [QA_FILE] [PARQUET] [SERVER_URL]
```

- `BASE_DIR` — the **run dir** (holds `0/` and optional `seed*/0/`), *not* the
  `.../0` subdir. A trailing `/0` is tolerated.
- `DATA_ROOT` — step3 dialogue root with both `parsed_dialog_gt.json` (gold) and
  `parsed_dialog_pred.json` (ASR).
- `QA_FILE` — default `outputs/step3_anony/qas/merged_qa_anoy.jsonl`.
- `PARQUET` — the Stage-2 Parquet (its `chunk_folders` map localizes evidence).
- `SERVER_URL` — default `http://127.0.0.1:5005/batch_process`.

```bash
bash diagnostic/run_probe_errors.sh \
  agents/qwen3.6-27b_..._dataset_pred_name_Season01_Clean_Anoy_no_thinking_tokens_2048 \
  outputs/step3_anony/S01_S03_Clean_Anoy \
  outputs/step3_anony/qas/merged_qa_anoy.jsonl \
  outputs/step3_anony/S01_S03_Clean_Anoy/dataset_gt_name_Season01_Clean_Anoy.parquet

# also run the Golden (gold-dialogue ceiling) probe:
RUN_GOLDEN=1 bash diagnostic/run_probe_errors.sh ...
```

#### Friends Season01 (pseudonymized)

The probe needs `memory_server.py` on port 5005 *and* the reward-model vLLM on
8002 — `run_pipeline.sh` does not leave them up, so start them first:

```bash
./launch_servers.sh            # both, backgrounded + health-checked
./launch_servers.sh status     # verify
```

```bash
bash diagnostic/run_probe_errors.sh \
  agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_Season01_Anony_no_thinking_tokens_2048 \
  outputs/step3_anony/Friends \
  outputs/step3_anony/Friends_Anony_QA/friends_s1_qa_anony.jsonl \
  outputs/step3_anony/Friends/dataset_gt_name_Season01_Anony.parquet
```

`DATA_ROOT` is the step3 tree, which holds both `parsed_dialog_gt.json` and
`parsed_dialog_pred.json` as required. Either Friends Parquet works for
localization — their `chunk_folders` are identical — but passing the gt one
matches the script's own auto-discovery default. The `QA_FILE` is the same file
Stage 3 uses (see the source-path note in 2d).

Probe stages: **T** (transcript) and **G** (gold, optional) are precomputed once
per `DATA_ROOT` into `<DATA_ROOT>/tg_probe_cache.json` and reused across every
base_dir / seed / compression variant; **C** (constructed memory) and **S**
(retrieval) run per instance. Writes `error_probe.json` +
`error_probe_debug.json` in each instance dir (skipped if both exist). Multiple
seeds are aggregated into `error_probe_seed_summary.json`.

Evidence localization matches `gt_source.sources[].file` against the Parquet's
`chunk_folders`, which are `{episode}/CHUNK_N` (e.g.
`Friends.Season01.Episode01/CHUNK_0`). A QA file whose `file` paths carry an
extra leading component will silently fail to localize — the QA still scores in
Stage 3, but the probe loses its evidence.

> **Fixed defect — Friends QA source paths.** As generated,
> `outputs/step3_anony/Friends_Anony_QA/friends_s1_qa_anony.jsonl` had all 157
> **Content QA** items prefixed with an extra `Friends/`
> (`Friends/Friends.Season01.Episode01/CHUNK_3/...`), so they did not localize:
> 120 of 278 sources mapped, 158 did not. (`normalize_qa_sources()` only repairs
> the perltqa and missing-`file` schemas, so it does not help here.)
>
> The prefix has been **stripped in place** — the file now maps 278/278. Only
> `gt_source.sources[].file` changed; questions, options and answers are
> byte-identical, so Stage-3 results produced before the fix remain valid (neither
> `run_qa_evaluation.py` nor `evaluate_agent_results.py` reads `gt_source`). The
> pre-fix file is kept at
> `outputs/step3_anony/friends_s1_qa_anony.jsonl.orig` — deliberately *outside*
> any QA dir, since `load_custom_qa_from_dir()` loads every `.json`/`.jsonl` in
> the directory it is given and a second copy would double the QA set.
>
> If you regenerate this QA set (e.g. for Season02), check the prefix again.
>
> (These items also carry a placeholder `session_timeline_date` of `2026-08-24`
> instead of the episode date. Harmless — no code under `diagnostic/` reads that
> field.)

### 4b. Compare probes (tables / CSV)

`diagnostic/compare_probes.py` — pools findings across a run's instance/seed
subdirs and reports original accuracy, T/C/S pass rates, and construction
dynamics (self-correction vs memory loss).

```bash
python diagnostic/compare_probes.py <run_folder1> <run_folder2> ... --csv out.csv
# or auto-scan a root of runs:
python diagnostic/compare_probes.py --scan agents
```

### 4c. Plot ablation figures

`diagnostic/plot_probe_ablation.py` — one x-axis category per named pipeline
variant (e.g. Full vs. dropping name-extraction vs. dropping global tracking).
Run with an env that has matplotlib (`mem`):

```bash
~/miniconda3/envs/mem/bin/python diagnostic/plot_probe_ablation.py \
  --folders '{"Full": "agents/..._Clean_Anoy_..._2048", "AnonGlobal": "agents/..._anon_global_..._2048"}'
# -> diagnostic/figures_ablation/{fig_probe_bars,fig_cascade,fig_memory_dynamics,fig_confusion_counts}
```

(Sibling `diagnostic/plot_probe_figures.py` groups by SNR family × compression
ratio instead of named variants.)

---

## Quick end-to-end example (clean, Season01)

```bash
cd /storage/home/tuochao/Mem-alpha-audio
export PYTHONPATH=/storage/home/tuochao/Mem-alpha-audio

# 0. Audio prep — only for a new show / noise condition; TBBT clean is already done
#    bash audio_script/run_mix_wham_noise.sh
#    python prepare_data/make_bazinga_timeline.py --data_dir <raw show dir>

# 1. Audio pipeline (or reuse the pre-built outputs/step3_anony/S01_S03_Clean_Anoy)
bash audio_script/run_demo_pipeline_bazinga.sh

# 2. Parquet (predicted + gold names)
python -m prepare_data.prepare_parquet_from_step3 \
  --data_dir outputs/step3_anony/S01_S03_Clean_Anoy \
  --season_filter Season01 --suffix Clean_Anoy
python -m prepare_data.prepare_parquet_from_step3 \
  --data_dir outputs/step3_anony/S01_S03_Clean_Anoy \
  --season_filter Season01 --suffix Clean_Anoy --use_gt_name

# 3. Memory + QA (SLURM sweep is easiest)
bash submit_sweep.sh sweep_configs/TBBT_clean_prediction.conf

# 4. Probe + plot (after the job finishes)
bash diagnostic/run_probe_errors.sh \
  agents/qwen3.6-27b_..._dataset_pred_name_Season01_Clean_Anoy_no_thinking_tokens_2048 \
  outputs/step3_anony/S01_S03_Clean_Anoy \
  outputs/step3_anony/qas/merged_qa_anoy.jsonl \
  outputs/step3_anony/S01_S03_Clean_Anoy/dataset_gt_name_Season01_Clean_Anoy.parquet
python diagnostic/compare_probes.py agents/qwen3.6-27b_..._Clean_Anoy_..._2048 --csv clean.csv
```

---

## Troubleshooting

- **`Cannot locate conda`** — the scripts search `~/miniconda3`, `~/anaconda3`,
  then `$CONDA_EXE`. Set `CONDA_EXE` if conda lives elsewhere.
- **Step3 hangs / connection refused** — the Qwen server at `QWEN_URL`
  (`localhost:8002/v1`) is not up.
- **Step3 `404 The model 'Qwen/Qwen3-32B' does not exist`** — `QWEN_MODEL_NAME`
  must match the server's `--served-model-name`, not the HF repo id. Both
  `launch_vllm.sh` and `launch_vllm_qwen36.sh` serve as `qwen3-32b`. Confirm with
  `curl -s http://localhost:8002/v1/models | python3 -m json.tool`.
- **Step1 predicts 1 speaker per chunk** — `REPETITION_PENALTY` > 1.0. See the
  warning in Stage 1; reset it to `1.0` **and delete the affected `step1/` tree**,
  since Step1 skips chunks that already have outputs.
- **Step1 stops producing output but the job is still alive** — a chunk blocked
  inside `generate()`. There is no error and no traceback because Python never
  exits, so `set -e` and the `ERR` trap never fire. It is not a long generation:
  `max_new_tokens` caps a runaway at ~7 min at observed throughput. Kill and
  re-run; resume is chunk-level.
- **An env override appears to do nothing** — check it is written as
  `${VAR:-default}` in the script. The VibeVoice knobs in
  `run_demo_pipeline_bazinga.sh` were plain assignments until they were fixed;
  the equivalents in `run_demo_pipeline_mosaic.sh`,
  `run_demo_pipeline_perltqa.sh` and `run_demo_step1_vibevoice.sh` still are, and
  those scripts never pass `--repetition_penalty` at all, so they inherit the
  CLI default of **1.2** from `backends/__init__.py:115`.
- **Probe errors: "no instance dir with results.json"** — you pointed `BASE_DIR`
  at the `.../0` subdir; pass the parent run dir instead.
- **Seed sweep produces identical results** — `MEM_TEMPERATURE` must be > 0;
  `submit_sweep.sh` refuses a seed sweep otherwise.
- **Co-located SLURM jobs collide** — `submit_pipeline.slurm` auto-picks free
  ports and per-job PID files; don't hardcode 8002/5005 when running multiple.
