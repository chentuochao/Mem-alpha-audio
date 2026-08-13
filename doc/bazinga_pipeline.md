# Bazinga (The Big Bang Theory) Audio Memory Pipeline

End-to-end guide for running the **Bazinga** dataset through the audio memory
agent: from raw episode audio → diarized/ASR'd dialogue → Parquet → memory
construction + QA evaluation → error probing / visualization.

- **Dataset:** The Big Bang Theory (TBBT), organized by season/episode.
- **Raw data path:** `/checkpoint/seamless/tuochao/data/bazinga/data/TheBigBangTheory`
  - Noisy variants live in sibling `*_SNRx` / `*_interf_SNRx` folders.
- **Repo root:** `/storage/home/tuochao/Mem-alpha-audio`

The pipeline has four stages:

| Stage | What it does | Entry point |
|-------|--------------|-------------|
| 1. Audio pipeline | diarization + ASR → speaker matching → speaker-name extraction | `audio_script/run_demo_pipeline_bazinga.sh` |
| 2. Parquet build | pack dialogue chunks into a Parquet dataset | `prepare_data/prepare_parquet_from_step3*.py` |
| 3. Memory + QA | build memory, then answer/score QA | `run_pipeline.sh` / `submit_sweep.sh` |
| 4. Error probing | attribute QA errors to a pipeline stage + plot | `diagnostic/run_probe_errors.sh` |

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
- Step3 name extraction expects `QWEN_URL=http://localhost:8002/v1`.
- Memory construction / QA expects a vLLM server; on SLURM this is launched
  automatically by `submit_pipeline.slurm` (see Stage 3).

**Models** (paths baked into `run_demo_pipeline_bazinga.sh`, override via env):
- Diarization: `diar_streaming_sortformer_4spk-v2.1` (NeMo backend only)
- ASR: `multitalker-parakeet-streaming-0.6b-v1` (NeMo) or `microsoft/VibeVoice-ASR` (VibeVoice)
- Speaker embedding: `wespeaker-voxceleb-resnet293-LM`

Everything below assumes you run from the repo root and have
`PYTHONPATH=/storage/home/tuochao/Mem-alpha-audio`.

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
```

Key knobs (edit at the top of the script or pass as env):
`METHOD`, `RAW_DATA_PATH`, `SEASONS`, `RUN_STEP1/2/3`, `RESET_STATE`,
`SIMILARITY_THRESHOLD`, `QWEN_URL`.

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
`--time_info_path`). Each chunk folder becomes `chunk_folders[i]` in the Parquet,
which is what the error probe later uses to map QA evidence → memory ids.

### 2a. From a fresh Step1/Step2/Step3 audio run

`prepare_data/prepare_audio_parquet.py` — reads
`{show}/{episode}/parsed_dialog_pred.json` + `speaker_name_map.json` and bakes
names in.

```bash
python -m prepare_data.prepare_audio_parquet \
  --data_dir Audio_Results/vibevoice/TheBigBangTheory/step2 \
  --output_root outputs/bazinga_data \
  --season_filter Season01
# -> outputs/bazinga_data/dataset_pred_name_Season01.parquet
# add --use_gt_name for the gold-name parquet
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
`TBBT_interf_SNR5/SNR10.conf`.

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

Probe stages: **T** (transcript) and **G** (gold, optional) are precomputed once
per `DATA_ROOT` into `<DATA_ROOT>/tg_probe_cache.json` and reused across every
base_dir / seed / compression variant; **C** (constructed memory) and **S**
(retrieval) run per instance. Writes `error_probe.json` +
`error_probe_debug.json` in each instance dir (skipped if both exist). Multiple
seeds are aggregated into `error_probe_seed_summary.json`.

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
- **Probe errors: "no instance dir with results.json"** — you pointed `BASE_DIR`
  at the `.../0` subdir; pass the parent run dir instead.
- **Seed sweep produces identical results** — `MEM_TEMPERATURE` must be > 0;
  `submit_sweep.sh` refuses a seed sweep otherwise.
- **Co-located SLURM jobs collide** — `submit_pipeline.slurm` auto-picks free
  ports and per-job PID files; don't hardcode 8002/5005 when running multiple.
