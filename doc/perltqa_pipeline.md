# PerLTQA Audio Memory Pipeline

End-to-end guide for running the **PerLTQA** dataset through the audio memory
agent. PerLTQA is *synthesized* two-party dialogue audio (built by TTS from the
PerLTQA long-term-memory QA corpus), so the flow mirrors the Bazinga pipeline
but with a few PerLTQA-specific differences (see [Differences vs Bazinga](#differences-vs-bazinga)).

- **Dataset:** PerLTQA — per-profile multi-session dialogues synthesized to audio.
- **Raw data path (name-replaced TTS):**
  `/checkpoint/seamless/tuochao/data/PerLTQA/dialogue_tts_en_name_replaced/`
  (holds `<Profile>/<dialogue_id>/` dirs; TTS output of
  `ctbox_tts/perltqa_dialogue_tts.py`).
- **Repo root:** `/storage/home/tuochao/Mem-alpha-audio`

The pipeline has four stages:

| Stage | What it does | Entry point |
|-------|--------------|-------------|
| 1. Audio pipeline | Step0 GT annotations → diar+ASR → speaker matching → name extraction | `audio_script/run_demo_pipeline_perltqa.sh` |
| 2. Parquet build | pack per-bundle dialogue chunks into Parquet | `prepare_data/prepare_audio_parquet_perltqa.py` |
| 3. Memory + QA | build memory, then answer/score QA | `run_pipeline.sh` / `submit_sweep.sh` |
| 4. Error probing | attribute QA errors to a pipeline stage + plot | `diagnostic/run_probe_errors_perltqa.sh` |

> See also [`doc/bazinga_pipeline.md`](bazinga_pipeline.md) — Stages 3 and 4 share
> the same scripts and knobs; this doc focuses on the PerLTQA specifics.

---

## Differences vs Bazinga

1. **Step0 annotations first.** PerLTQA has no reference transcripts on disk, so
   `run_demo_pipeline_perltqa.sh` first builds per-speaker ground truth
   (`*_annotation.json`, turn text + Silero VAD) with
   `ctbox_tts/generate_annotations.py`. Step1 **requires** these.
2. **No "seasons" — "bundles" instead.** PerLTQA groups profiles into *bundles*.
   Each bundle is treated like a Bazinga season: it gets its own speaker pool,
   name state, and Parquet. Step1 runs once; Step2/Step3 run either per-bundle or
   as one global pool.
3. **PerLTQA loaders.** Step1 uses `audio_script.Multi_ASR.step1_perltqa`; the
   Parquet builder is `prepare_audio_parquet_perltqa.py`.
4. **Free-form QA grading.** QA answers are not multiple-choice; the probe grades
   with `keyword` (default, no API) or `llm_judge` (via the Qwen server).
5. **`dataset=perltqa`** everywhere downstream (memory construction, QA, probe
   `--data_source perltqa`).

---

## Prerequisites

**Conda environments** (activated by name inside the scripts):

| Env | Used for |
|-----|----------|
| `ctbox` | Step0 GT annotations (`generate_annotations.py`) |
| `vibevoice` | Step1 ASR — VibeVoice backend (default `METHOD`) |
| `nemo` | Step1 ASR — NeMo backends (`nemo-streaming` / `nemo-offline`) |
| `mem` | Step2/Step3, QA evaluation, plotting |
| `vllm` | Memory construction (step 1 of `run_pipeline.sh`), reward-model server |

**A Qwen server** reachable at `QWEN_URL` (default `http://localhost:8002/v1`,
served name `qwen3-32b`) for Step3 name extraction and (optionally) `llm_judge`
grading. Memory construction / QA needs a vLLM server (auto-launched on SLURM).

**Models** (baked into the script, override via env): same as Bazinga —
`diar_streaming_sortformer_4spk-v2.1`, `multitalker-parakeet-streaming-0.6b-v1`
or `microsoft/VibeVoice-ASR`, and `wespeaker-voxceleb-resnet293-LM`.

Run from the repo root with `PYTHONPATH=/storage/home/tuochao/Mem-alpha-audio`.

---

## Stage 0 (prep) — session timeline

The Parquet builder needs a session-timeline JSON that maps each `conv_id`
(= `<Profile>_<dialogue_folder>`, e.g. `Cao_Lili_25_0_0_0`) to a date. Build it
once:

```bash
python prepare_data/make_perltqa_timeline.py
# reads /checkpoint/seamless/tuochao/data/PerLTQA/Dataset/en_v2/perltmem_en_v2.json
# -> outputs/perltqa_data/perltqa_session_timeline.json
```

---

## Stage 1 — Audio pipeline (annotations → diar+ASR → speaker tracking → names)

Script: `audio_script/run_demo_pipeline_perltqa.sh`. Sub-steps:

0. **Step0** (`ctbox_tts/generate_annotations.py`, `ctbox` env) — build
   `*_annotation.json` GT per dialogue. Skip with `RUN_ANNOTATE=0` if they exist.
1. **Step1** (`Multi_ASR/step1_perltqa.py`) — diar + ASR. Only transcribes
   profiles referenced by the **bundle manifests** (skips the ~110 QA-less
   profiles); it takes the union across both manifests so one Step1 run covers
   every bundle mode you might evaluate.
2. **Step2** (`Speaker_Track/step2_speaker_match_v2.py`) — speaker→pool matching.
3. **Step3** (`Speaker_Track/step3_speaker_name_extract.py`) — real names via Qwen.

**Bundle vs global mode** (controls how Step2/Step3 group + pool):

- `BUNDLE_MANIFEST` **unset** → one **global** pool over everything (default).
- `BUNDLE_MANIFEST=<manifest>` → one **independent** pool + name state *per
  bundle* (bundle == "season", profiles == "episodes"). Two manifests ship in the
  raw data folder:
  - `bundles_per_profile.json` → per-profile pools (30)
  - `bundles_multi.json` → multi-profile pools (3)

### Run it

```bash
cd /storage/home/tuochao/Mem-alpha-audio

# Default: build annotations + Step1 + global-pool Step2/Step3, VibeVoice.
bash audio_script/run_demo_pipeline_perltqa.sh

# Step1 once (annotations + ASR), skip pooling — reuse for both bundle modes:
RUN_STEP2=0 RUN_STEP3=0 bash audio_script/run_demo_pipeline_perltqa.sh

# Then per-profile pools (reuse Step1):
BUNDLE_MANIFEST=/checkpoint/seamless/tuochao/data/PerLTQA/dialogue_tts_en_name_replaced/bundles_per_profile.json \
  RUN_STEP1=0 bash audio_script/run_demo_pipeline_perltqa.sh

# Or multi-profile bundles (reuse Step1):
BUNDLE_MANIFEST=/checkpoint/seamless/tuochao/data/PerLTQA/dialogue_tts_en_name_replaced/bundles_multi.json \
  RUN_STEP1=0 bash audio_script/run_demo_pipeline_perltqa.sh
```

Key knobs: `METHOD`, `RAW_DATA_PATH`, `RUN_ANNOTATE/STEP1/STEP2/STEP3`,
`ANNOTATE_OVERWRITE`, `RESET_STATE`, `BUNDLE_MANIFEST`, `PP_MANIFEST`,
`MULTI_MANIFEST`, `SIMILARITY_THRESHOLD`, `QWEN_URL`, `QWEN_MODEL_NAME`.

### Outputs

```
Audio_Results/<METHOD>/<DATASET_TAG>/
  step1/                                  # diar + ASR
  step2/                                  # GLOBAL mode: single pool
    pool.npz  speakers_name_pool.json
    <conv_id>/CHUNK_N/parsed_dialog_{pred,gt}.json
  step2/bundle_<id>/                      # BUNDLE mode: one pool per bundle
    pool.npz  state.json
    <conv_id>/CHUNK_N/parsed_dialog_{pred,gt}.json
  logs/pipeline_perltqa_<timestamp>.log
```

`DATASET_TAG` = basename of `RAW_DATA_PATH`.

> Pre-built step3 trees already exist under `outputs/step3_perltqa_replaced_name/`
> (`bundle_0/`, `bundle_1/`, `bundle_2/`) and noisy variants
> `outputs/step3_perltqa_replaced_name_interf_SNR{0,5}/`.

---

## Stage 2 — Build the per-bundle Parquet dataset

Script: `prepare_data/prepare_audio_parquet_perltqa.py`. Unlike the Bazinga
builder, it iterates over **bundle folders** and writes one Parquet per bundle
(pred and/or gt names). Pred mode resolves `GLOBAL_SPK_N` → real name via that
bundle's `extracted_speaker_name.json`.

```bash
PYTHONPATH=. python prepare_data/prepare_audio_parquet_perltqa.py \
  --data_dir Audio_Results/vibevoice/dialogue_tts_en_name_replaced/step2 \
  --output_root outputs/step3_perltqa_replaced_name \
  --time_info_path outputs/perltqa_data/perltqa_session_timeline.json \
  --mode both
```

- `--mode` — `pred` | `gt` | `both` (default `both`).
- `--bundles` — restrict to named bundle folders (default: all found).
- `--data_source` — label written into the Parquet (default `perltqa`).
- `--suffix` — extra filename tag (e.g. an interference/SNR tag).

Output per bundle:

```
outputs/step3_perltqa_replaced_name/<bundle>/
  dataset_pred_name_<bundle>[_<suffix>].parquet
  dataset_gt_name_<bundle>[_<suffix>].parquet
  <conv_id>/CHUNK_N/parsed_dialog_{pred,gt}.json     # mirrored chunks
```

e.g. `outputs/step3_perltqa_replaced_name/bundle_0/dataset_pred_name_bundle_0_perltqa.parquet`.

---

## Stage 3 — Memory construction + QA evaluation

Identical machinery to Bazinga (`run_pipeline.sh` / `submit_sweep.sh`), just pass
`perltqa` as the dataset and point at a PerLTQA Parquet + QA dir. Run **one job
per bundle**.

### Option A — local single run

```bash
bash run_pipeline.sh <parquet_path> perltqa <custom_qa_dir>
```

```bash
bash run_pipeline.sh \
  outputs/step3_perltqa_replaced_name/bundle_0/dataset_pred_name_bundle_0_perltqa.parquet \
  perltqa \
  outputs/perltqa_data/qa_multi_name_replaced/bundle_0_filterd/
```

Step 1 (memory construction, `vllm` env) → Step 2 (QA evaluation, `mem` env).
Same optional env knobs as Bazinga: `COMPRESSION_STRATEGY` (`x1.5`/`x2`/`x3`/`x5`,
keys from `config/prompts_wrt_datasource_compression.yaml`), and `SEED` /
`MEM_TEMPERATURE` (> 0) / `ROLLOUT_LABEL` for seed-variance runs.

### Option B — SLURM sweep

`submit_sweep.sh` works unchanged — set `DATASET="perltqa"` in the config and
point `PARQUET_PATH` / `CUSTOM_QA_DIR` at one bundle. No PerLTQA sweep config
ships yet; copy `sweep_configs/example.conf`:

```bash
# sweep_configs/perltqa_bundle0.conf
PARQUET_PATH="outputs/step3_perltqa_replaced_name/bundle_0/dataset_pred_name_bundle_0_perltqa.parquet"
DATASET="perltqa"
CUSTOM_QA_DIR="outputs/perltqa_data/qa_multi_name_replaced/bundle_0_filterd/"
COMPRESSION_RATIOS=("none")     # or ("x3" "x4")
# SEEDS=(1 2 10); MEM_TEMPERATURE="0.3"   # seed sweep needs temperature > 0
```

```bash
bash submit_sweep.sh sweep_configs/perltqa_clean_bundle0.conf
squeue --me
```

To submit the clean/SNR5/SNR0 × compression grid for bundle 0, use the
multi-input config:

```bash
bash submit_sweep.sh sweep_configs/perltqa_snr_compression.conf
```

Set `CLEAN_RATIOS`, `SNR5_RATIOS`, and `SNR0_RATIOS` independently in the
config. The shipped grid uses
`none x2 x3 x4 x5` for every condition and one greedy run per cell (15 jobs).
Preview the exact 15 submissions without launching them with
`DRY_RUN=1 bash submit_sweep.sh sweep_configs/perltqa_snr_compression.conf`.

Run dirs land under `agents/`, e.g.
`qwen3.6-27b_..._perltqa_dataset_pred_name_bundle_0_no_thinking_tokens_2048`.

---

## Stage 4 — Error probing + visualization

### 4a. Run the behavioral error probe (PerLTQA wrapper)

Script: `diagnostic/run_probe_errors_perltqa.sh`. Same G/C/T/S probe semantics as
the Bazinga wrapper, but wired for PerLTQA: `--data_source perltqa` normalizes QA
`gt_source` refs (`Cao_Lili/25_0_0_0`) to on-disk chunk folders
(`Cao_Lili_25_0_0_0/CHUNK_0/...`), and grading uses a free-form scorer.

```bash
bash diagnostic/run_probe_errors_perltqa.sh [BASE_DIR] [DATA_ROOT] [QA_FILE] [PARQUET] [SERVER_URL]
```

- `BASE_DIR` — the **run dir** (holds `0/` and optional `seed*/0/`), not `.../0`.
- `DATA_ROOT` — bundle dialogue root (holds `parsed_dialog_{gt,pred}.json`), e.g.
  `outputs/step3_perltqa_replaced_name/bundle_0`.
- `QA_FILE` — e.g. `outputs/perltqa_data/qa_multi_name_replaced/bundle_0_filterd/qa.jsonl`.
- `PARQUET` — the bundle Parquet (its `chunk_folders` localizes evidence).
- `SERVER_URL` — default `http://127.0.0.1:5005/batch_process`.

```bash
bash diagnostic/run_probe_errors_perltqa.sh \
  agents/qwen3.6-27b_..._perltqa_dataset_pred_name_bundle_0_no_thinking_tokens_2048 \
  outputs/step3_perltqa_replaced_name/bundle_0 \
  outputs/perltqa_data/qa_multi_name_replaced/bundle_0_filterd/qa.jsonl \
  outputs/step3_perltqa_replaced_name/bundle_0/dataset_gt_name_bundle_0_perltqa.parquet
```

PerLTQA-specific env:
- `SCORER` — `llm_judge` (default; needs `QWEN_URL` / `QWEN_MODEL_NAME` /
  `OPENROUTER_API_KEY`, all defaulted to the local vLLM backend) or `keyword`
  (`;`-split containment, no API).
- `RUN_GOLDEN=1` — also run the gold-dialogue ceiling (G) probe.
- `BATCH_SIZE` (default 64), `PROBE_TIMEOUT` (default 1200) — tune the T/G
  precompute POST size / read timeout; smaller batches checkpoint the cache more
  often.

T/G probes are precomputed once per `DATA_ROOT` into
`<DATA_ROOT>/tg_probe_cache.json` and reused across base_dirs/seeds; C and S run
per instance. Outputs `error_probe.json` + `error_probe_debug.json` per instance
(skipped if both exist); multiple seeds aggregate into
`error_probe_seed_summary.json`.

For every completed experiment listed in a multi-input PerLTQA sweep config,
submit one Slurm job per agent directory (one job for each SNR/compression
combination). Concurrent jobs safely coordinate access to each SNR's shared T/G
cache with a file lock:

```bash
# Preview the discovered run directories and submissions.
DRY_RUN=1 bash diagnostic/submit_probe_sweep_perltqa.sh \
  sweep_configs/perltqa_snr_compression.conf

# Submit the jobs. The default is LLM-judge grading and probing.
bash diagnostic/submit_probe_sweep_perltqa.sh \
  sweep_configs/perltqa_snr_compression.conf
```

Each job starts its own local judge vLLM + memory server, writes
`evaluation_metrics.json`, `error_probe.json`, and `error_probe_debug.json` into
each run's `0/` directory, and logs to `logs/perltqa-probe-<jobid>.out`.
The noisy condition's `DATA_ROOT` is retained for its predicted transcripts and
T/G cache, while all SNR conditions reuse the identical clean GT Parquet under
`outputs/step3_perltqa_replaced_name/<bundle>/`. Override that root with
`PERLTQA_GT_ROOT` if needed.

### 4b / 4c. Compare + plot

Same tools as Bazinga (dataset-agnostic — they read the `error_probe.json` files):

```bash
python diagnostic/compare_probes.py <run_folder1> <run_folder2> ... --csv perltqa.csv
~/miniconda3/envs/mem/bin/python diagnostic/plot_probe_ablation.py \
  --folders '{"bundle_0": "agents/..._perltqa_..._bundle_0_..._2048"}'
# figures -> diagnostic/figures_ablation/
```

---

## Quick end-to-end example (bundle_0, global pool)

```bash
cd /storage/home/tuochao/Mem-alpha-audio
export PYTHONPATH=/storage/home/tuochao/Mem-alpha-audio

# 0. Timeline (once)
python prepare_data/make_perltqa_timeline.py

# 1. Audio pipeline (annotations + Step1 + bundle-mode Step2/Step3)
RUN_STEP2=0 RUN_STEP3=0 bash audio_script/run_demo_pipeline_perltqa.sh
BUNDLE_MANIFEST=/checkpoint/seamless/tuochao/data/PerLTQA/dialogue_tts_en_name_replaced/bundles_multi.json \
  RUN_STEP1=0 bash audio_script/run_demo_pipeline_perltqa.sh
# (or just reuse the pre-built outputs/step3_perltqa_replaced_name/bundle_0)

# 2. Per-bundle Parquet (pred + gt)
PYTHONPATH=. python prepare_data/prepare_audio_parquet_perltqa.py \
  --data_dir Audio_Results/vibevoice/dialogue_tts_en_name_replaced/step2 \
  --output_root outputs/step3_perltqa_replaced_name \
  --time_info_path outputs/perltqa_data/perltqa_session_timeline.json --mode both

# 3. Memory + QA
bash run_pipeline.sh \
  outputs/step3_perltqa_replaced_name/bundle_0/dataset_pred_name_bundle_0_perltqa.parquet \
  perltqa \
  outputs/perltqa_data/qa_multi_name_replaced/bundle_0_filterd/

# 4. Probe + compare
bash diagnostic/run_probe_errors_perltqa.sh \
  agents/qwen3.6-27b_..._perltqa_dataset_pred_name_bundle_0_no_thinking_tokens_2048 \
  outputs/step3_perltqa_replaced_name/bundle_0 \
  outputs/perltqa_data/qa_multi_name_replaced/bundle_0_filterd/qa.jsonl \
  outputs/step3_perltqa_replaced_name/bundle_0/dataset_gt_name_bundle_0_perltqa.parquet
python diagnostic/compare_probes.py agents/qwen3.6-27b_..._bundle_0_..._2048 --csv bundle0.csv
```

---

## Troubleshooting

- **Step1 says "no manifest profiles found; running on ALL dialogues"** — the
  `PP_MANIFEST` / `MULTI_MANIFEST` paths don't exist under `RAW_DATA_PATH`. Point
  them at the real `bundles_*.json` or it transcribes all ~141 profiles.
- **Step1 fails immediately** — missing `*_annotation.json`; run Step0 first
  (`RUN_ANNOTATE=1`, the default) or `ANNOTATE_OVERWRITE=1` to rebuild.
- **`no timeline entry for '<conv_id>'` (Parquet build)** — the timeline JSON is
  stale/missing; rebuild with `make_perltqa_timeline.py`, and confirm each
  `source_file` (minus `.json`) equals the chunk's `conv_id`.
- **`llm_judge` grading 401 "EMPTY key"** — `QWEN_URL` was unset so the client
  fell back to `api.openai.com`. The wrapper defaults it to the local vLLM
  backend; export your own to override, or use `SCORER=keyword`.
- **Probe: "no instance dir with results.json"** — you pointed `BASE_DIR` at the
  `.../0` subdir; pass the parent run dir.
- **Seed sweep produces identical results** — `MEM_TEMPERATURE` must be > 0.
- **Same conda / server caveats** as `doc/bazinga_pipeline.md`.
