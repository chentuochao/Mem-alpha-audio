# Mix_Mosaic Audio Memory Pipeline

End-to-end guide for running the **Mix_Mosaic** dataset (mixed Seamless-Interaction
dyads) through the audio memory agent. The flow mirrors the Bazinga / PerLTQA
pipelines; this doc covers the Mosaic-specific pieces (see
[Differences vs PerLTQA](#differences-vs-perltqa)).

- **Dataset:** Mix_Mosaic — two-party naturalistic conversations, each speaker
  recorded on a separate close-mic channel and summed to one mono mix by
  `audio_script/datasets/mix_interact.py`.
- **Raw data path:** `/checkpoint/seamless/tuochao/data/Mix_Mosaic/naturalistic/test/`
  — 65 speaker-pair folders, 816 conversations, 118 speakers, ~524k transcript tokens.
- **Noisy variants (already built):** `test_interf_SNR{0,5,10}/` (competing speech).
- **Repo root:** `/storage/home/tuochao/Mem-alpha-audio`

Raw layout — one folder per speaker *pair*, one sub-folder per conversation:

```
<data_dir>/
  Pxxx_Pyyy/                      # Pxxx / Pyyy are stable, globally unique speaker ids
    V00_Sxxxx_Ixxxxxxxx/
      mixed_conv.wav              # mono mix (the only backend input)
      transcript1.json            # speaker Pxxx: turn segments + word feats
      transcript2.json            # speaker Pyyy
      vad1.json  vad2.json        # per-speaker VAD (unused — GT VAD is rebuilt from turns)
  bundles.json                    # bundle manifest (see Stage 0)
```

The pipeline has five stages:

| Stage | What it does | Entry point |
|-------|--------------|-------------|
| 0. Bundles (+ noise) | group pairs into bundles; optionally mix noise/interference | `audio_script/make_mix_mosaic_bundles.py`, `run_mix_wham_{noise,inteference}_mosaic.sh` |
| 1. Audio pipeline | diar+ASR → speaker matching → name extraction | `audio_script/run_demo_pipeline_mosaic.sh` (+ `run_step1_mosaic_shards.sh`) |
| 2. Parquet build | pack per-bundle chunks into Parquet | `prepare_data/prepare_audio_parquet_perltqa.py` |
| 3. Memory + QA | build memory, then answer/score QA | `run_pipeline.sh` / `submit_sweep.sh` |
| 4. Error probing | attribute QA errors to a pipeline stage + plot | `diagnostic/run_probe_errors.sh` |

> See also [`doc/perltqa_pipeline.md`](perltqa_pipeline.md) and
> [`doc/bazinga_pipeline.md`](bazinga_pipeline.md) — Stages 3 and 4 share the same
> machinery.

---

## Differences vs PerLTQA

1. **No Step0 annotations.** Mix_Mosaic ships reference transcripts on disk
   (`transcript1/2.json`), so Step1 builds GT directly — there is no TTS
   annotation stage.
2. **Bundles group by *shared speaker*, not by profile.** `make_mix_mosaic_bundles.py`
   merges pair-folders that share a speaker into connected components and packs
   components into token-balanced bundles, so every conversation involving a given
   speaker lands in exactly one bundle (asserted at build time). A bundle plays the
   role of a Bazinga "season"; its pair-folder names are the Step2 `--season_filter`.
3. **`conv_id` = `<pair>_<clip>`** (e.g. `P0092_P0093_V00_S0062_I00000125`). The pair
   prefix is what makes `--season_filter` bundle selection work.
4. **Multiple-choice QA** — Mosaic reuses the Seamless QA format, so downstream
   `--dataset` / `--data_source` is **`seamlessinteraction_options`**, *not* `mosaic`
   (there is no `mosaic` entry in `run_memory_construction*.py`'s choices nor in
   `config/prompts_wrt_datasource_compression.yaml`; a mismatch raises `KeyError`).
5. **Sharded Step1.** 816 conversations × ~80 s each is too slow single-process, so
   `run_step1_mosaic_shards.sh` runs one shard per GPU (see Stage 1).
6. **Parquet builder is reused as-is** — `prepare_audio_parquet_perltqa.py` is
   layout-compatible (`{bundle}/{conv_id}/{CHUNK_N}/parsed_dialog_*.json`); only
   `--data_source` / `--suffix` / `--time_info_path` change.

---

## Prerequisites

**Conda environments** (activated by name inside the scripts):

| Env | Used for |
|-----|----------|
| `vibevoice` | Step1 ASR — VibeVoice backend (default `METHOD`) |
| `nemo` | Step1 ASR — NeMo backends; also the noise/interference mixers |
| `mem` | Step2/Step3, QA evaluation, probing, plotting |
| `vllm` | Memory construction (step 1 of `run_pipeline.sh`) |

**A Qwen server** at `QWEN_URL` (default `http://localhost:8002/v1`, served name
`qwen3-32b`) for Step3 name extraction. Memory construction / QA needs a vLLM
server (auto-launched on SLURM by `submit_sweep.sh`).

**Models** (baked into the scripts, override via env): `microsoft/VibeVoice-ASR`
or `diar_streaming_sortformer_4spk-v2.1` + `multitalker-parakeet-streaming-0.6b-v1`,
and `wespeaker-voxceleb-resnet293-LM`.

Run from the repo root with `PYTHONPATH=/storage/home/tuochao/Mem-alpha-audio`.

---

## Stage 0a — Bundles

`bundles.json` already exists in the raw folder (4 bundles, ~131k tokens each).
Rebuild only if the data changes:

```bash
python audio_script/make_mix_mosaic_bundles.py \
  --data-dir /checkpoint/seamless/tuochao/data/Mix_Mosaic/naturalistic/test \
  --num-bundles 4                      # or --target-tokens 130000
# -> <data-dir>/bundles.json
```

Each bundle records `folders` (pair-folder names — the Step2 `--season_filter`),
`conversations`, `speakers`, and `total_tokens`.

## Stage 0b — Noisy variants (optional)

Two mixers clone the raw tree into sibling `_SNRx` folders. Only `mixed_conv.wav`
is mixed; `transcript*.json` / `vad*.json` are copied verbatim (timestamps
unchanged, so GT stays aligned), and only conversations listed in `bundles.json`
are processed.

```bash
# WHAM ambient noise -> test_SNR0/, test_SNR5/
bash audio_script/run_mix_wham_noise_mosaic.sh

# Competing speech (LibriTTS babble, 1-4 tracks) -> test_interf_SNR{0,5,10}/
bash audio_script/run_mix_wham_inteference_mosaic.sh
```

Edit `SNRS`, `SEED`, `NUM_INTERF_{MIN,MAX}`, `GAP_MAX` at the top of each script.
The same noise/interference realization is reused across SNR levels for a given
conversation, so the only difference between `_SNR0` and `_SNR5` is the gain — a
clean A/B. Feed a variant back in with `RAW_DATA_PATH=<...>_interf_SNR5`.

## Stage 0c — Session timeline

The Parquet builder needs a timeline JSON mapping each `conv_id` to a date:

```json
{"sessions": [{"source_file": "P0043_P0108_V00_S0557_I00000135.json",
               "session_timeline_date": "2023-05-01",
               "pair_id": "P0043_P0108", "clip_id": "V00_S0557_I00000135"}, ...]}
```

The shipped file is `outputs/mosaic_step3/mosaic_session_timeline.json` (weekly
dates assigned in history order per pair). **There is no generator script in the
repo** — reuse this file, or produce one with the same schema (`source_file` minus
`.json` must equal the chunk's `conv_id`).

---

## Stage 1 — Audio pipeline (diar+ASR → speaker tracking → names)

Script: `audio_script/run_demo_pipeline_mosaic.sh`. Sub-steps:

1. **Step1** (`Multi_ASR/step1_mosaic.py`) — diar + ASR over every conversation,
   once, with no cross-conversation state.
2. **Step2** (`Speaker_Track/step2_speaker_match_v2.py`) — speaker→pool matching,
   **per bundle** (each bundle = an independent pool, selected by `--season_filter`
   on the bundle's pair folders).
3. **Step3** (`Speaker_Track/step3_speaker_name_extract.py`) — real names via Qwen.

`BUNDLE_MANIFEST` (default `<RAW_DATA_PATH>/bundles.json`) controls Step2/Step3
grouping; set it to `""` for a single global pool over everything.

### Run it (single process)

```bash
cd /storage/home/tuochao/Mem-alpha-audio
bash audio_script/run_demo_pipeline_mosaic.sh                       # all three steps
RAW_DATA_PATH=/checkpoint/.../test_interf_SNR5 \
  bash audio_script/run_demo_pipeline_mosaic.sh                     # noisy variant
BUNDLE_MANIFEST="" bash audio_script/run_demo_pipeline_mosaic.sh    # global pool
```

Key knobs: `METHOD` (`vibevoice` | `nemo-streaming` | `nemo-offline`),
`RAW_DATA_PATH`, `RUN_STEP1/2/3`, `RESET_STATE`, `BUNDLE_MANIFEST`,
`SIMILARITY_THRESHOLD`, `QWEN_URL`, `QWEN_MODEL_NAME`.

### Run it (sharded Step1 — recommended)

`run_step1_mosaic_shards.sh` splits the conversation list round-robin into
`NUM_SHARDS` shards and runs one process per GPU, all writing into the **same**
`output_dir`. Output is keyed by `conv_id`, so shards never collide and the tree is
identical to a single-process run.

```bash
# 4 shards on 4 GPUs (~1-1.5 h for 816 convs on H200s)
setsid nohup env RAW_DATA_PATH=/checkpoint/seamless/tuochao/data/Mix_Mosaic/naturalistic/test_interf_SNR5 \
  bash audio_script/run_step1_mosaic_shards.sh > /tmp/step1_snr5.log 2>&1 < /dev/null &

# then Step2/3 on the finished Step1 tree
RAW_DATA_PATH=/checkpoint/.../test_interf_SNR5 RUN_STEP1=0 \
  bash audio_script/run_demo_pipeline_mosaic.sh
```

Knobs: `NUM_SHARDS`, `GPUS` (default `"0 1 2 3"`), `METHOD`, `MAX_NEW_TOKENS`
(default 8192), `REPETITION_PENALTY` (1.2), `NO_REPEAT_NGRAM_SIZE`.
Per-shard logs: `Audio_Results/logs/step1_mosaic_<tag>_<method>_shard<i>_<stamp>.log`.
Step1 is **resumable** — conversations with existing outputs are skipped, so a
killed run is restarted with the same command.

> **Run one SNR condition at a time.** Each shard loads a full VibeVoice model;
> stacking several conditions on the same 4 GPUs causes >20× slowdowns and risks
> the OOM killer silently reaping shards mid-generation.

### Outputs

```
Audio_Results/<METHOD>/<DATASET_TAG>/
  step1/<conv_id>/CHUNK_N/                # diart_pred.npy transcript_{pred,gt}.json
                                          # vad_gt.json sample_info.json
  step2/bundle_<id>/                      # one pool per bundle
    pool.npz  state.json  speaker_map.json
    raw_speaker_tracking.json  extracted_speaker_name.json
    <conv_id>/CHUNK_N/parsed_dialog_{pred,gt}.json
  logs/pipeline_mosaic_<timestamp>.log
```

`DATASET_TAG` = basename of `RAW_DATA_PATH` (`test`, `test_interf_SNR5`, ...).

---

## Stage 2 — Build the per-bundle Parquet dataset

Reuses the PerLTQA builder. Pred mode resolves `GLOBAL_SPK_N` → real name via the
bundle's `extracted_speaker_name.json`.

```bash
# clean
PYTHONPATH=. python prepare_data/prepare_audio_parquet_perltqa.py \
  --data_dir Audio_Results/vibevoice/test/step2 \
  --output_root outputs/mosaic_step3 \
  --time_info_path outputs/mosaic_step3/mosaic_session_timeline.json \
  --data_source seamlessinteraction_options \
  --mode both --suffix mosaic

# interference SNR5
PYTHONPATH=. python prepare_data/prepare_audio_parquet_perltqa.py \
  --data_dir Audio_Results/vibevoice/test_interf_SNR5/step2 \
  --output_root outputs/mosaic_step3_interf_SNR5 \
  --time_info_path outputs/mosaic_step3/mosaic_session_timeline.json \
  --data_source seamlessinteraction_options \
  --mode both --suffix mosaic_interf_SNR5
```

- `--data_source seamlessinteraction_options` — **must** match the `--dataset`
  passed in Stage 3.
- `--suffix` — filename tag; use it to keep noise conditions apart.
- `--mode` — `pred` | `gt` | `both`; `--bundles` restricts to named bundle folders.
- The timeline is shared across noise conditions (same conversations, same dates).

Output per bundle:

```
outputs/mosaic_step3/bundle_0/
  dataset_pred_name_bundle_0_mosaic.parquet
  dataset_gt_name_bundle_0_mosaic.parquet
  <conv_id>/CHUNK_N/parsed_dialog_{pred,gt}.json     # mirrored chunks
```

---

## Stage 3 — Memory construction + QA evaluation

QA lives in `outputs/mosaic_step3/QA/bundle<N>/bundle_<N>_content_qa.jsonl`
(multiple-choice Content QA with `gt_source.sources[].{session_id,chunk_id,evidence_turns}`).
It was authored externally — there is no in-repo generator. Note the directory drops
the underscore (`QA/bundle0/`) while the Parquet dir keeps it (`bundle_0/`). The QA
set is shared by every noise condition.

### Option A — local single run

```bash
bash run_pipeline.sh \
  outputs/mosaic_step3/bundle_0/dataset_pred_name_bundle_0_mosaic.parquet \
  seamlessinteraction_options \
  outputs/mosaic_step3/QA/bundle0/
```

Step 1 (memory construction, `vllm` env) → Step 2 (QA evaluation, `mem` env).
Env knobs: `COMPRESSION_STRATEGY` (`x1.5`/`x2`/`x3`/`x5`, keys from
`config/prompts_wrt_datasource_compression.yaml`), and `SEED` / `MEM_TEMPERATURE`
(> 0) / `ROLLOUT_LABEL` for seed-variance runs.

### Option B — SLURM sweep

Configs ship for all four bundles:

```bash
bash submit_sweep.sh sweep_configs/mosaic_clean_bundle0.conf   # ...bundle{1,2,3}.conf
squeue --me
```

They sweep `COMPRESSION_RATIOS=("none" "x2" "x3" "x4" "x5")` × `SEEDS=("" 1 2)` at
`MEM_TEMPERATURE=0.3`. For a noisy condition, copy the config and repoint
`PARQUET_PATH` at the `_interf_SNR*` Parquet (keep `CUSTOM_QA_DIR` unchanged).

Run dirs land under `agents/`, e.g.
`qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_bundle_0_mosaic_no_thinking_tokens_2048`
(compression variants get a `_comp_x3` postfix).

---

## Stage 4 — Error probing + visualization

### 4a. Behavioral error probe

Mosaic uses the **generic** wrapper `diagnostic/run_probe_errors.sh` (the
multiple-choice format needs no PerLTQA free-form scorer):

```bash
bash diagnostic/run_probe_errors.sh [BASE_DIR] [DATA_ROOT] [QA_FILE] [PARQUET] [SERVER_URL]
```

```bash
bash diagnostic/run_probe_errors.sh \
  agents/qwen3.6-27b_Qwen_Qwen3.6-27B_seamlessinteraction_options_dataset_pred_name_bundle_0_mosaic_no_thinking_tokens_2048 \
  outputs/mosaic_step3/bundle_0 \
  outputs/mosaic_step3/QA/bundle0/bundle_0_content_qa.jsonl \
  outputs/mosaic_step3/bundle_0/dataset_gt_name_bundle_0_mosaic.parquet
```

- `BASE_DIR` — the **run dir** (holds `0/` and optional `seed*/0/`), not `.../0`.
- `DATA_ROOT` — the bundle folder holding `parsed_dialog_{gt,pred}.json`.
- `PARQUET` — pass the **gt** Parquet; its `chunk_folders` map localizes evidence
  deterministically.
- `SERVER_URL` — memory server, default `http://127.0.0.1:5005/batch_process`
  (must be up).
- `RUN_GOLDEN=1` also runs the gold-dialogue ceiling (G) probe;
  `BATCH_SIZE` (64) / `PROBE_TIMEOUT` (1200) tune the T/G precompute.

T/G probes are precomputed once per `DATA_ROOT` into `<DATA_ROOT>/tg_probe_cache.json`
and reused across run dirs and seeds; C and S run per instance. Each instance gets
`error_probe.json` + `error_probe_debug.json` (skipped if both exist); multiple
seeds aggregate into `error_probe_seed_summary.json`.

### 4b. Compare + plot

```bash
python diagnostic/compare_probes.py <run_dir1> <run_dir2> ... --csv mosaic.csv
~/miniconda3/envs/mem/bin/python diagnostic/plot_probe_ablation.py \
  --folders '{"clean": "agents/..._bundle_0_mosaic_..._2048",
              "interf_SNR5": "agents/..._bundle_0_mosaic_interf_SNR5_..._2048"}'
# figures -> diagnostic/figures_ablation/
```

---

## Quick end-to-end example (bundle_0, interference SNR5)

```bash
cd /storage/home/tuochao/Mem-alpha-audio
export PYTHONPATH=/storage/home/tuochao/Mem-alpha-audio

# 0. Noisy variant (bundles.json + timeline already exist)
bash audio_script/run_mix_wham_inteference_mosaic.sh

# 1a. Step1, 4 shards on 4 GPUs
RAW_DATA_PATH=/checkpoint/seamless/tuochao/data/Mix_Mosaic/naturalistic/test_interf_SNR5 \
  bash audio_script/run_step1_mosaic_shards.sh
# 1b. Step2/3 per bundle
RAW_DATA_PATH=/checkpoint/seamless/tuochao/data/Mix_Mosaic/naturalistic/test_interf_SNR5 \
  RUN_STEP1=0 bash audio_script/run_demo_pipeline_mosaic.sh

# 2. Per-bundle Parquet (pred + gt)
PYTHONPATH=. python prepare_data/prepare_audio_parquet_perltqa.py \
  --data_dir Audio_Results/vibevoice/test_interf_SNR5/step2 \
  --output_root outputs/mosaic_step3_interf_SNR5 \
  --time_info_path outputs/mosaic_step3/mosaic_session_timeline.json \
  --data_source seamlessinteraction_options \
  --mode both --suffix mosaic_interf_SNR5

# 3. Memory + QA
bash run_pipeline.sh \
  outputs/mosaic_step3_interf_SNR5/bundle_0/dataset_pred_name_bundle_0_mosaic_interf_SNR5.parquet \
  seamlessinteraction_options \
  outputs/mosaic_step3/QA/bundle0/

# 4. Probe + compare
bash diagnostic/run_probe_errors.sh \
  agents/qwen3.6-27b_..._dataset_pred_name_bundle_0_mosaic_interf_SNR5_no_thinking_tokens_2048 \
  outputs/mosaic_step3_interf_SNR5/bundle_0 \
  outputs/mosaic_step3/QA/bundle0/bundle_0_content_qa.jsonl \
  outputs/mosaic_step3_interf_SNR5/bundle_0/dataset_gt_name_bundle_0_mosaic_interf_SNR5.parquet
python diagnostic/compare_probes.py agents/..._bundle_0_mosaic_..._2048 --csv mosaic_b0.csv
```

---

## Troubleshooting

- **Step1 shard logs stop mid-generation with no traceback** — the process was
  killed (usually the host OOM killer). Check no other GPU job is running, then
  just relaunch: finished conversations are skipped.
- **A chunk takes far longer than the ~60-90 s/it norm** — greedy decoding
  collapsed into a repetition loop that never emits EOS and runs to
  `MAX_NEW_TOKENS`. `MAX_NEW_TOKENS=8192` + `REPETITION_PENALTY=1.2` bound the
  damage; `NO_REPEAT_NGRAM_SIZE=3` breaks the loop outright. Count runaways with
  `grep -c "hit_eos=False" Audio_Results/logs/step1_mosaic_*.log`.
- **`KeyError` in memory construction** — `--dataset` and the Parquet's
  `data_source` column disagree. Both must be `seamlessinteraction_options`;
  `mosaic` is not a valid dataset key.
- **`no timeline entry for '<conv_id>'` (Parquet build)** — a conversation is
  missing from `mosaic_session_timeline.json`; `source_file` minus `.json` must
  equal `<pair>_<clip>`.
- **QA dir not found** — `QA/bundle0/`, not `QA/bundle_0/`.
- **Probe: "no instance dir with results.json"** — you pointed `BASE_DIR` at
  `.../0`; pass the parent run dir.
- **Probe hangs / connection refused** — the memory server at `SERVER_URL` isn't up.
- **Seed sweep produces identical results** — `MEM_TEMPERATURE` must be > 0.
- **Step2 pool is empty for a bundle** — `--season_filter` matches on the `conv_id`
  prefix; confirm `bundles.json` `folders` still match the on-disk pair folders.
