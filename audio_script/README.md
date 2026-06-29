# Audio Pipeline (`audio_script/`)

A three-step pipeline that turns raw TV-show audio (the **Bazinga** / *The Big Bang
Theory* / *Friends* dataset) into a speaker-named, turn-level transcript:

```
raw audio + word-level GT          per-chunk diarization + ASR        cross-file speakers          real speaker names
  (.wav / .txt)        ──Step 1──▶   diart_pred.npy / transcript_*    ──Step 2──▶  speaker_map.json  ──Step 3──▶  extracted_speaker_name.json
                                     (+ Step1 eval: DER / cpWER)                  parsed_dialog_*.json
```

Each step has a thin bash driver (edit the paths at the top, then run it). The
drivers just activate the right conda env, set `PYTHONPATH`, and call a Python
module. Run them from the repo root (`Mem-alpha-audio/`).

> Only the scripts below are documented. Other `run_demo_step1_*.sh` variants
> (`_offline.sh`, `_vibevoice.sh`, `run_demo_step1.sh`) are alternate Step-1
> backends/datasets and are out of scope here.

---

## Step 1 — Diarization + multi-talker ASR

**Driver:** `run_demo_step1_bazinga.sh` → `python -m audio_script.Multi_ASR.step1_bazinga`

Splits each episode into dialogue chunks, then for every chunk runs speaker
diarization (who spoke when) and multi-talker ASR (what each speaker said).

### Backends (set via `METHOD`)
The driver selects one inference backend through the `METHOD` env var; override
without editing the file:

```bash
METHOD=nemo-offline ./audio_script/run_demo_step1_bazinga.sh
```

| `METHOD`         | Backend                                       | Conda env   |
| ---------------- | --------------------------------------------- | ----------- |
| `nemo-streaming` (default) | NeMo streaming SortFormer diarizer + cache-aware ASR | `nemo`      |
| `nemo-offline`   | NeMo offline SortFormer + offline ASR         | `nemo`      |
| `vibevoice`      | VibeVoice end-to-end multi-talker ASR         | `vibevoice` |

Key configuration knobs in the script: `DATA_PATH`, `OUTPUT_DIR`,
`SEASON_FILTER` (e.g. `("Season01")` to limit which episodes run),
`DIAR_MODEL_PATH` / `ASR_MODEL_PATH` and `MAX_NUM_OF_SPKS` (NeMo), and the
VibeVoice decode params (`VV_MODEL_PATH`, `MAX_NEW_TOKENS`, `TEMPERATURE`, …).

### Input
`DATA_PATH` points at a directory of Bazinga episodes:

```
data_dir/
    TheBigBangTheory.Season01.Episode01.en.wav     # audio
    TheBigBangTheory.Season01.Episode01.txt        # word-level annotation
    ...
```

The `.txt` annotation is space-separated, 9 columns:

```
file_id  speaker  start_time  end_time  word  confidence  listener  scene_context  misc
```

### Output
Results are written under `OUTPUT_DIR/<conv_id>/CHUNK_<id>/`, one folder per
chunk, each containing **5 artifacts**:

| File                   | Format | Contents |
| ---------------------- | ------ | -------- |
| `diart_pred.npy`       | numpy `(num_frames, num_speakers)` | Binary diarization activity matrix (frame = `feat_len_sec`, ~0.08 s). |
| `transcript_pred.json` | `{speaker_id: [{word, start, end}, …]}` | Predicted per-speaker word-level transcript. |
| `transcript_gt.json`   | `{speaker_id: [{word, start, end, …}, …]}` | Ground-truth words, timestamps shifted to be chunk-relative. |
| `vad_gt.json`          | `{speaker_id: [{start, end}, …]}` | Ground-truth voice-activity intervals per speaker. |
| `sample_info.json`     | JSON manifest | Pointers + metadata used by Step 2 / eval: `dataset`, `conv_id`, `chunk_id`, `audio_file`, `speakers`, `transcript_path`, `vad_path`, `diart_path`, `pred_transcript_path`, `feat_len_sec`, `time_stamp` (`[start_sample, end_sample]`). |

Chunks whose `diart_pred.npy` + `transcript_pred.json` + `sample_info.json`
already exist are skipped, so the step is resumable.

### Evaluating Step 1 — `evaluate_audio_results.py`
Scores the Step-1 output (diarization + ASR quality) before any speaker
matching is done.

```bash
python audio_script/evaluate_audio_results.py <OUTPUT_DIR>   # e.g. .../step1
```

- **Input:** a Step-1 output root. The script recursively finds every
  `sample_info.json`, loading the sibling `diart_pred.npy` and the GT
  VAD/transcript it points to.
- **Computes:**
  - **DER** (Diarization Error Rate) — best-permutation match of predicted vs.
    GT speakers, broken into miss / false-alarm / confusion seconds + frame accuracy.
  - **cpWER** (concatenated min-permutation Word Error Rate) — when predicted &
    GT transcripts are present.
- **Output:**
  - A `diarization_plot.png` next to each `sample_info.json` (GT vs. aligned
    prediction raster).
  - Printed per-sample metrics and a final batch summary (mean / median / min /
    max DER and cpWER).

> Note: the eval currently filters to `Season01` paths (hard-coded). Adjust in
> `evaluate_audio_results.py` if you need other seasons.

---

## Step 2 — Speaker embedding + cross-file matching

**Driver:** `run_demo_step2.sh` → `python audio_script/Speaker_Track/step2_speaker_match_v2.py`
**Conda env:** `mem` (WeSpeaker embeddings)

Takes the per-chunk Step-1 output and ties together the *same physical person*
across all chunks/episodes. For each chunk it slices the audio by the
diarization result, extracts a WeSpeaker embedding per local speaker, and links
local speakers into a **global** speaker pool via cosine-similarity matching.

Key knobs in the script: `DATA_DIR` (Step-1 `step1` folder), `OUTPUT_DIR`
(`step2` folder), `EMBEDDING_MODEL_DIR`, `SIMILARITY_THRESHOLD` (default 0.65),
`EMBEDDING_DEVICE`. The Python module also supports `--linker {greedy,asnorm,twopass}`.

### Input
The Step-1 `OUTPUT_DIR`, discovered as `data_dir/*/*/sample_info.json` (with the
sibling `diart_pred.npy` + `transcript_pred.json`).

### Output
Written under `OUTPUT_DIR`:

| File | Format | Contents |
| ---- | ------ | -------- |
| `speaker_map.json` | `{global_speaker_id: gt_speaker_label}` | Best-guess GT label for each discovered global speaker (majority vote). |
| `raw_speaker_tracking.json` | `{speaker_cluster_pred, speaker_cluster_gt}` | Raw predicted vs. GT speaker clustering used for accuracy scoring. |
| `<spk_pair>/<conv_id>/parsed_dialog_pred.json` | turn list `[{speaker, text, …}]` | Predicted conversation, speakers relabeled to global IDs. |
| `<spk_pair>/<conv_id>/parsed_dialog_gt.json`   | turn list `[{speaker, text, …}]` | Ground-truth conversation turns. |
| `speaker_segments/` | `.wav` | Per-speaker concatenated audio used for embedding extraction. |

It also prints DER / cpWER (recomputed) and speaker-tracking accuracy
(TP / FP / FN) to stdout.

---

## Step 3 — Speaker name extraction

**Driver:** `run_demo_step3.sh` → `python audio_script/Speaker_Track/step3_speaker_name_extract.py`
**Conda env:** `mem`

Reads the parsed dialogue from Step 2 and uses a Qwen3 LLM to infer each
speaker's **real name** from the conversation content (e.g. someone being
addressed by name). Requires a running Qwen-compatible OpenAI endpoint; the
driver sets `QWEN_URL` (default `http://localhost:8002/v1`). Relevant env vars:
`QWEN_URL` / `QWEN_BASE_URL`, `QWEN_MODEL_NAME` / `QWEN3_MODEL`,
`QWEN_API_KEY` / `OPENROUTER_API_KEY`.

Set `DATA_PATH` in the script to the Step-2 `step2` output folder.

### Input
The Step-2 `OUTPUT_DIR`, discovered as `data_dir/*/*/parsed_dialog_pred.json`
(anonymized as `Speaker_0`, `Speaker_1`, … before being sent to the LLM).

### Output
| File | Format | Contents |
| ---- | ------ | -------- |
| `<data_dir>/extracted_speaker_name.json` | `{global_speaker_id: real_name}` | Resolved speaker name per global speaker; unidentified speakers fall back to `Unknown_speakerNNN`. |

Identified names are also printed and (when GT names are available) checked
against ground truth with ✓/✗ markers.

---

## Quick start

```bash
# from repo root: Mem-alpha-audio/
# 1. diarization + ASR (edit DATA_PATH / OUTPUT_DIR / SEASON_FILTER first)
METHOD=nemo-streaming ./audio_script/run_demo_step1_bazinga.sh
python audio_script/evaluate_audio_results.py /path/to/.../step1   # optional: score Step 1

# 2. cross-file speaker matching (edit DATA_DIR / OUTPUT_DIR first)
./audio_script/run_demo_step2.sh

# 3. speaker name extraction (needs a running Qwen endpoint; edit DATA_PATH first)
./audio_script/run_demo_step3.sh
```
