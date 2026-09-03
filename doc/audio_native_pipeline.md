# Audio-Native Memory Pipeline (Qwen3-Omni)

The default pipeline is a **cascade**: audio → diar+ASR → speaker tracking → name
extraction → *text* chunks → memory construction with a text LLM. Every front-end error
is baked in before the memory agent sees anything.

This is the **audio-native** arm: the memory agent is
**Qwen3-Omni-30B-A3B-Instruct** and each chunk is delivered as the raw **audio slice**
plus its session timestamp — no transcript, no speaker roster. Everything else is
shared with the text arm (same chunk boundaries, same memory tools, same compression
strategies, same QA + probing), so the two are directly comparable.

| | text arm | audio arm |
|---|---|---|
| Step-1 script | `run_memory_construction_new.py` | `run_memory_construction_audio.py` |
| Agent class | `agent_prompting.MemoryAgentPrompting` | `agent_omni.MemoryAgentOmni` |
| Agent config | `config/qwen3.6-27B_agent.yaml` | `config/qwen3-omni-30b_agent.yaml` |
| Chunk input | `chunks[i]` (dialogue text) | `audio_chunks[i]` (wav + time range) |
| Prompt | `unified_prompt_multispeaker` | `unified_prompt_multispeaker_audio` |
| Wrapper | `run_pipeline.sh` | `run_pipeline_audio.sh` |
| QA (step 2) | `run_qa_evaluation.py` | **same, unchanged** |
| Error probe | `diagnostic/run_probe_errors.sh` (G/C/T/S) | `diagnostic/run_probe_errors_audio.sh` (G/C/S — no T) |

Memory itself is text in both arms, so QA (`run_qa_evaluation.py`) and scoring
(`evaluate_agent_results.py`) need no changes. The error probe needs one: the audio arm
has no transcription stage, so it runs C+S only via
`diagnostic/run_probe_errors_audio.sh` (= `probe_errors.py --no_transcript_probe`) —
see [Stage 3](#stage-3--error-probe-where-do-the-failures-come-from).

---

## Stage 0 — Model

`Qwen/Qwen3-Omni-30B-A3B-Instruct` (~62 GB bf16) must be in the shared HF cache. The
agent-side proxy cannot reach huggingface.co, so run this in your own shell:

```bash
HF_HOME=/checkpoint/seamless/tuochao/Models/huggingface/ \
HF_HUB_CACHE=/checkpoint/seamless/tuochao/Models/huggingface/ \
  hf download Qwen/Qwen3-Omni-30B-A3B-Instruct
```

vLLM 0.23 in the `vllm` env already registers `Qwen3OmniMoeForConditionalGeneration`
(thinker only → text output). It fits on one H200 at `tensor_parallel_size: 1`.

## Stage 1 — Add audio provenance to the Parquet

Step 1 of the audio front-end already recorded, per chunk, the source wav and its sample
range in `sample_info.json`. `add_audio_chunks.py` joins that onto an existing Parquet
via the `chunk_folders` column and writes a `<name>_audio.parquet` with one extra
column:

```
audio_chunks[i] = {conv_id, chunk_id, chunk_folder, audio_file,
                   start_sec, end_sec, duration_sec, timestamp, speakers}
```

```bash
PYTHONPATH=. python prepare_data/add_audio_chunks.py \
  --parquet outputs/mosaic_step3/bundle_0/dataset_pred_name_bundle_0_mosaic.parquet \
  --step1_dir Audio_Results/vibevoice/test/step1 \
  --time_info_path outputs/mosaic_step3/mosaic_session_timeline.json
```

- `--step1_dir` is the **step1** tree matching the step2 tree the Parquet was built from
  (`Audio_Results/<method>/<DATASET_TAG>/step1`). A noisy condition must point at its own
  tag (`test_interf_SNR5`), otherwise you would score clean audio.
- The `chunks` text column is kept untouched — it stays the denominator for the
  compression word budget and for `compression.json`, so ratios match the text arm.
- Missing `sample_info.json` / missing wav / timeline gap → hard error, never a silent skip.
- `pred` vs `gt` Parquets contain the **same audio**; they differ only in the text used
  for accounting. Use the `pred` one unless you are specifically studying the accounting.

Already built (clean Mosaic, 829 chunks / ~43 h audio):
`outputs/mosaic_step3/bundle_{0,1,2,3}/dataset_{pred,gt}_name_bundle_*_mosaic_audio.parquet`.

Bazinga / PerLTQA use the same command with their own `--step1_dir`
(`Audio_Results/vibevoice/TheBigBangTheory/step1`,
`.../dialogue_tts_en_name_replaced/step1`) and their own timeline JSON — no code change.

PerLTQA bundle_0, already built (217 chunks / 6.4 h audio, longest 172 s so nothing is
sub-sliced):

```bash
PYTHONPATH=. python prepare_data/add_audio_chunks.py \
  --parquet outputs/step3_perltqa_replaced_name/bundle_0/dataset_pred_name_bundle_0_perltqa.parquet \
  --step1_dir Audio_Results/vibevoice/dialogue_tts_en_name_replaced/step1 \
  --time_info_path outputs/perltqa_data/perltqa_session_timeline.json
# -> .../dataset_pred_name_bundle_0_perltqa_audio.parquet
```

## Stage 2 — Memory construction + QA

```bash
bash run_pipeline_audio.sh \
  outputs/mosaic_step3/bundle_0/dataset_pred_name_bundle_0_mosaic_audio.parquet \
  seamlessinteraction_options \
  outputs/mosaic_step3/QA/bundle0/
```

Env knobs are identical to `run_pipeline.sh` (`COMPRESSION_STRATEGY`, `SEED`,
`MEM_TEMPERATURE`, `ROLLOUT_LABEL`, `FORCE_REANSWER`) plus `AGENT_CONFIG` and
`MAX_AUDIO_SEC`.

On SLURM, `AUDIO_NATIVE=true` selects this wrapper:

```bash
AUDIO_NATIVE=true sbatch submit_pipeline.slurm \
  outputs/mosaic_step3/bundle_0/dataset_pred_name_bundle_0_mosaic_audio.parquet \
  seamlessinteraction_options outputs/mosaic_step3/QA/bundle0/
```

Run dir: `agents/qwen3-omni-30b_Qwen_Qwen3-Omni-30B-A3B-Instruct_seamlessinteraction_options_dataset_pred_name_bundle_0_mosaic_audio_no_thinking_tokens_2048/0/`
with the usual `agent_state.json`, `compression.json` (plus `input_audio_seconds`),
`chunks_and_function_calls.json` (plus `audio_chunk` provenance),
`final_responses.json`, `embeddings.npz`.

**Long chunks.** One turn = one audio clip. A chunk longer than `max_audio_sec`
(config default 480 s; 3 of the 829 Mosaic chunks) is split into equal sub-slices
processed as consecutive memory turns. All sub-turns keep the *original* `chunk_idx`, so
QA-evidence mapping and the probes are unaffected.

**Smoke test** before committing a full run — use a rollout label so the partial
`agent_state.json` cannot be mistaken for a finished run (step 1 skips any instance that
already has one):

```bash
conda activate vllm
ROLLOUT_LABEL=smoke python run_memory_construction_audio.py \
  --agent_config config/qwen3-omni-30b_agent.yaml \
  --dataset seamlessinteraction_options \
  --parquet_path outputs/mosaic_step3/bundle_0/dataset_pred_name_bundle_0_mosaic_audio.parquet \
  --batch_size 1 --limit_chunks 5 --rollout_label smoke
```

Then check `agents/qwen3-omni-30b_.../smoke/0/`:
- `final_responses.json` — are the responses `✿FUNCTION✿` tool calls, and is the content
  actually about what is *said in the audio*?
- `agent_state.json` — non-empty `semantic` / `episodic`.

If Omni does not emit the `✿FUNCTION✿` format reliably, the parsing fallback already in
`agent.py` is `_parse_function_calls_from_text` (plain JSON) — wire it in
`MemoryAgentOmni` before spending a full run.

### Verbosity: why the prompt caps tool calls

Given the memory tools and no further constraint, Omni does not summarize the recording —
it *narrates* it, one memory item per turn of dialogue: 30-40 tool calls per chunk,
generation running to `max_tokens` every single time (so the last call is cut mid-JSON
and dropped), and a memory **1.7x larger than the transcript it replaced**. Raising the
cap does not help: at 4096 and 8192 tokens it simply fills the larger budget (149 calls
at 8k).

Two levers fix it, and both are needed — measured on mosaic bundle_0 chunks 0-3 with
`debug/omni_verbosity_sweep.py` (calls per chunk / did every generation finish /
memory-words per transcript-word):

| variant | calls | finished | mem/text |
|---|---|---|---|
| uncapped, no penalty | 29,38,35,35 | no | 1.72 |
| cap 12 + penalty 1.1 | 15,34,30,34 | no | 1.67 |
| cap 8 + penalty 1.1 | 8,8,24,25 | no | 1.16 |
| **cap 5 + penalty 1.1** | **5,11,5,5** | **yes** | **0.40** |

Adherence collapses as the stated cap grows, so the defaults in
`config/qwen3-omni-30b_agent.yaml` are `max_memory_calls: 5` (rendered into the prompt's
`{max_memory_calls}` HARD LIMIT sentence, placed last) and `repetition_penalty: 1.1`
(breaks the enumeration loop). A third guard, `max_calls_executed`
(default `2 * max_memory_calls`), executes only the first N calls of a turn and logs a
`[WARN]`, so one runaway chunk cannot dominate the memory. On a 6-chunk end-to-end run
this gives **1.28x compression** (vs 0.33x before) with one warned chunk.

Re-run the tuning experiment after any prompt change:

```bash
python debug/omni_verbosity_sweep.py --chunks 4 --variants cap5_rep,cap8_rep,uncapped
```

### Other datasets: PerLTQA

Nothing dataset-specific in the code — `--dataset` just has to equal the parquet's
`data_source`, and both arms format the same family of prompt
(`unified_prompt_multispeaker` for text, `unified_prompt_multispeaker_audio` for audio),
so PerLTQA needs no prompt of its own.

Smoke-tested on bundle_0 (`debug/smoke_audio_construction.slurm`, 5 chunks, 1 GPU, ~7
min end to end including model load):

```bash
sbatch debug/smoke_audio_construction.slurm \
  outputs/step3_perltqa_replaced_name/bundle_0/dataset_pred_name_bundle_0_perltqa_audio.parquet \
  perltqa 5 smoke
```

| check | result |
|---|---|
| tool-call format | `✿FUNCTION✿` parsed on every chunk, 0 tool errors |
| calls per chunk | 5, 5, 5, 5, 5 — exactly the cap, every generation finished |
| memory density | 150 tokens/chunk vs the cascade's 165 — `max_memory_calls: 5` transfers as-is, no retune |
| compression | 2.13x (cascade full run: 2.55x) |
| core memory | non-empty (tiktoken cache resolved) |
| speaker naming | real names recovered from speech (Lindsay / Rowan / Baker Hayden); one chunk fell back to `Speaker A` where no name was spoken |
| session id | absent, as intended |

Full run ≈ 18 s/chunk × 217 ≈ 65 min of generation plus model load, then QA over 609
questions:

```bash
AUDIO_NATIVE=true FORCE_REANSWER=1 sbatch submit_pipeline.slurm \
  outputs/step3_perltqa_replaced_name/bundle_0/dataset_pred_name_bundle_0_perltqa_audio.parquet \
  perltqa outputs/perltqa_data/qa_multi_name_replaced/bundle_0_filterd/
```

Then probe it — PerLTQA is open-ended, so grade with the LLM judge exactly as
`run_probe_errors_perltqa.sh` does:

```bash
DATA_SOURCE=perltqa SCORER=llm_judge bash diagnostic/run_probe_errors_audio.sh \
  agents/qwen3-omni-30b_Qwen_Qwen3-Omni-30B-A3B-Instruct_perltqa_dataset_pred_name_bundle_0_perltqa_audio_no_thinking_tokens_2048 \
  outputs/step3_perltqa_replaced_name/bundle_0 \
  outputs/perltqa_data/qa_multi_name_replaced/bundle_0_filterd/qa.jsonl \
  outputs/step3_perltqa_replaced_name/bundle_0/dataset_pred_name_bundle_0_perltqa_audio.parquet
```

Cascade baseline to compare against:
`agents/qwen3.6-27b_Qwen_Qwen3.6-27B_perltqa_dataset_pred_name_bundle_0_perltqa_no_thinking_tokens_2048`
(avg score 0.716, 2.55x, 609 questions).

### Experimental rolling audio history

`run_memory_construction_audio_history.py` is an isolated experimental variant that
prepends the previous audio chunks to each current chunk. The preceding recordings are
labelled as **reference history only**: the model may use them to recognize voices and
resolve names or references, but every memory write must describe information stated or
revealed in the current recording.

Defaults:

- previous 5 original chunks (`--history_chunks 5`);
- at most 180 seconds from the start of each history chunk
  (`--history_max_audio_sec 180`), keeping the six-audio prompt within the 32k context;
- the normal 480-second current-chunk split;
- an isolated `_history5` run-directory suffix;
- at most 5 executed memory calls per current turn, even if Omni emits more.

Smoke test on the first seven Mosaic chunks:

```bash
sbatch debug/smoke_audio_history.slurm \
  outputs/mosaic_step3/bundle_0/dataset_pred_name_bundle_0_mosaic_audio.parquet \
  seamlessinteraction_options 7 5
```

Direct invocation:

```bash
python run_memory_construction_audio_history.py \
  --agent_config config/qwen3-omni-30b_agent.yaml \
  --dataset seamlessinteraction_options \
  --parquet_path outputs/mosaic_step3/bundle_0/dataset_pred_name_bundle_0_mosaic_audio.parquet \
  --history_chunks 5 \
  --history_max_audio_sec 180
```

Default output directory for the command above:

```text
agents/qwen3-omni-30b_Qwen_Qwen3-Omni-30B-A3B-Instruct_seamlessinteraction_options_dataset_pred_name_bundle_0_mosaic_audio_no_thinking_tokens_2048/history5/0/
```

The final component is the Parquet row index. Without an explicit `--run_dir_suffix`,
the history runner uses `history<N>` (for example, `history4` or `history5`) so it cannot
overwrite the ordinary audio-native result. A compression strategy is placed before the
history suffix, for example `..._tokens_2048_comp_x3_history5/0/`. The smoke launcher
uses a suffix such as `history5_smoke7`, producing
`..._tokens_2048_history5_smoke7/0/`.

To run QA against this memory directory, pass the same suffix:

```bash
python run_qa_evaluation.py \
  --agent_config config/qwen3-omni-30b_agent.yaml \
  --dataset seamlessinteraction_options \
  --parquet_path outputs/mosaic_step3/bundle_0/dataset_pred_name_bundle_0_mosaic_audio.parquet \
  --custom_qa_dir outputs/mosaic_step3/QA/bundle0/ \
  --run_dir_suffix history5
```

Each output directory contains `agent_state.json`, `compression.json`,
`data_instance_info.json`, `chunks_and_function_calls.json`, `final_responses.json`, and
`embeddings.npz` when embeddings are available.

The output's `chunks_and_function_calls.json` records the exact
`history_chunk_indices` used for every current turn. This arm should be treated as an
ablation: the earlier direct speaker-matching study found that multi-conversation windows
were unreliable, so QA and memory-contamination results must be compared with the
single-audio baseline before adopting it.

Initial Mosaic bundle-0 smoke test (first 6 chunks, history growing from 0 to 5): all six
prompts, including the full five-reference window, ran successfully and stopped normally,
and the execution guard kept
every turn at 5 memory calls. The executed semantic/episodic writes described the current
chunk rather than copying history. The model reused `Speaker A`/`Speaker B` consistently
across this same-pair sequence, but recovered no real names; this is only a plumbing and
prompt-boundary check, not evidence of an accuracy improvement. One four-history turn
emitted 8 calls before the runner discarded the last 3, confirming that the hard execution
cap is still necessary.

## Stage 3 — Error probe (where do the failures come from?)

Same behavioral probe as the text arm (`diagnostic/probe_errors.py`: re-answer each
question on a curated context and see whether the answer flips), **minus the T stage**.
The text arm's cascade is

```
audio → diar+ASR → transcript chunk → memory → retrieval → answer
                   └── T-probe ──┘   └─ C ─┘   └─── S ───┘
```

The audio arm has nothing between the recording and the memory agent, so there is no
transcription output to probe. A C-probe failure therefore *is* a construction failure —
the `construction` bucket here bundles **listening/perception** with memory writing.
What is left:

```
C-probe : final memory units traced to the evidence chunk(s)   ── QA ──▶ correct?
   ├─ no  ▶ construction   (the agent listened to that chunk and did not store the fact)
   └─ yes ▶ S-probe: evidence units ∩ what the retriever actually surfaced
              ├─ yes ▶ response    (it was shown, model still wrong)
              └─ no  ▶ retrieval   (in the store, not surfaced)
```

The optional **G-probe** (whole gold dialogue chunk = ceiling) is still meaningful and is
text-only, so it is *identical* to the text arm's and is read from the same
`<DATA_ROOT>/tg_probe_cache.json` — enable with `RUN_GOLDEN=1`, it will reuse the cascade
run's cached gold answers instead of recomputing them.

```bash
# memory server must be up (see diagnostic/README.md "Launching the local LLM backend")
bash diagnostic/run_probe_errors_audio.sh \
  agents/qwen3-omni-30b_..._mosaic_audio_no_thinking_tokens_2048
# writes <instance_dir>/error_probe.json + error_probe_debug.json + qa_evidence_map.json
```

Defaults in the wrapper are mosaic bundle_0
(`--data_root outputs/mosaic_step3/bundle_0`,
`--qa_file outputs/mosaic_step3/QA/bundle0/bundle_0_content_qa.jsonl`,
`--parquet .../dataset_pred_name_bundle_0_mosaic_audio.parquet`); pass them positionally
for another bundle/dataset. Under the hood it is just
`probe_errors.py --no_transcript_probe`, so any other run can be probed directly.

Nothing else needed changing, because:

- the `*_audio.parquet` keeps `chunk_folders` verbatim, so the QA → chunk_idx map is the
  same deterministic one the text arm uses (verified: 100/100 mosaic bundle_0 questions
  localize, mean 4.6 memory units per question);
- `chunks_and_function_calls.json` from `run_memory_construction_audio.py` has **one
  record per original `chunk_idx`** even when a long chunk was split into audio
  sub-turns, so chunk_idx → memory-id provenance is unambiguous;
- memory is text in both arms, so the C/S contexts and the QA server are identical.

Reading the numbers: `construction` in the audio arm and `transcription + construction`
in the text arm cover the same span of the pipeline, so **that pair is the like-for-like
comparison** — audio `construction` vs cascade `transcription + construction`. The
cascade's `transcription` slice tells you how much of its loss ASR alone is responsible
for, i.e. how much the audio arm could in principle win back.

## Stage 4 — Compare against the cascade

```bash
QWEN_URL="http://localhost:8002/v1" python evaluate_agent_results.py \
  --base_dir agents/qwen3-omni-30b_..._mosaic_audio_no_thinking_tokens_2048
python diagnostic/compare_probes.py <omni_run_dir> <text_run_dir> --csv audio_vs_text.csv
```

Compare QA accuracy, `compression.json` (same input-token denominator in both arms) and,
for a noise sweep, the audio arm on `_interf_SNR*` Parquets vs the cascade on the same
condition — that isolates how much of the cascade's noise degradation comes from ASR
rather than from memory.

---

## Notes / gotchas

- **No session id in the prompt.** The chunk metadata block carries only the session
  **date** and the excerpt's second offsets — deliberately no `conv_id`. A deployed
  listener has the recording and its timestamp; the session id is dataset bookkeeping.
  An earlier version did pass it (and told the model to write `... into session
  {conv_id}` in every episodic item): it cost **11% of the memory text** on mosaic
  bundle_0 (16.4k of 149k chars, 530 of 865 units) and never once bound a voice to a
  participant, so it was pure overhead on the compression budget.
  This is invisible to the probes — `diagnostic/probe_errors.py` localizes through
  `parquet chunk_folders → chunk_idx → memory ids`, never through memory text — but it
  does mean units are anchored only by a date, and dates repeat (mosaic bundle_0: 196
  chunks over 20 distinct dates, up to 18 sessions sharing one), so a `Speaker A` in the
  pooled store is not attributable to a specific recording.
- **Speaker names.** The audio arm gets no name map. The prompt tells the model to use a
  name only when it is actually spoken (self-intro / vocative) and otherwise a
  chunk-local `Speaker A/B` label with a voice descriptor. Chunk-local labels do **not**
  link across chunks — there is no cross-chunk speaker embedding — so expect
  name-dependent QA to be the main gap vs the cascade. On mosaic bundle_0 this shows up
  as **abstention, not error**: the audio arm answers "not sure" on 51/100 (vs 26/100 for
  the cascade) because questions name participants by `P####` while memory says
  `Speaker A`; on the questions it does commit to, it is 86% correct — the same precision
  as the cascade's 85%. The entire 42% vs 63% end-to-end gap is that abstention.
  Feeding past chunks into the context to fix this was measured and does **not** work; see
  [`doc/omni_cross_chunk_speaker_experiment.md`](omni_cross_chunk_speaker_experiment.md).
- **`--dataset` must match the Parquet's `data_source`** (Mosaic =
  `seamlessinteraction_options`), same as the text arm.
- **`limit_mm_per_prompt`** must include `audio: 1`; the Qwen3.6 text path forces
  `{"image": 0, "video": 0}` only, which would disable audio entirely.
- **`gdn_prefill_backend`** is a Qwen3-Next knob and is deliberately NOT set for Omni.
- The `vllm` env has no `librosa`; `MemoryAgentOmni.load_audio_slice` resamples with
  `scipy.signal.resample_poly` when a wav is not already 16 kHz.
- **`mm_encoder_attn_backend: TORCH_SDPA`** is required. vLLM 0.23's FlashAttention path
  for this model's multimodal encoders passes `cu_seqlens` on CPU and engine startup dies
  with `cu_seqlens_q must be on CUDA`.
- **`HF_HUB_OFFLINE=1`.** The Qwen3-Omni repo ships no `tokenizer.json` (only
  `vocab.json` + `merges.txt`); online, transformers tries to fetch it and errors out,
  offline it builds the tokenizer from the local files. `run_pipeline_audio.sh` sets this
  plus `HF_HOME=~/.cache/huggingface` (where the 66 GB snapshot lives).
- **`TIKTOKEN_CACHE_DIR`.** `memory.py` counts tokens with tiktoken, which downloads
  `o200k_base` on first use. When that download fails, `memory_update` raises and **core
  memory silently stays empty for the whole run** — the failure only shows up as
  `[tool memory_update error]` inside `chunks_and_function_calls.json`. The encoding is
  cached in-repo at `.cache/tiktoken/`; `run_pipeline_audio.sh` points at it.
