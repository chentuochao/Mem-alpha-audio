# Mem-Alpha Audio — Pipeline Reference

## Overview

Three sequential steps transform raw audio transcripts into a structured memory agent evaluation:

```
Raw Inputs (dialogs · QA pairs · timestamps)
          │
          ▼  prepare_step2_data.py
outputs/test_{split}.parquet
          │
          ▼  run_memory_construction.py  (local vLLM / Qwen)
agents/{name}/{idx}/agent_state.json  +  embeddings.npz
          │
          ▼  run_qa_evaluation.py  →  memory_server.py  (Qwen3-32B / GPT)
agents/{name}/{idx}/results.json
```

---

## Step A — Prepare Dataset

```bash
python prepare_step2_data.py
```

Merges conversation dialogs, QA pairs (per question-type JSONL), and per-clip
session timestamps into a single Parquet file.  For the `pred` split it also
maps predicted speaker IDs to canonical names via `speaker_map.json`.

### Inputs

| File / Source | Description |
|---|---|
| `outputs/demo_output_step2/**/{clip_id}/parsed_dialog_{split}.json` | Per-clip dialog turns with speaker labels |
| `outputs/QA_pairs_generated/*.jsonl` | QA pairs grouped by question type |
| `outputs/timeline_and_QA/longmemeval_style_session_timeline.jsonl` | Per-clip session date/timestamp |
| `outputs/demo_output_step2/speaker_map.json` | (pred split only) predicted → canonical speaker map |

### Output

| File | Description |
|---|---|
| `outputs/test_{split}.parquet` | Single-row Parquet: `chunks` (JSON array of dialog strings), `questions_and_answers` (JSON array), `data_source`, `metadata`, `num_chunks`, `num_questions` |

### Internal Flow

```
parsed_dialog_{split}.json ──┐
QA pairs (*.jsonl)           ├──▶ Merge & format chunks  ──▶ outputs/test_{split}.parquet
session_timeline.jsonl ──────┘
speaker_map.json (pred only)
```

---

## Step B — Memory Construction

```bash
python run_memory_construction.py \
  --agent_config config/memalpha-qwen3-4b_agent_0.05-0.1.yaml \
  --dataset seamlessinteraction_gt \
  --batch_size 1
```

Streams each conversation through a **vLLM**-hosted Qwen model chunk by chunk.
The model issues tool calls to populate a `Memory` object
(`add_semantic`, `add_episodic`, `add_core`).  Conversations whose
`agent_state.json` already exists are skipped automatically.

### Inputs

| File / Source | Description |
|---|---|
| `config/*.yaml` | Agent config: `model_name`, `enable_thinking`, `thinking_budget`, `max_new_tokens`, `vllm: true` |
| `config/prompts_wrt_datasource.yaml` | Per-datasource prompt templates (`unified_prompt`) and `including_core` flag |
| `outputs/test_{split}.parquet` | Dataset produced by Step A (loaded via `ConversationCreator`) |

### Outputs (per conversation, under `./agents/{agent_name}_{model_name}_{dataset}/{idx}/`)

| File | Description |
|---|---|
| `agent_state.json` | Serialised memory: `semantic[]`, `episodic[]`, `core[]`, embedding IDs |
| `embeddings.npz` | Compressed NumPy arrays: `semantic_matrix`, `episodic_matrix` |
| `chunks_and_function_calls.json` | Each raw chunk paired with the tool calls it triggered |
| `final_responses.json` | Raw vLLM text outputs for every chunk |
| `data_instance_info.json` | `data_source` + `global_idx` metadata |

### Internal Flow (per conversation)

```
Parquet dataset (ConversationCreator)
        │
        ▼
  Conversation chunks  (text segments)
        │  ◀─────────────────────────────────────────────────────────────┐
        ▼                                                                 │ next chunk
  Format prompt  (unified_prompt template + current memory state)        │
        │                                                                 │
        ▼                                                                 │
  vLLM inference  (Qwen)                                                  │
    ┌── enable_thinking=true ──────────────────────────────────────────┐  │
    │  Phase 1: generate up to thinking_budget tokens                  │  │
    │  Phase 2: continue for (max_new_tokens − thinking_budget) tokens │  │
    └─── enable_thinking=false ────────────────────────────────────────┘  │
    single pass, temperature=0                                            │
        │                                                                 │
        ▼                                                                 │
  Parse tool calls  (_parse_response)                                     │
        │                                                                 │
        ▼                                                                 │
  Execute tools → update Memory object ────────────────────────────────►─┘
    • add_to_semantic_memory
    • add_to_episodic_memory
    • add_to_core_memory  (if including_core)
        │  (all chunks done)
        ▼
  Save state → agent_state.json, embeddings.npz,
               chunks_and_function_calls.json, final_responses.json
```

### Thinking Budget Logic

| `enable_thinking` | Generation strategy |
|---|---|
| `true` | Phase 1 up to `thinking_budget` tokens; if `</think>` absent, append early-stopping text and continue for the remaining budget. |
| `false` | Single pass at temperature=0, up to `max_new_tokens`. |

### Key CLI Options

| Flag | Description |
|---|---|
| `--agent_config` | Path to YAML agent config (required) |
| `--dataset` | Dataset name; controls which Parquet is loaded |
| `--batch_size` | Conversations processed per vLLM batch call |
| `--exclude_memory` | Disable `core`, `semantic`, and/or `episodic` memory types |
| `--rollout_label` | Suffix appended to the output directory name |
| `--save_process` | Save detailed per-chunk logs (Qwen models) |

---

## Step C — QA Evaluation

```bash
python run_qa_evaluation.py \
  --agent_config config/memalpha-qwen3-4b_agent_0.05-0.1.yaml \
  --dataset seamlessinteraction_gt \
  --batch_size 1
```

Loads the memory states from Step B, groups questions by conversation, and
POSTs them alongside the memory payloads to a running **`memory_server.py`**
instance.  The server retrieves relevant memories, builds a system prompt,
and calls the QA LLM.

### Inputs

| File / Source | Description |
|---|---|
| `config/*.yaml` | Same agent config as Step B; `external_model_url` must point to the memory server |
| `agents/…/{idx}/agent_state.json` | Memory state produced by Step B |
| `agents/…/{idx}/embeddings.npz` | Embedding matrices produced by Step B |
| Dataset QA pairs | Questions and ground-truth answers (from `ConversationCreator`) |

### Outputs (per conversation)

| File | Description |
|---|---|
| `agents/…/{idx}/results.json` | Per-question: `question`, `response`, `answer`, `step_info` |
| `agents/…/{idx}/agentic_results.json` | (`--agentic_search`) Same format; answers via iterative memory tool calls |

### Internal Flow

```
agent_state.json + embeddings.npz ──┐
Dataset QA questions ───────────────┴──▶ Build HTTP payload
                                               │
                                               ▼
                                       memory_server.py
                                               │
                          ┌────── /batch_process ──── /agentic_process ──┐
                          │                                               │
                          ▼                                               ▼
                   BM25 search                                    BM25 (top_k=2)
                   top-k=20 per type                              LLM calls search_memory
                   (token guard: reduce                           tool iteratively (≤5×)
                    if > 30k tokens)                                      │
                          │                                               │
                          ▼                                               ▼
                   construct_system_prompt                        final answer
                   <core> + <semantic top-20>
                   + <episodic top-20>
                          │
                          ▼
                   QA LLM  (Qwen3-32B / GPT)
                          │
                          ▼
                   results.json / agentic_results.json
```

### Key CLI Options

| Flag | Description |
|---|---|
| `--agent_config` | Path to YAML agent config (required) |
| `--dataset` | Dataset name; must match the one used in Step B |
| `--batch_size` | Conversations evaluated per batch |
| `--agentic_search` | Use iterative memory tool calls instead of single BM25 pass |
| `--exclude_memory` | Ablate `core`, `semantic`, and/or `episodic` memory types |
| `--force_reanswer_questions` | Re-run evaluation even if `results.json` already exists |

---

## Memory Types

| Type | Description | Enabled by default |
|---|---|---|
| `core` | High-level facts about speakers (persistent across the full conversation) | Dataset-dependent (`including_core` flag in prompts_wrt_datasource.yaml) |
| `semantic` | Factual/conceptual memories extracted from dialog | Yes |
| `episodic` | Event memories with temporal context | Yes |

Use `--exclude_memory core semantic episodic` to ablate individual types.

---

## Output Directory Naming

The output root is automatically derived from the agent config and CLI flags:

```
./agents/{agent_name}_{model_name}_{dataset}[_ext_{ext_model}][_no_thinking][_exclude_...][_tokens_{n}][_rollout_{label}]/{conversation_idx}/
```

Example: `./agents/memalpha_Qwen_Qwen3-4B_seamlessinteraction_gt_tokens_2048/0/`



## Step D — Metrics Evaluation

QA accuracy
```
python evaluate_agent_results.py --base_dir /storage/home/tuochao/Mem-alpha-audio/agents/minimal_memory_agent_qwen_converted_YuWangX_Memalpha-4B_memalpha_ext_qwen3-32b_no_thinking_tokens_2048 --output /storage/home/tuochao/Mem-alpha-audio/agents/minimal_memory_agent_qwen_converted_YuWangX_Memalpha-4B_memalpha_ext_qwen3-32b_no_thinking_tokens_2048/evaluation_metrics.json
```

Compression Ratio
```
python evaluate_compression_ratio.py --base_dir ./agents/minimal_memory_agent_qwen_converted_YuWangX_Memalpha-4B_memalpha_ext_qwen3-32b_no_thinking_tokens_2048 --dataset memalpha
```
