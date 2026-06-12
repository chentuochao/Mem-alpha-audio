# Memory QA Error Tracing (`diagnostic/`)

Attribute every **failed** memory-QA question to the stage of the Mem-alpha
pipeline that actually caused the failure: **memory construction**, **retrieval**,
or the **QA (response) model** — plus gates for problems that are not the system's
fault (bad gold answer, judge/parse error, missing data).

The approach follows the *MemTrace* idea (`trace_err.pdf`): the decisive error is
the **earliest** operation whose correct output would have rescued the answer. We
walk the pipeline forward and find the first stage where the gold evidence
disappears.

---

## TL;DR

```bash
# from the repo root
python diagnostic/trace_errors.py \
    --instance_dir memory_result/<run_name>/0 \
    --qa_file      outputs/tmp_folder_for_95_qs/merged_95.jsonl \
    --dialog_root  outputs/bazinga/TheBigBangTheory/Season1
```

Writes `<instance_dir>/error_trace.json` and prints an attribution summary:

```
================ ERROR TRACE SUMMARY ================
questions      : 95
correct        : 64
failed         : 31
---- failure attribution ----
  gate:evidence_file_unavailable     8  (26%)
  construction:extraction            1  (3%)
  construction:update/deletion       4  (13%)
  retrieval                         16  (52%)
  response                           2  (6%)
=====================================================
```

---

## The strategy: a forward cascade

For each question we extract the **gold evidence turns** (from `gt_source`), then
ask, in order, where they go missing:

```
correct?  ──yes──▶  stage = correct
   │ no
GATES
  • answer not parseable (no \boxed letter)   ▶ gate:no_parseable_answer
  • question not found in QA file             ▶ gate:qa_not_found
  • evidence dialog file unavailable          ▶ gate:evidence_file_unavailable
  • no usable gold evidence                   ▶ gate:no_gold_evidence
   │
CONSTRUCTION  — is the evidence in the FULL stored memory? (agent_state.json)
   ├─ no  ▶ construction:extraction       (agent saw it, never wrote it)
   │      ▶ construction:update/deletion  (it was written, then lost)
   │ yes
RETRIEVAL     — is the evidence in the RETRIEVED memory? (results.json)
   ├─ no  ▶ retrieval                     (in the store, not surfaced to the QA model)
   │ yes
RESPONSE      ▶ response                  (evidence was shown, model still wrong)
```

Gates are excluded from "system blame" — they flag data/eval problems so they
don't pollute the construction/retrieval/response counts.

### How "is the evidence present?" is decided

Evidence is a **set of turns**. We test each turn against a memory blob, then
require enough turns to match.

1. **Per-turn match** (`matching._turn_present`) — a turn matches if **any** layer fires:
   - **lexical** (always on): n-gram phrase containment ≥ `LEX_TAU`
   - **embedding** (if `OPENAI_API_KEY` set): cosine ≥ `EMB_TAU` (`text-embedding-3-small`)
   - **LLM judge** (if `OPENROUTER_API_KEY` set): entailment "yes" on the top lexical candidates
2. **Per-question coverage** (`matching.present`) — evidence is "present" if
   `matched_turns / total_turns ≥ COVERAGE_TAU`.

The **construction vs retrieval vs response** decision is just `present()` applied
first to the full store, then to the retrieved set, with the same `COVERAGE_TAU`
as the boundary each time.

### Lexical matching (the default signal)

`matching.lexical_score(evidence, candidate)` = fraction of the turn's content-word
**n-grams** found in the candidate, using the longest n the turn supports
(trigrams → bigrams → single word). It is computed on the **utterance only**
(speaker prefix stripped) and is **asymmetric** (divided by the *evidence's*
n-gram count), so a long memory unit cannot match on a few scattered words.

- Verbatim quote → ~1.0
- Unrelated unit sharing a few words → ~0.0
- **Fails on paraphrase / synonyms / reordering** — that is what the embedding and
  LLM-judge layers are for. With lexical only, "missing from store" can be a false
  negative for paraphrased `core` content; enable the extra layers to tighten it.

---

## Thresholds (in `matching.py`)

| Constant | Default | Meaning |
|---|---|---|
| `LEX_TAU` | `0.6` | per-turn lexical: ≥60% of the turn's n-grams must appear in a unit |
| `EMB_TAU` | `0.55` | per-turn embedding cosine cutoff |
| `COVERAGE_TAU` | `0.5` | per-question: ≥50% of evidence turns must match for "present" |

Construction-vs-retrieval boundary truth table (`COVERAGE_TAU = 0.5`):

| store coverage | retrieved coverage | attribution |
|---|---|---|
| `< 0.5` | — | construction |
| `≥ 0.5` | `< 0.5` | retrieval |
| `≥ 0.5` | `≥ 0.5` | response |

Tuning: raise `LEX_TAU` for stricter per-turn matching (more construction calls);
raise `COVERAGE_TAU` toward 1.0 to require (nearly) all evidence present.

---

## Inputs

| Input | Source | Used for |
|---|---|---|
| `results.json` | `--instance_dir` | model response, gold answer, `retrieved_memory` |
| `agent_state.json` | `--instance_dir` | the full stored memory (`core` / `episodic` / `semantic`) |
| `chunks_and_function_calls.json` | `--instance_dir` | *optional* — splits construction into extraction vs update/deletion |
| QA file (`merged_95.jsonl`) | `--qa_file` | questions, options, gold answer, `gt_source.evidence_turns` |
| dialog transcripts | `--dialog_root` | resolves `evidence_turns` → actual turn text |

Notes:
- `chunks_and_function_calls.json` is only read on the construction branch. Without
  it, construction errors all collapse to `construction:extraction`.
- `--dialog_root` must point at transcripts with **real speaker names** (matching
  the memory store), not the anonymized `P0001` set. Multiple roots are allowed;
  the first listed wins on basename collisions.
- Each question's `evidence_turns` index into **its own** source file; the
  `DialogResolver` finds it by basename. Files it can't locate become
  `gate:evidence_file_unavailable`.

---

## Output: `error_trace.json`

```jsonc
{
  "instance_dir": "...",
  "matcher": { "lexical": true, "embedding": false, "llm_judge": false,
               "thresholds": { "lexical": 0.6, "embedding": 0.55 } },
  "summary": { "total": 95, "correct": 64, "failed": 31,
               "attribution": { "retrieval": 16, "construction:update/deletion": 4, ... } },
  "findings": [
    {
      "question": "...",
      "options": { "A": "...", "B": "..." },
      "gold": "A", "pred": "C", "correct": false,
      "stage": "retrieval",
      "evidence": ["leonard_hofstadter: ...", "..."],
      "detail": {
        "store_coverage":     { "coverage": 1.0,  "matched": 4, "total": 4, "matches": [ ... ] },
        "retrieved_coverage": { "coverage": 0.25, "matched": 1, "total": 4, "missing": [ ... ] },
        "best_episodic_rank": 2,
        "evidence_episodic_ranks": [ {"turn": "...", "rank": 234, "memory_id": "58e9"} ],
        "retrieved_episodic_count": 20
      }
    }
  ]
}
```

Each matched turn records **which memory unit** it matched (`memory_id` hash,
`memory_type` = core/episodic/semantic), **how** (`method` = lex/emb/llm), and the
`score`, so every attribution is auditable. For retrieval failures,
`evidence_episodic_ranks` vs `retrieved_episodic_count` shows *why* it was dropped
(e.g. evidence at rank 234, cutoff 20).

---

## CLI options

| Flag | Default | Description |
|---|---|---|
| `--instance_dir` | (a run under `memory_result/`) | dir with `results.json` / `agent_state.json` / `chunks_and_function_calls.json` |
| `--qa_file` | `outputs/tmp_folder_for_95_qs/merged_95.jsonl` | QA jsonl with `gt_source.evidence_turns` |
| `--dialog_root` | `outputs/bazinga/TheBigBangTheory/Season1` | dir(s) searched recursively for evidence dialog files |
| `--min_turn_words` | `3` | drop evidence turns whose utterance has fewer than N words (set `0` to disable) |
| `--out` | `<instance_dir>/error_trace.json` | output path |

Defaults are relative to the **repo root**, so run from there. Works both as a
script (`python diagnostic/trace_errors.py`) and as a module
(`python -m diagnostic.trace_errors`).

To enable the stronger matcher layers:

```bash
export OPENAI_API_KEY=...       # embedding layer  (text-embedding-3-small)
export OPENROUTER_API_KEY=...   # LLM entailment judge
# (also reads a .env if python-dotenv is installed)
```

---

## Code layout

```
diagnostic/
├── matching.py     # similarity: tokenization, n-gram lexical_score,
│                   #   EmbeddingMatcher, LLMJudge, present(), evidence_rank(),
│                   #   thresholds (LEX_TAU / EMB_TAU / COVERAGE_TAU)
├── data_utils.py   # loading & parsing: fix_space_in_text, extract_choice,
│                   #   gold_letter, load_qa, DialogResolver, evidence_texts,
│                   #   memory_records, retrieved_records
└── trace_errors.py # the cascade orchestrator + CLI (construction_subtype, trace)
```

Dependency direction is acyclic: `matching` → `data_utils` → `trace_errors`.

### Text normalization
The dialog source uses Penn-Treebank spacing (`"ca n't"`, `"they 're"`, `"Hi ."`)
while the memory store uses joined forms (`"can't"`, `"Hi."`).
`data_utils.fix_space_in_text` reconciles them **at load time** (applied to
evidence turns, memory units, and the construction trace), so all matcher layers
see consistent text.

---

## Known limitations

- **Lexical-only misses paraphrase/synonyms** → can over-count construction
  errors on summarized `core` content. Enable embedding + LLM judge to mitigate.
- **Threshold sensitivity**: cases near `LEX_TAU`/`COVERAGE_TAU` can flip buckets.
- **Episodic ranks are approximate**: computed from the lexical matcher's ordering,
  not the exact embedding ranking the memory server used at retrieval time.
- **Attribution is correlational, not yet causal**: it locates where evidence
  disappears but does not *prove* it. The rigorous confirmation is a counterfactual
  rescue (inject the correct value at the suspected stage, re-answer, see if it
  flips) — not yet implemented here.
```
