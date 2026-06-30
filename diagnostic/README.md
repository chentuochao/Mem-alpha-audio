# Memory QA Error Tracing (`diagnostic/`)

Attribute every **failed** memory-QA question to the stage of the Mem-alpha
pipeline that actually caused the failure: **transcription** (ASR + speaker
naming), **memory construction**, **retrieval**, or the **QA (response) model** —
plus gates for problems that are not the system's fault (missing data, bad gold
evidence).

The approach follows the *MemTrace* idea (`trace_err.pdf`): the decisive error is
the **earliest** operation whose correct output would have rescued the answer. We
walk the pipeline forward and find the first stage where the gold evidence
disappears.

`trace_errors_new.py` is the current tracer; it targets the **chunked** QA schema
(evidence anchored to per-chunk dialogue with string turn IDs). The run script
`run_trace_errors.sh` drives it.

---

## TL;DR

```bash
# from the repo root
bash diagnostic/run_trace_errors.sh ./agents/<run_name>/0
```

or directly:

```bash
python diagnostic/trace_errors_new.py \
    --instance_dir   ./agents/<run_name>/0 \
    --qa_file        outputs/tmp_folder_for_key_phrases_qa/merged_qa/merged_qa.jsonl \
    --transcript_root outputs/step3/vibevoice_TheBigBangTheory_predname
```

Writes `<instance_dir>/error_trace.json` and prints an attribution summary:

```
================ ERROR TRACE SUMMARY ================
questions      : 264
correct        : 205
failed         : 59
---- failure attribution ----
  transcription                      4  (7%)
  construction                      55  (93%)
=====================================================
```

---

## The strategy: a forward cascade

For each question we extract the **gold evidence turns** (from `gt_source`), then
ask, in order, where they go missing:

```
correct?  ──yes──▶  stage = correct
   │ no
GATES (not system blame)
  • question not found in QA file             ▶ gate:qa_not_found
  • evidence chunk file unavailable           ▶ gate:evidence_file_unavailable
  • no usable gold evidence                   ▶ gate:no_gold_evidence
   │
TRANSCRIPTION — is the evidence in the TRANSCRIBED dialogue? (parsed_dialog_pred.json)
   ├─ no  ▶ transcription                 (ASR or speaker-naming dropped it, BEFORE construction)
   │ yes                                   (matched only within the evidence's OWN chunk;
   │                                        sentence-wise, requires content AND speaker name)
   │
CONSTRUCTION  — is the evidence in the FULL stored memory? (agent_state.json)
   ├─ no  ▶ construction                  (transcript had it, store doesn't)
   │ yes
RETRIEVAL     — is the evidence in the RETRIEVED memory? (results.json)
   ├─ no  ▶ retrieval                     (in the store, not surfaced to the QA model)
   │ yes
RESPONSE      ▶ response                  (evidence was shown, model still wrong)
```

**Unparseable answers are NOT gated.** If the model's response has no extractable
choice (e.g. `"not sure"`, no `\boxed{X}`), the prediction is recorded as the
sentinel **`"not_parse"`**, counted as a failure, and traced through the full
cascade (so we still learn whether the evidence was even available). This differs
from the older `trace_errors_clean.py`, which gated these as
`gate:no_parseable_answer`.

Gates are excluded from "system blame" — they flag data problems so they don't
pollute the transcription/construction/retrieval/response counts.

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
first to the full store, then to the retrieved set, with `COVERAGE_TAU` as the
boundary each time. Construction is a single bucket (no extraction/update split).

### The transcription stage (sentence-wise, same-chunk)

The transcription check is special, because a **gold** evidence turn is often a
speaker-MERGE of several utterances, while the **pred** transcript is finely
segmented. Scoring the whole merged turn against a single pred turn under-counts
(no fragment covers it; the merge even invents cross-sentence n-grams that match
nothing). So this stage uses `matching.present_sentencewise`:

1. **Scope to the evidence's OWN chunk** — candidates come only from that chunk's
   `parsed_dialog_pred.json` (`trace_errors_new.evidence_chunks` →
   `TranscriptLoader.records_for_chunks`), not the whole episode.
2. **Speaker filter** — keep only pred turns whose speaker fuzzily matches the gold
   speaker (`matching.speaker_match`, e.g. pred `Sheldon` matches gold
   `sheldon_cooper`). A gold turn with no same-speaker pred turn is an automatic
   miss → catches speaker mis-attribution / dropped speakers.
3. **Split into sentences**, drop trivially short ones (< `SENT_MIN_CONTENT_TOKENS`
   content words; whole-utterance fallback if all are trivial).
4. **Match each sentence** against the best single same-speaker candidate; the turn
   is preserved if the content-token-weighted fraction of found sentences ≥
   `COVERAGE_TAU` (the fact-bearing sentence outweighs filler).

For each **missed** sentence the record carries a `reason`:

| `reason` | meaning | extra fields |
|---|---|---|
| `low_lex` | no turn (even ignoring speaker) clears `LEX_TAU` | `best_any_score` |
| `speaker_mismatch` | a turn clears `LEX_TAU`, but under a different speaker — content is in the transcript, attributed wrongly | `found_under_speaker`, `any_speaker_score` |

### Lexical matching (the default signal)

`matching.lexical_score(evidence, candidate)` = fraction of the turn's content-word
**n-grams** found in the candidate, using the longest n the turn supports
(trigrams → bigrams → single word). It is computed on the **utterance only**
(speaker prefix stripped) and is **asymmetric** (divided by the *evidence's*
n-gram count), so a long memory unit cannot match on a few scattered words.

- Verbatim quote → ~1.0
- Unrelated unit sharing a few words → ~0.0
- **Fails on paraphrase / synonyms / reordering / morphology** (e.g. `question` vs
  `questions`) — that is what the embedding and LLM-judge layers are for. With
  lexical only, "missing" can be a false negative; enable the extra layers to
  tighten it.

---

## Thresholds (in `matching.py`)

| Constant | Default | Meaning |
|---|---|---|
| `LEX_TAU` | `0.6` | per-turn/sentence lexical: ≥60% of the n-grams must appear in a unit |
| `EMB_TAU` | `0.55` | per-turn embedding cosine cutoff |
| `COVERAGE_TAU` | `0.5` | per-question (and per-turn sentence weight): ≥50% must match for "present" |
| `SENT_MIN_CONTENT_TOKENS` | `2` | transcription: drop sentences with fewer content words |

Construction-vs-retrieval boundary truth table (`COVERAGE_TAU = 0.5`):

| store coverage | retrieved coverage | attribution |
|---|---|---|
| `< 0.5` | — | construction |
| `≥ 0.5` | `< 0.5` | retrieval |
| `≥ 0.5` | `≥ 0.5` | response |

Tuning: raise `LEX_TAU` for stricter matching (more construction calls); raise
`COVERAGE_TAU` toward 1.0 to require (nearly) all evidence present.

---

## Inputs

| Input | Source | Used for |
|---|---|---|
| `results.json` | `--instance_dir` | model response, gold answer, `retrieved_memory` |
| `agent_state.json` | `--instance_dir` | the full stored memory (`core` / `episodic` / `semantic`) |
| QA file (`merged_qa.jsonl`) | `--qa_file` | questions, options, gold answer, `gt_source.sources[].evidence_turns` |
| transcript root | `--transcript_root` | holds **both** the gold evidence chunks (`parsed_dialog_gt.json`) and the transcribed dialogue (`parsed_dialog_pred.json`), as `<root>/<episode>/CHUNK_*/...` |

There is **no `--dialog_root`** (unlike `trace_errors_clean.py`): the gold evidence
chunks and the pred transcript live side by side under `--transcript_root`, so both
sides of the transcription match — and the evidence text itself — come from there.

### Chunked QA schema

```jsonc
"gt_source": {
  "slot": "S01E02_FACT_005",
  "target_speaker": "sheldon_cooper",
  "sources": [
    {
      "file": "TheBigBangTheory.Season01.Episode02/CHUNK_1/parsed_dialog_gt.json",
      "evidence_turns": ["S01E02_C001_T003"]   // S<season>E<episode>_C<chunk>_T<turn-in-chunk>
    }
  ]
}
```

- `file` is a path **relative to `--transcript_root`**; its first two components are
  the episode and chunk (used to scope the transcription match).
- each `evidence_turns` entry is a string ID whose trailing `T<NNN>` is the 0-based
  turn index within that chunk (`trace_errors_new._turn_index`).

### Migrating old (whole-episode) QA → chunked

Old QA used whole-episode files and **integer** turn indices
(`file: "...Episode01.json"`, `evidence_turns: [58, 59]`). Convert it with:

```bash
python diagnostic/convert_qa_to_chunked.py \
    --qa_file        outputs/tmp_folder_for_95_qs/merged_95.jsonl \
    --old_dialog_root outputs/bazinga/TheBigBangTheory/Season1 \
    --chunk_root     outputs/step3/vibevoice_TheBigBangTheory_predname
# -> outputs/tmp_folder_for_95_qs/merged_95_chunked.jsonl
```

It locates each old turn inside the chunk files by **text** (normalized substring,
lexical fallback), since the chunked dialogue is re-segmented (no 1:1 index). One
old source can fan out into several per-chunk sources. Turns it can't resolve (e.g.
the session-based `S1_main/session_*.json` layout, whose dialogs aren't on disk) are
dropped and reported in the summary.

---

## Output: `error_trace.json`

```jsonc
{
  "instance_dir": "...",
  "matcher": { "lexical": true, "embedding": false, "llm_judge": false,
               "thresholds": { "lexical": 0.6, "embedding": 0.55 } },
  "summary": { "total": 264, "correct": 205, "failed": 59,
               "attribution": { "transcription": 4, "construction": 55 } },
  "findings": [
    {
      "question": "...",
      "options": { "A": "...", "B": "..." },
      "gold": "B", "pred": "not_parse", "correct": false,
      "stage": "transcription",
      "evidence": ["leonard_hofstadter: ...", "..."],
      "detail": {
        "transcript_coverage": {
          "coverage": 0.33, "matched": 1, "total": 3,
          "matches": [
            { "turn": "leonard_hofstadter: ...", "found": false, "method": "sentencewise",
              "score": 0.0, "speaker_records": 3,
              "sentences": [
                { "sentence": "You are going to march yourself over there ...",
                  "found": false, "score": 0.0, "weight": 6,
                  "reason": "speaker_mismatch",
                  "found_under_speaker": "Sheldon", "any_speaker_score": 1.0 }
              ] }
          ]
        },
        "chunks": ["TheBigBangTheory.Season01.Episode02/CHUNK_8"]
      }
    }
  ]
}
```

For construction/retrieval failures each matched turn records **which memory unit**
it matched (`memory_id` hash, `memory_type` = core/episodic/semantic), **how**
(`method` = lex/emb/llm), and the `score`. For retrieval failures,
`evidence_episodic_ranks` vs `retrieved_episodic_count` shows *why* it was dropped
(e.g. evidence at rank 234, cutoff 20).

---

## CLI options (`trace_errors_new.py`)

| Flag | Default | Description |
|---|---|---|
| `--instance_dir` | (a run under `agents/`) | dir with `results.json` / `agent_state.json` |
| `--qa_file` | `outputs/tmp_folder_for_key_phrases_qa/merged_qa/merged_qa.jsonl` | chunked QA jsonl |
| `--transcript_root` | `outputs/step3/vibevoice_TheBigBangTheory_predname` | root holding the gold evidence chunks + pred transcript |
| `--min_turn_words` | `0` | drop evidence turns whose utterance has fewer than N words |
| `--out` | `<instance_dir>/error_trace.json` | output path |

Defaults are relative to the **repo root**, so run from there.

To enable the stronger matcher layers:

```bash
export OPENAI_API_KEY=...       # embedding layer  (text-embedding-3-small)
export OPENROUTER_API_KEY=...   # LLM entailment judge (construction stage)
# (also reads a .env if python-dotenv is installed)
```

### Launching the local LLM backend (for the judge / probe)

The LLM judge (and `probe_errors.py`'s QA server) talk to a local vLLM endpoint on
`:8002`. `run_trace_errors.sh` already points there (`QWEN_URL=http://localhost:8002/v1`),
so it will **hang** until the server is up. On a SLURM allocation, bring it up on the
**same node** as your job:

```bash
# 1) open a second shell ON THE SAME NODE as your running job
srun --pty --overlap --jobid <JOBID> bash

# 2) in that second shell, start vLLM (serves the Qwen judge on :8002)
./launch_vllm.sh

# 3) check vLLM is serving
curl http://localhost:8002/v1/models

# 4) (only for probe_errors.py) start the memory QA server, pointed at vLLM
QWEN_URL="http://localhost:8002/v1" python memory_server.py --port 5005 > server_outputs.log 2>&1 &

# 5) back in the FIRST shell, sanity-check the QA endpoint
curl http://127.0.0.1:5005/batch_process
```

Once `:8002` answers, `bash diagnostic/run_trace_errors.sh <INSTANCE_DIR>` runs with
the judge enabled. If you don't want the judge, run lexical-only by unsetting the key:
`env -u OPENROUTER_API_KEY python diagnostic/trace_errors_new.py --instance_dir ...`.

<!-- ---

## Code layout

```
diagnostic/
├── matching.py             # similarity: tokenization, n-gram lexical_score,
│                           #   EmbeddingMatcher, LLMJudge, present(),
│                           #   present_sentencewise(), speaker_match(),
│                           #   evidence_rank(), thresholds
├── data_utils.py           # loading & parsing: fix_space_in_text, extract_choice,
│                           #   gold_letter, load_qa, TranscriptLoader
│                           #   (episode_records / chunk_records / records_for_chunks),
│                           #   memory_records, retrieved_records
├── trace_errors_new.py     # the cascade orchestrator + CLI (chunked schema):
│                           #   ChunkDialogResolver, evidence_texts/chunks, trace()
├── convert_qa_to_chunked.py# migrate old whole-episode QA -> chunked schema
├── run_trace_errors.sh     # convenience runner for trace_errors_new.py
└── trace_errors_clean.py   # legacy tracer for the old whole-episode QA schema
```

Dependency direction is acyclic: `matching` → `data_utils` → `trace_errors_new`.

### Text normalization
The dialog source uses Penn-Treebank spacing (`"ca n't"`, `"they 're"`, `"Hi ."`)
while the memory store uses joined forms (`"can't"`, `"Hi."`).
`data_utils.fix_space_in_text` reconciles them **at load time** (evidence turns,
transcript turns, and memory units), so all matcher layers see consistent text.

---

## Alternative: behavioral probe (`probe_errors.py`)

The cascade attributes by **matching** — does the gold evidence string still
*appear* in the memory/transcript? That is correlational, sensitive to
`LEX_TAU`/`COVERAGE_TAU`, and blind to paraphrase (memory construction *rewrites*
content, so matching over-counts construction errors).

`probe_errors.py` attributes **behaviorally / causally** instead: it re-runs the
real QA model (same `/batch_process` server) on curated contexts and asks whether
the **answer flips**. It needs no matching thresholds and is immune to paraphrase.

It uses **exact provenance** rather than fuzzy localization: every memory unit is
created by a `new_memory_insert` recorded in `chunks_and_function_calls.json`, and
the returned unit id (e.g. `cd61`) is the *same* id stored in `agent_state.json`.
So "the memory constructed from turn T" = the **final** stored units whose insert
lives in the chunk that contained T (chunks are non-overlapping).

Per **failed** question, up to three oracle QA re-answers:

```
C-probe : final memory units traced to the evidence chunk(s)   ── QA ──▶ correct?
   ├─ no  ▶ T-probe: the matched transcript turns (raw ASR)    ── QA ──▶ correct?
   │          ├─ yes ▶ construction   (transcript had it, store lost it)
   │          └─ no  ▶ transcription  (ASR/naming dropped it upstream)
   └─ yes ▶ rescue: actual retrieved_memory ∪ evidence units   ── QA ──▶ correct?
              ├─ yes ▶ retrieval   (store had it, retriever didn't surface it)
              └─ no  ▶ response    (it was shown, model still wrong)
```

```bash
# memory server must be running (same one used to produce results.json)
bash diagnostic/run_probe_errors.sh memory_result/<run_name>/0 http://127.0.0.1:5005/batch_process
# writes <instance_dir>/error_probe.json
```

The two tools are complementary: run the matching cascade as cheap triage, then the
behavioral probe to causally confirm the suspect buckets.
-->

--- 

## Known limitations

- **Lexical-only misses paraphrase / synonyms / morphology** (e.g. `question` vs
  `questions`, since matching is exact-token n-gram with no stemming) → can
  over-count construction/transcription misses. Enable embedding + LLM judge to
  mitigate, or add a stemmer in `matching._tokens`.
- **Turn-level aggregation**: a question with several gold turns can read as a
  transcription failure even when the *answer-bearing* turn was preserved, if other
  turns miss (coverage < `COVERAGE_TAU`). The `sentences`/`reason` detail makes this
  visible per turn.
- **Same-chunk scoping** assumes the gold evidence chunk has a corresponding pred
  chunk with the content; if gold/pred chunk boundaries drift, a turn near a
  boundary could read as a miss even though it landed in the adjacent pred chunk.
- **Threshold sensitivity**: cases near `LEX_TAU`/`COVERAGE_TAU` can flip buckets.
- **Episodic ranks are approximate**: from the lexical matcher's ordering, not the
  exact embedding ranking the memory server used at retrieval time.
- **Attribution is correlational, not causal**: it locates where evidence
  disappears but does not *prove* it. The rigorous confirmation is the
  counterfactual rescue in `probe_errors.py`.
```
