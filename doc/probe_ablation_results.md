# Probe-ablation results (audio memory agent)

Generated with `diagnostic/plot_probe_ablation.py --group N`, which pools the
per-question `error_probe.json` files (`probe_errors.py`) of every instance/seed
subdir of each run folder. Figures + `metrics.json` / `metrics.md` per group live in
`diagnostic/figures_ablation/group<N>_<name>/`.

```bash
for g in 1 2 3 4 5 6; do
  ~/miniconda3/envs/mem/bin/python diagnostic/plot_probe_ablation.py --group $g
done
```

## Reading the metrics

| Column | Meaning |
|---|---|
| Comp ratio | mean measured input→memory compression (from `compression.json`); the `compxN`/`_xN` in a name is the *requested* level, this column is the *actual* ratio |
| T-probe | QA answered from the **ASR transcript** — audio-stage ceiling |
| C-probe | QA answered from the **constructed memory** — after memory construction |
| S-probe | QA answered from the **retrieved subset** of memory — after retrieval |
| E2E | accuracy of the real end-to-end run |
| Self-corr (T✗→C✓) | construction recovers an answer the transcript got wrong |
| Mem-loss (T✓→C✗) | construction drops an answer the transcript had |

Cascade reading: `100−T` = audio loss, `T−C` = construction loss, `C−S` = retrieval loss.

Naming: `compxN` / `_xN` = compression level; `BG_SNRx` = additive background noise at
input SNR *x* dB; `interf_SNRx` = interfering speech at input SNR *x* dB; **omni memory** =
memory built by the Qwen3-Omni-30B audio-native model, everything else uses the cascaded
(ASR → diarization → tracking → LLM) memory agent.

**Caveat on the omni rows:** audio-native runs are probed without a transcript stage, so
T-probe, self-correction and memory-loss are undefined (`n/a` in the tables below, but the
`fig_probe_bars` / `fig_cascade` figures still draw them as 0 — treat the T bar and the
"audio loss" band of the omni variants as meaningless, not as 0%).

---

## Group 1 — SeamlessInteraction Season01: noise × compression (`group1_seamless_season01_noise_compression`)

| Variant | Comp ratio | #QA | T-probe | C-probe | S-probe | E2E | Self-corr | Mem-loss |
|---|---|---|---|---|---|---|---|---|
| Clean | 1.79x | 1056 | 94.1% | 86.6% | 68.2% | 70.2% | 2.6% | 10.1% |
| Clean_x2 | 2.72x | 264 | 94.3% | 80.3% | 56.8% | 58.7% | 2.3% | 16.3% |
| Clean_x4 | 3.89x | 264 | 94.3% | 70.1% | 56.8% | 59.5% | 1.9% | 26.1% |
| Clean_x5 | 4.35x | 264 | 94.3% | 69.7% | 50.8% | 58.0% | 0.8% | 25.4% |
| BG_SNR5 | 1.84x | 792 | 91.3% | 82.4% | 65.3% | 66.4% | 3.7% | 12.5% |
| BG_SNR5_x3 | 2.39x | 1056 | 91.3% | 75.1% | 57.3% | 61.1% | 3.0% | 19.2% |
| BG_SNR5_x4 | 3.81x | 1056 | 91.3% | 46.6% | 38.9% | 41.8% | 2.1% | 46.8% |
| BG_SNR0 | 1.75x | 1056 | 79.7% | 79.6% | 61.3% | 62.0% | 7.9% | 8.0% |
| BG_SNR0_x3 | 2.31x | 1056 | 79.9% | 65.8% | 53.3% | 56.8% | 5.5% | 19.6% |
| BG_SNR0_x4 | 3.58x | 1056 | 79.9% | 45.6% | 34.9% | 39.2% | 3.6% | 37.9% |
| interf_SNR10 | 1.61x | 1056 | 80.3% | 74.0% | 58.2% | 59.8% | 7.1% | 13.4% |
| interf_SNR10_x3 | 2.06x | 1056 | 80.3% | 68.5% | 52.7% | 54.1% | 7.5% | 19.3% |
| interf_SNR10_x4 | 2.79x | 1056 | 80.5% | 53.6% | 41.4% | 45.3% | 5.4% | 32.3% |
| interf_SNR5 | 1.60x | 1056 | 65.3% | 61.4% | 47.9% | 49.8% | 7.8% | 11.7% |
| interf_SNR5_x3 | 2.19x | 1056 | 65.2% | 53.6% | 40.7% | 45.1% | 6.0% | 17.5% |
| interf_SNR5_x4 | 3.12x | 1056 | 65.2% | 33.5% | 26.9% | 30.7% | 3.1% | 34.8% |

Observations:

- **Interfering speech is far more damaging than background noise at matched SNR.**
  Background noise at 0 dB still leaves T = 79.7%; interfering speech at 5 dB drops T to
  65.3% and at 10 dB to 80.3%. Overlapping speech corrupts both ASR and diarization,
  background noise mainly ASR.
- **Noise and compression compound multiplicatively, not additively.** Clean at ~3.9x
  compression keeps C = 70.1% (−16.5 pts vs clean); BG_SNR5 at ~3.8x collapses to
  C = 46.6% (−35.8 pts), and interf_SNR5 at 3.1x to 33.5%. The `_x4` step is a cliff in
  every noisy family — memory-loss jumps to 35–47%.
- **Compression damage is entirely in construction, not audio.** T is flat within each
  noise family (e.g. 91.3% for all BG_SNR5 variants, by construction — same audio) while C
  falls by up to 36 pts; mem-loss rises monotonically with the ratio.
- **Retrieval costs a near-constant ~10–18 pts** (`C−S`) across all conditions, so it is not
  the term that noise or compression amplifies.
- E2E tracks S closely (usually S + 1–4 pts), confirming the probe chain is well calibrated.
- Seed spread (4 seeds where available): E2E pstdev is 1–4 pts for most cells but 6 pts for
  `BG_SNR5_x3` / `BG_SNR0_x3`, so mid-compression noisy runs are the least stable.
  `Clean_x2/x4/x5` are single-instance (264 QAs) — treat those three rows as noisier than
  the rest.

## Group 2 — Season01 pipeline ablation (`group2_seamless_season01_pipeline_ablation`)

| Variant | Comp ratio | #QA | T-probe | C-probe | S-probe | E2E | Self-corr | Mem-loss |
|---|---|---|---|---|---|---|---|---|
| ASR+local_diar+globaltracking+name_extraction (Full) | 1.79x | 1056 | 94.1% | 86.6% | 68.2% | 70.2% | 2.6% | 10.1% |
| ASR+local_diar+globaltracking | 1.73x | 1056 | 42.4% | 59.9% | 47.7% | 49.6% | 27.5% | 9.9% |
| ASR+local_diar+globaltracking (new prompt) | 1.77x | 1056 | 42.4% | 84.3% | 61.7% | 63.6% | 46.0% | 4.2% |
| ASR+local_diar | 1.82x | 1056 | 44.3% | 54.3% | 41.9% | 44.5% | 23.4% | 13.4% |

Observations:

- **Name extraction is the single most valuable stage: +20.6 pts E2E** (70.2% vs 49.6%).
  Without it the transcript carries anonymous speaker IDs, so T collapses to 42.4% — most
  QAs are speaker-attribution questions that anonymous transcripts simply cannot answer.
- **The anon-speaker prompt recovers most of that gap without any extra pipeline stage:**
  same T (42.4%, identical audio front-end) but C = 84.3% vs 59.9%, E2E 63.6% vs 49.6%
  (+14 pts). Self-correction of 46.0% means the construction LLM resolves anonymous IDs to
  people from context when the prompt tells it to. Cheapest win in the table.
- Global tracking on top of local diarization is worth ~5 pts E2E (49.6% vs 44.5%).
- The `ASR+local_diar+globaltracking` (old prompt) run has the largest seed spread in this
  study (E2E pstdev 9.6 pts) — its prompt makes the construction stage unstable, which the
  new prompt also fixes (pstdev 1.9 pts).
- Here T is *below* C: the T-probe is an audio-stage ceiling only when speakers are named.

## Group 3 — PerLTQA: noise + omni memory (`group3_perltqa_noise_and_omni`)

| Variant | Comp ratio | #QA | T-probe | C-probe | S-probe | E2E | Self-corr | Mem-loss |
|---|---|---|---|---|---|---|---|---|
| Perltqa clean | 2.54x | 2436 | 85.1% | 69.0% | 63.5% | 64.2% | 3.6% | 19.6% |
| Perltqa omni memory | 2.81x | 609 | n/a | 52.4% | 40.7% | 45.3% | n/a | n/a |
| Perltqa interf 5dB | 2.48x | 609 | 72.9% | 56.0% | 51.9% | 56.7% | 3.1% | 20.0% |
| Perltqa interf 0dB | 2.25x | 609 | 37.3% | 31.0% | 28.7% | 34.5% | 5.3% | 11.5% |

Observations:

- **The cascaded agent beats omni memory by 18.9 pts E2E on clean audio** (64.2% vs 45.3%)
  at a comparable compression ratio (2.54x vs 2.81x). Omni's C-probe (52.4%) is 16.6 pts
  below the cascade's, so the deficit originates in memory construction, not retrieval.
- Omni is still ahead of the cascade under 0 dB interfering speech (45.3% vs 34.5%) but
  behind it at 5 dB (56.7%) — i.e. omni is roughly equivalent to running the cascade on
  ~2–5 dB interfered audio.
- Interfering speech is brutal on PerLTQA: 0 dB removes 48 pts of transcript accuracy
  (85.1% → 37.3%) and 30 pts of E2E.
- Retrieval loss (`C−S`) is small here (~2–5 pts) — PerLTQA memories are compact, so
  construction dominates.
- Only the clean row is a 4-seed average (2436 QAs); the other three are single runs of
  609 QAs.

## Group 4 — PerLTQA compression sweep (`group4_perltqa_compression`)

| Variant | Comp ratio | #QA | T-probe | C-probe | S-probe | E2E | Self-corr | Mem-loss |
|---|---|---|---|---|---|---|---|---|
| Perltqa clean | 2.54x | 2436 | 85.1% | 69.0% | 63.5% | 64.2% | 3.6% | 19.6% |
| Perltqa compx2 | 3.10x | 609 | 85.1% | 67.2% | 61.4% | 64.0% | 3.9% | 21.8% |
| Perltqa compx3 | 3.78x | 609 | 84.7% | 63.9% | 59.3% | 60.4% | 3.9% | 24.8% |
| Perltqa compx4 | 4.19x | 609 | 85.1% | 58.0% | 54.7% | 58.1% | 3.0% | 30.0% |
| Perltqa compx5 | 4.99x | 609 | 84.7% | 55.8% | 51.9% | 56.2% | 3.8% | 32.7% |
| Perltqa compx8 | 7.81x | 609 | 85.1% | 43.7% | 38.8% | 46.6% | 3.3% | 44.7% |

Observations:

- **Compression to ~3.1x is nearly free: −0.2 pts E2E for +22% compression.** That is the
  operating point to pick.
- Beyond that, degradation is roughly linear in the *requested* level: 3.78x → −3.8 pts,
  4.19x → −6.1, 4.99x → −8.0, 7.81x → −17.6 pts E2E vs clean.
- Every point of loss is construction loss — T is flat at ~85% and memory-loss climbs
  monotonically 19.6% → 44.7%, while self-correction stays pinned at 3–4%.
- Requested vs measured ratio diverges: `compx8` yields 7.81x but `compx2` yields 3.10x
  (the clean baseline already compresses 2.54x), so read the ratio column, not the tag.

## Group 5 — Mosaic compression sweep (`group5_mosaic_compression`)

| Variant | Comp ratio | #QA | T-probe | C-probe | S-probe | E2E | Self-corr | Mem-loss |
|---|---|---|---|---|---|---|---|---|
| Mosaic clean | 3.41x | 100 | 88.0% | 81.0% | 69.0% | 63.0% | 7.0% | 14.0% |
| Mosaic compx2 | 3.64x | 100 | 88.0% | 80.0% | 62.0% | 57.0% | 5.0% | 13.0% |
| Mosaic compx3 | 4.27x | 100 | 88.0% | 80.0% | 69.0% | 50.0% | 7.0% | 15.0% |
| Mosaic compx4 | 5.19x | 100 | 88.0% | 69.0% | 58.0% | 44.0% | 7.0% | 26.0% |
| Mosaic compx5 | 6.20x | 100 | 88.0% | 70.0% | 53.0% | 48.0% | 7.0% | 25.0% |
| Mosaic compx6 | 6.82x | 100 | 88.0% | 67.0% | 52.0% | 48.0% | 6.0% | 27.0% |
| Mosaic compx8 | 10.21x | 100 | 88.0% | 56.0% | 46.0% | 42.0% | 7.0% | 39.0% |

Observations:

- Mosaic reaches much higher ratios than PerLTQA at similar quality — up to 10.21x with
  C = 56%. Construction holds up to ~4.3x (C = 80–81%, no loss) and then breaks: 5.19x
  costs 12 pts of C, 10.21x costs 25 pts.
- **E2E is *below* S on every row here** (e.g. clean 63% vs S 69%, compx3 50% vs 69%). That
  is the opposite of every other group and points at the real-run retrieval/answering path
  losing accuracy the S-probe's oracle subset does not — worth investigating separately;
  the S-probe overstates what the Mosaic pipeline actually delivers.
- **All Mosaic rows are single runs of only 100 QAs** (±~5 pts of binomial noise, no seed
  std), so the non-monotonic dips (compx4 44% vs compx5 48%) are within noise. Treat the
  Mosaic trend, not the individual cells.

## Group 6 — Mosaic: noise + omni memory (`group6_mosaic_noise_and_omni`)

| Variant | Comp ratio | #QA | T-probe | C-probe | S-probe | E2E | Self-corr | Mem-loss |
|---|---|---|---|---|---|---|---|---|
| Mosaic clean | 3.41x | 100 | 88.0% | 81.0% | 69.0% | 63.0% | 7.0% | 14.0% |
| Mosaic omni memory | 4.60x | 100 | n/a | 52.0% | 43.0% | 41.0% | n/a | n/a |
| Mosaic interf SNR10 | 3.27x | 100 | 83.0% | 71.0% | 60.0% | 55.0% | 1.0% | 13.0% |
| Mosaic interf SNR5 | 3.35x | 100 | 72.0% | 56.0% | 44.0% | 41.0% | 4.0% | 20.0% |

Observations:

- Same verdict as PerLTQA: **omni memory (41.0%) lands 22 pts below the clean cascade
  (63.0%)** — about equal to running the cascade at 5 dB interfering speech (41.0%). Omni
  did compress harder (4.60x vs 3.41x), which explains part but not all of the gap.
- Interfering speech costs 8 pts E2E at 10 dB and 22 pts at 5 dB; the loss is split between
  audio (T 88 → 72) and construction (C 81 → 56).
- Again 100 QAs, single run each.

---

## Cross-group summary

1. **Name extraction / speaker resolution is the highest-leverage component** (+20.6 pts
   E2E), and the anon-speaker prompt recovers ~70% of that benefit for free.
2. **Interfering speech ≫ background noise** at equal SNR, and its damage is not confined
   to ASR — it propagates through construction.
3. **A free compression budget exists** (~3x on PerLTQA, ~4.3x on Mosaic); past it,
   construction memory-loss rises steeply and noise multiplies the effect.
4. **The audio-native omni memory is not yet competitive** with the cascaded agent on clean
   audio (−19 pts PerLTQA, −22 pts Mosaic), with the gap opening at the construction stage.
5. **Sample sizes are uneven.** Groups 1–2 are 4-seed / 1056-QA and trustworthy; groups 3–6
   are mostly single 100–609-QA runs. Adding seeds for Mosaic and for the noisy/omni PerLTQA
   runs is the obvious next step, along with the Mosaic E2E-below-S anomaly.

## Reproducing / extending

Groups are defined in `FOLDER_GROUPS` at the top of `diagnostic/plot_probe_ablation.py`
(`FOLDERS1`…`FOLDERS6`). Add a dict + an entry there, or pass paths ad hoc:

```bash
~/miniconda3/envs/mem/bin/python diagnostic/plot_probe_ablation.py \
  --folders '{"name": "/path/to/agents/run_folder", ...}'
```

Each run writes `fig_summary_table`, `fig_probe_bars`, `fig_cascade`,
`fig_memory_dynamics`, `fig_confusion_counts` (`.png` + `.pdf`) plus `metrics.json` and
`metrics.md` into its group directory.
