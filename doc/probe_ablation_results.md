# Probe-ablation results (audio memory agent)

Regenerated on 2026-09-03 with `diagnostic/plot_probe_ablation.py --group N`.
The script pools the per-question `error_probe.json` files from every instance/seed
subdirectory of each configured run. Figures and machine-readable tables are stored in
`diagnostic/figures_ablation/group<N>_<name>/`.

```bash
for g in 1 2 3 4 5 6 7 8; do
  ~/miniconda3/envs/mem/bin/python diagnostic/plot_probe_ablation.py --group "$g"
done
```

## Reading the metrics

| Column | Meaning |
|---|---|
| Comp ratio | Measured input-to-memory compression from `compression.json`; the name records the requested level |
| T-probe | QA answered from the ASR transcript—the audio-stage ceiling |
| C-probe | QA answered from the constructed memory |
| S-probe | QA answered from the retrieved subset of memory |
| E2E | Accuracy of the original end-to-end run |
| Self-corr / T✗ | `P(C correct | T wrong or unsure)`: corrected T-stage failures divided by all T-stage failures |
| Mem-loss / T✓ | `P(C wrong or unsure | T correct)`: lost T-stage successes divided by all T-stage successes |

The final two columns are conditional rates, not percentages of all QA samples. Their raw
counts are the off-diagonal cells in `fig_confusion_counts`.

Cascade reading: `100−T` is audio loss, `T−C` is net construction loss, and `C−S` is net
retrieval loss. These differences are useful summaries but do not expose bidirectional
answer flips; the conditional self-correction and memory-loss rates do.

Naming: `compxN`/`xN` is the requested compression level; `BG_SNRx` is additive
background noise; `interf_SNRx` is interfering speech. Audio-native Omni runs have no
transcript stage, so T-probe, self-correction, and memory loss are undefined (`n/a`).

---

## Group 1 — SeamlessInteraction Season01: noise × compression

Output: `group1_seamless_season01_noise_compression`

![Season01 relative self-correction and construction-loss curves](../diagnostic/figures_ablation/group1_seamless_season01_noise_compression/fig_noise_compression_relative.png)

The x-axis uses the measured compression ratio from `compression.json`, not the
requested `xN` label.

| Variant | Comp ratio | #QA | T-probe | C-probe | S-probe | E2E | Self-corr / T✗ | Mem-loss / T✓ |
|---|---|---|---|---|---|---|---|---|
| Clean | 1.79x | 1056 | 94.1% | 86.6% | 68.2% | 70.2% | 43.5% | 10.8% |
| Clean_x2 | 2.72x | 264 | 94.3% | 80.3% | 56.8% | 58.7% | 40.0% | 17.3% |
| Clean_x4 | 3.89x | 264 | 94.3% | 70.1% | 56.8% | 59.5% | 33.3% | 27.7% |
| Clean_x5 | 4.35x | 264 | 94.3% | 69.7% | 50.8% | 58.0% | 13.3% | 26.9% |
| BG_SNR5 | 1.84x | 792 | 91.3% | 82.4% | 65.3% | 66.4% | 42.0% | 13.7% |
| BG_SNR5_x3 | 2.39x | 1056 | 91.3% | 75.1% | 57.3% | 61.1% | 34.8% | 21.1% |
| BG_SNR5_x4 | 3.81x | 1056 | 91.3% | 46.6% | 38.9% | 41.8% | 23.9% | 51.2% |
| BG_SNR0 | 1.75x | 1056 | 79.7% | 79.6% | 61.3% | 62.0% | 38.8% | 10.0% |
| BG_SNR0_x3 | 2.31x | 1056 | 79.9% | 65.8% | 53.3% | 56.8% | 27.4% | 24.5% |
| BG_SNR0_x4 | 3.58x | 1056 | 79.9% | 45.6% | 34.9% | 39.2% | 17.9% | 47.4% |
| interf_SNR10 | 1.61x | 1056 | 80.3% | 74.0% | 58.2% | 59.8% | 36.1% | 16.7% |
| interf_SNR10_x3 | 2.06x | 1056 | 80.3% | 68.5% | 52.7% | 54.1% | 38.0% | 24.1% |
| interf_SNR10_x4 | 2.79x | 1056 | 80.5% | 53.6% | 41.4% | 45.3% | 27.7% | 40.1% |
| interf_SNR5 | 1.60x | 1056 | 65.3% | 61.4% | 47.9% | 49.8% | 22.4% | 18.0% |
| interf_SNR5_x3 | 2.19x | 1056 | 65.2% | 53.6% | 40.7% | 45.1% | 17.1% | 26.9% |
| interf_SNR5_x4 | 3.12x | 1056 | 65.2% | 33.5% | 26.9% | 30.7% | 9.0% | 53.3% |

Key observations:

- Interfering speech is more damaging than background noise at matched SNR.
- Compression damage appears primarily during construction: T stays fixed within each
  audio condition while C falls as compression increases.
- The strongest noisy/compressed settings lose roughly half of the initially answerable
  questions during construction: BG-SNR5 x4 has 51.2% memory loss, and interfering-SNR5
  x4 has 53.3%.
- Retrieval commonly removes another 10–18 percentage points between C and S.

## Group 2 — Season01 pipeline ablation

Output: `group2_seamless_season01_pipeline_ablation`

| Variant | Comp ratio | #QA | T-probe | C-probe | S-probe | E2E | Self-corr / T✗ | Mem-loss / T✓ |
|---|---|---|---|---|---|---|---|---|
| ASR+local_diarization+globaltracking+name_extraction(Full) | 1.79x | 1056 | 94.1% | 86.6% | 68.2% | 70.2% | 43.5% | 10.8% |
| ASR+local_diarization+globaltracking | 1.73x | 1056 | 42.4% | 59.9% | 47.7% | 49.6% | 47.7% | 23.4% |
| ASR+local_diarization+globaltracking (new prompt) | 1.77x | 1056 | 42.4% | 84.3% | 61.7% | 63.6% | 79.9% | 9.8% |
| ASR+local_diarization | 1.82x | 1056 | 44.3% | 54.3% | 41.9% | 44.5% | 42.0% | 30.3% |

Key observations:

- Explicit name extraction remains the strongest pipeline component: the full system is
  20.6 E2E points above global tracking without name extraction.
- The anonymous-speaker prompt corrects 79.9% of questions that its anonymous transcript
  cannot answer while losing only 9.8% of transcript-answerable questions.
- Global tracking improves E2E by about five points over local diarization alone.

## Group 3 — PerLTQA: noise and audio-native memory

Output: `group3_perltqa_noise_and_omni`

| Variant | Comp ratio | #QA | T-probe | C-probe | S-probe | E2E | Self-corr / T✗ | Mem-loss / T✓ |
|---|---|---|---|---|---|---|---|---|
| Perltqa clean | 2.54x | 2436 | 85.1% | 69.0% | 63.5% | 64.2% | 23.9% | 23.0% |
| Perltqa omni memory | 2.81x | 609 | n/a | 52.4% | 40.7% | 45.3% | n/a | n/a |
| Perltqa interf 5dB | 2.48x | 609 | 72.9% | 56.0% | 51.9% | 56.7% | 11.5% | 27.5% |
| Perltqa interf 0dB | 2.25x | 609 | 37.3% | 31.0% | 28.7% | 34.5% | 8.4% | 30.8% |

Key observations:

- The clean cascade exceeds audio-native Omni by 18.9 E2E points at comparable measured
  compression.
- Interfering speech sharply reduces T-stage answerability: 85.1% clean, 72.9% at 5 dB,
  and 37.3% at 0 dB.
- Construction recovers fewer T failures as noise increases, while conditional memory
  loss rises from 23.0% clean to 30.8% at 0 dB.

## Group 4 — PerLTQA clean compression sweep

Output: `group4_perltqa_compression`

| Variant | Comp ratio | #QA | T-probe | C-probe | S-probe | E2E | Self-corr / T✗ | Mem-loss / T✓ |
|---|---|---|---|---|---|---|---|---|
| Perltqa clean | 2.54x | 2436 | 85.1% | 69.0% | 63.5% | 64.2% | 23.9% | 23.0% |
| Perltqa compx2 | 3.10x | 609 | 85.1% | 67.2% | 61.4% | 64.0% | 26.4% | 25.7% |
| Perltqa compx3 | 3.78x | 609 | 84.7% | 63.9% | 59.3% | 60.4% | 25.8% | 29.3% |
| Perltqa compx4 | 4.19x | 609 | 85.1% | 58.0% | 54.7% | 58.1% | 19.8% | 35.3% |
| Perltqa compx5 | 4.99x | 609 | 84.7% | 55.8% | 51.9% | 56.2% | 24.7% | 38.6% |
| Perltqa compx8 | 7.81x | 609 | 85.1% | 43.7% | 38.8% | 46.6% | 22.0% | 52.5% |

Key observations:

- The x2 run is effectively free relative to the pooled clean baseline: 64.0% versus
  64.2% E2E.
- Conditional memory loss rises from 23.0% at baseline to 52.5% at x8, while T remains
  approximately 85%; compression therefore acts mainly on construction.
- Measured compression differs from the requested label, so comparisons should use the
  `Comp ratio` column.

## Group 5 — Mosaic clean compression sweep

Output: `group5_mosaic_compression`

| Variant | Comp ratio | #QA | T-probe | C-probe | S-probe | E2E | Self-corr / T✗ | Mem-loss / T✓ |
|---|---|---|---|---|---|---|---|---|
| Mosaic clean | 3.41x | 100 | 88.0% | 81.0% | 69.0% | 63.0% | 58.3% | 15.9% |
| Mosaic compx2 | 3.64x | 100 | 88.0% | 80.0% | 62.0% | 57.0% | 41.7% | 14.8% |
| Mosaic compx3 | 4.27x | 100 | 88.0% | 80.0% | 69.0% | 50.0% | 58.3% | 17.0% |
| Mosaic compx4 | 5.19x | 100 | 88.0% | 69.0% | 58.0% | 44.0% | 58.3% | 29.5% |
| Mosaic compx5 | 6.20x | 100 | 88.0% | 70.0% | 53.0% | 48.0% | 58.3% | 28.4% |
| Mosaic compx6 | 6.82x | 100 | 88.0% | 67.0% | 52.0% | 48.0% | 50.0% | 30.7% |
| Mosaic compx8 | 10.21x | 100 | 88.0% | 56.0% | 46.0% | 42.0% | 58.3% | 44.3% |

Key observations:

- C remains near 80% through 4.27x measured compression, then declines at stronger
  settings.
- Conditional memory loss increases from 15.9% at baseline to 44.3% at x8.
- E2E is below S for every row, indicating an additional gap between the probe's curated
  retrieved subset and the actual retrieval/response path.
- Every row has only 100 questions, so small non-monotonic differences should not be
  overinterpreted.

## Group 6 — Mosaic: cascade versus audio-native memory

Output: `group6_mosaic_noise_and_omni`

| Variant | Comp ratio | #QA | T-probe | C-probe | S-probe | E2E | Self-corr / T✗ | Mem-loss / T✓ |
|---|---|---|---|---|---|---|---|---|
| Mosaic clean | 3.41x | 100 | 88.0% | 81.0% | 69.0% | 63.0% | 58.3% | 15.9% |
| Mosaic omni memory | 4.60x | 100 | n/a | 52.0% | 43.0% | 41.0% | n/a | n/a |
| Mosaic omni memory history 5 | 6.48x | 100 | n/a | 42.0% | 38.0% | 34.0% | n/a | n/a |

Key observations:

- Standard audio-native Omni is 22 E2E points below the clean cascade; history-5 is a
  further seven points lower.
- History-5 compresses substantially harder than ordinary Omni (6.48x versus 4.60x), so
  this comparison combines the history intervention with a different achieved memory
  budget.
- The history-5 C-to-S gap is only four points; most of its deficit is already present at
  construction.

## Group 7 — PerLTQA SNR × compression

Output: `group7_perltqa_snr_compression`

![PerLTQA relative self-correction and construction-loss curves](../diagnostic/figures_ablation/group7_perltqa_snr_compression/fig_noise_compression_relative.png)

The x-axis uses the measured compression ratio from `compression.json`.

| Variant | Comp ratio | #QA | T-probe | C-probe | S-probe | E2E | Self-corr / T✗ | Mem-loss / T✓ |
|---|---|---|---|---|---|---|---|---|
| Clean no comp | 2.54x | 2436 | 85.1% | 69.0% | 63.5% | 64.2% | 23.9% | 23.0% |
| Clean x3 | 3.78x | 609 | 84.7% | 63.9% | 59.3% | 60.4% | 25.8% | 29.3% |
| Clean x5 | 4.99x | 609 | 84.7% | 55.8% | 51.9% | 56.2% | 24.7% | 38.6% |
| Clean x8 | 7.81x | 609 | 85.1% | 43.7% | 38.8% | 46.6% | 22.0% | 52.5% |
| SNR5 no comp | 2.48x | 609 | 72.9% | 56.0% | 51.9% | 56.7% | 11.5% | 27.5% |
| SNR5 x3 | 3.66x | 609 | 73.7% | 53.7% | 49.9% | 52.4% | 14.4% | 32.3% |
| SNR5 x5 | 5.12x | 609 | 73.7% | 45.2% | 43.0% | 50.6% | 10.0% | 42.3% |
| SNR5 x8 | 8.46x | 609 | 73.7% | 33.2% | 29.7% | 40.1% | 3.8% | 56.3% |
| SNR0 no comp | 2.25x | 609 | 37.3% | 31.0% | 28.7% | 34.5% | 8.4% | 30.8% |
| SNR0 x3 | 3.58x | 609 | 37.3% | 28.6% | 24.5% | 33.5% | 7.1% | 35.2% |
| SNR0 x5 | 5.24x | 609 | 37.4% | 23.8% | 20.4% | 30.9% | 6.6% | 47.4% |
| SNR0 x8 | 10.25x | 609 | 37.4% | 15.9% | 14.1% | 22.3% | 2.6% | 61.8% |

Key observations:

- Without explicit compression, noise already reduces E2E from 64.2% clean to 56.7% at
  SNR5 and 34.5% at SNR0.
- At matched requested ratios, E2E falls consistently from clean to SNR5 to SNR0: at x3
  it is 60.4%, 52.4%, and 33.5%; at x8 it is 46.6%, 40.1%, and 22.3%.
- T is constant within each SNR family, while conditional memory loss increases with
  compression. From no compression to x8, it rises from 23.0% to 52.5% clean, from
  27.5% to 56.3% at SNR5, and from 30.8% to 61.8% at SNR0.
- Self-correction becomes rare under combined severe noise and compression: only 2.6% of
  T failures are recovered for SNR0 x8.

## Group 8 — Mosaic SNR × compression

Output: `group8_mosaic_snr_compression`

![Mosaic relative self-correction and construction-loss curves](../diagnostic/figures_ablation/group8_mosaic_snr_compression/fig_noise_compression_relative.png)

The x-axis uses the measured compression ratio from `compression.json`.

| Variant | Comp ratio | #QA | T-probe | C-probe | S-probe | E2E | Self-corr / T✗ | Mem-loss / T✓ |
|---|---|---|---|---|---|---|---|---|
| Clean no comp | 3.41x | 100 | 88.0% | 81.0% | 69.0% | 63.0% | 58.3% | 15.9% |
| Clean x2 | 3.64x | 100 | 88.0% | 80.0% | 62.0% | 57.0% | 41.7% | 14.8% |
| Clean x4 | 5.19x | 100 | 88.0% | 69.0% | 58.0% | 44.0% | 58.3% | 29.5% |
| Clean x6 | 6.82x | 100 | 88.0% | 67.0% | 52.0% | 48.0% | 50.0% | 30.7% |
| Clean x8 | 10.21x | 100 | 88.0% | 56.0% | 46.0% | 42.0% | 58.3% | 44.3% |
| SNR10 no comp | 3.27x | 100 | 83.0% | 71.0% | 60.0% | 55.0% | 5.9% | 15.7% |
| SNR10 x2 | 3.60x | 100 | 83.0% | 73.0% | 61.0% | 56.0% | 11.8% | 14.5% |
| SNR10 x4 | 5.00x | 100 | 83.0% | 67.0% | 53.0% | 49.0% | 5.9% | 20.5% |
| SNR10 x6 | 6.36x | 100 | 83.0% | 66.0% | 50.0% | 44.0% | 17.6% | 24.1% |
| SNR10 x8 | 10.70x | 100 | 83.0% | 47.0% | 34.0% | 35.0% | 5.9% | 44.6% |
| SNR5 no comp | 3.35x | 100 | 72.0% | 56.0% | 44.0% | 41.0% | 14.3% | 27.8% |
| SNR5 x2 | 3.38x | 100 | 72.0% | 63.0% | 49.0% | 44.0% | 17.9% | 19.4% |
| SNR5 x4 | 5.45x | 100 | 72.0% | 55.0% | 51.0% | 48.0% | 3.6% | 25.0% |
| SNR5 x6 | 7.63x | 100 | 72.0% | 51.0% | 45.0% | 40.0% | 3.6% | 30.6% |
| SNR5 x8 | 17.69x | 100 | 72.0% | 30.0% | 25.0% | 23.0% | 3.6% | 59.7% |

Key observations:

- Without explicit compression, E2E is 63% clean, 55% at SNR10, and 41% at SNR5.
- At x2, clean and SNR10 are close in E2E (57% and 56%), whereas SNR5 is 44%.
- Conditional memory loss increases with compression in every condition. At x8 it is
  44.3% clean, 44.6% at SNR10, and 59.7% at SNR5.
- SNR5 x8 overshoots its nominal target substantially, reaching 17.69x measured
  compression and the lowest E2E score, 23%.
- The 100-question sample size makes the x4/x6 non-monotonicity too small to interpret as
  a reliable reversal.

---

## Cross-group summary

1. Construction can both recover and destroy information. The conditional metrics make
   the distinction explicit: self-correction is measured only among T failures, while
   memory loss is measured only among T successes.
2. Name-aware prompting can recover a large fraction of transcript-stage failures; the
   anonymous-speaker prompt recovers 79.9% of them in Group 2.
3. Stronger compression consistently increases conditional memory loss. The effect is
   amplified by interfering speech, reaching 61.8% for PerLTQA SNR0 x8 and 59.7% for
   Mosaic SNR5 x8.
4. Audio-native Omni remains below the clean cascade on both datasets. On Mosaic, adding
   five reference-history chunks does not improve the current implementation and also
   produces a substantially higher achieved compression ratio.
5. Mosaic conclusions are based on 100-question single runs; PerLTQA SNR/compression
   cells contain 609 questions. Seeded replication remains important for uncertainty.

## Reproducing or extending

Groups are defined in `FOLDER_GROUPS` in `diagnostic/plot_probe_ablation.py`
(`FOLDERS1` through `FOLDERS8`). Run a predefined group with:

```bash
~/miniconda3/envs/mem/bin/python diagnostic/plot_probe_ablation.py --group 8
```

Or pass folders directly:

```bash
~/miniconda3/envs/mem/bin/python diagnostic/plot_probe_ablation.py \
  --folders '{"name": "/path/to/agents/run_folder"}'
```

Each invocation writes `fig_summary_table`, `fig_probe_bars`, `fig_cascade`,
`fig_memory_dynamics`, and `fig_confusion_counts` as PNG and PDF, plus `metrics.json`
and `metrics.md`. Groups 1, 7, and 8 additionally write
`fig_noise_compression_relative`, which plots the conditional self-correction and
construction-loss rates across the no-compression and compressed settings.
