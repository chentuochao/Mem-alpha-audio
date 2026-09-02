# Can Qwen3-Omni recognise the same speaker across audio chunks?

Experiment report. Code: `experiment/omni_memory/`. Raw results:
`experiment/omni_memory/outputs/`. Models: `Qwen3-Omni-30B-A3B-Instruct` throughout, with the
`-Thinking` checkpoint compared in §10. Data: mosaic `bundle_0` (196 recordings, 27 speakers).

---

## 1. The problem, in one picture

The audio memory arm reads one chunk at a time:

```
chunk 3  --> [Omni] --> "Speaker A said she has a sister"     (Speaker A = local to chunk 3)
chunk 12 --> [Omni] --> "Speaker B is training for a 10k"     (Speaker B = local to chunk 12)
```

If chunk 3's "Speaker A" and chunk 12's "Speaker B" are the same person, nothing in the
system knows it. Facts about one person can never be joined across recordings.

**The idea we wanted to test:** keep a few past chunks in the context window, labelled by
chunk id, and let the model notice the voices itself:

```
[chunk 9 audio][chunk 10 audio][chunk 11 audio][chunk 12 audio]
  --> [Omni] --> "the woman in chunk 9 is the woman in chunk 12"
```

**Result: this does not work.** More audio in the context makes the model *worse*, not
better. Details below.

---

## 2. How I approached it

The tempting move is to build the multi-chunk memory and see if QA improves. That answers
nothing when it fails: was it the audio, the window size, the prompt, the memory writer, or
the QA? So instead I tested the *underlying ability* first, on a ladder from the easiest
possible version of the task to the real one, changing **one thing at a time**:

| step | what the model hears | question asked | what it isolates |
|---|---|---|---|
| 1 | 2 clean clips, one voice each | same person? | can it match voices at all |
| 2 | same, but cut by the real pipeline's diarizer | same person? | cost of imperfect cuts |
| 3 | 2 whole chunks, two voices each | share a speaker? | can it separate *then* match |
| 4 | 4 whole chunks | who shares with whom? | the actual FIFO-window proposal |
| 5 | K clean reference voices + 1 new voice | which one, or new? | the alternative design |
| 6 | 1 whole chunk | who speaks when? | can it cut its own clips |

Each step has ground truth for free, because mosaic file names contain the true speaker
ids (`P0742_P0743_V00_S0589_I00000125` = a conversation between speakers `P0742` and
`P0743`), and `transcript_gt.json` / `vad_gt.json` say who talks when.

Two rules I applied throughout, both of which turned out to matter more than expected:

- **Always print the dumb baseline.** What score would "always say yes" get? If the model
  can't beat that, it knows nothing. This caught a fake result (§5).
- **Always include a control that can only be failed.** e.g. a set of recordings where
  *nobody* repeats: any "these two share a speaker" answer there is a hallucination.

---

## 3. What a probe actually looks like

### Step 1–3: the pair probe

The model receives two audio clips and one instruction, and must answer with JSON only:

```
[Recording R1]  <10 s of one person speaking>
[Recording R2]  <10 s of one person speaking>

Rate how likely it is that THE SAME PERSON speaks in both, 0-100.
Judge only the VOICE (timbre, pitch, rhythm). Two different people often discuss
the same topic, so the subject matter tells you nothing.

Answer with strict JSON only:
{"same_speaker_score": <0-100>, "reason": "<= 12 words"}
```

Three real answers from the run:

| truth | model answer |
|---|---|
| same person (`P5607`, two different recordings) | `{"same_speaker_score": 100, "reason": "Identical pitch, rhythm, and speaking style."}` ✔ |
| different people (`P0743` vs `P0742`) | `{"same_speaker_score": 0, "reason": "Different genders, pitches, and speaking styles."}` ✔ |
| same person (`P1298`) | `{"same_speaker_score": 0, "reason": "Male and female voices; different pitch and timbre."}` ✘ |

### Step 4: the window probe (the proposal being tested)

```
[Recording R1] <60 s of a two-person conversation>
[Recording R2] <60 s of a different two-person conversation>
[Recording R3] ...
[Recording R4] ...

Step 1: for each recording, list the distinct speakers as R1_S1, R1_S2, ...
Step 2: group the labels that are THE SAME PERSON across recordings.
It is possible that NO speaker appears in more than one recording.

{"speakers": {...}, "groups": [[...], ...], "n_distinct_people": <int>}
```

A real answer, on a window where R1 and R3 are the same two people and R2 and R4 are the
same two people (truth: 4 distinct people, 2 linked pairs):

```json
{"speakers": {"R1": ["R1_S1","R1_S2"], "R2": ["R2_S1","R2_S2","R2_S3","R2_S4"],
              "R3": ["R3_S1","R3_S2"], "R4": ["R4_S1","R4_S2","R4_S3","R4_S4"]},
 "groups": [["R1_S1"],["R1_S2"],["R2_S1"],["R2_S2"],["R2_S3"],["R2_S4"], ...]}
```

Two failures in one answer: it heard **four** speakers in a two-person recording, and it
linked **nothing** to anything.

### Step 5: the gallery probe (the alternative design)

```
Reference voices you already know (one clip per person):
[S1] <10 s>  [S2] <10 s>  ...  [S10] <10 s>
Now the query:
[Q] <10 s of one voice, from a recording you have not heard>

Which reference is Q, or is it somebody new?
{"match": "<S1..S10 or NEW>", "confidence": <0-100>, "runner_up": "..."}
```

Real answer, on a query whose speaker is **not** in the gallery:

```json
{"match": "S2", "confidence": 95, "runner_up": "NEW"}
```

Confidently wrong — and this is the model's typical behaviour at K=10 (see §6).

---

## 4. Results

### Step 1–2: matching one clean voice against another — works

200 balanced pairs; in 199 of 200 the two clips come from *different* recordings.

| what the model heard | how it was asked | accuracy | ROC-AUC |
|---|---|---|---|
| clean 10 s voice, cut using the true VAD | "yes or no?" | 0.595 | — |
| clean 10 s voice, cut using the true VAD | "score 0-100" | **0.835** | **0.835** |
| clean 10 s voice, cut by the real diarizer | "score 0-100" | 0.742 | 0.731 |

Reading: it can do this. Using the pipeline's own (imperfect) diarization to cut the clips
costs about 0.10 AUC — real, but not fatal.

*ROC-AUC = the chance that a true "same person" pair gets a higher score than a true
"different people" pair. 0.5 is coin-flipping, 1.0 is perfect.*

### Step 3: two-speaker recordings — mostly fails

Same question, but each clip is a whole 60 s conversation with two voices in it:

| clips | ROC-AUC |
|---|---|
| clean single voice | 0.835 |
| whole two-speaker chunk | **0.589** |

Reading: the model can compare *voices*, but it cannot first pull one voice out of a
conversation and then compare it. This already predicts the next result, because
two-speaker chunks are exactly what a FIFO window would hold.

### Step 4: the window proposal — worse than a dumb baseline

W=4 recordings per probe; scored on every recording pair ("do these two share a speaker?").

| window set | model F1 | "always yes" F1 | model acc | "always no" acc |
|---|---|---|---|---|
| balanced (33% of pairs truly share) | **0.462** | 0.500 | 0.537 | **0.667** |

The model loses to both trivial answers. Also:

- It gets the **number of distinct people** right on 8% of windows (it over-counts, mean
  error +0.8 people).
- **Shuffling the recording order** changed the answer on 26 of 34 windows.
- On the **no-recurrence control** (windows where nobody repeats) it still invented links
  on 14 of 66 pairs.
- 10 of 142 window answers were **unparsable JSON**, versus 0 of 640 pair answers.

### Step 5: the gallery alternative — works only with 2 voices

K reference voices + one query; 30% of queries are people not in the gallery.

| K | picks the right person | **top-2** | correctly says "NEW" | overall | chance |
|---|---|---|---|---|---|
| 2 | 0.690 | 0.786 | **0.917** | **0.758** | 0.500 |
| 5 | 0.702 | 0.798 | 0.278 | 0.575 | 0.200 |
| 10 | 0.448 | 0.724 | **0.000** | 0.313 | 0.100 |

Asking for a score per reference (the trick that fixed step 1) does **not** help here: the
confidence runs the wrong way — average top score 79.5 when the person *is* enrolled, 84.4
when they are *not*.

#### How the gallery scales — three different curves

"Accuracy drops with K" is true but hides three behaviours that scale differently, and the
distinction decides what the fix should be:

- **Ranking barely degrades.** Top-2 is flat (0.79 → 0.80 → 0.72) and identification
  *relative to chance* actually improves (1.4x → 3.5x → 4.5x). The right person keeps
  landing near the top of the list.
- **Rejection collapses, monotonically and completely** (0.92 → 0.28 → 0.00). Chance level
  cannot explain this one: "NEW" is always available and costs nothing.
- **The final pick degrades in steps, not smoothly.** Error is flat from K=2 to K=5
  (0.31 → 0.30) despite four times the distractors, then nearly doubles by K=10 (0.55).
  That looks like a crowding threshold somewhere around 5-10 voices in context.

**What this says about the model.** It has no stable internal notion of "same person" — it
answers to the *shape of the question*. Asked 1-vs-1 yes/no it was wildly over-conservative
(said yes on 9% of probes against a 48% true rate); asked K-way "pick one or NEW" it becomes
over-liberal until it never rejects at all. Same model, same audio, opposite biases, and the
graded run confirms there is no calibrated similarity underneath.

**It is not a cost problem.** A gallery costs 127 audio tokens per person, so all 27
speakers of bundle_0 would be ~3.4k tokens — trivial against a 65k context — and latency
grew only 2.2 s → 2.7 s from K=2 to K=10. Token budget is not the constraint; decision
quality is.

**Would K separate 1-vs-1 calls rescue it?** Tempting, since pairwise is the good regime,
but identification requires the true match to outrank *every* distractor. At AUC 0.835 that
is 0.835^26 ≈ 1% for 27 speakers if distractors were independent — speaker-verification
systems reach EER 1-3% (AUC ≈ 0.99) precisely because identification is so much harder than
verification. The measured numbers *beat* that pessimistic estimate (K=10 gave 0.45, the
independence model predicts ≈0.06), which tells us the distractors are not equally
confusable: half are the opposite pitch class and get rejected trivially. So the real driver
is the number of *acoustically similar* candidates, not raw K — but a 27-speaker gallery
still holds a dozen of those.

K=20 and K=27 were not measured; the trend above is from K = 2, 5, 10.


### Step 6: can Omni cut its own enrolment snippets?

The gallery design has a hidden dependency: somebody has to produce the clean
single-speaker clips. So far that was the cascade's diarizer. Can the model do it itself?

The task is deliberately *not* full diarization — a gallery only needs one good excerpt per
person, so the model is told it may skip anything it is unsure about:

```
[Recording R1] <60 s conversation>

Find short stretches where you are CONFIDENT exactly ONE person is speaking and nobody
talks over them. At most 2 per speaker, 3-10 s each. You do NOT need to cover the whole
recording — one reliable stretch is worth more than several doubtful ones.

{"snippets": [{"speaker": "S1", "start": <s>, "end": <s>, "confidence": <0-100>}, ...],
 "n_speakers": <int>}
```

A typical answer (78 recordings, 60 s each, two speakers in every one):

```json
{"snippets": [{"speaker": "S1", "start": 0.0, "end": 5.0, "confidence": 95},
              {"speaker": "S2", "start": 5.0, "end": 10.0, "confidence": 95}],
 "n_speakers": 2}
```

Scored the way a gallery would use the output — purity of what comes back, not coverage —
against the cascade diarizer on the identical clips:

| | Omni | cascade diarizer |
|---|---|---|
| snippet purity (of the speech inside, how much is one person) | 0.880 | **0.917** |
| snippets usable (purity ≥ 0.9, ≥ half speech) | 0.593 | **0.747** |
| usable snippets per recording | 1.44 | 1.40 |
| **recordings where BOTH speakers got a usable snippet** | **0.192** | **0.500** |
| two labels that are really the same person | 0.103 | — |
| speaker count correct (truth = 2) | 0.795 | — |

Reading: this is the closest Omni comes to the cascade anywhere in the study. Timestamps
were always inside the clip (189/189), 97% of each snippet was real speech rather than
silence, and purity 0.88 vs 0.92 is a small gap.

But the number that matters for a gallery is the fourth row. Enrolment needs a clip for
*each* person; Omni delivers both speakers in only 19% of recordings against the diarizer's
50%, largely because it answers with two 5-second snippets at the very start of the clip
(see the example above) instead of hunting for each voice. And in 10% of recordings its two
labels are the same person — which would create two gallery entries for one human, exactly
the fragmentation failure the whole exercise was meant to fix.

An independent check of the same ability, free from the earlier window runs: asked to list
the speakers in a recording that always has exactly two, Omni said 2 on 81% of 528
recordings, 3+ on 12%, and 1 on 7%.

**Conclusion:** Omni can point at clean speech, but not reliably enough, and not evenly
enough across speakers, to replace the diarizer in the snippet-cutting step. A gallery
built this way would be missing half its people.

### The baseline all of this has to beat

The existing WeSpeaker cascade on the same 27 speakers: **purity 0.935**, for the cost of
one small embedding model. Its known flaw is fragmentation — six speakers split into two
identities each (e.g. `P0832` = `GLOBAL_SPK_2` and `GLOBAL_SPK_4`). No Omni configuration
measured comes close to 0.935.

---

## 5. Two mistakes worth knowing about

**Mistake 1 — a yes/no question hid the model's ability.** The first pairwise run scored
0.595 accuracy and I nearly concluded the model was weak at voices. But it was answering
"same person" on only 9% of probes when the true rate was 48%: when it *did* say yes it was
right 94% of the time. The ability was there; the yes/no framing forced a badly placed
threshold. Asking for 0-100 and thresholding afterwards moved accuracy 0.595 → 0.835 on the
identical audio. **Do not measure this model with boolean prompts.**

**Mistake 2 — my first window set was rigged in the model's favour.** I built windows from
consecutive chunks in bundle order and got F1 0.913, which looks excellent. It was
meaningless: bundle order groups conversations by speaker pair, so 88% of recording pairs
genuinely shared a speaker, and "always say yes" scores 0.935 — *better than the model*.
The rebuilt balanced set (two recordings from each of two speaker-disjoint pairs) dropped
the model to 0.462. The dumb-baseline rule is what exposed this; the probe builder now
defaults to `--composition balanced`.

---

## 6. Why it degrades: it is the number of voices, not the length

| configuration | voices in the prompt | prompt tokens | works? |
|---|---|---|---|
| pair of clean clips | 2 | 0.5k | yes (0.835) |
| gallery K=2 + query | 3 | 0.7k | yes (0.758) |
| gallery K=10 + query | 11 | 1.8k | no (0.313) |
| window of 4 chunks | 8 | 3.4k | no (0.462) |

Nothing here is near the 65k context limit, so this is not a length problem. Performance
tracks how many distinct voices the model must hold apart at once.

One practical consequence: a **gallery prefix is identical for every query**, so the vLLM
server serves it from the prefix cache after the first call. A **FIFO window cannot reuse
anything** — dropping the oldest chunk shifts every later token, invalidating the cache at
every step. The design that works better is also the cheaper one.

Cost per probe (median): pair 2.5 s / 0.5k tokens; gallery K=10 2.7 s / 1.8k; window W=4
13 s / 3.4k. For comparison, the existing memory arm already spends ~100 s per chunk.

---

## 7. What the probes recommended (superseded by §8)

1. **Do not build the FIFO-of-old-chunks arm.** It is the worst case on both axes at once
   (two-speaker clips, many voices) and the most expensive.
2. **If Omni is to help with identity, give it one clean voice against at most 2–3
   candidates**, ask for a score rather than a decision, and apply the accept/reject
   threshold in code. That is the K=2 regime: 0.76 overall, 0.92 correct rejections.
3. **Shortlist-and-verify** is the shape that fits: WeSpeaker proposes the 1–2 nearest
   identities (cheap, already 0.935 purity), Omni only arbitrates the borderline ones.
   That attacks the cascade's real weakness — the six fragmented speakers — instead of
   replacing a component that works.
   Note this keeps the **diarizer in the loop either way**: step 6 shows Omni cannot cut
   its own enrolment clips evenly enough (both speakers covered on 19% of recordings vs
   the diarizer's 50%), so even a pure-gallery design would still depend on it.
4. **Check the prize before building anything.** Run the memory arm once with ground-truth
   speaker ids injected as text. If perfect speaker linking barely moves QA accuracy, no
   linking mechanism is worth its cost.

Stages C (name binding) and D (memory integration) of the original plan were not started:
the gate at step 4 failed.

---

## 8. The registry experiment: Omni-cut enrolment + two matchers

The probes said Omni can *point at* clean speech but cannot hold many voices apart. That
leaves one design worth building: let Omni cut its own enrolment clips, and hand the
matching to something else. Two matchers were run end-to-end over all 196 recordings of
bundle_0, with the enrolment source as the other axis:

| enrolment cut by | matcher | what it tests |
|---|---|---|
| Omni | Omni gallery | the fully self-contained design (1) |
| Omni | WeSpeaker + shipped linker | the hybrid design (2) |
| diarizer (same 10 s budget) | WeSpeaker | control: is the *cut* or the *matcher* at fault |
| oracle VAD (same budget) | WeSpeaker | upper bound |

The matcher for the WeSpeaker arms is the shipped one, unchanged —
`speaker_pool.WeSpeakerBackend` + `build_linker("greedy", 0.5)`, the mosaic pipeline's own
settings. Omni's snippets are converted into the same `(frames, n_spk)` activity matrix the
diarizer produces, so nothing downstream had to be adapted.

### Results (bundle_0: 196 recordings, 27 speakers)

| arm | purity | identities (true = 27) | fragmented | merged | enrolment purity |
|---|---|---|---|---|---|
| WeSpeaker + oracle cuts | **0.958** | 35 | 8 | 2 | 1.000 |
| WeSpeaker + diarizer cuts | 0.939 | 42 | 12 | 7 | 0.917 |
| *shipped cascade (reference)* | *0.935* | *37* | *11* | *9* | *—* |
| **WeSpeaker + Omni cuts — design (2)** | **0.879** | 46 | 17 | 16 | 0.852 |
| Omni gallery, recency-3 shortlist | 0.812 | **87** | 26 | 25 | 0.852 |
| **Omni gallery, unbounded — design (1)** | **0.362** | 42 | 25 | 14 | 0.852 |

**Design (2) works, nearly.** Omni-cut enrolment plus the existing WeSpeaker linker reaches
0.879 purity against the cascade's 0.935 — close enough to be interesting, not yet close
enough to drop the diarizer. And the gap has an obvious cause: registry purity tracks
enrolment purity almost linearly across the three cut sources (0.852 → 0.879,
0.917 → 0.939, 1.000 → 0.958). The matcher is not the bottleneck; the clips are. Better
snippet selection — multi-snippet enrolment, discarding low-purity clips, longer excerpts —
attacks the gap directly.

The diarizer-cut control also validates the comparison: at the *same* 10 s budget it scores
0.939, statistically the same as the shipped pipeline's 0.935 with full-length cuts. So the
10-second enrolment budget is not what separates the arms.

**Design (1) fails, as the probes predicted.** Showing Omni the whole gallery gives 0.362
purity: of 348 decisions where it claimed to recognise someone, only **18%** were the
person that entry was actually enrolled with. It does not converge to the "few
mega-identities" I expected from the K-sweep — it creates 42 identities while *also*
merging 14 of them, i.e. it splits and fuses at the same time. The NEW-rate wanders
(0.13 → 0.03 → 0.19 across the run) rather than collapsing to zero, and confidence is a
constant 95 on 380 of 390 decisions — no usable signal at all.

**The recency shortlist is a trap.** Showing only the 3 most recently matched entries keeps
Omni in its good regime and re-identification jumps from 18% to 57% — but the registry ends
with **87 identities for 27 people**. Because the right candidate is usually not in a
recency window, the model keeps saying NEW. Its 0.812 purity looks respectable only because
purity rewards splitting: many small, pure, duplicate identities. This is the case where
purity alone misleads and the identity count has to be read alongside it.

### Cost

Omni enrolment extraction: ~4.5 s per recording (~15 min for the bundle, one pass).
Omni gallery matching: ~0.9 s per decision (~5 min per arm, prefix cache absorbing the
growing gallery). WeSpeaker embedding: 2.1 s per clip on CPU, seconds to link. The hybrid
is also the cheapest of the three at inference time.

### Verdict

Of the two designs asked for, **(2) is the one to keep**: Omni cuts the clips, WeSpeaker
owns identity. It is 0.06 purity behind the shipped cascade today, and the deficit is
attributable to enrolment-clip quality rather than to the linker, so it is improvable.
**(1) should be dropped** — as a matcher, Omni is worse than the component it would
replace by a factor that no prompt fix in this study has come close to closing.

Next, in order: (a) raise enrolment purity (keep only clips above a purity proxy, enrol
several clips per speaker, prefer longer excerpts) and re-run design (2) — the numbers say
that is where the remaining 0.06 lives; (b) the oracle-QA check from §7, to find out
whether closing that gap changes any answer downstream.

---

## 9. The online gallery loop: extract → match → update, from an empty gallery

§8 pre-cut every enrolment clip in an offline pass, which quietly assumed the hard part
(finding each person's voice) was already solved. The deployed shape has no such pass — the
gallery is *built by* the extraction step:

```
for each incoming recording (2 speakers, nothing known in advance):
    samples = EXTRACT(recording)          # find the voices, cut one sample each
    for each sample:
        id = MATCH(sample, gallery)       # who is this, or nobody?
        if id is None:  gallery.add(sample)     # a new person; this clip becomes the reference
        else:           gallery.update(id, sample)
```

The gallery starts **empty**; every reference in it was produced by the extractor being
tested. `registry/run_gallery_loop.py --extractor {omni,diarizer} --matcher {omni,wespeaker}`.
Run on 10 recordings sampled across bundle_0 (17 distinct people, several recurring).

| extract | match | chunk-set F1 | exact set | purity | gallery (true: 17) | people ever enrolled |
|---|---|---|---|---|---|---|
| diarizer | WeSpeaker | **0.923** | **0.80** | 0.950 | 18 | **16/17** |
| diarizer | Omni, **one-by-one** | **0.842** | 0.60 | 0.850 | 16 | 16/17 |
| **Omni** | WeSpeaker | 0.750 | 0.20 | **0.960** | 12 | 12/17 |
| Omni | Omni, one-by-one | 0.625 | 0.10 | 0.857 | 12 | 12/17 |
| diarizer | Omni, K-way | 0.526 | 0.20 | 0.500 | 8 | 16/17 |
| Omni | Omni, K-way | 0.364 | 0.10 | 0.480 | 7 | 11/17 |

**How the matcher is *called* is worth more than which matcher it is.** The two Omni rows
differ only in prompting: "here are all K references, pick one or say NEW" (K-way) versus
one 1-vs-1 comparison per reference with the accept/reject threshold applied in code — the
same structure the embedding matcher uses. That single change takes chunk-set F1 from 0.526
to **0.842** and purity from 0.500 to 0.850, and stops the gallery collapsing (8 -> 16
identities for 17 people). It closes most of the gap to WeSpeaker (0.923), and it is the
fair comparison: the K-way version was using the model in the exact regime the probes showed
it fails in. The cost is K model calls per voice instead of one — 12.0 s per recording
against 3.5 s (K-way) and 6.7 s (embeddings), and that grows linearly with the number of
people known.

**Now both axes cost something, and they fail differently.**

*Omni as extractor costs recall.* It captured both speakers in only **3 of 10** recordings;
in **7 of 10**, every clip it returned was the same person — often two clean, high-purity
clips (`purity 1.0` each) labelled `S1` and `S2` but both the same human. WeSpeaker then
correctly merges them (cosine 0.68-0.81, against 0.06-0.16 for genuinely different voices).
The consequence is silent and permanent: only 12 of 17 people ever enter the gallery, so the
missing five can never be recognised later. The diarizer captured both speakers in 9 of 10.
Note the precision of the Omni-extract arm is 1.000 — what it does enrol is right; it simply
misses half the people.

*Omni as matcher costs precision and purity — but only when asked K-way.* Shown the whole
gallery at once it collapses to 7-8 entries for 17 people (purity 0.48-0.50): past a handful
of voices it stops saying NEW and folds everyone together, the K-sweep of §Step 5 playing
out in a live loop. Asked one reference at a time it keeps 16 identities and purity 0.850.

### Scaling from 10 to 50 recordings

10 recordings only contain 3 recurring people, so matching is barely exercised. At 50
recordings (27 people, **25 of them recurring**, up to 8 appearances each) the ranking
changes:

| extract | match | F1 @10 | **F1 @50** | purity @50 | gallery @50 (true: 27) | wall @50 |
|---|---|---|---|---|---|---|
| diarizer | WeSpeaker | 0.923 | **0.953** | 0.939 | 31 | 5.3 min |
| Omni | WeSpeaker | 0.750 | **0.823** | 0.929 | 34 | 8.1 min |
| diarizer | Omni, one-by-one | 0.842 | **0.764** | 0.745 | 39 | **25.9 min** |
| diarizer | Omni, K-way | 0.526 | **0.411** | 0.378 | 14 | 1.5 min |

**The two families move in opposite directions.** Both WeSpeaker arms *improve* with more
data (0.923 → 0.953, 0.750 → 0.823) — more recurrence means more chances to link, and the
centroids get better as they absorb more samples. Both Omni matchers *degrade*
(0.842 → 0.764, 0.526 → 0.411), because every new person makes the decision harder rather
than easier. The one-by-one matcher, which looked competitive at 10 recordings, has fallen
0.19 behind by 50 and its purity is down to 0.745 with 20 of 27 speakers fragmented.

**And its cost grows with the gallery, as predicted.** Per-recording latency went from 5.5 s
over the first ten recordings to **51.2 s** over the last ten — a 9x increase purely because
the gallery grew from 9 to 39 entries, since it issues one model call per known person. The
embedding matcher is flat (6.6 s → 7.1 s): one forward pass plus K cheap dot products. Over
the whole bundle (196 recordings, 27+ people) the one-by-one arm would cost hours where the
embedding arm costs ~20 minutes.

So the one-by-one framing is the right way to *use* an LLM matcher, and it is still the
wrong component for the job: it gets worse exactly where a memory needs it to get better.

**Control — the first 10 recordings in order**, which all come from one speaker pair, so
only 2 people exist:

| extract | match | chunk-set F1 | gallery (true: 2) |
|---|---|---|---|
| diarizer | WeSpeaker | 1.000 | 2 |
| diarizer | Omni | 1.000 | 4 |
| Omni | WeSpeaker | 0.889 | 2 |
| Omni | Omni | 0.824 | 5 |

With two people everything roughly works — which is the same K≤2 regime the gallery probe
identified. The differences only appear once the gallery has to hold more than a couple of
voices, so any evaluation restricted to a single conversation pair will look fine and tell
you nothing.

---

## 10. Earlier variant: whole recording as the query, offline enrolment

§8 handed every matcher a pre-cut single-speaker clip, so "find the voices" and "recognise
the voices" were separate steps and only the second was scored. A deployed memory does not
get that split for free: a recording arrives whole. So the experiment was re-run with a
fixed interface — **each arm receives the 2-speaker conversation and must answer "which
registry identities speak in this recording?"**, registering anyone new. 182-192 recordings,
sequential, registry grows as it goes.

New metric, `score_registry.py`: per-chunk **identity-set** precision/recall against the
recording's true speakers. Purity says the identities are internally coherent; this says
the memory actually knows who is in the room.

| detection | matching | chunk-set F1 | exact set | registry purity | identities (true: 27) |
|---|---|---|---|---|---|
| diarizer | WeSpeaker | **0.969** | 0.890 | **0.939** | 42 |
| **Omni** | **WeSpeaker** | **0.901** | 0.747 | **0.879** | 46 |
| Omni (implicit, one call) | Omni | 0.468 | 0.181 | 0.596 | **206** |
| diarizer | Omni | 0.274 | 0.068 | 0.194 | **9** |
| *oracle* | *WeSpeaker* | *0.978* | *0.916* | *0.958* | *35* |

**The matcher axis dominates.** Swapping the matcher moves chunk-set F1 by 0.4-0.7; swapping
the detector moves it by 0.07. Whoever decides identity determines whether the system works.

**Omni fails in opposite directions depending on what it is shown**, which is the clearest
evidence yet that it has no calibrated notion of "same person":

- *Whole recording + gallery, one call*: it names **8.7 identities per recording** as
  present, in recordings that contain two people, and still reports 1-2 new voices in 130 of
  182 chunks. The registry explodes to **206 identities for 27 people**. It also mis-counts
  the room: 73 of 182 recordings were reported as having 4 speakers.
- *Clean single-speaker clip + gallery*: the opposite — it almost never says NEW once the
  gallery grows, so 27 people collapse into **9 identities** (purity 0.194).

Same model, same audio, same registry; the answer follows the shape of the question, not the
voices. A memory built on either behaviour is broken in a way that is invisible from the
outside: one invents people, the other silently fuses them.

**Practical read.** Keeping the whole recording as the interface does not rescue the
LLM-as-matcher design — it makes it worse than the clip-level version (F1 0.468 vs 0.852 for
the best clip-level Omni arm). The Omni-detects / WeSpeaker-matches hybrid is unaffected by
the interface change and stays 0.07 behind the shipped cascade.

Run it with `registry/run_registry_chunk.py --detector {none,diarizer,omni}`; all arms,
including the §8 ones, are scored together by `registry/score_registry.py`.

---

## 11. Caveats

- One model (`Instruct`, not `Thinking`), one dataset (mosaic). TBBT and PerLTQA — and
  especially TTS voices — may behave differently.
- Prompts were tuned a little, not exhaustively (4 pair variants, 2 gallery variants).
  Step 1 shows phrasing is worth a lot, so the failing setups might improve — but window
  clustering is *below* trivial baselines, not marginally behind.
- Zero-shot only; no fine-tuning.
- The registry arms enrol at most one clip per speaker per recording, ~10 s, with no
  quality filtering — the most obvious lever on design (2) is untouched.
- The `shipped cascade` row is a reference, not a matched control: it was produced by the
  real pipeline over a slightly different clip set (hence 36 speakers seen against my 27).
  The like-for-like control is `WeSpeaker + diarizer cuts`, run through the same harness.
- Gallery enrolment is naive: the first clean 10 s of each speaker, one snippet, no quality
  filtering.
- "Pitch class" (median F0 split at 165 Hz) is used to separate easy from hard pairs. It is
  a proxy, deliberately not called gender — mosaic ships no speaker-sex metadata.

---


## 12. Instruct vs Thinking

`Qwen3-Omni-30B-A3B-Thinking` was run through the same probes, same manifests, same audio,
same scorer — only the served model changed. It is **not** a better perceiver of voices; it
is a more cautious decider, and it costs 4-40x more.

| probe | Instruct | Thinking |
|---|---|---|
| pairwise, clean clips, graded (ROC-AUC) | **0.835** | **0.625** |
| window clustering, balanced (F1) | 0.462 | 0.462 |
| window clustering, accuracy | 0.537 | **0.741** |
| gallery K=10, overall | 0.313 | **0.444** |
| gallery K=10, closed-set identification | **0.448** | 0.301 |
| gallery K=10, correctly says NEW | 0.000 | **0.732** |
| snippet extraction, purity | 0.864 | 0.861 |
| snippet extraction, both speakers covered | **0.308** | 0.211 |
| snippet extraction, usable clips per recording | **2.21** | 0.95 |

**Voice discrimination gets worse, not better** (AUC 0.835 → 0.625). Deliberation does not
help a perceptual judgement here; if anything the model talks itself out of correct
snap decisions.

**Rejection gets dramatically better** — 0/45 unknown speakers rejected by Instruct,
73% by Thinking. This is the one behaviour a memory genuinely needs, because a false merge
fuses two people's facts irreversibly. Thinking pays for it by over-rejecting: 36% of
speakers that *were* enrolled got declared NEW, so closed-set identification drops to 0.301.
The failure mode flips from unsafe (merge) to safe (split).

**Window clustering moves along the precision/recall curve, not past the baseline.**
Precision rises 0.377 → 0.750 and recall falls 0.596 → 0.333, leaving F1 identical at 0.462
— still at the always-yes baseline of 0.500. Accuracy does clear the always-no baseline
(0.741 vs 0.667) for the first time, but on n=54 chunk-pairs from 9 windows.

### The token budget is a trap, and a finding in itself

At `--max_tokens 3072`, **76% of window probes and 17% of gallery probes never produced
JSON** — the reasoning trace ran past the cap before closing `</think>`. Those runs look
scoreable but are a biased, easy subset (the probes whose reasoning happened to be short).
Raising to 8192 fixed parsing completely (9/9) but revealed the real cost: **5.9-7.1k
completion tokens and ~554 s per window probe**, against 13 s for Instruct. Extrapolated to
a bundle, that is >30 h of GPU for one pass of the cheapest arm, so the 8k window run was
stopped at 9 probes.

Median latency / completion tokens: pairwise 10.4 s / 257, gallery 18.1 s / 439, extraction
96.4 s / 2340, window 125 s / 3072 (capped) → 554 s / ~6500 (uncapped).

`extract_json()` now strips everything up to the last `</think>` before parsing, since the
reasoning routinely contains draft JSON that would otherwise be picked up instead of the
answer.

### Verdict

Thinking does not change the conclusion. It is worse at the perceptual task that the
hybrid design relies on, worse at producing enrolment clips (0.95 usable per recording vs
2.21), and far too slow for a per-chunk memory loop. The one thing worth borrowing is its
willingness to say "new person" — and the hybrid already gets that for free from a
similarity threshold on WeSpeaker embeddings, at no extra cost.

Reproduce: `bash experiment/omni_memory/run_thinking_suite.sh [probes|long]` against a
server started with `MODEL=Qwen/Qwen3-Omni-30B-A3B-Thinking SERVED_NAME=qwen3-omni-thinking
PORT=8011`.

---

## 13. Reproducing

```bash
cd /storage/home/tuochao/Mem-alpha-audio && export PYTHONPATH=$PWD
E=experiment/omni_memory

# 0. server (LIMIT_MM must allow gallery_size + 1 audios)
CUDA_VISIBLE_DEVICES=0,1 TENSOR_PARALLEL_SIZE=2 MAX_MODEL_LEN=65536 PORT=8010 \
  LIMIT_MM='{"audio":32}' MEDIA_PATH=$PWD/$E/outputs/clips \
  HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 bash baseline/launch_vllm_omni.sh

# 1-2. pairs of clean clips        (--source pred for the diarizer-cut version)
python $E/probes/build_speaker_probe_sets.py --mode pairs --bundle bundle_0 --n 200 --name pairs_b0_gt
python $E/probes/run_speaker_probe.py --manifest $E/outputs/pairs_b0_gt.jsonl \
    --out $E/outputs/pairs_b0_gt.score.pred.jsonl --base_url http://127.0.0.1:8010/v1 --pair_variant score
python $E/probes/score_speaker_probe.py --pred $E/outputs/pairs_b0_gt.score.pred.jsonl

# 3. pairs of whole two-speaker chunks
python $E/probes/build_speaker_probe_sets.py --mode pairs --clip_kind chunk --bundle bundle_0 \
    --n 120 --clip_sec 60 --name chunkpairs_b0_60s

# 4. windows  — ALWAYS balanced, and build the no-recurrence control too
python $E/probes/build_speaker_probe_sets.py --mode windows --bundle bundle_0 --W 4 \
    --clip_sec 60 --composition balanced --permute --limit 40 --name win_b0_W4_bal
python $E/probes/build_speaker_probe_sets.py --mode windows --bundle bundle_0 --W 4 \
    --clip_sec 60 --no_recurrence --limit 12 --name win_b0_W4_norec

# 5. gallery   (--gallery_variant choice | scores)
python $E/probes/build_voice_gallery.py --bundle bundle_0 --gallery_size 10 --n 150 --source pred
python $E/probes/run_speaker_probe.py --manifest $E/outputs/gallery_bundle_0_pred_K10.jsonl \
    --out $E/outputs/gallery_b0_pred_K10.pred.jsonl \
    --gallery $E/outputs/gallery_bundle_0_pred_K10.gallery.json \
    --base_url http://127.0.0.1:8010/v1 --gallery_variant choice

# 6. snippet extraction — can the model cut its own enrolment clips?
python $E/probes/build_speaker_probe_sets.py --mode extract --bundle bundle_0 \
    --n 80 --clip_sec 60 --name extract_b0_60s
python $E/probes/run_speaker_probe.py --manifest $E/outputs/extract_b0_60s.jsonl \
    --out $E/outputs/extract_b0_60s.instruct.pred.jsonl \
    --base_url http://127.0.0.1:8010/v1 --extract_variant snippets
# the scorer prints the cascade diarizer's purity on the same clips for comparison
python $E/probes/score_speaker_probe.py --pred $E/outputs/extract_b0_60s.instruct.pred.jsonl

# 7-8. registry arms: enrolment from three sources, then two matchers
python $E/probes/build_speaker_probe_sets.py --mode extract --bundle bundle_0 \
    --n 200 --clip_sec 180 --name enrol_extract_b0
python $E/probes/run_speaker_probe.py --manifest $E/outputs/enrol_extract_b0.jsonl \
    --out $E/outputs/enrol_extract_b0.pred.jsonl --extract_variant spread \
    --base_url http://127.0.0.1:8010/v1
python $E/registry/build_enrolment.py --source omni --pred $E/outputs/enrol_extract_b0.pred.jsonl \
    --name enrol_b0_omni
python $E/registry/build_enrolment.py --source diarizer --name enrol_b0_diarizer   # control
python $E/registry/build_enrolment.py --source gt       --name enrol_b0_gt         # upper bound

python $E/registry/run_registry_omni.py --enrolment $E/outputs/enrol_b0_omni.jsonl --mode unbounded
python $E/registry/run_registry_omni.py --enrolment $E/outputs/enrol_b0_omni.jsonl --mode recency3
for src in omni diarizer gt; do
  python $E/registry/run_registry_wespeaker.py --enrolment $E/outputs/enrol_b0_$src.jsonl \
      --embedding_device cpu --cache_embeddings $E/outputs/emb_b0_$src.npz
done
python $E/registry/score_registry.py --runs "$E/outputs/registry_*.json" \
    --cascade Audio_Results/vibevoice/test/step2/bundle_0/raw_speaker_tracking.json \
    --csv $E/outputs/registry_comparison.csv

# any probe can print the cascade baseline next to it
python $E/probes/score_speaker_probe.py --pred <pred.jsonl> \
    --cascade Audio_Results/vibevoice/test/step2/bundle_0/raw_speaker_tracking.json
```

`probes/`: `build_speaker_probe_sets.py` (makes probes + cuts audio),
`build_voice_gallery.py` (reference voices), `run_speaker_probe.py` (sends them to the
server, resumable), `score_speaker_probe.py` (metrics, baselines, cascade comparison).
`registry/`: `enrol_common.py` (Omni snippets -> the diarizer's activity-matrix interface),
`build_enrolment.py` (three enrolment sources), `run_registry_omni.py` /
`run_registry_wespeaker.py` (the two matchers), `score_registry.py` (one table for all arms).
