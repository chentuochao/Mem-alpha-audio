---
marp: true
paginate: true
---

<!--
Deck: "Can a speech LLM track who is speaking across recordings?"
Audience: mentor with NO prior context on this project.
Target: 12 minutes + questions. 13 main slides, 4 backup.

Render:  marp doc/omni_speaker_tracking_slides.md -o slides.html   (or paste into Slides/Keynote)
Speaker notes are the indented blocks under each slide — say these, don't read the slide.

Rule for delivery: every slide answers ONE question. If you are short on time, cut
slides 6 and 12 (they are supporting detail, not results).
-->

# Can a speech LLM track *who* is speaking across recordings?

An evaluation of Qwen3-Omni for cross-recording speaker identity

Tuochao · <date>

> **Say:** "We're building a memory system that listens to conversations. It works,
> except for one thing: it can't tell that the person talking today is the same person
> who talked last week. I spent this week testing whether the speech model itself can
> fix that. Short answer: no — but the experiment told us exactly what to build instead."

---

## The system, in one picture

```
recording 1  ──►  [ speech LLM ]  ──►  "Speaker A has a sister in Boston"
recording 2  ──►  [ speech LLM ]  ──►  "Speaker B is training for a 10k"
   ...                                        │
                                              ▼
                                     text memory  ──►  question answering
```

Each recording is processed **on its own**. The memory is text.

> **Say:** "The pipeline listens to one recording at a time, writes what it heard into a
> text memory, and later answers questions from that memory. Everything works per
> recording."

---

## The problem

Speaker A (recording 1) and Speaker B (recording 2) may be **the same person**.
Nothing in the system knows that.

- Facts about one person can never be joined across recordings
- "What did Mary say about her sister?" → unanswerable unless Mary is named out loud

**Today's fix:** a separate voice-fingerprint model (WeSpeaker) links speakers.
It is 93.5% accurate — but it is an extra component, and it makes mistakes.

> **Say:** "The labels are local to each recording. So the memory holds facts about
> 'Speaker A' fifty times over, never realising it's one person. We already have a
> classical fix — a speaker-embedding model — and the question was whether the speech
> LLM could just do it itself, and let us delete that component."

---

## The idea we tested

Give the model **several past recordings in its context**, labelled, and let it notice
the voices itself:

```
[recording 9] [recording 10] [recording 11] [recording 12]
        ──►  "the woman in 9 is the woman in 12"
```

If this works: one model does everything, no separate speaker system.

> **Say:** "The proposal was simple: modern models have huge context windows, so keep the
> last few recordings in context and let the model do the matching. That's what I set out
> to test."

---

## How I tested it: measure the ability, don't build the system

Building it first tells you nothing when it fails — was it the audio, the window size,
the prompt, the memory, or the question answering?

So: a **ladder**, changing one thing at a time.

| step | what the model hears | isolates |
|---|---|---|
| 1 | 2 clean clips, one voice each | can it match voices at all? |
| 2 | 2 whole conversations | can it separate, *then* match? |
| 3 | 4 whole conversations | the actual proposal |
| 4 | a set of known voices + 1 new voice | the alternative design |
| 5 | 1 conversation | can it cut its own clips? |

Ground truth is free: the dataset's filenames contain the true speaker IDs.

> **Say:** "The key methodological choice. Instead of building the thing and looking at
> end-to-end accuracy, I tested the underlying ability directly, from the easiest version
> to the real one. Each step changes exactly one variable, so a failure points at a cause.
> And the data gives us ground truth for free — 196 recordings, 27 people, each person in
> about 14 different conversations."

---

## What one test looks like

```
[Recording R1]  <10 s of one person>
[Recording R2]  <10 s of one person>

Rate 0-100 how likely it is the SAME PERSON speaks in both.
Judge the voice, not the topic.
```

Real answers:

| truth | model |
|---|---|
| same person | `{"score": 100, "reason": "Identical pitch, rhythm, speaking style"}` ✔ |
| different people | `{"score": 0, "reason": "Different genders, pitches"}` ✔ |
| same person | `{"score": 0, "reason": "Male and female voices"}` ✘ |

> **Say:** "Every test is one prompt with audio in it and a strict JSON answer, so scoring
> is automatic. 200 pairs, balanced, and in 199 of 200 the two clips come from different
> recordings — so this is genuinely cross-recording matching, not the easy case."

---

## Result 1 — it *can* match voices … if you ask properly

| how the same question was asked | accuracy |
|---|---|
| "Same person? yes / no" | 0.595 |
| "Score 0-100, threshold afterwards" | **0.835** |

Same model, same audio. The yes/no version said "yes" on 9% of pairs when the true rate
was 48%.

✅ Not a shortcut: on *same-gender* pairs (the hard ones) it scores **higher**, 0.879.

> **Say:** "First surprise, and a lesson for anything we build with these models: never
> ask for a boolean. The model has the knowledge but a badly placed internal threshold.
> Ask for a score, threshold it ourselves, and accuracy jumps 24 points. And I checked
> the obvious cheat — it isn't just detecting gender; it does better on the hard subset."

---

## Result 2 — it fails on real recordings, and on the actual proposal

| what the model hears | score |
|---|---|
| 2 clean clips, one voice each | **0.835** |
| 2 whole conversations (two voices each) | **0.589** |

**4 recordings at once, "who shares a speaker?":** F1 **0.462**
— worse than answering *"yes"* to everything (0.500).

Also: right number of people 8% of the time; shuffling the input changed the answer on
26 of 34 windows.

> **Say:** "It can compare voices, but it cannot pull one voice out of a two-person
> conversation and then compare it. And that's fatal, because conversations are exactly
> what the proposal would put in the context. When I gave it four recordings and asked it
> to group the speakers, it did worse than a program that always answers 'yes'. So the
> original idea is dead — and it's dead for a specific, understandable reason."

---

## Result 3 — the alternative also hits a wall

Keep a **gallery**: one clean clip per known person; ask "which one is this, or is it
someone new?"

| people in the gallery | picks right person | says "new" correctly | overall |
|---|---|---|---|
| 2 | 0.69 | **0.92** | **0.76** |
| 5 | 0.70 | 0.28 | 0.58 |
| 10 | 0.45 | **0.00** | 0.31 |

At 10 people it **never** says "new" — every stranger is merged into an existing person.

Why: its false-accept rate is ~11% per comparison. A registry of N people needs it to
shrink like 1/N. Real speaker-verification systems run at ~1%.

> **Say:** "Second design: don't show old conversations, show a gallery of known voices.
> This works with two people and collapses by ten. The important part is *what* collapses:
> not the ranking — the right person is still in its top two — but the decision to reject.
> At ten voices it merges every stranger into someone it already knows, which for a memory
> is the worst possible error: two people's facts become one person's. And there's a
> statistical reason it can't scale: identifying among N people needs a per-comparison
> error rate that shrinks with N, and this model is 10-100x away from that."

---

## Result 4 — but it *can* cut its own voice samples

Task: "find one clean 5-10 s stretch per speaker."

| | Omni | existing diarizer |
|---|---|---|
| clip purity (is it really one voice?) | 0.88 | 0.92 |
| covers **both** speakers | 0.31 | 0.50 |

Close on quality, weaker on coverage — it tends to grab two clips from the opening.

> **Say:** "One thing it does well: it can point at clean single-speaker audio. Purity 0.88
> against the classical diarizer's 0.92. That matters, because it means the model can
> prepare the *input* for a speaker system even if it can't do the matching."

---

## So we built it: gallery starts empty, grows as recordings arrive

```
recording arrives ──► EXTRACT one voice sample per speaker
                 ──► MATCH each against the gallery
                 ──► matched: update it   |   new: add it as a reference
```

10 recordings, 17 different people, nothing known in advance:

| who finds the voices | who decides identity | correct people per recording | gallery (true: 17) |
|---|---|---|---|
| existing diarizer | WeSpeaker | **0.92** | 18 |
| existing diarizer | Omni, **one voice at a time** | **0.84** | 16 |
| **Omni** | WeSpeaker | 0.75 | 12 |
| existing diarizer | Omni, all voices at once | 0.53 | 8 |
| Omni | Omni, all voices at once | 0.36 | 7 |

**How you ask matters as much as who you ask**: same model, same audio — comparing against
one reference at a time instead of all of them at once moves 0.53 → 0.84.

**The two jobs fail differently**
- Omni *finding* voices → misses people: it captured both speakers in only 3 of 10
  recordings; in 7 of 10 both its clips were the same person. 5 of 17 people never enter
  the gallery, so they can never be recognised later.
- Omni *deciding* identity → fuses people: the gallery collapses to 7 entries for 17
  people.

**Scaling to 50 recordings (27 people, 25 recurring)** — the two families diverge:

| | 10 recordings | 50 recordings | cost per recording |
|---|---|---|---|
| WeSpeaker | 0.92 | **0.95** ↑ | flat, 7 s |
| Omni, one at a time | 0.84 | 0.76 ↓ | 5 s → **51 s** |

More data helps the classical matcher and hurts the LLM.

> **Say:** "This is the real system: the gallery starts empty and is built from whatever
> the extractor finds. Two independent jobs — find the voices, decide who they are — and
> Omni is weaker at both, but in different ways. As a voice-finder it silently drops
> people: in seven of ten recordings, the two clips it labelled 'speaker 1' and 'speaker 2'
> were the same person. As a decider it fuses people: seventeen people become seven
> identities. The classical components do both jobs correctly."

---

## Does the "reasoning" model fix it?

Same tests, `Qwen3-Omni-Thinking`:

| | Instruct | Thinking |
|---|---|---|
| voice matching (AUC) | **0.835** | 0.625 |
| says "new" correctly | 0.000 | **0.732** |
| window grouping (F1) | 0.462 | 0.462 |
| seconds per window test | 13 s | **554 s** |

Worse at hearing, better at *refusing*, 40× slower. Conclusion unchanged.

⚠️ Trap: with a 3k token limit, **76% of its answers were truncated mid-reasoning** and
silently unscoreable.

> **Say:** "I re-ran everything on the reasoning variant. It's not a better listener — it's
> a more cautious decider. The one thing it gains, refusing to guess, we can get for free
> from a threshold on the classical model. And it's 40x slower, so it's out."

---

## Conclusion

**Don't** let the speech LLM own speaker identity — it is 10–100× short of what
identification at scale requires, and fails in the dangerous direction (merging people).

**Do** use it where it is strong: finding clean voice samples for a classical speaker
model. 0.90 vs 0.97 today, and the gap tracks clip quality — a third of its clips still
contain the other speaker.

Next:
1. Better clips (several per person, drop low-quality ones) → close the 0.06
2. **First**: inject perfect speaker IDs into the memory and see if QA even moves —
   if it doesn't, none of this is worth building

> **Say:** "Two takeaways. Negative: don't give this job to the LLM, and the reason is
> statistical, not fixable by prompting. Positive: it's a good front-end for the classical
> model, six points off today with an obvious lever. And before I spend more time, I want
> to run one cheap check — inject perfect speaker labels into the memory and see whether
> question-answering accuracy actually moves. If it doesn't, we should stop here."

---

# Backup

---

## Method details

- **Data**: Mosaic, bundle_0 — 196 recordings, 27 speakers, ~14 recordings per speaker.
  Ground truth from filenames + per-speaker voice-activity annotations.
- **Model**: Qwen3-Omni-30B-A3B (Instruct and Thinking), served with vLLM, 65k context.
- **Audio cost**: 13 tokens/second, so a 60 s clip ≈ 780 tokens.
- **Scoring**: purity = share of clips landing in a cluster whose majority is their true
  speaker — the same metric the existing system reports.
- Code: `experiment/omni_memory/`; report: `doc/omni_cross_chunk_speaker_experiment.md`.

---

## Two mistakes the controls caught

1. **A yes/no prompt hid the model's ability.** First run said 0.595 accuracy; the same
   comparison asked as a score gave 0.835.

2. **My first window test was rigged in the model's favour.** Built from consecutive
   recordings, it scored F1 0.913 — but the dataset orders recordings by speaker pair, so
   88% of pairs really did share a speaker and "always yes" scored 0.935, *beating* the
   model. Rebuilding the set balanced dropped it to 0.462.

**Rule adopted:** always print the trivial baseline, and always include a control that can
only be failed (e.g. windows where nobody repeats).

---

## Why it degrades: voices in context, not context length

| configuration | voices in prompt | prompt tokens | works? |
|---|---|---|---|
| pair of clean clips | 2 | 0.5k | yes (0.835) |
| gallery of 2 + query | 3 | 0.7k | yes (0.758) |
| gallery of 10 + query | 11 | 1.8k | no (0.313) |
| window of 4 recordings | 8 | 3.4k | no (0.462) |

Nothing is near the 65k limit. Cost isn't the constraint — decision quality is.

---

## Caveats

- One dataset (naturalistic two-person conversations), one model family, zero-shot.
- Enrolment uses one ~10 s clip per person per recording, no quality filtering — the most
  obvious improvement is untested.
- The 8k-token Thinking window run is n=9 (10 min per test).
- Prompts were tuned a little, not exhaustively — and slide 7 shows phrasing is worth a
  lot, so the failing configurations might improve somewhat. They would have to improve a
  great deal: window grouping is *below* trivial baselines, not marginally behind.
