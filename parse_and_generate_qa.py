import json
import os

DATA_JSON = "data/global_speaker_results.json"
OUTPUT_DIR = "mytest"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(DATA_JSON, "r") as f:
    data = json.load(f)

per_audio = data["per_audio_results"]

merged_lines = []
audio_count = 0
for audio_file, audio_data in per_audio.items():
    local_to_global = audio_data["local_to_global"]
    local_speakers = audio_data["local_speakers"]

    if not local_to_global or not local_speakers:
        continue

    all_segments = []
    for local_id, speaker_data in local_speakers.items():
        global_id = local_to_global.get(local_id)
        if global_id is None or not speaker_data["segments"]:
            continue
        for seg in speaker_data["segments"]:
            if seg["words"].strip():
                all_segments.append({
                    "global_id": global_id,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["words"].strip(),
                })

    if not all_segments:
        continue

    all_segments.sort(key=lambda s: s["start"])

    for seg in all_segments:
        merged_lines.append(f"{seg['global_id']}: {seg['text']}")

    audio_count += 1

txt_path = os.path.join(OUTPUT_DIR, "conv_000.txt")
with open(txt_path, "w") as f:
    f.write("\n".join(merged_lines) + "\n")

print(f"Merged {audio_count} audio conversations into 1 file: {txt_path}")
print(f"Total lines (chunks): {len(merged_lines)}")
print()
for i, line in enumerate(merged_lines):
    speaker, text = line.split(": ", 1)
    print(f"  [{i}] {speaker}: {text[:100]}...")

global_speakers = data["global_speakers"]
print("\n" + "=" * 80)
print("GLOBAL SPEAKER SUMMARY")
print("=" * 80)
for gid, gdata in global_speakers.items():
    full_text = " ".join(
        t["text"] for t in gdata["transcriptions"] if t["text"].strip()
    )
    word_count = len(full_text.split())
    print(f"\n{gid} (appears in {gdata['weight']} audio files, ~{word_count} words):")
    print(f"  {full_text[:300]}...")


# ---------------------------------------------------------------------------
# Design speaker-aware QA for each conversation
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 80)
print("GENERATING MERGED SPEAKER-AWARE QA")
print("=" * 80)

all_qa = [
    # --- Topic: childhood memories, acting, Christmas (audio 125) ---
    {
        "question": "What musical did GLOBAL_SPK_0 perform in during fourth grade?",
        "answer": "A Rumpus in the Rainforest.",
        "type": "speaker_attribution",
    },
    {
        "question": "What character did GLOBAL_SPK_0 play in the fourth-grade musical?",
        "answer": "The Tree Boa.",
        "type": "speaker_specific_fact",
    },
    {
        "question": "Why did GLOBAL_SPK_0 get to sing the whole song alone during the final performance?",
        "answer": "Because the other Triboa character didn't come to school that day, and GLOBAL_SPK_0 had prepared both parts.",
        "type": "speaker_specific_reasoning",
    },
    {
        "question": "Why was GLOBAL_SPK_0's older sister annoyed at GLOBAL_SPK_0?",
        "answer": "Because GLOBAL_SPK_0 kept using the boom box in her room to listen to the CD and practice the songs for the musical.",
        "type": "speaker_specific_fact",
    },
    {
        "question": "What childhood memory does GLOBAL_SPK_1 associate with Christmas?",
        "answer": "Watching Santa's location being tracked on the live map on TV news (channel five or channel seven news), showing Santa in his sleigh with reindeers and presents.",
        "type": "speaker_attribution",
    },
    {
        "question": "Is GLOBAL_SPK_1 a morning person?",
        "answer": "No, GLOBAL_SPK_1 says they are not a morning person at all.",
        "type": "speaker_specific_fact",
    },
    {
        "question": "Who ruined Santa for GLOBAL_SPK_0, and how?",
        "answer": "GLOBAL_SPK_0's older sister told them that Santa is not real.",
        "type": "speaker_specific_fact",
    },
    {
        "question": "What did GLOBAL_SPK_0 do to their older sister on Christmas morning in first grade?",
        "answer": "GLOBAL_SPK_0 woke up early, came into her sister's room, poked her in the eye, and said 'it's Christmas.'",
        "type": "speaker_specific_fact",
    },
    {
        "question": "Which speaker talked about siblings not wanting to share with each other?",
        "answer": "GLOBAL_SPK_1.",
        "type": "speaker_identification",
    },
    {
        "question": "How did GLOBAL_SPK_0 respond when their parents asked them to write a Christmas list after learning Santa might not be real?",
        "answer": "GLOBAL_SPK_0 said 'Santa would know,' suggesting they didn't fully believe Santa was not real yet.",
        "type": "speaker_specific_reasoning",
    },
    # --- Topic: fitness challenge ideas (audio 126) ---
    {
        "question": "What fitness activity did GLOBAL_SPK_1 want to start doing as part of the fitness challenge?",
        "answer": "Jump roping / jump rope.",
        "type": "speaker_specific_fact",
    },
    {
        "question": "Which speaker suggested adding a meditation class after each activity?",
        "answer": "GLOBAL_SPK_0.",
        "type": "speaker_identification",
    },
    {
        "question": "Why did GLOBAL_SPK_0 not take a second boxing class?",
        "answer": "Because they would have to wrap their hands by themselves and didn't know how to do that.",
        "type": "speaker_specific_fact",
    },
    {
        "question": "Why did GLOBAL_SPK_0 sign up for a boxing class?",
        "answer": "They wanted to blow off some steam.",
        "type": "speaker_specific_reasoning",
    },
    {
        "question": "What was GLOBAL_SPK_0's overall experience with the boxing class?",
        "answer": "It was a good experience — really intimidating because people were at different skill levels, but overall a high energy and really cool activity.",
        "type": "speaker_specific_fact",
    },
    {
        "question": "Which sports did GLOBAL_SPK_1 suggest for the fitness challenge?",
        "answer": "Jump rope, yoga, basketball, soccer, tennis, table tennis, and swimming.",
        "type": "speaker_specific_fact",
    },
    {
        "question": "What does GLOBAL_SPK_1 love about swimming?",
        "answer": "Getting into the water and feeling invigorated from the cold water, which makes their skin feel really good.",
        "type": "speaker_specific_fact",
    },
    {
        "question": "Which speaker mentioned wanting to punch someone in the face as motivation for boxing?",
        "answer": "GLOBAL_SPK_1.",
        "type": "speaker_identification",
    },
    # --- Topic: dreams, storytelling, Inside Out (audio 128) ---
    {
        "question": "What is GLOBAL_SPK_1's most profound dream?",
        "answer": "To tell stories that make an impact on people — stories that are creative, inspirational, and that change lives, not coming from a place of fear or judgment.",
        "type": "speaker_specific_fact",
    },
    {
        "question": "What kind of movie or story would GLOBAL_SPK_1 dream of being part of?",
        "answer": "Any Disney story or Disney movie, because they are warm, fuzzy, and leave you feeling happy and inspired. Specifically, GLOBAL_SPK_1 mentioned Inside Out 2.",
        "type": "speaker_specific_fact",
    },
    {
        "question": "Has GLOBAL_SPK_0 seen Inside Out or Inside Out 2?",
        "answer": "No, GLOBAL_SPK_0 has never seen Inside Out or Inside Out 2.",
        "type": "speaker_specific_fact",
    },
    {
        "question": "What did GLOBAL_SPK_0 connect with in GLOBAL_SPK_1's answer about dreams?",
        "answer": "The part about anxiety sometimes being a good thing — GLOBAL_SPK_0 thinks it shows that you care.",
        "type": "cross_speaker_reasoning",
    },
    {
        "question": "Who asked 'What is your most profound dream?'",
        "answer": "GLOBAL_SPK_0.",
        "type": "speaker_identification",
    },
    {
        "question": "What does GLOBAL_SPK_1 say about anxiety?",
        "answer": "A little bit of anxiety didn't hurt anyone and sometimes a little anxiety is actually kind of good.",
        "type": "speaker_specific_fact",
    },
    # --- Topic: regrets, apologies, twin sister, play (audio 130) ---
    {
        "question": "What does GLOBAL_SPK_0 regret about seventh grade?",
        "answer": "Staying friends with two girls who were mean to their twin sister, just because it felt good to feel wanted and liked.",
        "type": "speaker_specific_fact",
    },
    {
        "question": "Has GLOBAL_SPK_0 apologized to their twin sister?",
        "answer": "Yes, GLOBAL_SPK_0 has already apologized because they've been in each other's lives the whole time.",
        "type": "speaker_specific_fact",
    },
    {
        "question": "What did the main girl in the friend group do every Monday regarding GLOBAL_SPK_0's twin?",
        "answer": "She would ask 'how's the Rene drama,' basically trying to egg GLOBAL_SPK_0 on and get information about issues with their twin.",
        "type": "speaker_specific_fact",
    },
    {
        "question": "What play did GLOBAL_SPK_0 and their twin audition for?",
        "answer": "Their high school play during GLOBAL_SPK_0's junior year (winter).",
        "type": "speaker_specific_fact",
    },
    {
        "question": "What roles did GLOBAL_SPK_0 and their twin get in the play?",
        "answer": "GLOBAL_SPK_0 played the mistress and their twin played the wife.",
        "type": "speaker_specific_fact",
    },
    {
        "question": "Why were GLOBAL_SPK_0 and their twin worried about both being cast?",
        "answer": "Because they look too alike and the cast was small with not many family member characters in the script.",
        "type": "speaker_specific_reasoning",
    },
    {
        "question": "What misunderstanding happened about GLOBAL_SPK_0's comment on the mistress character?",
        "answer": "GLOBAL_SPK_0 said the mistress character isn't that emotional, and it got twisted into 'GLOBAL_SPK_0 doesn't think actors should have emotion,' which wasn't what they meant.",
        "type": "speaker_specific_fact",
    },
    {
        "question": "What does GLOBAL_SPK_1 find fun about the relationship between GLOBAL_SPK_0 and their sister?",
        "answer": "That when the sister helps GLOBAL_SPK_0 rehearse lines, GLOBAL_SPK_0 tells the sister to stop giving unsolicited feedback and just read the lines.",
        "type": "cross_speaker_reasoning",
    },
    {
        "question": "Which speaker has a twin sister?",
        "answer": "GLOBAL_SPK_0.",
        "type": "speaker_identification",
    },
    {
        "question": "What does GLOBAL_SPK_0 prefer about acting style for film and TV?",
        "answer": "GLOBAL_SPK_0 prefers real-sounding conversation over sounding theatrical/like someone with a theater degree.",
        "type": "speaker_specific_fact",
    },
]

qa_path = os.path.join(OUTPUT_DIR, "conv_000_qa.json")
with open(qa_path, "w") as f:
    json.dump(all_qa, f, indent=2)

print(f"  Written {qa_path} ({len(all_qa)} questions)")
print(f"\nDone! All files saved to '{OUTPUT_DIR}/'")
print(f"  - 1 merged .txt file  (conv_000.txt, {len(merged_lines)} chunks)")
print(f"  - 1 merged _qa.json   (conv_000_qa.json, {len(all_qa)} questions)")
print(f"\nYou can now run prepare_my_data.py (data_folder='mytest') to generate test.parquet")
