import pandas as pd
import json
import os
import glob
import jsonlines
import string

def chunk_dialog(dialog, min_dur=60.0, max_dur=600.0, gap_threshold=4.0):
    """
    Split a flat list of utterance dicts into smaller conversation chunks.

    Strategy:
      1. A natural boundary is detected when the silence gap between two
         consecutive utterances exceeds `gap_threshold` seconds.
      2. A chunk is cut at the earliest natural boundary AFTER `min_dur` seconds
         have accumulated, or hard-cut at `max_dur` regardless.
      3. If no boundary is found before `max_dur`, the chunk is cut between the
         two utterances whose gap is the largest within the window.

    Args:
        dialog: list of utterance dicts with keys 'start', 'end', 'text', 'speaker'.
        min_dur: minimum chunk duration in seconds.
        max_dur: maximum chunk duration in seconds.
        gap_threshold: silence gap (seconds) that marks a natural boundary.

    Returns:
        List of lists, each inner list is a group of utterance dicts.
    """
    if not dialog:
        return []

    chunks = []
    chunk_start = 0  # index into dialog

    while chunk_start < len(dialog):
        chunk_begin_time = dialog[chunk_start]["start"]
        best_gap_idx = None   # index of utterance *after* which to cut (largest gap)
        best_gap_val = -1.0

        i = chunk_start
        max_end_so_far = dialog[chunk_start]["end"]
        while i < len(dialog):
            max_end_so_far = max(max_end_so_far, dialog[i]["end"])
            elapsed = max_end_so_far - chunk_begin_time

            # Compute gap to next utterance (if any)
            if i + 1 < len(dialog):
                gap = dialog[i + 1]["start"] - max_end_so_far
            else:
                gap = 0.0

            # Track largest gap seen so far (fallback hard-cut)
            if gap > best_gap_val:
                best_gap_val = gap
                best_gap_idx = i

            # Natural boundary after min_dur
            if elapsed >= min_dur and gap >= gap_threshold:
                chunks.append(dialog[chunk_start: i + 1])
                chunk_start = i + 1
                break

            # Hard cut at max_dur
            if elapsed >= max_dur:
                # Cut at the largest gap seen within the window
                cut_at = best_gap_idx if best_gap_idx is not None else i
                chunks.append(dialog[chunk_start: cut_at + 1])
                chunk_start = cut_at + 1
                break

            i += 1
        else:
            # Reached end of dialog — append remaining utterances to the last chunk
            if chunks:
                chunks[-1] = chunks[-1] + dialog[chunk_start:]
            else:
                chunks.append(dialog[chunk_start:])
            break

    return chunks


def fix_space_in_text(text):
    punctuation_pattern = [" " + c for c in string.punctuation]
    punctuation_pattern.append(" n't")
    # punctuation_pattern.extend([" 'm", " 's", " 've", " 're", " 'll", " 'd", " 'n", " 't", " 'y", " 'z"])
    for pattern in punctuation_pattern:
        text = text.replace(pattern, pattern.strip())
    return text


anonymized = True
use_gt_speaker = False

data_folder = "outputs/bazinga_data/"
Season = "Season1"
split = "gt" # "pred"
if split == "gt":
    speaker_map = None
else:
    with open(os.path.join(data_folder, "speaker_map.json"), "r") as f:
        speaker_map = json.load(f)


if anonymized:
    qa_files = [
        "QA_designs/TV_series/TBBT_s1/speaker_atributted_qa_anonymized.jsonl",
        "QA_designs/TV_series/TBBT_s1/long_context_QA_anonymized.jsonl",
    ]
    input_folder= "QA_designs/TV_series/TBBT_s1/S1_main_anon/"
    output_path = os.path.join(data_folder, f"{Season}_{split}_SpeakerLabel{use_gt_speaker}_anony.parquet")

else:
    qa_files = [
        "QA_designs/TV_series/TBBT_s1/speaker_atributted_qa_deanonymized.jsonl",
        "QA_designs/TV_series/TBBT_s1/long_context_QA_unmasked.jsonl",
    ]
    input_folder = f"outputs/bazinga/TheBigBangTheory/{Season}"
    output_path = os.path.join(data_folder, f"{Season}_{split}_SpeakerLabel{use_gt_speaker}.parquet")


timestamp_file = "QA_designs/TV_series/TBBT_s1/S1_session_timeline.json"







'''
# load timestamp_file to json
with open(timestamp_file, "r") as f:
    time_info = json.load(f)
time_info = time_info["sessions"]
time_maps = {}
for item in time_info:
    time_maps[item["source_file"]] = item["session_timeline_date"]

qas = []
for qa_file in qa_files:
    question_type = os.path.basename(qa_file).replace(".jsonl", "")
    # read jsonl file
    print(qa_file)
    with jsonlines.open(qa_file) as reader:
        qa = [line for line in reader]

    for qa_item in qa:
        question = qa_item["question"]
        question_type = qa_item["category"]
        if "options" in qa_item.keys():
            options = qa_item["options"]
            for k,v in options.items():
                question += f"\n{k}. {v}"
            question += "\nC. not sure"
        qas.append({
            "question": question,
            "answer": qa_item["answer"],
            "type": question_type,
            "gt_source": qa_item["gt_source"],
        })
print(qas[0])
# list all the jsonl file in the input_folder
jsonl_files = sorted(
    glob.glob(os.path.join(input_folder, "*.json"))
)
print(f"Found {len(jsonl_files)} jsonl files")


MIN_CHUNK_DUR = 60.0    # seconds
MAX_CHUNK_DUR = 300.0   # seconds
GAP_THRESHOLD = 3.0     # seconds

speakers_pool = {}
chunks = []
for idx, json_file in enumerate(jsonl_files):
    session_name = os.path.basename(json_file)
    timestamp = time_maps[session_name]

    with open(json_file, "r") as f:
        dialog = json.load(f)

    sub_chunks = chunk_dialog(dialog, min_dur=MIN_CHUNK_DUR, max_dur=MAX_CHUNK_DUR, gap_threshold=GAP_THRESHOLD)

    for sub in sub_chunks:
        dialog_chunk = f"[Dialogue between multiple people on {timestamp}]\n"
        for turn in sub:
            if speaker_map is None:
                speaker = turn['speaker']
            else:
                speaker = speaker_map[turn['speaker']]

            if not use_gt_speaker:
                if speaker not in speakers_pool:
                    speakers_pool[speaker] = len(speakers_pool)
                speaker = "Speaker_" + str(speakers_pool[speaker])

            turn_text = fix_space_in_text(turn['text'])
            dialog_chunk += f"<{speaker}> {turn_text}\n"

        # print(dialog_chunk)
        # print("-"*100)
        chunks.append(dialog_chunk)


print("speakers_pool = ", speakers_pool)
samples = []
samples.append({
    "instance_id": 0,
    "prompt": "I will provide you with the conversation history between the different speakers and I need you to remember the details of the conversation for future reference.",
    "chunks": json.dumps(chunks),
    "questions_and_answers": json.dumps(qas),
    "data_source": "seamlessinteraction",
    "metadata": {"data_source": "seamlessinteraction", "metadata": "{}", "sample_id": 0},
    "num_chunks": len(chunks),
    "num_questions": len(qas),
})

df = pd.DataFrame(samples)
df.to_parquet(output_path, index=False)
print(f"Saved {len(samples)} samples to {output_path}")
'''
