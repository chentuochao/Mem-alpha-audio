import pandas as pd
import json
import os
import glob
import jsonlines
import string
from .preprocess_utils import chunk_dialog, fix_space_in_text



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
