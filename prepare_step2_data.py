import pandas as pd
import json
import os
import glob
import jsonlines


data_folder = "outputs/demo_output_step2"
qa_folder = "outputs/QA_pairs_generated"
dialog = "parsed_dialog_pred.json"

if dialog == "parsed_dialog_gt.json":
    speaker_map = None
else:
    with open(os.path.join(data_folder, "speaker_map.json"), "r") as f:
        speaker_map = json.load(f)


output_path = os.path.join("outputs/test_pred.parquet")


## list all the json files in the qa_folder
qa_files = sorted(
    glob.glob(os.path.join(qa_folder, "*.jsonl"))
)
print(f"Found {len(qa_files)} qa files")
# print(qa_files)
qas = []
speaker_pairs = []
for qa_file in qa_files:
    question_type = os.path.basename(qa_file).replace(".jsonl", "")
    # read jsonl file
    with jsonlines.open(qa_file) as reader:
        qa = [line for line in reader]
    
    for qa_item in qa:
        qas.append({
            "question": qa_item["question"],
            "answer": qa_item["answer"],
            "type": question_type,
            "gt_source": qa_item["gt_source"],
        })
        # print(qa_item["gt_source"])
        pair_id = qa_item["gt_source"]["file"]#.split("/")[0]
        if pair_id.startswith("P"):
            spk1, spk2 = pair_id.split("/")[0].split("_")
        else:
            spk1, spk2 = pair_id.split("/")[1].split("_")
        if spk1 not in speaker_pairs:
            speaker_pairs.append(spk1)
        if spk2 not in speaker_pairs:
            speaker_pairs.append(spk2)
print(speaker_pairs)
speaker_pairs.extend(["P0043", "P0108", "P1297"])

# only include the parsed_dialog_pred.json in the speaker_pairs list  

# list all the folders in the data_folder
pairs_folder = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]

valid_folder = []
for spk_pair in pairs_folder:
    spk1, spk2 = spk_pair.split("_")
    if spk1 in speaker_pairs or spk2 in speaker_pairs:
        valid_folder.append(spk_pair)
print(valid_folder)

json_files = []
for folder in valid_folder:
    # list all parsed_dialog_pred.json under folder in recursive way
    parsed_dialog_pred_files = glob.glob(os.path.join(data_folder, folder, "*", dialog), recursive=True)
    json_files.extend(parsed_dialog_pred_files)


# json_files = sorted(
#     glob.glob(os.path.join(data_folder, "**", "parsed_dialog_pred.json"), recursive=True)
# )
print(f"Found {len(json_files)} {dialog} files")

chunks = []
for idx, json_file in enumerate(json_files):
    with open(json_file, "r") as f:
        dialog = json.load(f)
    dialog_chunk = ""
    for turn in dialog:
        if speaker_map is None:
            speaker = turn['speaker']
        else:
            speaker = speaker_map[turn['speaker']]
        dialog_chunk += f"<{speaker}> {turn['text']}\n"
    chunks.append(dialog_chunk)


samples = []
samples.append({
    "instance_id": 0,
    "prompt": "I will provide you with the conversation history between the different speakers and I need you to remember the details of the conversation for future reference.",
    "chunks": json.dumps(chunks),
    "questions_and_answers": json.dumps(qas),
    "data_source": "seamlessinteraction_gt",
    "metadata": {"data_source": "seamlessinteraction_gt", "metadata": "{}", "sample_id": 0},
    "num_chunks": len(chunks),
    "num_questions": len(qas),
})

df = pd.DataFrame(samples)
df.to_parquet(output_path, index=False)
print(f"Saved {len(samples)} samples to {output_path}")
