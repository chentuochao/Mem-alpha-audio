import pandas as pd
import json
import os


data_folder = "mytest"

# list all the txt file in the data_folder
txt_files = [f for f in os.listdir(data_folder) if f.endswith('.txt')]


samples = []
for idx, txt_file in enumerate(txt_files):
    with open(os.path.join(data_folder, txt_file), 'r') as f:
        text = f.readlines()
    qa_json = txt_file.replace(".txt", "_qa.json")

    with open(os.path.join(data_folder, qa_json), 'r') as f:
        qa = json.load(f)

    chunks = []
    for line in text:
        # remove the first appear ":"
        line = line.split(": ", 1)[1]
        chunks.append(line)
    print(chunks)
    print(qa)
    samples.append({
        "instance_id": idx,
        "prompt": "I will provide you with the conversation history between the user and the assistant and I need you to remember the details of the conversation for future reference.",
        "chunks": json.dumps(chunks),
        "questions_and_answers": json.dumps(qa),
        "data_source": "my_custom_source",  # must match prompts_wrt_datasource.yaml
        "metadata": {"data_source": "my_custom_source", "metadata": "{}"},
        "num_chunks": len(chunks),
        "num_questions": len(qa),
    })


df = pd.DataFrame(samples)
df.to_parquet( os.path.join(data_folder, "test.parquet"), index=False)
