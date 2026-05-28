"""
Build a Parquet dataset from pre-chunked dialogue folders and speaker_name_map.json.

Input folder layout (same as step3_speaker_name_extract.py):
    data_dir/{show}/{episode}/parsed_dialog_pred.json   — one chunk per file
    data_dir/speaker_name_map.json                      — produced by step3

Each parsed_dialog_pred.json is a list of utterance dicts:
    [{"speaker": "SPEAKER_00", "text": "...", "start": 0.0, "end": 1.5}, ...]

Output: a single Parquet file with columns:
    instance_id, prompt, chunks, data_source, metadata, num_chunks
(no QA columns)
"""

import argparse
import glob
import json
import os
import string
import sys

import pandas as pd
from prepare_data.preprocess_utils import fix_space_in_text


def load_chunks_pred(data_dir: str, use_extracted_name: bool = False) -> list[str]:
    map_path = os.path.join(args.data_dir, "speaker_name_map.json")
    with open(map_path, "r") as f:
        speaker_name_map = json.load(f)
    
    # time_info_path = os.path.join(data_dir, "time_info.json")
    time_info_path = "QA_designs/bazinga/TBBT/s1_arc/S1_session_timeline.json"
    with open(time_info_path, "r") as f:
        time_maps = json.load(f)
    
    print(f"Loaded speaker name map with {len(speaker_name_map)} entries")

    subfolders = sorted(
        glob.glob(os.path.join(data_dir, "*", "*", "parsed_dialog_pred.json"))
    )
    print(f"Found {len(subfolders)} dialogue files")

    chunks = []
    for path in subfolders:
        with open(path, "r") as f:
            dialog = json.load(f)
        
        session_name = os.path.basename(os.path.dirname(path))
        timestamp = time_maps[session_name]

        dialog_chunk = f"[Dialogue between multiple people on {timestamp}]\n"
        for turn in dialog:
            raw_speaker = turn["speaker"]
            if use_extracted_name:
                speaker = speaker_name_map.get(raw_speaker, raw_speaker)
            else:
                speaker = raw_speaker

            turn_text = fix_space_in_text(turn["text"])
            dialog_chunk += f"<{speaker}> {turn_text}\n"

        chunks.append(dialog_chunk)

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Parquet from pre-chunked dialogues + speaker name map"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root folder with {show}/{episode}/parsed_dialog_pred.json",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output parquet path (default: <data_dir>/dataset.parquet)",
    )
    args = parser.parse_args()

    chunks = load_chunks_pred(args.data_dir)
    print(f"Loaded {len(chunks)} chunks")

    samples = [{
        "instance_id": 0,
        "prompt": "I will provide you with the conversation history between the "
                  "different speakers and I need you to remember the details of "
                  "the conversation for future reference.",
        "chunks": json.dumps(chunks),
        "data_source": "seamlessinteraction",
        "metadata": json.dumps({"data_source": "seamlessinteraction", "sample_id": 0}),
        "num_chunks": len(chunks),
    }]

    output_path = args.output or os.path.join(args.data_dir, "dataset.parquet")
    df = pd.DataFrame(samples)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(samples)} samples ({len(chunks)} chunks) to {output_path}")


if __name__ == "__main__":
    main()
