"""
Test script for Speaker_Name_tracker.py

Builds dialogue chunks from the Big Bang Theory Season 1 data
(same pipeline as prepare_bazinga_data.py), then runs the
speaker name tracker to resolve anonymous Speaker_X labels
to real character names.

Usage:
    # Set env vars for your Qwen endpoint, e.g.:
    export QWEN_URL="http://localhost:8000/v1"
    export OPENROUTER_API_KEY="EMPTY"
    export QWEN_MODEL_NAME="Qwen/Qwen3-32B"

    python mytest/test_speaker_name_tracker.py
"""

import glob
import json
import os
import sys
import string
import jsonlines
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import OpenAI
from audio_script.Speaker_Track.Speaker_Name_tracker import (
    identify_speakers,
    resolve_transcript,
    SpeakerRecord,
    update_registry,
    qwen3_chat,
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_prompt,
    strip_thinking,
)
from prepare_data.preprocess_utils import load_episode_chunks, fix_space_in_text


# ── Build chunks from episode data (mirrors prepare_bazinga_data.py) ─────────


# ── Test 4: End-to-end with real Qwen API call on episode data ───────────────

def test_e2e_speaker_identification(dialogue_folder, time_maps):
    jsonl_files = sorted(
        glob.glob(os.path.join(dialogue_folder, "*.json"))
    )
    print(f"Found {len(jsonl_files)} jsonl files")

    chunks, speakers_pool = load_episode_chunks(jsonl_files, time_maps)
    print(f"Loaded {len(chunks)} chunks, {len(speakers_pool)} unique speakers")

    # Override the module-level client/model to use the same env vars as evaluate_agent_results.py
    import audio_script.Speaker_Track.Speaker_Name_tracker as tracker
    tracker.client = OpenAI(
        base_url=os.getenv("QWEN_URL", os.getenv("QWEN_BASE_URL", "http://localhost:8000/v1")),
        api_key=os.getenv("OPENROUTER_API_KEY", os.getenv("QWEN_API_KEY", "EMPTY")),
    )
    tracker.QWEN3_MODEL = os.getenv("QWEN_MODEL_NAME", os.getenv("QWEN3_MODEL", "Qwen/Qwen3-32B"))

    registry = identify_speakers(chunks, enable_thinking=False)

    print("\n── Final Registry ──")
    for sid, rec in sorted(registry.items()):
        print(f"  {sid:12s} → {rec.display_name:20s} [{rec.status}]  (evidence: {len(rec.evidence)})")

    # Check that at least some speakers were identified
    identified = [sid for sid, rec in registry.items() if rec.name is not None]
    print(f"\nIdentified {len(identified)} / {len(registry)} speakers")

    # Verify against ground truth
    gt_reverse = {v: k for k, v in speakers_pool.items()}
    for sid, rec in registry.items():
        if rec.name:
            speaker_idx = int(sid.split("_")[1])
            if speaker_idx in gt_reverse:
                gt_name = gt_reverse[speaker_idx]
                firstname_rec = rec.name.split()[0]
                match = (
                    rec.name.lower() in gt_name.lower()
                    or gt_name.lower().startswith(rec.name.lower())
                    or gt_name.lower().startswith(firstname_rec.lower())
                )
                status = "✓" if match else "✗"
                print(f"  {status} [{rec.status}] {sid}: predicted={rec.name}, ground_truth={gt_name}")

    # Reassign speaker labels in every chunk:
    #   identified   → real name        e.g. <Sheldon>
    #   unidentified → Unknown_<sid>    e.g. <Unknown_Speaker_2>
    def _resolved_label(sid: str, rec: "SpeakerRecord") -> str:
        return rec.name if rec.name else f"Unknown_{sid}"
    print(speakers_pool)
    print(registry)

    resolved_chunks = []
    for chunk in chunks:
        # replace <Speaker_X> labels with resolved names
        for sid, rec in registry.items():
            chunk = chunk.replace(f"<{sid}>", f"<{_resolved_label(sid, rec)}>")

        # speakers_pool may contain speakers the tracker never saw (no registry entry);
        # replace their raw <Speaker_X> labels with <Unknown_Speaker_X>.
        for _, idx in speakers_pool.items():
            sid = f"Speaker_{idx}"
            if sid not in registry:
                chunk = chunk.replace(f"<{sid}>", f"<Unknown_{sid}>")
        resolved_chunks.append(chunk)

    # print(f"\n── Resolved transcripts ({len(resolved_chunks)} chunks) ──")
    # for i, chunk in enumerate(resolved_chunks):
    #     print(f"\n[Chunk {i}]\n{chunk[:500]}")

    assert len(identified) > 0, "Expected at least one speaker to be identified"
    print("[PASS] test_e2e_speaker_identification")

    return resolved_chunks, speakers_pool



def test_e2e_use_gt_speaker(dialogue_folder, time_maps):
    """Build resolved chunks using ground-truth speaker identities.

    No LLM call is made. Each <Speaker_X> label is replaced directly with
    the real character name derived from speakers_pool (e.g. 'sheldon_cooper'
    → 'Sheldon Cooper').
    """
    jsonl_files = sorted(glob.glob(os.path.join(dialogue_folder, "*.json")))
    print(f"Found {len(jsonl_files)} jsonl files")

    chunks, speakers_pool = load_episode_chunks(jsonl_files, time_maps, use_gt_speaker=True)
    print(f"Loaded {len(chunks)} chunks, {len(speakers_pool)} unique speakers")
    print(speakers_pool)
    exit(0)

    return chunks, speakers_pool


# ── Save resolved chunks to parquet ─────────────────────────────────────────

def save_parquet(resolved_chunks, qas, output_path):
    samples = [{
        "instance_id": 0,
        "prompt": "I will provide you with the conversation history between the different speakers and I need you to remember the details of the conversation for future reference.",
        "chunks": json.dumps(resolved_chunks),
        "questions_and_answers": json.dumps(qas),
        "data_source": "seamlessinteraction",
        "metadata": {"data_source": "seamlessinteraction", "metadata": "{}", "sample_id": 0},
        "num_chunks": len(resolved_chunks),
        "num_questions": len(qas),
    }]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(samples)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(samples)} samples ({len(resolved_chunks)} chunks, {len(qas)} QA pairs) to {output_path}")


def parse_qa(qa_files):
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
    return qas

if __name__ == "__main__":
    print("=" * 60)
    print("Running Speaker Name Tracker Tests")
    print("=" * 60)

    anonymized = False
    use_gt_speaker = True

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
            "QA_designs/bazinga/TBBT/s1_arc/speaker_atributted_qa_anonymized.jsonl",
            "QA_designs/bazinga/TBBT/s1_arc/long_context_QA_anonymized.jsonl",
        ]
        input_folder= "QA_designs/bazinga/TBBT/s1_arc/S1_main_anon/"
        output_path = os.path.join(data_folder, f"{Season}_{split}_SpeakerLabel{use_gt_speaker}_anony.parquet")

    else:
        qa_files = [
            "QA_designs/bazinga/TBBT/s1_arc/speaker_atributted_qa_deanonymized.jsonl",
            "QA_designs/bazinga/TBBT/s1_arc/long_context_QA_unmasked.jsonl",
        ]
        input_folder = f"outputs/bazinga/TheBigBangTheory/{Season}"
        output_path = os.path.join(data_folder, f"{Season}_{split}_SpeakerLabel{use_gt_speaker}.parquet")


    timestamp_file = "QA_designs/bazinga/TBBT/s1_arc/S1_session_timeline.json"
    with open(timestamp_file, "r") as f:
        time_info = json.load(f)
    time_info = time_info["sessions"]
    time_maps = {}
    for item in time_info:
        time_maps[item["source_file"]] = item["session_timeline_date"]


    qas = parse_qa(qa_files)

    # Online tests (require Qwen API)
    # test_qwen_api_connection()
    if use_gt_speaker:
        resolved_chunks, speakers_pool = test_e2e_use_gt_speaker(input_folder, time_maps)
    else:
        resolved_chunks, speakers_pool = test_e2e_speaker_identification(input_folder, time_maps)
    save_parquet(resolved_chunks, qas, output_path)

    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)
