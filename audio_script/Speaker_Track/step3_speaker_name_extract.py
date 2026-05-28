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
def fix_space_in_text(text):
    punctuation_pattern = [" " + c for c in string.punctuation]
    punctuation_pattern.append(" n't")
    # punctuation_pattern.extend([" 'm", " 's", " 've", " 're", " 'll", " 'd", " 'n", " 't", " 'y", " 'z"])
    for pattern in punctuation_pattern:
        text = text.replace(pattern, pattern.strip())
    return text


# def load_dialog(dialog):
#     dialog_chunk = ""
#     for turn in dialog:
#         speaker = turn["speaker"]
#         if speaker not in speakers_pool:
#             speakers_pool[speaker] = len(speakers_pool)
#         anon_speaker = "Speaker_" + str(speakers_pool[speaker])
#         turn_text = fix_space_in_text(turn["text"])

#         if use_gt_speaker:
#             dialog_chunk += f"<{speaker}> {turn_text}\n"
#         else:
#             dialog_chunk += f"<{anon_speaker}> {turn_text}\n"
#     chunks.append(dialog_chunk)

def load_gt_episode_chunks(dialogue_folder):
    ### find all subfolder with {xxx}/{xxx}/parsed_dialog_gt.json sub-folders
    subfolders = glob.glob(os.path.join(dialogue_folder, "*", "*", "parsed_dialog_gt.json"))
    print(f"Found {len(subfolders)} subfolders")

    folder_names = []
    speakers_pool = {}
    chunks = []

    for subfolder in subfolders:
        folder_name = os.path.dirname(subfolder)
        folder_names.append(folder_name)

        gt_file = os.path.join(folder_name, "parsed_dialog_gt.json")
        with open(gt_file, "r") as f:
            dialog = json.load(f)
        
        dialog_chunk = f"[Dialogue between multiple people]\n"
        for turn in dialog:
            speaker = turn["speaker"]
            if speaker not in speakers_pool:
                speakers_pool[speaker] = len(speakers_pool)
            anon_speaker = "Speaker_" + str(speakers_pool[speaker])
            turn_text = fix_space_in_text(turn["text"])
            
            dialog_chunk += f"{anon_speaker}: {turn_text}\n"

        chunks.append(dialog_chunk)

    return chunks, speakers_pool, folder_names
    


# ── Test 4: End-to-end with real Qwen API call on episode data ───────────────

def test_e2e_speaker_identification(dialogue_folder):
    jsonl_files = sorted(
        glob.glob(os.path.join(dialogue_folder, "*.json"))
    )
    print(f"Found {len(jsonl_files)} jsonl files")

    chunks, speakers_pool, folder_list = load_gt_episode_chunks(dialogue_folder)
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



def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Speaker name extraction"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing Friends *.en.wav and *.txt episode files",
    )

    args = parser.parse_args()
    resolved_chunks, speakers_pool = test_e2e_speaker_identification(args.data_dir)


if __name__ == "__main__":
    main()