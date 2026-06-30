import glob
import json
import os
import sys
import string
import jsonlines
import pandas as pd
import argparse
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
    registry_to_dict,
    registry_from_dict,
)
def fix_space_in_text(text):
    punctuation_pattern = [" " + c for c in string.punctuation]
    punctuation_pattern.append(" n't")
    punctuation_pattern.extend([" 'm", " 's", " 've", " 're", " 'll", " 'd", " 'n", " 't", " 'y", " 'z"])
    for pattern in punctuation_pattern:
        text = text.replace(pattern, pattern.strip())
    return text



def load_gt_episode_chunks(dialogue_folder):
    ### find all subfolder with {xxx}/{xxx}/parsed_dialog_gt.json sub-folders
    subfolders = glob.glob(os.path.join(dialogue_folder, "*", "*", "parsed_dialog_gt.json"))
    # sorted the subfolders
    subfolders = sorted(subfolders)
    print(f"Found {len(subfolders)} subfolders")

    folder_names = []
    speakers_pool = {}
    chunks = []

    for subfolder in subfolders:
        folder_name = os.path.dirname(subfolder)
        folder_names.append(folder_name)

        with open(subfolder, "r") as f:
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



def load_pred_episode_chunks(dialogue_folder, speakers_pool=None, season_filter=None):
    ### find all subfolder with {xxx}/{xxx}/parsed_dialog_pred.json sub-folders
    subfolders = glob.glob(os.path.join(dialogue_folder, "*", "*", "parsed_dialog_pred.json"))
    # sorted the subfolders
    subfolders = sorted(subfolders)
    # Optional single-season filter: keep only chunks whose path contains the
    # given substring (e.g. "Season03"). None/empty = keep all.
    if season_filter:
        subfolders = [s for s in subfolders if season_filter in s]
    print(f"Found {len(subfolders)} subfolders")

    # speakers_pool maps a (global) speaker label -> stable Speaker_<idx>.
    # Pass in a persistent pool so the same global speaker keeps the same index
    # across incremental (season-by-season) runs; new speakers are appended.
    if speakers_pool is None:
        speakers_pool = {}
    folder_names = []
    chunks = []

    for subfolder in subfolders:
        folder_name = os.path.dirname(subfolder)
        folder_names.append(folder_name)

        with open(subfolder, "r") as f:
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



# ── Persistent state (single JSON file across incremental runs) ──────────────

def load_state(state_path):
    """Load {speakers_pool, processed_chunks, registry} from one JSON file.

    Returns (speakers_pool, processed_ids, registry). Missing file -> empty
    state, so the first run starts fresh.
    """
    if state_path and os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
        speakers_pool = state.get("speakers_pool", {})
        processed_ids = set(state.get("processed_chunks", []))
        registry = registry_from_dict(state.get("registry", {}))
        print(
            f"[state] loaded {len(speakers_pool)} speaker(s), "
            f"{len(processed_ids)} processed chunk(s) <- {state_path}"
        )
        return speakers_pool, processed_ids, registry
    if state_path:
        print(f"[state] no existing state at {state_path}; starting fresh")
    return {}, set(), {}


def save_state(state_path, speakers_pool, processed_ids, registry):
    out_dir = os.path.dirname(os.path.abspath(state_path))
    os.makedirs(out_dir, exist_ok=True)
    state = {
        "speakers_pool": speakers_pool,
        "processed_chunks": sorted(processed_ids),
        "registry": registry_to_dict(registry),
    }
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    print(
        f"[state] saved {len(speakers_pool)} speaker(s), "
        f"{len(processed_ids)} processed chunk(s) -> {state_path}"
    )


# ── End-to-end speaker name identification (incremental-aware) ────────────────

def run_speaker_identification(dialogue_folder, state_path=None, season_filter=None,
                               update_pool=False):
    # Restore cross-run state (empty on first run / when --state_path omitted).
    speakers_pool, processed_ids, registry = load_state(state_path)

    chunks, speakers_pool, folder_names = load_pred_episode_chunks(
        dialogue_folder, speakers_pool=speakers_pool, season_filter=season_filter
    )
    print(f"Loaded {len(chunks)} chunks, {len(speakers_pool)} unique speakers")

    # Override the module-level client/model to use the same env vars as evaluate_agent_results.py
    import audio_script.Speaker_Track.Speaker_Name_tracker as tracker
    tracker.client = OpenAI(
        base_url=os.getenv("QWEN_URL", os.getenv("QWEN_BASE_URL", "http://localhost:8000/v1")),
        api_key=os.getenv("OPENROUTER_API_KEY", os.getenv("QWEN_API_KEY", "EMPTY")),
    )
    tracker.QWEN3_MODEL = os.getenv("QWEN_MODEL_NAME", os.getenv("QWEN3_MODEL", "Qwen/Qwen3-32B"))

    # Incremental: only new chunks (by folder path) hit the LLM; Phase 2 then
    # re-resolves names over ALL accumulated evidence.
    registry = identify_speakers(
        chunks,
        dialogue_ids=folder_names,
        registry=registry,
        processed_ids=processed_ids,
        enable_thinking=False,
    )

    print("\n── Final Registry ──")
    for sid, rec in sorted(registry.items()):
        print(f"  {sid:12s} → {rec.display_name:20s} [{rec.status}]  (evidence: {len(rec.evidence)})")

    identified = [sid for sid, rec in registry.items() if rec.name is not None]
    print(f"\nIdentified {len(identified)} / {len(registry)} speakers")

    # Build the cumulative {global_speaker_id: resolved_name} map. speakers_pool
    # spans every speaker seen across all runs so far, so this map is complete.
    speaker_name_map = {}
    for global_id, idx in speakers_pool.items():
        sid = f"Speaker_{idx}"
        if sid in registry and registry[sid].name:
            speaker_name_map[global_id] = registry[sid].name
        else:
            speaker_name_map[global_id] = f"Unknown_speaker{idx:03d}"

    # Record which season this run covered (reserved key; downstream consumers
    # only look speakers up by key, so this metadata is ignored by them).
    # if season_filter:
    #     speaker_name_map["__season_filter__"] = season_filter
    if season_filter:
        out_path = os.path.join(dialogue_folder, f"extracted_speaker_name_{season_filter}.json")
    else:
        out_path = os.path.join(dialogue_folder, "extracted_speaker_name.json")

    with open(out_path, "w") as f:
        json.dump(speaker_name_map, f, indent=2)
    print(f"\nSaved speaker name map to {out_path}")
    print(json.dumps(speaker_name_map, indent=2))

    # Persist updated state for the next incremental run.
    if state_path and update_pool:
        save_state(state_path, speakers_pool, processed_ids, registry)

    return speaker_name_map, speakers_pool


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Speaker name extraction"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Step-2 output dir containing */*/parsed_dialog_pred.json",
    )
    parser.add_argument(
        "--state_path",
        type=str,
        default=None,
        help="Single JSON file holding the cross-run name state "
             "(speaker pool + evidence registry + processed chunks). "
             "If it exists it is loaded so only new chunks hit the LLM and "
             "names are re-resolved over all evidence; updated state is saved "
             "back here. Omit for a one-shot run.",
    )
    parser.add_argument(
        "--update_pool",
        action="store_true",
        help="whether update the pool file or not",
    )
    parser.add_argument(
        "--season_filter",
        type=str,
        default=None,
        help="Single season substring (e.g. Season03). Only chunks whose path "
             "contains it are processed; None = all. Recorded in the saved "
             "extracted_speaker_name.json under '__season_filter__'.",
    )
    args = parser.parse_args()
    run_speaker_identification(
        args.data_dir,
        state_path=args.state_path,
        season_filter=args.season_filter,
        update_pool=args.update_pool,
    )


if __name__ == "__main__":
    main()
