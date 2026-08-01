"""
Build an ANONYMIZED Parquet dataset from an ALREADY-PROCESSED step3 folder.

This is a variant of prepare_parquet_from_step3.py. It reads the same
already-named dialogue chunks, but instead of writing the real speaker names
into the prompt, it replaces them with anonymous labels. The goal is to test
whether the memory model can recover / bind speaker identities from the
conversation content alone, without being handed the names.

Two anonymization modes:

  --anon_mode global   (experiment 1)
      Cluster names ACROSS all episodes/chunks (within the season filter) into a
      single global map, in deterministic first-appearance order over the sorted
      chunk files. Every occurrence of a real name becomes "Speaker{X}", so the
      same real person keeps the same label everywhere. This gives GLOBAL speaker
      differentiation with no names.

  --anon_mode local    (experiment 2)
      Cluster names WITHIN each chunk only. Each chunk ("conversation") gets a
      fresh map, and the label is "Conversation{Y}_Speaker{X}" where X resets
      per chunk and Y is a running conversation index accumulated across chunks
      and episodes (0-based, aligned with chunk_folders). This gives only LOCAL
      speaker differentiation: speakers are distinguishable within a conversation
      but NOT tied together across conversations.

Input folder layout (e.g. outputs/step3_anony/S01_S03_Clean_Anoy):
    data_dir/{episode}/{CHUNK_N}/parsed_dialog_gt.json    — gt named dialogue
    data_dir/{episode}/{CHUNK_N}/parsed_dialog_pred.json  — pred named dialogue

Each parsed_dialog_*.json is a list of utterance dicts:
    [{"speaker": "erik_larsen", "text": "...", "start": 0.0, "end": 1.5}, ...]

Output goes into a NEW sibling folder "<data_dir>_anon_{global|local}" (the
original folder is never modified). It mirrors the input chunk tree with the
speaker names replaced, plus the parquet:
    <data_dir>_anon_{mode}/{episode}/{CHUNK_N}/parsed_dialog_{gt|pred}.json
    <data_dir>_anon_{mode}/dataset_{gt|pred}_name_anon_{mode}_{season}{_suffix}.parquet

Example:
    python -m prepare_data.prepare_parquet_from_step3_anon \
        --data_dir outputs/step3_anony/S01_S03_Clean_Anoy \
        --season_filter Season01 --suffix Clean --anon_mode global
    -> outputs/step3_anony/S01_S03_Clean_Anoy_anon_global/
         dataset_pred_name_anon_global_Season01_Clean.parquet  (+ mirrored chunks)
"""

import argparse
import glob
import json
import os
import shutil

import pandas as pd
from prepare_data.preprocess_utils import fix_space_in_text


DEFAULT_TIME_INFO_PATH = "outputs/bazinga_data/TBBT_all_seasons_session_timeline.json"


def load_time_maps(time_info_path: str) -> dict:
    with open(time_info_path, "r") as f:
        time_info = json.load(f)
    time_info = time_info["sessions"]
    time_maps = {}
    for item in time_info:
        keyname = item["source_file"].replace(".json", "")
        time_maps[keyname] = item["session_timeline_date"]
    return time_maps


def load_chunks_anon(
    data_dir: str,
    dialog_filename: str,
    anon_mode: str,
    season_filter: str = None,
    time_info_path: str = DEFAULT_TIME_INFO_PATH,
) -> tuple[list[str], list[str], list[list[dict]], dict]:
    """Read already-named dialogue chunks and emit ANONYMIZED prompt strings.

    `dialog_filename` selects gt vs pred (parsed_dialog_gt.json /
    parsed_dialog_pred.json). `anon_mode` selects the labeling scheme:

      "global"  -> "Speaker{X}"                          (one map across all chunks)
      "local"   -> "Conversation{Y}_Speaker{X}"          (fresh map per chunk)

    Returns (chunks, chunk_folders, anon_dialogs, global_name_map), all
    index-aligned per chunk. `anon_dialogs` holds the original utterance dicts
    with only the "speaker" field replaced by the anon label (start/end/text
    untouched), so callers can re-save an anonymized parsed_dialog_*.json. For
    local mode the returned map is empty (labels are chunk-scoped and provenance
    lives in chunk_folders).
    """
    assert anon_mode in ("global", "local"), f"bad anon_mode: {anon_mode}"

    time_maps = load_time_maps(time_info_path)

    subfolders = sorted(
        glob.glob(os.path.join(data_dir, "*", "*", dialog_filename))
    )
    # Single-season filter: keep only chunks whose path contains the season
    # substring (e.g. "Season03"). None/empty = keep all.
    if season_filter:
        subfolders = [p for p in subfolders if season_filter in p]
    print(f"Found {len(subfolders)} dialogue files ({dialog_filename})"
          + (f" (season_filter={season_filter})" if season_filter else "")
          + f" [anon_mode={anon_mode}]")

    chunks = []
    chunk_folders = []      # per-chunk provenance: "{episode}/{CHUNK_N}", chunk-aligned
    anon_dialogs = []       # per-chunk anonymized utterance dicts, chunk-aligned

    # Global map: real name -> stable integer id, assigned in first-appearance
    # order over the emitted chunks. Only used for anon_mode="global".
    global_name_map = {}

    # Conversation index Y, accumulated across chunks and episodes. It advances
    # once per EMITTED chunk, so it stays aligned with `chunks` / `chunk_folders`
    # (skipped chunks with no timeline entry do not consume a Y).
    conversation_idx = 0

    for path in subfolders:
        with open(path, "r") as f:
            dialog = json.load(f)

        session_name = path.split("/")[-3]
        if session_name not in time_maps:
            print(f"  WARNING: no timeline entry for '{session_name}' in "
                  f"{time_info_path}; skipping {path}")
            continue
        timestamp = time_maps[session_name]

        # Fresh per-chunk map for local mode.
        local_name_map = {}

        dialog_chunk = f"[Dialogue between multiple people on {timestamp}]\n"
        anon_turns = []
        for turn in dialog:
            speaker = turn["speaker"]
            turn_text = fix_space_in_text(turn["text"])

            if anon_mode == "global":
                if speaker not in global_name_map:
                    global_name_map[speaker] = len(global_name_map)
                label = f"Speaker{global_name_map[speaker]}"
            else:  # local
                if speaker not in local_name_map:
                    local_name_map[speaker] = len(local_name_map)
                label = f"Conversation{conversation_idx}_Speaker{local_name_map[speaker]}"

            dialog_chunk += f"<{label}> {turn_text}\n"
            # Copy the utterance verbatim, swapping only the speaker label so the
            # re-saved parsed_dialog_*.json keeps start/end/text intact.
            anon_turns.append({**turn, "speaker": label})

        chunks.append(dialog_chunk)
        # Relative folder of THIS chunk ("{episode}/CHUNK_N"); its position in
        # `chunk_folders` is the chunk_idx (== conversation_idx) used downstream.
        chunk_folders.append(os.path.relpath(os.path.dirname(path), data_dir))
        anon_dialogs.append(anon_turns)
        conversation_idx += 1

    if anon_mode == "global":
        print(f"Global speaker map: {len(global_name_map)} unique speakers")

    return chunks, chunk_folders, anon_dialogs, global_name_map


def main():
    parser = argparse.ArgumentParser(
        description="Prepare an ANONYMIZED Parquet from an already-processed "
                    "step3 folder (names replaced by Speaker labels)."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="step3 folder, e.g. outputs/step3_anony/S01_S03_Clean_Anoy",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to write the anonymized folder + parquet. Default: a new "
             "sibling folder '<data_dir>_anon_{global|local}'.",
    )
    parser.add_argument(
        "--anon_mode", type=str, choices=["global", "local"], default="global",
        help="global: one Speaker{X} map across all chunks (experiment 1). "
             "local: Conversation{Y}_Speaker{X}, fresh map per chunk "
             "(experiment 2).",
    )
    parser.add_argument(
        "--use_gt_name", action="store_true",
        help="Anonymize parsed_dialog_gt.json instead of parsed_dialog_pred.json",
    )
    parser.add_argument(
        "--season_filter", type=str, default=None,
        help="Single season substring (e.g. Season01) to build a per-season "
             "parquet. None = all seasons.",
    )
    parser.add_argument(
        "--suffix", type=str, default=None,
        help="Extra tag appended after the season in the filename, e.g. 'Clean' "
             "-> dataset_pred_name_anon_global_Season01_Clean.parquet",
    )
    parser.add_argument(
        "--time_info_path", type=str, default=DEFAULT_TIME_INFO_PATH,
        help=f"Season timeline JSON (default: {DEFAULT_TIME_INFO_PATH}).",
    )
    parser.add_argument(
        "--dump_name_map", action="store_true",
        help="Also write the global name->label map next to the parquet "
             "(global mode only), for debugging/eval.",
    )

    args = parser.parse_args()

    # New sibling folder "<data_dir>_anon_{mode}" unless overridden. normpath so a
    # trailing slash doesn't produce "<data_dir>/_anon_global".
    output_dir = args.output_dir or (
        os.path.normpath(args.data_dir) + f"_anon_{args.anon_mode}"
    )
    os.makedirs(output_dir, exist_ok=True)

    season_suffix = f"_{args.season_filter}" if args.season_filter else ""
    extra_suffix = f"_{args.suffix}" if args.suffix else ""

    if args.use_gt_name:
        dialog_filename = "parsed_dialog_gt.json"
        name_tag = "gt_name"
    else:
        dialog_filename = "parsed_dialog_pred.json"
        name_tag = "pred_name"

    chunks, chunk_folders, anon_dialogs, global_name_map = load_chunks_anon(
        args.data_dir,
        dialog_filename=dialog_filename,
        anon_mode=args.anon_mode,
        season_filter=args.season_filter,
        time_info_path=args.time_info_path,
    )
    print(f"Loaded {len(chunks)} chunks")

    # Re-save the anonymized dialogue, mirroring the input tree under output_dir:
    #   output_dir/{episode}/{CHUNK_N}/{dialog_filename}
    # Also copy the untouched parsed_dialog_gt.json alongside (unless gt IS the
    # anonymized primary, in which case copying it back would clobber the anon
    # version) so the new folder keeps the original names for reference/eval.
    copied_gt = 0
    for rel_folder, anon_turns in zip(chunk_folders, anon_dialogs):
        dst_dir = os.path.join(output_dir, rel_folder)
        os.makedirs(dst_dir, exist_ok=True)
        with open(os.path.join(dst_dir, dialog_filename), "w") as f:
            json.dump(anon_turns, f, indent=2, ensure_ascii=False)

        if not args.use_gt_name:
            src_gt = os.path.join(args.data_dir, rel_folder, "parsed_dialog_gt.json")
            if os.path.exists(src_gt):
                shutil.copyfile(src_gt, os.path.join(dst_dir, "parsed_dialog_gt.json"))
                copied_gt += 1
    print(f"Wrote {len(anon_dialogs)} anonymized {dialog_filename} under {output_dir}")
    if not args.use_gt_name:
        print(f"Copied {copied_gt} unchanged parsed_dialog_gt.json alongside")

    output_path = os.path.join(
        output_dir,
        f"dataset_{name_tag}_anon_{args.anon_mode}"
        f"{season_suffix}{extra_suffix}.parquet",
    )

    samples = [{
        "instance_id": 0,
        "prompt": "I will provide you with the conversation history between the "
                  "different speakers and I need you to remember the details of "
                  "the conversation for future reference.",
        "chunks": json.dumps(chunks),
        # Per-chunk source folder ("{episode}/CHUNK_N"), index-aligned with `chunks`.
        "chunk_folders": json.dumps(chunk_folders),
        "data_source": "seamlessinteraction_options",
        "metadata": json.dumps({"data_source": "seamlessinteraction_options", "sample_id": 0}),
        "num_chunks": len(chunks),
    }]

    df = pd.DataFrame(samples)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(samples)} samples ({len(chunks)} chunks) to {output_path}")

    if args.dump_name_map and args.anon_mode == "global":
        map_path = output_path.replace(".parquet", "_name_map.json")
        # real_name -> "Speaker{X}"
        readable = {name: f"Speaker{idx}" for name, idx in global_name_map.items()}
        with open(map_path, "w") as f:
            json.dump(readable, f, indent=2, ensure_ascii=False)
        print(f"Saved global name map ({len(readable)} speakers) to {map_path}")


if __name__ == "__main__":
    main()
