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


# Default (Season 1) timeline. Each season needs its own timeline file; override
# via --time_info_path when running per-season.
DEFAULT_TIME_INFO_PATH = "outputs/bazinga_data/TBBT_all_seasons_session_timeline.json"

# to record which season the map was produced for.
SEASON_FILTER_KEY = "__season_filter__"


# def read_embedded_season(data_dir: str) -> str | None:
#     """Return the season step3 stamped into extracted_speaker_name.json (or None)."""
#     map_path = os.path.join(data_dir, "extracted_speaker_name.json")
#     if os.path.exists(map_path):
#         with open(map_path, "r") as f:
#             return json.load(f).get(SEASON_FILTER_KEY)
#     return None


def load_time_maps(time_info_path: str) -> dict:
    with open(time_info_path, "r") as f:
        time_info = json.load(f)
    time_info = time_info["sessions"]
    time_maps = {}
    for item in time_info:
        keyname = item["source_file"].replace(".json", "")
        time_maps[keyname] = item["session_timeline_date"]
    return time_maps

def load_chunks_gt(
    data_dir: str,
    output_root: str = None,
    season_filter: str = None,
    time_info_path: str = DEFAULT_TIME_INFO_PATH,
) -> list[str]:
    time_maps = load_time_maps(time_info_path)

    subfolders = sorted(
        glob.glob(os.path.join(data_dir, "*", "*", "parsed_dialog_gt.json"))
    )
    # Single-season filter (matches step3): keep only chunks whose path contains
    # the season substring (e.g. "Season03"). None/empty = keep all.
    if season_filter:
        subfolders = [p for p in subfolders if season_filter in p]
    print(f"Found {len(subfolders)} dialogue files"
          + (f" (season_filter={season_filter})" if season_filter else ""))

    chunks = []
    for path in subfolders:
        with open(path, "r") as f:
            dialog = json.load(f)

        session_name = path.split("/")[-3]
        if session_name not in time_maps:
            print(f"  WARNING: no timeline entry for '{session_name}' in "
                  f"{time_info_path}; skipping {path}")
            continue
        timestamp = time_maps[session_name]

        dialog_chunk = f"[Dialogue between multiple people on {timestamp}]\n"
        named_dialog = []
        for turn in dialog:
            raw_speaker = turn["speaker"]
            speaker = raw_speaker
            turn_text = fix_space_in_text(turn["text"])
            dialog_chunk += f"<{speaker}> {turn_text}\n"
            turn["text"] = turn_text

            named_dialog.append(dict(turn))

        chunks.append(dialog_chunk)

        # Save the raw chunk mirroring the input layout under output_root,
        # e.g. outputs/step3/{show}/{episode}/parsed_dialog_gt.json
        if output_root is not None:
            rel = os.path.relpath(os.path.dirname(path), data_dir)
            out_dir = os.path.join(output_root, rel)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "parsed_dialog_gt.json"), "w") as f:
                json.dump(named_dialog, f, indent=2, ensure_ascii=False)

    if output_root is not None:
        print(f"Saved {len(subfolders)} dialogue chunk(s) under {output_root}")

    return chunks

def load_chunks_pred(
    data_dir: str,
    use_extracted_name: bool = False,
    output_root: str = None,
    season_filter: str = None,
    time_info_path: str = DEFAULT_TIME_INFO_PATH,
) -> list[str]:
    if season_filter is None:
        map_path = os.path.join(data_dir, "extracted_speaker_name.json")
    else:
        map_path = os.path.join(data_dir, f"extracted_speaker_name_{season_filter}.json")
    with open(map_path, "r") as f:
        speaker_name_map = json.load(f)

    # step3 stamps the season it covered under a reserved key; drop it so it is
    # never treated as a speaker. Fall back to it if no explicit season is given.
    embedded_season = speaker_name_map.pop(SEASON_FILTER_KEY, None)
    if not season_filter:
        season_filter = embedded_season

    time_maps = load_time_maps(time_info_path)
    print(f"Loaded speaker name map with {len(speaker_name_map)} entries")

    subfolders = sorted(
        glob.glob(os.path.join(data_dir, "*", "*", "parsed_dialog_pred.json"))
    )
    # Single-season filter (matches step3): keep only chunks whose path contains
    # the season substring. None/empty = keep all.
    if season_filter:
        subfolders = [p for p in subfolders if season_filter in p]
    print(f"Found {len(subfolders)} dialogue files"
          + (f" (season_filter={season_filter})" if season_filter else ""))
    chunks = []
    for path in subfolders:
        with open(path, "r") as f:
            dialog = json.load(f)
        session_name = path.split("/")[-3]
        if session_name not in time_maps:
            print(f"  WARNING: no timeline entry for '{session_name}' in "
                  f"{time_info_path}; skipping {path}")
            continue
        timestamp = time_maps[session_name]

        dialog_chunk = f"[Dialogue between multiple people on {timestamp}]\n"
        named_dialog = []
        for turn in dialog:
            raw_speaker = turn["speaker"]
            resolved_speaker = speaker_name_map.get(raw_speaker, raw_speaker)
            if use_extracted_name:
                speaker = resolved_speaker
            else:
                speaker = raw_speaker

            turn_text = fix_space_in_text(turn["text"])
            dialog_chunk += f"<{speaker}> {turn_text}\n"

            # raw turn with the speaker replaced by the recognized name
            new_turn = dict(turn)
            new_turn["speaker"] = resolved_speaker
            named_dialog.append(new_turn)

        chunks.append(dialog_chunk)

        # Save the raw chunk (with replaced names) mirroring the input layout
        # under output_root, e.g. outputs/step3/{show}/{episode}/parsed_dialog_pred.json
        rel = os.path.relpath(os.path.dirname(path), data_dir)

        if output_root is not None:
            out_dir = os.path.join(output_root, rel)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "parsed_dialog_pred.json"), "w") as f:
                json.dump(named_dialog, f, indent=2, ensure_ascii=False)

    if output_root is not None:
        print(f"Saved {len(subfolders)} named dialogue chunk(s) under {output_root}")

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Parquet from pre-chunked dialogues + speaker name map"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root folder with {show}/{episode}/parsed_dialog_pred.json",
    )
    # parser.add_argument(
    #     "--output", type=str, default=None,
    #     help="Output parquet path (default: <data_dir>/dataset.parquet)",
    # )
    parser.add_argument(
        "--output_root", type=str, default="outputs/step3",
        help="Root dir to save named dialogue chunks (default: outputs/step3)",
    )

    parser.add_argument(
        "--use_gt_name", action='store_true'
    )
    parser.add_argument(
        "--season_filter", type=str, default=None,
        help="Single season substring (e.g. Season03) to build a per-season "
             "parquet, matching step3. If omitted, falls back to the season "
             "stamped in extracted_speaker_name.json; None = all seasons.",
    )
    parser.add_argument(
        "--time_info_path", type=str, default=DEFAULT_TIME_INFO_PATH,
        help=f"Season timeline JSON (default: {DEFAULT_TIME_INFO_PATH}). Each "
             "season has its own timeline; override when running per-season.",
    )

    args = parser.parse_args()

    # Effective season: explicit flag wins; otherwise use what step3 stamped into
    # the speaker-name map (pred only). Used both for filtering and output naming.
    season = args.season_filter
    # if season is None and not args.use_gt_name:
    #     season = read_embedded_season(args.data_dir)

    season_suffix = f"_{season}" if season else ""

    if args.use_gt_name:
        chunks = load_chunks_gt(
            args.data_dir, output_root=args.output_root,
            season_filter=season, time_info_path=args.time_info_path,
        )
        output_path = os.path.join(
            args.output_root, f"dataset_gt_name{season_suffix}.parquet"
        )
    else:
        chunks = load_chunks_pred(
            args.data_dir, use_extracted_name=True, output_root=args.output_root,
            season_filter=season, time_info_path=args.time_info_path,
        )
        output_path = os.path.join(
            args.output_root, f"dataset_pred_name{season_suffix}.parquet"
        )
    print(f"Loaded {len(chunks)} chunks")

    samples = [{
        "instance_id": 0,
        "prompt": "I will provide you with the conversation history between the "
                  "different speakers and I need you to remember the details of "
                  "the conversation for future reference.",
        "chunks": json.dumps(chunks),
        "data_source": "seamlessinteraction_options",
        "metadata": json.dumps({"data_source": "seamlessinteraction_options", "sample_id": 0}),
        "num_chunks": len(chunks),
    }]


    df = pd.DataFrame(samples)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(samples)} samples ({len(chunks)} chunks) to {output_path}")


if __name__ == "__main__":
    main()
