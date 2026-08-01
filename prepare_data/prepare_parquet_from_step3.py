"""
Build a Parquet dataset from an ALREADY-PROCESSED step3 folder.

Unlike prepare_audio_parquet.py, this script assumes the dialogue files already
have their final speaker names baked in (both gt and pred), so there is no
speaker_name_map lookup and no re-saving of the dialogue chunks. It simply reads
the existing chunks, filters by season, and writes a single parquet.

Input folder layout (e.g. outputs/step3_anony/S01_S03_Clean_Anoy):
    data_dir/{episode}/{CHUNK_N}/parsed_dialog_gt.json    — gt named dialogue
    data_dir/{episode}/{CHUNK_N}/parsed_dialog_pred.json  — pred named dialogue

Each parsed_dialog_*.json is a list of utterance dicts:
    [{"speaker": "erik_larsen", "text": "...", "start": 0.0, "end": 1.5}, ...]

Output (written into data_dir by default):
    dataset_gt_name_{season}{_suffix}.parquet    (--use_gt_name)
    dataset_pred_name_{season}{_suffix}.parquet  (default)

Example:
    python -m prepare_data.prepare_parquet_from_step3 \
        --data_dir outputs/step3_anony/S01_S03_Clean_Anoy \
        --season_filter Season01 --suffix Clean --use_gt_name
    -> outputs/step3_anony/S01_S03_Clean_Anoy/dataset_gt_name_Season01_Clean.parquet
"""

import argparse
import glob
import json
import os

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


def load_chunks(
    data_dir: str,
    dialog_filename: str,
    season_filter: str = None,
    time_info_path: str = DEFAULT_TIME_INFO_PATH,
) -> tuple[list[str], list[str]]:
    """Read already-named dialogue chunks and format them into prompt strings.

    `dialog_filename` selects gt vs pred (parsed_dialog_gt.json /
    parsed_dialog_pred.json). Names are used verbatim from the file.
    """
    time_maps = load_time_maps(time_info_path)

    subfolders = sorted(
        glob.glob(os.path.join(data_dir, "*", "*", dialog_filename))
    )
    # Single-season filter: keep only chunks whose path contains the season
    # substring (e.g. "Season03"). None/empty = keep all.
    if season_filter:
        subfolders = [p for p in subfolders if season_filter in p]
    print(f"Found {len(subfolders)} dialogue files ({dialog_filename})"
          + (f" (season_filter={season_filter})" if season_filter else ""))

    chunks = []
    chunk_folders = []      # per-chunk provenance: "{episode}/{CHUNK_N}", chunk-aligned
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
        for turn in dialog:
            speaker = turn["speaker"]
            turn_text = fix_space_in_text(turn["text"])
            dialog_chunk += f"<{speaker}> {turn_text}\n"

        chunks.append(dialog_chunk)
        # Relative folder of THIS chunk ("{episode}/CHUNK_N"); its position in
        # `chunk_folders` is the chunk_idx used in chunks_and_function_calls.json.
        chunk_folders.append(os.path.relpath(os.path.dirname(path), data_dir))

    return chunks, chunk_folders

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Parquet from an already-processed step3 folder "
                    "(names already baked into the dialogue files)."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="step3 folder, e.g. outputs/step3_anony/S01_S03_Clean_Anoy",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to write the parquet (default: --data_dir)",
    )
    parser.add_argument(
        "--use_gt_name", action="store_true",
        help="Use parsed_dialog_gt.json instead of parsed_dialog_pred.json",
    )
    parser.add_argument(
        "--season_filter", type=str, default=None,
        help="Single season substring (e.g. Season01) to build a per-season "
             "parquet. None = all seasons.",
    )
    parser.add_argument(
        "--suffix", type=str, default=None,
        help="Extra tag appended after the season in the filename, e.g. 'Clean' "
             "-> dataset_gt_name_Season01_Clean.parquet",
    )
    parser.add_argument(
        "--time_info_path", type=str, default=DEFAULT_TIME_INFO_PATH,
        help=f"Season timeline JSON (default: {DEFAULT_TIME_INFO_PATH}).",
    )

    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)

    season_suffix = f"_{args.season_filter}" if args.season_filter else ""
    extra_suffix = f"_{args.suffix}" if args.suffix else ""

    if args.use_gt_name:
        dialog_filename = "parsed_dialog_gt.json"
        name_tag = "gt_name"
    else:
        dialog_filename = "parsed_dialog_pred.json"
        name_tag = "pred_name"

    chunks, chunk_folders = load_chunks(
        args.data_dir,
        dialog_filename=dialog_filename,
        season_filter=args.season_filter,
        time_info_path=args.time_info_path,
    )
    print(f"Loaded {len(chunks)} chunks")

    output_path = os.path.join(
        output_dir, f"dataset_{name_tag}{season_suffix}{extra_suffix}.parquet"
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


if __name__ == "__main__":
    main()
