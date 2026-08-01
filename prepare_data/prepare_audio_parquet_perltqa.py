"""
Build Parquet datasets from pre-chunked PerLTQA dialogue bundles.

This is the PerLTQA-specific counterpart of prepare_audio_parquet.py. The key
difference is the input layout: the PerLTQA step2 folder groups conversations
into *bundles* (bundle_0, bundle_1, ...). Each bundle is treated like a "season":
it has its own extracted_speaker_name.json and produces its own parquet.

Input folder layout:
    data_dir/{bundle}/{conv_id}/{CHUNK_N}/parsed_dialog_pred.json   — pred mode
    data_dir/{bundle}/{conv_id}/{CHUNK_N}/parsed_dialog_gt.json     — gt mode
    data_dir/{bundle}/extracted_speaker_name.json                  — pred mode only

    where conv_id == <Profile>_<dialogue_folder> (e.g. "Cao_Lili_25_0_0_0"), which
    must match a `source_file` (minus .json) in the session timeline.

Each parsed_dialog_*.json is a list of utterance dicts:
    [{"speaker": "GLOBAL_SPK_0", "text": "...", "start": 0.0, "end": 1.5}, ...]

Output (one per bundle):
    output_root/{bundle}/dataset_{pred|gt}_name_{bundle}[_{suffix}].parquet
    output_root/{bundle}/{conv_id}/{CHUNK_N}/parsed_dialog_{pred|gt}.json   (mirror)

Parquet columns:
    instance_id, prompt, chunks, chunk_folders, data_source, metadata, num_chunks

Example:
    python prepare_data/make_perltqa_timeline.py
    PYTHONPATH=. python prepare_data/prepare_audio_parquet_perltqa.py \
        --data_dir Audio_Results/vibevoice/dialogue_tts_en_v2/step2 \
        --output_root outputs/step3/perltqa \
        --time_info_path outputs/perltqa_data/perltqa_session_timeline.json \
        --mode both
"""

import argparse
import glob
import json
import os

import pandas as pd

from prepare_data.preprocess_utils import fix_space_in_text
from prepare_data.prepare_audio_parquet import (
    SEASON_FILTER_KEY,
    load_time_maps,
)

DEFAULT_TIME_INFO_PATH = "outputs/perltqa_data/perltqa_session_timeline.json"


def find_bundles(data_dir: str, bundle_filter: list[str] | None = None) -> list[str]:
    """Return sorted bundle folder names directly under data_dir.

    A bundle is any immediate subfolder that contains at least one chunked
    dialogue file. `bundle_filter` (if given) keeps only those exact names.
    """
    bundles = []
    for name in sorted(os.listdir(data_dir)):
        bpath = os.path.join(data_dir, name)
        if not os.path.isdir(bpath):
            continue
        has_chunks = glob.glob(
            os.path.join(bpath, "*", "*", "parsed_dialog_pred.json")
        ) or glob.glob(os.path.join(bpath, "*", "*", "parsed_dialog_gt.json"))
        if has_chunks:
            bundles.append(name)
    if bundle_filter:
        wanted = set(bundle_filter)
        bundles = [b for b in bundles if b in wanted]
    return bundles


def load_bundle_chunks(
    bundle_dir: str,
    time_maps: dict,
    use_gt_name: bool,
    output_root: str = None,
) -> tuple[list[str], list[str]]:
    """Load and format every chunk in a single bundle.

    Returns (chunks, chunk_folders) where chunk_folders[i] == "{conv_id}/CHUNK_N"
    is bundle-relative and index-aligned with chunks[i].
    """
    dialog_filename = "parsed_dialog_gt.json" if use_gt_name else "parsed_dialog_pred.json"

    # Pred mode resolves GLOBAL_SPK_N -> real name via the bundle's own map.
    speaker_name_map = {}
    if not use_gt_name:
        map_path = os.path.join(bundle_dir, "extracted_speaker_name.json")
        with open(map_path, "r") as f:
            speaker_name_map = json.load(f)
        # step3 may stamp the season under a reserved key; never treat as speaker.
        speaker_name_map.pop(SEASON_FILTER_KEY, None)
        print(f"  Loaded speaker name map with {len(speaker_name_map)} entries")

    subfolders = sorted(
        glob.glob(os.path.join(bundle_dir, "*", "*", dialog_filename))
    )
    print(f"  Found {len(subfolders)} {dialog_filename} files")

    chunks = []
    chunk_folders = []
    for path in subfolders:
        with open(path, "r") as f:
            dialog = json.load(f)

        # data_dir/{bundle}/{conv_id}/{CHUNK_N}/parsed_dialog_*.json
        # -> conv_id is the folder two levels up from the file.
        session_name = path.split("/")[-3]
        if session_name not in time_maps:
            print(f"    WARNING: no timeline entry for '{session_name}'; skipping {path}")
            continue
        timestamp = time_maps[session_name]

        dialog_chunk = f"[Dialogue between multiple people on {timestamp}]\n"
        named_dialog = []
        for turn in dialog:
            raw_speaker = turn["speaker"]
            resolved_speaker = speaker_name_map.get(raw_speaker, raw_speaker)
            speaker = resolved_speaker if not use_gt_name else raw_speaker

            turn_text = fix_space_in_text(turn["text"])
            dialog_chunk += f"<{speaker}> {turn_text}\n"

            new_turn = dict(turn)
            new_turn["text"] = turn_text
            if not use_gt_name:
                new_turn["speaker"] = resolved_speaker
            named_dialog.append(new_turn)

        chunks.append(dialog_chunk)
        rel = os.path.relpath(os.path.dirname(path), bundle_dir)  # "{conv_id}/CHUNK_N"
        chunk_folders.append(rel)

        # Mirror the raw chunk (with replaced names for pred) under output_root.
        if output_root is not None:
            out_dir = os.path.join(output_root, rel)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, dialog_filename), "w") as f:
                json.dump(named_dialog, f, indent=2, ensure_ascii=False)

    if output_root is not None:
        print(f"  Saved {len(chunks)} dialogue chunk(s) under {output_root}")

    return chunks, chunk_folders


def build_bundle_parquet(
    data_dir: str,
    bundle: str,
    output_root: str,
    use_gt_name: bool,
    time_maps: dict,
    data_source: str,
    suffix: str = "",
) -> None:
    """Build one parquet (+ mirrored chunks) for a single bundle."""
    bundle_dir = os.path.join(data_dir, bundle)
    bundle_out_root = os.path.join(output_root, bundle)  # subfolder per bundle
    os.makedirs(bundle_out_root, exist_ok=True)

    name_mode = "gt" if use_gt_name else "pred"
    print(f">>> Bundle '{bundle}' ({name_mode} name)")

    chunks, chunk_folders = load_bundle_chunks(
        bundle_dir, time_maps, use_gt_name=use_gt_name, output_root=bundle_out_root,
    )
    print(f"  Loaded {len(chunks)} chunks")

    samples = [{
        "instance_id": 0,
        "prompt": "I will provide you with the conversation history between the "
                  "different speakers and I need you to remember the details of "
                  "the conversation for future reference.",
        "chunks": json.dumps(chunks),
        # Per-chunk source folder ("{conv_id}/CHUNK_N"), bundle-relative and
        # index-aligned with `chunks` so chunk_folders[chunk_idx] recovers the
        # origin of chunk `chunk_idx`. Used to map QA evidence -> memory ids.
        "chunk_folders": json.dumps(chunk_folders),
        "data_source": data_source,
        "metadata": json.dumps(
            {"data_source": data_source, "sample_id": 0, "bundle": bundle}
        ),
        "num_chunks": len(chunks),
    }]

    # Suffix the parquet with the bundle name (the "season" for PerLTQA) and an
    # optional user-supplied suffix (e.g. an interference/SNR tag).
    extra = f"_{suffix}" if suffix else ""
    output_path = os.path.join(
        bundle_out_root, f"dataset_{name_mode}_name_{bundle}{extra}.parquet"
    )
    pd.DataFrame(samples).to_parquet(output_path, index=False)
    print(f"  Saved {len(samples)} sample(s) ({len(chunks)} chunks) to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare per-bundle Parquet datasets for PerLTQA"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="PerLTQA step2 root containing bundle folders "
             "({bundle}/{conv_id}/{CHUNK_N}/parsed_dialog_*.json)",
    )
    parser.add_argument(
        "--output_root", type=str, default="outputs/step3",
        help="Root dir; each bundle gets its own subfolder (default: outputs/step3)",
    )
    parser.add_argument(
        "--mode", type=str, default="both", choices=["pred", "gt", "both"],
        help="Which parquet(s) to build per bundle (default: both)",
    )
    parser.add_argument(
        "--bundles", type=str, nargs="*", default=None,
        help="Restrict to these bundle folder names (default: all bundles found)",
    )
    parser.add_argument(
        "--time_info_path", type=str, default=DEFAULT_TIME_INFO_PATH,
        help=f"PerLTQA session timeline JSON (default: {DEFAULT_TIME_INFO_PATH}), "
             "built by prepare_data/make_perltqa_timeline.py. Each session's "
             "source_file (minus .json) must equal the chunk's conv_id.",
    )
    parser.add_argument(
        "--data_source", type=str, default="perltqa",
        help="data_source label written into the parquet (default: perltqa)",
    )
    parser.add_argument(
        "--suffix", type=str, default="",
        help="Optional extra suffix appended to the parquet name, i.e. "
             "dataset_{pred|gt}_name_{bundle}_{suffix}.parquet (e.g. an "
             "interference/SNR tag). Empty = no extra suffix.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise SystemExit(f"ERROR: data_dir not found: {args.data_dir}")

    time_maps = load_time_maps(args.time_info_path)
    bundles = find_bundles(args.data_dir, args.bundles)
    if not bundles:
        raise SystemExit(f"ERROR: no bundles with chunks found under {args.data_dir}")

    print("===============================================================")
    print(f"data_dir     : {args.data_dir}")
    print(f"output_root  : {args.output_root}")
    print(f"mode         : {args.mode}")
    print(f"bundles      : {', '.join(bundles)}")
    print("===============================================================")

    os.makedirs(args.output_root, exist_ok=True)
    modes = {"pred": [False], "gt": [True], "both": [False, True]}[args.mode]
    for bundle in bundles:
        for use_gt_name in modes:
            build_bundle_parquet(
                args.data_dir, bundle, args.output_root,
                use_gt_name=use_gt_name, time_maps=time_maps,
                data_source=args.data_source, suffix=args.suffix,
            )

    print("Done.")


if __name__ == "__main__":
    main()
