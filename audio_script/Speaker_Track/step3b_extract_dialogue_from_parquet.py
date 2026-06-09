import argparse
import json
import os

import pandas as pd


def extract_dialogue_from_parquet(parquet_path, output_root, episode_name=None):
    """Load a parquet file (same format as step3) and dump each chunk's dialogue
    to <output_root>/<episode_name>/CHUNK_<i>/dialogue.txt.

    The parquet's `chunks` column is a JSON-encoded list of dialogue strings,
    one per chunk. One row == one episode.
    """
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {parquet_path}: {df.shape[0]} row(s), columns={list(df.columns)}")

    for _, row in df.iterrows():
        # Resolve the episode/folder name for this row.
        if episode_name is not None:
            name = episode_name
        else:
            name = str(row.get("instance_id", "instance"))

        chunks = row["chunks"]
        if isinstance(chunks, str):
            chunks = json.loads(chunks)

        episode_dir = os.path.join(output_root, name)
        for i, chunk in enumerate(chunks):
            chunk_dir = os.path.join(episode_dir, f"CHUNK_{i}")
            os.makedirs(chunk_dir, exist_ok=True)
            out_path = os.path.join(chunk_dir, "dialogue.txt")
            with open(out_path, "w") as f:
                f.write(chunk)

        print(f"Wrote {len(chunks)} chunk(s) to {episode_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-chunk dialogue from a step3-format parquet into dialogue.txt files"
    )
    parser.add_argument(
        "--parquet",
        type=str,
        required=True,
        help="Path to the parquet file (must have a `chunks` column)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/step3",
        help="Root directory for output (default: outputs/step3)",
    )
    parser.add_argument(
        "--episode_name",
        type=str,
        default=None,
        help="Folder name for the episode (default: the row's instance_id)",
    )
    args = parser.parse_args()

    extract_dialogue_from_parquet(args.parquet, args.output_root, args.episode_name)


if __name__ == "__main__":
    main()
