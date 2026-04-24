import json
import re
import pandas as pd
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "mytest/test.parquet"

df = pd.read_parquet(path)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

row = df.iloc[0].to_dict()
raw_chunks = row["chunks"]

# Parse chunks JSON string → list of strings
chunks: list[str] = json.loads(raw_chunks) if isinstance(raw_chunks, str) else list(raw_chunks)

def split_chunk(text: str) -> list[dict]:
    """Split a chunk string into turns by <User> / <Assistant> tags."""
    pattern = re.compile(r"<(User|Assistant)>(.*?)(?=<(?:User|Assistant)>|$)", re.DOTALL)
    turns = []
    for m in pattern.finditer(text):
        turns.append({"role": m.group(1), "content": m.group(2).strip()})
    # If no tags found, return the whole chunk as a single entry
    if not turns:
        turns = [{"role": "raw", "content": text.strip()}]
    return turns

print(f"\nTotal chunks: {len(chunks)}\n")
for i, chunk in enumerate(chunks):
    turns = split_chunk(chunk)
    print(f"--- Chunk {i} ({len(turns)} turn(s)) ---")
    for turn in turns:
        preview = turn["content"][:120].replace("\n", " ")
        print(f"  [{turn['role']}] {preview}")
    print()
