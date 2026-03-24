"""
Test script for InterActDataset from SeamlessInteraction_loader.py

Usage:
    # Test with synthetic mock data (no real data needed)
    python test_dataset.py --mock

    # Test with real data
    python test_dataset.py --data_path /checkpoint/seamless/data/Mosaic

    # Test with real data, specific split
    python test_dataset.py --data_path /checkpoint/seamless/data/Mosaic --split test --format naturalistic
"""

import argparse
import json
import os
import sys
import tempfile
import shutil

import jsonlines
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "audio_script"))
from datasets.SeamlessInteraction_loader import (
    InterActDataset,
)



def test_dataset_init(data_path, diag_format, split):
    """Test InterActDataset construction."""
    print("=" * 60)
    print(f"TEST: InterActDataset.__init__(data_path={data_path}, "
          f"format={diag_format}, split={split})")
    print("=" * 60)

    try:
        ds = InterActDataset(
            data_path=data_path,
            diag_format=diag_format,
            split=split,
        )
        print(f"  Dataset length: {len(ds)}")
        print(f"  Valid pairs:    {len(ds.valid_data)}")
        print(f"  Audio pairs:    {len(ds.audio_files)}")
        print(f"  Transcript pairs: {len(ds.transcript_files)}")
        print(f"  VAD pairs:      {len(ds.vad_files)}")

        assert len(ds) == len(ds.valid_data)
        assert len(ds.audio_files) == len(ds)
        assert len(ds.transcript_files) == len(ds)
        assert len(ds.vad_files) == len(ds)

        if len(ds) == 0:
            print("  [WARN] Dataset is empty -- no valid paired files found.")
        else:
            print("  [PASS]")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        ds = None
    print()
    return ds


def test_load_sample(ds, idx=0):
    """Test InterActDataset.load_sample()."""
    print("=" * 60)
    print(f"TEST: InterActDataset.load_sample(idx={idx})")
    print("=" * 60)

    if ds is None or len(ds) == 0:
        print("  [SKIP] No data available.")
        print()
        return None

    try:
        sample = ds.load_sample(idx)
        print(f"  Keys: {list(sample.keys())}")
        print(f"  IDs:  {sample['id']}")
        print(f"  Speaker IDs: {sample['speaker_id']}")
        print(f"  Sample rate: {sample['sr']}")
        for i, aud in enumerate(sample["audios"]):
            print(f"  Audio[{i}] shape: {aud.shape}, dtype: {aud.dtype}")
        for i, tr in enumerate(sample["transcripts"]):
            print(f"  Transcript[{i}]: {len(tr)} entries")
            if len(tr) > 0:
                print(f"    First entry: {tr[0]}")
        for i, v in enumerate(sample["vad"]):
            print(f"  VAD[{i}]: {len(v)} entries")
            if len(v) > 0:
                print(f"    First entry: {v[0]}")

        assert len(sample["audios"]) == 2
        assert len(sample["transcripts"]) == 2
        assert len(sample["vad"]) == 2
        assert len(sample["id"]) == 2
        assert len(sample["speaker_id"]) == 2
        print("  [PASS]")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sample = None
    print()
    return sample


def main():
    ds = test_dataset_init("/checkpoint/seamless/data/Mosaic", "naturalistic", "test")
    sample = test_load_sample(ds, idx=0)


if __name__ == "__main__":
    main()
