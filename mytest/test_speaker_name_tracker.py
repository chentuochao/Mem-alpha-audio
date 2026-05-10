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

import json
import os
import sys
import string

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
from prepare_data.prepare_bazinga_data import chunk_dialog, fix_space_in_text


# ── Build chunks from episode data (mirrors prepare_bazinga_data.py) ─────────

def load_episode_chunks(
    json_path: str,
    timestamp: str = "2023-05-01",
    min_dur: float = 60.0,
    max_dur: float = 300.0,
    gap_threshold: float = 3.0,
):
    with open(json_path, "r") as f:
        dialog = json.load(f)

    sub_chunks = chunk_dialog(dialog, min_dur=min_dur, max_dur=max_dur, gap_threshold=gap_threshold)

    speakers_pool = {}
    chunks = []
    for sub in sub_chunks:
        dialog_chunk = ""
        for turn in sub:
            speaker = turn["speaker"]
            if speaker not in speakers_pool:
                speakers_pool[speaker] = len(speakers_pool)
            anon_speaker = "Speaker_" + str(speakers_pool[speaker])
            turn_text = fix_space_in_text(turn["text"])
            dialog_chunk += f"<{anon_speaker}> {turn_text}\n"
        chunks.append(dialog_chunk)

    return chunks, speakers_pool


# ── Test 1: Unit test for strip_thinking ─────────────────────────────────────

def test_strip_thinking():
    raw = '<think>\nSome internal reasoning\n</think>\n{"extractions": []}'
    cleaned = strip_thinking(raw)
    assert cleaned == '{"extractions": []}', f"strip_thinking failed: {cleaned}"
    print("[PASS] test_strip_thinking")


# ── Test 2: Unit test for update_registry ────────────────────────────────────

def test_update_registry():
    registry = {}
    extractions = [
        {
            "speaker_id": "Speaker_0",
            "name": "Sheldon",
            "cue_type": "vocative",
            "evidence_utterance": "Sheldon, this was your idea.",
            "confidence": 0.9,
        },
        {
            "speaker_id": "Speaker_1",
            "name": "Leonard",
            "cue_type": "vocative",
            "evidence_utterance": "Leonard, I don't think I can do this.",
            "confidence": 0.9,
        },
    ]
    registry = update_registry(registry, extractions, "test_dialogue_1")

    assert registry["Speaker_0"].name == "Sheldon"
    assert registry["Speaker_0"].status == "confirmed"  # conf >= 0.9
    assert registry["Speaker_1"].name == "Leonard"
    assert registry["Speaker_1"].status == "confirmed"
    print("[PASS] test_update_registry")


# ── Test 3: Unit test for build_extraction_prompt ────────────────────────────

def test_build_extraction_prompt():
    registry = {
        "Speaker_0": SpeakerRecord(speaker_id="Speaker_0", name="Sheldon", status="confirmed"),
    }
    dialogue = "<Speaker_0> Hello\n<Speaker_1> Hi there"
    prompt = build_extraction_prompt(dialogue, registry)
    assert "Sheldon" in prompt
    assert "Speaker_0" in prompt
    assert "New dialogue to analyze" in prompt
    print("[PASS] test_build_extraction_prompt")


# ── Test 4: End-to-end with real Qwen API call on episode data ───────────────

def test_e2e_speaker_identification():
    episode_path = os.path.join(
        os.path.dirname(__file__), "..",
        "outputs", "bazinga", "TheBigBangTheory", "Season1",
        "TheBigBangTheory.Season01.Episode01.json",
    )
    print(episode_path)
    if not os.path.exists(episode_path):
        print("[SKIP] test_e2e_speaker_identification — episode data not found")
        return

    chunks, speakers_pool = load_episode_chunks(episode_path)
    print(f"\nLoaded {len(chunks)} chunks from Episode 01")
    print(f"Speaker pool (ground truth): {speakers_pool}")

    # for i in range(len(chunks)):
    #     print(f"Chunk {i}:\n{chunks[i][:500]}\n...")
    # exit(0)

    # Override the module-level client/model to use the same env vars as evaluate_agent_results.py
    import audio_script.Speaker_Track.Speaker_Name_tracker as tracker
    tracker.client = OpenAI(
        base_url=os.getenv("QWEN_URL", os.getenv("QWEN_BASE_URL", "http://localhost:8000/v1")),
        api_key=os.getenv("OPENROUTER_API_KEY", os.getenv("QWEN_API_KEY", "EMPTY")),
    )
    tracker.QWEN3_MODEL = os.getenv("QWEN_MODEL_NAME", os.getenv("QWEN3_MODEL", "Qwen/Qwen3-32B"))

    # Use first 3 chunks to keep the test fast
    test_chunks = chunks[:3]
    registry = identify_speakers(test_chunks, enable_thinking=True)

    print("\n── Final Registry ──")
    for sid, rec in sorted(registry.items()):
        print(f"  {sid:12s} → {rec.display_name:20s} [{rec.status}]  (evidence: {len(rec.evidence)})")

    # Resolve the last test chunk
    resolved = resolve_transcript(test_chunks[-1], registry)
    print(f"\n── Resolved transcript (chunk 3) ──\n{resolved[:800]}")

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
                match = rec.name.lower() in gt_name.lower() or gt_name.lower().startswith(rec.name.lower())
                status = "✓" if match else "✗"
                print(f"  {status} {sid}: predicted={rec.name}, ground_truth={gt_name}")

    assert len(identified) > 0, "Expected at least one speaker to be identified"
    print("[PASS] test_e2e_speaker_identification")


# ── Test 5: Qwen API connectivity check ─────────────────────────────────────

def test_qwen_api_connection():
    import audio_script.Speaker_Track.Speaker_Name_tracker as tracker
    tracker.client = OpenAI(
        base_url=os.getenv("QWEN_URL", os.getenv("QWEN_BASE_URL", "http://localhost:8000/v1")),
        api_key=os.getenv("OPENROUTER_API_KEY", os.getenv("QWEN_API_KEY", "EMPTY")),
    )
    tracker.QWEN3_MODEL = os.getenv("QWEN_MODEL_NAME", os.getenv("QWEN3_MODEL", "Qwen/Qwen3-32B"))

    try:
        result = qwen3_chat(
            system="You are a test assistant.",
            user='Return exactly: {"status": "ok"}',
            temperature=0.0,
            max_tokens=64,
        )
        print(f"API response: {result}")
        assert "ok" in result.lower(), f"Unexpected response: {result}"
        print("[PASS] test_qwen_api_connection")
    except Exception as e:
        print(f"[SKIP] test_qwen_api_connection — API not reachable: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Speaker Name Tracker Tests")
    print("=" * 60)

    # Online tests (require Qwen API)
    # test_qwen_api_connection()
    test_e2e_speaker_identification()

    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)
