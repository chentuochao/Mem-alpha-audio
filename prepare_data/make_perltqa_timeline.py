#!/usr/bin/env python3
"""
Build a session-timeline JSON for PerLTQA, matching the format of
``outputs/bazinga_data/TBBT_all_seasons_session_timeline.json`` so that
``prepare_audio_parquet.py`` can stamp each dialogue with its real datetime.

Where the time comes from
-------------------------
PerLTQA already carries a timestamp per dialogue: in ``perltmem_en_v2.json`` each
dialogue's ``contents`` is keyed by a datetime string (e.g. "2022-05-12 10:00").
We use that as ``session_timeline_date``.

Key = conv_id
-------------
``prepare_audio_parquet.load_time_maps`` keys the map by ``source_file`` minus
".json", and later looks it up as ``path.split('/')[-3]`` — which for the Step-1/2
layout is the **conv_id** ``<Profile>_<dialogue_folder>`` (e.g. "Cao_Lili_25_0_0_0").
That conv_id is the same whether Step-2/3 ran per-profile or in multi bundles, so
**a single timeline file works for both**.

Output (bazinga-compatible)::
    {
      "history_start_date": null,
      "timeline_unit": "datetime",
      "session_timeline_method": "perltqa_original_datetime",
      "session_timeline_explanation": "...",
      "sessions": [
        {"source_file": "<conv_id>.json", "profile": "...", "dialogue_key": "...",
         "session_timeline_date": "2022-05-12 10:00"},
        ...
      ]
    }

Example::
    python prepare_data/make_perltqa_timeline.py
    python prepare_data/make_perltqa_timeline.py \
        --perltmem /path/perltmem_en_v2.json --out outputs/perltqa_data/perltqa_session_timeline.json
"""

import argparse
import json
import os
import re

DEFAULT_PERLTMEM = "/checkpoint/seamless/tuochao/data/PerLTQA/Dataset/en_v2/perltmem_en_v2.json"
DEFAULT_OUT = "outputs/perltqa_data/perltqa_session_timeline.json"


def safe_filename(name: str) -> str:
    """Mirror perltqa_dialogue_tts.safe_filename (folder/conv_id naming)."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    return name.strip("_") or "unnamed"


def build_sessions(perltmem: dict) -> list:
    sessions = []
    for profile, pv in perltmem.items():
        p_safe = safe_filename(profile)
        for dkey, dv in (pv.get("dialogues", {}) or {}).items():
            contents = dv.get("contents", {}) or {}
            if not contents:
                continue
            # a dialogue's contents is keyed by its datetime; take the first key
            timestamp = next(iter(contents.keys()))
            conv_id = f"{p_safe}_{safe_filename(dkey)}"   # == Step-1/2 conv_id
            sessions.append({
                "source_file": f"{conv_id}.json",
                "profile": profile,
                "dialogue_key": dkey,
                "session_timeline_date": timestamp,
            })
    # deterministic order
    sessions.sort(key=lambda s: s["source_file"])
    return sessions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perltmem", default=DEFAULT_PERLTMEM,
                    help="perltmem_en_v2.json (source of dialogue timestamps)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output timeline JSON path")
    args = ap.parse_args()

    with open(args.perltmem, "r", encoding="utf-8") as f:
        perltmem = json.load(f)

    sessions = build_sessions(perltmem)

    payload = {
        "history_start_date": None,
        "timeline_unit": "datetime",
        "session_timeline_method": "perltqa_original_datetime",
        "session_timeline_explanation": (
            "session_timeline_date is the original PerLTQA dialogue timestamp "
            "(the datetime key of each dialogue's `contents` in perltmem). "
            "source_file (minus .json) equals the Step-1/2 conv_id "
            "<Profile>_<dialogue_folder>, identical across per-profile and multi "
            "bundling."
        ),
        "sessions": sessions,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # quick report
    dates = [s["session_timeline_date"] for s in sessions]
    print(f"profiles: {len(perltmem)}")
    print(f"sessions (dialogues): {len(sessions)}")
    print(f"date range: {min(dates)}  ->  {max(dates)}" if dates else "no sessions")
    print(f"example: {sessions[0] if sessions else None}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
