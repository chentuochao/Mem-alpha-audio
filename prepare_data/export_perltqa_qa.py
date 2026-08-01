#!/usr/bin/env python3
"""
Export the per-bundle QA from a PerLTQA bundle manifest into custom_qa_dir-ready
files, so run_qa_evaluation.py --custom_qa_dir can consume them.

run_qa_evaluation.load_custom_qa_from_dir globs a *directory* and reads each file
as JSONL (one QA object per line), loading everything in that dir into ONE
conversation instance. So each bundle gets its own sub-directory:

    <out>/qa_<mode>/bundle_<id>/qa.jsonl        # that bundle's QAs (one per line)

Each line has at least {question, answer}; profile / evidence_chunks are kept for
traceability (ignored by the evaluator). data_source is set at eval time via
--dataset perltqa, so it isn't needed here.

Example:
    python prepare_data/export_perltqa_qa.py \
        --manifest /checkpoint/.../dialogue_tts_en_v2/bundles_multi.json
    python prepare_data/export_perltqa_qa.py \
        --manifest /checkpoint/.../dialogue_tts_en_v2/bundles_per_profile.json
"""

import argparse
import json
import os

DEFAULT_OUT = "/storage/home/tuochao/Mem-alpha-audio/outputs/perltqa_data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True,
                    help="bundles_multi.json or bundles_per_profile.json")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"output root (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    mode = manifest.get("meta", {}).get("mode", "bundles")
    root = os.path.join(args.out, f"qa_{mode}")
    os.makedirs(root, exist_ok=True)

    total_qa = 0
    n_bundles = 0
    empty = 0
    for b in manifest["bundles"]:
        bid = b["bundle_id"]
        qa = b.get("qa", []) or []
        if not qa:
            empty += 1
            continue
        bundle_dir = os.path.join(root, f"bundle_{bid}")
        os.makedirs(bundle_dir, exist_ok=True)
        out_path = os.path.join(bundle_dir, "qa.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in qa:
                line = {
                    "question": item["question"],
                    "answer": item["answer"],
                    "profile": item.get("profile"),
                    "evidence_chunks": item.get("evidence_chunks", []),
                    "memory_type": item.get("memory_type", "dialogues"),
                    "data_source": "perltqa",
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        total_qa += len(qa)
        n_bundles += 1

    print(f"mode={mode}")
    print(f"bundles with QA written : {n_bundles}  (skipped {empty} empty)")
    print(f"total QA lines          : {total_qa}")
    print(f"layout                  : {root}/bundle_<id>/qa.jsonl")
    print(f"use with                : --custom_qa_dir {root}/bundle_<id>/")


if __name__ == "__main__":
    main()
