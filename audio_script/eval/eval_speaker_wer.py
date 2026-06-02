"""
Evaluate speaker-aware WER by comparing parsed_dialog_gt.json and
parsed_dialog_pred.json across all subfolders.

For each subfolder/chunk:
  1. Load gt and pred dialog JSONs
  2. Map pred GLOBAL_SPK_XXX -> real speaker ID via speaker_map.json
  3. Concatenate text per speaker (ignore unknown/FP speakers)
  4. Compute WER per matched speaker, then aggregate

Usage:
    python -m audio_script.eval.eval_speaker_wer \
        --data_dir outputs/demo_output_step2
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import editdistance


def word_error_rate(
    hypotheses: List[str], references: List[str], use_cer=False
) -> float:
    scores = 0
    words = 0
    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )
    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()
        words += len(r_list)
        scores += editdistance.eval(h_list, r_list)
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float("inf")
    return wer


def concat_text_by_speaker(dialog: List[Dict]) -> Dict[str, str]:
    """Concatenate all utterance texts per speaker from a dialog list."""
    speaker_texts: Dict[str, List[str]] = defaultdict(list)
    for turn in dialog:
        speaker_texts[turn["speaker"]].append(turn["text"])
    return {spk: " ".join(texts) for spk, texts in speaker_texts.items()}


def match_speaker_name(pred_name: str, gt_name: str) -> bool:
    """Check if predicted speaker name matches gt name via partial/fuzzy matching."""
    pred_lower = pred_name.lower().strip()
    gt_lower = gt_name.lower().strip()
    if not pred_lower or not gt_lower:
        return False
    firstname_pred = pred_lower.split()[0]
    return (
        pred_lower in gt_lower
        or gt_lower.startswith(pred_lower)
        or gt_lower.startswith(firstname_pred)
    )


def build_speaker_matching(
    pred_speakers: List[str], gt_speakers: List[str]
) -> Dict[str, str]:
    """Match pred speaker names to gt speaker names using fuzzy matching.

    Returns a mapping from pred speaker name to gt speaker name.
    """
    pred_to_gt: Dict[str, str] = {}
    used_gt = set()
    for pred_spk in pred_speakers:
        for gt_spk in gt_speakers:
            if gt_spk in used_gt:
                continue
            if match_speaker_name(pred_spk, gt_spk):
                pred_to_gt[pred_spk] = gt_spk
                used_gt.add(gt_spk)
                break
    return pred_to_gt


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate speaker-aware WER between gt and pred dialogs"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root output directory containing {spk_pair}/{conv_id}/ subfolders",
    )
    parser.add_argument(
        "--use_cer", action="store_true",
        help="Compute CER instead of WER",
    )
    args = parser.parse_args()

    speaker_map_path = os.path.join(args.data_dir, "extracted_speaker_name.json")
    with open(speaker_map_path, "r") as f:
        speaker_map = json.load(f)

    gt_files = sorted(glob.glob(
        os.path.join(args.data_dir, "*", "*", "parsed_dialog_gt.json")
    ))
    print(f"Found {len(gt_files)} chunks")

    all_hypotheses = []
    all_references = []
    per_chunk_wers = []
    skipped_speakers = 0
    matched_speakers = 0

    for gt_path in gt_files:
        chunk_dir = os.path.dirname(gt_path)
        pred_path = os.path.join(chunk_dir, "parsed_dialog_pred.json")
        if not os.path.exists(pred_path):
            print(f"  WARNING: missing pred file for {chunk_dir}, skipping")
            continue

        rel_path = os.path.relpath(chunk_dir, args.data_dir)

        with open(gt_path, "r") as f:
            gt_dialog = json.load(f)
        with open(pred_path, "r") as f:
            pred_dialog = json.load(f)

        gt_by_speaker = concat_text_by_speaker(gt_dialog)

        # Map pred speakers to extracted names via speaker_map
        mapped_pred_dialog = []
        for turn in pred_dialog:
            raw_spk = turn["speaker"]
            mapped_spk = speaker_map.get(raw_spk, raw_spk)
            if mapped_spk.startswith("FP_") or mapped_spk.startswith("GLOBAL_SPK_"):
                continue
            mapped_pred_dialog.append({"speaker": mapped_spk, "text": turn["text"]})

        pred_by_speaker = concat_text_by_speaker(mapped_pred_dialog)

        # Fuzzy-match pred speaker names to gt speaker names
        pred_to_gt = build_speaker_matching(
            list(pred_by_speaker.keys()), list(gt_by_speaker.keys())
        )
        matched_gt = set(pred_to_gt.values())
        gt_only = set(gt_by_speaker.keys()) - matched_gt
        pred_only = set(pred_by_speaker.keys()) - set(pred_to_gt.keys())

        if gt_only:
            skipped_speakers += len(gt_only)
        if pred_only:
            skipped_speakers += len(pred_only)

        chunk_hyps = []
        chunk_refs = []
        hyps_cat = ""
        refs_cat = ""
        for pred_spk, gt_spk in sorted(pred_to_gt.items()):
            chunk_hyps.append(pred_by_speaker[pred_spk])
            hyps_cat += pred_by_speaker[pred_spk]
            chunk_refs.append(gt_by_speaker[gt_spk])
            refs_cat += gt_by_speaker[gt_spk]
            matched_speakers += 1

        if chunk_hyps:
            chunk_wer = word_error_rate(chunk_hyps, chunk_refs, use_cer=args.use_cer)
            # chunk_wer = word_error_rate([hyps_cat], [refs_cat], use_cer=args.use_cer)
            per_chunk_wers.append(chunk_wer)
            all_hypotheses.extend(chunk_hyps)
            all_references.extend(chunk_refs)
            print(f"  {rel_path}: WER={chunk_wer:.4f}  "
                  f"(matched={len(pred_to_gt)}, "
                  f"gt_only={len(gt_only)}, pred_only={len(pred_only)}, "
                  f"mapping={pred_to_gt})")
        else:
            print(f"  {rel_path}: NO matched speakers "
                  f"(gt={set(gt_by_speaker.keys())}, pred={set(pred_by_speaker.keys())})")

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  Total chunks evaluated: {len(per_chunk_wers)}/{len(gt_files)}")
    print(f"  Matched speakers: {matched_speakers}")
    print(f"  Skipped (unmatched) speakers: {skipped_speakers}")

    if all_hypotheses:
        overall_wer = word_error_rate(all_hypotheses, all_references, use_cer=args.use_cer)
        metric_name = "CER" if args.use_cer else "WER"
        print(f"  Overall speaker-aware {metric_name}: {overall_wer:.4f}")
        if per_chunk_wers:
            avg_wer = sum(per_chunk_wers) / len(per_chunk_wers)
            print(f"  Average per-chunk {metric_name}: {avg_wer:.4f}")
    else:
        print("  No matched speakers found — cannot compute WER.")


if __name__ == "__main__":
    main()
