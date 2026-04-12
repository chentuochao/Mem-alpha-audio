"""
Step 1 Evaluation: Compute DER and cpWER from saved inference results.

Reads the step1_manifest.json produced by step1_diarize_asr.py and evaluates
each conversation against its ground-truth VAD and transcript files.

Metrics:
  - DER  (Diarization Error Rate): compares diar.npy vs vad1.json / vad2.json
  - cpWER (concatenated min-permutation WER): compares predicted per-speaker
    transcripts vs transcript1.json / transcript2.json
"""

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np

import matplotlib.pyplot as plt
from audio_script.datasets.turn_annotation import AlignedProcess
from audio_script.eval.eval_utils import eval_der_seamlessinteraction, eval_cpwer_seamlessinteraction
# ── Helpers ──────────────────────────────────────────────────────────────
TURN_GAP_TH = 1.5





def discover_samples(data_dir: str) -> List[Dict]:
    """
    Walk the directory tree and find all sample folders containing
    sample_info.json produced by Step 1.

    Returns a list of entry dicts, each augmented with ``sample_dir``,
    ``diart_path``, and ``transcript_path``.
    """
    samples = []
    for info_path in sorted(glob.glob(os.path.join(data_dir, "*", "*", "sample_info.json"))):
        sample_dir = os.path.dirname(info_path)
        diar_path = os.path.join(sample_dir, "diart_pred.npy")
        transcript_path = os.path.join(sample_dir, "transcript_pred.json")
        if not os.path.exists(diar_path) or not os.path.exists(transcript_path):
            continue
        with open(info_path, "r") as f:
            info = json.load(f)
        info["sample_dir"] = sample_dir
        info["diart_path"] = diar_path
        info["transcript_path"] = transcript_path
        samples.append(info)
    return samples


# ── Main ────
def main():
    parser = argparse.ArgumentParser(
        description="Step 1 Evaluation: DER and cpWER from saved inference results"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root output directory from Step 1 containing "
             "{spk_pair}/{conv_id}/sample_info.json sub-folders",
    )
    parser.add_argument(
        "--frame_duration",
        type=float,
        default=None,
        help="Override frame duration (seconds) for DER. "
             "If not set, uses feat_len_sec from sample_info.",
    )
    args = parser.parse_args()

    samples = discover_samples(args.data_dir)
    print(f"Found {len(samples)} samples under {args.data_dir}")
    if not samples:
        print("No samples found. Check your --data_dir path.")
        return

    all_ders = []
    all_cpwers = []

    for entry in samples:
        spk_pair = entry["spk_pair"]
        speaker0, speaker1 = spk_pair.split("_")
        conv_id = entry["conv_id"]

        if not conv_id == "V03_S0033_I00000130":
            continue
        print(f"\n{'=' * 70}")
        print(f"Evaluating: {spk_pair} / {conv_id}")
        print(f"{'=' * 70}")

        # ── Load predictions ──────────────────────────────────────────
        diar_result = np.load(entry["diart_path"])
        with open(entry["pred_transcript_path"], "r") as f:
            word_list = json.load(f)
        # print(word_list)
        frame_duration = args.frame_duration or entry.get("feat_len_sec", 0.08)

        # ── Evaluate DER ──────────────────────────────────────────────
        diart_gt_files = {
            speaker0: entry["vad1_path"],
            speaker1: entry["vad2_path"],
        }
        der, best_perm, der_details = eval_der_seamlessinteraction(diar_result, diart_gt_files, frame_duration)
        all_ders.append({
            "conv_id": conv_id,
            "der": der, **der_details,
        })

        # ── Evaluate cpWER ────────────────────────────────────────────
        trans_gt_files = {
            speaker0: entry["transcript1_path"],
            speaker1: entry["transcript2_path"],
        }
        cpwer, best_perm = eval_cpwer_seamlessinteraction(word_list, trans_gt_files)
        all_cpwers.append({
            "spk_pair": spk_pair, "conv_id": conv_id, "cpwer": cpwer,
        })
        # if der > 2.0:
        #     print("  [SKIP cpWER] DER too high")
        #     exit(0)

        perm_index = []
        match_num = min([len(best_perm), 2])
        for i in range(match_num):
            pred_id = best_perm[i]
            gt_index = i
            perm_index.append({"gt_idx": i, "pred_idx": pred_id })

        perm_index_file = entry["diart_path"].replace("diart_pred.npy", "perm_index.json")
        with open(perm_index_file, "w") as f:
            json.dump(perm_index, f, indent=2)

        ### visualize the speaker turn
        speaker_aware_turn = parse_transcript(word_list)
        print("-"*20)
        print("Prediction turn")
        print("-"*20)
        print_turns(speaker_aware_turn)


        print("-"*20)
        print("GT turn")
        print("-"*20)
        turns_gt = []
        for turn in gt_trans1:
            turns_gt.append(turn)
        for turn in gt_trans2:
            turns_gt.append(turn)
        # sort the turns by start time
        turns_gt.sort(key=lambda key: (key['start'], -key['end']))
        print_turns(turns_gt)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 70}")

    if all_ders:

        avg_der = np.mean([d["der"] for d in all_ders])
        avg_miss = np.mean([d["miss"] for d in all_ders])
        avg_fa = np.mean([d["fa"] for d in all_ders])
        avg_conf = np.mean([d["conf"] for d in all_ders])
        # median value
        median_der = np.median([d["der"] for d in all_ders])
        print(f"\nDER ({len(all_ders)} sessions)")
        print(f"  Avg DER:  {avg_der:.4f}")
        print(f"  Median DER: {median_der:.4f}")
        print(f"  Avg Miss: {avg_miss:.2f}s  |  Avg FA: {avg_fa:.2f}s  |  Avg Conf: {avg_conf:.2f}s")
        # for d in all_ders:
        #     print(f"    {d['spk_pair']}/{d['conv_id']}: DER={d['der']:.4f}")

    if all_cpwers:
        avg_cpwer = np.mean([c["cpwer"] for c in all_cpwers])
        median_cpwer = np.median([c["cpwer"] for c in all_cpwers])
        print(f"\ncpWER ({len(all_cpwers)} sessions)")
        print(f"  Avg cpWER: {avg_cpwer:.4f}")
        print(f"  Median cpWER: {median_cpwer:.4f}")
        # for c in all_cpwers:
        #     print(f"    {c['spk_pair']}/{c['conv_id']}: cpWER={c['cpwer']:.4f}")


    # plot the distribution of DER and cpWER
    ders = [d["der"] for d in all_ders]
    ders = [d for d in ders if d <= 2]
    plt.hist(ders, bins=100)
    plt.savefig( os.path.join(args.data_dir, "der_distribution.png"))
    plt.close()

    cpuers = [c["cpwer"] for c in all_cpwers]
    plt.hist(cpuers, bins=100)
    plt.savefig( os.path.join(args.data_dir, "cpwer_distribution.png"))
    plt.close()
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
