"""
Count token statistics of transcriptions in Mix_Mosaic dataset.
Aggregates token counts by speaker pair, and plots the distribution.
"""

import os
import json
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TOKEN_SCALE = 1.33  # word-to-subword multiplier


def count_tokens_in_transcript(transcript_data):
    """Count whitespace-delimited tokens from a transcript JSON (list of utterance dicts)."""
    total_tokens = 0
    for entry in transcript_data:
        text = entry.get("text", "")
        if text:
            total_tokens += len(text.split())
    return total_tokens


def collect_stats(data_root):
    pair_stats = defaultdict(lambda: {
        "total_tokens": 0,
        "num_conversations": 0,
        "per_conv_tokens": [],
    })

    total_tokens_all = 0
    total_convs = 0

    for dirpath, dirnames, filenames in os.walk(data_root):
        if "transcript1.json" not in filenames:
            continue

        conv_id = os.path.basename(dirpath)
        speaker_pair = os.path.basename(os.path.dirname(dirpath))

        conv_tokens = 0
        for tf in ["transcript1.json", "transcript2.json"]:
            tf_path = os.path.join(dirpath, tf)
            if not os.path.exists(tf_path):
                continue
            with open(tf_path, "r") as f:
                transcript = json.load(f)
            conv_tokens += count_tokens_in_transcript(transcript) * TOKEN_SCALE

        pair_stats[speaker_pair]["total_tokens"] += conv_tokens
        pair_stats[speaker_pair]["num_conversations"] += 1
        pair_stats[speaker_pair]["per_conv_tokens"].append((conv_id, conv_tokens))

        total_tokens_all += conv_tokens
        total_convs += 1

    return pair_stats, total_tokens_all, total_convs


def print_table(pair_stats, total_tokens_all, total_convs, data_root):
    print("=" * 80)
    print(f"Token Statistics for: {data_root}")
    print("=" * 80)
    print(f"{'Speaker Pair':<30} {'Conversations':>15} {'Total Tokens':>15} {'Avg Tokens/Conv':>18}")
    print("-" * 80)

    for pair in sorted(pair_stats.keys()):
        stats = pair_stats[pair]
        avg = stats["total_tokens"] / stats["num_conversations"] if stats["num_conversations"] > 0 else 0
        print(f"{pair:<30} {stats['num_conversations']:>15} {stats['total_tokens']:>15.0f} {avg:>18.1f}")

    print("-" * 80)
    avg_all = total_tokens_all / total_convs if total_convs > 0 else 0
    print(f"{'TOTAL':<30} {total_convs:>15} {total_tokens_all:>15.0f} {avg_all:>18.1f}")
    print("=" * 80)


def plot_distributions(pair_stats, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    all_conv_tokens = []
    pair_names = []
    pair_totals = []
    pair_avgs = []

    for pair in sorted(pair_stats.keys()):
        stats = pair_stats[pair]
        tokens_list = [t for _, t in stats["per_conv_tokens"]]
        all_conv_tokens.extend(tokens_list)
        pair_names.append(pair)
        pair_totals.append(stats["total_tokens"])
        n = stats["num_conversations"]
        pair_avgs.append(stats["total_tokens"] / n if n > 0 else 0)

    # --- Figure 1: histogram of per-conversation token counts ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_conv_tokens, bins=40, edgecolor="black", alpha=0.75, color="#4C72B0")
    ax.set_xlabel("Token count per conversation", fontsize=12)
    ax.set_ylabel("Number of conversations", fontsize=12)
    ax.set_title("Distribution of token counts across all conversations", fontsize=14)
    mean_val = np.mean(all_conv_tokens)
    median_val = np.median(all_conv_tokens)
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {mean_val:.0f}")
    ax.axvline(median_val, color="orange", linestyle="--", linewidth=1.5, label=f"Median = {median_val:.0f}")
    ax.legend(fontsize=11)
    fig.tight_layout()
    path1 = os.path.join(out_dir, "token_hist_all_convs.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"Saved: {path1}")

    # --- Figure 2: total tokens per speaker pair (bar chart) ---
    fig, ax = plt.subplots(figsize=(max(8, len(pair_names) * 0.6), 6))
    x = np.arange(len(pair_names))
    ax.bar(x, pair_totals, color="#55A868", edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Total tokens", fontsize=12)
    ax.set_title("Total token count per speaker pair", fontsize=14)
    fig.tight_layout()
    path2 = os.path.join(out_dir, "token_total_per_pair.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"Saved: {path2}")

    # --- Figure 3: box plot of per-conv tokens grouped by speaker pair ---
    grouped_tokens = []
    labels = []
    for pair in sorted(pair_stats.keys()):
        tokens_list = [t for _, t in pair_stats[pair]["per_conv_tokens"]]
        grouped_tokens.append(tokens_list)
        labels.append(pair)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 6))
    bp = ax.boxplot(grouped_tokens, patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#C44E52")
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Tokens per conversation", fontsize=12)
    ax.set_title("Token distribution per speaker pair (box plot)", fontsize=14)
    fig.tight_layout()
    path3 = os.path.join(out_dir, "token_boxplot_per_pair.png")
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"Saved: {path3}")


def main():
    parser = argparse.ArgumentParser(description="Count token statistics in Mix_Mosaic transcriptions.")
    parser.add_argument("--data_root", type=str,
                        default="/checkpoint/seamless/tuochao/data/Mix_Mosaic",
                        help="Root path to Mix_Mosaic dataset")
    parser.add_argument("--plot_dir", type=str, default=None,
                        help="Directory to save plots (default: <data_root>/plots)")
    args = parser.parse_args()

    data_root = args.data_root
    plot_dir = args.plot_dir or os.path.join(data_root, "plots")

    pair_stats, total_tokens_all, total_convs = collect_stats(data_root)

    if total_convs == 0:
        print("No conversations found. Check --data_root path.")
        return

    print_table(pair_stats, total_tokens_all, total_convs, data_root)
    plot_distributions(pair_stats, plot_dir)


if __name__ == "__main__":
    main()
