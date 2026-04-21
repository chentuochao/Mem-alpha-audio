import os
import json
import argparse
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def count_memory_tokens(memory_state, tokenizer):
    """Count total tokens in a memory state (core + semantic + episodic)."""
    total = 0

    if 'core' in memory_state and memory_state['core']:
        core = memory_state['core']
        if isinstance(core, str):
            total += len(tokenizer(core).input_ids)
        elif isinstance(core, list):
            for item in core:
                if isinstance(item, str):
                    total += len(tokenizer(item).input_ids)

    for mem_type in ('semantic', 'episodic'):
        entries = memory_state.get(mem_type, [])
        if entries:
            for item in entries:
                total += len(tokenizer(list(item.values())[0]).input_ids)

    return total


def count_input_tokens(chunks, tokenizer):
    """Count total tokens across all input chunks."""
    total = 0
    for chunk in chunks:
        if isinstance(chunk, str):
            total += len(tokenizer(chunk).input_ids)
    return total


def load_parquet_data(dataset):
    """Load the parquet dataset used during memory construction."""
    dataset_to_path = {
        'memalpha': 'data/memalpha/test.parquet',
        'memalpha_train': 'data/memalpha/train.parquet',
        'memalpha_sample': 'data/memalpha_sample/train.parquet',
        'memoryagentbench': 'data/memoryagentbench/test.parquet',
        'accurate_retrieval': 'data/memoryagentbench/test.parquet',
        'test_time_learning': 'data/memoryagentbench/test.parquet',
        'long_range_understanding': 'data/memoryagentbench/test.parquet',
        'booksum': 'data/memalpha/test.parquet',
        'perltqa': 'data/memalpha/test.parquet',
        'pubmed-rct': 'data/memalpha/test.parquet',
        'squad': 'data/memalpha/processed_squad_data_filtered.parquet',
        'seamlessinteraction_gt': 'outputs/test_gt.parquet',
        'seamlessinteraction_pred': 'outputs/test_pred.parquet',
    }

    if dataset not in dataset_to_path:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: {list(dataset_to_path.keys())}")

    data = pd.read_parquet(dataset_to_path[dataset])

    source_filters = {
        'accurate_retrieval': ['ruler_qa1_197K', 'ruler_qa2_421K', 'longmemeval_s*'],
        'test_time_learning': [
            'icl_banking77_5900shot_balance', 'icl_clinic150_7050shot_balance',
            'icl_nlu_8296shot_balance', 'icl_trec_coarse_6600shot_balance',
            'icl_trec_fine_6400shot_balance',
        ],
        'long_range_understanding': ['infbench_sum_eng_shots2'],
        'booksum': ['booksum'],
        'perltqa': ['perltqa'],
        'pubmed-rct': ['pubmed-rct'],
    }
    if dataset in source_filters:
        data = data[data['data_source'].isin(source_filters[dataset])]

    return data


def evaluate_compression(base_dir, dataset, tokenizer_name="Qwen/Qwen3-32B"):
    """Compute compression ratio for all agent directories under base_dir."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data = load_parquet_data(dataset)

    results = []

    for root, dirs, files in os.walk(base_dir):
        if 'agent_state.json' not in files or 'data_instance_info.json' not in files:
            continue

        with open(os.path.join(root, 'data_instance_info.json'), 'r') as f:
            info = json.load(f)
        global_idx = info.get('global_idx')
        data_source = info.get('data_source', '')

        if global_idx is None or global_idx >= len(data):
            print(f"Warning: skipping {root}, global_idx={global_idx} out of range")
            continue

        row = data.iloc[global_idx]
        chunks = json.loads(row['chunks']) if isinstance(row['chunks'], str) else row['chunks']
        input_tokens = count_input_tokens(chunks, tokenizer)

        with open(os.path.join(root, 'agent_state.json'), 'r') as f:
            memory_state = json.load(f)
        memory_tokens = count_memory_tokens(memory_state, tokenizer)

        compression_ratio = input_tokens / memory_tokens if memory_tokens > 0 else float('inf')

        results.append({
            'agent_dir': root,
            'global_idx': global_idx,
            'data_source': data_source,
            'input_tokens': input_tokens,
            'memory_tokens': memory_tokens,
            'num_chunks': len(chunks),
            'num_semantic': len(memory_state.get('semantic', [])),
            'num_episodic': len(memory_state.get('episodic', [])),
            'has_core': bool(memory_state.get('core')),
            'compression_ratio': compression_ratio,
        })

    return results


def print_summary(results):
    """Print compression ratio summary grouped by data source."""
    if not results:
        print("No results found.")
        return

    grouped = {}
    for r in results:
        src = r['data_source']
        if src not in grouped:
            grouped[src] = []
        grouped[src].append(r)

    print("\nCompression Ratio Summary")
    print("=" * 90)
    print(f"{'Data Source':<35} {'Count':>6} {'Avg Input':>10} {'Avg Memory':>11} {'Avg Ratio':>10} {'Std Ratio':>10}")
    print("-" * 90)

    all_input = []
    all_memory = []
    all_ratio = []

    for source in sorted(grouped.keys()):
        items = grouped[source]
        input_tokens = [r['input_tokens'] for r in items]
        memory_tokens = [r['memory_tokens'] for r in items]
        ratios = [r['compression_ratio'] for r in items if r['compression_ratio'] != float('inf')]

        avg_input = np.mean(input_tokens)
        avg_memory = np.mean(memory_tokens)
        avg_ratio = np.mean(ratios) if ratios else float('inf')
        std_ratio = np.std(ratios) if len(ratios) > 1 else 0.0

        print(f"{source:<35} {len(items):>6} {avg_input:>10.0f} {avg_memory:>11.0f} {avg_ratio:>10.2f}x {std_ratio:>9.2f}")

        all_input.extend(input_tokens)
        all_memory.extend(memory_tokens)
        all_ratio.extend(ratios)

    print("-" * 90)
    print(f"{'OVERALL':<35} {len(results):>6} {np.mean(all_input):>10.0f} {np.mean(all_memory):>11.0f} {np.mean(all_ratio):>10.2f}x {np.std(all_ratio):>9.2f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description='Evaluate memory compression ratio')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory containing agent results (e.g., ./agents/my_agent_memalpha)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=[
                            'memalpha', 'memalpha_train', 'memalpha_sample',
                            'memoryagentbench', 'accurate_retrieval',
                            'test_time_learning', 'long_range_understanding',
                            'booksum', 'perltqa', 'pubmed-rct', 'squad',
                            'seamlessinteraction_gt', 'seamlessinteraction_pred',
                        ],
                        help='Dataset used during memory construction')
    parser.add_argument('--tokenizer', type=str, default='Qwen/Qwen3-32B',
                        help='HuggingFace tokenizer for token counting')
    parser.add_argument('--output', type=str, default='compression_metrics.json',
                        help='Output filename (saved inside base_dir)')

    args = parser.parse_args()

    print(f"Computing compression ratios for {args.base_dir} (dataset={args.dataset})")
    results = evaluate_compression(args.base_dir, args.dataset, args.tokenizer)

    print_summary(results)

    output_path = os.path.join(args.base_dir, args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    main()
