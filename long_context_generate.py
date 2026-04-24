"""
Decoupled generation step for long context evaluation.

Runs model inference and saves raw predictions to a JSON file without computing
any evaluation scores.  Scoring is handled separately by long_context_score.py.

Output format (one JSON array):
    [
      {
        "question":     "...",
        "response":     "...",   # model prediction
        "answer":       "...",   # gold answer
        "data_source":  "...",
        "index":        <int>
      },
      ...
    ]

Usage examples:
    python long_context_generate.py --model gpt-4o-mini --dataset memalpha
    python long_context_generate.py --model qwen3-32b --dataset pubmed-rct --test_samples 100
    python long_context_generate.py --model gpt-4o-mini --dataset booksum \\
        --output_file results/my_predictions.json
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import logging
from datetime import datetime

import dotenv
dotenv.load_dotenv()

from long_context_eval import LongContextEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LongContextGenerator(LongContextEvaluator):
    """
    Generation-only variant of LongContextEvaluator.

    Overrides _compute_score to always return None so that the inherited
    run_test_evaluation never blocks on scoring API calls.  All other
    generation logic (batching, BM25, MemAgent, checkpoints, …) is reused
    unchanged from the parent.
    """

    def _compute_score(self, data_source, predicted_answer, gold_answer, question=None):
        """Skip scoring – predictions are scored later by long_context_score.py."""
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_output_path(model: str, dataset: str, without_chunks: bool) -> str:
    safe_model = model.replace(".", "_").replace("-", "_")
    chunks_suffix = "_no_chunks" if without_chunks else "_with_chunks"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(
        "results",
        f"{safe_model}{chunks_suffix}_{dataset}_predictions_{timestamp}.json",
    )


def save_predictions(results: list, output_file: str) -> None:
    """Convert internal result dicts to the flat predictions format and write JSON."""
    predictions = []
    for i, r in enumerate(results):
        predictions.append(
            {
                "question": r.get("question", ""),
                "response": r.get("predicted_answer", ""),
                "answer": r.get("answer", ""),
                "data_source": r.get("data_source", ""),
                "index": r.get("index", i),
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d predictions to %s", len(predictions), output_file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate QA predictions and save to JSON (no scoring).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        choices=[
            "gpt-4o-mini", "gpt-4o-mini-bm25", "gpt-4.1-mini",
            "qwen3-32b", "qwen3-32b-bm25",
            "memagent-7b", "memagent-14b",
            "mem1",
        ],
        help="Model to use for generation",
    )
    parser.add_argument(
        "--without_chunks",
        action="store_true",
        help="Run without chunks (direct question answering)",
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=-1,
        help="Number of test samples to evaluate (-1 for all)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="memalpha",
        choices=[
            "memalpha", "pubmed-rct", "booksum", "perltqa",
            "seamlessinteraction_gt", "seamlessinteraction_pred",
            "long_range_understanding", "accurate_retrieval",
            "test_time_learning", "longmemeval", "memoryagentbench",
            "squad", "hotpotqa",
        ],
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path for the output predictions JSON (auto-generated if not specified)",
    )
    args = parser.parse_args()

    output_file = args.output_file or _default_output_path(
        args.model, args.dataset, args.without_chunks
    )

    logger.info(
        "Configuration: model=%s  without_chunks=%s  dataset=%s  test_samples=%d",
        args.model, args.without_chunks, args.dataset, args.test_samples,
    )

    generator = LongContextGenerator(
        dataset=args.dataset,
        model_name=args.model,
        without_chunks=args.without_chunks,
        force_rescore=False,
    )

    # Clean up any corrupted checkpoints first
    generator.clean_corrupted_checkpoints()

    # Load data and run generation (scoring is skipped via _compute_score override)
    generator.load_datasets(args.dataset)
    generator._load_existing_results(args.dataset)
    test_results = generator.run_test_evaluation(args.test_samples, args.dataset)

    save_predictions(test_results, output_file)

    print(f"\nGeneration complete.")
    print(f"  Predictions : {len(test_results)}")
    print(f"  Output file : {output_file}")

    return test_results, output_file


if __name__ == "__main__":
    main()
