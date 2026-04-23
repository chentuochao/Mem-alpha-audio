"""
Decoupled scoring step for long context evaluation.

Reads a predictions JSON file produced by long_context_generate.py, computes
evaluation scores for every entry, prints per-source metrics, and optionally
saves a scored results file.

Expected input format (produced by long_context_generate.py):
    [
      {
        "question":    "...",
        "response":    "...",
        "answer":      "...",
        "data_source": "...",
        "index":       <int>
      },
      ...
    ]

Usage examples:
    # Score a predictions file
    python long_context_score.py --predictions results/gpt_4o_mini_memalpha_predictions.json

    # Save scored results to a specific file
    python long_context_score.py \\
        --predictions results/gpt_4o_mini_memalpha_predictions.json \\
        --output_file results/gpt_4o_mini_memalpha_scored.json

    # LLM-based scoring for 'friends' / 'longmemeval_s*' datasets requires model clients;
    # the script initialises them automatically from environment variables.
"""

import re
import os
import json
import argparse
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import dotenv
dotenv.load_dotenv()

from memalpha.utils import evaluate_eurlex
from memalpha.llm_agent.metrics import evaluate_wrt_source, _extract_answer_from_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

QWEN_URL = os.getenv("QWEN_URL")


# ---------------------------------------------------------------------------
# Scoring logic  (mirrors LongContextEvaluator._compute_score exactly)
# ---------------------------------------------------------------------------

def compute_score(
    data_source: str,
    predicted_answer: str,
    gold_answer: Any,
    question: Optional[str] = None,
    azure_client=None,
    qwen_client=None,
) -> float:
    """
    Compute evaluation score for a single prediction.

    Parameters
    ----------
    data_source      : dataset identifier string
    predicted_answer : raw model response
    gold_answer      : ground-truth answer (str, list, or dict depending on dataset)
    question         : original question text (required for LLM-based scoring)
    azure_client     : AzureOpenAI client – required only for 'friends' dataset
    qwen_client      : OpenAI client pointed at Qwen – required for 'longmemeval_s*'
    """
    if "<think>" in predicted_answer and "</think>" in predicted_answer:
        predicted_answer = predicted_answer.split("</think>")[1].strip()
    if "<think>" in predicted_answer:
        predicted_answer = "Empty"

    # ------------------------------------------------------------------
    # booksum / infbench_sum_eng_shots2  →  keyword hit-rate
    # ------------------------------------------------------------------
    if data_source == "booksum" or data_source == "infbench_sum_eng_shots2":
        if not isinstance(gold_answer, list):
            gold_answer = gold_answer.split(", ")
        hit = sum(1 for kw in gold_answer if kw.lower() in predicted_answer.lower())
        return hit / len(gold_answer)

    # ------------------------------------------------------------------
    # eurlex  →  F1 via boxed extraction
    # ------------------------------------------------------------------
    elif data_source == "eurlex":
        extracted = _extract_answer_from_response(predicted_answer)
        return evaluate_eurlex([extracted], [gold_answer])

    # ------------------------------------------------------------------
    # friends  →  LLM judge (CORRECT / WRONG)
    # ------------------------------------------------------------------
    elif data_source == "friends":
        if question is None:
            return 0.0
        if azure_client is None:
            logger.warning(
                "Skipping LLM-based scoring for 'friends': no azure_client available. "
                "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT env vars."
            )
            return 0.0

        ACCURACY_PROMPT = (
            "Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. "
            "You will be given the following data:\n"
            "  (1) a question (posed by one user to another user),\n"
            "  (2) a 'gold' (ground truth) answer,\n"
            "  (3) a generated answer\n"
            "which you will score as CORRECT/WRONG.\n\n"
            "The point of the question is to ask about the reason of someone saying something. "
            "The gold answer is a concise and short answer referring to the evidence happened before.\n\n"
            "The key is to identify whether the answer correctly identifies the evidence mentioned in "
            "the golden answer. Rephrasing is allowed, as long as it is related to the evidence. "
            "All other answers that are too general, unrelated to the evidence, should be marked as WRONG.\n\n"
            "Question: {question}\n"
            "Gold answer: {gold_answer}\n"
            "Generated answer: {generated_answer}\n\n"
            "First, provide a short (one sentence) explanation of your reasoning, "
            "then finish with CORRECT or WRONG.\n"
            "Do NOT include both CORRECT and WRONG in your response.\n"
            "Return the label in JSON format with key \"label\"."
        )
        response = azure_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": ACCURACY_PROMPT.format(
                    question=question.split("\n\n")[-1],
                    gold_answer=gold_answer,
                    generated_answer=predicted_answer,
                ),
            }],
            max_tokens=100,
            temperature=0.1,
        )
        label = json.loads(response.choices[0].message.content)["label"]
        return 1.0 if label == "CORRECT" else 0.0

    # ------------------------------------------------------------------
    # wos46985  →  hierarchical partial score
    # ------------------------------------------------------------------
    elif data_source == "wos46985":
        extracted_answer = _extract_answer_from_response(predicted_answer).strip("\"'").strip()
        gold_answer_str = str(gold_answer).strip("\"'").strip()

        if not re.match(r"^\d+\s*>\s*\d+\s*>\s*\d+$", extracted_answer):
            return 0.0

        try:
            pred_parts = [p.strip() for p in extracted_answer.split(">")]
            gold_parts = [p.strip() for p in gold_answer_str.split(">")]
            if len(pred_parts) != 3 or len(gold_parts) != 3:
                return 0.0

            weights = [0.5, 0.3, 0.2]
            total = 0.0
            for i in range(3):
                if pred_parts[i] == gold_parts[i]:
                    total += weights[i]
                else:
                    break
            return total
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # pubmed-rct  →  exact single-digit match
    # ------------------------------------------------------------------
    elif data_source == "pubmed-rct":
        extracted = _extract_answer_from_response(predicted_answer).strip("\"'").strip()
        if not re.match(r"^\d+$", extracted):
            return 0.0
        return 1.0 if extracted == str(gold_answer).strip("\"'").strip() else 0.0

    # ------------------------------------------------------------------
    # test_time_learning / pubmed*  →  exact single-digit match
    # ------------------------------------------------------------------
    elif data_source == "test_time_learning" or data_source.startswith("pubmed"):
        extracted = _extract_answer_from_response(predicted_answer).strip("\"'").strip()
        if not re.match(r"^\d+$", extracted):
            return 0.0
        return 1.0 if extracted == str(gold_answer).strip("\"'").strip() else 0.0

    # ------------------------------------------------------------------
    # perltqa  →  substring containment (multi-answer aware)
    # ------------------------------------------------------------------
    elif data_source == "perltqa":
        if ";" in gold_answer:
            parts = gold_answer.split(";")
            hits = sum(1 for p in parts if p.lower().strip() in predicted_answer.lower())
            return hits / len(parts)
        return 1.0 if gold_answer.lower() in predicted_answer.lower() else 0.0

    # ------------------------------------------------------------------
    # arxiv-classification  →  exact single-digit match
    # ------------------------------------------------------------------
    elif data_source == "arxiv-classification":
        extracted = _extract_answer_from_response(predicted_answer).strip("\"'").strip()
        if not re.match(r"^\d+$", extracted):
            return 0.0
        return 1.0 if extracted == str(gold_answer).strip("\"'").strip() else 0.0

    # ------------------------------------------------------------------
    # narrativeqa  →  containment
    # ------------------------------------------------------------------
    elif data_source == "narrativeqa":
        if isinstance(gold_answer, list):
            answer_text = str(gold_answer[0]) if gold_answer else ""
        else:
            answer_text = (
                gold_answer.get("text", gold_answer)
                if isinstance(gold_answer, dict)
                else str(gold_answer)
            )
        return 1.0 if answer_text.lower() in predicted_answer.lower() else 0.0

    # ------------------------------------------------------------------
    # longmemeval_s*  →  LLM judge (yes / no)
    # ------------------------------------------------------------------
    elif data_source.startswith("longmemeval_s"):
        if qwen_client is None:
            logger.warning(
                "Skipping LLM-based scoring for '%s': no qwen_client available. "
                "Set QWEN_URL env var.",
                data_source,
            )
            return 0.0

        template = (
            "I will give you a question, a rubric for desired personalized response, "
            "and a response from a model. Please answer yes if the response satisfies "
            "the desired response. Otherwise, answer no. The model does not need to "
            "reflect all the points in the rubric. The response is correct as long as "
            "it recalls and utilizes the user's personal information correctly.\n\n"
            "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
        prompt = template.format(question, gold_answer, predicted_answer)
        response = qwen_client.chat.completions.create(
            model="qwen3-32b",
            messages=[{"role": "user", "content": prompt}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content.strip().lower()
        return 1.0 if ("yes" in content and "no" not in content) else 0.0

    # ------------------------------------------------------------------
    # RULER accurate-retrieval  →  containment
    # ------------------------------------------------------------------
    elif data_source in ("ruler_qa1_197K", "ruler_qa2_421K"):
        if isinstance(gold_answer, list):
            answer_text = str(gold_answer[0]) if gold_answer else ""
        else:
            answer_text = (
                gold_answer.get("text", gold_answer)
                if isinstance(gold_answer, dict)
                else str(gold_answer)
            )
        return 1.0 if answer_text.lower() in predicted_answer.lower() else 0.0

    # ------------------------------------------------------------------
    # squad / hotpotqa  →  containment
    # ------------------------------------------------------------------
    elif data_source in ("squad", "hotpotqa"):
        if isinstance(gold_answer, list):
            answer_text = str(gold_answer[0]) if gold_answer else ""
        else:
            answer_text = (
                gold_answer.get("text", gold_answer)
                if isinstance(gold_answer, dict)
                else str(gold_answer)
            )
        return 1.0 if answer_text.lower() in predicted_answer.lower() else 0.0

    # ------------------------------------------------------------------
    # Fallback  →  memory-agent-bench generic scorer
    # ------------------------------------------------------------------
    else:
        return evaluate_wrt_source({"output": predicted_answer}, gold_answer, data_source)


# ---------------------------------------------------------------------------
# Client initialisation helpers
# ---------------------------------------------------------------------------

def _try_build_azure_client():
    """Return an AzureOpenAI client if credentials are available, else None."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not api_key or not endpoint:
        return None
    try:
        from openai import AzureOpenAI
        return AzureOpenAI(api_key=api_key, api_version="2025-01-01-preview", azure_endpoint=endpoint)
    except Exception as exc:
        logger.warning("Could not initialise AzureOpenAI client: %s", exc)
        return None


def _try_build_qwen_client():
    """Return an OpenAI-compat Qwen client if QWEN_URL is set, else None."""
    if not QWEN_URL:
        return None
    try:
        from openai import OpenAI
        return OpenAI(base_url=QWEN_URL, api_key="EMPTY")
    except Exception as exc:
        logger.warning("Could not initialise Qwen client: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_metrics(scored: List[Dict]) -> Dict:
    """
    Compute per-source average scores, mirroring LongContextEvaluator.calculate_metrics.

    Groups results by (index, data_source), averages scores within each group,
    then averages across groups per source.
    """
    index_to_results: Dict[Any, List[Dict]] = defaultdict(list)
    for r in scored:
        index_to_results[r.get("index", 0)].append(r)

    index_summary = {
        idx: {
            "score": float(np.mean([x["score"] for x in items if x.get("score") is not None])),
            "source": items[0].get("data_source", "unknown"),
        }
        for idx, items in index_to_results.items()
    }

    source_groups: Dict[str, List[float]] = defaultdict(list)
    for v in index_summary.values():
        source_groups[v["source"]].append(v["score"])

    metrics = {
        src: {"score": float(np.mean(scores)), "count": len(scores)}
        for src, scores in source_groups.items()
    }
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score predictions produced by long_context_generate.py.",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to the predictions JSON file produced by long_context_generate.py",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional path to save scored results JSON (default: <predictions>_scored.json)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load predictions
    # ------------------------------------------------------------------
    if not os.path.exists(args.predictions):
        raise FileNotFoundError(f"Predictions file not found: {args.predictions}")

    with open(args.predictions, "r", encoding="utf-8") as f:
        predictions: List[Dict] = json.load(f)

    logger.info("Loaded %d predictions from %s", len(predictions), args.predictions)

    # ------------------------------------------------------------------
    # Determine which datasets are present and initialise clients lazily
    # ------------------------------------------------------------------
    data_sources = {p.get("data_source", "") for p in predictions}
    needs_azure = "friends" in data_sources
    needs_qwen = any(ds.startswith("longmemeval_s") for ds in data_sources)

    azure_client = _try_build_azure_client() if needs_azure else None
    qwen_client = _try_build_qwen_client() if needs_qwen else None

    # ------------------------------------------------------------------
    # Score every prediction
    # ------------------------------------------------------------------
    scored: List[Dict] = []
    for pred in predictions:
        score = compute_score(
            data_source=pred.get("data_source", ""),
            predicted_answer=pred.get("response", ""),
            gold_answer=pred.get("answer", ""),
            question=pred.get("question"),
            azure_client=azure_client,
            qwen_client=qwen_client,
        )
        scored.append({**pred, "score": score})

    # ------------------------------------------------------------------
    # Compute and print metrics
    # ------------------------------------------------------------------
    metrics = calculate_metrics(scored)

    print("\n=== Evaluation Results ===")
    all_scores = [s["score"] for s in scored if s.get("score") is not None]
    print(f"Total predictions  : {len(scored)}")
    print(f"Overall avg score  : {np.mean(all_scores):.4f}" if all_scores else "No valid scores.")
    print()
    print(f"{'Data source':<40} {'Avg score':>10} {'Count':>8}")
    print("-" * 60)
    for src, info in sorted(metrics.items()):
        print(f"{src:<40} {info['score']:>10.4f} {info['count']:>8}")
    print()

    # ------------------------------------------------------------------
    # Save scored results
    # ------------------------------------------------------------------
    output_file = args.output_file
    if output_file is None:
        base, ext = os.path.splitext(args.predictions)
        output_file = f"{base}_scored{ext}"

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": scored}, f, indent=2, ensure_ascii=False)

    logger.info("Scored results saved to %s", output_file)

    return scored, metrics


if __name__ == "__main__":
    main()
