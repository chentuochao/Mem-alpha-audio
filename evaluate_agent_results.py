import re
import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from typing import List, Dict, Any, Optional
from openai import OpenAI
from transformers import AutoTokenizer

from memalpha.llm_agent.metrics import evaluate_wrt_source, _extract_answer_from_response

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token-counting helpers (shared with compression-ratio evaluation)
# ---------------------------------------------------------------------------

def count_memory_tokens(memory_state: dict, tokenizer) -> int:
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
        for item in memory_state.get(mem_type) or []:
            total += len(tokenizer(list(item.values())[0]).input_ids)
    return total


def count_input_tokens(chunks: list, tokenizer) -> int:
    """Count total tokens across all input chunks."""
    return sum(len(tokenizer(c).input_ids) for c in chunks if isinstance(c, str))


def load_parquet_data(dataset: str) -> pd.DataFrame:
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

class AgentResultsEvaluator:
    """
    Evaluator for agent results that uses data source specific evaluation methods
    borrowed from long_context_eval.py
    """

    def __init__(self, tokenizer_name: str = "Qwen/Qwen3-32B"):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.client = OpenAI(
            base_url=os.getenv("QWEN_URL"),
            api_key=os.getenv("OPENROUTER_API_KEY", "EMPTY")
        )
        self.qwen_model = os.getenv("QWEN_MODEL_NAME")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _compute_rouge_score(self, prediction: str, reference: str) -> float:
        """Compute ROUGE score between prediction and reference text."""
        scores = self.rouge_scorer.score(reference, prediction)
        rouge1_f1 = scores['rouge1'].fmeasure
        rouge2_f1 = scores['rouge2'].fmeasure
        rougeL_f1 = scores['rougeL'].fmeasure

        return (rouge1_f1 + rouge2_f1 + rougeL_f1) / 3.0

    def _compute_score(self, data_source, predicted_answer, gold_answer, question=None):

        if "<think>" and "</think>" in predicted_answer:
            predicted_answer = predicted_answer.split("</think>")[1].strip()
        if "<think>" in predicted_answer:
            predicted_answer = "Empty"

        """Compute evaluation score based on data source."""
        if data_source == 'booksum' or data_source == 'infbench_sum_eng_shots2':

            if not isinstance(gold_answer, list):
                gold_answer = gold_answer.split(", ")

            hit = 0
            for keyword in gold_answer:
                if keyword.lower() in predicted_answer.lower():
                    hit += 1

            return hit / len(gold_answer)

        elif data_source == 'pubmed-rct' or "icl" in data_source:
            # PUBMED dataset evaluation: MUST be ONLY a single digit
            extracted_answer = _extract_answer_from_response(predicted_answer)

            # Remove quotes and strip whitespace
            extracted_answer = extracted_answer.strip('"\'').strip()

            # STRICT pattern: must be EXACTLY a single digit with nothing else
            single_digit_pattern = r'^\d+$'

            if isinstance(gold_answer, list):
                gold_answer = gold_answer[0]

            if not re.match(single_digit_pattern, extracted_answer):
                return 0.0

            gold_num = str(gold_answer).strip('"\'').strip()

            return 1.0 if extracted_answer == gold_num else 0.0

        elif data_source == 'squad' or data_source == 'hotpotqa':
            # Default: containment score for other data sources
            if isinstance(gold_answer, list):
                answer_text = str(gold_answer[0]) if gold_answer else ""
            else:
                answer_text = gold_answer.get('text', gold_answer) if isinstance(gold_answer, dict) else str(gold_answer)

            return 1.0 if answer_text.lower() in predicted_answer.lower() else 0.0

        elif data_source == 'perltqa':

            if ";" in gold_answer:
                gold_answer = gold_answer.split(";")
                total_hit = 0
                for answer in gold_answer:
                    if answer.lower().strip() in predicted_answer:
                        total_hit += 1
                return total_hit / len(gold_answer)

            else:
                return 1.0 if gold_answer.lower() in predicted_answer.lower() else 0.0

        elif "seamless" in data_source:
            template = (
                "You are an evaluation judge. Given a question, a reference answer, and a model's response, "
                "classify the model's response into exactly one of three categories:\n\n"
                "1. **correct**: The model's response contains the key information from the reference answer. "
                "Paraphrasing, elaboration, or additional details are acceptable as long as the core answer is correct.\n"
                "2. **not_sure**: The model does not give a specific answer, hedges, express it is not sure, "
                " states that the information is not available in its memory/context, or require more information to answer the question.\n"
                "3. **wrong**: The model gives a specific, concrete answer, but it is factually incorrect "
                "compared to the reference answer.\n\n"
                "Question: {question}\n\n"
                "Reference Answer: {gold_answer}\n\n"
                "Model Response: {predicted_answer}\n\n"
                "Respond with ONLY one word: correct, not_sure, or wrong."
            )
            prompt = template.format(
                question=question,
                gold_answer=gold_answer,
                predicted_answer=predicted_answer
            )

            response = self.client.chat.completions.create(
                model=self.qwen_model,
                messages=[{"role": "user", "content": prompt}],
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            )

            judgment = response.choices[0].message.content.strip().lower()
            if "correct" in judgment and "not_sure" not in judgment and "wrong" not in judgment:
                return 1.0
            elif "not_sure" in judgment:
                return 0.5
            else:
                return 0.0

        elif data_source == 'lme_train' or data_source == 'longmemeval_s*':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, gold_answer, predicted_answer)

            response = self.client.chat.completions.create(
                model=self.qwen_model,
                messages=[{"role": "user", "content": prompt}],
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            )

            if "yes" in response.choices[0].message.content.strip().lower() and "no" not in response.choices[0].message.content.strip().lower():
                return 1.0
            else:
                return 0.0

        else:
            # memory agent bench
            return evaluate_wrt_source({'output': predicted_answer}, gold_answer, data_source)

    def evaluate_agent_results(
        self,
        agent_dir: str,
        parquet_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate results for a single agent directory.

        Args:
            agent_dir: Path to agent results directory containing
                       data_instance_info.json and results.json.
            parquet_data: Optional pre-loaded parquet DataFrame used to compute
                          compression ratio.  When *None* the compression metrics
                          are omitted from the returned dict.

        Returns:
            Dictionary containing evaluation metrics.
        """
        # Load data source info
        data_instance_info_path = os.path.join(agent_dir, "data_instance_info.json")
        with open(data_instance_info_path, 'r') as f:
            data_instance_info = json.load(f)
        data_source = data_instance_info.get('data_source')

        if not data_source:
            raise ValueError(f"No data_source found in {data_instance_info_path}")

        # Load results
        results_path = os.path.join(agent_dir, "results.json")
        with open(results_path, 'r') as f:
            results = json.load(f)

        # Load memory state and count memory tokens
        memory_state_path = os.path.join(agent_dir, "agent_state.json")
        total_memory_length = 0
        memory_state = None
        if os.path.exists(memory_state_path):
            try:
                with open(memory_state_path, 'r') as f:
                    memory_state = json.load(f)
                total_memory_length = count_memory_tokens(memory_state, self.tokenizer)
            except Exception as e:
                logger.warning(f"Could not load memory state from {memory_state_path}: {e}")

        # Optionally compute compression ratio from parquet data
        input_tokens = None
        compression_ratio = None
        if parquet_data is not None and memory_state is not None:
            global_idx = data_instance_info.get('global_idx')
            if global_idx is not None and global_idx < len(parquet_data):
                row = parquet_data.iloc[global_idx]
                chunks = json.loads(row['chunks']) if isinstance(row['chunks'], str) else row['chunks']
                input_tokens = count_input_tokens(chunks, self.tokenizer)
                compression_ratio = (
                    input_tokens / total_memory_length if total_memory_length > 0 else float('inf')
                )
            else:
                logger.warning(f"global_idx {global_idx} out of range for {agent_dir}")

        if not results:
            logger.warning(f"No results found in {results_path}")
            return {
                "data_source": data_source,
                "error": "No results found"
            }

        # Group results by instance and evaluate
        instance_scores = {}

        for result in results:
            # Get instance identifier (use index if no explicit instance_id)
            instance_id = result.get('instance_id', result.get('question_id', len(instance_scores)))

            score = self._compute_score(
                data_source=data_source,
                predicted_answer=result['response'],
                gold_answer=result['answer'],
                question=result['question']
            )

            if isinstance(score, bool):
                score = 1.0 if score else 0.0

            result['score'] = score

            if "seamless" in data_source:
                if score == 1.0:
                    result['judgment'] = 'correct'
                elif score == 0.5:
                    result['judgment'] = 'not_sure'
                else:
                    result['judgment'] = 'wrong'
            print(result['question'], result['judgment'] )
            # Group scores by instance
            if instance_id not in instance_scores:
                instance_scores[instance_id] = []
            instance_scores[instance_id].append(score)

        # Calculate average score for each instance
        instance_avg_scores = []
        for instance_id, scores_list in instance_scores.items():
            instance_avg = np.mean(scores_list)
            instance_avg_scores.append(instance_avg)

        # Overall scores for compatibility with existing code
        scores = instance_avg_scores

        # Save updated results with scores
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Calculate metrics
        metrics = {
            "data_source": data_source,
            "num_instances": len(instance_scores),
            "num_total_results": len(results),
            "avg_score_per_instance": np.mean(scores) if scores else 0.0,
            "min_instance_score": np.min(scores) if scores else 0.0,
            "max_instance_score": np.max(scores) if scores else 0.0,
            "std_instance_score": np.std(scores) if scores else 0.0,
            "total_memory_length": total_memory_length,
            # Compression metrics (present only when --dataset is supplied)
            "input_tokens": input_tokens,
            "compression_ratio": compression_ratio,
            # Keep backward compatibility
            "num_questions": len(instance_scores),
            "avg_score": np.mean(scores) if scores else 0.0,
            "min_score": np.min(scores) if scores else 0.0,
            "max_score": np.max(scores) if scores else 0.0,
            "std_score": np.std(scores) if scores else 0.0
        }

        if "seamless" in data_source:
            judgments = [r.get('judgment', '') for r in results]
            total = len(judgments)
            metrics["num_correct"] = judgments.count('correct')
            metrics["num_not_sure"] = judgments.count('not_sure')
            metrics["num_wrong"] = judgments.count('wrong')
            metrics["pct_correct"] = metrics["num_correct"] / total if total else 0.0
            metrics["pct_not_sure"] = metrics["num_not_sure"] / total if total else 0.0
            metrics["pct_wrong"] = metrics["num_wrong"] / total if total else 0.0

        return metrics

def evaluate_all_agents(
    base_dir: str,
    dataset: Optional[str] = None,
    tokenizer_name: str = "Qwen/Qwen3-32B",
) -> List[Dict[str, Any]]:
    """
    Evaluate results for all agent directories under the base directory.

    Args:
        base_dir: Base directory containing agent result directories.
        dataset: Optional dataset name used during memory construction.
                 When provided, compression ratio is computed for each agent.
        tokenizer_name: HuggingFace tokenizer for token counting.

    Returns:
        List of evaluation metrics for each agent.
    """
    evaluator = AgentResultsEvaluator(tokenizer_name=tokenizer_name)
    all_metrics = []

    parquet_data: Optional[pd.DataFrame] = None
    if dataset is not None:
        print(f"Loading parquet data for dataset '{dataset}' ...")
        parquet_data = load_parquet_data(dataset)

    for root, dirs, files in os.walk(base_dir):
        if 'data_instance_info.json' in files and 'results.json' in files:
            metrics = evaluator.evaluate_agent_results(root, parquet_data=parquet_data)
            metrics['agent_dir'] = root
            all_metrics.append(metrics)
    return all_metrics

def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate agent results (QA + optional compression ratio)')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory containing agent results')
    parser.add_argument('--output', type=str, default='evaluation_metrics.json',
                        help='Output file to save metrics (written inside base_dir)')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=[
                            'memalpha', 'memalpha_train', 'memalpha_sample',
                            'memoryagentbench', 'accurate_retrieval',
                            'test_time_learning', 'long_range_understanding',
                            'booksum', 'perltqa', 'pubmed-rct', 'squad',
                            'seamlessinteraction_gt', 'seamlessinteraction_pred',
                        ],
                        help='Dataset used during memory construction (enables compression ratio)')
    parser.add_argument('--tokenizer', type=str, default='Qwen/Qwen3-32B',
                        help='HuggingFace tokenizer for token counting')

    args = parser.parse_args()

    all_metrics = evaluate_all_agents(
        args.base_dir,
        dataset=args.dataset,
        tokenizer_name=args.tokenizer,
    )

    output_path = os.path.join(args.base_dir, args.output)
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Group metrics by data source
    grouped_metrics: Dict[str, list] = {}
    for metrics in all_metrics:
        src = metrics['data_source']
        grouped_metrics.setdefault(src, []).append(metrics)

    compute_compression = args.dataset is not None

    # Header
    col_w = 100 if compute_compression else 70
    print("\nEvaluation Summary by Data Source:")
    print("=" * col_w)

    for data_source, metrics_list in grouped_metrics.items():
        print(f"\nData Source: {data_source}")
        print("-" * 40)

        total_instances = sum(m['num_instances'] for m in metrics_list)
        avg_scores = [m['avg_score_per_instance'] for m in metrics_list]
        overall_avg = np.mean(avg_scores) if avg_scores else 0.0
        overall_min = min(m['min_instance_score'] for m in metrics_list)
        overall_max = max(m['max_instance_score'] for m in metrics_list)
        overall_std = np.std(avg_scores) if len(avg_scores) > 1 else 0.0

        memory_lengths = [m.get('total_memory_length', 0) for m in metrics_list]
        avg_memory_length = np.mean(memory_lengths) if memory_lengths else 0.0

        print(f"  Total agents:                     {len(metrics_list)}")
        print(f"  Total instances:                  {total_instances}")
        print(f"  Avg score per instance:           {overall_avg:.3f}")
        print(f"  Min/Max instance score:           {overall_min:.3f} / {overall_max:.3f}")
        print(f"  Std across agents:                {overall_std:.3f}")
        print(f"  Avg memory tokens:                {avg_memory_length:.0f}")

        if compute_compression:
            ratios = [
                m['compression_ratio'] for m in metrics_list
                if m.get('compression_ratio') not in (None, float('inf'))
            ]
            input_tkns = [m['input_tokens'] for m in metrics_list if m.get('input_tokens') is not None]
            avg_input = np.mean(input_tkns) if input_tkns else float('nan')
            avg_ratio = np.mean(ratios) if ratios else float('nan')
            std_ratio = np.std(ratios) if len(ratios) > 1 else 0.0
            print(f"  Avg input tokens:                 {avg_input:.0f}")
            print(f"  Avg compression ratio:            {avg_ratio:.2f}x  (std {std_ratio:.2f})")

        if "seamless" in data_source:
            total_correct = sum(m.get('num_correct', 0) for m in metrics_list)
            total_not_sure = sum(m.get('num_not_sure', 0) for m in metrics_list)
            total_wrong = sum(m.get('num_wrong', 0) for m in metrics_list)
            total_q = total_correct + total_not_sure + total_wrong
            if total_q:
                print(f"  Judgment breakdown ({total_q} questions):")
                print(f"    correct:  {total_correct:>5} ({total_correct/total_q*100:.1f}%)")
                print(f"    not_sure: {total_not_sure:>5} ({total_not_sure/total_q*100:.1f}%)")
                print(f"    wrong:    {total_wrong:>5} ({total_wrong/total_q*100:.1f}%)")

    print("\nMetrics saved to:", output_path)

if __name__ == "__main__":
    main()
