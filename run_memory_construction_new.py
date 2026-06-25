import os

os.environ['HF_HOME'] = '/checkpoint/seamless/tuochao/Models/huggingface/'
os.environ['HF_HUB_CACHE'] = '/checkpoint/seamless/tuochao/Models/huggingface/'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from datetime import datetime
import json
import vllm
import yaml
import argparse
import numpy as np
import time
from conversation_creator import ConversationCreator
# Use the prompting variant so larger Qwen3 models (27B / 35B-A3B) can be
# sharded across multiple GPUs via vLLM init knobs in the agent_config.
# Aliased to MemoryAgent so the rest of this script is unchanged.
from agent_prompting import MemoryAgentPrompting as MemoryAgent
from memory import Memory
from functions import get_memory_tool_schemas
from conversation_creator import get_out_dir
import torch
import random

SEED_NUM = 0
def init_random_seed():
    torch.manual_seed(SEED_NUM)
    np.random.seed(SEED_NUM)
    random.seed(SEED_NUM)
    torch.cuda.manual_seed(SEED_NUM)
    torch.cuda.manual_seed_all(SEED_NUM)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def load_agent_config(config_path):
    """Load agent configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Validate required fields
    required_fields = ['agent_name', 'model_name']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in config file: {config_path}")

    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Memory Construction Step")
    parser.add_argument("--agent_config", type=str, required=True, help="Path to agent configuration YAML file")
    parser.add_argument("--dataset", type=str, default="LOCOMO", choices=['squad', 'seamlessinteraction', 'seamlessinteraction_options', 'squad_test', 'hotpotqa', 'booksum', 'friends', 'wos46985', 'pubmed-rct', 'arxiv-classification', 'eurlex', 'accurate_retrieval', 'long_range_understanding', 'conflict_resolution', 'test_time_learning', "LOCOMO", "LongMemEval", "MemAgent_Bench", "memalpha", "memalpha_train", 'memalpha_sample', "detectiveqa", 'memoryagentbench', 'perltqa', 'narrativeqa', 'accurate_retrieval', 'test_time_learning', 'cr_train']) # Restricted choices
    parser.add_argument("--parquet_path", type=str, default=None, help="Path to parquet file")
    parser.add_argument("--load_db_from", type=str, default=None) # Memory databse
    parser.add_argument("--chunk_size", type=int, default=4096, help="Chunk size for MemAgent_Bench dataset")  # add parameter chunk_size
    parser.add_argument("--save_process", action="store_true", help="Enable process tracking for Qwen models (saves detailed logs)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for batch processing")
    parser.add_argument("--rollout_label", type=str, default=None, help="Label to append to output directory path, e.g., rollout_1")
    parser.add_argument(
        "--compression_strategy",
        type=str,
        default="default",
        help="Compression strategy controlling memory verbosity + per-chunk token budget. "
             "Must match a key under 'compression_strategies' in config/prompts_wrt_datasource_compression.yaml "
             "(e.g. default, x1.5, x2, x3, x5). Names are nominal targets, not exact ratios. "
             "The 'default' strategy is the legacy behavior and writes to the original folder without a '_comp_' postfix."
    )
    parser.add_argument(
        "--exclude_memory",
        nargs='*',
        default=[],
        help="Space or comma separated list of memory components to disable. Choose from: core, episodic, semantic."
    )

    args = parser.parse_args()
    allowed_memory_types = {"core", "semantic", "episodic"}
    normalized_exclusions = []
    for entry in args.exclude_memory:
        # Allow comma separated values in addition to whitespace separation
        parts = [part.strip().lower() for part in entry.split(",") if part.strip()]
        normalized_exclusions.extend(parts)

    invalid = sorted(set(normalized_exclusions) - allowed_memory_types)
    if invalid:
        parser.error(f"Invalid memory types for --exclude_memory: {', '.join(invalid)}. Allowed values: core, semantic, episodic.")

    args.exclude_memory = set(normalized_exclusions)
    return args


def count_memory_tokens(state: dict, tokenizer) -> int:
    """Count total tokens stored in a memory state (core + semantic + episodic)."""
    total = 0
    if state.get('core'):
        core = state['core']
        if isinstance(core, str):
            total += len(tokenizer(core).input_ids)
        elif isinstance(core, list):
            for item in core:
                if isinstance(item, str):
                    total += len(tokenizer(item).input_ids)
    for mem_type in ('semantic', 'episodic'):
        for item in state.get(mem_type) or []:
            total += len(tokenizer(list(item.values())[0]).input_ids)
    return total


def count_input_tokens(chunks: list, tokenizer) -> int:
    """Count total tokens across all raw input chunks."""
    return sum(len(tokenizer(c).input_ids) for c in chunks if isinstance(c, str))


def run_memory_construction_batch(args, agent_config, batch_indices, batch_chunks, batch_sources):
    """Process chunks and build memory for a batch of conversations."""

    # Use the compression-enabled prompt config (adds {compression_instruction}
    # placeholder + compression_strategies block on top of the original prompts).
    with open('config/prompts_wrt_datasource_compression.yaml', 'r') as f:
        prompts_wrt_datasource = yaml.safe_load(f)

    # Resolve the compression strategy (verbosity instruction + token-budget multiplier).
    compression_strategies = prompts_wrt_datasource.get('compression_strategies', {})
    if args.compression_strategy not in compression_strategies:
        raise ValueError(
            f"Unknown --compression_strategy '{args.compression_strategy}'. "
            f"Available: {', '.join(sorted(compression_strategies.keys()))}"
        )
    strategy = compression_strategies[args.compression_strategy]
    compression_instruction = strategy['instruction']
    budget_ratio = float(strategy.get('budget_ratio', 1.0))
    base_max_new_tokens = agent_config.get('max_new_tokens', 2048)
    # Hard per-chunk output cap for this strategy. Smaller cap -> smaller memory -> higher ratio.
    effective_max_new_tokens = max(64, int(base_max_new_tokens * budget_ratio))
    print(f"[DEBUG] Compression strategy '{args.compression_strategy}': "
          f"budget_ratio={budget_ratio}, max_new_tokens={effective_max_new_tokens} "
          f"(base={base_max_new_tokens})")

    batch_size = len(batch_chunks)

    # Get including_core parameter from agent_config, default to False
    # including_core = agent_config.get('including_core', False)
    batch_memories = [
        Memory(
            including_core=prompts_wrt_datasource[batch_sources[idx]]['including_core'],
            disabled_memory_types=args.exclude_memory
        )
        for idx in range(batch_size)
    ]
    memory_agent_template = MemoryAgent(agent_config=agent_config)

    # Check if agent_state.json exists for all batch items
    batch_out_dirs = [get_out_dir(agent_config, args, batch_indices[i]) for i in range(batch_size)]
    all_states_exist = True

    for i in range(batch_size):
        out_dir = batch_out_dirs[i]
        # Check if agent_state.json exists for this batch item
        if not os.path.exists(f"{out_dir}/agent_state.json"):
            all_states_exist = False
            break

    # Load existing states if all exist, otherwise process chunks
    if all_states_exist:
        print(f"[DEBUG] All agent states already exist, skipping memory construction...")
        return

    print(f"[DEBUG] Not all agent states exist, proceeding with chunk processing...")

    max_chunks = max(len(chunk_list) for chunk_list in batch_chunks) if len(batch_chunks) > 0 else 0

    # Initialize function calls storage for each batch item
    batch_function_calls_log = [[] for _ in range(batch_size)]
    batch_final_responses = {i: [] for i in range(batch_size)}

    # print("max_chunks = ", max_chunks)
    for chunk_idx in range(max_chunks):
        # pass chunk by chunk
        print(f"[DEBUG] Processing chunk {chunk_idx + 1}/{max_chunks}")

        max_new_tokens = effective_max_new_tokens

        remaining_indices = []
        current_chunks = []
        for i, chunk_list in enumerate(batch_chunks):
            if chunk_idx < len(chunk_list):
                prompt_key = 'unified_prompt_multispeaker' if "seamlessinteraction" in batch_sources[i] else 'unified_prompt'
                # prompt_key = 'unified_prompt'
                current_chunks.append(prompts_wrt_datasource[prompt_key].format(
                    context=chunk_list[chunk_idx],
                    max_new_tokens=int(max_new_tokens * 0.8),
                    compression_instruction=compression_instruction,
                ))
                remaining_indices.append(i)
        if len(remaining_indices) == 0:
            break

        prompts = []
        for chunk, memory in zip(current_chunks, [batch_memories[i] for i in remaining_indices]):
            processed_text = MemoryAgent.process_text_with_qwen_pipeline(
                text=chunk,
                tokenizer=memory_agent_template.tokenizer,
                functions=[tool["function"] for tool in get_memory_tool_schemas(memory)],
                status='memorie',
                enable_thinking=agent_config['enable_thinking'],
                return_text=True,
                memory=memory
            )
            prompts.append(processed_text)
        assert agent_config['vllm']

        # Import SamplingParams from vLLM for batch processing
        from vllm import SamplingParams

        if agent_config['enable_thinking']:
            # First generation until thinking budget
            thinking_budget = agent_config.get('thinking_budget', 1024)
            max_new_tokens = effective_max_new_tokens

            thinking_sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=thinking_budget,
                stop_token_ids=[memory_agent_template.tokenizer.eos_token_id],
                seed=SEED_NUM,
            )

            outputs = memory_agent_template.model.generate(prompts, thinking_sampling_params)
            first_responses = [output.outputs[0].text for output in outputs]

            # Collect all texts that need second generation
            second_gen_indices = []
            second_gen_texts = []
            has_early_stopping = []  # Track which ones have early stopping text
            finished_indices = []

            early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"

            for i, (first_response, prompt) in enumerate(zip(first_responses, prompts)):
                # Check if the generation has already finished or thinking process is complete
                if (memory_agent_template.tokenizer.eos_token_id not in memory_agent_template.tokenizer(first_response).input_ids
                    and "</think>" not in first_response):
                    print(f"thinking budget is reached for prompt {i}")
                    # Add early stopping text and prepare for batch second generation
                    continued_text = prompt + first_response + early_stopping_text
                    second_gen_indices.append(i)
                    second_gen_texts.append(continued_text)
                    has_early_stopping.append(True)
                elif ("</think>" in first_response
                      and memory_agent_template.tokenizer.eos_token_id not in memory_agent_template.tokenizer(first_response).input_ids):
                    # Thinking completed, continue generation after thinking
                    continued_text = prompt + first_response
                    second_gen_indices.append(i)
                    second_gen_texts.append(continued_text)
                    has_early_stopping.append(False)
                else:
                    # Generation finished or no continuation needed
                    finished_indices.append(i)

            # Batch second generation for all texts that need it
            second_gen_responses = []
            if second_gen_texts:
                remaining_sampling_params = SamplingParams(
                    temperature=0.7,
                    max_tokens=max_new_tokens - thinking_budget,
                    stop_token_ids=[memory_agent_template.tokenizer.eos_token_id],
                    seed = SEED_NUM,
                )
                second_outputs = memory_agent_template.model.generate(second_gen_texts, remaining_sampling_params)
                second_gen_responses = [output.outputs[0].text.strip() for output in second_outputs]

            # Combine all responses in correct order
            final_responses = [None] * len(first_responses)

            # Fill in second generation responses
            for i, idx in enumerate(second_gen_indices):
                if has_early_stopping[i]:
                    # Budget reached case: include early stopping text
                    final_responses[idx] = first_responses[idx] + early_stopping_text + second_gen_responses[i]
                else:
                    # Thinking complete case: no early stopping text
                    final_responses[idx] = first_responses[idx] + second_gen_responses[i]

            # Fill in finished responses
            for idx in finished_indices:
                final_responses[idx] = first_responses[idx].strip()
        else:
            # Single generation without thinking budget
            max_new_tokens = effective_max_new_tokens
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=max_new_tokens,
                stop_token_ids=[memory_agent_template.tokenizer.eos_token_id],
                seed=SEED_NUM,
            )
            outputs = memory_agent_template.model.generate(prompts, sampling_params)
            final_responses = [output.outputs[0].text.strip() for output in outputs]

        # Batch process responses and execute function calls
        batch_assistant_messages = []
        batch_function_calls = []

        # Parse all responses in batch
        for i, (response, memory_idx) in enumerate(zip(final_responses, remaining_indices)):
            assistant_messages = memory_agent_template._parse_response(response)
            batch_assistant_messages.append((assistant_messages, memory_idx))
            batch_final_responses[memory_idx].append(response)

            # Collect function calls for batch execution
            function_calls_messages = [msg for msg in assistant_messages if msg.get("function_call")]
            if function_calls_messages:
                for assistant_msg in function_calls_messages:
                    batch_function_calls.append((assistant_msg["function_call"], memory_idx))

        # Execute function calls in batch and update memory
        if batch_function_calls:
            for function_call, memory_idx in batch_function_calls:
                tool_result = memory_agent_template._run_tool_from_function_call(
                    function_call,
                    batch_memories[memory_idx]
                )
                # Store the function call and result in the appropriate batch item log
                function_call_record = {
                    'function_call': function_call,
                    'tool_result': tool_result,
                    'chunk_idx': chunk_idx,
                    'timestamp': time.time()
                }
                batch_function_calls_log[memory_idx].append(function_call_record)

    # Save all memory states after chunk processing
    for i in range(batch_size):
        batch_idx = batch_indices[i]
        memory = batch_memories[i]
        out_dir = batch_out_dirs[i]

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Save memory state
        state = {
            'semantic': memory.semantic,
            'episodic': memory.episodic,
            'conversation_history': [],
            'step': max_chunks,
            'semantic_embedding_ids': memory.semantic_embedding_ids,
            'episodic_embedding_ids': memory.episodic_embedding_ids
        }

        # Only save core memory if it's available
        if memory.including_core and memory.core is not None:
            state['core'] = memory.core
        print("store memory ", f"{out_dir}/agent_state.json")
        with open(f"{out_dir}/agent_state.json", "w") as f:
            json.dump(state, f, indent=2)

        # Compute and save compression ratio
        input_tokens = count_input_tokens(batch_chunks[i], memory_agent_template.tokenizer)
        memory_tokens = count_memory_tokens(state, memory_agent_template.tokenizer)
        compression_ratio = input_tokens / memory_tokens if memory_tokens > 0 else float('inf')
        compression_info = {
            'input_tokens': input_tokens,
            'memory_tokens': memory_tokens,
            'compression_ratio': compression_ratio,
        }
        with open(f"{out_dir}/compression.json", "w") as f:
            json.dump(compression_info, f, indent=2)
        print(f"compression ratio: {compression_ratio:.2f}x  (input={input_tokens}, memory={memory_tokens})")

        # Save data instance info
        data_instance_info = {
            'data_source': batch_sources[i],
            'global_idx': batch_idx
        }
        with open(f"{out_dir}/data_instance_info.json", "w") as f:
            json.dump(data_instance_info, f, indent=2)

        # Save chunks with their corresponding function calls in a single file
        chunks_with_function_calls = []
        for chunk_idx, chunk in enumerate(batch_chunks[i]):
            # Get function calls for this specific chunk
            chunk_function_calls = [
                fc for fc in batch_function_calls_log[i]
                if fc.get('chunk_idx') == chunk_idx
            ]

            chunks_with_function_calls.append({
                'chunk_idx': chunk_idx,
                'raw_chunk': chunk,
                'function_calls': chunk_function_calls
            })

        with open(f"{out_dir}/chunks_and_function_calls.json", "w") as f:
            json.dump(chunks_with_function_calls, f, indent=2)

        with open(f"{out_dir}/final_responses.json", "w") as f:
            json.dump(batch_final_responses[i], f, indent=2)

        # Save embeddings if available
        if (memory.semantic_embedding_matrix.size > 0 or
            memory.episodic_embedding_matrix.size > 0):
            np.savez_compressed(f"{out_dir}/embeddings.npz",
                              semantic_matrix=memory.semantic_embedding_matrix,
                              episodic_matrix=memory.episodic_embedding_matrix)


def main():
    init_random_seed()
    args = parse_args()

    # Load agent configuration
    agent_config = load_agent_config(args.agent_config)

    # Print loaded configuration
    print(f"Loaded agent configuration:")
    print(f"  Agent name: {agent_config['agent_name']}")
    print(f"  Model name: {agent_config['model_name']}")
    if 'enable_thinking' in agent_config:
        print(f"  Enable thinking: {agent_config['enable_thinking']}")
    print(f"  Save process (Qwen only): {args.save_process}")
    if args.exclude_memory:
        print(f"  Disabled memories: {', '.join(sorted(args.exclude_memory))}")
    else:
        print(f"  Disabled memories: None")

    conversation_creator = ConversationCreator(args.dataset, args.chunk_size, parquet_path = args.parquet_path)

    all_chunks = conversation_creator.chunks() # TODO: Note we don't skip already completed conversations in chunking process of conversation_creator.py since it's easy to mess up the index sequence in eval.py, but we can fix it later (return empty chunks instead of skipping)

    # all_queries_and_answers = conversation_creator.get_query_and_answer()
    # Handle cases where some instances might have empty Q&A lists
    # QA list = item[0] - [q_idx, question, answer, data_source, category]
    # all_sources = []
    # for item in all_queries_and_answers:
    #     if len(item) > 0:
    #         all_sources.append(item[0][3])
    #     else:
    #         # Default source for empty Q&A lists based on dataset
    #         all_sources.append(args.dataset)

    # # just for debug
    # for item in all_queries_and_answers:
    #     if len(item) > 0:
    #         assert len(np.unique([x[3] for x in item])) == 1, "all sources should be the same"
    all_sources = conversation_creator.data['data_source'].tolist()
    print(f"Processing {len(all_chunks)} conversations for dataset {args.dataset}...")
    # Process all conversations using batch processing
    all_indices = list(range(len(all_chunks)))
    for i in range(0, len(all_indices), args.batch_size):
        batch_indices = all_indices[i:i+args.batch_size]
        batch_chunks = [all_chunks[idx] for idx in batch_indices]
        batch_sources = [all_sources[idx] for idx in batch_indices]
        run_memory_construction_batch(args, agent_config, batch_indices, batch_chunks, batch_sources)


if __name__ == '__main__':
    main()
