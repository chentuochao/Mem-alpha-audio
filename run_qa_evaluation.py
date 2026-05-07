import os

os.environ['HF_HOME'] = '/checkpoint/seamless/tuochao/Models/huggingface/'
os.environ['HF_HUB_CACHE'] = '/checkpoint/seamless/tuochao/Models/huggingface/'

import json
import yaml
import argparse
import numpy as np
import requests
from conversation_creator import ConversationCreator
from memory import Memory
from conversation_creator import get_out_dir

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


def get_results_filename(agentic_search=False):
    """Get the appropriate results filename based on search method."""
    return "agentic_results.json" if agentic_search else "results.json"


def load_custom_qa_from_dir(qa_dir, data_source='seamlessinteraction'):
    """Load custom QA pairs from all JSON/JSONL files in a directory.

    Each file is treated as newline-delimited JSON (one object per line).
    Returns a flat list of [q_idx, question, answer, data_source] tuples,
    or [q_idx, question, answer, data_source, category] for seamlessinteraction.
    """
    qa_items = []
    q_idx = 0
    for filename in sorted(os.listdir(qa_dir)):
        filepath = os.path.join(qa_dir, filename)
        if not os.path.isfile(filepath):
            continue
        if not (filename.endswith('.json') or filename.endswith('.jsonl')):
            continue
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                question = item.get('question', '')
                answer = item.get('answer', '')
                if question:
                    entry = [q_idx, question, answer, data_source]
                    if data_source in ['seamlessinteraction']:
                        entry.append(item.get('category', ''))
                    qa_items.append(entry)
                    q_idx += 1
    print(f"Loaded {len(qa_items)} custom QA items from {qa_dir}")
    return qa_items


def parse_args():
    parser = argparse.ArgumentParser(description="QA Evaluation Step")
    parser.add_argument("--agent_config", type=str, required=True, help="Path to agent configuration YAML file")
    parser.add_argument("--dataset", type=str, default="LOCOMO", choices=['squad', 'seamlessinteraction', 'squad_test', 'hotpotqa', 'booksum', 'friends', 'wos46985', 'pubmed-rct', 'arxiv-classification', 'eurlex', 'accurate_retrieval', 'long_range_understanding', 'conflict_resolution', 'test_time_learning', "LOCOMO", "LongMemEval", "MemAgent_Bench", "memalpha", "memalpha_train", 'memalpha_sample', "detectiveqa", 'memoryagentbench', 'perltqa', 'narrativeqa', 'accurate_retrieval', 'test_time_learning', 'cr_train']) # Restricted choices
    parser.add_argument("--parquet_path", type=str, default=None, help="Path to parquet file")
    parser.add_argument("--chunk_size", type=int, default=4096, help="Chunk size for MemAgent_Bench dataset")  # add parameter chunk_size
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for batch processing")
    parser.add_argument("--agentic_search", action="store_true", help="Use agentic memory search instead of simple batch processing")
    parser.add_argument("--rollout_label", type=str, default=None, help="Label to append to output directory path, e.g., rollout_1")
    parser.add_argument("--force_reanswer_questions", action="store_true", help="Force reanswering all questions even if results file already exists")
    parser.add_argument("--custom_qa_dir", type=str, default=None,
                        help="Directory containing custom QA JSON/JSONL files to use instead of dataset QA. "
                             "All items are loaded into the first (and only) conversation instance.")
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


def load_memory_state(memory, out_dir):
    """Load a saved memory state from disk into a Memory object."""
    state_file = f"{out_dir}/agent_state.json"
    if not os.path.exists(state_file):
        raise FileNotFoundError(
            f"Agent state not found: {state_file}. "
            "Run run_memory_construction.py first."
        )

    # Load agent state
    with open(state_file, "r") as f:
        state = json.load(f)

    # Restore memory state
    # Only restore core memory if it's available in the memory object
    if memory.including_core and memory.core is not None:
        memory.core = state.get('core', [])

    if memory.is_memory_type_enabled('semantic'):
        memory.semantic = state.get('semantic', [])
        memory.semantic_embedding_ids = state.get('semantic_embedding_ids', [])
    else:
        memory.semantic = []
        memory.semantic_embedding_ids = []

    if memory.is_memory_type_enabled('episodic'):
        memory.episodic = state.get('episodic', [])
        memory.episodic_embedding_ids = state.get('episodic_embedding_ids', [])
    else:
        memory.episodic = []
        memory.episodic_embedding_ids = []

    # Load embeddings if available
    embeddings_file = f"{out_dir}/embeddings.npz"
    if os.path.exists(embeddings_file):
        embeddings = np.load(embeddings_file)
        if memory.is_memory_type_enabled('semantic'):
            memory.semantic_embedding_matrix = embeddings['semantic_matrix']
        else:
            memory.semantic_embedding_matrix = np.array([])

        if memory.is_memory_type_enabled('episodic'):
            memory.episodic_embedding_matrix = embeddings['episodic_matrix']
        else:
            memory.episodic_embedding_matrix = np.array([])
    else:
        memory.semantic_embedding_matrix = np.array([])
        memory.episodic_embedding_matrix = np.array([])

    return state


def run_qa_evaluation_batch(
        args,
        agent_config,
        batch_indices,
        batch_chunks,
        batch_queries_and_answers,
        batch_sources):
    """Answer questions using the previously constructed memory."""

    with open('config/prompts_wrt_datasource.yaml', 'r') as f:
        prompts_wrt_datasource = yaml.safe_load(f)

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

    batch_out_dirs = [get_out_dir(agent_config, args, batch_indices[i]) for i in range(batch_size)]
    print(f"[DEBUG] Loading existing agent states for all batch items, skipping chunk processing...")
    for i in range(batch_size):
        load_memory_state(batch_memories[i], batch_out_dirs[i])

    max_chunks = max(len(chunk_list) for chunk_list in batch_chunks) if len(batch_chunks) > 0 else 0
    print(f"[DEBUG] Loaded existing states, proceeding directly to question answering...")

    # TODO: check if the results file exists for all batch items
    results_filename = get_results_filename(args.agentic_search)
    all_results_exist = True
    if not args.force_reanswer_questions:
        for i in range(batch_size):
            out_dir = batch_out_dirs[i]
            if not os.path.exists(f"{out_dir}/{results_filename}"):
                all_results_exist = False
                break
        if all_results_exist:
            all_results = []
            for i in range(batch_size):
                out_dir = batch_out_dirs[i]
                with open(f"{out_dir}/{results_filename}", "r") as f:
                    all_results.extend(json.load(f))
            return all_results
    else:
        all_results_exist = False

    # First, save all memory states in parallel
    all_results = []

    for i in range(batch_size):
        batch_idx = batch_indices[i]
        memory = batch_memories[i]
        out_dir = batch_out_dirs[i]

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Save memory state (in case it wasn't saved during chunk processing)
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

        with open(f"{out_dir}/agent_state.json", "w") as f:
            json.dump(state, f, indent=2)

        # Save data instance info
        data_instance_info = {
            'data_source': batch_sources[i],
            'global_idx': batch_idx
        }
        with open(f"{out_dir}/data_instance_info.json", "w") as f:
            json.dump(data_instance_info, f, indent=2)

        # Save embeddings if available
        if (memory.semantic_embedding_matrix.size > 0 or
            memory.episodic_embedding_matrix.size > 0):
            np.savez_compressed(f"{out_dir}/embeddings.npz",
                              semantic_matrix=memory.semantic_embedding_matrix,
                              episodic_matrix=memory.episodic_embedding_matrix)

    # Collect all questions for batch processing
    all_questions = []
    question_metadata = []  # Store metadata for each question
    for i in range(batch_size):
        batch_idx = batch_indices[i]
        queries_and_answers = batch_queries_and_answers[i]
        memory = batch_memories[i]

        for item in queries_and_answers:
            if args.dataset == "LOCOMO":
                question_idx, question, answer, category = item
                question_metadata.append({
                    'batch_idx': batch_idx,
                    'memory_idx': i,
                    'question': question,
                    'answer': answer,
                    'category': category,
                    'dataset_type': 'LOCOMO'
                })
            elif args.dataset == "MemAgent_Bench":
                question_idx, question, answer, category, source = item
                question_metadata.append({
                    'batch_idx': batch_idx,
                    'memory_idx': i,
                    'question': question,
                    'answer': answer,
                    'category': category,
                    'source': source,
                    'dataset_type': 'MemAgent_Bench'
                })
            else:
                question_idx, question, answer, data_source = item[:4]
                meta_entry = {
                    'batch_idx': batch_idx,
                    'memory_idx': i,
                    'question': question,
                    'answer': answer,
                    'data_source': data_source,
                    'dataset_type': args.dataset
                }
                if len(item) >= 5:
                    meta_entry['category'] = item[4]
                question_metadata.append(meta_entry)

            all_questions.append(question)

    # Process questions in batches using temporary agents or external model
    question_responses = []
    question_step_infos = []

    # Check if we should use external model for batch inference
    print("number of question_metadata", len(question_metadata))
    if agent_config.get('infer_with_full_memory', False) and agent_config.get('external_model_url'):

        # Prepare questions grouped by memory (batch item)
        questions_by_memory = {}
        for i, meta in enumerate(question_metadata):
            memory_idx = meta['memory_idx']
            question = meta['question']

            if meta.get("data_source", None):
                data_source = meta['data_source']
                query_prompt = prompts_wrt_datasource[data_source]['query_prompt']
                if query_prompt is not None:
                    question = query_prompt + "\n\n" + question

            # print(i, question)
            if memory_idx not in questions_by_memory:
                questions_by_memory[memory_idx] = []
            questions_by_memory[memory_idx].append({'question': question, 'metadata_idx': i})

        # Prepare payload for memory server
        batch_memories_for_server = []
        questions_for_server = []

        for memory_idx in sorted(questions_by_memory.keys()):
            memory = batch_memories[memory_idx]
            # Prepare memory dict for server
            memory_dict = {
                'episodic': memory.episodic,
                'semantic': memory.semantic
            }
            # Only include core if it exists
            if memory.including_core and memory.core is not None:
                memory_dict['core'] = memory.core

            batch_memories_for_server.append(memory_dict)
            questions_for_server.append([q['question'] for q in questions_by_memory[memory_idx]])

        # Make request to memory server
        payload = {
            "memories": batch_memories_for_server,
            'questions': questions_for_server
        }

        # Choose endpoint based on agentic_search flag
        base_url = agent_config['external_model_url']
        if base_url.endswith('/batch_process'):
            base_url = base_url[:-len('/batch_process')]

        if args.agentic_search:
            endpoint = f"{base_url}/agentic_process"
        else:
            endpoint = f"{base_url}/batch_process"

        response = requests.post(endpoint, json=payload)

        if response.status_code != 200:
            raise Exception(f"Memory server request failed with status {response.status_code}: {response.text}")

        server_response_json = response.json()
        server_results = server_response_json.get('result', [])
        server_retrieved_memories = server_response_json.get('retrieved_memories', [])

        # Process server results and map back to original question order
        question_responses = [None] * len(all_questions)
        question_step_infos = [None] * len(all_questions)

        result_idx = 0
        for memory_idx in sorted(questions_by_memory.keys()):
            memory_questions = questions_by_memory[memory_idx]
            memory_results = server_results[result_idx] if result_idx < len(server_results) else []
            memory_retrieved = server_retrieved_memories[result_idx] if result_idx < len(server_retrieved_memories) else []

            for i, q_info in enumerate(memory_questions):
                metadata_idx = q_info['metadata_idx']
                response_text = memory_results[i] if i < len(memory_results) else "No response from server"
                retrieved_mem = memory_retrieved[i] if i < len(memory_retrieved) else None

                question_responses[metadata_idx] = response_text
                step_info = {
                    "step": max_chunks,
                    "final_response": response_text,
                    "memory_server_used": True,
                    "batch_processed": True,
                    "agentic_search_used": args.agentic_search,
                    "retrieved_memory": retrieved_mem,
                }
                question_step_infos[metadata_idx] = step_info

            result_idx += 1

    elif agent_config.get('external_model_url'):
        # Handle case without infer_with_full_memory flag
        # This case would need similar implementation based on your requirements
        raise NotImplementedError("Memory server without infer_with_full_memory not yet implemented")

    else:
        raise NotImplementedError("Only memory server inference is supported for batch processing")

    # Group results by batch item and save
    batch_results_dict = {}
    for i, meta in enumerate(question_metadata):
        batch_idx = meta['batch_idx']
        if batch_idx not in batch_results_dict:
            batch_results_dict[batch_idx] = []

        # Format result based on dataset type
        step_info = question_step_infos[i] or {}
        retrieved_memory = step_info.get('retrieved_memory')
        if meta['dataset_type'] == 'LOCOMO':
            result = {
                'question': meta['question'],
                'response': question_responses[i],
                'answer': meta['answer'],
                'category': meta['category'],
                'retrieved_memory': retrieved_memory,
            }
        elif meta['dataset_type'] == 'MemAgent_Bench':
            result = {
                'question': meta['question'],
                'response': question_responses[i],
                'answer': meta['answer'],
                'category': meta['category'],
                'source': meta['source'],
                'retrieved_memory': retrieved_memory,
            }
        else:
            result = {
                'question': meta['question'],
                'response': question_responses[i],
                'answer': meta['answer'],
                'retrieved_memory': retrieved_memory,
            }
            if 'category' in meta:
                result['category'] = meta['category']

        batch_results_dict[batch_idx].append(result)

    # Save results for each batch item
    for i, batch_idx in enumerate(batch_indices):
        batch_results = batch_results_dict[batch_idx]
        out_dir = batch_out_dirs[i]

        results_filename = get_results_filename(args.agentic_search)
        with open(f"{out_dir}/{results_filename}", "w") as f:
            json.dump(batch_results, f, indent=2)

        all_results.extend(batch_results)

    return all_results


def main():

    args = parse_args()

    # Load agent configuration
    agent_config = load_agent_config(args.agent_config)

    # Print loaded configuration
    print(f"Loaded agent configuration:")
    print(f"  Agent name: {agent_config['agent_name']}")
    print(f"  Model name: {agent_config['model_name']}")
    if 'enable_thinking' in agent_config:
        print(f"  Enable thinking: {agent_config['enable_thinking']}")
    if args.exclude_memory:
        print(f"  Disabled memories: {', '.join(sorted(args.exclude_memory))}")
    else:
        print(f"  Disabled memories: None")

    conversation_creator = ConversationCreator(args.dataset, args.chunk_size, parquet_path = args.parquet_path)

    all_chunks = conversation_creator.chunks() # TODO: Note we don't skip already completed conversations in chunking process of conversation_creator.py since it's easy to mess up the index sequence in eval.py, but we can fix it later (return empty chunks instead of skipping)

    all_queries_and_answers = conversation_creator.get_query_and_answer()

    # Override QA with custom files if --custom_qa_dir is specified.
    # Assumes a single conversation instance (batch_size=1, chunk_count=1).
    if args.custom_qa_dir:
        custom_qa = load_custom_qa_from_dir(args.custom_qa_dir, data_source=args.dataset)
        all_queries_and_answers = [custom_qa] * len(all_chunks)

    # Handle cases where some instances might have empty Q&A lists
    all_sources = []
    for item in all_queries_and_answers:
        if len(item) > 0:
            all_sources.append(item[0][3])  # index 3 is always data_source
        else:
            # Default source for empty Q&A lists based on dataset
            all_sources.append(args.dataset)

    # just for debug
    for item in all_queries_and_answers:
        if len(item) > 0:
            assert len(np.unique([x[3] for x in item])) == 1, "all sources should be the same"

    print(f"Evaluating {len(all_chunks)} conversations for dataset {args.dataset}...")

    # Process all conversations using batch processing
    all_indices = list(range(len(all_chunks)))
    for i in range(0, len(all_indices), args.batch_size):
        batch_indices = all_indices[i:i+args.batch_size]
        batch_chunks = [all_chunks[idx] for idx in batch_indices]
        batch_sources = [all_sources[idx] for idx in batch_indices]
        batch_queries_and_answers = [all_queries_and_answers[idx] for idx in batch_indices]
        run_qa_evaluation_batch(args, agent_config, batch_indices, batch_chunks, batch_queries_and_answers, batch_sources)


if __name__ == '__main__':
    main()
