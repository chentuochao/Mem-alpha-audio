
# step1
# python run_memory_construction.py --agent_config config/memalpha-qwen3-4b_agent_0.05-0.1.yaml --dataset seamlessinteraction_gt --batch_size 1

# step2
# python run_qa_evaluation.py --agent_config config/memalpha-qwen3-4b_agent_0.05-0.1.yaml --dataset seamlessinteraction_gt --batch_size 1 --force_reanswer_questions


# step3 evaluate the QA accuracy
QWEN_URL="http://localhost:8002/v1" python evaluate_agent_results.py --base_dir ./agents/minimal_memory_agent_qwen_converted_YuWangX_Memalpha-4B_seamlessinteraction_gt_ext_qwen3-32b_no_thinking_tokens_2048/ --dataset seamlessinteraction_gt


# step4 evaluate the compression ratio
# python evaluate_compression_ratio.py --base_dir ./agents/minimal_memory_agent_qwen_converted_YuWangX_Memalpha-4B_seamlessinteraction_gt_ext_qwen3-32b_no_thinking_tokens_2048/ --dataset seamlessinteraction_gt
