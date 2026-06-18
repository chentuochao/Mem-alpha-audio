"""
test_load_qwen.py

Minimal standalone smoke test for loading a (large) Qwen model in vLLM and
running a trivial prompt. NO Mem-alpha code is imported (no agent, memory,
conversation_creator, qwen_agent), so this isolates whether vLLM itself can
load + generate with the model, independent of the construction pipeline.

Examples:
    # single GPU, eager (fastest startup, recommended first test)
    CUDA_VISIBLE_DEVICES=0 python test_load_qwen.py --model Qwen/Qwen3.6-27B

    # two GPUs
    CUDA_VISIBLE_DEVICES=0,1 python test_load_qwen.py \
        --model Qwen/Qwen3.6-27B --tensor-parallel-size 2

    # let torch.compile run (to reproduce the hang with vs. without eager)
    python test_load_qwen.py --model Qwen/Qwen3.6-27B --no-enforce-eager
"""

import os

# Match the cache dirs used by the main scripts so weights are reused.
os.environ.setdefault('HF_HOME', '/checkpoint/seamless/tuochao/Models/huggingface/')
os.environ.setdefault('HF_HUB_CACHE', '/checkpoint/seamless/tuochao/Models/huggingface/')

import argparse
import time


def parse_args():
    p = argparse.ArgumentParser(description="vLLM load + generate smoke test")
    p.add_argument("--model", type=str, default="Qwen/Qwen3.6-27B",
                   help="HF repo id or local path of the model")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=None,
                   help="Cap context length (helps avoid KV-cache OOM)")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--enforce-eager", dest="enforce_eager", action="store_true",
                   default=True, help="Skip torch.compile/CUDA-graph (default)")
    p.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false",
                   help="Allow torch.compile (reproduces the slow/hang path)")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--prompt", type=str,
                   default="Hello! In one sentence, what is a large language model?")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print(f"model                  : {args.model}")
    print(f"tensor_parallel_size   : {args.tensor_parallel_size}")
    print(f"gpu_memory_utilization : {args.gpu_memory_utilization}")
    print(f"max_model_len          : {args.max_model_len}")
    print(f"dtype                  : {args.dtype}")
    print(f"enforce_eager          : {args.enforce_eager}")
    print(f"CUDA_VISIBLE_DEVICES   : {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    print("=" * 70)

    # Import here so the prints above show even if vLLM import is slow.
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print("[1/4] loading tokenizer ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"      tokenizer loaded in {time.time() - t0:.1f}s")

    llm_kwargs = dict(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    print("[2/4] constructing vLLM engine (this is where loading happens) ...")
    t0 = time.time()
    llm = LLM(**llm_kwargs)
    print(f"      engine ready in {time.time() - t0:.1f}s")

    # Build a proper chat-formatted prompt. enable_thinking=False to match the
    # construction configs; drop the kwarg if this model's template rejects it.
    print("[3/4] building prompt ...")
    messages = [{"role": "user", "content": args.prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Some newer templates don't accept enable_thinking.
        print("      (template rejected enable_thinking; retrying without it)")
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    print("[4/4] generating ...")
    t0 = time.time()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    outputs = llm.generate([text], sampling_params)
    dt = time.time() - t0

    print("\n" + "=" * 70)
    print("PROMPT:")
    print(args.prompt)
    print("-" * 70)
    print("OUTPUT:")
    print(outputs[0].outputs[0].text.strip())
    print("=" * 70)
    print(f"generation took {dt:.1f}s")
    print("SUCCESS: model loaded and generated.")


if __name__ == "__main__":
    main()
