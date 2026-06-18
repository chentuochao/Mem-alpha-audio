"""
test_load_qwen.py

Minimal standalone smoke test for loading Qwen3.5/3.6 in vLLM as a TEXT-ONLY
LLM and running a trivial prompt. No Mem-alpha code is imported.

Everything is HARDCODED (no CLI args). The vision / multimodal tower is
disabled so vLLM does NOT run the max-image-size warmup that stalls the
Qwen3_5ForConditionalGeneration VL checkpoint:
  - limit_mm_per_prompt={"image": 0, "video": 0}  -> skip multimodal profiling
  - gdn_prefill_backend="triton"                  -> skip FlashInfer GDN JIT

Edit the CONFIG block below if you need to change the model path / GPU count.

Run:
    CUDA_VISIBLE_DEVICES=0 python test_load_qwen.py
"""

import os

# Match the cache dirs used by the main scripts so weights are reused.
os.environ.setdefault('HF_HOME', '/checkpoint/seamless/tuochao/Models/huggingface/')
os.environ.setdefault('HF_HUB_CACHE', '/checkpoint/seamless/tuochao/Models/huggingface/')

import time

# ===========================================================================
# CONFIG (hardcoded)
# ===========================================================================
MODEL = "Qwen/Qwen3.6-27B"          # HF repo id or local path
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.9
MAX_MODEL_LEN = None                  # set an int (e.g. 8192) to cap KV cache
DTYPE = "bfloat16"
MAX_TOKENS = 64
PROMPT = "Hello! In one sentence, what is a large language model?"
# ===========================================================================


def main():
    print("=" * 70)
    print(f"model                  : {MODEL}")
    print(f"tensor_parallel_size   : {TENSOR_PARALLEL_SIZE}")
    print(f"CUDA_VISIBLE_DEVICES   : {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    print("mode                   : TEXT-ONLY (vision/multimodal disabled)")
    print("=" * 70)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print("[1/4] loading tokenizer ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    print(f"      tokenizer loaded in {time.time() - t0:.1f}s")

    llm_kwargs = dict(
        model=MODEL,
        dtype=DTYPE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=True,                       # skip torch.compile / CUDA graphs
        # --- disable vision / multimodal entirely (text-only) ---
        limit_mm_per_prompt={"image": 0, "video": 0},
        gdn_prefill_backend="triton",             # skip FlashInfer GDN JIT
    )
    if MAX_MODEL_LEN is not None:
        llm_kwargs["max_model_len"] = MAX_MODEL_LEN

    print("[2/4] constructing vLLM engine (this is where loading happens) ...")
    t0 = time.time()
    llm = LLM(**llm_kwargs)
    print(f"      engine ready in {time.time() - t0:.1f}s")

    print("[3/4] building prompt ...")
    messages = [{"role": "user", "content": PROMPT}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        print("      (template rejected enable_thinking; retrying without it)")
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    print("[4/4] generating ...")
    t0 = time.time()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    outputs = llm.generate([text], sampling_params)
    dt = time.time() - t0

    print("\n" + "=" * 70)
    print("PROMPT:")
    print(PROMPT)
    print("-" * 70)
    print("OUTPUT:")
    print(outputs[0].outputs[0].text.strip())
    print("=" * 70)
    print(f"generation took {dt:.1f}s")
    print("SUCCESS: model loaded and generated (text-only).")


if __name__ == "__main__":
    main()
