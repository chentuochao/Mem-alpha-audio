"""
agent_prompting.py

A thin extension of `MemoryAgent` (from agent.py) for running memory
construction with larger Qwen3-family models (e.g. Qwen3.6-27B dense,
Qwen3.6-35B-A3B MoE) on multi-GPU nodes.

The ONLY behavioral difference vs. the base MemoryAgent is how the vLLM
engine is constructed: the base class hardcodes

    self.model = LLM(model=qwen_model, dtype="bfloat16")

which cannot shard a 27B/35B model across GPUs. Here we inject extra vLLM
init kwargs (tensor_parallel_size, gpu_memory_utilization, max_model_len,
dtype) read from the agent_config, WITHOUT modifying agent.py.

All Qwen-specific prompting, function-call formatting, thinking-budget
logic and response parsing are inherited unchanged from MemoryAgent, so
this only works for Qwen3-family models (same as the base class).

Usage (drop-in replacement for MemoryAgent):

    from agent_prompting import MemoryAgentPrompting as MemoryAgent
    agent = MemoryAgent(agent_config=agent_config)

Config keys consumed here (all optional, with safe defaults):
    tensor_parallel_size:   int   (default 1)
    gpu_memory_utilization: float (default 0.9)
    max_model_len:          int   (default None -> vLLM infers)
    dtype:                  str   (default "bfloat16")
"""

import contextlib

import agent as _agent_module
from agent import MemoryAgent


class MemoryAgentPrompting(MemoryAgent):
    """MemoryAgent variant with configurable vLLM engine construction."""

    def __init__(self, agent_config: dict = None, *args, **kwargs) -> None:
        agent_config = agent_config or {}

        # Collect optional vLLM overrides from the config. These get merged
        # into the LLM(...) call that the base __init__ makes.
        vllm_overrides = {
            "dtype": agent_config.get("dtype", "bfloat16"),
            "tensor_parallel_size": agent_config.get("tensor_parallel_size", 1),
            "gpu_memory_utilization": agent_config.get("gpu_memory_utilization", 0.9),
        }
        # Only pass max_model_len when explicitly set; None lets vLLM infer it.
        if agent_config.get("max_model_len") is not None:
            vllm_overrides["max_model_len"] = agent_config["max_model_len"]

        # enforce_eager=True skips torch.compile + CUDA-graph capture. For
        # hybrid Mamba/attention MoE models (Qwen3-Next style) that capture
        # phase is extremely slow and can appear to hang on multi-GPU TP, so
        # this is the recommended escape hatch. Slower per-token inference but
        # reliable, fast startup. Omit / set false once startup is confirmed.
        if agent_config.get("enforce_eager") is not None:
            vllm_overrides["enforce_eager"] = bool(agent_config["enforce_eager"])

        # Generic passthrough for any other vLLM LLM(...) kwargs from config.
        extra = agent_config.get("vllm_kwargs") or {}
        if isinstance(extra, dict):
            vllm_overrides.update(extra)

        # Run the base __init__ with the module-level LLM symbol temporarily
        # wrapped so our overrides are applied to the engine it builds. This
        # keeps agent.py untouched while still inheriting any other __init__
        # behavior verbatim.
        with self._patched_vllm(vllm_overrides):
            super().__init__(agent_config=agent_config, *args, **kwargs)

    @staticmethod
    @contextlib.contextmanager
    def _patched_vllm(overrides: dict):
        """Temporarily wrap agent.LLM to merge in extra init kwargs."""
        original_llm = getattr(_agent_module, "LLM", None)
        if original_llm is None:
            # vLLM not imported in agent.py (VLLM_AVAILABLE is False there).
            # Nothing to patch; let the base class raise its own error.
            yield
            return

        def wrapped_llm(*a, **kw):
            # Caller-provided kwargs win only where we don't override; our
            # overrides take precedence (e.g. dtype, tensor_parallel_size).
            merged = {**kw, **overrides}
            return original_llm(*a, **merged)

        _agent_module.LLM = wrapped_llm
        try:
            yield
        finally:
            _agent_module.LLM = original_llm
