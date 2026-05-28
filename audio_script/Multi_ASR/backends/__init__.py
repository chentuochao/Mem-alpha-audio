"""Pluggable inference backends for Step 1 multi-talker diarization + ASR.

Use :py:func:`build_backend` to instantiate a backend from an
``argparse.Namespace``, or import the backend class directly for programmatic
use.  Adding a new backend means:

  1. Subclass :py:class:`BaseBackend` in a new module.
  2. Add it to ``METHODS`` and :py:func:`build_backend` below.
  3. Optionally register any backend-specific CLI flags via
     :py:func:`add_backend_args`.
"""

from __future__ import annotations

import argparse

import torch

from .base import BaseBackend
from .nemo_offline import NemoOfflineBackend
from .nemo_streaming import NemoStreamingBackend
from .vibevoice import VibeVoiceBackend


METHODS = ("nemo-streaming", "nemo-offline", "vibevoice")


def _resolve_attn_implementation(requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    if device == "cuda" and torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except ImportError:
            return "sdpa"
    return "sdpa"


def build_backend(args: argparse.Namespace) -> BaseBackend:
    """Instantiate the backend selected by ``args.method``."""
    if args.method == "nemo-streaming":
        if not args.diar_model_path or not args.asr_model_path:
            raise SystemExit(
                "nemo-streaming requires --diar_model_path and --asr_model_path"
            )
        return NemoStreamingBackend(
            diar_model_path=args.diar_model_path,
            asr_model_path=args.asr_model_path,
            max_num_of_spks=args.max_num_of_spks,
            device=args.device,
        )

    if args.method == "nemo-offline":
        if not args.diar_model_path or not args.asr_model_path:
            raise SystemExit(
                "nemo-offline requires --diar_model_path and --asr_model_path"
            )
        return NemoOfflineBackend(
            diar_model_path=args.diar_model_path,
            asr_model_path=args.asr_model_path,
            max_num_of_spks=args.max_num_of_spks,
            device=args.device,
        )

    if args.method == "vibevoice":
        if not args.model_path:
            raise SystemExit("vibevoice requires --model_path")
        dtype = (
            torch.float32 if args.device in ("mps", "xpu", "cpu") else torch.bfloat16
        )
        attn_impl = _resolve_attn_implementation(
            args.attn_implementation, args.device
        )
        return VibeVoiceBackend(
            model_path=args.model_path,
            device=args.device,
            dtype=dtype,
            attn_implementation=attn_impl,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
        )

    raise SystemExit(f"Unknown --method {args.method!r} (choose from {METHODS})")


def add_backend_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register ``--method`` plus all backend-specific CLI flags on ``parser``."""
    parser.add_argument(
        "--method",
        required=True,
        choices=list(METHODS),
        help="Inference backend to use.",
    )

    nemo = parser.add_argument_group("NeMo backend options")
    nemo.add_argument("--diar_model_path", type=str, default=None,
                      help="Path to NeMo diarization model (.nemo).")
    nemo.add_argument("--asr_model_path", type=str, default=None,
                      help="Path to NeMo ASR model (.nemo).")
    nemo.add_argument("--max_num_of_spks", type=int, default=6,
                      help="Max number of speakers (default: 6; offline often uses 4).")

    vv = parser.add_argument_group("VibeVoice backend options")
    vv.add_argument("--model_path", type=str, default=None,
                    help="Path to the VibeVoice ASR model checkpoint.")
    vv.add_argument("--max_new_tokens", type=int, default=32768)
    vv.add_argument("--temperature", type=float, default=0.0)
    vv.add_argument("--top_p", type=float, default=1.0)
    vv.add_argument("--num_beams", type=int, default=1)
    vv.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["flash_attention_2", "sdpa", "eager", "auto"],
    )

    return parser


__all__ = [
    "BaseBackend",
    "NemoStreamingBackend",
    "NemoOfflineBackend",
    "VibeVoiceBackend",
    "METHODS",
    "build_backend",
    "add_backend_args",
]
