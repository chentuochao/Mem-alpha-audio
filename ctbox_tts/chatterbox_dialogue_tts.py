# chatterbox_dialogue_tts.py

import json
import re
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torchaudio as ta

# ============================================================
# Patch Perth watermarker before ChatterboxTTS initialization
# ============================================================
def _patch_perth_watermarker():
    """
    Some environments have perth.PerthImplicitWatermarker = None,
    which makes ChatterboxTTS.from_pretrained crash with:
        TypeError: 'NoneType' object is not callable

    We patch it to DummyWatermarker locally, without modifying site-packages.
    """
    try:
        import perth

        wm_cls = getattr(perth, "PerthImplicitWatermarker", None)
        dummy_cls = getattr(perth, "DummyWatermarker", None)

        if wm_cls is None or not callable(wm_cls):
            if dummy_cls is not None and callable(dummy_cls):
                perth.PerthImplicitWatermarker = dummy_cls
                print("[Patch] perth.PerthImplicitWatermarker -> perth.DummyWatermarker")
            else:
                raise RuntimeError(
                    "perth.PerthImplicitWatermarker is unavailable, "
                    "and perth.DummyWatermarker is also unavailable."
                )

    except ImportError:
        raise ImportError(
            "Failed to import perth. Please install chatterbox-tts dependencies correctly."
        )


_patch_perth_watermarker()

from chatterbox.tts import ChatterboxTTS

# ============================================================
# 1. 显式硬编码：Profile 人物 -> 参考语音
# ============================================================
# 这里必须换成你们真实的 reference audio 路径
REFERENCE_VOICE_MAP = {
    "Wang Xiaoming": "ref_voices/wang_xiaoming.wav",
    "Wang Xiaohong": "ref_voices/wang_xiaohong.wav",
    "Zhang Wei": "ref_voices/zhang_wei.wav",
    "Li Ting": "ref_voices/li_ting.wav",
    "Zhao Ming": "ref_voices/zhao_ming.wav",
    "Liu Li": "ref_voices/liu_li.wav",
    "Zhang Wen": "ref_voices/zhang_wen.wav",
    "Li Ming": "ref_voices/li_ming.wav",
    "Chen Hua": "ref_voices/chen_hua.wav",
    "Liu Qiang": "ref_voices/liu_qiang.wav",
}


# ============================================================
# 2. 全局模型：文件 import 时初始化一次
# ============================================================
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_TTS_MODEL = ChatterboxTTS.from_pretrained(device=_DEVICE)
_SAMPLE_RATE = int(_TTS_MODEL.sr)


# ============================================================
# 3. 工具函数
# ============================================================
def _safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("_") or "speaker"


def _normalize_text_list(text: Union[str, List[str]]) -> List[str]:
    """
    支持两种输入：
    1. str: 当成单轮文本；如果里面有换行，则按多轮切分
    2. list[str]: 每个元素是一轮
    """
    if isinstance(text, str):
        parts = [x.strip() for x in text.split("\n") if x.strip()]
        return parts

    return [x.strip() for x in text if x and x.strip()]


def _to_mono_np(wav) -> np.ndarray:
    """
    Chatterbox 输出可能是 torch.Tensor，也可能是 np.ndarray。
    这里统一转成 mono float32 numpy array。
    """
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().float().numpy()

    wav = np.asarray(wav, dtype=np.float32)
    wav = np.squeeze(wav)

    if wav.ndim == 2:
        # [C, T]
        if wav.shape[0] <= 2:
            wav = wav.mean(axis=0)
        # [T, C]
        else:
            wav = wav.mean(axis=1)

    if wav.ndim != 1:
        raise ValueError(f"Unexpected wav shape after TTS generation: {wav.shape}")

    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    return wav.astype(np.float32)


def _generate_tts(
    speaker_name: str,
    text: str,
    exaggeration: Optional[float] = None,
    cfg_weight: Optional[float] = None,
) -> np.ndarray:
    if speaker_name not in REFERENCE_VOICE_MAP:
        raise ValueError(
            f"Missing reference voice for speaker: {speaker_name}. "
            f"Please add it to REFERENCE_VOICE_MAP."
        )

    ref_audio_path = Path(REFERENCE_VOICE_MAP[speaker_name])

    if not ref_audio_path.exists():
        raise FileNotFoundError(
            f"Reference voice file not found for speaker {speaker_name}: "
            f"{ref_audio_path}"
        )

    kwargs = {
        "audio_prompt_path": str(ref_audio_path),
    }

    if exaggeration is not None:
        kwargs["exaggeration"] = exaggeration

    if cfg_weight is not None:
        kwargs["cfg_weight"] = cfg_weight

    wav = _TTS_MODEL.generate(text, **kwargs)
    return _to_mono_np(wav)


# ============================================================
# 4. 主函数：对方只需要调用这个
# ============================================================
def generate_dialogue_tts(
    speaker1_name: str,
    speaker2_name: str,
    speaker1_text: Union[str, List[str]],
    speaker2_text: Union[str, List[str]],
    output_dir: str = "tts_outputs",
    save_stereo_wav: bool = True,
    save_npy: bool = True,
    save_json: bool = True,
    exaggeration: Optional[float] = None,
    cfg_weight: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generate two-channel dialogue TTS using Chatterbox.

    Rules:
    - speaker1 starts first by default.
    - speaker1 and speaker2 speak alternately.
    - speaker1 occupies the left channel.
    - speaker2 occupies the right channel.
    - When one speaker is speaking, the other channel is zero-padded.
    - Final left/right np arrays have exactly the same length.
    - Save channel mapping to JSON.

    Returns:
        {
            "speaker1_tts": np.ndarray,  # left channel
            "speaker2_tts": np.ndarray,  # right channel
            "metadata": dict
        }
    """

    speaker1_turns = _normalize_text_list(speaker1_text)
    speaker2_turns = _normalize_text_list(speaker2_text)

    if len(speaker1_turns) == 0 and len(speaker2_turns) == 0:
        raise ValueError("speaker1_text and speaker2_text are both empty.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    turns = []
    max_turns = max(len(speaker1_turns), len(speaker2_turns))

    # speaker1 默认先说，然后交替
    for i in range(max_turns):
        if i < len(speaker1_turns):
            turns.append(
                {
                    "speaker_name": speaker1_name,
                    "speaker_role": "speaker1",
                    "channel": "left",
                    "text": speaker1_turns[i],
                }
            )

        if i < len(speaker2_turns):
            turns.append(
                {
                    "speaker_name": speaker2_name,
                    "speaker_role": "speaker2",
                    "channel": "right",
                    "text": speaker2_turns[i],
                }
            )

    left_chunks = []
    right_chunks = []
    turn_metadata = []

    cursor = 0

    for turn_idx, turn in enumerate(turns):
        wav = _generate_tts(
            speaker_name=turn["speaker_name"],
            text=turn["text"],
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

        num_samples = len(wav)
        zero_padding = np.zeros(num_samples, dtype=np.float32)

        if turn["channel"] == "left":
            left_chunks.append(wav)
            right_chunks.append(zero_padding)
        else:
            left_chunks.append(zero_padding)
            right_chunks.append(wav)

        turn_metadata.append(
            {
                "turn_index": turn_idx,
                "speaker_role": turn["speaker_role"],
                "speaker_name": turn["speaker_name"],
                "channel": turn["channel"],
                "text": turn["text"],
                "start_sample": int(cursor),
                "end_sample": int(cursor + num_samples),
                "num_samples": int(num_samples),
                "start_sec": float(cursor / _SAMPLE_RATE),
                "end_sec": float((cursor + num_samples) / _SAMPLE_RATE),
            }
        )

        cursor += num_samples

    speaker1_tts = np.concatenate(left_chunks).astype(np.float32)
    speaker2_tts = np.concatenate(right_chunks).astype(np.float32)

    # 显式 zero padding，保证两个 np array 长度完全一致
    target_len = max(len(speaker1_tts), len(speaker2_tts))

    if len(speaker1_tts) < target_len:
        speaker1_tts = np.pad(
            speaker1_tts,
            (0, target_len - len(speaker1_tts)),
            mode="constant",
        )

    if len(speaker2_tts) < target_len:
        speaker2_tts = np.pad(
            speaker2_tts,
            (0, target_len - len(speaker2_tts)),
            mode="constant",
        )

    speaker1_file = output_dir / f"{_safe_filename(speaker1_name)}_TTS.npy"
    speaker2_file = output_dir / f"{_safe_filename(speaker2_name)}_TTS.npy"
    stereo_wav_file = output_dir / "dialogue_TTS.wav"
    metadata_file = output_dir / "channel_map.json"

    if save_npy:
        np.save(speaker1_file, speaker1_tts)
        np.save(speaker2_file, speaker2_tts)

    if save_stereo_wav:
        stereo = np.stack([speaker1_tts, speaker2_tts], axis=0)
        stereo_tensor = torch.from_numpy(stereo)
        ta.save(str(stereo_wav_file), stereo_tensor, _SAMPLE_RATE)

    metadata = {
        "sample_rate": _SAMPLE_RATE,
        "num_samples": int(target_len),
        "duration_sec": float(target_len / _SAMPLE_RATE),
        "channel_map": {
            "left": {
                "speaker_role": "speaker1",
                "speaker_name": speaker1_name,
                "array_name": "speaker1_tts",
                "file": str(speaker1_file) if save_npy else None,
            },
            "right": {
                "speaker_role": "speaker2",
                "speaker_name": speaker2_name,
                "array_name": "speaker2_tts",
                "file": str(speaker2_file) if save_npy else None,
            },
        },
        "files": {
            "speaker1_tts_npy": str(speaker1_file) if save_npy else None,
            "speaker2_tts_npy": str(speaker2_file) if save_npy else None,
            "stereo_wav": str(stereo_wav_file) if save_stereo_wav else None,
            "channel_map_json": str(metadata_file) if save_json else None,
        },
        "turns": turn_metadata,
    }

    if save_json:
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {
        "speaker1_tts": speaker1_tts,
        "speaker2_tts": speaker2_tts,
        "metadata": metadata,
    }