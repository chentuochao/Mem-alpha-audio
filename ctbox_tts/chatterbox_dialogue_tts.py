# chatterbox_dialogue_tts.py

import json
import re
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torchaudio as ta

def get_interval(sr):
    if np.random.rand() < 0.3:
        interval_single = [-int(0.5*sr), 1.5*sr]
        gap = np.random.randint(low = interval_single[0], high = interval_single[1] )
    else:
        intervals = [[-1.5, -1.2], [-1.2, -0.9], [-0.9, -0.8], [-0.8, -0.7000000000000001], [-0.7000000000000001, -0.6000000000000001], [-0.6000000000000001, -0.5000000000000001], [-0.5000000000000001, -0.40000000000000013], [-0.40000000000000013, -0.30000000000000016], [-0.30000000000000016, -0.20000000000000015], [-0.20000000000000015, -0.10000000000000014], [-0.10000000000000014, -1.3877787807814457e-16], [-1.3877787807814457e-16, 0.09999999999999987], [0.09999999999999987, 0.19999999999999987], [0.19999999999999987, 0.2999999999999999], [0.2999999999999999, 0.3999999999999999], [0.3999999999999999, 0.4999999999999999], [0.4999999999999999, 0.5999999999999999], [0.5999999999999999, 0.6999999999999998], [0.6999999999999998, 0.7999999999999998], [0.7999999999999998, 0.8999999999999998], [0.8999999999999998, 0.9999999999999998], [0.9999999999999998, 1.0999999999999999], [1.0999999999999999, 1.2], [1.2, 1.3], [1.3, 1.4000000000000001], [1.4000000000000001, 1.5000000000000002], [1.5000000000000002, 1.6000000000000003], [1.6000000000000003, 1.7000000000000004], [1.7000000000000004, 1.8000000000000005], [1.8000000000000005, 1.9000000000000006], [1.9000000000000006, 2.0000000000000004], [2, 2.5]]
        probs = [0.0016664352173309262, 0.0024996528259963896, 0.004999305651992779, 0.00611026246354673, 0.00860991528954312, 0.012914872934314679, 0.026107485071517843, 0.036522705179836135, 0.05360366615747813, 0.07012914872934314, 0.08193306485210387, 0.10970698514095265, 0.1333148173864741, 0.12373281488682128, 0.09332037217053187, 0.06401888626579642, 0.0452714900708235, 0.035828357172614914, 0.022913484238300235, 0.015414525760311068, 0.012776003332870435, 0.010137480905429801, 0.006665740869323705, 0.004999305651992779, 0.003888348840438828, 0.0033328704346618524, 0.00180530481877517, 0.001944174420219414, 0.0016664352173309262, 0.0013886960144424386, 0.0013886960144424386, 0.0013886960144424386]
        _id = np.random.choice(len(intervals), p=probs)
        gap = np.random.uniform(low = intervals[_id][0], high = intervals[_id][1])
        gap = int(gap*sr)

    return gap



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

    # 1) Render every turn's waveform first, so we can place them on a shared
    #    timeline with inter-turn gaps/overlaps from get_interval().
    rendered = []  # list of (turn, wav, num_samples)
    for turn in turns:
        wav = _generate_tts(
            speaker_name=turn["speaker_name"],
            text=turn["text"],
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        rendered.append((turn, wav, len(wav)))

    # 2) Compute the absolute start sample of each turn.
    #    get_interval() returns an inter-turn offset in SAMPLES:
    #        gap > 0  -> silence inserted between turns
    #        gap < 0  -> the next turn starts before the previous ends (overlap)
    #    The first turn always starts at 0; starts are clamped to be >= 0.
    placements = []  # list of (start, end, gap)
    prev_end = 0
    for idx, (turn, wav, num_samples) in enumerate(rendered):
        if idx == 0:
            gap = 0
            start = 0
        else:
            gap = int(get_interval(_SAMPLE_RATE))
            start = max(0, prev_end + gap)
        end = start + num_samples
        placements.append((start, end, gap))
        prev_end = end

    total_len = max((end for _, end, _ in placements), default=0)

    # 3) Add each turn into its channel buffer at its absolute offset.
    #    Using += (not assignment) so overlapping turns mix instead of
    #    overwriting one another.
    left_buffer = np.zeros(total_len, dtype=np.float32)
    right_buffer = np.zeros(total_len, dtype=np.float32)
    turn_metadata = []

    for turn_idx, ((turn, wav, num_samples), (start, end, gap)) in enumerate(
        zip(rendered, placements)
    ):
        if turn["channel"] == "left":
            left_buffer[start:end] += wav
        else:
            right_buffer[start:end] += wav

        turn_metadata.append(
            {
                "turn_index": turn_idx,
                "speaker_role": turn["speaker_role"],
                "speaker_name": turn["speaker_name"],
                "channel": turn["channel"],
                "text": turn["text"],
                "gap_samples": int(gap),
                "gap_sec": float(gap / _SAMPLE_RATE),
                "start_sample": int(start),
                "end_sample": int(end),
                "num_samples": int(num_samples),
                "start_sec": float(start / _SAMPLE_RATE),
                "end_sec": float(end / _SAMPLE_RATE),
            }
        )

    # Overlapping turns on the same channel can sum past [-1, 1]; clip to be safe.
    speaker1_tts = np.clip(left_buffer, -1.0, 1.0).astype(np.float32)
    speaker2_tts = np.clip(right_buffer, -1.0, 1.0).astype(np.float32)

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


# ============================================================
# 5. 多人版本：支持任意说话人数量
# ============================================================
def generate_multispeaker_dialogue_tts(
    speaker_names: List[str],
    speaker_texts: List[Dict[str, str]],
    output_dir: str = "tts_outputs",
    save_per_speaker_npy: bool = True,
    save_multichannel_wav: bool = True,
    save_mono_wav: bool = True,
    save_json: bool = True,
    exaggeration: Optional[float] = None,
    cfg_weight: Optional[float] = None,
    use_intervals: bool = True,
) -> Dict[str, Any]:
    """
    Generate multi-channel dialogue TTS for an arbitrary number of speakers.

    Args:
        speaker_names: ordered list of distinct speaker names. Each speaker
            gets its own dedicated channel; channel index == position in this
            list. Every name must exist in REFERENCE_VOICE_MAP.
        speaker_texts: the conversation as an ordered list of turns. Each turn
            is a dict mapping a speaker name to that turn's text, e.g.
                [{"Alice": "Hi Bob, Carol."},
                 {"Bob":   "Hey Alice!"},
                 {"Carol": "Good to see you both."}]
            List order == speaking order. A turn dict normally holds a single
            {name: text} pair; if it holds several, they are flattened into
            consecutive turns in insertion order.
        output_dir: where to write outputs.
        save_per_speaker_npy: save each speaker's isolated track as <name>_TTS.npy.
        save_multichannel_wav: save an N-channel wav (one channel per speaker).
        save_mono_wav: save a mono mix-down of all speakers.
        save_json: save channel_map.json with per-speaker channels + per-turn
            timing (including the gap/overlap chosen for each turn).
        exaggeration, cfg_weight: optional Chatterbox knobs (forwarded per turn).
        use_intervals: if True, insert get_interval() gaps/overlaps between
            turns; if False, turns are placed strictly back-to-back.

    Returns:
        {
            "speaker_tts": {name: np.ndarray, ...},  # per-speaker isolated tracks
            "mono_mix": np.ndarray,                   # all speakers summed (mono)
            "metadata": dict,
        }
    """
    if not speaker_names:
        raise ValueError("speaker_names must be a non-empty list.")
    if len(set(speaker_names)) != len(speaker_names):
        raise ValueError(f"speaker_names must be unique, got: {speaker_names}")

    # name -> channel index
    channel_of = {name: i for i, name in enumerate(speaker_names)}
    num_channels = len(speaker_names)

    # ---- flatten the ordered turns into (speaker_name, text) pairs ----
    turns: List[Tuple[str, str]] = []
    for turn_idx, turn in enumerate(speaker_texts):
        if not isinstance(turn, dict) or not turn:
            raise ValueError(
                f"speaker_texts[{turn_idx}] must be a non-empty "
                f"{{speaker_name: text}} dict, got: {turn!r}"
            )
        for name, text in turn.items():
            if name not in channel_of:
                raise ValueError(
                    f"speaker_texts[{turn_idx}] references unknown speaker "
                    f"{name!r}; not in speaker_names {speaker_names}."
                )
            text = (text or "").strip()
            if text:
                turns.append((name, text))

    if not turns:
        raise ValueError("speaker_texts contains no non-empty turns.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) render every turn's waveform
    rendered = []  # list of (name, wav, num_samples)
    for name, text in turns:
        wav = _generate_tts(
            speaker_name=name,
            text=text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        rendered.append((name, wav, len(wav)))

    # 2) place each turn on a shared timeline with gaps/overlaps
    #    gap > 0 -> silence between turns; gap < 0 -> overlap (clamped to >= 0)
    placements = []  # list of (start, end, gap)
    prev_end = 0
    for idx, (name, wav, num_samples) in enumerate(rendered):
        if idx == 0:
            gap = 0
            start = 0
        else:
            gap = int(get_interval(_SAMPLE_RATE)) if use_intervals else 0
            start = max(0, prev_end + gap)
        end = start + num_samples
        placements.append((start, end, gap))
        prev_end = end

    total_len = max((end for _, end, _ in placements), default=0)

    # 3) mix each turn into its speaker's channel buffer (+= so overlaps blend)
    channels = [np.zeros(total_len, dtype=np.float32) for _ in range(num_channels)]
    turn_metadata = []
    for turn_idx, ((name, wav, num_samples), (start, end, gap)) in enumerate(
        zip(rendered, placements)
    ):
        ch = channel_of[name]
        channels[ch][start:end] += wav
        turn_metadata.append(
            {
                "turn_index": turn_idx,
                "speaker_name": name,
                "channel": ch,
                "text": turns[turn_idx][1],
                "gap_samples": int(gap),
                "gap_sec": float(gap / _SAMPLE_RATE),
                "start_sample": int(start),
                "end_sample": int(end),
                "num_samples": int(num_samples),
                "start_sec": float(start / _SAMPLE_RATE),
                "end_sec": float(end / _SAMPLE_RATE),
            }
        )

    # clip each isolated channel to [-1, 1] (same-speaker overlaps can sum)
    channels = [np.clip(c, -1.0, 1.0).astype(np.float32) for c in channels]
    speaker_tts = {name: channels[channel_of[name]] for name in speaker_names}

    # mono mix-down: sum all channels, normalize if it would clip
    mono_mix = np.sum(np.stack(channels, axis=0), axis=0).astype(np.float32)
    peak = float(np.max(np.abs(mono_mix))) if total_len else 0.0
    if peak > 1.0:
        mono_mix = (mono_mix / peak * 0.95).astype(np.float32)

    # ---- save outputs ----
    per_speaker_files = {}
    if save_per_speaker_npy:
        for name in speaker_names:
            f = output_dir / f"{_safe_filename(name)}_TTS.npy"
            np.save(f, speaker_tts[name])
            per_speaker_files[name] = str(f)

    multichannel_wav_file = output_dir / "dialogue_multichannel_TTS.wav"
    if save_multichannel_wav:
        multi = np.stack(channels, axis=0)  # [num_channels, T]
        ta.save(str(multichannel_wav_file), torch.from_numpy(multi), _SAMPLE_RATE)

    mono_wav_file = output_dir / "dialogue_mono_TTS.wav"
    if save_mono_wav:
        ta.save(
            str(mono_wav_file),
            torch.from_numpy(mono_mix).unsqueeze(0),  # [1, T]
            _SAMPLE_RATE,
        )

    metadata_file = output_dir / "channel_map.json"
    metadata = {
        "sample_rate": _SAMPLE_RATE,
        "num_samples": int(total_len),
        "duration_sec": float(total_len / _SAMPLE_RATE),
        "num_speakers": num_channels,
        "use_intervals": bool(use_intervals),
        "channel_map": {
            name: {
                "channel": channel_of[name],
                "speaker_name": name,
                "npy_file": per_speaker_files.get(name),
            }
            for name in speaker_names
        },
        "files": {
            "per_speaker_npy": per_speaker_files if save_per_speaker_npy else None,
            "multichannel_wav": str(multichannel_wav_file) if save_multichannel_wav else None,
            "mono_wav": str(mono_wav_file) if save_mono_wav else None,
            "channel_map_json": str(metadata_file) if save_json else None,
        },
        "turns": turn_metadata,
    }
    if save_json:
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {
        "speaker_tts": speaker_tts,
        "mono_mix": mono_mix,
        "metadata": metadata,
    }