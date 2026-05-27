# prepare_profile_ref_voices.py
import io
import librosa

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio


# ============================================================
# 1. 你的 profile 人物 -> 目标性别
# ============================================================
# 这里是根据你当前 profile 里的关系设定做的固定映射：
# Wang Xiaoming: male
# Wang Xiaohong: elder sister -> female
# Zhang Wei: male friend
# Li Ting: girlfriend -> female
# Zhao Ming: male friend
# Liu Li: colleague, 这里先设为 female
# Zhang Wen: mentor, 这里先设为 male
# Li Ming: neighbor, 这里先设为 male
# Chen Hua: community elder, 这里先设为 male
# Liu Qiang: stranger, 这里先设为 male

PROFILE_SPEAKERS = [
    {"profile_name": "Wang Xiaoming", "gender": "M", "out_name": "wang_xiaoming.wav"},
    {"profile_name": "Wang Xiaohong", "gender": "F", "out_name": "wang_xiaohong.wav"},
    {"profile_name": "Zhang Wei", "gender": "M", "out_name": "zhang_wei.wav"},
    {"profile_name": "Li Ting", "gender": "F", "out_name": "li_ting.wav"},
    {"profile_name": "Zhao Ming", "gender": "M", "out_name": "zhao_ming.wav"},
    {"profile_name": "Liu Li", "gender": "F", "out_name": "liu_li.wav"},
    {"profile_name": "Zhang Wen", "gender": "M", "out_name": "zhang_wen.wav"},
    {"profile_name": "Li Ming", "gender": "M", "out_name": "li_ming.wav"},
    {"profile_name": "Chen Hua", "gender": "M", "out_name": "chen_hua.wav"},
    {"profile_name": "Liu Qiang", "gender": "M", "out_name": "liu_qiang.wav"},
]


# ============================================================
# 2. 配置
# ============================================================

HF_DATASET = "sdialog/voices-libritts"
OUT_DIR = Path("ref_voices")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 24000
REF_DURATION_SEC = 10.0

REFERENCE_MAP_JSON = OUT_DIR / "reference_voice_map.json"
REFERENCE_MAP_PY = OUT_DIR / "reference_voice_map.py"


def trim_or_pad_audio(wav: np.ndarray, sr: int, duration_sec: float) -> np.ndarray:
    """
    裁剪或补零到固定长度。
    Chatterbox 其实不强制必须固定长度，但统一成 10s 更干净。
    """
    wav = np.asarray(wav, dtype=np.float32)

    if wav.ndim == 2:
        # Hugging Face Audio 通常已经是 mono，但这里兜底
        wav = wav.mean(axis=1)

    target_len = int(sr * duration_sec)

    if len(wav) >= target_len:
        return wav[:target_len].astype(np.float32)

    padded = np.zeros(target_len, dtype=np.float32)
    padded[: len(wav)] = wav
    return padded


def peak_normalize(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """
    简单 peak normalize，避免过小或爆音。
    """
    max_abs = float(np.max(np.abs(wav))) if len(wav) > 0 else 0.0
    if max_abs < 1e-8:
        return wav.astype(np.float32)

    wav = wav / max_abs * peak
    return wav.astype(np.float32)


# def load_voice_bank_streaming():
#     """
#     streaming=True 不会下载完整数据集。
#     cast_column 会把 audio decode 成 waveform，并重采样到 TARGET_SR。
#     """
#     ds = load_dataset(
#         HF_DATASET,
#         split="train",
#         streaming=True,
#     )

#     ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
#     return ds

def load_voice_bank_streaming():
    ds = load_dataset(
        HF_DATASET,
        split="train",
        streaming=True,
    )

    # 关键：不要让 datasets 自动 decode audio
    # 否则会触发 torchcodec，然后又回到 FFmpeg / GLIBCXX 问题
    ds = ds.cast_column("audio", Audio(decode=False))
    return ds

def prepare_reference_voices() -> Dict[str, str]:
    ds = load_voice_bank_streaming()

    pending: List[dict] = PROFILE_SPEAKERS.copy()
    reference_voice_map: Dict[str, str] = {}
    selected_metadata = []

    used_identifiers = set()

    for ex in ds:
        if not pending:
            break

        gender = ex.get("gender")
        identifier = str(ex.get("identifier"))
        source_name = ex.get("name")
        subset = ex.get("subset")
        file_name = ex.get("file_name")
        total_duration_s = ex.get("total_duration_s")

        if identifier in used_identifiers:
            continue

        # 找第一个需要该 gender 的 profile speaker
        match_idx = None
        for i, item in enumerate(pending):
            if item["gender"] == gender:
                match_idx = i
                break

        if match_idx is None:
            continue

        profile_item = pending.pop(match_idx)
        profile_name = profile_item["profile_name"]
        out_path = OUT_DIR / profile_item["out_name"]

        # audio = ex["audio"]
        # wav = audio["array"]
        # sr = int(audio["sampling_rate"])


        audio_obj = ex["audio"]

        if audio_obj.get("bytes") is not None:
            with io.BytesIO(audio_obj["bytes"]) as f:
                wav, sr = sf.read(f, dtype="float32")
        else:
            raise RuntimeError(
                f"audio bytes is None. audio_obj={audio_obj}. "
                "The dataset may be storing only paths instead of embedded audio bytes."
            )

        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        wav = trim_or_pad_audio(wav, sr, REF_DURATION_SEC)
        wav = peak_normalize(wav)

        sf.write(out_path, wav, sr)

        used_identifiers.add(identifier)
        reference_voice_map[profile_name] = str(out_path)

        selected_metadata.append(
            {
                "profile_name": profile_name,
                "assigned_gender": gender,
                "reference_audio": str(out_path),
                "libritts_identifier": identifier,
                "libritts_speaker_name": source_name,
                "libritts_subset": subset,
                "libritts_file_name": file_name,
                "source_total_duration_s": total_duration_s,
                "saved_duration_s": REF_DURATION_SEC,
                "sample_rate": sr,
            }
        )

        print(
            f"[OK] {profile_name:15s} <- "
            f"{gender} / {identifier} / {source_name} / {subset} "
            f"-> {out_path}"
        )

    if pending:
        missing = [x["profile_name"] for x in pending]
        raise RuntimeError(
            f"Not enough matching voices were found. Missing: {missing}"
        )

    payload = {
        "dataset": HF_DATASET,
        "target_sample_rate": TARGET_SR,
        "reference_duration_sec": REF_DURATION_SEC,
        "reference_voice_map": reference_voice_map,
        "selected_metadata": selected_metadata,
    }

    with REFERENCE_MAP_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with REFERENCE_MAP_PY.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated by prepare_profile_ref_voices.py\n\n")
        f.write("REFERENCE_VOICE_MAP = {\n")
        for k, v in reference_voice_map.items():
            f.write(f'    "{k}": "{v}",\n')
        f.write("}\n")

    print()
    print(f"[DONE] Saved reference map JSON: {REFERENCE_MAP_JSON}")
    print(f"[DONE] Saved Python map file: {REFERENCE_MAP_PY}")
    print()
    print("Copy this into chatterbox_dialogue_tts.py:")
    print()
    print("REFERENCE_VOICE_MAP = {")
    for k, v in reference_voice_map.items():
        print(f'    "{k}": "{v}",')
    print("}")

    return reference_voice_map


if __name__ == "__main__":
    prepare_reference_voices()