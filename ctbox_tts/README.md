# Chatterbox Dialogue TTS

This module wraps Chatterbox TTS into a simple Python function for two-speaker dialogue generation.

Directly import and call the function in Python.


---

my use to generate the perlTQA synthezied dialogue
python perltqa_dialogue_tts.py --count-only      # stats only (verified ✓)
python perltqa_dialogue_tts.py --prepare-only     # build the 550-voice bank
python perltqa_dialogue_tts.py --limit 5          # test: first 5 dialogues
python perltqa_dialogue_tts.py --limit 0          # full run (heavy)
Useful flags: --skip-prepare (reuse existing map), --overwrite, --ref-dir, --output-dir.


---

## Installation

Create and activate a conda environment:

```bash
conda env create -f environment.yml
```

Check CUDA:

```bash
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

---

## Reference Voices

Each speaker name is mapped to a fixed reference voice:

```python
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
```

Prepare reference voices:

```bash
python audio_prepare.py
```

Expected output:

```text
ref_voices/
├── wang_xiaoming.wav
├── wang_xiaohong.wav
├── zhang_wei.wav
├── li_ting.wav
├── zhao_ming.wav
├── liu_li.wav
├── zhang_wen.wav
├── li_ming.wav
├── chen_hua.wav
├── liu_qiang.wav
└── reference_voice_map.json
```

---

## Input Format

Import the function:

```python
from chatterbox_dialogue_tts import generate_dialogue_tts
```

Call it with:

```python
result = generate_dialogue_tts(
    speaker1_name="Wang Xiaoming",
    speaker2_name="Wang Xiaohong",
    speaker1_text=[
        "Hello Xiaohong, this is a test.",
        "The left channel should contain my voice.",
    ],
    speaker2_text=[
        "Hi Xiaoming, I can hear you clearly.",
        "The right channel should contain my voice.",
    ],
    output_dir="tts_outputs/test",
)
```

Required inputs:

```text
speaker1_name: name of the first speaker
speaker2_name: name of the second speaker
speaker1_text: string or list of strings
speaker2_text: string or list of strings
```

If `speaker1_text` and `speaker2_text` are lists, each item is treated as one dialogue turn.

The dialogue order is:

```text
speaker1 turn 0
speaker2 turn 0
speaker1 turn 1
speaker2 turn 1
...
```

Speaker 1 always starts first.

---

## Output

The function returns:

```python
{
    "speaker1_tts": np.ndarray,
    "speaker2_tts": np.ndarray,
    "metadata": dict,
}
```

Saved files:

```text
tts_outputs/test/
├── Wang_Xiaoming_TTS.npy
├── Wang_Xiaohong_TTS.npy
├── dialogue_TTS.wav
└── channel_map.json
```

---

## Guarantees

This module guarantees:

1. `speaker1_tts` and `speaker2_tts` have the same length.
2. Speaker 1 is always assigned to the left channel.
3. Speaker 2 is always assigned to the right channel.
4. When one speaker is speaking, the other channel is zero-padded.
5. The speaker-to-channel mapping is explicitly saved in `channel_map.json`.

Timeline example:

```text
speaker1_tts: [speech][zeros ][speech][zeros ]
speaker2_tts: [zeros ][speech][zeros ][speech]
```

So the final audio is aligned and can be used as two synchronized channels.

---

## Minimal Test

```python
from chatterbox_dialogue_tts import generate_dialogue_tts

result = generate_dialogue_tts(
    speaker1_name="Wang Xiaoming",
    speaker2_name="Wang Xiaohong",
    speaker1_text="Hello, this is speaker one.",
    speaker2_text="Hi, this is speaker two.",
    output_dir="tts_outputs/minimal_test",
)

print(result["speaker1_tts"].shape)
print(result["speaker2_tts"].shape)
print(result["metadata"]["channel_map"])
```