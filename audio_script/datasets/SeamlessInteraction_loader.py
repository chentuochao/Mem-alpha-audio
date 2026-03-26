import argparse
import json
import os
from pprint import pprint
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import glob
import os
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import jsonlines
import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .turn_annotation import AlignedProcess
# ==================================================================================================== #
# Constants
# ==================================================================================================== #
ANNOTATION_LABEL = "annotation"
ANNOTATION_TYPE_LABEL = "annotation_type"
INTERACT_BENCHMARK_PATH = "/checkpoint/seamless/data/InterAct_benchmark/"
SAMPLE_ID_LABEL = "sample_id"

DYADIC_JSONL_KEY_ID_0 = "id_0"
DYADIC_JSONL_KEY_ID_1 = "id_1"


# ==================================================================================================== #
# Helper Functions
# ==================================================================================================== #
def load_annotation(sample_id, benchmark_dataset_path, version, annotation_type):
    path = os.path.join(
        benchmark_dataset_path,
        version,
        annotation_type,
        ANNOTATION_LABEL,
        f"{sample_id}.json",
    )
    if not os.path.exists(path):
        raise ValueError(f"Provided annotation path ({path}) does not exist!")
    with open(path) as f:
        annotation = json.load(f)[ANNOTATION_LABEL]
    return annotation


def load_jsonl_file(file_path):
    with jsonlines.open(file_path) as reader:
        data = [obj for obj in reader]
    return data


def pair_files(file_list):
    # Create a dictionary to store the pairs
    pairs = {}
    for file in file_list:
        file = os.path.basename(file)

        match = re.match(r"(V\d+_S\d+_I\d+)_.*", file)
        file_id = file.split(".")[0]
        if match:
            key = match.group(1)
            if key not in pairs:
                pairs[key] = []
            # extract the speaker id from the file name Pxxx
            m = re.search(r"(P\d+)$", file_id)
            speaker_id = m.group(1) if m else None

            pairs[key].append({
                "conv_id": key,
                "file_id": file_id,
                "speaker_id": speaker_id,
            })
    # Filter out files that don't have a matched pair
    paired_files = [files for files in pairs.values() if len(files) > 1]
    return paired_files


class InterActDataset(object):
    def __init__(
        self,
        data_path,
        diag_format="naturalistic",
        split="train",
        sample_rate=16000,
    ):
        self.sample_rate = sample_rate

        VAD_PATH = os.path.join(data_path, diag_format, split, "metadata", "vad")
        transcript_PATH = os.path.join(data_path, diag_format, split, "metadata", "transcript")
        audio_PATH = os.path.join(data_path, diag_format, split, "audio")

        files = glob.glob(transcript_PATH + "/*.jsonl")
        self.all_data = pair_files(files)



        self.transcript_files = []
        self.audio_files = []
        self.vad_files = []
        self.valid_data = []
        self.speaker_data = []

        for s1_data, s2_data in self.all_data:
            all_files_finds = True
            conv_id = s1_data["conv_id"]
            s1 = s1_data["file_id"]
            s2 = s2_data["file_id"]
            s1_speaker_id = s1_data["speaker_id"]
            s2_speaker_id = s2_data["speaker_id"]
            audio1 = os.path.join(audio_PATH, s1 + ".wav")
            audio2 = os.path.join(audio_PATH, s2 + ".wav")

            transcript1 = os.path.join(transcript_PATH, s1 + ".jsonl")
            transcript2 = os.path.join(transcript_PATH, s2 + ".jsonl")

            vad1 = os.path.join(VAD_PATH, s1 + ".jsonl")
            vad2 = os.path.join(VAD_PATH, s2 + ".jsonl")

            if (not os.path.exists(audio1)) or not (os.path.exists(audio2)):
                all_files_finds = False
            if (not os.path.exists(transcript1)) or not (os.path.exists(transcript2)):
                all_files_finds = False
            if (not os.path.exists(vad1)) or not (os.path.exists(vad2)):
                all_files_finds = False

            if all_files_finds:
                self.valid_data.append((conv_id, s1, s2))
                self.transcript_files.append((transcript1, transcript2))
                self.audio_files.append((audio1, audio2))
                self.vad_files.append((vad1, vad2))
                self.speaker_data.append((s1_speaker_id, s2_speaker_id))

    def __len__(self):
        return len(self.valid_data)

    def parse_conversation(self, sample):
        amplitude_range = [0.5, 0.95]
        audio1 = sample['audios'][0]
        audio2 = sample['audios'][1]
        min_len = min([len(audio1), len(audio2)])
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        audio1 = audio1 / np.max(np.abs(audio1)) * np.random.uniform(amplitude_range[0], amplitude_range[1]) # random amplitude
        audio2 = audio2 / np.max(np.abs(audio2)) * np.random.uniform(amplitude_range[0], amplitude_range[1]) # random amplitude
        audio1 = audio1.astype(np.float32)
        audio2 = audio2.astype(np.float32)
        audio = (audio1 + audio2) / 2

        transcript1 = sample['transcripts'][0]
        transcript2 = sample['transcripts'][1]
        # parse the transcript for turn annotation
        aligned_process = AlignedProcess(transcript1, transcript2, sample['speaker_id'][0], sample['speaker_id'][1])
        transA, transB = aligned_process.get_parsed_dialog()

        temp_dialogs = transA + transB
        temp_dialogs.sort(key=lambda key: (key['start'], -key['end']))
        for utt in temp_dialogs:
            print(utt["dialog_type"], utt["speaker"], utt["start"], utt["end"], utt["text"] )

        # parse the diarization result
        vad1 = sample['vad'][0]
        vad2 = sample['vad'][1]
        print("VAD=", len(vad1), len(vad2))
        print(vad1, vad2)


        parsed_sample = {
            "conv_id": sample['conv_id'],
            "id": sample['id'],
            "speaker_id": sample['speaker_id'],
            "audios": audio,
            "transcripts": [transA, transB],
            "vad": sample['vad'],
        }
        return parsed_sample




    def load_sample(self, idx) -> Dict[str, Any]:
        conv_id, s1, s2 = self.valid_data[idx]
        audiof1, audiof2 = self.audio_files[idx]
        audio1, sr = librosa.load(audiof1, sr=self.sample_rate, mono=True)
        # audio1 = audio1[np.newaxis, :]
        audio2, sr = librosa.load(audiof2, sr=self.sample_rate, mono=True)
        # audio2 = audio2[np.newaxis, :]

        transcriptf1, transcriptf2 = self.transcript_files[idx]
        transcript1 = load_jsonl_file(transcriptf1)
        transcript2 = load_jsonl_file(transcriptf2)

        vadf1, vadf2 = self.vad_files[idx]
        vad1 = load_jsonl_file(vadf1)
        vad2 = load_jsonl_file(vadf2)

        s1_speaker_id, s2_speaker_id = self.speaker_data[idx]

        sample0 = {
            "id": [s1, s2],
            "conv_id": conv_id,
            "speaker_id": [s1_speaker_id, s2_speaker_id],
            "audios": [audio1, audio2],
            "sr": sr,
            "transcripts": [transcript1, transcript2],
            "vad": [vad1, vad2],
        }
        return self.parse_conversation(sample0)
