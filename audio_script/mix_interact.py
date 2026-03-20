import librosa
import numpy as np
import soundfile as sf
import os
import argparse
import json
import time
import threading
import queue
import random
import string
import logging
import traceback
import shutil


output_folder = "mix_conversation_dataset"
audios = [
    ["/checkpoint/seamless/data/Mosaic/naturalistic/test/audio/V00_S0062_I00000125_P0092.wav", "/checkpoint/seamless/data/Mosaic/naturalistic/test/audio/V00_S0062_I00000125_P0093.wav"],
    ["/checkpoint/seamless/data/Mosaic/naturalistic/test/audio/V00_S0062_I00000126_P0092.wav", "/checkpoint/seamless/data/Mosaic/naturalistic/test/audio/V00_S0062_I00000126_P0093.wav"],
    ["/checkpoint/seamless/data/Mosaic/naturalistic/test/audio/V00_S0062_I00000128_P0092.wav", "/checkpoint/seamless/data/Mosaic/naturalistic/test/audio/V00_S0062_I00000128_P0093.wav"],
    ["/checkpoint/seamless/data/Mosaic/naturalistic/test/audio/V00_S0062_I00000129_P0092.wav", "/checkpoint/seamless/data/Mosaic/naturalistic/test/audio/V00_S0062_I00000129_P0093.wav"],
    ["/checkpoint/seamless/data/Mosaic/naturalistic/test/audio/V00_S0062_I00000130_P0092.wav", "/checkpoint/seamless/data/Mosaic/naturalistic/test/audio/V00_S0062_I00000130_P0093.wav"],

]


for audio_pair in audios:
    audio_list = []
    audio_names = []
    for audio_file in audio_pair:
        audio, sr = librosa.load(audio_file, sr=None, mono=True)
        audio_list.append(audio)
        audio_names.append(os.path.basename(audio_file).split(".")[0])
        trans_file = audio_file.replace("audio", "metadata/transcript").replace(".wav", ".jsonl")
        # copy the transcript file to the output folder
        shutil.copy(trans_file, os.path.join(output_folder, os.path.basename(trans_file)))


    min_len = min(len(audio) for audio in audio_list)
    audio_list = [audio[:min_len] for audio in audio_list]
    audio_array = np.array(audio_list)
    audio_array = audio_array.mean(axis=0)
    audio_array = audio_array / np.max(np.abs(audio_array)) * 0.8
    audio_array = audio_array.astype(np.float32)


    outname = "_".join(audio_names)
    sf.write(os.path.join(output_folder, f"{outname}.wav"), audio_array, sr)
    print(f"Mixed audio saved to mixed_audio.wav")
