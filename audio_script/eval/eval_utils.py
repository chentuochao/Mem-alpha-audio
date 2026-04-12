import json

from typing import List, Dict
import numpy as np
from .multitalker_metrics import compute_der, calculate_session_cpWER, normalize_string

## print function
def print_turns(turns):
    for utt in turns:
        # print(utt["dialog_type"], utt["speaker"], utt["start"], utt["end"], utt["text"] )
        speaker = utt["speaker"]
        start = utt["start"]
        end = utt["end"]
        text = utt["text"]
        print(f"{speaker}[{start:.1f}-{end:.1f}]: {text }")


### evaluation functions for SeamlessInteraction dataset
def load_vad_json(path: str) -> List[Dict]:
    """Load a VAD file (plain JSON array or JSONL) → [{start, end}, ...]"""
    with open(path, "r") as f:
        text = f.read().strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        pass
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries



def vad_segments_to_binary(vad_segments: List[Dict], total_frames: int,
                           frame_duration: float = 0.01) -> np.ndarray:
    """Convert a list of {start, end} VAD segments to a binary vector."""
    binary = np.zeros(total_frames, dtype=np.float32)
    for seg in vad_segments:
        s = int(seg["start"] / frame_duration)
        e = int(seg["end"] / frame_duration)
        e = min(e, total_frames)
        if s < total_frames:
            binary[s:e] = 1.0
    return binary



def eval_der_seamlessinteraction(pred, gt_files, frame_duration=0.08):
    """
    pred - (T, num_speakers)
    gt_files - List[Dict]: {"SPEAK0": path_to_vad1, "SPEAK1": path_to_vad2}
    """
    speaker_gt = []
    gt_matrix = []

    total_frames = pred.shape[0]

    for spk, vad_path in gt_files.items():
        vad_segments = load_vad_json(vad_path)
        gt_array = vad_segments_to_binary(vad_segments, total_frames, frame_duration)
        speaker_gt.append(spk)
        gt_matrix.append(gt_array)
    gt_matrix = np.stack(gt_matrix, axis=0)
    pred_matrix = pred.T  # (num_speakers, T)

    der, der_details = compute_der(pred_matrix, gt_matrix, frame_duration=frame_duration)

    best_perm = der_details["col_ind"]
    der_details["speaker_gt"] = speaker_gt
    # print(f"best perm: {best_perm}")
    # print(f"  DER: {der:.4f}  "
    #         f"(miss={der_details['miss']:.2f}s, fa={der_details['fa']:.2f}s, "
    #         f"conf={der_details['conf']:.2f}s, total={der_details['total']:.2f}s)")

    return der, best_perm, der_details



## eval the transcriptions of the otuput
def extract_text_from_transcript(transcript) -> str:
    """Load a transcript JSON and return concatenated segment-level text."""

    words = []
    for seg in transcript:
        words.append(seg["text"])
    trans = " ".join(words)
    # trans = trans.lower()
    trans = normalize_string(trans)
    return trans


def build_speaker_transcripts(word_list: Dict[str, List[Dict]]) -> List[str]:
    """
    From a word_list dict {speaker_id: [{word, start, end, ...}, ...]},
    return a list of concatenated text strings for each non-empty speaker.
    """
    speakers_list = sorted(list(word_list.keys()))
    transcripts_plain = []
    valid_speakers = []
    for speaker in speakers_list:
        if len(word_list[speaker]) == 0:
            continue
        trans = ""
        for word in word_list[speaker]:
            trans += word["word"]

        trans = normalize_string(trans)
        transcripts_plain.append(trans)
        valid_speakers.append(speaker)

    return transcripts_plain, valid_speakers


def parse_transcript(word_list: Dict) -> List[Dict]:
    # parse the output of words list
    # check the speaker number
    speaker_transcripts = {}
    valid_speakers = []
    transcripts = []
    for speaker in word_list.keys():
        words = word_list[speaker]
        if len(words) == 0:
            continue
        # sorted the words by "start" time
        words = sorted(words, key=lambda x: x['start'])
        transcript = ""
        for word in words:
            transcript += word["word"]
        transcripts.append(transcript)
        valid_speakers.append(speaker)
        speaker_transcripts[speaker] = [{
            "speaker": speaker,
            "start": words[0]['start'],
            "end": words[-1]['end'],
            "words": words
        }]

    speaker_aware_turn = []
    transA, transB = None, None
    if len(valid_speakers) == 0:
        print(f"No valid speakers found for!")
        return []

    elif len(valid_speakers) == 1:
        print(f"Only one valid speaker found for")
        words = speaker_transcripts[valid_speakers[0]]["words"]
        transcript = transcripts[0]
        speaker_aware_turn = [{
            "dialog_type": "dialog",
            "speaker": valid_speakers[0],
            "start": words[0]['start'],
            "end": words[-1]['end'],
            "text": transcript,
            "wfeats": words
        }]
    elif len(valid_speakers) == 2:
        speaker0 = valid_speakers[0]
        speaker1 = valid_speakers[1]
        aligned_process = AlignedProcess(speaker_transcripts[speaker0], speaker_transcripts[speaker1], speaker0, speaker1, interval_character='', turn_gap_threshold = TURN_GAP_TH)
        transA, transB = aligned_process.get_parsed_dialog()
        speaker_aware_turn = transA + transB
        speaker_aware_turn.sort(key=lambda key: (key['start'], -key['end']))

    else:
        # find the top2 speaker with longest transcript
        # sort the valid_speakers by the length of transcripts
        # print(transcripts)
        # print(valid_speakers)
        lengths = [len(t) for t in transcripts]
        sorted_pairs = sorted(zip(lengths, valid_speakers), reverse=True)  # longer first
        valid_speakers = [speaker for length, speaker in sorted_pairs]
        valid_speakers = valid_speakers[:2]
        speaker0 = valid_speakers[0]
        speaker1 = valid_speakers[1]
        # print(valid_speakers)
        # exit(0)
        aligned_process = AlignedProcess(speaker_transcripts[speaker0], speaker_transcripts[speaker1], speaker0, speaker1, interval_character='', turn_gap_threshold = TURN_GAP_TH)
        transA, transB = aligned_process.get_parsed_dialog()
        speaker_aware_turn = transA + transB
        speaker_aware_turn.sort(key=lambda key: (key['start'], -key['end']))

    # print(speaker_aware_turn)
    # for utt in speaker_aware_turn:
    #     print(utt["dialog_type"], utt["speaker"], utt["start"], utt["end"], utt["text"] )

    return speaker_aware_turn


def eval_cpwer_seamlessinteraction(pred_transcripts, gt_files):
    """
        pred_transcripts - {
            "SPEAK0": [{"word": "hello", "start": 0.0, "end": 1.0}, ...],
            "SPEAK1": [{"word": "world", "start": 1.0, "end": 2.0}, ...],
        }
        gt_files - Dict: {"SPEAK0": path_to_trans1, "SPEAK1": path_to_trans2}
    """
    spk_hypothesis, speakers_pred = build_speaker_transcripts(pred_transcripts)

    spk_reference = []
    speaker_gt = []
    for spk, gt_path in gt_files.items():
        with open(gt_path, "r") as f:
            gt_trans = json.load(f)
        ref_text = extract_text_from_transcript(gt_trans)
        spk_reference.append(ref_text)
        speaker_gt.append(spk)

    cpwer, _, _, best_perm = calculate_session_cpWER(spk_hypothesis, spk_reference)
    best_perm = [speakers_pred[i] for i in best_perm]
    # print(f"  Best permutation: {best_perm}")
    # print(f"  cpWER: {cpwer:.4f}")

    return cpwer, best_perm
