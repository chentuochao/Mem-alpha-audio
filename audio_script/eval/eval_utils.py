import json

from typing import List, Dict
import numpy as np
from .multitalker_metrics import compute_der, calculate_session_cpWER, normalize_string
from audio_script.datasets.turn_annotation import AlignedProcess
import string
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
TURN_GAP_TH = 1.5


def normalize_string(input_string):
    result = input_string.lower()
    # return result
    for char in string.punctuation:
        result = result.replace(char, "")
    return result

## print function
def print_turns(turns):
    for utt in turns:
        # print(utt["dialog_type"], utt["speaker"], utt["start"], utt["end"], utt["text"] )
        speaker = utt["speaker"]
        start = utt["start"]
        end = utt["end"]
        text = utt["text"]
        print(f"{speaker}[{start:.1f}-{end:.1f}]: {text }")


def remove_first_punction(text):
    for char in string.punctuation:
        if text.startswith(char):
            text = text[1:]
    return text.strip()

def parse_turn(turns):
    dialog = []
    for utt in turns:
        start = utt["start"]
        end = utt["end"]
        text = utt["text"]
        text = remove_first_punction(text.strip())
        dialog.append({
            "speaker": utt["speaker"],
            "start": start,
            "end": end,
            "text": text
        })
    return dialog




def best_match_tp_fp_fn(pred_speakers, gt_speakers):
    n_pred, n_gt = len(pred_speakers), len(gt_speakers)
    gt_ids = [g[0] for g in gt_speakers]
    score = np.zeros((n_pred, n_gt), dtype=int)
    for i, p in enumerate(pred_speakers):
        c = Counter(p)
        for j, gt_id in enumerate(gt_ids):
            score[i, j] = c.get(gt_id, 0)
    m = max(n_pred, n_gt)
    cost = np.zeros((m, m), dtype=int)
    cost[:n_pred, :n_gt] = -score
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_pred = set()
    matched_gt = set()
    tp_total = 0
    fp_total = 0
    fn_total = 0
    for r, c in zip(row_ind, col_ind):
        if r < n_pred and c < n_gt:
            matched_pred.add(r)
            matched_gt.add(c)
            tp = score[r, c]
            tp_total += tp
            fp_total += len(pred_speakers[r]) - tp
            fn_total += len(gt_speakers[c]) - tp
    # unmatched pred => FP
    for i in range(n_pred):
        if i not in matched_pred:
            fp_total += len(pred_speakers[i])
    # unmatched gt => FN
    for j in range(n_gt):
        if j not in matched_gt:
            fn_total += len(gt_speakers[j])
    return tp_total, fp_total, fn_total


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
    """
        From a word_list dict {speaker_id: [{word, start, end, ...}, ...]},
        return a list of concatenated text strings for each non-empty speaker.
    """
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
    # print(speaker_transcripts, valid_speakers)

    if len(valid_speakers) == 0:
        print(f"No valid speakers found for!")
        return []

    elif len(valid_speakers) == 1:
        print(f"Only one valid speaker found for")
        words = speaker_transcripts[valid_speakers[0]][0]["words"]
        transcript = ""
        for w in words:
            transcript += w["word"]
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



def parse_transcript_morespeakers(word_list: Dict) -> List[Dict]:
    """Parse a word-list dict into speaker-aware turns for 1–8 speakers.

    Takes the output format of transcript_gt.json:
        {speaker_id: [{word, start, end, score, ...}, ...], ...}

    and returns a sorted list of turn dicts, each with keys:
        dialog_type, speaker, start, end, text, wfeats.

    Steps:
        1. Filter out speakers with no words.
        2. Sort each speaker's words by start time.
        3. Wrap each speaker's word list into a single segment dict
           (the format AlignedProcess expects).
        4. For 0 speakers  -> return empty list.
           For 1 speaker   -> return one turn covering all words.
           For 2+ speakers -> use AlignedProcess_Morespeakers to detect
                              turns, backchannels, and overlaps.

    Args:
        word_list: dict mapping speaker names to lists of word dicts.
                   Each word dict must have at least: word, start, end, score.

    Returns:
        List of turn dicts sorted by (start, -end), with dialog_type labels.
    """
    # --- Step 1 & 2: Collect valid speakers and sort their words ---
    speaker_transcripts = {}
    valid_speakers = []
    for speaker in word_list.keys():
        words = word_list[speaker]
        if len(words) == 0:
            continue
        # Sort words by start time to ensure chronological order.
        words = sorted(words, key=lambda x: x['start'])
        valid_speakers.append(speaker)
        # Wrap the flat word list into a single-segment list,
        # which is the format AlignedProcess / AlignedProcess_Morespeakers
        # expects as input per speaker.
        speaker_transcripts[speaker] = [{
            "speaker": speaker,
            "start": words[0]['start'],
            "end": words[-1]['end'],
            "words": words,
        }]

    # --- Step 3: Handle based on number of valid speakers ---
    if len(valid_speakers) == 0:
        print("No valid speakers found!")
        return []

    if len(valid_speakers) == 1:
        # Only one speaker: no turn-taking to detect, just return one turn.
        print(f"Only one valid speaker found: {valid_speakers[0]}")
        words = speaker_transcripts[valid_speakers[0]][0]["words"]
        transcript = "".join(w["word"] for w in words)
        return [{
            "dialog_type": "dialog",
            "speaker": valid_speakers[0],
            "start": words[0]['start'],
            "end": words[-1]['end'],
            "text": transcript,
            "wfeats": words,
        }]

    # --- 2+ speakers: use AlignedProcess_Morespeakers ---
    # Build the ordered lists that the constructor expects.
    transcripts_list = [speaker_transcripts[spk] for spk in valid_speakers]

    aligned_process = AlignedProcess_Morespeakers(
        transcripts=transcripts_list,
        speaker_names=valid_speakers,
        interval_character=' ',
        turn_gap_threshold=TURN_GAP_TH,
    )

    # get_parsed_dialog_combined returns all speakers' turns in one
    # sorted list, already tagged with dialog_type and human-readable
    # speaker names.
    speaker_aware_turn = aligned_process.get_parsed_dialog_combined()
    return speaker_aware_turn


def eval_cpwer_seamlessinteraction(pred_transcripts, gt_files, limit_hypo_number = False):
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

    cpwer, _, _, best_perm = calculate_session_cpWER(spk_hypothesis, spk_reference, limit_hypo_number = limit_hypo_number)
    best_perm = [speakers_pred[i] for i in best_perm]
    # print(f"  Best permutation: {best_perm}")
    # print(f"  cpWER: {cpwer:.4f}")

    return cpwer, best_perm
