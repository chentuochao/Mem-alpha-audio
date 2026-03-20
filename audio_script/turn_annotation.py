from typing import Optional, Tuple, Iterator, List, Set, Any
from dataclasses import dataclass
from queue import Empty, Full
from pathlib import Path
import os
import io
import re
import multiprocessing as mp
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import string

from itertools import zip_longest

AUDIO_SILENCE_CODES = torch.tensor([
    153,  14, 819,  15, 713, 220, 855,  42, 975, 865, 460, 815, 487, 221, 800, 100
]).unsqueeze(1) # [num_codebooks, 1]
from src.data.tokenizer import Llama3Tokenizer, SLMCodecSpecialTokens
from src.data.iterator.diag_base import Turn, TurnEndType, TurnType

def normalize_string(input_string):
    result = input_string
    # return result
    for char in string.punctuation:
        result = result.replace(char, '')
    return result.lower()

### old data to merge audio and text stream
def _is_valid_seg(seg):
    """
    Whisperx results in errors when aligining segments
    comprising numerical values. Errors can be in the form of
    missing timestamps (not 'start' or 'end' keys), or startime
    >= endtime within a segment.
    """
    if ('end' in seg) and ('start' in seg):
        if seg['start'] is None or seg['end'] is None:
            return False
        if seg['start'] >= seg['end']:
            return False
        return True
    else:
        return False

    return True


def _get_valid_segments(segments):
    _segments = []
    prev_seg_valid = True
    for _, seg in enumerate(segments):
        if len(_segments) == 0:
            # Continue until we find first valid segment
            if _is_valid_seg(seg):
                _seg = seg.copy()
                _segments.append(_seg)
            continue

        if _is_valid_seg(seg):
            if not prev_seg_valid:
                _segments[-1]['end'] = seg['end']
            _seg = seg.copy()
            _segments.append(_seg)
            prev_seg_valid = True
        else:
            _segments[-1]['text'] = ' '.join(
                [_segments[-1]['text'], seg['text']]
            )
            prev_seg_valid = False

    return _segments

BACKCHANNELS = [
    "yeah",
    "ok",
    "okay",
    "mm",
    "mmm"
    "hm",
    "hmm",
    "hmmm",
    "uh",
    "huh",
    "ha",
    "hah",
    "haha",
    "ah",
    "um",
    "umhum",
    "uhhuh",
    "right",
    "nice",
    "good",
    "fine",
    "oh",
    "really",
    "god",
    "hey",
    "so",
    "well",
    "all",
    "cool",
    "wow",
    "yep",
    "yo",
    "eh",
    "sure",
    "gotcha",
    "and",
    "yes",
    "ow",
    "yum",
    "sign",
    "laugh",
    "shoot"
]

def destyle(input_string):
    result = input_string.strip().lower()
    # return result
    for char in string.punctuation:
        result = result.replace(char, ' ')
    return result

def check_backchannel(turn_text):
    turn_text = destyle(turn_text)
    words = turn_text.split()
    if len(words) > 4:
        return 0

    for w in words:
        if w not in BACKCHANNELS:
            return 0

    return len(words)


def join_utterance_separated_by(dialogs, separated_by=0.5):
    drefined = []

    lasts = [None for _ in range(2)]
    dic = {'A': 0, 'B': 1}
    for idx, curr in enumerate(dialogs):
        # If current text is entriely contained within the last utterance
        last_current = lasts[dic[curr['speaker']]]
        if last_current is None:
            lasts[dic[curr['speaker']]] = curr
            continue

        # Join utterances from current speaker < separated_by
        if last_current is not None and curr['start'] - last_current['end'] < separated_by:
            last_current['text'] += f" {curr['text']}"
            last_current['end'] = curr['end']
            last_current['wfeats'].extend(curr['wfeats'])

        else:
            drefined.append(last_current)
            lasts[dic[curr['speaker']]] = curr

    drefined.append(lasts[0])
    drefined.append(lasts[1])

    if all([x is None for x in drefined]):
        return []

    drefined.sort(key=lambda x: (x['start'], -x['end']))

    return drefined


def join_utterance_separated_by2(dialogs, separated_by=0.5):
    drefined = []

    last_current = None
    dic = {'A': 0, 'B': 1}

    for idx, curr in enumerate(dialogs):
        # If current text is entriely contained within the last utterance

        if last_current is None:
            last_current = curr
            continue

        # Join utterances from current speaker < separated_by
        if last_current is not None and curr['start'] - last_current['end'] < separated_by and (curr['speaker'] == last_current["speaker"]):
            last_current['text'] += f" {curr['text']}"
            last_current['end'] = curr['end']
            last_current['wfeats'].extend(curr['wfeats'])

        else:
            drefined.append(last_current)
            last_current = curr

    drefined.append(last_current)

    if all([x is None for x in drefined]):
        return []

    drefined.sort(key=lambda x: (x['start'], -x['end']))

    return drefined


def combine_dialogue_without_timings(dialog, separated_by=2, dont_cat=False):
    # [diagA, diagB], diagA = [turn1, turn2, ...], turn1 = [word1, word2, ....]
    combined = dialog[0]
    combined.extend(dialog[1])
    combined.sort(key=lambda key: key['start'])

    if not dont_cat:
        combined = join_utterance_separated_by(
            combined, separated_by=separated_by)
    else:
        combined = join_utterance_separated_by2(
            combined, separated_by=separated_by)
    # [turn1A, turn1B, turn2A, ......] sorted
    return combined



def pairwise_remove_backchannels(dialogs, pre_silence=1, post_silence=1, bc_duration0=1):
    # [turn1A, turn1B, turn2A, ......] sorted
    dialogsA = [x for x in dialogs if x['speaker'] == 'A']
    dialogsB = [x for x in dialogs if x['speaker'] == 'B']

    if len(dialogsA) == 0 or len(dialogsB) == 0:
        return dialogs, []

    assert len(dialogsA) + len(dialogsB) == len(
        dialogs), f"Dialogs not separated by speaker: {len(dialogsA)} + {len(dialogsB)} != {len(dialogs)}; type(dialogs[0])={type(dialogs[0])}"

    def remove_bc_from_channel(dialogs, end_of_utterance_time=0, bc_duration0=1):
        #  [turn1, turn2, ...], turn1 = [word1, word2, ....]
        last_end = 0
        new_dialog = []
        new_bc = []
        for idx, dialog in enumerate(dialogs):
            turn_text = dialog["text"]
            bc_in = check_backchannel(turn_text)

            if bc_in <= 1:
                bc_duration = 100
            elif bc_in <= 2:
                bc_duration = 20
            else:
                bc_duration = bc_duration0

            # Pre silence is 1s, Post silence is 1s and Utterance Length is less than 1
            duration = dialog['end'] - dialog['start']
            pre_sil = dialog['start'] - last_end

            last_end = dialog['end']

            post_sil = end_of_utterance_time - dialog["end"]
            if idx != len(dialogs) - 1:
                post_sil = dialogs[idx+1]['start'] - dialog['end']

            if bc_in > 0 and duration <= bc_duration and pre_sil >= pre_silence and post_sil >= post_silence:
                new_bc.append(dialog)
                continue

            new_dialog.append(dialog)

        assert(len(new_dialog) + len(new_bc) == len(dialogs)), f"BC extraction not equal {len(new_dialog)} + {len(new_bc)} != {len(dialogs)};"
        return new_dialog, new_bc

    end_of_utterance_time = max(dialogsA[-1]['end'], dialogsB[-1]['end'])
    new_dialogsA, new_bcA = remove_bc_from_channel(
        dialogsA, end_of_utterance_time, bc_duration0)
    new_dialogsB, new_bcB = remove_bc_from_channel(
        dialogsB, end_of_utterance_time, bc_duration0)

    new_dialogs = new_dialogsA + new_dialogsB
    new_bc = new_bcA + new_bcB

    new_dialogs.sort(key=lambda key: (key['start'], -key['end']))
    new_bc.sort(key=lambda key: (key['start'], -key['end']))

    # [turn1A, turn1B, turn2A, ......] sorted
    # [bc1a, bc1b, bc2a, bc2b, ....] sorted
    assert(len(new_dialogs) + len(new_bc) == len(dialogs)),  f"BC extraction not equal {len(new_dialogs)} + {len(new_bc)} != {len(dialogs)};"

    return new_dialogs, new_bc


def pairwise_remove_backchannels2(dialogs, pre_silence=1, post_silence=1, bc_duration0=1):
    # [turn1A, turn1B, turn2A, ......] sorted
    dialogsA = [x for x in dialogs if x['speaker'] == 'A']
    dialogsB = [x for x in dialogs if x['speaker'] == 'B']

    if len(dialogsA) == 0 or len(dialogsB) == 0:
        return dialogs, []

    assert len(dialogsA) + len(dialogsB) == len(
        dialogs), f"Dialogs not separated by speaker: {len(dialogsA)} + {len(dialogsB)} != {len(dialogs)}; type(dialogs[0])={type(dialogs[0])}"

    def remove_bc_from_channel(dialogs, end_of_utterance_time=0, bc_duration0=1):
        #  [turn1, turn2, ...], turn1 = [word1, word2, ....]
        last_end = 0
        new_dialog = []
        new_bc = []
        for idx, dialog in enumerate(dialogs):
            turn_text = dialog["text"]
            bc_in = check_backchannel(turn_text)

            if bc_in <= 1:
                bc_duration = 100
            elif bc_in <= 2:
                bc_duration = 20
            else:
                bc_duration = bc_duration0

            # Pre silence is 1s, Post silence is 1s and Utterance Length is less than 1
            duration = dialog['end'] - dialog['start']

            # check front
            check_front = False
            if idx != 0:
                bc_prev = check_backchannel(dialogs[idx-1]['text'] )
            else:
                bc_prev = True
            pre_sil = dialog['start'] - last_end
            if bc_prev:
                check_front = True
            elif pre_sil >= pre_silence:
                check_front = True

            # check back
            check_back = False
            if idx != len(dialogs) - 1:
                bc_next = check_backchannel(dialogs[idx+1]['text'] )
                post_sil = dialogs[idx+1]['start'] - dialog['end']
            else:
                bc_next = True
                post_sil = end_of_utterance_time - dialog["end"]

            if bc_next:
                check_back = True
            elif post_sil >= post_silence:
                check_back = True

            last_end = dialog['end']

            if bc_in > 0 and duration <= bc_duration and check_front and check_back:
                new_bc.append(dialog)
                continue

            new_dialog.append(dialog)

        assert(len(new_dialog) + len(new_bc) == len(dialogs)), f"BC extraction not equal {len(new_dialog)} + {len(new_bc)} != {len(dialogs)};"
        return new_dialog, new_bc

    end_of_utterance_time = max(dialogsA[-1]['end'], dialogsB[-1]['end'])
    new_dialogsA, new_bcA = remove_bc_from_channel(
        dialogsA, end_of_utterance_time, bc_duration0)
    new_dialogsB, new_bcB = remove_bc_from_channel(
        dialogsB, end_of_utterance_time, bc_duration0)

    new_dialogs = new_dialogsA + new_dialogsB
    new_bc = new_bcA + new_bcB

    new_dialogs.sort(key=lambda key: (key['start'], -key['end']))
    new_bc.sort(key=lambda key: (key['start'], -key['end']))

    # [turn1A, turn1B, turn2A, ......] sorted
    # [bc1a, bc1b, bc2a, bc2b, ....] sorted
    assert(len(new_dialogs) + len(new_bc) == len(dialogs)),  f"BC extraction not equal {len(new_dialogs)} + {len(new_bc)} != {len(dialogs)};"

    return new_dialogs, new_bc

def remove_overlaps(dialogs):
    if len(dialogs) == 0:
        return [], []

    drefined = [dialogs[0]]
    overlaps = []
    for idx, curr in enumerate(dialogs[1:]):
        if drefined[-1]["start"] <= curr["start"] <= drefined[-1]["end"]:
            if drefined[-1]["start"] <= curr["end"] <= drefined[-1]["end"]:
                overlaps.append(curr)
                continue

        drefined.append(curr)

    # [turn1A, turn1B, turn2A, ......] sorted
    # [overlap1a, overlap1b, overlap2a, overlap2b, ....] sorted
    assert(len(drefined) + len(overlaps) == len(dialogs)),  f"Overlap extraction not equal {len(drefined)} + {len(overlaps)} != {len(dialogs)};"
    return drefined, overlaps


def combine_consecutive_trps(dialogs, bc=[], overlap=[]):
    # combine the consecutive turn
    temp_dialogs = [x | {"dialog_type": "dialog"} for x in dialogs]
    temp_bc = [x | {"dialog_type": "backchannel"} for x in bc]
    temp_overlaps = [x | {"dialog_type": "overlap"} for x in overlap]
    temp_dialogs = temp_dialogs + temp_bc + temp_overlaps
    temp_dialogs.sort(key=lambda key: (key['start'], -key['end']))
    if len(temp_dialogs) == 0:
        return [], []

    # print()
    # for utt in temp_dialogs:
    #     print(utt["dialog_type"], utt["start"], utt["end"], utt["speaker"], utt["text"] )

    start_idx = 0
    combined_dialogs = []
    combined_backchannels = []

    for i in range(0, len(temp_dialogs)):
        if temp_dialogs[i]['dialog_type'] == "dialog":
            combined_dialogs = [temp_dialogs[i]]
            start_idx = (i + 1)
            break
        elif temp_dialogs[i]['dialog_type'] == "backchannel":
            combined_backchannels.append(temp_dialogs[i])

    assert (len(combined_dialogs) == 1)
    assert (combined_dialogs[0]['dialog_type'] == "dialog")

    num_merge = 0
    for idx in range(start_idx, len(temp_dialogs)):
        # print(idx, temp_dialogs[idx]['dialog_type'])
        if temp_dialogs[idx]['dialog_type'] in ["dialog", "backchannel"] and combined_dialogs[-1]['dialog_type'] == "dialog":
            if combined_dialogs[-1]['speaker'] == temp_dialogs[idx]['speaker'] and temp_dialogs[idx]['start'] - combined_dialogs[-1]['end'] < 5:
                # backchannel is to be combined with the same speaker's utterance
                # print("mergt!!!!!", combined_dialogs[-1]['text'] , temp_dialogs[idx]['text'],  temp_dialogs[idx]['start'])
                combined_dialogs[-1]['text'] += f" {temp_dialogs[idx]['text']}"
                combined_dialogs[-1]['end'] = temp_dialogs[idx]['end']
                # combined_dialogs[-1]['dialog_type'] = temp_dialogs[idx]['dialog_type']
                combined_dialogs[-1]['wfeats'].extend(
                    temp_dialogs[idx]['wfeats'])
                num_merge += 1
            elif temp_dialogs[idx]['dialog_type'] == "backchannel":
                combined_backchannels.append(temp_dialogs[idx])
            else:
                combined_dialogs.append(temp_dialogs[idx])

    assert(len(combined_dialogs) + len(combined_backchannels) + len(temp_overlaps) + num_merge == len(temp_dialogs)),  f"Overlap extraction not equal {len(combined_dialogs)} + {len(combined_backchannels)} + {len(temp_overlaps)} + {num_merge} != {len(temp_dialogs)};"

    return combined_dialogs, combined_backchannels



def combine_consecutive_trps2(dialogs, bc=[], overlap=[]):
    # combine the consecutive turn
    temp_dialogs = [x | {"dialog_type": "dialog"} for x in dialogs]
    temp_bc = [x | {"dialog_type": "backchannel"} for x in bc]
    temp_overlaps = [x | {"dialog_type": "overlap"} for x in overlap]


    temp_dialogs = temp_dialogs + temp_overlaps
    temp_dialogs.sort(key=lambda key: (key['start'], -key['end']))
    # print(len(temp_dialogs), len(dialogs), len(bc), len(overlap))
    if len(temp_dialogs) == 0:
        return [], []

    # print()
    # for utt in temp_dialogs:
    #     print(utt["dialog_type"], utt["start"], utt["end"], utt["speaker"], utt["text"] )

    combined_dialogs = [temp_dialogs[0]]
    combined_backchannels = []

    num_merge = 0
    prev_max_end = combined_dialogs[-1]["end"]

    for idx in range(1, len(temp_dialogs)):
        prev_max_end = max([combined_dialogs[-1]["end"], prev_max_end])
        if combined_dialogs[-1]['speaker'] == temp_dialogs[idx]['speaker'] \
            and temp_dialogs[idx]['start'] - prev_max_end < 5 \
            and (temp_dialogs[idx]['dialog_type'] == "dialog" and combined_dialogs[-1]['dialog_type'] == "dialog"):
            # print("merge!!!", prev_max_end, temp_dialogs[idx]['start'], {temp_dialogs[idx]['text']})
            # backchannel is to be combined with the same speaker's utterance
            combined_dialogs[-1]['text'] += f" {temp_dialogs[idx]['text']}"
            combined_dialogs[-1]['end'] = temp_dialogs[idx]['end']
            combined_dialogs[-1]['wfeats'].extend(
                temp_dialogs[idx]['wfeats'])
            num_merge += 1
        else:
            combined_dialogs.append(temp_dialogs[idx])

    combined_dialogs_new = []
    combined_overlap = []
    for utt in combined_dialogs:
        if utt['dialog_type'] == "dialog":
            combined_dialogs_new.append(utt)
        else:
             combined_overlap.append(utt)

    combined_dialogs = combined_dialogs_new

    assert(len(combined_overlap) == len(temp_overlaps)), f"Overlap number changed after meger!!!!  {len(combined_overlap)} != {len(temp_overlaps)} "
    assert(len(combined_dialogs) + len(combined_overlap) + num_merge == len(temp_dialogs)),  f"Overlap extraction not equal {len(combined_dialogs)} + {len(combined_backchannels)} + {len(temp_overlaps)} + {num_merge} != {len(temp_dialogs)};"

    return combined_dialogs, temp_bc




def separate_by_speaker(dialog_ord):
    new_dialogA = []
    new_dialogB = []
    for idx, turn in enumerate(dialog_ord):
        if turn['speaker'] == 'A':
            new_dialogA.append(turn)
        else:
            new_dialogB.append(turn)

    assert(len(new_dialogA) + len(new_dialogB) == len(dialog_ord)),  f"Speaker split not equal {len(new_dialogA)} + {len(new_dialogB)} != {len(dialog_ord)};"
    return {'speakerA': new_dialogA, 'speakerB': new_dialogB}




### post-process
def handle_channel(
    turnA ,
    curr_turnA ,
    turnB ,
    curr_turnB,
    yield_int_thresh=0.1,
) -> tuple[Turn, Turn]:
    if turnA is None:
        raise ValueError("turnA is None")
    if turnB is None:
        raise ValueError("turnB is None")

    if turnB["start"] < turnA["end"] - yield_int_thresh:
        # turnB overlapped by turnA
        if turnB["end"] < turnA["end"]:
            pass

        curr_turnA.update(
            word=curr_turnA.word + " " + turnA["text"],
            turn_end_type=TurnEndType.YIELD,
            start=turnA["start"],
            end=turnA["end"],
            other_next_id=curr_turnB.id,
        )

        curr_turnB.update(turn_type=TurnType.INTERRUPT, other_prev_id=curr_turnA.id)
    else:
        new_turn_type = (
            TurnType.NORMAL
            if curr_turnA.turn_type == TurnType.NONE
            else curr_turnA.turn_type
        )
        curr_turnA.update(
            word=curr_turnA.word + " " + turnA["text"],
            turn_type=new_turn_type,
            start=turnA["start"],
            end=turnA["end"],
            other_next_id=curr_turnB.id,
        )
        curr_turnB.update(other_prev_id=curr_turnA.id)

    return curr_turnA, curr_turnB




class AlignedProcess():
    def __init__(self,
                tokenizer,
                special_tokens,
                transcriptA,
                transcriptB,
                trp_separated_by = 1,
                pre_silence=0.5,
                post_silence=0.5,
                bc_duration=3,
                yield_int_thresh = 0.2,
                include_backchannels = True,
                include_overlap = True,
                use_same_token = True,
                code_freq = 25,
                dont_cat = False,
                dont_cat2 = False,
                different_silence = False,
                remove_word = False,
                use_end_token = False
            ):

        self.trp_separated_by = trp_separated_by
        self.pre_silence = pre_silence
        self.post_silence = post_silence
        self.bc_duration = bc_duration
        self.yield_int_thresh = yield_int_thresh
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.include_backchannels = include_backchannels
        self.include_overlap = include_overlap
        self.use_same_token = use_same_token
        self.code_freq = code_freq
        self.dont_cat = dont_cat
        self.remove_word = remove_word

        ### split the trasnscriptions into small segments
        if dont_cat2:
            annoA = self.split_trans2(transcriptA, "A")
            annoB = self.split_trans2(transcriptB, "B")
        else:
            annoA = self.split_trans(transcriptA, "A")
            annoB = self.split_trans(transcriptB, "B")

        if dont_cat2:
            self.turn_start = special_tokens.spk_tags[0]
            self.overlap_start = special_tokens.spk_tags[0]
        else:
            self.turn_start = special_tokens.spk_tags[2]
            self.overlap_start = special_tokens.spk_tags[2]

        self.backchannel_start = special_tokens.spk_tags[3]

        self.use_end_token = use_end_token
        self.turn_end_token = None
        # if self.use_end_token:
        self.turn_end_token = special_tokens.spk_tags[4]
        assert(different_silence == False)

        if different_silence:
            self.inter_silence = special_tokens.spk_tags[4]
            self.intra_silence = special_tokens.text_pad
        else:
            self.inter_silence = special_tokens.text_pad
            self.intra_silence = special_tokens.text_pad

        self.max_len = max([annoA[-1]["end"], annoB[-1]["end"]])
        dialog = [annoA, annoB]
        # self.print_diag(annoA)
        # self.print_diag(annoB)
        dialog = self.preprocess_diag(dialog)

        diagA = dialog["dialog"]["speakerA"]
        diagB = dialog["dialog"]["speakerB"]

        self.dialog = dialog

        # turnsA, turnsB = self.postprocess_diag(diagA, diagB)
        # print()
        # self.print_turn(turnsA)
        # self.print_turn(turnsB)
    def print_final_diag(self):
        dialogs = self.dialog["dialog"]["speakerA"]
        backchannel = self.dialog["backchannel"]["speakerA"]
        overlap = self.dialog["overlap"]["speakerA"]

        temp_dialogs = [x | {"dialog_type": "dialog"} for x in dialogs]
        temp_bc = [x | {"dialog_type": "backchannel"} for x in backchannel]
        temp_overlaps = [x | {"dialog_type": "overlap"} for x in overlap]

        dialogs2 = self.dialog["dialog"]["speakerB"]
        backchannel2 = self.dialog["backchannel"]["speakerB"]
        overlap2 = self.dialog["overlap"]["speakerB"]

        temp_dialogs2 = [x | {"dialog_type": "dialog"} for x in dialogs2]
        temp_bc2 = [x | {"dialog_type": "backchannel"} for x in backchannel2]
        temp_overlaps2 = [x | {"dialog_type": "overlap"} for x in overlap2]


        temp_dialogs = temp_dialogs + temp_bc + temp_overlaps+ temp_dialogs2+temp_bc2+temp_overlaps2
        temp_dialogs.sort(key=lambda key: (key['start'], -key['end']))
        print()
        print("final.....", len(temp_dialogs))
        for utt in temp_dialogs:
            print(utt["dialog_type"], utt["start"], utt["end"], utt["speaker"], utt["text"] )

    def preprocess_diag(self, dialog):
        ### combined the small segments if the segments are very close < 0.5s !!!
        dialog = combine_dialogue_without_timings(
            dialog, separated_by=self.trp_separated_by, dont_cat = self.dont_cat
        )
        # print()
        # print("after combining......")
        # self.print_diag(dialog)

        ## find the backchannel from the segments
        if not self.dont_cat:
            dialog, backchannels = pairwise_remove_backchannels(
                dialog, self.pre_silence, self.post_silence, self.bc_duration
            )
        else:
            dialog, backchannels = pairwise_remove_backchannels2(
                dialog, self.pre_silence, self.post_silence, self.bc_duration
            )
        # print()
        # print("Back channeling......")
        # self.print_diag(backchannels)

        ## find the overlap from the segments
        dialog, overlaps = remove_overlaps(dialog)
        # print()
        # print("Overlap......")
        # self.print_diag(overlaps)
        # print()

        ## combine the consecutive chunks and backchannels
        if not self.dont_cat:
            dialog, backchannels = combine_consecutive_trps(
                dialog, backchannels, overlaps
            )
        else:
            dialog, backchannels = combine_consecutive_trps2(
                dialog, backchannels, overlaps
            )
        # print()
        # print("combines......")
        # self.print_diag(dialog)

        new_dialog = {}
        new_dialog["dialog"] = separate_by_speaker(dialog)
        new_dialog["backchannel"] = separate_by_speaker(backchannels)
        new_dialog["overlap"] = separate_by_speaker(overlaps)

        return new_dialog

    def postprocess_diag(
        self, dialogA, dialogB
    ) -> tuple[list[Turn], list[Turn]]:
        """
        Helper function to extract tokens from each speaker's dialog
        """

        turnsA = []
        turnsB = []
        conv_id = 0

        curr_turnA = Turn.default("A", conv_id)
        curr_turnB = Turn.default("B", conv_id)

        dialogs = list(zip_longest(dialogA + [None], dialogB + [None], fillvalue=None))
        # print("dialogs", len(dialogs))
        idxA, idxB = 0, 0
        while idxA < len(dialogs) and idxB < len(dialogs):
            turnA, turnB = dialogs[idxA][0], dialogs[idxB][1]
            # print(idxA, idxB, len(turnsA), len(turnsB))
            if turnA is None and turnB is not None:
                curr_turnB.update(
                    start=turnB["start"],
                    end=turnB["end"],
                    word=curr_turnB.word + " " + turnB["text"],
                    speaker="B",
                    turn_index=idxA + idxB,
                )
                turnsB.append(curr_turnB)
                idxB += 1
                curr_turnB = Turn.default("B", conv_id)
                continue

            if turnB is None and turnA is not None:
                curr_turnA.update(
                    start=turnA["start"],
                    end=turnA["start"],
                    word=curr_turnA.word + " " + turnA["text"],
                    speaker="A",
                    turn_index=idxA + idxB,
                )
                turnsA.append(curr_turnA)
                idxA += 1

                curr_turnA = Turn.default("A", conv_id)
                continue

            if turnA is None and turnB is None:
                idxA += 1
                idxB += 1
                continue

            if turnA["start"] < turnB["start"]:
                curr_turnA, curr_turnB = handle_channel(
                    turnA,
                    curr_turnA,
                    turnB,
                    curr_turnB,
                    yield_int_thresh=self.yield_int_thresh,
                )
                curr_turnA.update(turn_index=idxA + idxB)
                turnsA.append(curr_turnA)
                curr_turnA = Turn.default("A", conv_id)
                idxA += 1
            else:
                curr_turnB, curr_turnA = handle_channel(
                    turnB,
                    curr_turnB,
                    turnA,
                    curr_turnA,
                    yield_int_thresh=self.yield_int_thresh,
                )
                curr_turnB.update(turn_index=idxA + idxB)
                turnsB.append(curr_turnB)
                curr_turnB = Turn.default("B", conv_id)
                idxB += 1

        return turnsA, turnsB

    def tokenize_diag(self, tokens, spk, offset_idx = 0, predict_sot = True):
        # add the text tokens to the first stream for agent side
        # Obtain segments
        if spk == 0:
            spk_id = "speakerA" # speaker A is user
        else:
            spk_id = "speakerB"


        tokens = torch.full_like(tokens, fill_value=self.inter_silence)

        dialogs = self.dialog["dialog"][spk_id]
        backchannel = self.dialog["backchannel"][spk_id]
        overlap = self.dialog["overlap"][spk_id]

        temp_dialogs = [x | {"dialog_type": "dialog"} for x in dialogs]
        temp_bc = [x | {"dialog_type": "backchannel"} for x in backchannel]
        temp_overlaps = [x | {"dialog_type": "overlap"} for x in overlap]
        temp_dialogs = temp_dialogs + temp_bc + temp_overlaps
        temp_dialogs.sort(key=lambda key: (key['start'], -key['end']))
        assert(len(temp_dialogs) > 0), "Error: diaglogue is empty, it cannot be tokenized!!!!!"
        # print()
        # print("-----------------------------")
        # segments = []
        # for n, seg in enumerate(temp_dialogs):
        #     for word in seg['wfeats']:
        #         _seg = word.copy()
        #         _seg['text'] = _seg.pop('word')
        #         segments.append(_seg)
        # print(segments)
        prev_end = -1
        for sno, sent in enumerate(temp_dialogs):
            # print(sent["dialog_type"], sent["start"], sent["end"], sent["speaker"], sent["text"])
            # insert intra silence tokens
            if len(sent["wfeats"]) == 0:
                continue
            turn_start = math.floor(sent["start"] * self.code_freq)
            turn_end = math.floor(sent["wfeats"][-1]["start"] * self.code_freq)
            tokens[0, turn_start:turn_end] = self.intra_silence

            for wn0, word in enumerate(sent["wfeats"]):
                text_tokens = self.tokenizer.encode(word['word'] + ' ', bos=False, eos=False)
                if self.remove_word:
                    text_tokens = [ self.intra_silence for _ in text_tokens]

                if wn0 == 0:
                    if sent["dialog_type"] == "backchannel" and (not self.use_same_token):
                        start_token = self.backchannel_start
                    elif sent["dialog_type"] == "overlap" and (not self.use_same_token):
                        start_token = self.overlap_start
                    else:
                        start_token = self.turn_start
                    text_tokens2 = [start_token]
                    for w in text_tokens:
                        text_tokens2.append(w)
                    text_tokens = text_tokens2

                if wn0 == len(sent["wfeats"]) - 1 and self.use_end_token:
                    text_tokens.append(self.turn_end_token)

                start_idx = math.floor(word['start'] * self.code_freq) - len(text_tokens) - offset_idx

                if start_idx <= prev_end:
                    if start_idx > prev_end - 5: ### cannot be more than 5 tokens
                        start_idx = prev_end + 1
                    else:
                        print("Overlap warning", start_idx, prev_end, word['word'])

                for i in range(len(text_tokens)):
                    idx = start_idx + i
                    if idx >= 0 and idx < tokens.shape[-1]:
                        tokens[0, idx] = text_tokens[i]

                prev_end = start_idx + len(text_tokens) - 1
        return tokens

    def tokenize_user_side(self, tokens, spk, offset_idx = 0, predict_sot = False, predict_eot = True, remove = True):
        # add the text tokens to the first stream for user side
        # Obtain segments
        if spk == 0:
            spk_id = "speakerA" # speaker A is user
        else:
            spk_id = "speakerB" # speaker B is agent


        tokens = torch.full_like(tokens, fill_value=self.inter_silence)

        dialogs = self.dialog["dialog"][spk_id]
        backchannel = self.dialog["backchannel"][spk_id]
        overlap = self.dialog["overlap"][spk_id]

        temp_dialogs = [x | {"dialog_type": "dialog"} for x in dialogs]
        temp_bc = [x | {"dialog_type": "backchannel"} for x in backchannel]
        temp_overlaps = [x | {"dialog_type": "overlap"} for x in overlap]
        temp_dialogs = temp_dialogs + temp_bc + temp_overlaps
        temp_dialogs.sort(key=lambda key: (key['start'], -key['end']))
        assert(len(temp_dialogs) > 0), "Error: diaglogue is empty, it cannot be tokenized!!!!!"
        # print()
        # print("-----------------------------")
        # segments = []
        # for n, seg in enumerate(temp_dialogs):
        #     for word in seg['wfeats']:
        #         _seg = word.copy()
        #         _seg['text'] = _seg.pop('word')
        #         segments.append(_seg)
        # print(segments)
        prev_end = -1
        for sno, sent in enumerate(temp_dialogs):
            # print(sent["dialog_type"], sent["start"], sent["end"], sent["speaker"], sent["text"])
            # insert intra silence tokens
            if len(sent["wfeats"]) == 0:
                continue
            turn_start = math.floor(sent["start"] * self.code_freq)
            turn_end = math.floor(sent["wfeats"][-1]["start"] * self.code_freq)
            tokens[0, turn_start:turn_end] = self.intra_silence

            for wn0, word in enumerate(sent["wfeats"]):
                text_tokens = self.tokenizer.encode(word['word'] + ' ', bos=False, eos=False)
                if remove:
                    text_tokens = [ self.intra_silence for _ in text_tokens]

                if wn0 == 0 and predict_sot:
                    start_token = self.turn_end_token
                    text_tokens2 = [start_token]
                    for w in text_tokens:
                        text_tokens2.append(w)
                    text_tokens = text_tokens2

                if wn0 == len(sent["wfeats"]) - 1 and predict_eot:
                    # print(sent["dialog_type"], word)
                    if sent["dialog_type"] == "backchannel" and (not self.use_same_token):
                        end_token = self.backchannel_start
                    elif sent["dialog_type"] == "overlap" and (not self.use_same_token):
                        end_token = self.overlap_start
                    else:
                        end_token = self.turn_start

                    text_tokens.append(end_token)

                # start_idx = math.floor(word['start'] * self.code_freq) - len(text_tokens) - offset_idx
                start_idx = math.ceil(word['end'] * self.code_freq)

                if start_idx <= prev_end:
                    if start_idx > prev_end - 5: ### cannot be more than 5 tokens
                        start_idx = prev_end + 1
                    else:
                        print("Overlap warning", start_idx, prev_end, word['word'])

                for i in range(len(text_tokens)):
                    idx = start_idx + i
                    if idx >= 0 and idx < tokens.shape[-1]:
                        tokens[0, idx] = text_tokens[i]

                prev_end = start_idx + len(text_tokens) - 1
        return tokens

    def tokenize_diag_vad(self, tokens, spk, offset_idx = 0):
        # add the text tokens to the first stream
        # Obtain segments
        if spk == 0:
            spk_id = "speakerA"
        else:
            spk_id = "speakerB"


        tokens = torch.full_like(tokens, fill_value=self.special_tokens.text_pad)

        dialogs = self.dialog["dialog"][spk_id]
        backchannel = self.dialog["backchannel"][spk_id]
        overlap = self.dialog["overlap"][spk_id]

        temp_dialogs = [x | {"dialog_type": "dialog"} for x in dialogs]
        temp_bc = [x | {"dialog_type": "backchannel"} for x in backchannel]
        temp_overlaps = [x | {"dialog_type": "overlap"} for x in overlap]
        temp_dialogs = temp_dialogs + temp_bc + temp_overlaps
        temp_dialogs.sort(key=lambda key: (key['start'], -key['end']))
        assert(len(temp_dialogs) > 0), "Error: diaglogue is empty, it cannot be tokenized!!!!!"

        for sno, sent in enumerate(temp_dialogs):
            # insert intra silence tokens
            # for w in sent["wfeats"]:
            #     print(sno, w["start"], w["end"], w["word"])
            if len(sent["wfeats"]) == 0:
                continue
            for wn0, word in enumerate(sent["wfeats"]):
                # print(sno, word["start"], word["end"], word["word"])
                start_idx = math.floor((word['start'] - 0.4)* self.code_freq)
                end_idx = math.ceil((word['end'] +  0.4) * self.code_freq)

                for idx in range(start_idx, end_idx):
                    if idx >= 0 and idx < tokens.shape[-1]:
                        tokens[0, idx] = self.turn_start
        return tokens

    def print_diag(self, diag):
        for utt in diag:
            print(utt["start"], utt["end"], utt["speaker"], utt["text"])

    def print_turn(self, turns):
        for turn in turns:
            print(turn.turn_index, turn.speaker, turn.start, turn.end, turn.speaker, turn.word, turn.turn_type, turn.turn_end_type)

    def split_trans(self, transcript, speaker):
        anno = []
        for n, seg in enumerate(transcript):
            key = f"{speaker}{n}"
            segments = []
            prev_end = None
            for word in seg['words']:
                _seg = word.copy()
                _seg['word'] = _seg.pop('word')
                # if prev_end is not None:
                #     print(_seg['start'] - prev_end, _seg['word'])
                # prev_end = _seg['end']
                _seg['speaker'] = speaker
                segments.append(_seg)

            segments = self.fix_invalid_segments(segments)


            if len(segments) == 0:
                continue

            anno.append({
                "start": segments[0]["start"],
                "end": segments[-1]["end"],
                "wfeats": segments,
                "text": " ".join([x["word"] for x in segments]),
                "speaker": speaker
            })

        # Obtain valid segments

        return anno



    def split_trans2(self, transcript, speaker):
        anno = []
        for n, seg in enumerate(transcript):
            key = f"{speaker}{n}"
            segments = []
            prev_end = None
            for word in seg['words']:
                _seg = word.copy()
                _seg['word'] = _seg.pop('word')
                # if prev_end is not None:
                #     print(_seg['start'] - prev_end, _seg['word'])
                # prev_end = _seg['end']
                _seg['speaker'] = speaker
                segments.append(_seg)

            segments = self.fix_invalid_segments(segments)
            if (len(segments) == 0):
                continue

            new_segments = []
            temp_segs = []
            # prev_end = 9999
            # for wi, w in enumerate(segments):
            #     if (w["end"] - w["start"]) > 6: ## large than 6 second it is impossible
            #         print("Warning!!!! whisper aligment inaccurate!!!!", wi, len(segments), w)
            #         if wi == 0:
            #             w["start"] = w["end"] - 0.2
            #             temp_segs.append(w)
            #         elif wi == len(segments) - 1:
            #             w["end"] = w["start"] + 0.2
            #             temp_segs.append(w)
            #         else:
            #             w["end"] = w["start"] + 0.2
            #             temp_segs.append(w)
            #             new_segments.append(temp_segs)
            #             temp_segs = []
            #     elif (w["start"] - prev_end) > 6:
            #         print("Warning!!!! whisper aligment inaccurate!!!!", wi, len(segments), prev_end, w, segments[wi - 1])
            #         new_segments.append(temp_segs)
            #         temp_segs = []
            #         temp_segs.append(w)
            #     else:
            #         temp_segs.append(w)
            #     prev_end = w["end"]
            for wi in range(len(segments)):
                w = segments[wi]
                if (w["end"] - w["start"]) > 6: ## large than 7 second it is impossible
                    # print("Warning!!!! whisper aligment inaccurate!!!!", wi, len(segments), w)
                    if wi == 0:
                        w["start"] = w["end"] - 0.2
                        temp_segs.append(w)
                    elif wi == len(segments) - 1:
                        w["end"] = w["start"] + 0.2
                        temp_segs.append(w)
                    else:
                        w["end"] = w["start"] + 0.2
                        temp_segs.append(w)
                        new_segments.append(temp_segs)
                        temp_segs = []
                    continue

                if wi == len(segments) - 1:
                    temp_segs.append(w)
                    continue

                next_w = segments[wi + 1]
                if (next_w["start"] - w["end"]) > 6:
                    # print("Warning!!!! whisper aligment inaccurate!!!!", wi, len(segments), w, next_w)
                    if wi == 0:
                        w["start"] = next_w["start"] - 0.2
                        w["end"] = next_w["start"]
                        temp_segs.append(w)
                    elif w["score"] < 0.1:
                        w["start"] = next_w["start"] - 0.2
                        w["end"] = next_w["start"]
                        new_segments.append(temp_segs)
                        temp_segs = []
                        temp_segs.append(w)
                    else:
                        temp_segs.append(w)
                        new_segments.append(temp_segs)
                        temp_segs = []
                else:
                    temp_segs.append(w)


            if len(temp_segs) > 0:
                new_segments.append(temp_segs)

            for segments in new_segments:
                if len(segments) == 0:
                    continue

                anno.append({
                    "start": segments[0]["start"],
                    "end": segments[-1]["end"],
                    "wfeats": segments,
                    "text": " ".join([x["word"] for x in segments]),
                    "speaker": speaker
                })

        # for a in anno:
        #     print(a["wfeats"])
        # Obtain valid segments

        return anno


    def fix_invalid_segments(self, segments):
        _segments = []
        prev_seg_valid = True

        ### first fix sth when the start or end is not all empty
        for _, seg in enumerate(segments):
            if len(_segments) == 0:
                # Continue until we find first valid segment
                if _is_valid_seg(seg):
                    _seg = seg.copy()
                    _segments.append(_seg)
                continue

            if _is_valid_seg(seg):
                if not prev_seg_valid:
                    _segments[-1]['end'] = seg['start']
                _seg = seg.copy()
                _segments.append(_seg)
                prev_seg_valid = True
            else:
                _segments[-1]['word'] = ' '.join(
                    [_segments[-1]['word'], seg['word']]
                )
                prev_seg_valid = False

        return _segments

    def save_diag(self, file_path):
        max_len = int((self.max_len  + 4)* self.code_freq)
        tokens1 = torch.full((18, max_len), fill_value=self.special_tokens.null_id)
        tokens2 = torch.full((18, max_len), fill_value=self.special_tokens.null_id)

        tokens1 = self.tokenize_diag(tokens1, 0, offset_idx = 0)
        tokens2 = self.tokenize_diag(tokens2, 1, offset_idx = 0)

        stream1 = []
        stream2 = []
        for j in range(max_len):
            w1 = self.tokenizer.decode(tokens1[0, j:j+1].long().tolist())
            w2 = self.tokenizer.decode(tokens2[0, j:j+1].long().tolist())
            stream1.append(w1)
            stream2.append(w2)

        def chunk_streams(s1, s2, chunk_size=100):
            """Yield 2-row DataFrames with chunk_size columns from two token streams."""
            for i in range(0, len(s1), chunk_size):
                chunk1 = s1[i:i+chunk_size]
                chunk2 = s2[i:i+chunk_size]

                # Pad if last chunk is shorter than chunk_size
                if len(chunk1) < chunk_size:
                    chunk1 += [''] * (chunk_size - len(chunk1))
                    chunk2 += [''] * (chunk_size - len(chunk2))

                chunk3 = [" " for _ in chunk1]
                df = pd.DataFrame([chunk1, chunk2, chunk3])
                df.index = ['SpeakerA', 'SpeakerB', ' ']
                yield df
        # Example token streams
        # stream1 = [f"t1_{i}" for i in range(250)]
        # stream2 = [f"t2_{i}" for i in range(250)]
        # Display each chunk
        chunk_dfs = list(chunk_streams(stream1, stream2, chunk_size=200))
        # Concatenate all chunks vertically (stack rows)
        # This will create a DataFrame with 2 * number_of_chunks rows and 100 columns
        concatenated_df = pd.concat(chunk_dfs, axis=0)
        # concatenated_df.reset_index(drop=True, inplace=True)
        # Save to CSV
        concatenated_df.to_csv(file_path, index=True)



def _encode_speech_with_turntaking(
    tokenizer: Llama3Tokenizer, special_tokens: SLMCodecSpecialTokens,
    processor: AlignedProcess, audio_codes: list, speech_codes: list,
    num_audio_codebooks: int, semantic_stream: bool = True,
    num_speech_codebooks: int = 1, code_freq: int = 25,
    delay_codes: bool = False
):
    if num_speech_codebooks != 1:
        raise NotImplementedError

    # Audio codes - only functional when delay_codes = True
    a_codes = torch.tensor(audio_codes)[:num_audio_codebooks]
    audio_silence = AUDIO_SILENCE_CODES.repeat( # - 1s silence
        (1, code_freq)
    )[:num_audio_codebooks] # [num_codebooks, code_freq]
    a_codes = torch.cat([audio_silence, a_codes], dim=-1)
    for k in range(num_audio_codebooks):
        if delay_codes:
            a_codes[k] = torch.roll(a_codes[k], k + 1) # additional +1 means current audio condition on the previous speech tokens
        a_codes[k] += special_tokens.codec_offsets[k]
    a_codes = a_codes[:, code_freq:]
    # Silence
    _silence = AUDIO_SILENCE_CODES.repeat( # 2s silence
        (1, 2 * code_freq)
    )[:num_audio_codebooks] # [num_codebooks, code_freq]
    for k in range(num_audio_codebooks):
        if delay_codes:
            _silence[k] = torch.roll(_silence[k], k + 1)
        _silence[k] += special_tokens.codec_offsets[k]
    _silence = _silence[:, code_freq:]
    _silence = torch.cat([
        torch.full_like(_silence[:2], fill_value=special_tokens.null_id),
        _silence
    ], dim=0)
    # print("_silence", _silence.shape)
    if num_audio_codebooks < 16:
        a_codes = F.pad(a_codes, (0, 0, 0, 16 - num_audio_codebooks), mode='constant', value=special_tokens.null_id)
        _silence = F.pad(_silence, (0, 0, 0, 16 - num_audio_codebooks), mode='constant', value=special_tokens.null_id)

    # Speech codes
    s_codes = torch.tensor(speech_codes) + special_tokens.speech_offset
    s_pad = a_codes.shape[-1] - s_codes.shape[-1]
    # print(s_pad)
    if s_pad > 0:
        s_codes = torch.cat([
            s_codes,
            torch.full_like(s_codes[:s_pad], fill_value=s_codes[-1])
        ], dim=-1)
    else:
        s_codes = s_codes[:a_codes.shape[-1]]
    s_codes = s_codes.unsqueeze(0)
    # print("s_codes", s_codes.shape)

    if not semantic_stream:
        s_codes = torch.full_like(s_codes, fill_value=special_tokens.null_id)

    text_tokens = torch.full_like(a_codes[0:1], fill_value=special_tokens.text_pad)
    tokens = torch.cat([text_tokens, s_codes, a_codes], dim=0)

    # agent
    # The end of the Text corresponding word_i is present $delay_speech$ timesteps before the start of first speech token of word_{i}
    # decide padding or remove the beginning of speech/audio stream
    tokens[0:1] = processor.tokenize_diag(tokens[0:1], 1, offset_idx = 0)
    tokens = tokens.permute(1, 0)

    return tokens


def _encode_user_with_turntaking(
    tokenizer: Llama3Tokenizer, special_tokens: SLMCodecSpecialTokens,
    processor: AlignedProcess, audio_codes: list, speech_codes: list,
    num_audio_codebooks: int, semantic_stream: bool = True,
    num_speech_codebooks: int = 1, code_freq: int = 25,
    delay_codes: bool = False, predict_sot = False, predict_eot = True, remove = True
):
    if num_speech_codebooks != 1:
        raise NotImplementedError

    # Audio codes - only functional when delay_codes = True
    a_codes = torch.tensor(audio_codes)[:num_audio_codebooks]
    audio_silence = AUDIO_SILENCE_CODES.repeat( # - 1s silence
        (1, code_freq)
    )[:num_audio_codebooks] # [num_codebooks, code_freq]
    a_codes = torch.cat([audio_silence, a_codes], dim=-1)
    for k in range(num_audio_codebooks):
        if delay_codes:
            a_codes[k] = torch.roll(a_codes[k], k + 1) # additional +1 means current audio condition on the previous speech tokens
        a_codes[k] += special_tokens.codec_offsets[k]
    a_codes = a_codes[:, code_freq:]
    # Silence
    _silence = AUDIO_SILENCE_CODES.repeat( # 2s silence
        (1, 2 * code_freq)
    )[:num_audio_codebooks] # [num_codebooks, code_freq]
    for k in range(num_audio_codebooks):
        if delay_codes:
            _silence[k] = torch.roll(_silence[k], k + 1)
        _silence[k] += special_tokens.codec_offsets[k]
    _silence = _silence[:, code_freq:]
    _silence = torch.cat([
        torch.full_like(_silence[:2], fill_value=special_tokens.null_id),
        _silence
    ], dim=0)
    # print("_silence", _silence.shape)

    # Speech codes
    s_codes = torch.tensor(speech_codes) + special_tokens.speech_offset
    s_pad = a_codes.shape[-1] - s_codes.shape[-1]
    # print(s_pad)
    if s_pad > 0:
        s_codes = torch.cat([
            s_codes,
            torch.full_like(s_codes[:s_pad], fill_value=s_codes[-1])
        ], dim=-1)
    else:
        s_codes = s_codes[:a_codes.shape[-1]]
    s_codes = s_codes.unsqueeze(0)

    if not semantic_stream:
        s_codes = torch.full_like(s_codes, fill_value=special_tokens.null_id)

    text_tokens = torch.full_like(a_codes[0:1], fill_value=special_tokens.text_pad)
    tokens = torch.cat([text_tokens, s_codes, a_codes], dim=0)

    # agent
    # The end of the Text corresponding word_i is present $delay_speech$ timesteps before the start of first speech token of word_{i}
    # decide padding or remove the beginning of speech/audio stream
    tokens[0:1] = processor.tokenize_user_side(tokens[0:1], 0, offset_idx = 0, predict_sot = predict_sot, predict_eot = predict_eot, remove = remove)
    tokens = tokens.permute(1, 0)

    return tokens

def _encode_speech_with_vad(
    tokenizer: Llama3Tokenizer, special_tokens: SLMCodecSpecialTokens,
    processor: AlignedProcess, audio_codes: list, speech_codes: list,
    num_audio_codebooks: int, semantic_stream: bool = True,
    num_speech_codebooks: int = 1, code_freq: int = 25,
    delay_codes: bool = False
):
    if num_speech_codebooks != 1:
        raise NotImplementedError

    # Audio codes - only functional when delay_codes = True
    a_codes = torch.tensor(audio_codes)[:num_audio_codebooks]
    audio_silence = AUDIO_SILENCE_CODES.repeat( # - 1s silence
        (1, code_freq)
    )[:num_audio_codebooks] # [num_codebooks, code_freq]
    a_codes = torch.cat([audio_silence, a_codes], dim=-1)
    for k in range(num_audio_codebooks):
        if delay_codes:
            a_codes[k] = torch.roll(a_codes[k], k + 1) # additional +1 means current audio condition on the previous speech tokens
        a_codes[k] += special_tokens.codec_offsets[k]
    a_codes = a_codes[:, code_freq:]
    # Silence
    _silence = AUDIO_SILENCE_CODES.repeat( # 2s silence
        (1, 2 * code_freq)
    )[:num_audio_codebooks] # [num_codebooks, code_freq]
    for k in range(num_audio_codebooks):
        if delay_codes:
            _silence[k] = torch.roll(_silence[k], k + 1)
        _silence[k] += special_tokens.codec_offsets[k]
    _silence = _silence[:, code_freq:]
    _silence = torch.cat([
        torch.full_like(_silence[:2], fill_value=special_tokens.null_id),
        _silence
    ], dim=0)
    # print("_silence", _silence.shape)

    # Speech codes
    s_codes = torch.tensor(speech_codes) + special_tokens.speech_offset
    s_pad = a_codes.shape[-1] - s_codes.shape[-1]
    # print(s_pad)
    if s_pad > 0:
        s_codes = torch.cat([
            s_codes,
            torch.full_like(s_codes[:s_pad], fill_value=s_codes[-1])
        ], dim=-1)
    else:
        s_codes = s_codes[:a_codes.shape[-1]]
    s_codes = s_codes.unsqueeze(0)
    # print("s_codes", s_codes.shape)

    if not semantic_stream:
        s_codes = torch.full_like(s_codes, fill_value=special_tokens.null_id)

    text_tokens = torch.full_like(a_codes[0:1], fill_value=special_tokens.text_pad)
    tokens = torch.cat([text_tokens, s_codes, a_codes], dim=0)

    # agent
    # The end of the Text corresponding word_i is present $delay_speech$ timesteps before the start of first speech token of word_{i}
    # decide padding or remove the beginning of speech/audio stream
    tokens[0:1] = processor.tokenize_diag_vad(tokens[0:1], 0, offset_idx = 0)
    tokens = tokens.permute(1, 0)

    return tokens
