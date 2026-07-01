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
import string

from itertools import zip_longest
from .diag_base import Turn, TurnEndType, TurnType


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
        if seg['start'] > seg['end']:
            return False
        return True
    else:
        return False

    return True



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







class AlignedProcess():
    def __init__(self,
                transcriptA,
                transcriptB,
                speakerA,
                speakerB,
                trp_separated_by = 1,
                pre_silence=0.5,
                post_silence=0.5,
                bc_duration=3,
                yield_int_thresh = 0.2,
                include_backchannels = True,
                include_overlap = True,
                interval_character = ' ',
                turn_gap_threshold = 6,
            ):
        self.interval_character = interval_character
        self.turn_gap_threshold = turn_gap_threshold
        
        self.trp_separated_by = trp_separated_by
        self.pre_silence = pre_silence
        self.post_silence = post_silence
        self.bc_duration = bc_duration
        self.yield_int_thresh = yield_int_thresh

        self.include_backchannels = include_backchannels
        self.include_overlap = include_overlap
        self.dont_cat = False

        self.speakerA = speakerA
        self.speakerB = speakerB

        ### split the trasnscriptions into small segments
        annoA = self.split_trans(transcriptA, "A")
        annoB = self.split_trans(transcriptB, "B")
        assert len(annoA) > 0 or len(annoB) > 0, "Empty annotation at A and B"
        durations = []
        if len(annoA) > 0:
            durations.append(annoA[-1]["end"])
        if len(annoB) > 0:
            durations.append(annoB[-1]["end"])
        self.max_len = max(durations)
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
    def get_parsed_dialog(self):
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


        temp_dialogs = temp_dialogs + temp_bc + temp_overlaps
        temp_dialogs.sort(key=lambda key: (key['start'], -key['end']))
        for idx, utt in enumerate(temp_dialogs):
            if utt['speaker'] == 'A':
                temp_dialogs[idx]['speaker'] = self.speakerA
            elif utt['speaker'] == 'B':
                temp_dialogs[idx]['speaker'] = self.speakerB
            else:
                raise ValueError(f"Unknown speaker {utt['speaker']}")

        temp_dialogs2 = temp_dialogs2 + temp_bc2 + temp_overlaps2
        temp_dialogs2.sort(key=lambda key: (key['start'], -key['end']))
        for idx, utt in enumerate(temp_dialogs2):
            if utt['speaker'] == 'A':
                temp_dialogs2[idx]['speaker'] = self.speakerA
            elif utt['speaker'] == 'B':
                temp_dialogs2[idx]['speaker'] = self.speakerB
            else:
                raise ValueError(f"Unknown speaker {utt['speaker']}")

        return temp_dialogs, temp_dialogs2

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

    def print_diag(self, diag):
        for utt in diag:
            print(utt["start"], utt["end"], utt["speaker"], utt["text"])

    def print_turn(self, turns):
        for turn in turns:
            print(turn.turn_index, turn.speaker, turn.start, turn.end, turn.speaker, turn.word, turn.turn_type, turn.turn_end_type)


    def split_trans(self, transcript, speaker):
        anno = []
        for n, seg in enumerate(transcript):
            key = f"{speaker}_{n}"
            segments = []
            prev_end = None
            for word in seg['words']:
                _seg = word.copy()
                _seg['word'] = _seg.pop('word')
                _seg['speaker'] = speaker
                segments.append(_seg)

            segments = self.fix_invalid_segments(segments)
            if (len(segments) == 0):
                continue

            new_segments = []
            temp_segs = []

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
                # split to turn when it is larger than 6 second
                if (next_w["start"] - w["end"]) > self.turn_gap_threshold :
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
                    "text": self.interval_character.join([x["word"] for x in segments]),
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
                _segments[-1]['word'] = self.interval_character.join(
                    [_segments[-1]['word'], seg['word']]
                )
                prev_seg_valid = False

        return _segments


# ---------------------------------------------------------------------------
# Multi-speaker helper functions (generalizations of the 2-speaker helpers)
# ---------------------------------------------------------------------------

def join_utterance_separated_by_multi(dialogs, speaker_labels, separated_by=0.5):
    """Merge consecutive utterances from the *same* speaker when the gap < separated_by.

    This is the N-speaker generalization of ``join_utterance_separated_by``.
    Instead of tracking only two "last seen" slots (A, B), we keep one slot per
    speaker label so the merging logic works for 1–8 speakers identically.

    Args:
        dialogs: list of utterance dicts, each with keys
                 'start', 'end', 'text', 'wfeats', 'speaker'.
        speaker_labels: list of all possible speaker label strings, e.g.
                        ["S0", "S1", "S2"].
        separated_by: maximum gap (seconds) between two consecutive utterances
                      of the same speaker for them to be merged.

    Returns:
        A new list of utterance dicts sorted by (start, -end).
    """
    # One "last utterance" buffer per speaker, keyed by speaker label.
    lasts = {label: None for label in speaker_labels}
    drefined = []

    for curr in dialogs:
        spk = curr['speaker']
        last = lasts[spk]

        # First utterance we see for this speaker – just buffer it.
        if last is None:
            lasts[spk] = curr
            continue

        # If the gap is small enough, merge into the buffered utterance.
        if curr['start'] - last['end'] < separated_by:
            last['text'] += f" {curr['text']}"
            last['end'] = curr['end']
            last['wfeats'].extend(curr['wfeats'])
        else:
            # Gap too large – flush the buffer and start a new one.
            drefined.append(last)
            lasts[spk] = curr

    # Flush any remaining buffered utterances.
    for last in lasts.values():
        if last is not None:
            drefined.append(last)

    if all(x is None for x in drefined):
        return []

    drefined.sort(key=lambda x: (x['start'], -x['end']))
    return drefined


def join_utterance_separated_by2_multi(dialogs, separated_by=0.5):
    """Merge consecutive utterances when same speaker AND gap < separated_by.

    Unlike ``join_utterance_separated_by_multi`` this version does NOT keep
    per-speaker buffers.  It simply walks the sorted list and merges adjacent
    entries that share a speaker and are close enough.  This is the N-speaker
    version of ``join_utterance_separated_by2``.

    Args:
        dialogs: sorted list of utterance dicts.
        separated_by: max gap (seconds) for merging.

    Returns:
        Sorted list of (possibly merged) utterance dicts.
    """
    drefined = []
    last_current = None

    for curr in dialogs:
        if last_current is None:
            last_current = curr
            continue

        # Merge if same speaker and gap small enough.
        if (curr['speaker'] == last_current['speaker']
                and curr['start'] - last_current['end'] < separated_by):
            last_current['text'] += f" {curr['text']}"
            last_current['end'] = curr['end']
            last_current['wfeats'].extend(curr['wfeats'])
        else:
            drefined.append(last_current)
            last_current = curr

    if last_current is not None:
        drefined.append(last_current)

    if all(x is None for x in drefined):
        return []

    drefined.sort(key=lambda x: (x['start'], -x['end']))
    return drefined


def combine_dialogue_without_timings_multi(dialog_lists, speaker_labels,
                                           separated_by=2, dont_cat=False):
    """Combine N per-speaker segment lists into one sorted turn list.

    Generalization of ``combine_dialogue_without_timings`` which only handled
    two lists (dialog[0], dialog[1]).

    Args:
        dialog_lists: list of N lists, one per speaker, each containing
                      utterance dicts.
        speaker_labels: corresponding speaker label strings.
        separated_by: gap threshold forwarded to the join function.
        dont_cat: if True, use the simpler sequential merge variant.

    Returns:
        A single sorted list of (possibly merged) utterance dicts.
    """
    # Flatten all per-speaker lists into one.
    combined = []
    for spk_list in dialog_lists:
        combined.extend(spk_list)
    combined.sort(key=lambda k: k['start'])

    # Merge close utterances from the same speaker.
    if not dont_cat:
        combined = join_utterance_separated_by_multi(
            combined, speaker_labels, separated_by=separated_by)
    else:
        combined = join_utterance_separated_by2_multi(
            combined, separated_by=separated_by)

    return combined


def pairwise_remove_backchannels_multi(dialogs, speaker_labels,
                                       pre_silence=1, post_silence=1,
                                       bc_duration0=1):
    """Remove backchannel utterances, generalized to N speakers.

    The original ``pairwise_remove_backchannels`` hard-coded speakers 'A'/'B'.
    This version partitions by arbitrary speaker labels, applies the same
    backchannel detection per channel, then recombines.

    Args:
        dialogs: sorted list of utterance dicts.
        speaker_labels: list of speaker label strings.
        pre_silence / post_silence: silence thresholds for BC detection.
        bc_duration0: maximum duration for a backchannel candidate.

    Returns:
        (non_bc_dialogs, bc_dialogs) – both sorted by (start, -end).
    """
    # Partition utterances by speaker.
    per_speaker = {label: [] for label in speaker_labels}
    for utt in dialogs:
        per_speaker[utt['speaker']].append(utt)

    # Skip if any speaker has zero utterances (nothing to compare against).
    non_empty = [label for label, utts in per_speaker.items() if len(utts) > 0]
    if len(non_empty) <= 1:
        return dialogs, []

    def remove_bc_from_channel(channel_dialogs, end_of_utterance_time, bc_dur):
        """Identify and separate backchannel utterances in a single channel."""
        last_end = 0
        new_dialog = []
        new_bc = []
        for idx, dialog in enumerate(channel_dialogs):
            turn_text = dialog["text"]
            bc_in = check_backchannel(turn_text)

            # Adaptive duration threshold based on word count.
            if bc_in <= 1:
                bc_duration = 100
            elif bc_in <= 2:
                bc_duration = 20
            else:
                bc_duration = bc_dur

            duration = dialog['end'] - dialog['start']
            pre_sil = dialog['start'] - last_end
            last_end = dialog['end']

            post_sil = end_of_utterance_time - dialog["end"]
            if idx != len(channel_dialogs) - 1:
                post_sil = channel_dialogs[idx + 1]['start'] - dialog['end']

            if (bc_in > 0 and duration <= bc_duration
                    and pre_sil >= pre_silence and post_sil >= post_silence):
                new_bc.append(dialog)
                continue

            new_dialog.append(dialog)

        assert len(new_dialog) + len(new_bc) == len(channel_dialogs)
        return new_dialog, new_bc

    # Find the global end-of-utterance time across all speakers.
    end_time = max(utts[-1]['end'] for utts in per_speaker.values() if utts)

    all_dialog = []
    all_bc = []
    for label in speaker_labels:
        if len(per_speaker[label]) == 0:
            continue
        d, b = remove_bc_from_channel(per_speaker[label], end_time, bc_duration0)
        all_dialog.extend(d)
        all_bc.extend(b)

    all_dialog.sort(key=lambda k: (k['start'], k['end']))
    all_bc.sort(key=lambda k: (k['start'], k['end']))

    assert len(all_dialog) + len(all_bc) == len(dialogs)
    return all_dialog, all_bc


def pairwise_remove_backchannels2_multi(dialogs, speaker_labels,
                                        pre_silence=1, post_silence=1,
                                        bc_duration0=1):
    """Variant of backchannel removal with neighbour-aware silence checks.

    Mirrors ``pairwise_remove_backchannels2`` but works with N speakers.
    The difference from the non-"2" version is that we also consider whether
    the neighbouring utterances are themselves backchannels when deciding the
    silence thresholds.

    Args / Returns: same as ``pairwise_remove_backchannels_multi``.
    """
    per_speaker = {label: [] for label in speaker_labels}
    for utt in dialogs:
        per_speaker[utt['speaker']].append(utt)

    non_empty = [l for l, u in per_speaker.items() if len(u) > 0]
    if len(non_empty) <= 1:
        return dialogs, []

    def remove_bc_from_channel(channel_dialogs, end_of_utterance_time, bc_dur):
        last_end = 0
        new_dialog = []
        new_bc = []
        for idx, dialog in enumerate(channel_dialogs):
            turn_text = dialog["text"]
            bc_in = check_backchannel(turn_text)

            if bc_in <= 1:
                bc_duration = 100
            elif bc_in <= 2:
                bc_duration = 20
            else:
                bc_duration = bc_dur

            duration = dialog['end'] - dialog['start']

            # Front check: relax silence requirement if previous utt is a BC.
            check_front = False
            bc_prev = check_backchannel(channel_dialogs[idx - 1]['text']) if idx != 0 else True
            pre_sil = dialog['start'] - last_end
            if bc_prev or pre_sil >= pre_silence:
                check_front = True

            # Back check: relax silence requirement if next utt is a BC.
            check_back = False
            if idx != len(channel_dialogs) - 1:
                bc_next = check_backchannel(channel_dialogs[idx + 1]['text'])
                post_sil = channel_dialogs[idx + 1]['start'] - dialog['end']
            else:
                bc_next = True
                post_sil = end_of_utterance_time - dialog["end"]

            if bc_next or post_sil >= post_silence:
                check_back = True

            last_end = dialog['end']

            if bc_in > 0 and duration <= bc_duration and check_front and check_back:
                new_bc.append(dialog)
                continue

            new_dialog.append(dialog)

        assert len(new_dialog) + len(new_bc) == len(channel_dialogs)
        return new_dialog, new_bc

    end_time = max(u[-1]['end'] for u in per_speaker.values() if u)

    all_dialog = []
    all_bc = []
    for label in speaker_labels:
        if not per_speaker[label]:
            continue
        d, b = remove_bc_from_channel(per_speaker[label], end_time, bc_duration0)
        all_dialog.extend(d)
        all_bc.extend(b)

    all_dialog.sort(key=lambda k: (k['start'], k['end']))
    all_bc.sort(key=lambda k: (k['start'], k['end']))

    assert len(all_dialog) + len(all_bc) == len(dialogs)
    return all_dialog, all_bc


def separate_by_speaker_multi(dialog_ord, speaker_labels):
    """Split a sorted utterance list into per-speaker sub-lists.

    Generalization of ``separate_by_speaker`` which only handled 'A'/'B'.

    Args:
        dialog_ord: sorted list of utterance dicts.
        speaker_labels: list of all possible speaker label strings.

    Returns:
        dict mapping each speaker label to its list of utterances,
        e.g. {"S0": [...], "S1": [...], "S2": [...]}.
    """
    per_speaker = {label: [] for label in speaker_labels}
    for turn in dialog_ord:
        per_speaker[turn['speaker']].append(turn)

    total = sum(len(v) for v in per_speaker.values())
    assert total == len(dialog_ord), (
        f"Speaker split mismatch: {total} != {len(dialog_ord)}")
    return per_speaker


# ---------------------------------------------------------------------------
# AlignedProcess_Morespeakers – supports 1 to 8 speakers
# ---------------------------------------------------------------------------

class AlignedProcess_Morespeakers():
    """Parse small ASR chunks into turn-level transcriptions for N speakers.

    This is the multi-speaker generalization of ``AlignedProcess``, which was
    limited to exactly 2 speakers (A and B).  ``AlignedProcess_Morespeakers``
    accepts a list of 1–8 (transcript, speaker_name) pairs and internally
    labels them "S0" … "S7".

    The processing pipeline is identical to the original:
        1. ``split_trans``   – split raw ASR segments into word-level chunks
        2. ``combine``       – merge chunks close in time within same speaker
        3. ``remove BC``     – detect and separate backchannel utterances
        4. ``remove overlap``– detect and separate overlapping utterances
        5. ``combine trps``  – merge consecutive same-speaker turns around BCs

    Args:
        transcripts: list of transcript objects, one per speaker.  Each is a
                     list of ASR segment dicts with a 'words' key.
        speaker_names: list of human-readable speaker names (same length as
                       *transcripts*).  These are stored and used for output
                       but internally we use canonical labels S0–S7.
        trp_separated_by: gap (seconds) below which consecutive same-speaker
                          chunks are merged.
        pre_silence / post_silence: silence thresholds for BC detection.
        bc_duration: max duration for a backchannel candidate.
        yield_int_thresh: (reserved for downstream use).
        include_backchannels: whether to keep BC info in the result.
        include_overlap: whether to keep overlap info in the result.
        interval_character: character used to join words in a segment text.
        turn_gap_threshold: gap (seconds) above which a single ASR segment
                            is split into separate turns.
    """

    # Maximum number of speakers supported.
    MAX_SPEAKERS = 10

    def __init__(
        self,
        transcripts: list,
        speaker_names: list,
        trp_separated_by=1,
        pre_silence=0.5,
        post_silence=0.5,
        bc_duration=3,
        yield_int_thresh=0.2,
        include_backchannels=True,
        include_overlap=True,
        interval_character=' ',
        turn_gap_threshold=6,
    ):
        # --- Validate inputs ---
        num_speakers = len(transcripts)
        assert len(speaker_names) == num_speakers, (
            f"Number of transcripts ({num_speakers}) must match number of "
            f"speaker names ({len(speaker_names)})")
        assert 1 <= num_speakers <= self.MAX_SPEAKERS, (
            f"Number of speakers must be between 1 and {self.MAX_SPEAKERS}, "
            f"got {num_speakers}")

        # --- Store configuration ---
        self.interval_character = interval_character
        self.turn_gap_threshold = turn_gap_threshold
        self.trp_separated_by = trp_separated_by
        self.pre_silence = pre_silence
        self.post_silence = post_silence
        self.bc_duration = bc_duration
        self.yield_int_thresh = yield_int_thresh
        self.include_backchannels = include_backchannels
        self.include_overlap = include_overlap
        self.dont_cat = False

        # --- Build speaker label mappings ---
        # Internal canonical labels: "S0", "S1", ..., "S{N-1}"
        self.num_speakers = num_speakers
        self.speaker_labels = [f"S{i}" for i in range(num_speakers)]

        # Map internal label -> human-readable name (e.g. "S0" -> "Alice").
        self.speaker_names = {
            self.speaker_labels[i]: speaker_names[i]
            for i in range(num_speakers)
        }
        # Reverse map: human-readable name -> internal label.
        self.speaker_names_inv = {v: k for k, v in self.speaker_names.items()}

        # --- Step 1: Split each speaker's transcript into word-level segments ---
        annotations = []
        durations = []
        for i, transcript in enumerate(transcripts):
            anno = self.split_trans(transcript, self.speaker_labels[i])
            annotations.append(anno)
            if len(anno) > 0:
                durations.append(anno[-1]["end"])

        assert len(durations) > 0, "All speakers have empty transcripts"
        self.max_len = max(durations)

        # --- Steps 2-5: Preprocess (combine, BC removal, overlap, merge) ---
        self.dialog = self.preprocess_diag(annotations)

    # ------------------------------------------------------------------
    # Preprocessing pipeline
    # ------------------------------------------------------------------

    def preprocess_diag(self, dialog_lists):
        """Run the full preprocessing pipeline on N speaker segment lists.

        Steps:
            1. Merge segments that are temporally close (< trp_separated_by).
            2. Detect and separate backchannel utterances.
            3. Detect and separate overlapping utterances.
            4. Re-merge consecutive same-speaker turns around BCs/overlaps.

        Args:
            dialog_lists: list of N lists of utterance dicts, one per speaker.

        Returns:
            dict with keys "dialog", "backchannel", "overlap", each mapping
            to a per-speaker dict produced by ``separate_by_speaker_multi``.
        """
        # Step 1: Combine temporally close segments across all speakers.
        dialog = combine_dialogue_without_timings_multi(
            dialog_lists, self.speaker_labels,
            separated_by=self.trp_separated_by, dont_cat=self.dont_cat)

        # Step 2: Detect and separate backchannels.
        if not self.dont_cat:
            dialog, backchannels = pairwise_remove_backchannels_multi(
                dialog, self.speaker_labels,
                self.pre_silence, self.post_silence, self.bc_duration)
        else:
            dialog, backchannels = pairwise_remove_backchannels2_multi(
                dialog, self.speaker_labels,
                self.pre_silence, self.post_silence, self.bc_duration)

        # Step 3: Detect and separate overlaps.
        dialog, overlaps = remove_overlaps(dialog)

        # Step 4: Merge consecutive same-speaker turns around BCs/overlaps.
        if not self.dont_cat:
            dialog, backchannels = combine_consecutive_trps(
                dialog, backchannels, overlaps)
        else:
            dialog, backchannels = combine_consecutive_trps2(
                dialog, backchannels, overlaps)

        # Step 5: Split results back into per-speaker dicts.
        result = {
            "dialog": separate_by_speaker_multi(dialog, self.speaker_labels),
            "backchannel": separate_by_speaker_multi(
                backchannels, self.speaker_labels),
            "overlap": separate_by_speaker_multi(
                overlaps, self.speaker_labels),
        }
        return result

    # ------------------------------------------------------------------
    # Output methods
    # ------------------------------------------------------------------

    def get_parsed_dialog(self):
        """Return a list of per-speaker utterance lists with human-readable names.

        Each list contains all dialog + backchannel + overlap utterances for
        one speaker, sorted by time.  The 'speaker' field is replaced with
        the human-readable speaker name passed at construction.

        Returns:
            list of N lists (one per speaker), each sorted by (start, -end).
        """
        result = []
        for label in self.speaker_labels:
            # Gather dialog, backchannel, and overlap entries for this speaker.
            dialogs = self.dialog["dialog"].get(label, [])
            bcs = self.dialog["backchannel"].get(label, [])
            overlaps = self.dialog["overlap"].get(label, [])

            # Tag each utterance with its dialog_type.
            tagged = ([x | {"dialog_type": "dialog"} for x in dialogs]
                      + [x | {"dialog_type": "backchannel"} for x in bcs]
                      + [x | {"dialog_type": "overlap"} for x in overlaps])

            # Sort by time.
            tagged.sort(key=lambda k: (k['start'], k['end']))

            # Replace internal label with the human-readable speaker name.
            for utt in tagged:
                utt['speaker'] = self.speaker_names[label]

            result.append(tagged)

        return result

    def get_parsed_dialog_combined(self):
        """Return a single sorted list of all utterances across all speakers.

        Convenience wrapper around ``get_parsed_dialog`` that flattens the
        per-speaker lists into one timeline.

        Returns:
            list of utterance dicts sorted by (start, end).
        """
        per_speaker = self.get_parsed_dialog()
        combined = []
        for spk_list in per_speaker:
            combined.extend(spk_list)
        combined.sort(key=lambda k: (k['start'], k['end']))
        return combined

    def print_final_diag(self):
        """Print the full processed dialog in chronological order."""
        combined = self.get_parsed_dialog_combined()
        print()
        print(f"final..... {len(combined)} utterances, "
              f"{self.num_speakers} speakers")
        for utt in combined:
            print(utt["dialog_type"], utt["start"], utt["end"],
                  utt["speaker"], utt["text"])

    # ------------------------------------------------------------------
    # Segment splitting (reused from AlignedProcess)
    # ------------------------------------------------------------------

    def print_diag(self, diag):
        for utt in diag:
            print(utt["start"], utt["end"], utt["speaker"], utt["text"])

    def split_trans(self, transcript, speaker):
        """Split a raw ASR transcript into word-level utterance segments.

        Each ASR segment may contain many words.  This method:
          - Validates word timestamps (via ``fix_invalid_segments``).
          - Splits segments that have abnormally long gaps between words
            (> ``turn_gap_threshold``).
          - Clamps impossibly long word durations (> 6 s).

        Args:
            transcript: list of ASR segment dicts, each with a 'words' key
                        containing a list of word-level dicts with 'word',
                        'start', 'end', 'score' keys.
            speaker: internal speaker label string (e.g. "S0").

        Returns:
            list of utterance dicts with keys:
                start, end, wfeats, text, speaker.
        """
        anno = []
        for n, seg in enumerate(transcript):
            segments = []
            for word in seg['words']:
                _seg = word.copy()
                _seg['word'] = _seg.pop('word')
                _seg['speaker'] = speaker
                segments.append(_seg)

            segments = self.fix_invalid_segments(segments)
            if len(segments) == 0:
                continue

            new_segments = []
            temp_segs = []

            for wi in range(len(segments)):
                w = segments[wi]

                # Clamp words longer than 6 seconds (likely ASR alignment error).
                if (w["end"] - w["start"]) > 6:
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
                # print(w['word'], next_w["start"], w["end"], next_w["start"] - w["end"], self.turn_gap_threshold)
                # Split into separate turns when the gap exceeds threshold.
                if (next_w["start"] - w["end"]) > self.turn_gap_threshold:
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
                    "text": self.interval_character.join(
                        [x["word"] for x in segments]),
                    "speaker": speaker,
                })

        return anno

    def fix_invalid_segments(self, segments):
        """Fix or discard word segments with missing/invalid timestamps.

        Words that lack valid start/end times are merged into the previous
        valid word's text.  This prevents downstream timestamp errors.

        Args:
            segments: list of word-level dicts.

        Returns:
            Cleaned list of word-level dicts with valid timestamps.
        """
        _segments = []
        prev_seg_valid = True

        for _, seg in enumerate(segments):
            if len(_segments) == 0:
                if _is_valid_seg(seg):
                    _segments.append(seg.copy())
                continue

            if _is_valid_seg(seg):
                if not prev_seg_valid:
                    _segments[-1]['end'] = seg['start']
                _segments.append(seg.copy())
                prev_seg_valid = True
            else:
                _segments[-1]['word'] = self.interval_character.join(
                    [_segments[-1]['word'], seg['word']])
                prev_seg_valid = False

        return _segments
