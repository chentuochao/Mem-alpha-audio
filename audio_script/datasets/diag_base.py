from collections.abc import Generator
import logging
import math
import bisect
import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger(__name__)


class TurnType(IntEnum):
    """
    (2) is reserved for the YIELD turn end type
    """

    NONE = 0
    NORMAL = 1
    BACKCHANNEL = 3
    INTERRUPT = 4
    OVERLAP = 5

    NON_OVERLAP = 6


class TurnEndType(IntEnum):
    NONE = 0
    NORMAL = 1
    YIELD = 2


@dataclass
class TurnID:
    """
    Represents a unique ID for a turn in a dialog with the following attributes:
        - dataset: the dataset ID
        - conv_id: the conversation ID. This is unique within a dataset
        - turn_type: the type of turn (normal, backchannel, interrupt, overlap)
        - speaker: the speaker (A/B)
        - turn_id: the turn ID. This is unique within a conversation irrespective of speaker and turn type.
            We can define a global ordering of turns in a conversation by sorting by this ID.
            turn_id is based on the order of start times of the turns in a conversation.
    """

    id: int

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __hash__(self):
        return hash(self.id)

    @property
    def value(self):
        return self.id

    def update_id(
        self,
        conv_id: int = 0,
        turn_type: TurnType = TurnType.NORMAL,
        speaker: str = "A",
        turn_id: int = 0,
    ):
        """
        Builds a unique ID for a turn in a dialog based on the given attributes:
            - conv_id: the conversation ID. This is unique within a dataset. If dataset is None, this must be a string
            - turn_type: the type of turn (normal, backchannel, interrupt, overlap)
            - speaker: the speaker (A/B)
            - turn_id: the turn ID. This is unique within a conversation irrespective of speaker.
                        Two turns of different turn types (normal/interrupt, backchannel/overlap) can have the same turn_id.

        """
        if isinstance(turn_type, TurnType):
            turn_type = turn_type.value

        if isinstance(speaker, str):
            if speaker not in {"A", "B"}:
                raise ValueError(f"Speaker must be 'A' or 'B', not {speaker}")

            speaker = 0 if speaker == "A" else 1


        self.id = (
            (conv_id << 34)
            | (turn_type << 30)
            | (speaker << 29)
            | turn_id
        )

        return self

    @staticmethod
    def default():
        return TurnID(0)

    @staticmethod
    def build_id(
        conv_id: int = 0,
        turn_type: TurnType = TurnType.NORMAL,
        speaker: str = "A",
        turn_id: int = 0,
    ):
        return TurnID.default().update_id(conv_id, turn_type, speaker, turn_id)


@dataclass
class Turn:
    """
    Represents a turn in a dialog with the following attributes:
        - word: the text of the entire turn
        - turn_type: the type of turn (normal, backchannel, interrupt, overlap)
        - turn_end_type: the type of turn ending (normal, yield)
        - start: the start time of the turn
        - end: the end time of the turn
        - id: the unique ID of the turn
        - curr_prev_id: the unique ID of the previous turn from the same speaker (includes bc/overlap)
        - curr_next_id: the unique ID of the next turn from the same speaker (includes bc/overlap)
        - other_prev_id: the unique ID of the previous turn from the other speaker.
                If the current turn is a backchannel or overlap turn and completely overlapped
                this will be the ID of the turn that the current turn is
                responding to
                If the current turn is a normal turn then this will be the ID of the prior normal turn from the other speaker
        - other_next_id: the unique ID of the next turn from the other speaker
                If the current turn is a backchannel turn (not overlap) and completely overlapped
                this will be the ID of the turn that the current turn is
                responding to.
                If the current turn is a normal turn then this will be the ID of the next normal turn from the other speaker
        - overlaps: a list of IDs of turns that are completely (not partially) overlapped by this turn
        - overlapped_by: the ID of the turn (if it exists) that this turn is completely (not partially) overlapped by
    """

    word: str = field(repr=False)
    turn_type: TurnType = field(repr=True)
    turn_end_type: TurnEndType = field(repr=True)
    start: float
    end: float
    speaker: str
    conv_id: str

    turn_index: int = 0

    id: TurnID = field(default_factory=TurnID.default)
    curr_prev_id: TurnID  = field(default_factory=TurnID.default)
    curr_next_id: TurnID  = field(default_factory=TurnID.default)
    other_prev_id: TurnID  = field(default_factory=TurnID.default)
    other_next_id: TurnID  = field(default_factory=TurnID.default)

    overlaps: list[TurnID] = field(default_factory=list)
    overlapped_by: TurnID  = None

    def __post_init__(self):
        self.id.update_id(
            conv_id=self.conv_id,
            turn_id=self.turn_index,
            turn_type=self.turn_type,
            speaker=self.speaker,
        )

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def to_dict(self):
        return {
            "word": self.word,
            "turn_type": self.turn_type,
            "turn_end_type": self.turn_end_type,
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "conv_id": self.conv_id,
            "turn_index": self.turn_index,
            "id": self.id.value,
            "curr_prev_id": (
                self.curr_prev_id.value if self.curr_prev_id is not None else None
            ),
            "curr_next_id": (
                self.curr_next_id.value if self.curr_next_id is not None else None
            ),
            "other_prev_id": (
                self.other_prev_id.value if self.other_prev_id is not None else None
            ),
            "other_next_id": (
                self.other_next_id.value if self.other_next_id is not None else None
            ),
            "overlaps": [x.value for x in self.overlaps],
            "overlapped_by": (
                self.overlapped_by.value if self.overlapped_by is not None else None
            ),
        }

    @staticmethod
    def from_dict(data):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.tolist()

        return Turn(
            data["word"],
            data["turn_type"],
            data["turn_end_type"],
            data["start"],
            data["end"],
            data["speaker"],
            data["conv_id"],
            data["turn_index"],
            TurnID(data["id"]),
            TurnID(data["curr_prev_id"]),
            TurnID(data["curr_next_id"]),
            TurnID(data["other_prev_id"]),
            TurnID(data["other_next_id"]),
            [TurnID(x) for x in data["overlaps"]],
            TurnID(data["overlapped_by"]),
        )

    @staticmethod
    def default(speaker: str = "A", conv_id: str = ""):
        return Turn(
            "", TurnType.NORMAL, TurnEndType.NORMAL, 0.0, math.inf, speaker, conv_id
        )

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)

        self.id.update_id(
            conv_id=self.conv_id,
            turn_id=self.turn_index,
            turn_type=self.turn_type,
            speaker=self.speaker,
        )

    def get_turn_duration(self):
        return self.end - self.start

    def get_turn_length(self):
        return len(self.word.split())
