"""Shared constants for the Multi_ASR pipeline (Step 1).

These are imported by both the inference backends and the dataset-agnostic
pipeline utilities so all callers agree on the audio sample rate and the
diarization frame stride.
"""

SR = 16000
"""Mono audio sample rate (Hz) used throughout Step 1."""

FRAME_LEN_SEC = 0.08
"""Diarization frame stride (seconds per frame in the diar_pred matrix)."""



### chunk parameters    
CHUNK_MIN_DURATION = 60.0
CHUNK_MAX_DURATION = 300.0
CHUNK_GAP_THRESHOLD = 3.0