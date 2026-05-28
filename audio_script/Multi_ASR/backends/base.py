"""Backend interface for Step 1 multi-talker diarization + ASR."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class BaseBackend:
    """Unified inference interface for a single audio chunk.

    Subclasses implement :py:meth:`transcribe`, which consumes a mono 16 kHz
    numpy chunk and returns:

      - ``word_list``   : ``{ "speaker_<id>": [{"word", "start", "end", "score", ...}, ...] }``
      - ``diar_binary`` : ``np.ndarray`` of shape ``(num_frames, num_speakers)``
        with dtype bool; ``num_frames`` is the audio duration divided by
        ``FRAME_LEN_SEC``.

    Optionally override :py:meth:`extra_manifest` to add backend-specific
    fields to the per-chunk ``sample_info.json``.
    """

    name: str = "base"

    def transcribe(
        self,
        audio: np.ndarray,
        audio_file: Optional[str] = None,
    ) -> Tuple[Dict[str, List[Dict]], np.ndarray]:
        raise NotImplementedError

    def extra_manifest(self) -> Dict:
        return {"mode": self.name}
