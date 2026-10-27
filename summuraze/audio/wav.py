"""Чтение wav после ингеста в float32 — так его едят и Silero, и Whisper."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

SAMPLE_RATE = 16_000


def read_pcm(path: Path) -> np.ndarray:
    """wav 16 kHz mono s16 → float32 в [-1, 1). Другой формат — значит, файл не из ингеста."""
    import numpy as np

    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(
                f"ожидался wav 16 kHz mono s16 (выход ингеста), а пришёл другой: {path}"
            )
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
