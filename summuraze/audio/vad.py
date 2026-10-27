"""VAD: wav → интервалы речи, которые потом получает ASR.

Whisper на тишине и музыке пишет текст, которого не было: «Спасибо за просмотр», циклы,
повторы последней фразы. VAD убирает главный источник этих галлюцинаций и заодно 20–40 %
времени ASR.

Модель — Silero v6 в ONNX из состава faster-whisper, вместе с его же разметкой вероятностей
в отрезки. Ни torch, ни отдельного пакета `silero-vad` не нужно: faster-whisper уже стоит ради
ASR. Почему не пакет `silero-vad` — ADR 2026-10-27.

Интервалы — в секундах от начала исходного файла: ингест не обрезает начало записи, поэтому
сэмпл wav и секунда исходника совпадают. Пересчитать таймкоды обратно после склейки речи — работа
ASR-адаптера (контракт `ASRProvider.transcribe`), VAD ничего не склеивает.

Память: весь wav читается в float32, и модель держит ещё две его копии. Замер на двух часах —
1.7 ГБ пика при 28 с работы (BENCHMARKS.md). ASR в этот момент ещё не загружена, так что в 14 ГБ
это проходит с запасом; резать на куски ради памяти пока незачем.

Refs #10
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import TYPE_CHECKING

from summuraze.types import Interval

if TYPE_CHECKING:
    from summuraze.config import VADSettings

SAMPLE_RATE = 16_000


class NoSpeechError(RuntimeError):
    """В записи не нашлось речи: саммари по пустому транскрипту бессмысленно."""


def detect_speech(wav: Path, cfg: VADSettings) -> list[Interval]:
    """Найти интервалы речи в wav 16 kHz mono s16 (выход ингеста)."""
    import numpy as np
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    with wave.open(str(wav), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"VAD ждёт wav 16 kHz mono s16, а получил другой: {wav}")
        frames = w.readframes(w.getnframes())
    if not frames:
        return []
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    del frames

    chunks = get_speech_timestamps(
        audio,
        VadOptions(
            threshold=cfg.threshold,
            min_speech_duration_ms=cfg.min_speech_ms,
            min_silence_duration_ms=cfg.min_silence_ms,
            speech_pad_ms=cfg.speech_pad_ms,
        ),
        sampling_rate=SAMPLE_RATE,
    )
    duration = len(audio) / SAMPLE_RATE
    return [
        Interval(start=c["start"] / SAMPLE_RATE, end=min(c["end"] / SAMPLE_RATE, duration))
        for c in chunks
    ]
