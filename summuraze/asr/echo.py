"""Заглушка ASR: режет запись на равные куски и ничего не распознаёт.

Нужна этапу 0, чтобы конвейер проходил целиком до появления faster-whisper (#11), и тестам —
сквозной прогон без моделей. Таймкоды настоящие (от длительности wav), текст — нет.

Контракт `speech` соблюдает честно: режет только интервалы речи от VAD и не выходит за них, —
так на заглушке проверяется, что таймкоды идут от начала исходника, а не от склейки речи.
"""

from __future__ import annotations

import wave
from pathlib import Path

from summuraze.types import Interval, Segment, Transcript

CHUNK_SECONDS = 30.0


class EchoASR:
    name = "echo"

    def transcribe(
        self,
        audio: Path,
        *,
        language: str | None = None,
        speech: list[Interval] | None = None,
        terms: list[str] | None = None,
    ) -> Transcript:
        with wave.open(str(audio), "rb") as w:
            duration = w.getnframes() / w.getframerate()

        if speech is None:
            speech = [Interval(start=0, end=duration)]
        segments: list[Segment] = []
        for interval in speech:
            start = interval.start
            while start < interval.end:
                end = min(start + CHUNK_SECONDS, interval.end)
                text = f"эхо-сегмент {len(segments) + 1}"
                segments.append(Segment(start=start, end=end, text=text))
                start = end
        return Transcript(
            source=audio,
            duration=duration,
            language=language or "ru",
            asr_model="echo",
            segments=segments,
        )
