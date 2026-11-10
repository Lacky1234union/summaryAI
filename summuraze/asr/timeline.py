"""Склейка речи и обратный пересчёт таймкодов в систему исходного файла.

ASR получает интервалы речи склеенными подряд: Whisper кодирует окнами по 30 с, и подавать каждый
интервал отдельным окном значило бы гонять энкодер по сотням полупустых окон (на двухчасовой
лекции VAD даёт ~750 интервалов против ~220 окон склейки). Расплата — таймкоды модели идут по
склейке, и их надо вернуть в секунды исходника. Это главная ловушка #10/#11 (CLAUDE.md, «Таймкоды
после VAD»), поэтому пересчёт живёт здесь, отдельно от вендора, и покрыт своими тестами.

Слово относится к куску склейки по своей середине: слово на стыке двух кусков не должно
получить начало из одного куска, а конец из другого — между ними в исходнике может быть минута
тишины.

Refs #11
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from summuraze.types import Interval


@dataclass(frozen=True)
class SpeechTimeline:
    """Куски речи исходника, уложенные встык. `packed_starts[i]` — начало куска i в склейке."""

    chunks: tuple[Interval, ...]
    packed_starts: tuple[float, ...]

    @classmethod
    def from_intervals(cls, speech: list[Interval]) -> SpeechTimeline:
        chunks = tuple(sorted(speech, key=lambda i: i.start))
        starts: list[float] = []
        t = 0.0
        for c in chunks:
            starts.append(t)
            t += c.duration
        return cls(chunks, tuple(starts))

    @property
    def packed_duration(self) -> float:
        return self.packed_starts[-1] + self.chunks[-1].duration if self.chunks else 0.0

    def chunk_of(self, packed: float) -> int:
        """В каком куске лежит момент склейки (границы — в пользу следующего куска)."""
        return max(bisect.bisect_right(self.packed_starts, packed) - 1, 0)

    def to_source(self, packed: float, chunk: int | None = None) -> float:
        """Момент склейки → секунды исходника. `chunk` прибивает кусок явно (для концов слов)."""
        i = self.chunk_of(packed) if chunk is None else chunk
        c = self.chunks[i]
        return min(max(c.start + packed - self.packed_starts[i], c.start), c.end)
