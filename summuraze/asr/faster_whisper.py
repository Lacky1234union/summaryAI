"""ASR на faster-whisper (CTranslate2, int8 на CPU).

Единственное место, где живёт `faster_whisper` как распознаватель (guardrail #1); импорт —
ленивый, в конструкторе, а конструктор зовёт фабрика только при промахе кэша.

Модель грузится в конструкторе и живёт, пока жив объект: оркестратор удаляет провайдер сразу
после стадии, до создания LLM (CLAUDE.md, «Память»).

Речь от VAD склеивается подряд и уходит модели одним массивом; таймкоды слов возвращаются в
систему исходника через `SpeechTimeline`. Встроенный `vad_filter` faster-whisper выключен: VAD —
отдельная стадия (#10), второй проход отрезал бы речь иначе, чем записано в её артефакте.

Сегмент Whisper, слова которого попали в разные куски склейки, режется по стыку: иначе в
транскрипте была бы «фраза» через минуту тишины исходника, и таймкод вёл бы не туда.

Refs #11
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from summuraze.asr.timeline import SpeechTimeline
from summuraze.audio.wav import SAMPLE_RATE, read_pcm
from summuraze.types import Interval, Segment, Transcript, Word

if TYPE_CHECKING:
    from summuraze.config import ASRSettings

_CPU_SYS = Path("/sys/devices/system/cpu")


def physical_cores() -> int:
    """Физические ядра, без гиперпотоков.

    Дефолт CTranslate2 — 4 потока; на Ryzen 7 5700U (8 ядер / 16 потоков) 8 потоков дают RTF 0.43
    против 0.63 на 4 (BENCHMARKS.md). Гиперпотоки делят с ядром те же блоки, ускорения не дают.
    """
    cores = {
        p.read_text().strip()
        for p in _CPU_SYS.glob("cpu[0-9]*/topology/thread_siblings_list")
        if p.is_file()
    }
    return len(cores) or os.cpu_count() or 4


class FasterWhisperASR:
    name = "faster-whisper"

    def __init__(self, cfg: ASRSettings) -> None:
        from faster_whisper import WhisperModel

        self.model_name = cfg.model
        self._model = WhisperModel(
            cfg.model,
            device=cfg.device,
            compute_type=cfg.compute_type,
            cpu_threads=cfg.threads or physical_cores(),
        )

    def transcribe(
        self,
        audio: Path,
        *,
        language: str | None = None,
        speech: list[Interval] | None = None,
        terms: list[str] | None = None,
    ) -> Transcript:
        import numpy as np

        pcm = read_pcm(audio)
        duration = len(pcm) / SAMPLE_RATE
        if speech is None:
            speech = [Interval(start=0, end=duration)]
        timeline = SpeechTimeline.from_intervals(speech)
        if not timeline.chunks:
            return Transcript(
                source=audio,
                duration=duration,
                language=language or "",
                asr_model=self.model_name,
            )

        packed = np.concatenate(
            [pcm[int(c.start * SAMPLE_RATE) : int(c.end * SAMPLE_RATE)] for c in timeline.chunks]
        )
        del pcm
        raw, info = self._model.transcribe(
            packed,
            language=language,
            word_timestamps=True,
            vad_filter=False,
            # Словарь терминов — подсказка модели; как его собирать и проверять — #13.
            initial_prompt=", ".join(terms) if terms else None,
        )
        segments = [s for seg in raw for s in _to_source(seg, timeline)]
        return Transcript(
            source=audio,
            duration=duration,
            language=info.language,
            asr_model=self.model_name,
            segments=segments,
        )


def _to_source(seg: Any, timeline: SpeechTimeline) -> Iterable[Segment]:
    """Сегмент faster-whisper (время склейки) → наши сегменты (время исходника)."""
    meta = {"avg_logprob": seg.avg_logprob, "no_speech_prob": seg.no_speech_prob}
    if not seg.words:
        start_chunk = timeline.chunk_of(seg.start)
        start = timeline.to_source(seg.start, start_chunk)
        end = timeline.to_source(seg.end, start_chunk)
        if seg.text.strip():
            yield Segment(start=start, end=max(end, start), text=seg.text.strip(), **meta)
        return

    runs: list[tuple[int, list[Word]]] = []
    for w in seg.words:
        chunk = timeline.chunk_of((w.start + w.end) / 2)
        start = timeline.to_source(w.start, chunk)
        word = Word(
            start=start,
            end=max(timeline.to_source(w.end, chunk), start),
            text=w.word,
            prob=min(max(w.probability, 0.0), 1.0),
        )
        if runs and runs[-1][0] == chunk:
            runs[-1][1].append(word)
        else:
            runs.append((chunk, [word]))

    for _, words in runs:
        text = "".join(w.text for w in words).strip()
        if text:
            yield Segment(start=words[0].start, end=words[-1].end, text=text, words=words, **meta)
