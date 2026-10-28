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

Термины (#13) уходят в `hotwords`, а не в `initial_prompt`: `initial_prompt` действует только
на первое окно, дальше его вытесняет распознанный текст — на термины после первых 30 секунд
он не влияет вовсе (замер в BENCHMARKS.md). `hotwords` подставляется в каждое окно, но длиннее
`HOTWORDS_MAX_TOKENS` ломает распознавание — лишние термины отрезаются с предупреждением.

Refs #11
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from summuraze.asr.base import ASRModelMissing
from summuraze.asr.timeline import SpeechTimeline
from summuraze.audio.wav import SAMPLE_RATE, read_pcm
from summuraze.types import Interval, Segment, Transcript, Word

if TYPE_CHECKING:
    from summuraze.config import ASRSettings

# Замер на large-v3-turbo: 54 токена (10 терминов) — термины вытянуты, текст цел; 84 — уже теряются
# слова; 114 и больше — терминов ноль, в тексте появляются непроизнесённые слова из подсказки.
HOTWORDS_MAX_TOKENS = 60

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
        try:
            self._model = WhisperModel(
                cfg.model,
                device=cfg.device,
                compute_type=cfg.compute_type,
                cpu_threads=cfg.threads or physical_cores(),
                # --local-only (#20): только с диска. Скачивание само по себе транскрипт не
                # увозит, но «ни одного сетевого запроса» — это ни одного.
                local_files_only=cfg.local_only,
            )
        except Exception as e:
            if not cfg.local_only:
                raise
            raise ASRModelMissing(
                f"модели {cfg.model!r} нет на диске, а --local-only запрещает её скачать. "
                "Запусти один раз без --local-only — модель скачается и останется в кэше"
            ) from e

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
            hotwords=self._hotwords(terms or []),
        )
        segments = [s for seg in raw for s in _to_source(seg, timeline)]
        return Transcript(
            source=audio,
            duration=duration,
            language=info.language,
            asr_model=self.model_name,
            segments=segments,
        )

    def _hotwords(self, terms: list[str]) -> str | None:
        tokenizer = self._model.hf_tokenizer
        hotwords, dropped = _fit_hotwords(terms, lambda text: len(tokenizer.encode(text).ids))
        if dropped:
            logger.warning(
                "в подсказку Whisper не влезли {} терминов из {} (лимит {} токенов): {}. "
                "Поставь важные выше в словаре; для остальных работают замены "
                "«слышится -> пишется»",
                len(dropped),
                len(terms),
                HOTWORDS_MAX_TOKENS,
                ", ".join(dropped),
            )
        return hotwords


def _fit_hotwords(terms: list[str], count_tokens: Any) -> tuple[str | None, list[str]]:
    """Взять термины по порядку, пока подсказка влезает в бюджет; вернуть её и отрезанное."""
    taken: list[str] = []
    for i, term in enumerate(terms):
        if count_tokens(" " + ", ".join([*taken, term])) > HOTWORDS_MAX_TOKENS:
            return (", ".join(taken) or None), terms[i:]
        taken.append(term)
    return (", ".join(taken) or None), []


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
