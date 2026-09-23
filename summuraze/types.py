"""Контракты, через которые общаются стадии конвейера.

Единый источник истины для типов. Вендорских импортов здесь нет и быть не должно: конвейер знает
только эти модели и протоколы, а `faster_whisper` / `anthropic` / `pyannote` живут каждый в своём
адаптере (guardrail #1). Поэтому смена STT или LLM не трогает ни одну стадию.

Время везде — **секунды от начала исходного файла**, а не от начала склейки речевых интервалов.
После VAD (#10) ASR получает только речь, и соблазн считать таймкоды от неё велик: тогда вся
навигация по саммари молча съезжает, а заметно это не сразу (см. CLAUDE.md, «Что легко сломать»).

Refs #4
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator


class Interval(BaseModel):
    """Отрезок времени в записи, в секундах от начала исходного файла."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)

    @model_validator(mode="after")
    def _end_after_start(self) -> Interval:
        if self.end < self.start:
            raise ValueError(f"конец отрезка раньше начала: {self.start} > {self.end}")
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start


class Word(Interval):
    """Слово с таймкодом.

    `prob` нужен не для красоты: по нему подсвечиваются места, где ASR угадывал (#12), и по нему
    же verify (#22) решает, насколько доверять утверждению, опирающемуся на этот фрагмент.
    """

    text: str
    prob: float = Field(ge=0, le=1)


class Segment(Interval):
    """Фраза: то, что ASR отдаёт одним куском.

    `speaker` заведён с самого начала и до этапа 5 всегда `None`. Дописывать поле в уже
    закэшированные артефакты дороже, чем завести сразу (ADR «Начинаем с лекций на один голос»).

    `suspect` ставит детектор галлюцинаций (#12). Сегмент не удаляется, а помечается: «здесь плохо
    слышно» должно дойти до промпта саммари, иначе модель достроит смысл по битому тексту.
    """

    text: str
    words: list[Word] = Field(default_factory=list)
    speaker: str | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    suspect: bool = False
    suspect_reason: str | None = None


class Transcript(BaseModel):
    """Результат распознавания — самостоятельный артефакт, а не промежуточный буфер.

    `asr_model` попадает в шапку готового документа: чем расшифровано, влияет на доверие к тексту.
    """

    source: Path
    duration: float = Field(ge=0)
    language: str
    asr_model: str
    segments: list[Segment] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def speech_duration(self) -> float:
        """Сколько в записи собственно речи — остальное тишина, её ASR не расшифровывает."""
        return sum(s.duration for s in self.segments)


class SpeakerTurn(Interval):
    """Отрезок, в котором говорит один человек.

    Источник — либо раздельные дорожки записи (этап 6, точно и бесплатно), либо
    diarization-модель (#25, дорого на CPU).
    """

    speaker: str


class TokenUsage(BaseModel):
    """Расход токенов за один запрос к LLM.

    `cached_input` отдельно, потому что кэш-хит у провайдеров дешевле промаха в десятки раз, и
    нулевой `cached_input` при повторном прогоне той же записи — сигнал, что префикс нестабилен
    (#15). Это диагностика, а не бухгалтерия.
    """

    input: int = Field(ge=0)
    output: int = Field(ge=0)
    cached_input: int = Field(default=0, ge=0)


class LLMResult(BaseModel):
    text: str
    model: str
    usage: TokenUsage
    cost_usd: float | None = None


@runtime_checkable
class ASRProvider(Protocol):
    """Распознавание речи. Реализации: faster-whisper (#11), whisper.cpp (#9)."""

    name: str

    def transcribe(
        self,
        audio: Path,
        *,
        language: str | None = None,
        speech: list[Interval] | None = None,
        terms: list[str] | None = None,
    ) -> Transcript:
        """Расшифровать аудио.

        `speech` — интервалы речи от VAD (#10); реализация обязана вернуть таймкоды в системе
        исходного файла, а не склейки. `terms` — словарь терминов для `initial_prompt` (#13).
        """
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Саммаризация. Один адаптер обслуживает и API, и локальный сервер (#15)."""

    name: str
    model: str
    context_window: int

    def complete(
        self,
        *,
        system: str,
        prefix: str,
        question: str,
        max_tokens: int,
    ) -> LLMResult:
        """Один запрос к модели.

        `prefix` и `question` разделены не для удобства: кэш префикса требует, чтобы транскрипт шёл
        первым и байт в байт, а всё изменчивое — после него. Слить их в один `user` означает
        потерять кэш и платить полную цену за каждый прогон (#15).
        """
        ...


@runtime_checkable
class SpeakerSource(Protocol):
    """Источник атрибуции речи: раздельные дорожки (этап 6) или модель (#25)."""

    name: str

    def turns(self, audio: Path) -> list[SpeakerTurn]: ...
