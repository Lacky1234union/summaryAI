"""Проверки контрактов: инварианты моделей и то, что протоколы описывают реальную форму."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from summuraze.types import (
    ASRProvider,
    Interval,
    LLMResult,
    Segment,
    SpeakerTurn,
    TokenUsage,
    Transcript,
    Word,
)


class TestInterval:
    def test_duration(self) -> None:
        assert Interval(start=1.5, end=4.0).duration == pytest.approx(2.5)

    def test_zero_length_is_allowed(self) -> None:
        """Нулевой отрезок законен: VAD умеет отдавать вырожденные интервалы."""
        assert Interval(start=3.0, end=3.0).duration == 0

    def test_end_before_start_rejected(self) -> None:
        with pytest.raises(ValidationError, match="конец отрезка раньше начала"):
            Interval(start=5.0, end=2.0)

    def test_negative_time_rejected(self) -> None:
        """Отрицательный таймкод — признак того, что кто-то пересчитал время после VAD."""
        with pytest.raises(ValidationError):
            Interval(start=-1.0, end=2.0)


class TestWord:
    def test_probability_bounds(self) -> None:
        for bad in (-0.1, 1.1):
            with pytest.raises(ValidationError):
                Word(start=0, end=1, text="раз", prob=bad)


class TestSegment:
    def test_speaker_is_none_until_stage_five(self) -> None:
        """Поле есть с самого начала, но до этапа 5 всегда пустое."""
        assert Segment(start=0, end=1, text="раз").speaker is None

    def test_suspect_defaults_to_false(self) -> None:
        assert Segment(start=0, end=1, text="раз").suspect is False


class TestTranscript:
    def _transcript(self) -> Transcript:
        return Transcript(
            source=Path("lecture.mp4"),
            duration=100.0,
            language="ru",
            asr_model="large-v3-turbo",
            segments=[
                Segment(start=0, end=2, text="  первая  "),
                Segment(start=10, end=13, text="вторая"),
                Segment(start=20, end=21, text="   "),
            ],
        )

    def test_text_joins_and_strips(self) -> None:
        assert self._transcript().text == "первая вторая"

    def test_speech_duration_excludes_silence(self) -> None:
        """Сумма речи меньше длительности записи — между сегментами тишина."""
        t = self._transcript()
        assert t.speech_duration == pytest.approx(6.0)
        assert t.speech_duration < t.duration


class TestUsage:
    def test_cached_input_defaults_to_zero(self) -> None:
        assert TokenUsage(input=100, output=10).cached_input == 0

    def test_cost_is_optional(self) -> None:
        """У локальной модели цены нет — это не ошибка, а отсутствие цены."""
        r = LLMResult(text="итог", model="local", usage=TokenUsage(input=1, output=1))
        assert r.cost_usd is None


class TestProtocols:
    def test_conforming_class_satisfies_asr_protocol(self) -> None:
        class FakeASR:
            name = "fake"

            def transcribe(
                self,
                audio: Path,
                *,
                language: str | None = None,
                speech: list[Interval] | None = None,
                terms: list[str] | None = None,
            ) -> Transcript:
                return Transcript(
                    source=audio, duration=0.0, language=language or "ru", asr_model="fake"
                )

        assert isinstance(FakeASR(), ASRProvider)

    def test_class_without_method_does_not_satisfy_protocol(self) -> None:
        class NotASR:
            name = "no"

        assert not isinstance(NotASR(), ASRProvider)


def test_module_has_no_vendor_imports() -> None:
    """Guardrail #1: в контрактах не может быть вендорских импортов.

    Если faster_whisper или anthropic просочатся сюда, конвейер окажется прибит к вендору,
    а CLI начнёт грузить рантайм STT на --help.
    """
    source = (Path(__file__).parents[2] / "summuraze" / "types.py").read_text(encoding="utf-8")
    forbidden = ("faster_whisper", "ctranslate2", "torch", "anthropic", "openai", "pyannote")
    found = [name for name in forbidden if name in source]
    assert not found, f"вендорские импорты в контрактах: {found}"


def test_speaker_turn_keeps_speaker() -> None:
    assert SpeakerTurn(start=0, end=1, speaker="Егор").speaker == "Егор"
