"""Детектор галлюцинаций Whisper: выдумки помечены, живая речь — нет.

Тексты выдумок — реальные ответы large-v3-turbo на тишину, тон и шум (2026-10-27, #12).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from summuraze.asr.faster_whisper import FasterWhisperASR
from summuraze.asr.repetition import flag_suspects
from summuraze.audio.ingest import ingest
from summuraze.audio.vad import detect_speech
from summuraze.config import ASRSettings, VADSettings
from summuraze.types import Segment


def _seg(text: str, start: float = 0, **kw: float) -> Segment:
    return Segment(start=start, end=start + 2, text=text, **kw)


def _flags(*texts: str) -> list[str | None]:
    return [s.suspect_reason for s in flag_suspects([_seg(t, i * 2) for i, t in enumerate(texts)])]


@pytest.mark.parametrize(
    "text",
    [
        "Продолжение следует...",
        "Спасибо за просмотр!",
        "Субтитры сделал DimaTorzok",
        "Редактор субтитров А.Семкин Корректор А.Егорова",
        "Thank you.",
        "Thanks for watching!",
        "Спасибо.",
        "ПРОДОЛЖЕНИЕ СЛЕДУЕТ",
    ],
)
def test_stock_hallucinations_flagged(text: str) -> None:
    [seg] = flag_suspects([_seg(text)])
    assert seg.suspect and seg.suspect_reason


@pytest.mark.parametrize(
    "text",
    [
        "Короче, сейчас разговаривал с Иваном, в общем, как бы забавно не звучало.",
        "Он создавал себе SIP-сервер, где добавлял LLM, которая общалась с ним.",
        "Спасибо за вопрос, давайте разберём его на примере преобразования Фурье.",
        "Продолжение следует из того, что ряд сходится абсолютно, а значит и равномерно.",
        "Да, да, да, именно так.",
        "Как-то так.",
    ],
)
def test_real_speech_not_flagged(text: str) -> None:
    [seg] = flag_suspects([_seg(text)])
    assert not seg.suspect and seg.suspect_reason is None


def test_word_loop_within_segment() -> None:
    [reason] = _flags("и вот это и вот это и вот это и вот это")
    assert reason is not None and "повтор «и вот это»" in reason


def test_single_word_needs_four_repeats() -> None:
    assert _flags("так так так") == [None]
    assert _flags("так так так так")[0] is not None


def test_compressible_long_text() -> None:
    # Цикл с вариациями, который n-граммы не ловят: каждая фраза чуть другая.
    text = " ".join(f"Мы рассмотрим пример номер {i % 2}." for i in range(8))
    [seg] = flag_suspects([_seg(text)])
    assert seg.suspect


def test_neighbour_loop() -> None:
    flags = _flags(
        "Теперь рассмотрим второй пример.",
        "Теперь рассмотрим второй пример.",
        "Здесь всё иначе.",
        "Теперь рассмотрим второй пример.",
    )
    assert flags[0] is None
    assert flags[1] == "повтор предыдущего сегмента"
    assert flags[2] is None
    assert flags[3] == "повтор предыдущего сегмента", "цикл через одну фразу — тоже цикл"


def test_short_replies_may_repeat() -> None:
    assert _flags("Да.", "Да.") == [None, None]


def test_low_confidence() -> None:
    [seg] = flag_suspects([_seg("какой-то текст", no_speech_prob=0.9, avg_logprob=-1.4)])
    assert seg.suspect_reason == (
        "похоже на тишину (no_speech 0.90); низкая уверенность (logprob -1.40)"
    )


def test_input_not_mutated_and_reasons_joined() -> None:
    original = _seg("Thank you.", avg_logprob=-2.0)
    [seg] = flag_suspects([original])
    assert not original.suspect
    assert seg.suspect_reason is not None and "; низкая уверенность" in seg.suspect_reason


@pytest.mark.slow
def test_real_model_hallucinations_on_noise_are_all_flagged(
    compose: Callable[..., list[tuple[float, float]]],
    tmp_path: Path,
    cached_whisper: ASRSettings,
) -> None:
    """Без VAD Whisper пишет на тишине и тоне текст — каждая такая строка должна быть помечена;
    речь через VAD — ни одна."""
    asr = FasterWhisperASR(cached_whisper)

    compose(tmp_path / "bad.m4a", [("silence", 20), ("tone", 25), ("silence", 10)])
    bad = ingest(tmp_path / "bad.m4a", tmp_path / "bad.wav")
    hallucinated = asr.transcribe(bad.wav, language="ru").segments
    assert all(s.suspect for s in flag_suspects(hallucinated)), hallucinated

    compose(tmp_path / "ok.m4a", [("silence", 5), ("speech",), ("silence", 5), ("speech2",)])
    ok = ingest(tmp_path / "ok.m4a", tmp_path / "ok.wav")
    speech = detect_speech(ok.wav, VADSettings())
    real = asr.transcribe(ok.wav, language="ru", speech=speech).segments
    assert real and not any(s.suspect for s in flag_suspects(real)), real
