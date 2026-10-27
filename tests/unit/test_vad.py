"""VAD: речь находится там, где она есть в исходнике, тишина и тон речью не считаются."""

from __future__ import annotations

import wave
from collections.abc import Callable
from pathlib import Path

import pytest

from summuraze.audio.ingest import ingest
from summuraze.audio.vad import detect_speech
from summuraze.config import VADSettings
from summuraze.types import Interval

Compose = Callable[..., list[tuple[float, float]]]

# Запас VAD по краям (speech_pad_ms=400) плюс шаг окна Silero 32 мс.
TOLERANCE = 0.6


def _speech(compose: Compose, tmp_path: Path, parts: list[tuple], **cfg: object) -> tuple:
    spans = compose(tmp_path / "rec.m4a", parts)
    info = ingest(tmp_path / "rec.m4a", tmp_path / "rec.wav")
    return spans, detect_speech(info.wav, VADSettings(**cfg)), info.duration


def test_speech_found_silence_and_tone_skipped(compose: Compose, tmp_path: Path) -> None:
    parts = [("silence", 10), ("speech",), ("tone", 8), ("speech",), ("silence", 5)]
    spans, speech, _ = _speech(compose, tmp_path, parts)

    assert len(speech) == len(spans) == 2
    for got, (start, end) in zip(speech, spans, strict=True):
        assert got.start == pytest.approx(start, abs=TOLERANCE)
        assert got.end == pytest.approx(end, abs=TOLERANCE)

    tone = Interval(start=spans[0][1] + TOLERANCE, end=spans[1][0] - TOLERANCE)
    assert not any(i.start < tone.end and tone.start < i.end for i in speech), (
        "тон 440 Гц — не речь: на нём Whisper и галлюцинирует"
    )


def test_timecodes_count_from_source_start(compose: Compose, tmp_path: Path) -> None:
    """Главная ловушка #10: речь на 37-й секунде исходника — это 37 с, а не 0 с склейки."""
    spans, speech, _ = _speech(compose, tmp_path, [("silence", 37), ("speech",)])
    assert speech[0].start == pytest.approx(37.0, abs=TOLERANCE)
    assert speech[0].end == pytest.approx(spans[0][1], abs=TOLERANCE)


def test_interval_never_exceeds_recording(compose: Compose, tmp_path: Path) -> None:
    # Речь до самого конца: паддинг не должен вытолкнуть конец интервала за длину записи.
    _, speech, duration = _speech(compose, tmp_path, [("silence", 2), ("speech",)])
    assert speech[-1].end <= duration


def test_silence_only_gives_nothing(compose: Compose, tmp_path: Path) -> None:
    _, speech, _ = _speech(compose, tmp_path, [("silence", 5), ("tone", 5)])
    assert speech == []


def test_settings_reach_silero(compose: Compose, tmp_path: Path) -> None:
    """Пороги из конфига работают, а не только лежат в .env: длинная пауза склеивает речь."""
    parts = [("silence", 3), ("speech",), ("silence", 4), ("speech",), ("silence", 3)]
    _, apart, _ = _speech(compose, tmp_path, parts)
    _, merged, _ = _speech(compose, tmp_path, parts, min_silence_ms=10_000)
    assert (len(apart), len(merged)) == (2, 1)


def test_empty_wav(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
    assert detect_speech(path, VADSettings()) == []


def test_rejects_wav_that_did_not_pass_ingest(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44_100)
        w.writeframes(b"\0" * 400)
    with pytest.raises(ValueError, match="16 kHz mono"):
        detect_speech(path, VADSettings())
