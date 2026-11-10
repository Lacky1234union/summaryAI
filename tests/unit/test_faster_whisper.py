"""ASR на faster-whisper: таймкоды склейки речи возвращаются в систему исходного файла.

Быстрые тесты подменяют `WhisperModel` фейком, который отвечает таймкодами склейки, — так
пересчёт проверяется без модели и в CI. Тесты с меткой `slow` гоняют настоящую модель
(`make test-all`) и пропускаются, если её нет в кэше HuggingFace.
"""

from __future__ import annotations

import wave
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from summuraze.asr.base import make_asr
from summuraze.asr.faster_whisper import FasterWhisperASR
from summuraze.asr.timeline import SpeechTimeline
from summuraze.audio.ingest import ingest
from summuraze.config import ASRSettings
from summuraze.types import ASRProvider, Interval

SR = 16_000


def _silence_wav(path: Path, seconds: float) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"\0\0" * int(seconds * SR))
    return path


def _word(start: float, end: float, text: str, prob: float = 0.9) -> SimpleNamespace:
    return SimpleNamespace(start=start, end=end, word=text, probability=prob)


def _segment(words: list[SimpleNamespace] | None, start: float = 0, end: float = 0, text: str = ""):
    if words:
        start, end = words[0].start, words[-1].end
        text = "".join(w.word for w in words)
    return SimpleNamespace(
        start=start, end=end, text=text, words=words, avg_logprob=-0.3, no_speech_prob=0.01
    )


class FakeModel:
    """Отвечает заранее заданными сегментами во времени склейки и запоминает, что получил."""

    segments: list[SimpleNamespace] = []
    created: list[dict[str, Any]] = []

    def __init__(self, name: str, **kwargs: Any) -> None:
        FakeModel.created.append({"name": name, **kwargs})
        self.calls: list[dict[str, Any]] = []
        # Токенизатор-фейк: 10 токенов на слово — в бюджет 60 влезает ровно 6 слов.
        self.hf_tokenizer = SimpleNamespace(
            encode=lambda text: SimpleNamespace(ids=[0] * 10 * len(text.split()))
        )

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[list[SimpleNamespace], Any]:
        self.calls.append({"samples": len(audio), **kwargs})
        return list(FakeModel.segments), SimpleNamespace(language=kwargs["language"] or "ru")


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> type[FakeModel]:
    import faster_whisper

    FakeModel.segments = []
    FakeModel.created = []
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    return FakeModel


def _asr(**cfg: Any) -> FasterWhisperASR:
    return FasterWhisperASR(ASRSettings(provider="faster-whisper", **cfg))


# Речь 10–15 с и 40–45 с исходника; в склейке это 0–5 и 5–10.
SPEECH = [Interval(start=10, end=15), Interval(start=40, end=45)]


class TestTimeline:
    def test_maps_packed_time_back_to_source(self) -> None:
        t = SpeechTimeline.from_intervals(SPEECH)
        assert t.packed_duration == 10
        assert t.to_source(1.0) == 11.0
        assert t.to_source(6.0) == 41.0

    def test_boundary_belongs_to_next_chunk(self) -> None:
        t = SpeechTimeline.from_intervals(SPEECH)
        assert t.chunk_of(5.0) == 1
        assert t.to_source(5.0) == 40.0

    def test_pinned_chunk_clamps_inside_it(self) -> None:
        """Конец слова, вылезший за кусок, не перескакивает через тишину исходника."""
        t = SpeechTimeline.from_intervals(SPEECH)
        assert t.to_source(5.3, chunk=0) == 15.0
        assert t.to_source(4.8, chunk=1) == 40.0

    def test_unsorted_input_and_empty(self) -> None:
        t = SpeechTimeline.from_intervals(list(reversed(SPEECH)))
        assert t.to_source(1.0) == 11.0
        assert SpeechTimeline.from_intervals([]).packed_duration == 0


class TestAdapter:
    def test_model_loaded_from_config(self, fake: type[FakeModel]) -> None:
        _asr(model="small", compute_type="int8", threads=6)
        assert fake.created == [
            {
                "name": "small",
                "device": "cpu",
                "compute_type": "int8",
                "cpu_threads": 6,
                "local_files_only": False,
            }
        ]

    def test_factory_builds_adapter(self, fake: type[FakeModel]) -> None:
        asr = make_asr(ASRSettings(provider="faster-whisper"))
        assert isinstance(asr, ASRProvider) and asr.name == "faster-whisper"

    def test_only_speech_is_sent_and_vad_filter_is_off(
        self, fake: type[FakeModel], tmp_path: Path
    ) -> None:
        asr = _asr()
        asr.transcribe(_silence_wav(tmp_path / "a.wav", 50), language="ru", speech=SPEECH)
        call = asr._model.calls[0]
        assert call["samples"] == 10 * SR
        assert call["vad_filter"] is False, "VAD — отдельная стадия, второй раз резать нельзя"
        assert call["word_timestamps"] is True
        assert call["condition_on_previous_text"] is False, "#70: иначе текст рвётся на обрывки"

    def test_words_return_to_source_time_and_split_at_gap(
        self, fake: type[FakeModel], tmp_path: Path
    ) -> None:
        # Whisper склеил в одну фразу слова по обе стороны 25 секунд тишины исходника.
        fake.segments = [
            _segment([_word(1.0, 1.5, " раз"), _word(4.8, 5.3, " два"), _word(6.0, 6.5, " три")])
        ]
        t = _asr().transcribe(_silence_wav(tmp_path / "a.wav", 50), speech=SPEECH)

        assert [(s.start, s.end, s.text) for s in t.segments] == [
            (11.0, 11.5, "раз"),
            (40.0, 41.5, "два три"),
        ]
        assert [(w.start, w.end) for w in t.segments[1].words] == [(40.0, 40.3), (41.0, 41.5)]
        assert t.segments[0].avg_logprob == -0.3 and t.segments[0].no_speech_prob == 0.01
        assert t.duration == 50 and t.language == "ru" and t.asr_model == "large-v3-turbo"

    def test_segment_without_words(self, fake: type[FakeModel], tmp_path: Path) -> None:
        fake.segments = [_segment(None, 6.0, 7.0, " текст "), _segment(None, 7.0, 7.5, "  ")]
        t = _asr().transcribe(_silence_wav(tmp_path / "a.wav", 50), speech=SPEECH)
        assert [(s.start, s.end, s.text) for s in t.segments] == [(41.0, 42.0, "текст")]

    def test_no_vad_means_whole_file(self, fake: type[FakeModel], tmp_path: Path) -> None:
        fake.segments = [_segment([_word(30.0, 30.4, " слово", prob=1.0000001)])]
        asr = _asr()
        t = asr.transcribe(_silence_wav(tmp_path / "a.wav", 50), terms=["Фурье", "FFT"])
        assert asr._model.calls[0]["samples"] == 50 * SR
        assert asr._model.calls[0]["hotwords"] == "Фурье, FFT"
        assert t.segments[0].start == 30.0
        assert t.segments[0].words[0].prob == 1.0

    def test_empty_speech_skips_model(self, fake: type[FakeModel], tmp_path: Path) -> None:
        asr = _asr()
        t = asr.transcribe(_silence_wav(tmp_path / "a.wav", 5), language="ru", speech=[])
        assert t.segments == [] and asr._model.calls == []


@pytest.mark.slow
def test_real_model_keeps_word_on_its_source_second(
    compose: Callable[..., list[tuple[float, float]]], tmp_path: Path, cached_whisper: ASRSettings
) -> None:
    """Сквозная проверка ловушки #10/#11 на настоящем Whisper: речь на 37-й секунде исходника
    после VAD и склейки распознаётся на 37-й секунде, а не на нулевой."""
    from summuraze.audio.vad import detect_speech
    from summuraze.config import VADSettings

    cfg = cached_whisper
    spans = compose(
        tmp_path / "rec.m4a", [("silence", 37), ("speech",), ("silence", 20), ("speech2",)]
    )
    info = ingest(tmp_path / "rec.m4a", tmp_path / "rec.wav")
    speech = detect_speech(info.wav, VADSettings())

    t = FasterWhisperASR(cfg).transcribe(info.wav, language="ru", speech=speech)

    # Слова, а не точный текст: robotic-голос espeak Whisper слышит с ошибками («фурии»).
    text = t.text.lower()
    assert "сегодня" in text and "быстрый" in text, text
    for (start, end), first in zip(
        spans, _first_words_after(t, [s for s, _ in spans]), strict=True
    ):
        assert first.start == pytest.approx(start, abs=0.7)
        assert first.start < end
    assert all(w.start <= w.end for s in t.segments for w in s.words)


def _first_words_after(t: Any, starts: list[float]) -> list[Any]:
    words = [w for s in t.segments for w in s.words]
    return [min((w for w in words if w.start >= s - 1), key=lambda w: w.start) for s in starts]


def test_hotwords_cut_to_budget_with_warning(fake: type[FakeModel], tmp_path: Path) -> None:
    from loguru import logger

    warnings: list[str] = []
    sink = logger.add(warnings.append, level="WARNING", format="{message}")
    try:
        asr = _asr()
        terms = ["Фурье", "Котельников", "окно Хэмминга", "Найквист", "Лаплас", "Тьюки"]
        asr.transcribe(_silence_wav(tmp_path / "a.wav", 5), terms=terms)
    finally:
        logger.remove(sink)
    # 6 слов × 10 = 60 токенов — ровно лимит; седьмое слово уже не влезает.
    assert asr._model.calls[0]["hotwords"] == "Фурье, Котельников, окно Хэмминга, Найквист, Лаплас"
    assert len(warnings) == 1 and "1 терминов из 6" in warnings[0] and "Тьюки" in warnings[0]


def test_no_terms_no_hotwords(fake: type[FakeModel], tmp_path: Path) -> None:
    asr = _asr()
    asr.transcribe(_silence_wav(tmp_path / "a.wav", 5))
    assert asr._model.calls[0]["hotwords"] is None


@pytest.mark.slow
def test_real_model_hotwords_fix_term(
    compose: Callable[..., list[tuple[float, float]]], tmp_path: Path, cached_whisper: ASRSettings
) -> None:
    """espeak-«Фурье» без подсказки Whisper слышит как «фурии»; с подсказкой — пишет правильно."""
    compose(tmp_path / "rec.m4a", [("silence", 1), ("speech",)])
    wav = ingest(tmp_path / "rec.m4a", tmp_path / "rec.wav").wav
    asr = FasterWhisperASR(cached_whisper)
    plain = asr.transcribe(wav, language="ru").text
    hinted = asr.transcribe(wav, language="ru", terms=["Фурье"]).text
    assert "Фурье" not in plain, plain
    assert "Фурье" in hinted, hinted


def test_local_only_never_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """--local-only (#20): модель только с диска; нет её — понятная ошибка, а не скачивание."""
    import faster_whisper

    from summuraze.asr.base import ASRModelMissing

    seen: list[bool] = []

    def missing(name: str, **kw: Any) -> None:
        seen.append(kw["local_files_only"])
        raise FileNotFoundError("нет в кэше HF")

    monkeypatch.setattr(faster_whisper, "WhisperModel", missing)
    with pytest.raises(ASRModelMissing, match="запрещает её скачать"):
        _asr(model="small", local_only=True)
    with pytest.raises(FileNotFoundError):
        _asr(model="small")
    assert seen == [True, False]
