"""Сквозной прогон на заглушках: стадии соединены, кэш работает, провайдеры не грузятся зря."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from summuraze.asr import repetition
from summuraze.asr.echo import EchoASR
from summuraze.audio.ingest import IngestError
from summuraze.audio.vad import NoSpeechError
from summuraze.cli import app
from summuraze.config import ASRSettings, LLMSettings, Settings, get_settings
from summuraze.pipeline.orchestrator import Pipeline, StageEvent
from summuraze.pipeline.stages import SUSPECT_RULE, transcript_prefix
from summuraze.render.markdown import fmt_timecode, short_timecode
from summuraze.store.db import Database
from summuraze.summarize.echo import EchoLLM
from summuraze.types import ASRProvider, LLMProvider, LLMResult, Segment, Transcript

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")

# Речь двумя кусками с паузой: секунды известны, по ним проверяется смещение таймкодов.
LAYOUT = [("silence", 10), ("speech",), ("silence", 10), ("speech2",), ("silence", 5)]


@pytest.fixture(scope="module")
def spans(
    tmp_path_factory: pytest.TempPathFactory, compose: Callable[..., list[tuple[float, float]]]
) -> tuple[Path, list[tuple[float, float]]]:
    path = tmp_path_factory.mktemp("media") / "lecture.m4a"
    return path, compose(path, LAYOUT)


@pytest.fixture
def lecture(spans: tuple[Path, list[tuple[float, float]]]) -> Path:
    return spans[0]


class Counting:
    """Фабрики, считающие, сколько раз провайдер действительно создавался."""

    def __init__(self) -> None:
        self.asr = 0
        self.llm = 0

    def make_asr(self, cfg: ASRSettings) -> ASRProvider:
        self.asr += 1
        return EchoASR()

    def make_llm(self, cfg: LLMSettings) -> LLMProvider:
        self.llm += 1
        return EchoLLM()


def _pipeline(counting: Counting, events: list[StageEvent] | None = None) -> Pipeline:
    return Pipeline(
        Settings(),
        asr_factory=counting.make_asr,
        llm_factory=counting.make_llm,
        on_event=events.append if events is not None else None,
    )


def test_full_run_produces_note(lecture: Path) -> None:
    with _pipeline(Counting()) as p:
        result = p.run(lecture)
    assert result.note is not None and result.note.is_file()
    text = result.note.read_text(encoding="utf-8")
    assert "llm_model" not in text and "> [!quote]- Транскрипт" in text
    assert result.transcript is not None
    first = result.transcript.segments[0]
    assert f"#t={first.start:.1f}|▶ {short_timecode(first.start)}]] эхо-сегмент 1" in text


def test_asr_sees_only_speech_with_source_timecodes(
    spans: tuple[Path, list[tuple[float, float]]],
) -> None:
    """Таймкоды после VAD — от начала исходника: вторая речь на ~25-й секунде, а не на 6-й."""
    lecture, truth = spans
    with _pipeline(Counting()) as p:
        result = p.run(lecture, until="asr")
    assert result.speech is not None and result.transcript is not None
    segments = result.transcript.segments
    assert [round(s.start) for s in segments] == [round(start) for start, _ in truth]
    pause = (truth[0][1] + 1, truth[1][0] - 1)
    assert not any(s.start < pause[1] and pause[0] < s.end for s in segments)


def test_vad_disabled_gives_asr_whole_file(lecture: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUMMURAZE_VAD_ENABLED", "false")
    events: list[StageEvent] = []
    with _pipeline(Counting(), events) as p:
        result = p.run(lecture, until="asr")
    assert result.speech is None and result.transcript is not None
    assert result.transcript.segments[0].start == 0
    assert {e.stage for e in events} == {"ingest", "asr"}
    assert {e.total for e in events} == {2}


def test_vad_param_change_reruns_vad_and_asr_but_not_ingest(
    lecture: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counting = Counting()
    with _pipeline(counting) as p:
        p.run(lecture, until="asr")
    monkeypatch.setenv("SUMMURAZE_VAD_MIN_SILENCE_MS", "20000")
    with _pipeline(counting) as p:
        result = p.run(lecture, until="asr")
    assert counting.asr == 2
    assert result.cached == {"ingest": True, "vad": False, "asr": False}
    assert result.speech is not None and len(result.speech) == 1


def test_no_speech_stops_before_asr(
    tmp_path: Path, compose: Callable[..., list[tuple[float, float]]]
) -> None:
    music = tmp_path / "music.m4a"
    compose(music, [("tone", 5)])
    counting = Counting()
    with _pipeline(counting) as p, pytest.raises(NoSpeechError, match="не найдено речи"):
        p.run(music)
    assert counting.asr == 0
    result = CliRunner().invoke(app, ["run", str(music)])
    assert result.exit_code == 1 and "не найдено речи" in result.output


def test_second_run_takes_everything_from_cache(lecture: Path) -> None:
    counting = Counting()
    with _pipeline(counting) as p:
        p.run(lecture)
        events: list[StageEvent] = []
        p._on_event = events.append
        second = p.run(lecture)

    assert (counting.asr, counting.llm) == (1, 1), "повторный прогон не должен создавать модели"
    assert second.cached == {
        "ingest": True,
        "vad": True,
        "asr": True,
        "summarize": True,
        "audio": True,
        "render": False,
    }
    finished = {e.stage: e.status for e in events if e.status != "start"}
    assert finished == {
        "ingest": "cached",
        "vad": "cached",
        "asr": "cached",
        "summarize": "cached",
        "audio": "cached",
        "render": "done",
    }


def test_asr_param_change_reruns_asr_but_not_ingest(
    lecture: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counting = Counting()
    with _pipeline(counting) as p:
        p.run(lecture)
    monkeypatch.setenv("SUMMURAZE_ASR_LANGUAGE", "en")
    with _pipeline(counting) as p:
        result = p.run(lecture)
    assert counting.asr == 2
    assert counting.llm == 2, "новый транскрипт — новое саммари: ключи идут цепочкой"
    assert result.cached is not None
    assert (result.cached["ingest"], result.cached["vad"]) == (True, True)


def test_threads_do_not_invalidate_transcript(
    lecture: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Число потоков — скорость, а не результат: час транскрибации не должен пропасть."""
    counting = Counting()
    with _pipeline(counting) as p:
        p.run(lecture, until="asr")
    monkeypatch.setenv("SUMMURAZE_ASR_THREADS", "4")
    with _pipeline(counting) as p:
        p.run(lecture, until="asr")
    assert counting.asr == 1


def test_until_asr_stops_before_llm(lecture: Path) -> None:
    counting = Counting()
    with _pipeline(counting) as p:
        result = p.run(lecture, until="asr")
    assert counting.llm == 0
    assert result.transcript is not None and result.note is None


def test_failed_stage_is_logged_and_not_cached(lecture: Path) -> None:
    class Broken(Counting):
        def make_asr(self, cfg: ASRSettings) -> ASRProvider:
            raise RuntimeError("модель не скачалась")

    with _pipeline(Broken()) as p, pytest.raises(RuntimeError):
        p.run(lecture)

    counting = Counting()
    with _pipeline(counting) as p:
        p.run(lecture)
    assert counting.asr == 1, "упавшая стадия не должна оставить ничего в кэше"

    with Database(Settings().paths.db_path) as db:
        statuses = [r.status for r in db.recent_runs()]
    assert statuses == ["done", "failed"]


def test_moved_file_hits_cache_and_note_follows_new_name(lecture: Path, tmp_path: Path) -> None:
    counting = Counting()
    with _pipeline(counting) as p:
        p.run(lecture)
        moved = Path(shutil.copy2(lecture, tmp_path / "переименованная.m4a"))
        result = p.run(moved)
    assert counting.asr == 1
    assert result.note is not None and "переименованная" in result.note.name
    assert result.audio.source == moved


def test_missing_file_fails_before_touching_db(tmp_path: Path) -> None:
    with _pipeline(Counting()) as p, pytest.raises(IngestError, match="не найден"):
        p.run(tmp_path / "nope.mp4")


def test_unknown_profile_fails_before_asr(lecture: Path) -> None:
    counting = Counting()
    with _pipeline(counting) as p, pytest.raises(ValueError, match="неизвестный профиль"):
        p.run(lecture, profile="nope")
    assert counting.asr == 0


def test_transcript_prefix_is_stable() -> None:
    """Префикс уходит модели первым: в нём не должно быть путей и времени прогона."""
    t = Transcript(
        source=Path("/where/ever.mp4"),
        duration=100,
        language="ru",
        asr_model="m",
        segments=[Segment(start=0, end=5, text=" раз "), Segment(start=3661, end=3670, text="два")],
    )
    other = t.model_copy(update={"source": Path("/elsewhere.mp4")})
    assert transcript_prefix(t) == "[00:00:00] раз\n[01:01:01] два"
    assert transcript_prefix(t) == transcript_prefix(other)


def test_fmt_timecode() -> None:
    assert fmt_timecode(0) == "00:00:00"
    assert fmt_timecode(754.9) == "00:12:34"
    assert fmt_timecode(4 * 3600 + 5) == "04:00:05"


class TestCLI:
    def test_run_twice_reports_cache(self, lecture: Path) -> None:
        runner = CliRunner()
        first = runner.invoke(app, ["run", str(lecture)])
        assert first.exit_code == 0, first.output
        assert "Заметка:" in first.output
        second = runner.invoke(app, ["run", str(lecture)])
        assert second.output.count("из кэша") == 5, second.output

    def test_transcribe_then_log(self, lecture: Path) -> None:
        runner = CliRunner()
        assert runner.invoke(app, ["transcribe", str(lecture)]).exit_code == 0
        out = runner.invoke(app, ["transcribe", str(lecture)]).output
        assert "речь:" in out and "интервалов: 2" in out, out
        out = runner.invoke(app, ["log"]).output
        assert "lecture.m4a" in out and "готово" in out and "00:35" in out
        assert "· echo" not in out, "без саммари строки LLM нет"
        assert runner.invoke(app, ["run", str(lecture)]).exit_code == 0
        out = runner.invoke(app, ["log"]).output
        assert "локально · echo" in out and "Итого за 3 прогонов" in out

    def test_log_when_empty(self) -> None:
        assert "ещё не было" in CliRunner().invoke(app, ["log"]).output

    def test_notes_go_to_output_dir(
        self, lecture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vault = tmp_path / "vault" / "Лекции"
        monkeypatch.setenv("SUMMURAZE_OUTPUT_DIR", str(vault))
        get_settings.cache_clear()
        missing = CliRunner().invoke(app, ["run", str(lecture)])
        assert missing.exit_code == 1 and "папки для заметок нет" in missing.output
        assert "[1/" not in missing.output, "отказ — до первой стадии, а не после ASR"

        vault.mkdir(parents=True)
        result = CliRunner().invoke(app, ["run", str(lecture)])
        assert result.exit_code == 0, result.output
        [note] = vault.glob("*.md")
        media = vault / "Записи"
        assert sorted(p.name for p in media.iterdir()) == [
            f"{note.stem}.ogg",
            f"{note.stem}.ru.srt",
        ]
        text = note.read_text(encoding="utf-8")
        assert f"![[{note.stem}.ogg]]" in text and f'audio: "[[{note.stem}.ogg]]"' in text

    def test_rerun_does_not_rewrite_unchanged_files(
        self, lecture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LiveSync рассылает изменённое: повторный прогон не должен трогать 15 МБ аудио."""
        vault = tmp_path / "vault"
        vault.mkdir()
        monkeypatch.setenv("SUMMURAZE_OUTPUT_DIR", str(vault))
        get_settings.cache_clear()
        runner = CliRunner()
        assert runner.invoke(app, ["run", str(lecture)]).exit_code == 0
        files = sorted(vault.rglob("*.*"))
        before = {p: p.stat().st_mtime_ns for p in files}
        assert runner.invoke(app, ["run", str(lecture)]).exit_code == 0
        assert {p: p.stat().st_mtime_ns for p in files} == before

    def test_note_audio_disabled(
        self, lecture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SUMMURAZE_NOTE_AUDIO", "false")
        get_settings.cache_clear()
        result = CliRunner().invoke(app, ["run", str(lecture)])
        assert result.exit_code == 0 and "аудио для заметки" not in result.output
        [note] = (tmp_path / "data" / "notes").glob("*.md")
        text = note.read_text(encoding="utf-8")
        assert "[[" not in text and "`00:09`" in text

    def test_unknown_profile_is_a_clean_error(self, lecture: Path) -> None:
        result = CliRunner().invoke(app, ["run", str(lecture), "--profile", "nope"])
        assert result.exit_code == 1
        assert "неизвестный профиль" in result.output

    def test_lang_flag_overrides_config(self, lecture: Path) -> None:
        runner = CliRunner()
        out = runner.invoke(app, ["transcribe", str(lecture), "--lang", "EN"]).output
        assert "язык en" in out, out
        out = runner.invoke(app, ["transcribe", str(lecture), "--lang", "auto"]).output
        assert "распознавание… из кэша" not in out, "другой язык — другой ключ кэша ASR"


class Hallucinating(Counting):
    """ASR, который вставляет шаблонную выдумку Whisper; LLM запоминает, что ей отправили."""

    def __init__(self) -> None:
        super().__init__()
        self.sent: list[dict[str, object]] = []

    def make_asr(self, cfg: ASRSettings) -> ASRProvider:
        self.asr += 1
        echo = EchoASR()

        class Stub:
            name = "stub"

            def transcribe(self, audio: Path, **kw: object) -> Transcript:
                t = echo.transcribe(audio, **kw)  # type: ignore[arg-type]
                fake = Segment(start=t.duration - 2, end=t.duration, text="Продолжение следует...")
                return t.model_copy(update={"segments": [*t.segments, fake]})

        return Stub()

    def make_llm(self, cfg: LLMSettings) -> LLMProvider:
        self.llm += 1
        llm, sent = EchoLLM(), self.sent

        class Recording:
            name = llm.name
            model = llm.model
            context_window = llm.context_window
            count_tokens = llm.count_tokens
            close = llm.close

            def complete(self, **kw: object) -> LLMResult:
                sent.append(kw)
                return llm.complete(**kw)  # type: ignore[arg-type]

        return Recording()  # type: ignore[return-value]


def test_suspect_segment_reaches_prompt_and_note(lecture: Path) -> None:
    h = Hallucinating()
    with _pipeline(h) as p:
        result = p.run(lecture)
    assert result.note is not None
    [call] = h.sent
    assert "[?] Продолжение следует" in str(call["prefix"])
    assert SUSPECT_RULE in str(call["system"])
    assert "⚠️ *Продолжение следует...* — шаблонная фраза" in result.note.read_text("utf-8")


def test_detector_change_redoes_summary_but_not_asr(
    lecture: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counting = Counting()
    with _pipeline(counting) as p:
        p.run(lecture)
    monkeypatch.setattr(repetition, "VERSION", repetition.VERSION + 1)
    with _pipeline(counting) as p:
        result = p.run(lecture)
    assert (counting.asr, counting.llm) == (1, 2)
    assert result.cached is not None and result.cached["asr"] is True


def test_glossary_terms_reach_asr_and_replacements_skip_it(lecture: Path, tmp_path: Path) -> None:
    """Термины — в ключе ASR (меняют распознавание); замены — поверх кэша, ASR не трогают."""
    seen: list[object] = []

    class Spy(Counting):
        def make_asr(self, cfg: ASRSettings) -> ASRProvider:
            self.asr += 1
            echo = EchoASR()

            class Rec:
                name = "spy"

                def transcribe(self, audio: Path, **kw: object) -> Transcript:
                    seen.append(kw["terms"])
                    return echo.transcribe(audio, **kw)  # type: ignore[arg-type]

            return Rec()

    course = tmp_path / "course.txt"
    course.write_text("Фурье\n", encoding="utf-8")
    spy = Spy()
    with _pipeline(spy) as p:
        p.run(lecture, until="asr")
        p.run(lecture, until="asr", terms=course)
        course.write_text("Фурье\nэхо -> Echo\n", encoding="utf-8")
        result = p.run(lecture, terms=course)
    assert seen == [None, ["Фурье"], ["Фурье", "Echo"]], "термины — в ключе ASR"
    assert result.transcript is not None
    assert result.transcript.segments[0].text.startswith("Echo-сегмент")

    # Поменялось только, как писать (термины те же) — ASR из кэша, саммари заново.
    course.write_text("Фурье\nEcho\nэхо -> echo\n", encoding="utf-8")
    with _pipeline(spy) as p:
        again = p.run(lecture, terms=course)
    assert len(seen) == 3, "правка замен не должна перезапускать распознавание"
    assert again.cached is not None and again.cached["summarize"] is False
    assert again.transcript is not None
    assert again.transcript.segments[0].text.startswith("echo-сегмент")


def test_other_profile_gets_own_note_and_reuses_asr(lecture: Path) -> None:
    counting = Counting()
    with _pipeline(counting) as p:
        main = p.run(lecture)
        raw = p.run(lecture, profile="raw")
    assert main.note is not None and raw.note is not None
    assert raw.note.name == main.note.name.replace(".md", " (raw).md")
    assert main.note.is_file(), "сырой профиль не затирает конспект"
    assert (counting.asr, counting.llm) == (1, 2)


def test_debug_flag_gives_full_note(lecture: Path, tmp_path: Path) -> None:
    """--debug — прежний вид (#50): модели, стоимость, транскрипт развёрнут, полные таймкоды."""
    result = CliRunner().invoke(app, ["run", str(lecture), "--debug"])
    assert result.exit_code == 0, result.output
    [note] = (tmp_path / "data" / "notes").glob("*.md")
    text = note.read_text(encoding="utf-8")
    assert 'llm_model: "echo"' in text and "## Транскрипт" in text and "|00:00:09]]" in text
