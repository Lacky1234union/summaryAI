"""Сквозной прогон на заглушках: стадии соединены, кэш работает, провайдеры не грузятся зря."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from summuraze.asr.echo import EchoASR
from summuraze.audio.ingest import IngestError
from summuraze.cli import app
from summuraze.config import ASRSettings, LLMSettings, Settings, get_settings
from summuraze.pipeline.orchestrator import Pipeline, StageEvent
from summuraze.pipeline.stages import transcript_prefix
from summuraze.render.markdown import fmt_timecode
from summuraze.store.db import Database
from summuraze.summarize.echo import EchoLLM
from summuraze.types import ASRProvider, LLMProvider, Segment, Transcript

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")

DURATION = 65.0


@pytest.fixture(scope="module")
def lecture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("media") / "lecture.m4a"
    subprocess.run(
        [
            *("ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"),
            *("-f", "lavfi", "-i", f"sine=frequency=300:duration={DURATION}"),
            *("-c:a", "aac", str(path)),
        ],
        check=True,
    )
    return path


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
    assert "Эхо-саммари" in text
    assert "`00:01:00` эхо-сегмент 3" in text, "таймкоды должны идти от начала записи"
    assert result.audio.duration == pytest.approx(DURATION, abs=0.1)


def test_second_run_takes_everything_from_cache(lecture: Path) -> None:
    counting = Counting()
    with _pipeline(counting) as p:
        p.run(lecture)
        events: list[StageEvent] = []
        p._on_event = events.append
        second = p.run(lecture)

    assert (counting.asr, counting.llm) == (1, 1), "повторный прогон не должен создавать модели"
    assert second.cached == {"ingest": True, "asr": True, "summarize": True, "render": False}
    finished = {e.stage: e.status for e in events if e.status != "start"}
    assert finished == {
        "ingest": "cached",
        "asr": "cached",
        "summarize": "cached",
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
    assert result.cached is not None and result.cached["ingest"] is True


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
        assert second.output.count("из кэша") == 3, second.output

    def test_transcribe_then_log(self, lecture: Path) -> None:
        runner = CliRunner()
        assert runner.invoke(app, ["transcribe", str(lecture)]).exit_code == 0
        out = runner.invoke(app, ["log"]).output
        assert "lecture.m4a" in out and "done" in out

    def test_log_when_empty(self) -> None:
        assert "ещё не было" in CliRunner().invoke(app, ["log"]).output

    def test_notes_go_to_output_dir(
        self, lecture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vault = tmp_path / "vault" / "Лекции"
        monkeypatch.setenv("SUMMURAZE_OUTPUT_DIR", str(vault))
        get_settings.cache_clear()
        result = CliRunner().invoke(app, ["run", str(lecture)])
        assert result.exit_code == 0, result.output
        assert [p.suffix for p in vault.iterdir()] == [".md"]

    def test_unknown_profile_is_a_clean_error(self, lecture: Path) -> None:
        result = CliRunner().invoke(app, ["run", str(lecture), "--profile", "nope"])
        assert result.exit_code == 1
        assert "неизвестный профиль" in result.output
