"""Срок хранения (#49): аудио уходит, заметка и транскрипт остаются, ссылки не ведут в пустоту."""

from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from summuraze.cli import app
from summuraze.config import get_settings
from summuraze.render.markdown import detach_audio
from summuraze.store import retention
from tests.conftest import SPEECH_CLIPS

runner = CliRunner()
NOW = datetime(2026, 10, 28, 12, 0).astimezone()


@pytest.mark.parametrize(("text", "days"), [("30d", 30), ("30", 30), ("2w", 14), ("1.5д", 1.5)])
def test_parse_days(text: str, days: float) -> None:
    assert retention.parse_days(text) == days


def test_cutoff() -> None:
    before = retention.cutoff(before="2026-10-01", older_than=None, retention_days=None, now=NOW)
    assert before == datetime(2026, 10, 1).astimezone()
    older = retention.cutoff(before=None, older_than="2w", retention_days=5, now=NOW)
    assert older == NOW - timedelta(days=14), "флаг важнее срока из конфига"
    assert retention.cutoff(
        before=None, older_than=None, retention_days=5, now=NOW
    ) == NOW - timedelta(days=5)
    for bad in [
        {"before": "1 октября"},
        {"older_than": "месяц"},
        {"before": "2026-10-01", "older_than": "3d"},
        {},
    ]:
        with pytest.raises(ValueError):
            retention.cutoff(
                before=bad.get("before"),
                older_than=bad.get("older_than"),
                retention_days=None,
                now=NOW,
            )


def test_detach_audio() -> None:
    note = (
        '---\ntitle: "Л"\naudio: "[[2026-10-01 пара.ogg]]"\ntags: ["summuraze"]\n---\n\n# Л\n\n'
        "![[2026-10-01 пара.ogg]]\n\n"
        "- тезис [[2026-10-01 пара.ogg#t=95.0|00:01:35]]\n"
        "- чужая [[другая.ogg#t=1.0|00:00:01]]\n"
    )
    out = detach_audio(note, "2026-10-01 пара.ogg", "2026-11-01")
    assert "- тезис `00:01:35`" in out
    assert "> [!info] Аудио удалено 2026-11-01" in out
    assert 'audio_removed: "2026-11-01"' in out and "audio:" not in out
    assert "[[другая.ogg#t=1.0|00:00:01]]" in out, "ссылки на другие записи не трогаем"
    assert detach_audio(out, "2026-10-01 пара.ogg", "2026-11-02") == out, "повтор — без изменений"


# ── Сквозные: прогон на заглушках, состаривание записи в БД, prune ───────────

pytestmark_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")


@pytest.fixture
def rec_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "rec"
    d.mkdir()
    monkeypatch.setenv("SUMMURAZE_RECORD_DIR", str(d))
    return d


def _processed(source: Path) -> Path:
    shutil.copy(SPEECH_CLIPS["speech"], source)
    result = runner.invoke(app, ["run", str(source)])
    assert result.exit_code == 0, result.output
    return source


def _age(tmp_path: Path, source: Path, days: float = 60) -> None:
    """Сделать вид, что запись впервые увидели `days` дней назад."""
    seen = (datetime.now().astimezone() - timedelta(days=days)).isoformat(timespec="seconds")
    conn = sqlite3.connect(tmp_path / "data" / "summuraze.sqlite")
    with conn:
        conn.execute(
            "UPDATE recordings SET first_seen = ? WHERE path = ?", (seen, str(source.resolve()))
        )
    conn.close()


def _audio_dirs(tmp_path: Path) -> list[Path]:
    root = tmp_path / "data" / "artifacts"
    return [p for stage in ("ingest", "note-audio") for p in root.glob(f"{stage}/??/*")]


@pytestmark_ffmpeg
def test_prune_shows_then_deletes_audio_keeps_text(tmp_path: Path, rec_dir: Path) -> None:
    source = _processed(rec_dir / "пара 1.opus")
    _age(tmp_path, source)
    notes = tmp_path / "data" / "notes"
    [note] = notes.glob("*.md")
    media = list((notes / "Записи").iterdir())
    assert {p.suffix for p in media} == {".ogg", ".srt"} and len(_audio_dirs(tmp_path)) == 2
    assert ".ogg#t=" in note.read_text()

    shown = runner.invoke(app, ["prune", "--older-than", "30d"])
    assert shown.exit_code == 0, shown.output
    assert "пара 1.opus" in shown.output and "Ничего не удалено" in shown.output
    assert source.exists() and all(p.exists() for p in media), "без --yes ничего не трогаем"

    done = runner.invoke(app, ["prune", "--older-than", "30d", "--yes"])
    assert done.exit_code == 0, done.output
    assert "заметок обновлено: 1" in done.output
    assert not source.exists() and not any(p.exists() for p in media)
    assert _audio_dirs(tmp_path) == []
    text = note.read_text()
    assert ".ogg" not in text.replace("audio_removed", "") and "Аудио удалено" in text
    assert list((tmp_path / "data" / "artifacts" / "asr").glob("??/*")), "транскрипт остался"
    assert "#" in runner.invoke(app, ["log"]).output, "история прогонов осталась"

    again = runner.invoke(app, ["prune", "--older-than", "30d", "--yes"])
    assert "удалять нечего" in again.output


@pytestmark_ffmpeg
def test_fresh_records_are_kept(tmp_path: Path, rec_dir: Path) -> None:
    source = _processed(rec_dir / "свежая.opus")
    result = runner.invoke(app, ["prune", "--before", "2000-01-01", "--yes"])
    assert "удалять нечего" in result.output and source.exists()


@pytestmark_ffmpeg
def test_foreign_original_needs_explicit_flag(tmp_path: Path, rec_dir: Path) -> None:
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    source = _processed(downloads / "чужая.opus")
    _age(tmp_path, source)
    result = runner.invoke(app, ["prune", "--older-than", "30d", "--yes"])
    assert "не трогаю (1)" in result.output
    assert source.exists() and _audio_dirs(tmp_path) == [], "кэш и хранилище — да, оригинал — нет"

    # Повторный прогон того же файла снова публикует аудио — и срок снова к нему применим.
    assert runner.invoke(app, ["run", str(source)]).exit_code == 0
    result = runner.invoke(app, ["prune", "--older-than", "30d", "--include-foreign", "--yes"])
    assert result.exit_code == 0, result.output
    assert not source.exists()


@pytestmark_ffmpeg
def test_retention_days_prunes_old_after_run(
    tmp_path: Path, rec_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = _processed(rec_dir / "старая.opus")
    _age(tmp_path, old, days=10)
    monkeypatch.setenv("SUMMURAZE_RETENTION_DAYS", "7")
    get_settings.cache_clear()
    new = rec_dir / "новая.opus"
    shutil.copy(SPEECH_CLIPS["speech2"], new)
    result = runner.invoke(app, ["run", str(new)])
    assert result.exit_code == 0, result.output
    assert "Срок хранения 7 дн.: удалено аудио 1 записей" in result.output
    assert not old.exists() and new.exists()
    assert [p.name for p in (tmp_path / "data" / "notes" / "Записи").glob("*.ogg")] == [
        p.name for p in (tmp_path / "data" / "notes" / "Записи").glob("*новая*.ogg")
    ]


def test_prune_without_term_is_a_clean_error() -> None:
    result = runner.invoke(app, ["prune"])
    assert result.exit_code == 1 and "не задан срок" in result.output
