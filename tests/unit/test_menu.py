"""Меню `summuraze` без аргументов (#38): пункты зовут те же команды, настройки — в тот же .env."""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from summuraze import cli, menu
from summuraze.cli import app
from summuraze.config import get_settings
from summuraze.pipeline.orchestrator import StageEvent
from tests.conftest import SPEECH_CLIPS

runner = CliRunner()


@pytest.fixture
def tty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(menu, "interactive", lambda: True)
    # Не настоящая ~/Recordings: её записи сдвинули бы номера пунктов.
    monkeypatch.setenv("SUMMURAZE_RECORD_DIR", str(tmp_path / "нет записей"))
    # Окно выбора файла есть не везде (в CI нет дисплея) — пункты меню не должны от этого плыть.
    monkeypatch.setattr(menu, "_dialog_available", lambda: False)


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Что меню отдало бы xdg-open — вместо настоящего запуска Obsidian."""
    calls: list[str] = []
    monkeypatch.setattr(menu, "_launch", lambda target: calls.append(target) or True)
    return calls


def test_not_a_terminal_gives_hint_instead_of_hanging() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "summuraze run" in result.output


def test_help_and_subcommands_still_work() -> None:
    assert "record" in runner.invoke(app, ["--help"]).output
    assert runner.invoke(app, ["version"]).exit_code == 0


def test_exit_and_eof(tty: None) -> None:
    assert runner.invoke(app, [], input="0\n").exit_code == 0
    assert runner.invoke(app, [], input="").exit_code == 0, "Ctrl+D — выход, не трейс"


def test_wrong_number_asks_again(tty: None) -> None:
    result = runner.invoke(app, [], input="7\n0\n")
    assert result.exit_code == 0
    assert "Номер от 0 до 4." in result.output and result.output.count("Что делаем?") == 1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")
def test_from_file_runs_pipeline_and_offers_note(
    tty: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opened: list[str]
) -> None:
    rec_dir = tmp_path / "rec"
    rec_dir.mkdir()
    shutil.copy(SPEECH_CLIPS["speech"], rec_dir / "пара 1.opus")
    (rec_dir / "заметки.txt").write_text("не запись")
    monkeypatch.setenv("SUMMURAZE_RECORD_DIR", str(rec_dir))
    # из файла → первая запись → конспект → открыть → выход
    result = runner.invoke(app, [], input="2\n1\n1\ny\n0\n")
    assert result.exit_code == 0, result.output
    assert "заметки.txt" not in result.output, "не-медиа в списке записей не показываем"
    assert "[4/7] саммари" in result.output
    [note] = (tmp_path / "data" / "notes").glob("*.md")
    assert f"Заметка: {note}" in result.output
    assert opened == [str(note)], "без хранилища — .md как есть"


def test_from_file_missing_path_returns_to_menu(tty: None, tmp_path: Path) -> None:
    # из файла → ввести путь → несуществующий → снова меню → выход
    result = runner.invoke(app, [], input=f"2\n1\n'{tmp_path}/нет.mp3'\n0\n")
    assert result.exit_code == 0
    assert "Файла нет" in result.output and result.output.count("Что делаем?") == 2


def test_failed_run_returns_to_menu(
    tty: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = tmp_path / "битая.mp3"
    broken.write_bytes(b"not audio")
    monkeypatch.setenv("SUMMURAZE_RECORD_DIR", str(tmp_path))
    result = runner.invoke(app, [], input="2\n1\n1\n0\n")
    assert result.exit_code == 0, result.output
    assert "Ошибка:" in result.output and result.output.count("Что делаем?") == 2


def test_interrupt_returns_to_menu(tty: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(name: str | None, **kw: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_record", interrupted)
    result = runner.invoke(app, [], input="1\nлекция\n\n0\n")
    assert result.exit_code == 0
    assert "Прервано." in result.output and result.output.count("Что делаем?") == 2


def test_record_passes_name_and_offers_note(
    tty: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opened: list[str]
) -> None:
    seen: list[tuple[str | None, object]] = []
    note = tmp_path / "n.md"

    def fake(name: str | None, **kw: object) -> Path:
        seen.append((name, kw.get("pair")))
        return note

    monkeypatch.setattr(cli, "_record", fake)
    result = runner.invoke(app, [], input="1\n\n\nn\n1\nЦОС 3\nn\nn\n0\n")
    assert result.exit_code == 0, result.output
    assert seen == [(None, True), ("ЦОС 3", False)], "пустое имя — по дате; пара — по умолчанию"
    assert opened == []


def test_recent_notes_open_in_obsidian(
    tty: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opened: list[str]
) -> None:
    vault = tmp_path / "Хранилище Obsidian"
    vault.mkdir()
    old, new = vault / "старая.md", vault / "новая лекция.md"
    old.write_text("1")
    new.write_text("2")
    past = time.time() - 3600
    os.utime(old, (past, past))
    monkeypatch.setenv("SUMMURAZE_OUTPUT_DIR", str(vault))
    result = runner.invoke(app, [], input="3\n1\n0\n")
    assert result.exit_code == 0, result.output
    assert result.output.index("новая лекция") < result.output.index("старая"), "свежие сверху"
    [uri] = opened
    assert uri.startswith("obsidian://open?path=") and "%20" in uri and " " not in uri


def test_recent_notes_empty(tty: None) -> None:
    assert "Заметок пока нет" in runner.invoke(app, [], input="3\n0\n").output


# ── Настройки ───────────────────────────────────────────────────────────────


@pytest.fixture
def real_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """conftest ставит провайдера echo переменной окружения — она перекрыла бы .env."""
    monkeypatch.delenv("SUMMURAZE_LLM_PROVIDER")
    get_settings.cache_clear()


def test_switch_to_local_and_back(
    tty: None, real_llm: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(menu, "_server_alive", lambda url: False)
    env = tmp_path / ".env"
    env.write_text("# мой комментарий\nSUMMURAZE_LLM_MODEL=old\nOPENROUTER_API_KEY=k\n")
    # настройки → локально (адрес и модель по умолчанию) → назад → выход
    result = runner.invoke(app, [], input="4\n2\n\n\n0\n0\n")
    assert result.exit_code == 0, result.output
    assert "не отвечает" in result.output
    text = env.read_text()
    assert text.startswith("# мой комментарий\nSUMMURAZE_LLM_MODEL=local\n")
    assert f"SUMMURAZE_LLM_BASE_URL={menu.LOCAL_URL}" in text
    get_settings.cache_clear()
    assert get_settings().llm.is_local

    result = runner.invoke(app, [], input="4\n1\n0\n0\n")
    assert "Сохранено" in result.output and "Ключа нет" not in result.output
    get_settings.cache_clear()
    llm = get_settings().llm
    assert not llm.is_local and llm.model == "deepseek/deepseek-v4.1-flash"


def test_openrouter_without_key_warns(tty: None, real_llm: None) -> None:
    result = runner.invoke(app, [], input="4\n1\n0\n0\n")
    assert "Ключа нет" in result.output


def test_invalid_combination_is_rolled_back(
    tty: None, real_llm: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """local_only + OpenRouter не проходит валидацию — .env остаётся как был."""
    env = tmp_path / ".env"
    before = f"SUMMURAZE_LOCAL_ONLY=true\nSUMMURAZE_LLM_BASE_URL={menu.LOCAL_URL}\n"
    env.write_text(before)
    result = runner.invoke(app, [], input="4\n1\n0\n0\n")
    assert result.exit_code == 0, result.output
    assert "Не сохранено" in result.output
    assert env.read_text() == before


def test_env_variable_shadowing_is_reported(tty: None, tmp_path: Path) -> None:
    # SUMMURAZE_LLM_PROVIDER=echo из окружения (conftest) перекрывает запись в .env.
    result = runner.invoke(app, [], input="4\n1\n0\n0\n")
    assert "перекрыто переменной окружения: SUMMURAZE_LLM_PROVIDER" in result.output


def test_output_dir(tty: None, tmp_path: Path) -> None:
    vault = tmp_path / "My Vault"
    vault.mkdir()
    result = runner.invoke(app, [], input=f"4\n3\n{tmp_path}/нет\n3\n'{vault}'\n0\n0\n")
    assert result.exit_code == 0, result.output
    assert "Папки нет" in result.output
    assert f'SUMMURAZE_OUTPUT_DIR="{vault}"' in (tmp_path / ".env").read_text()
    get_settings.cache_clear()
    assert get_settings().paths.notes_dir == vault, "кавычки .env читаются обратно"


def test_write_env_keeps_layout(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("# A=пример\nA=1\n\nB = 2\n")
    menu.write_env(env, {"A": "x y", "B": "3", "C": "4"})
    assert env.read_text() == '# A=пример\nA="x y"\n\nB=3\nC=4\n'


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  ", None),
        ("/a/b c.mp3", "/a/b c.mp3"),
        ("'/a/b c.mp3' ", "/a/b c.mp3"),
        ("/a/b\\ c.mp3", "/a/b c.mp3"),
        ("file:///a/%D0%BB%20c.mp3", "/a/л c.mp3"),
        ("'/a/unbalanced", "'/a/unbalanced"),
    ],
)
def test_parse_path(raw: str, expected: str | None) -> None:
    got = menu.parse_path(raw)
    assert (str(got) if got else None) == expected


def test_status_line(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "заглушка echo" in menu._status(get_settings())
    monkeypatch.delenv("SUMMURAZE_LLM_PROVIDER")
    get_settings.cache_clear()
    assert "API · deepseek/deepseek-v4.1-flash @ openrouter.ai" in menu._status(get_settings())


def test_dialog(monkeypatch: pytest.MonkeyPatch) -> None:
    class Done:
        def __init__(self, code: int, out: str) -> None:
            self.returncode, self.stdout = code, out

    monkeypatch.setattr(menu.subprocess, "run", lambda *a, **kw: Done(0, "/x/лекция.mp4\n"))
    assert menu._dialog() == Path("/x/лекция.mp4")
    monkeypatch.setattr(menu.subprocess, "run", lambda *a, **kw: Done(1, ""))
    assert menu._dialog() is None, "окно закрыли — назад"
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert not menu._dialog_available()


# ── Прогресс стадии ─────────────────────────────────────────────────────────


def test_stage_ticker_in_terminal(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Распознавание идёт десятки минут — в терминале строка стадии тикает, а не замирает."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    progress = cli._StageProgress()
    progress(StageEvent("asr", 3, 6, "start"))
    time.sleep(1.2)
    progress(StageEvent("asr", 3, 6, "done", 1.2))
    out = capsys.readouterr().out
    assert "\r[3/6] распознавание… 00:01" in out
    assert out.endswith("\r[3/6] распознавание… 1.2 с\033[K\n")
    assert not progress.halt()


def test_stage_line_outside_terminal(capsys: pytest.CaptureFixture[str]) -> None:
    progress = cli._StageProgress()
    progress(StageEvent("ingest", 1, 6, "start"))
    progress(StageEvent("ingest", 1, 6, "cached"))
    assert capsys.readouterr().out == "[1/6] ингест… из кэша\n"


def test_open_note_without_xdg_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(menu.shutil, "which", lambda name: None)
    menu.open_note(tmp_path / "n.md", get_settings())
    assert "нет xdg-open" in capsys.readouterr().out


def test_ctrl_c_works_again_after_recording() -> None:
    """`_record_loop` глушит SIGINT на остановку ffmpeg; после паузы Ctrl+C снова прерывает
    конвейер — иначе и меню после записи не закрылось бы по Ctrl+C."""
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        cli._absorb_echoed_sigint(window=0)
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)
