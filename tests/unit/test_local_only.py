"""Запрет сети и согласие на отправку наружу (#20).

«Локальный STT» приватности не даёт: саммари по API увозит транскрипт целиком. Поэтому проверка —
на уровне сокетов: под `--local-only` процесс не открывает ни одного соединения.
"""

from __future__ import annotations

import shutil
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from summuraze import cli
from summuraze.cli import app
from summuraze.config import LLMSettings, Settings, get_settings
from summuraze.pipeline.orchestrator import Pipeline
from summuraze.store.db import Database
from summuraze.summarize.base import NetworkForbidden, make_llm
from tests.conftest import SPEECH_CLIPS

runner = CliRunner()
REMOTE = "https://openrouter.ai/api/v1"


@pytest.fixture
def connections(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[object]]:
    """Каждая попытка соединения записывается и проваливается — в сеть тест не уходит никогда."""
    attempts: list[object] = []

    def refuse(self: socket.socket, address: object) -> None:
        attempts.append(address)
        raise OSError("сеть запрещена тестом")

    def refuse_lookup(host: object, *a: object, **kw: object) -> list[object]:
        attempts.append(host)
        raise OSError("сеть запрещена тестом")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse_lookup)
    yield attempts


@pytest.fixture
def remote_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUMMURAZE_LLM_PROVIDER", "openai-compat")
    monkeypatch.setenv("SUMMURAZE_LLM_BASE_URL", REMOTE)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    get_settings.cache_clear()


@pytest.fixture
def lecture(tmp_path: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("нужен ffmpeg")
    path = tmp_path / "лекция.opus"
    shutil.copy(SPEECH_CLIPS["speech"], path)
    return path


def _runs(tmp_path: Path) -> int:
    with Database(tmp_path / "data" / "summuraze.sqlite") as db:
        return len(db.recent_runs())


def test_factory_refuses_remote_even_when_called_directly(connections: list[object]) -> None:
    """Фабрика — последний рубеж: мимо preflight и валидатора конфига тоже не пройти."""
    cfg = LLMSettings(provider="openai-compat", base_url=REMOTE, api_key="k", local_only=True)
    with pytest.raises(NetworkForbidden, match="уходит наружу"):
        make_llm(cfg)
    assert connections == []


def test_local_only_flag_with_remote_llm_fails_before_anything(
    remote_llm: None, lecture: Path, tmp_path: Path, connections: list[object]
) -> None:
    result = runner.invoke(app, ["run", str(lecture), "--local-only"])
    assert result.exit_code == 1
    assert "--local-only" in result.output and "уходит наружу" in result.output
    assert connections == [] and not (tmp_path / "data" / "summuraze.sqlite").exists(), (
        "ни ингеста, ни записи в истории — отказ на старте"
    )


def test_local_only_run_opens_no_sockets(
    lecture: Path, tmp_path: Path, connections: list[object]
) -> None:
    result = runner.invoke(app, ["run", str(lecture), "--local-only"])
    assert result.exit_code == 0, result.output
    assert "Заметка:" in result.output
    assert connections == []


def test_first_remote_send_needs_terminal(
    remote_llm: None, lecture: Path, tmp_path: Path, connections: list[object]
) -> None:
    result = runner.invoke(app, ["run", str(lecture)])
    assert result.exit_code == 1
    assert "openrouter.ai" in result.output and "разрешить в терминале" in result.output
    assert "Аудио наружу не уходит" in result.output
    assert connections == [] and _runs(tmp_path) == 0, "спросили до транскрибации"


def test_declined_consent_stops_run(
    remote_llm: None,
    lecture: Path,
    tmp_path: Path,
    connections: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    result = runner.invoke(app, ["run", str(lecture)], input="n\n")
    assert result.exit_code == 1 and "Не отправляю" in result.output
    assert connections == [] and _runs(tmp_path) == 0


def test_consent_is_asked_once_per_host(
    remote_llm: None, monkeypatch: pytest.MonkeyPatch, connections: list[object]
) -> None:
    asked: list[str] = []
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    monkeypatch.setattr(cli.typer, "confirm", lambda text, **kw: asked.append(text) or True)
    with Pipeline(get_settings()) as pipeline:
        cli._confirm_remote(pipeline)
        cli._confirm_remote(pipeline)
        assert pipeline.db.has_consent("openrouter.ai")
    with Pipeline(get_settings()) as pipeline:
        cli._confirm_remote(pipeline)
    assert len(asked) == 1, "ответ запомнен в БД, а не в памяти процесса"

    other = Settings().llm.model_copy(update={"base_url": "https://api.deepseek.com/v1"})
    with Pipeline(Settings().model_copy(update={"llm": other})) as pipeline:
        cli._confirm_remote(pipeline)
    assert len(asked) == 2, "другой хост — другой вопрос"
    assert connections == []


def test_local_llm_needs_no_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUMMURAZE_LLM_PROVIDER", "openai-compat")
    monkeypatch.setenv("SUMMURAZE_LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    get_settings.cache_clear()
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    with Pipeline(get_settings()) as pipeline:
        cli._confirm_remote(pipeline)  # не спрашивает и не падает
        assert not pipeline.db.has_consent("127.0.0.1")
