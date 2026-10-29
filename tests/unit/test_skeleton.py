"""Проверки каркаса: пакет импортируется, CLI собирается и отвечает, гейт живой."""

from __future__ import annotations

import subprocess
import sys
import time

from typer.testing import CliRunner

from summuraze import __version__
from summuraze.cli import app

runner = CliRunner()


def test_package_importable() -> None:
    assert __version__


def test_version_is_a_subcommand() -> None:
    """`summuraze version` должна быть подкомандой, а не всем приложением.

    При единственной команде Typer схлопывает приложение в неё — тогда `version`
    становится не подкомандой, а самим CLI, и добавление второй команды ломает интерфейс.
    Удерживается пустым callback'ом в cli.py.
    """
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.output


def test_cli_starts_fast() -> None:
    """CLI не должен грузить torch и faster-whisper на --help.

    Guardrail #1: вендорские импорты — лениво, внутри команд. Порог щадящий: ловим
    не медленный компьютер, а импорт рантайма STT на верхнем уровне (это секунды, не миллисекунды).
    """
    start = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-m", "summuraze.cli", "--help"],
        capture_output=True,
        timeout=30,
    )
    elapsed = time.monotonic() - start
    assert result.returncode == 0, result.stderr.decode()
    assert elapsed < 3.0, f"CLI стартует {elapsed:.1f} с — похоже, тяжёлый импорт наверху"


def test_cli_restores_ctrl_c_ignored_by_background_launch() -> None:
    """Запуск в фоне (`cmd &`) наследует SIGINT = SIG_IGN — без возврата обработчика ни Ctrl+C,
    ни «Остановить» в вебе прогон не останавливают (#64)."""
    import signal

    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        assert runner.invoke(app, ["version"]).exit_code == 0
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGINT, previous)
