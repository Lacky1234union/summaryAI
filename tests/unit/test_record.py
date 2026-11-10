"""Запись: две дорожки в один файл, уровни, остановка, переживание падения, сквозной `record`.

Звукового сервера в CI нет: вместо PipeWire ffmpeg получает синтетические входы `lavfi` —
команда, разбор уровней и остановка те же, меняются только `-f pulse -i …`.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from summuraze.audio import pairs
from summuraze.audio import record as rec
from summuraze.audio.ingest import ingest
from summuraze.cli import _meter, app
from tests.conftest import SPEECH_CLIPS

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")

TONE = ["-re", "-f", "lavfi", "-i", "sine=f=300:r=48000"]
SILENCE = ["-re", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono"]


def _streams(path: Path) -> list[str]:
    out = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-show_entries", "stream=codec_name:stream_tags=title"),
            *("-of", "csv=p=0", str(path)),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def test_two_tracks_levels_and_stop(tmp_path: Path) -> None:
    out = tmp_path / "rec.mka"
    seen = []
    with rec.recording(out, [TONE, SILENCE], ["микрофон", "система"], bitrate_kbps=32) as r:
        for levels in r.levels():
            seen.append(levels)
            if levels.at >= 1:
                break
    assert _streams(out) == ["opus,микрофон", "opus,система"]
    assert all(lv.silent() == (False, True) for lv in seen), "тон слышен, тишина — нет"
    duration = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert duration != "N/A" and float(duration) >= 1, "после остановки длительность дописана"


def test_finalize_tolerates_broken_file(tmp_path: Path) -> None:
    broken = tmp_path / "b.mka"
    broken.write_bytes(b"not a matroska")
    rec.finalize(broken)
    assert broken.read_bytes() == b"not a matroska" and not (tmp_path / "b.mka.part").exists()
    rec.finalize(tmp_path / "missing.mka")


def test_kill_minus_nine_keeps_recorded(tmp_path: Path) -> None:
    """Падение посреди пары: всё, кроме последней секунды, остаётся читаемым."""
    out = tmp_path / "crash.mka"
    r = rec.Recording(rec.build_command([TONE, TONE], ["a", "b"], out, bitrate_kbps=32), 2)
    for levels in r.levels():
        if levels.at >= 3:
            break
    os.kill(r._proc.pid, signal.SIGKILL)
    r._proc.wait()
    info = ingest(out, tmp_path / "crash.wav")
    assert info.audio_streams == 2 and info.duration >= 1.5


def test_ffmpeg_failure_is_reported(tmp_path: Path) -> None:
    bad = ["-f", "lavfi", "-i", "no_such_filter_xyz"]
    with (
        pytest.raises(rec.RecordError, match="ffmpeg прервал запись"),
        rec.recording(tmp_path / "x.mka", [bad], ["a"], bitrate_kbps=32) as r,
    ):
        list(r.levels())


def test_single_track(tmp_path: Path) -> None:
    out = tmp_path / "mic.mka"
    with rec.recording(out, [TONE], ["микрофон"], bitrate_kbps=32) as r:
        for levels in r.levels():
            assert len(levels.peaks) == 1
            break
    assert _streams(out) == ["opus,микрофон"]


def test_parse_levels() -> None:
    lines = iter(
        [
            "frame:0    pts:0       pts_time:0",
            "lavfi.astats.1.Peak_level=-18.5",
            "lavfi.astats.2.Peak_level=-inf",
            "frame:1    pts:12000   pts_time:0.25",
            "lavfi.astats.1.Peak_level=-60",
            "lavfi.astats.2.Peak_level=-3.25",
        ]
    )
    assert list(rec.parse_levels(lines, 2)) == [
        rec.Levels(0.25, (-18.5, -120.0)),
        rec.Levels(0.5, (-60.0, -3.25)),
    ]


def test_silence_watch() -> None:
    watch = rec.SilenceWatch(minutes=1)
    quiet, loud = (-90.0, -90.0), (-20.0, -90.0)
    assert not watch.update(rec.Levels(10, quiet))
    assert not watch.update(rec.Levels(60, quiet))
    assert not watch.update(rec.Levels(65, loud)), "звук сбрасывает отсчёт"
    assert not watch.update(rec.Levels(70, quiet))
    assert watch.update(rec.Levels(130, quiet))
    assert not rec.SilenceWatch(minutes=0).update(rec.Levels(10_000, quiet))


def test_record_name(tmp_path: Path) -> None:
    now = lambda: 1_761_652_800.0  # noqa: E731
    first = rec.record_name(tmp_path, "ЦОС: лекция #3", now)
    assert first.name == "ЦОС_ лекция _3.mka"
    first.write_bytes(b"x")
    second = rec.record_name(tmp_path, "ЦОС: лекция #3", now)
    assert second != first and second.name.startswith("ЦОС_ лекция _3 20")
    assert rec.record_name(tmp_path, None, now).name.startswith("запись 20")


def test_default_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = {"get-default-source": "mic0", "get-default-sink": "spk0"}
    monkeypatch.setattr(rec, "_pactl", lambda *a: answers[a[0]])
    assert rec.default_sources(mic=None, system=None) == [
        rec.Source("mic0", "микрофон"),
        rec.Source("spk0.monitor", "система"),
    ]
    assert rec.default_sources(mic="usb", system="NONE") == [rec.Source("usb", "микрофон")]


def test_muted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rec, "_pactl", lambda *a: "Mute: yes")
    assert rec.muted("mic0")

    def broken(*a: str) -> str:
        raise rec.RecordError("нет pactl")

    monkeypatch.setattr(rec, "_pactl", broken)
    assert not rec.muted("mic0"), "не смогли спросить — не пугаем"


def test_meter() -> None:
    sources = [rec.Source("m", "микрофон"), rec.Source("s", "система")]
    line = _meter(3725, sources, (-12.0, -90.0))
    assert line.startswith("● 01:02:05")
    assert "микрофон ▮▮▮▮▮▮▮▮▯▯  -12 дБ" in line and "система ▯▯▯▯▯▯▯▯▯▯  тишина" in line


def _fake_inhibit(path: Path, body: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Подменный systemd-inhibit: пишет свои аргументы и делает `body`."""
    args = path.parent / "args"
    path.write_text(f'#!/bin/sh\necho "$@" > {args}\n{body}\n')
    path.chmod(0o755)
    monkeypatch.setattr(rec.shutil, "which", lambda name: str(path))
    return args


def test_inhibit_held_and_released(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pid_file = tmp_path / "pid"
    body = f'echo $$ > {pid_file}; while [ "${{1#--}}" != "$1" ]; do shift; done; exec "$@"'
    args = _fake_inhibit(tmp_path / "systemd-inhibit", body, monkeypatch)
    with rec.inhibit_sleep() as held:
        assert held
        line = args.read_text()
        assert "--what=sleep:idle:handle-lid-switch" in line and "sleep infinity" in line
        pid = int(pid_file.read_text())
        os.kill(pid, 0)  # живой: блокировка держится
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # снята вместе с `sleep infinity`


def test_inhibit_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Отказ polkit: systemd-inhibit выходит сразу — блокировки нет, запись идёт."""
    _fake_inhibit(tmp_path / "systemd-inhibit", "exit 1", monkeypatch)
    with rec.inhibit_sleep() as held:
        assert not held


def test_inhibit_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rec.shutil, "which", lambda name: None)
    with rec.inhibit_sleep() as held:
        assert not held


@contextmanager
def _no_inhibit(why: str = "") -> Iterator[bool]:
    yield True


class TestCLI:
    @pytest.fixture(autouse=True)
    def _keep_real_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Тесты не должны запрещать сон машине, на которой бегут."""
        monkeypatch.setattr(rec, "inhibit_sleep", _no_inhibit)

    def _fake_inputs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """«Микрофон» — синтетическая речь, «система» — тишина; запись кончается сама."""
        # Без -re: в реальном времени здесь проверять нечего, а набор тестов ждал бы секунды.
        speech = ["-i", str(SPEECH_CLIPS["speech"])]
        silence = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono:d=6"]
        monkeypatch.setattr(
            rec, "_pactl", lambda *a: "Mute: no" if a[0] == "get-source-mute" else "dev"
        )
        monkeypatch.setattr(rec, "pulse_inputs", lambda sources: [speech, silence][: len(sources)])

    def test_record_then_note(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fake_inputs(monkeypatch)
        monkeypatch.setenv("SUMMURAZE_RECORD_DIR", str(tmp_path / "rec"))
        result = CliRunner().invoke(app, ["record", "--name", "пара 1"])
        assert result.exit_code == 0, result.output
        recorded = tmp_path / "rec" / "пара 1.mka"
        assert _streams(recorded) == ["opus,микрофон", "opus,система"]
        assert "Заметка:" in result.output and "[4/7] саммари" in result.output

    def test_no_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fake_inputs(monkeypatch)
        monkeypatch.setenv("SUMMURAZE_RECORD_DIR", str(tmp_path / "rec"))
        result = CliRunner().invoke(app, ["record", "--no-run", "--mic-only"])
        assert result.exit_code == 0, result.output
        [recorded] = (tmp_path / "rec").glob("*.mka")
        assert _streams(recorded) == ["opus,микрофон"]
        assert "Заметка" not in result.output

    def test_preflight_fails_before_recording(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Пустой ключ — до записи: узнать о нём после двухчасовой пары поздно."""
        self._fake_inputs(monkeypatch)
        monkeypatch.setenv("SUMMURAZE_RECORD_DIR", str(tmp_path / "rec"))
        monkeypatch.setenv("SUMMURAZE_LLM_PROVIDER", "openai-compat")
        result = CliRunner().invoke(app, ["record"])
        assert result.exit_code == 1 and "OPENROUTER_API_KEY" in result.output
        assert "● Запись" not in result.output and not (tmp_path / "rec").exists()

    def test_muted_mic_warns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fake_inputs(monkeypatch)
        monkeypatch.setattr(
            rec, "_pactl", lambda *a: "Mute: yes" if a[0] == "get-source-mute" else "dev"
        )
        monkeypatch.setenv("SUMMURAZE_RECORD_DIR", str(tmp_path / "rec"))
        result = CliRunner().invoke(app, ["record", "--no-run"])
        assert "микрофон выключен в системе" in result.output

    def test_silence_auto_stop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Забыл остановить после пары — запись встаёт сама после N минут тишины."""
        quiet = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono:d=600"]
        monkeypatch.setattr(rec, "_pactl", lambda *a: "dev")
        monkeypatch.setattr(rec, "pulse_inputs", lambda sources: [quiet, quiet])
        monkeypatch.setenv("SUMMURAZE_RECORD_DIR", str(tmp_path / "rec"))
        monkeypatch.setenv("SUMMURAZE_RECORD_SILENCE_STOP_MINUTES", "0.25")
        result = CliRunner().invoke(app, ["record", "--no-run"])
        assert "Тишина 0.25 мин — останавливаю запись" in result.output, result.output
        assert "микрофон: тишина 10 с" in result.output
        [recorded] = (tmp_path / "rec").glob("*.mka")
        assert recorded.stat().st_size > 0

    def test_pair_stops_itself(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Конец пары — запись встаёт сама; время потока, поэтому без ожидания в реальном."""
        quiet = ["-f", "lavfi", "-i", "sine=f=300:r=48000:d=600"]
        monkeypatch.setattr(rec, "_pactl", lambda *a: "dev")
        monkeypatch.setattr(rec, "pulse_inputs", lambda sources: [quiet, quiet][: len(sources)])
        monkeypatch.setenv("SUMMURAZE_RECORD_DIR", str(tmp_path / "rec"))
        seen: list[tuple[list[object], float, float]] = []

        def soon(now: datetime, starts: list[object], *, minutes: float, margin: float) -> object:
            seen.append((starts, minutes, margin))
            return pairs.StopPlan(now + timedelta(seconds=3), None, "пара 13:00–14:20, стоп")

        monkeypatch.setattr(pairs, "plan_stop", soon)
        result = CliRunner().invoke(app, ["record", "--no-run", "--mic-only", "--pair"])
        assert result.exit_code == 0, result.output
        assert "пара 13:00–14:20, стоп. Ctrl+C — остановить раньше" in result.output
        assert "Время вышло — останавливаю запись" in result.output
        assert seen[0][1:] == (80, 10) and len(seen[0][0]) == 7, "сетка ВШЭ по умолчанию"
        [recorded] = (tmp_path / "rec").glob("*.mka")
        duration = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"]
            + [str(recorded)],
            capture_output=True,
            text=True,
        ).stdout
        assert float(duration) < 600, "встала сама, а не по концу входа"

    def test_bad_until(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SUMMURAZE_RECORD_DIR", str(tmp_path / "rec"))
        result = CliRunner().invoke(app, ["record", "--no-run", "--until", "полтретьего"])
        assert result.exit_code == 2 and "ЧЧ:ММ" in result.output
        assert not (tmp_path / "rec").exists(), "до записи"

    def test_no_inhibit_warns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        @contextmanager
        def refused(why: str = "") -> Iterator[bool]:
            yield False

        self._fake_inputs(monkeypatch)
        monkeypatch.setattr(rec, "inhibit_sleep", refused)
        monkeypatch.setenv("SUMMURAZE_RECORD_DIR", str(tmp_path / "rec"))
        result = CliRunner().invoke(app, ["record", "--no-run"])
        assert "не закрывай крышку" in result.output
