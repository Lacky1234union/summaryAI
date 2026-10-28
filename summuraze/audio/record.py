"""Запись: микрофон и звук системы двумя дорожками в один файл (#37).

Один процесс ffmpeg читает два входа PipeWire (через его PulseAudio-слой): микрофон и монитор
вывода. Дорожки пишутся **раздельно**: для лекции ингест их сводит (ADR #5), а созвонам на
этапе 6 (#28) раздельные дорожки нужны в исходном виде — после записи их уже не разделить.

Контейнер — Matroska (`.mka`), кодек — Opus: Matroska пишется потоково, поэтому оборванная
запись (Ctrl+C, падение, выключенный ноутбук) читается до последнего кластера, а не теряется
целиком, как mp4 без финального `moov`. Opus 32 кбит/с на дорожку — ~29 МБ на час записи против
~115 МБ у wav (#49).

Уровни обеих дорожек считает тот же ffmpeg и печатает в stdout: немой микрофон виден в первые
секунды, а не после пары. Источники ffmpeg — параметр (`inputs`), поэтому тесты подставляют
синтетический `lavfi` и обходятся без звукового сервера.

Refs #37
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from summuraze.audio.ingest import IngestError

SAMPLE_RATE = 48_000
LEVEL_INTERVAL = 0.25  # секунды между отсчётами уровня
SILENCE_DB = -50.0  # пик ниже — тишина (шум выключенного микрофона ≈ −90 дБ)

_LEVEL = re.compile(r"lavfi\.astats\.(\d)\.Peak_level=(-?inf|-?[\d.]+)")


class RecordError(RuntimeError):
    """Запись не стартовала или оборвалась — сообщение для человека."""


@dataclass(frozen=True)
class Source:
    name: str  # имя источника PulseAudio/PipeWire
    title: str  # подпись дорожки в файле и на экране


@dataclass(frozen=True)
class Levels:
    """Пиковые уровни за последний отсчёт, дБ; порядок — как у источников."""

    at: float  # секунды от начала записи
    peaks: tuple[float, ...]

    def silent(self) -> tuple[bool, ...]:
        return tuple(p < SILENCE_DB for p in self.peaks)


def default_sources(*, mic: str | None, system: str | None) -> list[Source]:
    """Микрофон и монитор вывода по умолчанию; `system="none"` — только микрофон."""
    sources = [Source(mic or _pactl("get-default-source"), "микрофон")]
    if (system or "").lower() != "none":
        sources.append(Source(system or f"{_pactl('get-default-sink')}.monitor", "система"))
    return sources


def muted(source: str) -> bool:
    """Выключен ли источник программно — самая частая причина немой записи."""
    try:
        return "yes" in _pactl("get-source-mute", source).lower()
    except RecordError:
        return False


def pulse_inputs(sources: Sequence[Source]) -> list[list[str]]:
    return [
        ["-f", "pulse", "-sample_rate", str(SAMPLE_RATE), "-channels", "1", "-i", s.name]
        for s in sources
    ]


def build_command(
    inputs: Sequence[Sequence[str]], titles: Sequence[str], output: Path, *, bitrate_kbps: int
) -> list[str]:
    """ffmpeg: каждая дорожка — в файл отдельно, их уровни — одним потоком в stdout."""
    n = len(inputs)
    # Раскладка mono явно: без неё amerge не может согласовать форматы входов PulseAudio.
    split = "".join(
        f"[{i}:a]aresample={SAMPLE_RATE},aformat=channel_layouts=mono,asplit[rec{i}][lvl{i}];"
        for i in range(n)
    )
    meter_in = "".join(f"[lvl{i}]" for i in range(n))
    merge = f"{meter_in}amerge=inputs={n}," if n > 1 else f"{meter_in}"
    meter = (
        f"{merge}asetnsamples=n={int(SAMPLE_RATE * LEVEL_INTERVAL)},"
        "astats=metadata=1:reset=1:measure_overall=none:measure_perchannel=Peak_level,"
        "ametadata=mode=print:file=-:direct=1,anullsink"
    )
    cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y"]
    for args in inputs:
        cmd += list(args)
    cmd += ["-filter_complex", split + meter]
    for i, title in enumerate(titles):
        cmd += ["-map", f"[rec{i}]", f"-metadata:s:a:{i}", f"title={title}"]
    cmd += [
        *("-c:a", "libopus", "-b:a", f"{bitrate_kbps}k", "-application", "voip"),
        # Потоковая запись: без возврата к началу файла, кластер — секунда, сброс на диск после
        # каждого пакета. Без этого ffmpeg держит в буфере всё, и `kill -9` через 6 с оставлял
        # пустой файл. Так при падении теряется не больше секунды.
        *("-live", "1", "-cluster_time_limit", "1000", "-flush_packets", "1"),
        *("-f", "matroska", str(output)),
    ]
    return cmd


def parse_levels(lines: Iterator[str], channels: int) -> Iterator[Levels]:
    """Строки `ametadata=print` → отсчёты уровней (`-inf` → −120 дБ)."""
    current: dict[int, float] = {}
    count = 0
    for line in lines:
        match = _LEVEL.search(line)
        if not match:
            continue
        channel, value = int(match.group(1)), match.group(2)
        current[channel] = -120.0 if "inf" in value else float(value)
        if len(current) == channels:
            count += 1
            yield Levels(count * LEVEL_INTERVAL, tuple(current[c + 1] for c in range(channels)))
            current = {}


class Recording:
    """Идущая запись. `stop()` — штатная остановка: ffmpeg дописывает файл и выходит."""

    def __init__(self, cmd: list[str], channels: int) -> None:
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            # Своя группа процессов: Ctrl+C в терминале ловит Python и останавливает запись
            # сам, а не ffmpeg получает сигнал посреди кадра.
            start_new_session=True,
        )
        self._channels = channels
        self._stderr: list[str] = []
        self._drain = threading.Thread(target=self._read_stderr, daemon=True)
        self._drain.start()

    def levels(self) -> Iterator[Levels]:
        assert self._proc.stdout is not None
        yield from parse_levels(iter(self._proc.stdout.readline, ""), self._channels)

    def stop(self, timeout: float = 15.0) -> None:
        if self._proc.poll() is None:
            try:
                assert self._proc.stdin is not None
                self._proc.stdin.write("q")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                self._proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        # ffmpeg мог выйти сам (конец входа): закрыть канал явно, иначе BrokenPipeError
        # вылетит позже, при сборке мусора.
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.close()
        except OSError:
            pass
        self._drain.join(timeout=2)

    def check(self) -> None:
        """Ошибка ffmpeg — исключением с его последней строкой, а не молчаливо пустым файлом."""
        code = self._proc.poll()
        if code not in (None, 0, 255):  # 255 — штатный выход ffmpeg по `q`
            # Первая строка — причина; дальше ffmpeg пишет следствия («Nothing was written…»).
            reason = self._stderr[0] if self._stderr else f"код выхода {code}"
            raise RecordError(f"ffmpeg прервал запись: {reason}")

    def _read_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            if line.strip():
                self._stderr.append(line.strip())


@contextmanager
def recording(
    output: Path,
    inputs: Sequence[Sequence[str]],
    titles: Sequence[str],
    *,
    bitrate_kbps: int,
) -> Generator[Recording, None, None]:
    output.parent.mkdir(parents=True, exist_ok=True)
    rec = Recording(build_command(inputs, titles, output, bitrate_kbps=bitrate_kbps), len(inputs))
    try:
        yield rec
    finally:
        rec.stop()
    rec.check()
    finalize(output)


def finalize(path: Path) -> None:
    """Дописать длительность и индекс перемотки: потоковый режим их не пишет (`-live 1`).

    Перепаковка без перекодирования — секунды даже на часовой записи. Не вышло — оставляем как
    есть: файл и без индекса читается целиком, ингесту этого хватает.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return
    tmp = path.with_name(path.name + ".part")
    result = subprocess.run(
        [
            *(_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y"),
            *("-i", str(path), "-map", "0", "-c", "copy", "-f", "matroska", str(tmp)),
        ],
        capture_output=True,
    )
    if result.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
        tmp.replace(path)
    else:
        tmp.unlink(missing_ok=True)


@dataclass
class SilenceWatch:
    """Авто-стоп: все дорожки молчат дольше `minutes`. 0 — выключено."""

    minutes: float
    _since: float | None = None

    def update(self, levels: Levels) -> bool:
        if self.minutes <= 0:
            return False
        if all(levels.silent()):
            if self._since is None:
                self._since = levels.at
            return levels.at - self._since >= self.minutes * 60
        self._since = None
        return False


def record_name(directory: Path, name: str | None, now: Callable[[], float] = time.time) -> Path:
    stamp = time.strftime("%Y-%m-%d %H-%M", time.localtime(now()))
    stem = re.sub(r'[\\/:*?"<>|#^\[\]]', "_", name).strip() if name else f"запись {stamp}"
    path = directory / f"{stem}.mka"
    if path.exists():  # то же имя второй раз за день — не затираем прошлую лекцию
        path = directory / f"{stem} {stamp}.mka"
    return path


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise IngestError(
            "не найден ffmpeg: поставь ffmpeg (pacman -S ffmpeg / apt install ffmpeg)"
        )
    return path


def _pactl(*args: str) -> str:
    binary = shutil.which("pactl")
    if binary is None:
        raise RecordError(
            "нет pactl — запись через PipeWire/PulseAudio недоступна (pipewire-pulse)"
        )
    result = subprocess.run(
        [binary, *args], capture_output=True, text=True, env={**os.environ, "LC_ALL": "C"}
    )
    if result.returncode != 0:
        raise RecordError(f"pactl {' '.join(args)}: {result.stderr.strip() or result.returncode}")
    return result.stdout.strip()
