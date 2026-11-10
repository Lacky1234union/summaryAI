"""Запуск прогона из браузера (#23): тот же `summuraze run`, отдельным процессом.

Отдельный процесс, а не поток сервера: прогон идёт ровно тем путём, что из терминала (ядро одно),
его можно остановить сигналом, и модели Whisper и llama-server не живут в памяти веба после него.
Прогресс процесс пишет в SQLite сам (`runs.stage`), веб его только читает.

Всё, что в терминале спрашивается до транскрибации, здесь проверяется до запуска: preflight
(ключ, папка заметок, модель `--local`) и согласие на отправку транскрипта наружу (#20). У
процесса нет терминала — без согласия в БД он откажет, но узнать об этом через час Whisper поздно.

Тяжёлый прогон — один за раз: ASR и локальная LLM двух прогонов в 14 ГБ не влезут.
FIFO и watch живут в `queue.py` (#24); Launcher повторно проверяет занятость перед стартом.
"""

from __future__ import annotations

import os
import re
import signal
import statistics
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from summuraze.config import Settings
from summuraze.menu import MEDIA_SUFFIXES
from summuraze.store.db import Database
from summuraze.web.view import alive

Provider = Literal["api", "local"]
LOG_TAIL = 4000


class LaunchError(ValueError):
    """Запуск невозможен — сообщение для человека; `consent` — нужен ответ про этот хост."""

    def __init__(self, message: str, *, consent: str | None = None) -> None:
        super().__init__(message)
        self.consent = consent


class Choice(BaseModel):
    """Чем делать саммари: как в настройках (`api`) или своей моделью (`local`, `--local`)."""

    id: Provider
    title: str
    model: str
    host: str
    local: bool
    ready: bool
    problem: str | None = None
    consent_needed: bool = False
    usd: float | None = None
    basis: str


class MediaFile(BaseModel):
    name: str
    path: str
    size: int
    modified: float
    recording_id: int | None


def media_files(settings: Settings, db: Database, limit: int = 50) -> list[MediaFile]:
    folder = settings.record.directory
    files = [p for p in folder.glob("*") if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        MediaFile(
            name=p.name,
            path=str(p.resolve()),
            size=p.stat().st_size,
            modified=p.stat().st_mtime,
            recording_id=db.recording_by_path(p),
        )
        for p in files[:limit]
    ]


def allowed(path: Path, settings: Settings, db: Database) -> bool:
    """Из браузера запускаем только свои записи: папка записей или уже известный конвейеру
    файл. Произвольный путь с другого устройства в сети — нет."""
    if not path.is_file():
        return False
    folders = [settings.record.directory]
    if settings.watch.dir is not None:
        folders.append(settings.watch.dir.expanduser())
    inside = path.resolve().parent in [p.resolve() for p in folders]
    return inside or db.recording_by_path(path) is not None


def settings_for(settings: Settings, provider: Provider) -> Settings:
    return settings.localized() if provider == "local" else settings


def choices(settings: Settings, db: Database, duration: float | None) -> list[Choice]:
    result = []
    providers: tuple[Provider, ...] = ("api", "local")
    for provider in providers:
        try:
            s = settings_for(settings, provider)
        except ValueError as e:
            result.append(
                Choice(
                    id=provider,
                    title="своя модель (--local)",
                    model="—",
                    host="локально",
                    local=True,
                    ready=False,
                    problem=str(e),
                    basis="",
                )
            )
            continue
        llm = s.llm
        problem = _preflight_problem(s)
        usd, basis = estimate(db, s, duration)
        result.append(
            Choice(
                id=provider,
                title="своя модель (--local)" if provider == "local" else "как в настройках",
                model=llm.model,
                host="локально" if llm.is_local else llm.host,
                local=llm.is_local,
                ready=problem is None,
                problem=problem,
                consent_needed=not llm.is_local and not db.has_consent(llm.host),
                usd=usd,
                basis=basis,
            )
        )
    return result


def estimate(db: Database, settings: Settings, duration: float | None) -> tuple[float | None, str]:
    """Ожидаемая цена прогона: медиана того, во что секунда записи обошлась этой модели раньше.
    Конспект из кэша стоит $0 — об этом говорит подпись, а не ноль в оценке."""
    if settings.llm.is_local:
        return 0.0, "своя модель — бесплатно"
    rates = db.cost_per_second(settings.llm.model)
    if duration is None or not rates:
        return None, "цены этой модели ещё не было — она будет в списке после прогона"
    usd = statistics.median(rates) * duration
    runs = f"{len(rates)} прошл{'ому прогону' if len(rates) == 1 else 'ым прогонам'}"
    return usd, f"по {runs} {settings.llm.model}; из кэша — $0"


def _preflight_problem(settings: Settings) -> str | None:
    from summuraze.pipeline.orchestrator import Pipeline
    from summuraze.summarize.budget import TranscriptTooLong

    try:
        with Pipeline(settings) as pipeline:
            pipeline.preflight(until="render")
    except (ValueError, TranscriptTooLong) as e:
        return str(e)
    return None


class Process(Protocol):
    pid: int

    def wait(self) -> int: ...


Spawn = Callable[..., Process]


@dataclass
class Launch:
    process: Process
    log: Path
    previous_run: int | None
    name: str
    code: int | None = None

    def state(self) -> dict[str, Any]:
        tail = ""
        if self.log.is_file():
            with self.log.open("rb") as f:
                f.seek(max(0, self.log.stat().st_size - LOG_TAIL))
                tail = f.read().decode("utf-8", errors="replace")
        return {
            "pid": self.process.pid,
            "running": self.code is None,
            "code": self.code,
            "previous_run": self.previous_run,
            "log": tail,
        }


class Launcher:
    """Процессы прогонов, запущенных этим сервером: их лог и код выхода для страницы записи."""

    def __init__(self, log_dir: Path, spawn: Spawn = subprocess.Popen) -> None:
        self.log_dir = log_dir
        self._spawn = spawn
        self._launches: dict[int, Launch] = {}
        self._lock = threading.Lock()

    def start(
        self,
        settings: Settings,
        db: Database,
        path: Path,
        *,
        profile: str,
        provider: Provider,
        consent: bool,
    ) -> int:
        """Запустить `summuraze run`; вернуть id записи, чтобы страница открыла её."""
        from summuraze.summarize.profiles import get_profile

        get_profile(profile)
        if not allowed(path, settings, db):
            raise LaunchError(f"запускать можно только записи из {settings.record.directory}")
        chosen = settings_for(settings, provider)
        problem = _preflight_problem(chosen)
        if problem is not None:
            raise LaunchError(problem)
        with self._lock:
            # Свой только что запущенный процесс мог ещё не дойти до записи прогона в БД.
            mine = next((lc for lc in self._launches.values() if lc.code is None), None)
            busy = running(db) or (mine.name if mine is not None else None)
            if busy is not None:
                raise LaunchError(f"уже идёт прогон «{busy}» — дождись его или останови")
            llm = chosen.llm
            if not llm.is_local and not db.has_consent(llm.host):
                if not consent:
                    raise LaunchError(
                        f"саммари делает {llm.host} ({llm.model}): туда уйдёт текст транскрипта "
                        "целиком. Аудио наружу не уходит никогда.",
                        consent=llm.host,
                    )
                db.grant_consent(llm.host)
            recording_id, _ = db.register_source(path)
            latest = db.latest_run(recording_id)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log = self.log_dir / f"{datetime.now():%Y%m%d-%H%M%S-%f}-{recording_id}.log"
            argv = [sys.executable, "-m", "summuraze.cli", "run", str(path), "--profile", profile]
            if provider == "local":
                argv.append("--local")
            with log.open("wb") as out:
                process = self._spawn(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    cwd=Path.cwd(),  # .env читается из текущего каталога — тот же, что у веба
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    # Свой сеанс: Ctrl+C по серверу не обрывает час транскрибации.
                    start_new_session=True,
                )
            launch = Launch(process, log, latest.run_id if latest else None, path.name)
            self._launches[recording_id] = launch
            threading.Thread(target=_reap, args=(launch,), daemon=True).start()
        return recording_id

    def state(self, recording_id: int) -> dict[str, Any] | None:
        launch = self._launches.get(recording_id)
        return launch.state() if launch is not None else None


def _reap(launch: Launch) -> None:
    # Без wait() завершившийся процесс остался бы зомби, и `alive(pid)` считал бы его живым.
    launch.code = launch.process.wait()


def running(db: Database) -> str | None:
    """Имя записи, чей прогон сейчас идёт, — кем бы он ни был запущен."""
    for row in db.latest_runs():
        if row.status == "running" and alive(row.pid):
            return Path(row.path).name
    return None


def cancel(db: Database, recording_id: int) -> bool:
    """Остановить идущий прогон записи так же, как Ctrl+C в терминале: оркестратор запишет
    «ошибка: KeyboardInterrupt», llama-server погасит `finally`."""
    row = db.latest_run(recording_id)
    if row is None or row.status != "running" or row.pid is None or not alive(row.pid):
        return False
    # pid мог достаться другому процессу, если прогон умер без finish_run: сигнал — только
    # процессу summuraze.
    try:
        cmdline = Path(f"/proc/{row.pid}/cmdline").read_bytes()
    except OSError:
        return False
    if b"summuraze" not in cmdline:
        return False
    os.kill(row.pid, signal.SIGINT)
    return True


def upload_path(folder: Path, name: str) -> Path:
    """Путь для загружаемого файла в папке записей: без каталогов в имени и без перезаписи."""
    clean = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", Path(name).name).strip(" .")
    target = folder / (clean or "запись")
    if target.suffix.lower() not in MEDIA_SUFFIXES:
        raise LaunchError(f"это не запись: {name} (нужен аудио- или видеофайл)")
    folder.mkdir(parents=True, exist_ok=True)
    n = 2
    while target.exists():
        target = folder / f"{Path(clean).stem} ({n}){target.suffix}"
        n += 1
    return target
