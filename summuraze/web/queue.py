"""Дисковая очередь и watch-папка; обработка — тем же CLI, строго по одному файлу (#24)."""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from summuraze.config import Settings
from summuraze.menu import MEDIA_SUFFIXES
from summuraze.pipeline.lock import exclusive
from summuraze.store.db import Database
from summuraze.summarize.profiles import get_profile
from summuraze.web import runs
from summuraze.web.view import alive


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


class Queue:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = settings.paths.db_path
        with Database(self.path):
            pass

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def jobs(self) -> list[dict[str, Any]]:
        conn = self.connect()
        try:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM job_queue WHERE status IN ('pending','running')"
                    " OR id IN (SELECT id FROM job_queue ORDER BY id DESC LIMIT 100)"
                    " ORDER BY id DESC"
                )
            ]
        finally:
            conn.close()

    def submit(
        self,
        path: Path,
        *,
        profile: str = "lecture",
        provider: runs.Provider = "api",
        consent: bool = False,
    ) -> dict[str, Any]:
        get_profile(profile)
        with Database(self.path) as db:
            if not runs.allowed(path, self.settings, db):
                raise runs.LaunchError(
                    "запускать можно только записи из папки записей или watch-папки"
                )
            chosen = runs.settings_for(self.settings, provider)
            problem = runs._preflight_problem(chosen)
            if problem:
                raise runs.LaunchError(problem)
            if not chosen.llm.is_local and not db.has_consent(chosen.llm.host):
                if not consent:
                    raise runs.LaunchError(
                        f"нужно разрешить отправку текста на {chosen.llm.host}",
                        consent=chosen.llm.host,
                    )
                db.grant_consent(chosen.llm.host)
            rid, _ = db.register_source(path)
        conn = self.connect()
        try:
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                args = (str(path.resolve()), profile, provider)
                existing = conn.execute(
                    "SELECT * FROM job_queue WHERE path=? AND profile=? AND provider=?"
                    " AND status IN ('pending','running')",
                    args,
                ).fetchone()
                if existing:
                    return dict(existing)
                cur = conn.execute(
                    "INSERT INTO job_queue(recording_id,path,profile,provider,status,created_at)"
                    " VALUES (?,?,?,?,'pending',?)",
                    (rid, *args, now()),
                )
                return dict(
                    conn.execute("SELECT * FROM job_queue WHERE id=?", (cur.lastrowid,)).fetchone()
                )
        finally:
            conn.close()

    def cancel(self, job_id: int) -> bool:
        conn = self.connect()
        try:
            with conn:
                cur = conn.execute(
                    "UPDATE job_queue SET status='cancelled',finished_at=?"
                    " WHERE id=? AND status='pending'",
                    (now(), job_id),
                )
                return cur.rowcount > 0
        finally:
            conn.close()


def notify(job: dict[str, Any]) -> None:
    title = "Конспект готов" if job["status"] == "done" else "Не удалось сделать конспект"
    message = Path(job["path"]).name
    logger.info("{}: {}", title, message)
    if binary := shutil.which("notify-send"):
        # Отсутствие сервиса уведомлений не должно останавливать очередь.
        with suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(
                [binary, "--app-name=Summuraze", title, message],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )


class Worker:
    def __init__(
        self,
        settings: Settings,
        launcher: runs.Launcher,
        *,
        on_complete: Callable[[dict[str, Any]], None] = notify,
    ) -> None:
        self.settings = settings
        self.queue = Queue(settings)
        self.launcher = launcher
        self.on_complete = on_complete
        self.watch_error: str | None = None
        self._observed: dict[Path, tuple[int, int, float]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="summuraze-queue", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as e:
                logger.warning("очередь: {}", e)
            self._stop.wait(1)

    def tick(self) -> None:
        # Межпроцессная блокировка: web и watch могут быть запущены одновременно.
        try:
            with exclusive(self.settings.paths.data_dir / "queue.lock", blocking=False):
                self.scan()
                if not self._stop.is_set():
                    self._advance()
        except BlockingIOError:
            return

    def _finish(
        self, conn: sqlite3.Connection, job: dict[str, Any], status: str, error: str | None
    ) -> None:
        with conn:
            conn.execute(
                "UPDATE job_queue SET status=?,finished_at=?,error=? WHERE id=?",
                (status, now(), error, job["id"]),
            )
        self.on_complete({**job, "status": status, "error": error})

    def _advance(self) -> None:
        conn = self.queue.connect()
        try:
            active = conn.execute(
                "SELECT * FROM job_queue WHERE status='running' ORDER BY id LIMIT 1"
            ).fetchone()
            with Database(self.queue.path) as db:
                if active is not None:
                    job = dict(active)
                    launch = self.launcher.state(job["recording_id"])
                    if launch is not None and launch["pid"] != job["pid"]:
                        launch = None
                    # После рестарта CLI продолжает жить; его итог берём из общей БД.
                    if launch is not None and launch["running"]:
                        return
                    if launch is None and job["pid"] and alive(job["pid"]):
                        return
                    row = db.latest_run(job["recording_id"])
                    done = (
                        row is not None
                        and row.run_id != job["previous_run"]
                        and row.status == "done"
                    )
                    if launch is not None:
                        done = launch["code"] == 0
                    error = None if done else (row.error if row else None) or "процесс прерван"
                    self._finish(conn, job, "done" if done else "failed", error)
                if runs.running(db):
                    return
                pending = conn.execute(
                    "SELECT * FROM job_queue WHERE status='pending' ORDER BY id LIMIT 1"
                ).fetchone()
                if pending is None:
                    return
                job = dict(pending)
                row = db.latest_run(job["recording_id"])
                with conn:
                    claimed = conn.execute(
                        "UPDATE job_queue SET status='running',started_at=?,previous_run=?"
                        " WHERE id=? AND status='pending'",
                        (now(), row.run_id if row else None, job["id"]),
                    )
                if not claimed.rowcount:
                    return  # отменили между чтением и захватом задачи
                try:
                    self.launcher.start(
                        self.settings,
                        db,
                        Path(job["path"]),
                        profile=job["profile"],
                        provider=job["provider"],
                        consent=False,
                    )
                except (ValueError, OSError) as e:
                    self._finish(conn, job, "failed", str(e))
                    return
                launch = self.launcher.state(job["recording_id"])
                assert launch is not None
                with conn:
                    conn.execute(
                        "UPDATE job_queue SET pid=? WHERE id=?", (launch["pid"], job["id"])
                    )
        finally:
            conn.close()

    def scan(self, *, clock: float | None = None) -> None:
        folder = self.settings.watch.dir
        if folder is None:
            return
        folder = folder.expanduser().resolve()
        if not folder.is_dir():
            self.watch_error = f"watch-папки нет: {folder}"
            return
        self.watch_error = None
        at = time.monotonic() if clock is None else clock
        present: set[Path] = set()
        conn = self.queue.connect()
        try:
            for path in sorted(folder.iterdir()):
                if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
                    continue
                # Ссылка из watch-папки не должна незаметно отправить чужой файл.
                if path.resolve().parent != folder:
                    continue
                present.add(path)
                stat = path.stat()
                fingerprint = (stat.st_size, stat.st_mtime_ns)
                observed = self._observed.get(path)
                if observed is None or observed[:2] != fingerprint:
                    self._observed[path] = (*fingerprint, at)
                    continue
                if not stat.st_size or at - observed[2] < self.settings.watch.stable_seconds:
                    continue
                seen = conn.execute(
                    "SELECT size,mtime_ns FROM watch_seen WHERE path=?", (str(path),)
                ).fetchone()
                if seen is not None and tuple(seen) == fingerprint:
                    continue
                try:
                    self.queue.submit(
                        path,
                        profile=self.settings.watch.profile,
                        provider=self.settings.watch.provider,
                    )
                except (ValueError, OSError) as e:
                    self.watch_error = str(e)
                    continue
                with conn:
                    conn.execute(
                        "INSERT INTO watch_seen VALUES (?,?,?) ON CONFLICT(path) DO UPDATE"
                        " SET size=excluded.size,mtime_ns=excluded.mtime_ns",
                        (str(path), *fingerprint),
                    )
            self._observed = {p: value for p, value in self._observed.items() if p in present}
        finally:
            conn.close()
