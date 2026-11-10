"""FIFO, восстановление и недокопированные файлы — без моделей и сети (#24)."""

from __future__ import annotations

import shutil
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from typer.testing import CliRunner

from summuraze.cli import app as cli
from summuraze.config import Settings, get_settings
from summuraze.pipeline.lock import exclusive
from summuraze.store.db import Database
from summuraze.web.app import create_app
from summuraze.web.queue import Queue, Worker
from summuraze.web.runs import Launcher, LaunchError
from tests.conftest import SPEECH_CLIPS
from tests.unit.test_web_run import FakeSpawn


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Settings, list[Path]]:
    folder = tmp_path / "Recordings"
    folder.mkdir()
    monkeypatch.setenv("SUMMURAZE_RECORD_DIR", str(folder))
    get_settings.cache_clear()
    paths = []
    for name, clip in SPEECH_CLIPS.items():
        target = folder / f"{name}.opus"
        shutil.copy(clip, target)
        paths.append(target)
    return get_settings(), paths


@pytest.fixture
def fake() -> Any:
    spawn = FakeSpawn()
    yield spawn
    spawn.finish()


def wait_for(check: Any) -> None:
    deadline = time.monotonic() + 60
    while not check():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_fifo_and_duplicate_click(setup: tuple[Settings, list[Path]], fake: FakeSpawn) -> None:
    settings, paths = setup
    completed = []
    launcher = Launcher(settings.paths.data_dir / "logs", fake)
    worker = Worker(settings, launcher, on_complete=completed.append)
    first = worker.queue.submit(paths[0])
    second = worker.queue.submit(paths[1])
    assert worker.queue.submit(paths[0])["id"] == first["id"]
    worker.tick()
    worker.tick()
    assert len(fake.calls) == 1
    assert [j["status"] for j in worker.queue.jobs()] == ["pending", "running"]
    fake.processes[0].done.set()
    wait_for(lambda: not launcher.state(first["recording_id"])["running"])
    worker.tick()
    assert len(fake.calls) == 2 and fake.calls[1][0][-3] == str(paths[1])
    assert completed[0]["id"] == first["id"] and completed[0]["status"] == "done"
    fake.processes[1].done.set()
    wait_for(lambda: not launcher.state(second["recording_id"])["running"])
    worker.tick()
    assert all(j["status"] == "done" for j in worker.queue.jobs())
    assert len(completed) == 2
    assert worker.queue.submit(paths[0])["id"] != first["id"], "можно повторить завершённую задачу"


def test_pending_cancel_and_external_pipeline(
    setup: tuple[Settings, list[Path]], fake: FakeSpawn
) -> None:
    settings, paths = setup
    worker = Worker(
        settings, Launcher(settings.paths.data_dir / "logs", fake), on_complete=lambda j: None
    )
    with Database(settings.paths.db_path) as db:
        rid, _ = db.register_source(paths[0])
        run = db.start_run(rid)
    job = worker.queue.submit(paths[1])
    worker.tick()
    assert fake.calls == []
    assert worker.queue.cancel(job["id"])
    assert not worker.queue.cancel(job["id"])
    assert not worker.queue.cancel(9999)
    with Database(settings.paths.db_path) as db:
        db.finish_run(run, "done")
    worker.tick()
    assert fake.calls == []
    assert Queue(settings).jobs()[0]["status"] == "cancelled", "статус переживает рестарт"


def test_worker_lock_and_pipeline_lock(setup: tuple[Settings, list[Path]], fake: FakeSpawn) -> None:
    settings, paths = setup
    worker = Worker(
        settings, Launcher(settings.paths.data_dir / "logs", fake), on_complete=lambda j: None
    )
    worker.queue.submit(paths[0])
    with exclusive(settings.paths.data_dir / "queue.lock"):
        worker.tick()
        assert fake.calls == []
    worker.tick()
    assert len(fake.calls) == 1
    lock = settings.paths.data_dir / "pipeline.lock"
    with exclusive(lock), pytest.raises(BlockingIOError), exclusive(lock, blocking=False):
        pytest.fail("вторая тяжёлая стадия не должна получить блокировку")
    with exclusive(lock, blocking=False):
        pass


def test_restart_recovers_done_and_interrupted(
    setup: tuple[Settings, list[Path]], fake: FakeSpawn
) -> None:
    settings, paths = setup
    queue = Queue(settings)
    job = queue.submit(paths[0])
    with closing(queue.connect()) as conn, conn:
        conn.execute("UPDATE job_queue SET status='running',pid=999999999 WHERE id=?", (job["id"],))
    completed = []
    worker = Worker(
        settings, Launcher(settings.paths.data_dir / "logs", fake), on_complete=completed.append
    )
    worker.tick()
    assert completed[-1]["status"] == "failed"
    next_job = queue.submit(paths[1])
    with Database(settings.paths.db_path) as db:
        run = db.start_run(next_job["recording_id"])
        db.finish_run(run, "done")
    with closing(queue.connect()) as conn, conn:
        conn.execute(
            "UPDATE job_queue SET status='running',pid=999999999 WHERE id=?", (next_job["id"],)
        )
    worker.tick()
    assert completed[-1]["status"] == "done"


def test_watch_waits_for_stable_file_and_deduplicates_after_restart(
    setup: tuple[Settings, list[Path]],
    fake: FakeSpawn,
    tmp_path: Path,
) -> None:
    settings, paths = setup
    folder = tmp_path / "watch"
    folder.mkdir()
    settings = settings.model_copy(
        update={"watch": settings.watch.model_copy(update={"dir": folder, "stable_seconds": 10})}
    )
    worker = Worker(
        settings, Launcher(settings.paths.data_dir / "logs", fake), on_complete=lambda j: None
    )
    target = folder / "новая.opus"
    target.write_bytes(b"first")
    (folder / "copy.opus.part").write_bytes(b"partial")
    (folder / "empty.opus").touch()
    (folder / "link.opus").symlink_to(paths[0])
    worker.scan(clock=0)
    worker.scan(clock=9)
    assert worker.queue.jobs() == []
    target.write_bytes(b"first-second")
    worker.scan(clock=10)
    worker.scan(clock=19)
    assert worker.queue.jobs() == []
    worker.scan(clock=20)
    jobs = worker.queue.jobs()
    assert len(jobs) == 1 and jobs[0]["path"] == str(target)
    restarted = Worker(settings, worker.launcher, on_complete=lambda j: None)
    restarted.scan(clock=30)
    restarted.scan(clock=40)
    assert len(restarted.queue.jobs()) == 1
    target.unlink()
    restarted.scan(clock=50)
    assert target not in restarted._observed


def test_preflight_consent_and_missing_source(
    setup: tuple[Settings, list[Path]], fake: FakeSpawn
) -> None:
    settings, paths = setup
    remote = settings.model_copy(
        update={
            "llm": settings.llm.model_copy(
                update={
                    "provider": "openai-compat",
                    "api_key": SecretStr("test"),
                }
            )
        }
    )
    queue = Queue(remote)
    with pytest.raises(LaunchError, match="разрешить"):
        queue.submit(paths[0])
    assert queue.jobs() == []
    job = queue.submit(paths[0], consent=True)
    paths[0].unlink()
    worker = Worker(
        remote, Launcher(settings.paths.data_dir / "logs", fake), on_complete=lambda j: None
    )
    worker.tick()
    assert queue.jobs()[0]["id"] == job["id"] and queue.jobs()[0]["status"] == "failed"
    assert fake.calls == []


def test_queue_http_and_watch_cli_errors(
    setup: tuple[Settings, list[Path]], fake: FakeSpawn
) -> None:
    settings, paths = setup
    client = TestClient(create_app(settings, spawn=fake))
    first = client.post("/api/runs", json={"path": str(paths[0])}).json()
    second = client.post("/api/runs", json={"path": str(paths[1])}).json()
    assert len(fake.calls) == 1
    assert (
        client.get(f"/api/recordings/{second['recording_id']}").json()["queue_job"]["status"]
        == "pending"
    )
    assert client.post(f"/api/queue/{second['job_id']}/cancel").status_code == 200
    assert client.post(f"/api/queue/{first['job_id']}/cancel").status_code == 409
    assert len(client.get("/api/queue").json()["items"]) == 2
    result = CliRunner().invoke(cli, ["watch"])
    assert result.exit_code == 1 and "SUMMURAZE_WATCH_DIR" in result.output


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")
def test_real_watch_pipeline_without_manual_commands(
    setup: tuple[Settings, list[Path]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, paths = setup
    folder = tmp_path / "inbox"
    folder.mkdir()
    monkeypatch.setenv("SUMMURAZE_WATCH_DIR", str(folder))
    monkeypatch.setenv("SUMMURAZE_WATCH_STABLE_SECONDS", "0.05")
    get_settings.cache_clear()
    settings = get_settings()
    completed = []
    worker = Worker(
        settings, Launcher(settings.paths.data_dir / "logs"), on_complete=completed.append
    )
    worker.start()
    try:
        target = folder / "живая-очередь.opus"
        shutil.copy(paths[0], target)
        wait_for(lambda: bool(completed))
        assert completed[0]["status"] == "done"
        with Database(settings.paths.db_path) as db:
            outputs = db.outputs(completed[0]["recording_id"])
        note = next(path for kind, path in outputs if kind == "note")
        assert note.is_file() and "Транскрипт" in note.read_text()
    finally:
        worker.close()


def test_concurrent_submit_deduplicates(setup: tuple[Settings, list[Path]]) -> None:
    from concurrent.futures import ThreadPoolExecutor

    settings, paths = setup
    queue = Queue(settings)
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = list(pool.map(lambda _: queue.submit(paths[0]), range(8)))
    assert len({job["id"] for job in jobs}) == 1
    assert len(queue.jobs()) == 1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")
def test_real_child_survives_worker_restart(setup: tuple[Settings, list[Path]]) -> None:
    settings, paths = setup
    first = Worker(settings, Launcher(settings.paths.data_dir / "logs"), on_complete=lambda j: None)
    first.queue.submit(paths[0])
    first.tick()
    completed = []
    second = Worker(
        settings, Launcher(settings.paths.data_dir / "other-logs"), on_complete=completed.append
    )

    def done() -> bool:
        second.tick()
        return bool(completed)

    wait_for(done)
    assert completed[0]["status"] == "done"
    with Database(settings.paths.db_path) as db:
        assert len(db.recent_runs()) == 1, "после рестарта начатый файл не запускается повторно"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")
def test_watch_cli_processes_file_and_stops_cleanly(
    setup: tuple[Settings, list[Path]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import signal
    import subprocess
    import sys

    settings, paths = setup
    folder = tmp_path / "cli-inbox"
    folder.mkdir()
    monkeypatch.setenv("SUMMURAZE_WATCH_STABLE_SECONDS", "0.05")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/nonexistent")
    log_path = tmp_path / "watch.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "summuraze.cli", "watch", str(folder)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        try:
            shutil.copy(paths[0], folder / "из-watch.opus")
            wait_for(lambda: "Готово:" in log_path.read_text())
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=10) == 0
            assert "Очередь сохранена" in log_path.read_text()
            assert Queue(settings).jobs()[0]["status"] == "done"
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
