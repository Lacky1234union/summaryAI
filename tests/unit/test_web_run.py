"""Запуск прогона из браузера (#23): тот же `summuraze run` отдельным процессом, после preflight
и согласия на отправку — того, что в терминале спрашивается до транскрибации."""

from __future__ import annotations

import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from summuraze.config import get_settings
from summuraze.store.db import Database
from summuraze.web import runs
from summuraze.web.app import create_app
from tests.conftest import SPEECH_CLIPS


class FakeProcess:
    """Процесс, который «идёт», пока тест его не отпустит."""

    def __init__(self) -> None:
        self.pid = os.getpid()
        self.done = threading.Event()

    def wait(self) -> int:
        self.done.wait(10)
        return 0


class FakeSpawn:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.processes: list[FakeProcess] = []

    def __call__(self, argv: list[str], **kw: Any) -> FakeProcess:
        self.calls.append((argv, kw))
        self.processes.append(FakeProcess())
        return self.processes[-1]

    def finish(self) -> None:
        for p in self.processes:
            p.done.set()


@pytest.fixture
def record_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "Recordings"
    folder.mkdir()
    monkeypatch.setenv("SUMMURAZE_RECORD_DIR", str(folder))
    get_settings.cache_clear()
    return folder


@pytest.fixture
def lecture(record_dir: Path) -> Path:
    path = record_dir / "пара 1.opus"
    shutil.copy(SPEECH_CLIPS["speech"], path)
    return path


@pytest.fixture
def spawn() -> Any:
    fake = FakeSpawn()
    yield fake
    fake.finish()


@pytest.fixture
def client(record_dir: Path, spawn: FakeSpawn) -> TestClient:
    return TestClient(create_app(get_settings(), spawn=spawn))


def _remote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUMMURAZE_LLM_PROVIDER", "openai-compat")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    get_settings.cache_clear()


def _post(client: TestClient, path: Path, **body: Any) -> Any:
    return client.post("/api/runs", json={"path": str(path), **body})


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="нужен ffmpeg")
def test_files_and_options(lecture: Path, client: TestClient, tmp_path: Path) -> None:
    (lecture.parent / "заметки.txt").write_text("не запись")
    files = client.get("/api/files").json()
    assert [f["name"] for f in files["items"]] == ["пара 1.opus"]
    assert files["items"][0]["recording_id"] is None

    o = client.get("/api/options", params={"path": str(lecture)}).json()
    assert o["duration"] == pytest.approx(5.45, abs=0.2), "до ингеста — по контейнеру"
    assert [p["id"] for p in o["profiles"]] == ["lecture", "raw"]
    api, local = o["providers"]
    assert api["ready"] and api["usd"] == 0 and api["local"], "echo — бесплатно"
    assert not local["ready"] and "SUMMURAZE_LOCAL_LLM_MODEL" in local["problem"]

    elsewhere = tmp_path / "чужой.opus"
    shutil.copy(lecture, elsewhere)
    assert client.get("/api/options", params={"path": str(elsewhere)}).status_code == 404


def test_estimate_from_past_runs(lecture: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _remote(monkeypatch)
    settings = get_settings()
    with Database(settings.paths.db_path) as db:
        usd, basis = runs.estimate(db, settings, 600)
        assert usd is None and "ещё не было" in basis
        rid, _ = db.register_source(lecture)
        db.set_duration(rid, 100)
        for cost in (0.01, 0.03, 0.02, 0.0):  # $0 — прогон из кэша, не в счёт
            run = db.start_run(rid)
            db.finish_run(run, "done", llm_model=settings.llm.model, cost_usd=cost)
        usd, basis = runs.estimate(db, settings, 600)
    assert usd == pytest.approx(0.12), "медиана $0.0002 за секунду × 10 минут"
    assert basis.startswith("по 3 прошлым прогонам")


def test_start_spawns_cli_run(lecture: Path, client: TestClient, spawn: FakeSpawn) -> None:
    r = _post(client, lecture, profile="raw")
    assert r.status_code == 202
    rid = r.json()["recording_id"]
    ((argv, kw),) = spawn.calls
    assert argv[1:] == ["-m", "summuraze.cli", "run", str(lecture), "--profile", "raw"]
    assert kw["stdin"] == subprocess.DEVNULL and kw["start_new_session"] is True
    assert kw["cwd"] == Path.cwd(), ".env читается из того же каталога"

    launch = client.get(f"/api/recordings/{rid}").json()["launch"]
    assert launch["running"] and launch["previous_run"] is None and launch["log"] == ""
    again = _post(client, lecture)
    assert again.status_code == 400 and "уже идёт" in again.json()["detail"], "двойной клик"
    assert len(spawn.calls) == 1


def test_remote_needs_consent_once(
    lecture: Path, spawn: FakeSpawn, monkeypatch: pytest.MonkeyPatch
) -> None:
    _remote(monkeypatch)
    client = TestClient(create_app(get_settings(), spawn=spawn))
    options = client.get("/api/options", params={"path": str(lecture)}).json()
    assert options["providers"][0]["consent_needed"] is True
    refused = _post(client, lecture)
    assert refused.status_code == 409 and refused.json()["consent"] == "openrouter.ai"
    assert spawn.calls == []
    assert _post(client, lecture, consent=True).status_code == 202
    with Database(get_settings().paths.db_path) as db:
        assert db.has_consent("openrouter.ai"), "процессу без терминала спрашивать не придётся"


@pytest.mark.parametrize(
    ("body", "where", "error"),
    [
        ({"profile": "встреча"}, "record", "неизвестный профиль"),
        ({}, "elsewhere", "только записи из"),
        ({"provider": "local"}, "record", "SUMMURAZE_LOCAL_LLM_MODEL"),
    ],
)
def test_refusals_before_spawn(
    lecture: Path,
    client: TestClient,
    spawn: FakeSpawn,
    tmp_path: Path,
    body: dict[str, str],
    where: str,
    error: str,
) -> None:
    path = lecture
    if where == "elsewhere":
        path = tmp_path / "чужой.opus"
        shutil.copy(lecture, path)
    r = _post(client, path, **body)
    assert r.status_code == 400 and error in r.json()["detail"]
    assert spawn.calls == []


def test_preflight_problem_refuses(
    lecture: Path, spawn: FakeSpawn, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUMMURAZE_OUTPUT_DIR", "/нет/такого/хранилища")
    get_settings.cache_clear()
    app = TestClient(create_app(get_settings(), spawn=spawn))
    r = app.post("/api/runs", json={"path": str(lecture)})
    assert r.status_code == 400 and "папки для заметок нет" in r.json()["detail"]
    assert spawn.calls == []


def test_local_adds_flag(
    lecture: Path, spawn: FakeSpawn, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = tmp_path / "qwen.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setenv("SUMMURAZE_LOCAL_LLM_MODEL", str(model))
    get_settings.cache_clear()
    client = TestClient(create_app(get_settings(), spawn=spawn))
    local = client.get("/api/options", params={"path": str(lecture)}).json()["providers"][1]
    assert local["ready"] and local["model"] == "qwen" and local["usd"] == 0
    assert _post(client, lecture, provider="local").status_code == 202
    assert spawn.calls[0][0][-1] == "--local"


def test_upload_into_record_dir(record_dir: Path, client: TestClient) -> None:
    first = client.put("/api/upload", params={"name": "../../лекция.m4a"}, content=b"abc")
    assert first.status_code == 200
    assert (record_dir / "лекция.m4a").read_bytes() == b"abc", "каталоги из имени отброшены"
    second = client.put("/api/upload", params={"name": "лекция.m4a"}, content=b"de")
    assert second.json()["name"] == "лекция (2).m4a", "без перезаписи"
    bad = client.put("/api/upload", params={"name": "virus.sh"}, content=b"x")
    assert bad.status_code == 400 and not list(record_dir.glob("*.part"))


def _running_run(lecture: Path, pid: int) -> int:
    db_path = get_settings().paths.db_path
    with Database(db_path) as db:
        rid, _ = db.register_source(lecture)
        run = db.start_run(rid)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE runs SET pid = ? WHERE id = ?", (pid, run))
    return rid


def test_cancel_sends_ctrl_c_to_summuraze_only(lecture: Path, client: TestClient) -> None:
    ours = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", "summuraze"],
        stderr=subprocess.PIPE,
        text=True,
    )
    foreign = subprocess.Popen(["sleep", "30"])
    try:
        rid = _running_run(lecture, foreign.pid)
        assert client.post(f"/api/recordings/{rid}/cancel").status_code == 409, "чужой pid"
        _running_run(lecture, ours.pid)
        assert client.post(f"/api/recordings/{rid}/cancel").status_code == 200
        # Код выхода после KeyboardInterrupt — 1 или −SIGINT, смотря по сборке Python (в CI −2).
        assert ours.wait(10) in (1, -signal.SIGINT)
        assert ours.stderr is not None
        assert "KeyboardInterrupt" in ours.stderr.read(), "как Ctrl+C в терминале"
    finally:
        ours.kill()
        foreign.kill()
        foreign.wait()
    assert client.post("/api/recordings/999/cancel").status_code == 409


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")
def test_real_process_runs_the_pipeline(lecture: Path) -> None:
    """Настоящий `python -m summuraze.cli run` без терминала: заглушки echo, прогон до заметки."""
    client = TestClient(create_app(get_settings()))
    rid = _post(client, lecture).json()["recording_id"]
    deadline = time.monotonic() + 60
    while (launch := client.get(f"/api/recordings/{rid}").json()["launch"])["running"]:
        assert time.monotonic() < deadline, launch["log"]
        time.sleep(0.2)
    assert launch["code"] == 0, launch["log"]
    assert "Заметка:" in launch["log"]
    card = client.get(f"/api/recordings/{rid}").json()["card"]
    assert card["status"] == "done" and card["profile"] == "lecture"
    assert card["run_id"] != launch["previous_run"]
