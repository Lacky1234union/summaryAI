"""Ручная версия переживает генерацию, удаление кэша и конфликт двух вкладок (#24)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from summuraze.config import get_settings
from summuraze.store import notes
from summuraze.store.db import Database
from summuraze.web.app import create_app
from tests.conftest import SPEECH_CLIPS
from tests.unit.test_web import _run

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")


@pytest.fixture
def recording(tmp_path: Path) -> tuple[int, Path]:
    source = tmp_path / "лекция.opus"
    shutil.copy(SPEECH_CLIPS["speech"], source)
    result = _run(source)
    assert result.recording_id is not None
    return result.recording_id, source


def test_edit_export_and_regenerate_preserves_manual(recording: tuple[int, Path]) -> None:
    rid, source = recording
    client = TestClient(create_app(get_settings()))
    url = f"/api/recordings/{rid}"
    original = client.get(url + "/document").json()
    assert original["revision"] == 0 and not original["edited"]
    assert "Сборка мусора" in original["machine"]
    text = "# Моя правка\n\n- Своими словами [[запись.ogg#t=10|00:10]]\n"
    saved = client.put(url + "/document", json={"markdown": text, "revision": 0}).json()
    assert saved["edited"] and saved["revision"] > 0
    assert saved["machine"] == original["machine"]
    download = client.get(url + "/markdown")
    assert download.text == text
    assert "attachment;" in download.headers["content-disposition"]
    assert client.get(url + "/markdown?machine=true").text == original["machine"]
    export = client.post(url + "/export")
    assert export.status_code == 200
    note = Path(export.json()["path"])
    assert note.read_text() == text
    result = _run(source)
    assert result.cached and result.cached["summarize"]
    assert note.read_text() == text, "перегенерация не затирает правки"
    assert client.get(url + "/document").json()["machine"] == original["machine"]
    source.unlink()
    shutil.rmtree(get_settings().paths.artifacts_dir)
    assert client.get(url + "/document").json()["markdown"] == text
    assert client.post(url + "/export").status_code == 200


def test_stale_tab_and_idempotent_save(recording: tuple[int, Path]) -> None:
    rid, _ = recording
    client = TestClient(create_app(get_settings()))
    url = f"/api/recordings/{rid}/document"
    first = client.put(url, json={"markdown": "первая", "revision": 0}).json()
    stale = client.put(url, json={"markdown": "потерянная", "revision": 0})
    assert stale.status_code == 409
    assert client.get(url).json()["markdown"] == "первая"
    repeated = client.put(url, json={"markdown": "первая", "revision": first["revision"]}).json()
    assert repeated["revision"] == first["revision"]
    second = client.put(url, json={"markdown": "вторая", "revision": first["revision"]}).json()
    assert second["revision"] > first["revision"]
    with Database(get_settings().paths.db_path) as db:
        assert db.note_version(rid, "lecture", "machine") is not None
        assert db.note_version(rid, "lecture", "manual")["markdown"] == "вторая"


def test_profiles_separate_and_missing_documents(recording: tuple[int, Path]) -> None:
    rid, source = recording
    client = TestClient(create_app(get_settings()))
    root = f"/api/recordings/{rid}"
    assert client.get(root + "/document?profile=raw").status_code == 404
    assert client.get(root + "/document?profile=unknown").status_code == 400
    assert client.get("/api/recordings/9999/document").status_code == 404
    assert client.put(root + "/document", json={"markdown": "x", "revision": -1}).status_code == 422
    from summuraze.summarize.profiles import Block, RawDoc

    _run(source, RawDoc(blocks=[Block(title="Блок", at="00:00:00")]), profile="raw")
    client.put(root + "/document", json={"markdown": "правка lecture", "revision": 0})
    raw = client.get(root + "/document?profile=raw").json()
    assert not raw["edited"] and "правка lecture" not in raw["markdown"]
    assert client.get(root + "/document?profile=lecture").json()["edited"]


def test_legacy_note_bootstraps_snapshot(recording: tuple[int, Path]) -> None:
    rid, _ = recording
    settings = get_settings()
    import sqlite3
    from contextlib import closing

    with closing(sqlite3.connect(settings.paths.db_path)) as conn, conn:
        conn.execute("DELETE FROM note_versions")
    with Database(settings.paths.db_path) as db:
        value = notes.document(db, settings, rid, "lecture")
        assert value["markdown"] == value["machine"]
        assert db.note_version(rid, "lecture", "machine") is not None


def test_pruned_audio_does_not_return_in_exports(recording: tuple[int, Path]) -> None:
    rid, _ = recording
    settings = get_settings()
    with Database(settings.paths.db_path) as db:
        data = notes.document(db, settings, rid, "lecture")
        assert "#t=" in data["markdown"]
        notes.save(db, settings, rid, "lecture", data["markdown"] + "\nМоя правка", 0)
        db.mark_pruned(rid)
        current = notes.document(db, settings, rid, "lecture")
        assert "#t=" not in current["markdown"] and "Аудио удалено" in current["markdown"]
        path = notes.export_to_vault(db, settings, rid, "lecture")
        assert "#t=" not in path.read_text()
