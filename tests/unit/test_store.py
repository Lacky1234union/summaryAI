"""Кэш артефактов и SQLite: повторный прогон не вызывает стадию, прерванный не оставляет мусора."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import pytest

from summuraze.store.artifacts import MANIFEST, ArtifactKey, ArtifactStore
from summuraze.store.db import Database
from summuraze.types import TokenUsage


def _key(**overrides: object) -> ArtifactKey:
    base: dict[str, object] = {
        "stage": "asr",
        "version": 1,
        "inputs": ("abc",),
        "params": {"model": "small"},
    }
    base.update(overrides)
    return ArtifactKey(**base)  # type: ignore[arg-type]


class TestKey:
    def test_same_inputs_same_key(self) -> None:
        assert _key().digest == _key().digest

    def test_param_order_does_not_matter(self) -> None:
        a = _key(params={"model": "small", "lang": "ru"})
        b = _key(params={"lang": "ru", "model": "small"})
        assert a.digest == b.digest

    @pytest.mark.parametrize(
        "change",
        [{"version": 2}, {"inputs": ("abd",)}, {"params": {"model": "medium"}}, {"stage": "vad"}],
    )
    def test_any_significant_change_changes_key(self, change: dict[str, object]) -> None:
        assert _key(**change).digest != _key().digest

    def test_non_json_params_rejected(self) -> None:
        """Path в параметрах сделал бы ключ зависимым от расположения файла — кэш бы не попадал."""
        with pytest.raises(TypeError):
            _key(params={"out": Path("/tmp")})

    def test_bad_stage_name_rejected(self) -> None:
        with pytest.raises(ValueError):
            _key(stage="../escape")


class TestStore:
    def test_second_run_does_not_call_stage(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        calls = 0

        def build(out: Path) -> None:
            nonlocal calls
            calls += 1
            (out / "transcript.json").write_text("{}")

        first, hit1 = store.get_or_build(_key(), build)
        second, hit2 = store.get_or_build(_key(), build)
        assert calls == 1
        assert (hit1, hit2) == (False, True)
        assert first == second
        assert (second / "transcript.json").read_text() == "{}"

    def test_new_stage_version_rebuilds(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        calls = 0

        def build(out: Path) -> None:
            nonlocal calls
            calls += 1

        store.get_or_build(_key(version=1), build)
        store.get_or_build(_key(version=2), build)
        assert calls == 2

    def test_failed_build_leaves_nothing(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)

        def build(out: Path) -> None:
            (out / "half.json").write_text("{")
            raise RuntimeError("ASR упал на сороковой минуте")

        with pytest.raises(RuntimeError):
            store.get_or_build(_key(), build)
        assert store.get(_key()) is None
        assert not [p for p in tmp_path.rglob("*") if p.is_file()]

    def test_directory_without_manifest_is_not_a_hit(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.path(_key()).mkdir(parents=True)
        assert store.get(_key()) is None

    def test_manifest_describes_key(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        path, _ = store.get_or_build(_key(), lambda out: None)
        manifest = (path / MANIFEST).read_text()
        assert '"stage": "asr"' in manifest
        assert '"model": "small"' in manifest


class TestGC:
    def _age(self, path: Path, days: float) -> None:
        old = time.time() - days * 86400
        os.utime(path, (old, old))

    def test_removes_only_unused(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)

        def write(out: Path) -> None:
            (out / "a.bin").write_bytes(b"x" * 1000)

        old, _ = store.get_or_build(_key(version=1), write)
        fresh, _ = store.get_or_build(_key(version=2), write)
        self._age(old / MANIFEST, 40)

        dry = store.gc(30, dry_run=True)
        assert dry.removed == 1 and dry.freed_bytes > 1000
        assert old.exists()

        result = store.gc(30)
        assert result.removed == 1
        assert not old.exists()
        assert fresh.exists()

    def test_cache_hit_protects_from_gc(self, tmp_path: Path) -> None:
        """gc удаляет давно не использованное, а не давно созданное."""
        store = ArtifactStore(tmp_path)
        path, _ = store.get_or_build(_key(), lambda out: None)
        self._age(path / MANIFEST, 40)
        store.get(_key())
        assert store.gc(30).removed == 0

    def test_removes_stale_tmp_but_not_fresh(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        bucket = tmp_path / "asr" / "ab"
        stale = bucket / "abcd.tmp-111"
        live = bucket / "abce.tmp-222"
        stale.mkdir(parents=True)
        live.mkdir()
        self._age(stale, 2)
        assert store.gc(30).removed == 1
        assert not stale.exists() and live.exists()

    def test_missing_root_is_fine(self, tmp_path: Path) -> None:
        assert ArtifactStore(tmp_path / "nope").gc(30).removed == 0


class TestDatabase:
    def test_source_hash_is_cached_by_fingerprint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = tmp_path / "lecture.mp4"
        src.write_bytes(b"lecture")
        with Database(tmp_path / "db.sqlite") as db:
            rid, digest = db.register_source(src)

            import hashlib

            def boom(*a: object, **k: object) -> None:
                raise AssertionError("неизменённый файл перехеширован")

            monkeypatch.setattr(hashlib, "file_digest", boom)
            assert db.register_source(src) == (rid, digest)

    def test_changed_file_gets_new_hash(self, tmp_path: Path) -> None:
        src = tmp_path / "lecture.mp4"
        src.write_bytes(b"v1")
        with Database(tmp_path / "db.sqlite") as db:
            _, h1 = db.register_source(src)
            src.write_bytes(b"v2 longer")
            _, h2 = db.register_source(src)
        assert h1 != h2

    def test_moved_file_is_same_recording(self, tmp_path: Path) -> None:
        a = tmp_path / "a.mp4"
        a.write_bytes(b"same")
        with Database(tmp_path / "db.sqlite") as db:
            r1, h1 = db.register_source(a)
            b = a.rename(tmp_path / "b.mp4")
            r2, h2 = db.register_source(b)
        assert (r1, h1) == (r2, h2)

    def test_run_history(self, tmp_path: Path) -> None:
        src = tmp_path / "lecture.mp4"
        src.write_bytes(b"x")
        with Database(tmp_path / "db.sqlite") as db:
            rid, _ = db.register_source(src)
            run = db.start_run(rid)
            db.record_stage(run, "ingest", "k1", cached=True, seconds=0.01)
            db.record_stage(run, "asr", "k2", cached=False, seconds=600)
            db.finish_run(
                run,
                "done",
                asr_model="large-v3-turbo",
                llm_model="deepseek-flash",
                usage=TokenUsage(input=40_000, output=4_000, cached_input=39_000),
                cost_usd=0.01,
            )
            [row] = db.recent_runs()
        assert row.status == "done"
        assert row.source.endswith("lecture.mp4")
        assert (row.stages_total, row.stages_cached) == (2, 1)
        assert row.cost_usd == pytest.approx(0.01)

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        path = tmp_path / "db.sqlite"
        Database(path).close()
        Database(path).close()
        conn = sqlite3.connect(path)
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        finally:
            conn.close()


def test_cli_gc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from summuraze.cli import app
    from summuraze.config import get_settings

    monkeypatch.setenv("SUMMURAZE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    store = ArtifactStore(get_settings().paths.artifacts_dir)
    path, _ = store.get_or_build(_key(), lambda out: (out / "a.wav").write_bytes(b"0" * 2**20))
    old = time.time() - 40 * 86400
    os.utime(path / MANIFEST, (old, old))

    runner = CliRunner()
    dry = runner.invoke(app, ["gc", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "будет удалено: 1" in dry.output
    assert path.exists()

    real = runner.invoke(app, ["gc"])
    assert "удалено: 1" in real.output
    assert not path.exists()
    get_settings.cache_clear()
