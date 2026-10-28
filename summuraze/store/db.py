"""SQLite: какие записи обработаны, чем, сколько это стоило и что взялось из кэша.

Один пользователь, одна машина — Postgres здесь был бы контейнером, который надо поднимать перед
тем, как расшифровать лекцию (см. ARCHITECTURE.md). WAL — чтобы будущий веб-интерфейс (#23) мог
читать историю, пока конвейер пишет.

Помимо истории, БД хранит sha256 исходников по отпечатку (путь, размер, mtime): хеш —
корень цепочки ключей кэша, а перечитывать двухгигабайтное видео ради него при каждом запуске
незачем.

Схема мигрирует через `PRAGMA user_version`: новая миграция — новый элемент в `_MIGRATIONS`,
старые не правятся никогда.

Refs #6
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from summuraze.types import TokenUsage

RunStatus = Literal["running", "done", "failed"]

_MIGRATIONS: list[str] = [
    """
    CREATE TABLE recordings (
        id         INTEGER PRIMARY KEY,
        sha256     TEXT NOT NULL UNIQUE,
        path       TEXT NOT NULL,          -- где файл видели последний раз
        size       INTEGER NOT NULL,
        mtime_ns   INTEGER NOT NULL,
        duration   REAL,
        first_seen TEXT NOT NULL
    );
    CREATE INDEX recordings_fingerprint ON recordings (path, size, mtime_ns);

    CREATE TABLE runs (
        id            INTEGER PRIMARY KEY,
        recording_id  INTEGER NOT NULL REFERENCES recordings (id),
        started_at    TEXT NOT NULL,
        finished_at   TEXT,
        status        TEXT NOT NULL CHECK (status IN ('running', 'done', 'failed')),
        error         TEXT,
        asr_model     TEXT,
        llm_provider  TEXT,               -- base_url: что куда уехало, видно потом
        llm_model     TEXT,
        input_tokens  INTEGER,
        output_tokens INTEGER,
        cached_tokens INTEGER,
        cost_usd      REAL
    );

    CREATE TABLE stage_runs (
        id       INTEGER PRIMARY KEY,
        run_id   INTEGER NOT NULL REFERENCES runs (id),
        stage    TEXT NOT NULL,
        key      TEXT NOT NULL,
        cached   INTEGER NOT NULL,
        seconds  REAL NOT NULL,
        at       TEXT NOT NULL
    );
    CREATE INDEX stage_runs_run ON stage_runs (run_id);
    """,
    # #49: что рендер положил в хранилище — чтобы срок хранения нашёл аудио и заметки записи,
    # даже если папку заметок потом поменяли. pruned_at — аудио записи уже удалено.
    """
    ALTER TABLE recordings ADD COLUMN pruned_at TEXT;

    CREATE TABLE outputs (
        recording_id INTEGER NOT NULL REFERENCES recordings (id),
        kind         TEXT NOT NULL CHECK (kind IN ('note', 'audio', 'subtitles')),
        path         TEXT NOT NULL,
        PRIMARY KEY (recording_id, path)
    );
    """,
    # #20: согласие отправлять транскрипт на внешний хост — спрашивается один раз на хост.
    """
    CREATE TABLE consents (
        host       TEXT PRIMARY KEY,
        granted_at TEXT NOT NULL
    );
    """,
]

OutputKind = Literal["note", "audio", "subtitles"]


@dataclass(frozen=True)
class RunRow:
    id: int
    source: str
    started_at: str
    finished_at: str | None
    status: RunStatus
    error: str | None
    asr_model: str | None
    llm_provider: str | None
    llm_model: str | None
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    cost_usd: float | None
    duration: float | None  # длительность записи
    stages_total: int
    stages_cached: int

    @property
    def seconds(self) -> float | None:
        """Сколько шёл прогон."""
        if self.finished_at is None:
            return None
        started = datetime.fromisoformat(self.started_at)
        return (datetime.fromisoformat(self.finished_at) - started).total_seconds()


@dataclass(frozen=True)
class RecordingRow:
    id: int
    path: str
    first_seen: str
    duration: float | None


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _migrate(self) -> None:
        current: int = self._conn.execute("PRAGMA user_version").fetchone()[0]
        for version, script in enumerate(_MIGRATIONS[current:], start=current + 1):
            with self._conn:
                self._conn.executescript(script)
                self._conn.execute(f"PRAGMA user_version = {version}")

    # ── Записи ──────────────────────────────────────────────────────────────

    def register_source(self, path: Path) -> tuple[int, str]:
        """`(id записи, sha256)` исходника. Хеш пересчитывается, только если файл изменился."""
        resolved = str(path.resolve())
        st = path.stat()
        row = self._conn.execute(
            "SELECT id, sha256 FROM recordings WHERE path = ? AND size = ? AND mtime_ns = ?",
            (resolved, st.st_size, st.st_mtime_ns),
        ).fetchone()
        if row is not None:
            return row["id"], row["sha256"]

        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        with self._conn:
            # Тот же файл мог переехать или быть переименован: тогда обновляем, где он лежит.
            self._conn.execute(
                """
                INSERT INTO recordings (sha256, path, size, mtime_ns, first_seen)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (sha256) DO UPDATE
                    SET path = excluded.path, size = excluded.size, mtime_ns = excluded.mtime_ns
                """,
                (digest, resolved, st.st_size, st.st_mtime_ns, _now()),
            )
        rid = self._conn.execute(
            "SELECT id FROM recordings WHERE sha256 = ?", (digest,)
        ).fetchone()["id"]
        return rid, digest

    def set_duration(self, recording_id: int, duration: float) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE recordings SET duration = ? WHERE id = ?", (duration, recording_id)
            )

    def record_outputs(self, recording_id: int, outputs: list[tuple[OutputKind, Path]]) -> None:
        """Запомнить, что рендер положил в хранилище. Новый рендер — аудио снова есть."""
        with self._conn:
            self._conn.executemany(
                "INSERT OR IGNORE INTO outputs (recording_id, kind, path) VALUES (?, ?, ?)",
                [(recording_id, kind, str(path.resolve())) for kind, path in outputs],
            )
            self._conn.execute(
                "UPDATE recordings SET pruned_at = NULL WHERE id = ?", (recording_id,)
            )

    def outputs(self, recording_id: int) -> list[tuple[OutputKind, Path]]:
        rows = self._conn.execute(
            "SELECT kind, path FROM outputs WHERE recording_id = ? ORDER BY kind, path",
            (recording_id,),
        ).fetchall()
        return [(row["kind"], Path(row["path"])) for row in rows]

    def unpruned_recordings(self) -> list[RecordingRow]:
        rows = self._conn.execute(
            "SELECT id, path, first_seen, duration FROM recordings WHERE pruned_at IS NULL"
            " ORDER BY first_seen"
        ).fetchall()
        return [RecordingRow(**dict(row)) for row in rows]

    def stage_keys(self, recording_id: int, stages: tuple[str, ...]) -> set[tuple[str, str]]:
        """`(стадия, ключ)` всех артефактов этих стадий, когда-либо собранных для записи."""
        marks = ", ".join("?" * len(stages))
        rows = self._conn.execute(
            f"""
            SELECT DISTINCT s.stage, s.key FROM stage_runs s
            JOIN runs r ON r.id = s.run_id
            WHERE r.recording_id = ? AND s.stage IN ({marks})
            """,
            (recording_id, *stages),
        ).fetchall()
        return {(row["stage"], row["key"]) for row in rows}

    def mark_pruned(self, recording_id: int) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE recordings SET pruned_at = ? WHERE id = ?", (_now(), recording_id)
            )

    # ── Прогоны ─────────────────────────────────────────────────────────────

    def start_run(self, recording_id: int) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO runs (recording_id, started_at, status) VALUES (?, ?, 'running')",
                (recording_id, _now()),
            )
        assert cur.lastrowid is not None
        return cur.lastrowid

    def record_stage(
        self, run_id: int, stage: str, key: str, *, cached: bool, seconds: float
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO stage_runs (run_id, stage, key, cached, seconds, at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, stage, key, int(cached), seconds, _now()),
            )

    def finish_run(
        self,
        run_id: int,
        status: RunStatus,
        *,
        error: str | None = None,
        asr_model: str | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        usage: TokenUsage | None = None,
        cost_usd: float | None = None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE runs SET finished_at = ?, status = ?, error = ?, asr_model = ?,
                    llm_provider = ?, llm_model = ?, input_tokens = ?, output_tokens = ?,
                    cached_tokens = ?, cost_usd = ?
                WHERE id = ?
                """,
                (
                    _now(),
                    status,
                    error,
                    asr_model,
                    llm_provider,
                    llm_model,
                    usage.input if usage else None,
                    usage.output if usage else None,
                    usage.cached_input if usage else None,
                    cost_usd,
                    run_id,
                ),
            )

    def recent_runs(self, limit: int = 20) -> list[RunRow]:
        rows = self._conn.execute(
            """
            SELECT r.id, rec.path AS source, r.started_at, r.finished_at, r.status, r.error,
                   r.asr_model, r.llm_provider, r.llm_model, r.input_tokens, r.output_tokens,
                   r.cached_tokens, r.cost_usd, rec.duration,
                   COUNT(s.id) AS stages_total, COALESCE(SUM(s.cached), 0) AS stages_cached
            FROM runs r
            JOIN recordings rec ON rec.id = r.recording_id
            LEFT JOIN stage_runs s ON s.run_id = r.id
            GROUP BY r.id
            ORDER BY r.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [RunRow(**dict(row)) for row in rows]

    # ── Согласие на отправку наружу ─────────────────────────────────────────

    def has_consent(self, host: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM consents WHERE host = ?", (host,)).fetchone()
        return row is not None

    def grant_consent(self, host: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO consents (host, granted_at) VALUES (?, ?)", (host, _now())
            )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
