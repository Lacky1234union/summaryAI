"""Машинный Markdown и ручная версия хранятся отдельно от кэша модели (#24)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from summuraze.config import Settings
from summuraze.pipeline.lock import exclusive
from summuraze.render.markdown import detach_audio, rewrite_note
from summuraze.store.db import Database


def path_for(db: Database, settings: Settings, recording_id: int, profile: str) -> Path | None:
    suffix = "" if profile == "lecture" else f" ({profile})"
    paths = [
        path
        for kind, path in db.outputs(recording_id)
        if kind == "note"
        and path.parent == settings.paths.notes_dir.resolve()
        and (path.stem.endswith(suffix) if suffix else not path.stem.endswith(" (raw)"))
    ]
    return paths[-1] if paths else None


def document(db: Database, settings: Settings, recording_id: int, profile: str) -> dict[str, Any]:
    row = db.latest_run(recording_id)
    if row is None:
        raise FileNotFoundError("нет такой записи")
    machine = db.note_version(recording_id, profile, "machine")
    if machine is None:
        # Заметки до миграции #24 ещё не имели снимка машинной версии в SQLite.
        path = path_for(db, settings, recording_id, profile)
        if path is None or not path.is_file():
            raise FileNotFoundError("Markdown ещё не создан — дождись готового конспекта")
        db.save_note_version(recording_id, profile, "machine", path.read_text(encoding="utf-8"))
        machine = db.note_version(recording_id, profile, "machine")
    assert machine is not None
    manual = db.note_version(recording_id, profile, "manual")

    def current(text: object) -> str:
        value = str(text)
        if row.pruned_at:
            for kind, path in db.outputs(recording_id):
                if kind == "audio":
                    value = detach_audio(value, path.name, row.pruned_at[:10])
        return value

    return {
        "profile": profile,
        "markdown": current(manual["markdown"] if manual else machine["markdown"]),
        "machine": current(machine["markdown"]),
        "revision": manual["id"] if manual else 0,
        "edited": manual is not None,
        "updated_at": (manual or machine)["created_at"],
    }


def save(
    db: Database, settings: Settings, recording_id: int, profile: str, text: str, revision: int
) -> dict[str, Any]:
    with exclusive(settings.paths.data_dir / "notes.lock"):
        document(db, settings, recording_id, profile)
        db.save_note_version(recording_id, profile, "manual", text, expected=revision)
        return document(db, settings, recording_id, profile)


def export_to_vault(db: Database, settings: Settings, recording_id: int, profile: str) -> Path:
    with exclusive(settings.paths.data_dir / "notes.lock"):
        data = document(db, settings, recording_id, profile)
        path = path_for(db, settings, recording_id, profile)
        if path is None:
            raise FileNotFoundError(
                "нет пути заметки в текущей папке вывода — прогони запись снова"
            )
        rewrite_note(path, data["markdown"])
        return path
