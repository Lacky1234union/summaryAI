"""FastAPI поверх `view` (#23): JSON для страницы, звук для плеера, сама страница — статикой.

Без сборщика фронтенда: одна страница на чистом JS. Это локальный инструмент, а не продукт, и
`npm install` ради списка и плеера был бы весомее всего остального проекта.

Соединение с SQLite — на запрос: обработчики FastAPI идут в пуле потоков, а соединение sqlite3
к потоку привязано. WAL (`store/db.py`) даёт читать, пока конвейер пишет.

Звук отдаёт `FileResponse`: он понимает `Range`, без этого плеер не перематывает — а перемотка
по таймкоду и есть смысл страницы.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from summuraze.config import Settings
from summuraze.store.artifacts import ArtifactStore
from summuraze.store.db import Database
from summuraze.web import view

STATIC = Path(__file__).parent / "static"
_AUDIO_TYPES = {".ogg": "audio/ogg", ".wav": "audio/wav", ".mka": "audio/x-matroska"}


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="summuraze", docs_url=None, redoc_url=None, openapi_url=None)
    store = ArtifactStore(settings.paths.artifacts_dir)

    def db() -> Database:
        return Database(settings.paths.db_path)

    @app.get("/api/recordings")
    def recordings() -> dict[str, Any]:
        with db() as conn:
            items = view.cards(conn)
        # Своё «сейчас» — чтобы секундомер стадии не зависел от часов устройства, на котором
        # открыта страница.
        return {"now": view.utc_now(), "items": [c.model_dump() for c in items]}

    @app.get("/api/recordings/{recording_id}")
    def recording(recording_id: int) -> dict[str, Any]:
        with db() as conn:
            found = view.detail(conn, store, settings, recording_id)
        if found is None:
            raise HTTPException(404, "нет такой записи")
        return {"now": view.utc_now(), **found.model_dump()}

    @app.get("/api/recordings/{recording_id}/audio")
    def audio(recording_id: int) -> FileResponse:
        with db() as conn:
            path = view.audio_file(conn, store, recording_id)
        if path is None:
            raise HTTPException(404, "звука записи нет: удалён по сроку хранения или из кэша")
        return FileResponse(path, media_type=_AUDIO_TYPES.get(path.suffix.lower()))

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
