"""FastAPI поверх `view` (#23): JSON для страницы, звук для плеера, сама страница — статикой.

Без сборщика фронтенда: одна страница на чистом JS. Это локальный инструмент, а не продукт, и
`npm install` ради списка и плеера был бы весомее всего остального проекта.

Соединение с SQLite — на запрос: обработчики FastAPI идут в пуле потоков, а соединение sqlite3
к потоку привязано. WAL (`store/db.py`) даёт читать, пока конвейер пишет.

Звук отдаёт `FileResponse`: он понимает `Range`, без этого плеер не перематывает — а перемотка
по таймкоду и есть смысл страницы.

Запуск прогона — `runs.py`: тот же `summuraze run` отдельным процессом. Загрузка файла — телом
PUT, а не формой: multipart потребовал бы ещё зависимость ради одного поля.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from summuraze.config import Settings
from summuraze.store.artifacts import ArtifactStore
from summuraze.store.db import Database
from summuraze.web import runs, view

STATIC = Path(__file__).parent / "static"
_AUDIO_TYPES = {".ogg": "audio/ogg", ".wav": "audio/wav", ".mka": "audio/x-matroska"}


class RunRequest(BaseModel):
    path: str
    profile: str = "lecture"
    provider: runs.Provider = "api"
    consent: bool = False


def create_app(settings: Settings, *, spawn: runs.Spawn | None = None) -> FastAPI:
    app = FastAPI(title="summuraze", docs_url=None, redoc_url=None, openapi_url=None)
    store = ArtifactStore(settings.paths.artifacts_dir)
    log_dir = settings.paths.data_dir / "web-runs"
    launcher = runs.Launcher(log_dir) if spawn is None else runs.Launcher(log_dir, spawn)

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
        return {"now": view.utc_now(), "launch": launcher.state(recording_id), **found.model_dump()}

    @app.get("/api/recordings/{recording_id}/audio")
    def audio(recording_id: int) -> FileResponse:
        with db() as conn:
            path = view.audio_file(conn, store, recording_id)
        if path is None:
            raise HTTPException(404, "звука записи нет: удалён по сроку хранения или из кэша")
        return FileResponse(path, media_type=_AUDIO_TYPES.get(path.suffix.lower()))

    @app.get("/api/files")
    def files() -> dict[str, Any]:
        with db() as conn:
            items = runs.media_files(settings, conn)
        return {"folder": str(settings.record.directory), "items": [f.model_dump() for f in items]}

    @app.get("/api/options")
    def options(path: str) -> dict[str, Any]:
        """Чем можно сделать конспект этого файла и во что он обойдётся."""
        from summuraze.audio.ingest import IngestError, duration
        from summuraze.menu import PROFILE_TITLES
        from summuraze.summarize.profiles import PROFILES

        source = Path(path)
        with db() as conn:
            if not runs.allowed(source, settings, conn):
                raise HTTPException(404, "нет такой записи в папке записей")
            known = conn.recording_by_path(source)
            row = conn.latest_run(known) if known is not None else None
            length = row.duration if row is not None else None
            if length is None:
                try:
                    length = duration(source)
                except IngestError as e:
                    raise HTTPException(400, str(e)) from None
            choices = runs.choices(settings, conn, length)
        return {
            "path": str(source),
            "duration": length,
            "profiles": [{"id": n, "title": PROFILE_TITLES.get(n, n)} for n in PROFILES],
            "providers": [c.model_dump() for c in choices],
        }

    @app.post("/api/runs")
    def start(request: RunRequest) -> JSONResponse:
        with db() as conn:
            try:
                recording_id = launcher.start(
                    settings,
                    conn,
                    Path(request.path),
                    profile=request.profile,
                    provider=request.provider,
                    consent=request.consent,
                )
            except runs.LaunchError as e:
                status = 409 if e.consent else 400
                return JSONResponse({"detail": str(e), "consent": e.consent}, status_code=status)
            except ValueError as e:  # неизвестный профиль
                raise HTTPException(400, str(e)) from None
        return JSONResponse({"recording_id": recording_id}, status_code=202)

    @app.post("/api/recordings/{recording_id}/cancel")
    def stop(recording_id: int) -> dict[str, bool]:
        with db() as conn:
            if not runs.cancel(conn, recording_id):
                raise HTTPException(409, "у записи нет идущего прогона")
        return {"ok": True}

    @app.put("/api/upload")
    async def upload(name: str, request: Request) -> dict[str, str]:
        try:
            target = runs.upload_path(settings.record.directory, name)
        except runs.LaunchError as e:
            raise HTTPException(400, str(e)) from None
        partial = target.with_name(target.name + ".part")
        try:
            with partial.open("wb") as out:
                async for chunk in request.stream():
                    out.write(chunk)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
        return {"path": str(target.resolve()), "name": target.name}

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
