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

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from summuraze.config import Settings
from summuraze.pipeline.lock import exclusive
from summuraze.store import notes
from summuraze.store.artifacts import ArtifactStore
from summuraze.store.db import Database
from summuraze.web import runs, view
from summuraze.web.queue import Worker

STATIC = Path(__file__).parent / "static"
_AUDIO_TYPES = {".ogg": "audio/ogg", ".wav": "audio/wav", ".mka": "audio/x-matroska"}


class RunRequest(BaseModel):
    path: str
    profile: str = "lecture"
    provider: runs.Provider = "api"
    consent: bool = False


class EditRequest(BaseModel):
    markdown: str = Field(max_length=2_000_000)
    revision: int = Field(ge=0)
    profile: str = "lecture"


def create_app(settings: Settings, *, spawn: runs.Spawn | None = None) -> FastAPI:
    store = ArtifactStore(settings.paths.artifacts_dir)
    log_dir = settings.paths.data_dir / "web-runs"
    launcher = runs.Launcher(log_dir) if spawn is None else runs.Launcher(log_dir, spawn)
    worker = Worker(settings, launcher)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        worker.start()
        try:
            yield
        finally:
            worker.close()

    app = FastAPI(
        title="summuraze", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )

    def db() -> Database:
        return Database(settings.paths.db_path)

    def profile_for(conn: Database, recording_id: int, profile: str | None) -> str:
        from summuraze.summarize.profiles import get_profile

        row = conn.latest_run(recording_id)
        if row is None:
            raise HTTPException(404, "нет такой записи")
        chosen = profile or row.profile or "lecture"
        try:
            get_profile(chosen)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        return chosen

    def read_document(recording_id: int, profile: str | None) -> dict[str, Any]:
        with db() as conn:
            chosen = profile_for(conn, recording_id, profile)
            try:
                with exclusive(settings.paths.data_dir / "notes.lock"):
                    return notes.document(conn, settings, recording_id, chosen)
            except FileNotFoundError as e:
                raise HTTPException(404, str(e)) from None

    @app.get("/api/recordings/{recording_id}/document")
    def document(recording_id: int, profile: str | None = None) -> dict[str, Any]:
        return read_document(recording_id, profile)

    @app.put("/api/recordings/{recording_id}/document")
    def edit(recording_id: int, request: EditRequest) -> dict[str, Any]:
        with db() as conn:
            chosen = profile_for(conn, recording_id, request.profile)
            try:
                return notes.save(
                    conn, settings, recording_id, chosen, request.markdown, request.revision
                )
            except FileNotFoundError as e:
                raise HTTPException(404, str(e)) from None
            except ValueError as e:
                raise HTTPException(409, str(e)) from None

    @app.get("/api/recordings/{recording_id}/markdown")
    def markdown(
        recording_id: int, profile: str | None = None, machine: bool = False
    ) -> PlainTextResponse:
        data = read_document(recording_id, profile)
        text = data["machine"] if machine else data["markdown"]
        filename = quote(f"конспект-{recording_id}.md")
        return PlainTextResponse(
            text,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
        )

    @app.post("/api/recordings/{recording_id}/export")
    def export(recording_id: int, profile: str | None = None) -> dict[str, str]:
        with db() as conn:
            chosen = profile_for(conn, recording_id, profile)
            try:
                path = notes.export_to_vault(conn, settings, recording_id, chosen)
            except FileNotFoundError as e:
                raise HTTPException(404, str(e)) from None
        return {"path": str(path)}

    @app.get("/api/queue")
    def queue() -> dict[str, Any]:
        return {
            "items": worker.queue.jobs(),
            "watch_error": worker.watch_error,
            "watch_dir": str(settings.watch.dir) if settings.watch.dir else None,
        }

    @app.post("/api/queue/{job_id}/cancel")
    def cancel_queued(job_id: int) -> dict[str, bool]:
        if not worker.queue.cancel(job_id):
            raise HTTPException(409, "отменить можно только ожидающую задачу")
        return {"ok": True}

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
        queued = next(
            (
                job
                for job in worker.queue.jobs()
                if job["recording_id"] == recording_id and job["status"] in ("pending", "running")
            ),
            None,
        )
        return {
            "now": view.utc_now(),
            "launch": launcher.state(recording_id),
            "queue_job": queued,
            **found.model_dump(),
        }

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
        try:
            job = worker.queue.submit(
                Path(request.path),
                profile=request.profile,
                provider=request.provider,
                consent=request.consent,
            )
        except runs.LaunchError as e:
            status = 409 if e.consent else 400
            return JSONResponse({"detail": str(e), "consent": e.consent}, status_code=status)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        worker.tick()
        return JSONResponse(
            {"recording_id": job["recording_id"], "job_id": job["id"]}, status_code=202
        )

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
