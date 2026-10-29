"""Веб (#23): список и страница записи читают то, что записал конвейер, — без своей копии данных."""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from summuraze.cli import app as cli_app
from summuraze.config import Settings, get_settings
from summuraze.pipeline.orchestrator import Pipeline, RunResult, StageName
from summuraze.store.db import Database
from summuraze.summarize.echo import EchoLLM
from summuraze.summarize.profiles import Block, Claim, LectureDoc, Question, RawDoc, Section, Term
from summuraze.types import LLMResult, Segment, TokenUsage, Transcript
from summuraze.web.app import create_app
from tests.conftest import SPEECH_CLIPS

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")


def _transcript() -> Transcript:
    return Transcript(
        source=Path("/x/лекция.opus"),
        duration=100.0,
        language="ru",
        asr_model="fixed-whisper",
        segments=[Segment(start=i * 10.0, end=i * 10.0 + 9, text=f"фраза {i}.") for i in range(10)],
    )


def _lecture_doc() -> LectureDoc:
    return LectureDoc(
        title="Сборка мусора",
        tldr=[
            Claim(text="Память освобождает сборщик", at="00:00:10"),
            Claim(text="Это выдумка модели", at="00:00:20"),
        ],
        sections=[
            Section(
                title="Поколения",
                at="00:00:30",
                theses=[
                    Claim(text="Память освобождает сборщик!", at="00:00:10"),  # повтор tldr
                    Claim(text="Старое поколение — реже", at="без таймкода"),
                ],
            )
        ],
        glossary=[Term(term="GC", definition="сборщик мусора", at="00:00:40")],
        examples=[],
        qa=[Question(question="А паузы?", answer="Короткие", at="00:00:50")],
        reading=[],
        unclear=[],
    )


class Writer(EchoLLM):
    """Документ — на вопрос саммари; на вопрос проверки — «нет» выдумке, «да» остальному."""

    def __init__(self, doc: Any) -> None:
        super().__init__()
        self.doc = doc

    def complete(self, **kw: Any) -> LLMResult:
        if "items" not in kw["schema"]["properties"]:
            text = self.doc.model_dump_json()
        else:
            items = []
            for line in kw["question"].splitlines():
                if m := re.match(r"(\d+)\. \[(\S+)\] (.*)", line):
                    lie = "выдумка" in m[3]
                    items.append(
                        {
                            "n": int(m[1]),
                            "verdict": "нет" if lie else "да",
                            "at": "" if lie else m[2],
                            "note": "не говорил" if lie else "",
                        }
                    )
            text = json.dumps({"items": items}, ensure_ascii=False)
        return LLMResult(
            text=text, model="writer", usage=TokenUsage(input=10, output=10), cost_usd=0.001
        )


class FixedASR:
    name = "fixed"

    def transcribe(self, audio: Path, **kw: object) -> Transcript:
        return _transcript().model_copy(update={"source": audio})


@pytest.fixture
def lecture(tmp_path: Path) -> Path:
    path = tmp_path / "лекция.opus"
    shutil.copy(SPEECH_CLIPS["speech"], path)
    return path


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(get_settings()))


def _run(
    source: Path,
    doc: Any = None,
    *,
    until: StageName = "render",
    profile: str = "lecture",
    terms: Path | None = None,
) -> RunResult:
    with Pipeline(
        Settings(),
        asr_factory=lambda cfg: FixedASR(),  # type: ignore[arg-type,return-value]
        llm_factory=lambda cfg: Writer(doc or _lecture_doc()),
    ) as p:
        return p.run(source, until=until, profile=profile, terms=terms)


def test_list_shows_run_profile_destination_and_total_cost(
    lecture: Path, client: TestClient
) -> None:
    _run(lecture)
    _run(lecture)  # из кэша: $0, но запись уже стоила $0.002
    data = client.get("/api/recordings").json()
    assert data["now"]
    (card,) = data["items"]
    assert card["name"] == "лекция.opus" and card["status"] == "done"
    assert card["profile"] == "lecture" and card["where"] == "локально"
    assert card["llm_model"] == "writer"
    assert card["cost_usd"] == pytest.approx(0.002), "саммари и проверка первого прогона"
    assert card["duration"] > 0 and card["stage"] == "render"


def test_recording_page_has_document_marks_and_transcript(
    lecture: Path, client: TestClient
) -> None:
    result = _run(lecture)
    data = client.get(f"/api/recordings/{result.recording_id}").json()
    assert data["asr_model"] == "fixed-whisper" and data["audio"] is True
    assert [line["start"] for line in data["lines"]] == [i * 10.0 for i in range(10)]
    assert result.note is not None and data["notes"] == [str(result.note.resolve())]
    # Повтор тезиса проверяется, хоть в заметку и не идёт: проверка считает документ модели.
    assert data["verification"].startswith("4 из 6")

    doc = data["document"]
    assert doc["title"] == "Сборка мусора" and doc["llm_model"] == "writer"
    assert [p["kind"] for p in doc["parts"]] == ["summary", "section", "glossary", "qa"]
    summary, section, glossary, qa = doc["parts"]
    assert [(i["text"], i["at"]) for i in summary["items"]] == [
        ("Память освобождает сборщик", 10.0),
        ("Это выдумка модели", 20.0),
    ]
    assert summary["items"][1]["mark"] == {
        "status": "unsupported",
        "note": "не говорил",
        "evidence": None,
    }
    assert section["at"] == 30.0
    (thesis,) = section["items"]  # повтор из «Коротко» отсеян — как в заметке
    assert thesis["at"] is None and thesis["mark"]["status"] == "no_timecode"
    assert glossary["items"][0]["term"] == "GC" and glossary["items"][0]["text"] == "сборщик мусора"
    assert qa["items"][0]["answer"] == "Короткие" and qa["items"][0]["at"] == 50.0


def test_raw_profile_gives_blocks_over_transcript(lecture: Path, client: TestClient) -> None:
    doc = RawDoc(
        blocks=[Block(title="Второй", at="00:00:52"), Block(title="Первый", at="00:00:00")]
    )
    result = _run(lecture, doc, profile="raw")
    data = client.get(f"/api/recordings/{result.recording_id}").json()
    assert data["document"]["parts"] == []
    assert data["document"]["blocks"] == [
        {"title": "Первый", "at": 0.0},
        {"title": "Второй", "at": 50.0},
    ]


def test_transcribe_only_shows_transcript_without_document(
    lecture: Path, client: TestClient
) -> None:
    result = _run(lecture, until="asr")
    data = client.get(f"/api/recordings/{result.recording_id}").json()
    assert data["document"] is None and data["missing"] is None
    assert len(data["lines"]) == 10
    assert data["card"]["profile"] is None and data["card"]["status"] == "done"


def test_transcript_has_the_replacements_the_summary_saw(
    lecture: Path, client: TestClient, tmp_path: Path
) -> None:
    terms = tmp_path / "курс.txt"
    terms.write_text("фраза -> строка\n", encoding="utf-8")
    result = _run(lecture, terms=terms)
    terms.unlink()  # словарь курса нигде не записан, кроме манифеста саммари
    data = client.get(f"/api/recordings/{result.recording_id}").json()
    assert data["lines"][3]["text"] == "строка 3."


def test_audio_seeks_by_range(lecture: Path, client: TestClient) -> None:
    result = _run(lecture)
    r = client.get(f"/api/recordings/{result.recording_id}/audio", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    assert r.headers["content-type"] == "audio/ogg"


def test_without_audio_page_still_opens(lecture: Path, client: TestClient) -> None:
    result = _run(lecture)
    artifacts = get_settings().paths.artifacts_dir
    shutil.rmtree(artifacts / "note-audio")
    shutil.rmtree(artifacts / "ingest")
    lecture.unlink()
    assert client.get(f"/api/recordings/{result.recording_id}/audio").status_code == 404
    data = client.get(f"/api/recordings/{result.recording_id}").json()
    assert data["audio"] is False and data["document"] is not None


def test_transcript_removed_by_gc_is_explained(lecture: Path, client: TestClient) -> None:
    result = _run(lecture, until="asr")
    shutil.rmtree(get_settings().paths.artifacts_dir / "asr")
    data = client.get(f"/api/recordings/{result.recording_id}").json()
    assert data["lines"] == [] and "gc" in data["missing"]


def test_running_stage_and_killed_run(lecture: Path, client: TestClient) -> None:
    db_path = get_settings().paths.db_path
    with Database(db_path) as db:
        rid, _ = db.register_source(lecture)
        run = db.start_run(rid, profile="lecture")
        db.set_stage(run, "asr")
    (card,) = client.get("/api/recordings").json()["items"]
    assert card["status"] == "running" and card["stage_title"] == "распознавание"
    assert card["stage_started_at"] and card["profile"] is None, "конспекта ещё нет"

    dead = subprocess.Popen(["true"])
    dead.wait()
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE runs SET pid = ? WHERE id = ?", (dead.pid, run))
    (card,) = client.get("/api/recordings").json()["items"]
    assert card["status"] == "interrupted" and card["stage"] == "asr"


def test_newest_recording_first_and_never_run(
    lecture: Path, client: TestClient, tmp_path: Path
) -> None:
    other = tmp_path / "другая.opus"
    shutil.copy(SPEECH_CLIPS["speech2"], other)
    with Database(get_settings().paths.db_path) as db:
        idle, _ = db.register_source(other)
    _run(lecture)
    items = client.get("/api/recordings").json()["items"]
    assert [c["name"] for c in items] == ["лекция.opus", "другая.opus"]
    assert items[1]["status"] is None
    data = client.get(f"/api/recordings/{idle}").json()
    assert data["lines"] == [] and "не расшифровывалась" in data["missing"]


def test_unknown_recording_and_index(client: TestClient) -> None:
    assert client.get("/api/recordings/999").status_code == 404
    assert client.get("/api/recordings/999/audio").status_code == 404
    page = client.get("/")
    assert page.status_code == 200 and "app.js" in page.text
    assert client.get("/app.js").status_code == 200


def test_cli_web_serves_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append(kw))
    result = CliRunner().invoke(cli_app, ["web", "--no-open", "--port", "9000"])
    assert result.exit_code == 0, result.output
    assert calls[0]["host"] == "127.0.0.1" and calls[0]["port"] == 9000
    assert "http://127.0.0.1:9000/" in result.output and "пароля нет" not in result.stderr

    result = CliRunner().invoke(cli_app, ["web", "--no-open", "--host", "0.0.0.0"])
    assert "пароля нет" in result.stderr and calls[1]["host"] == "0.0.0.0"
