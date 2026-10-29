"""Что показывает веб (#23): записи и их прогоны из SQLite, транскрипт и конспект из кэша стадий.

Веб — читатель того же хранилища, что пишет конвейер: своей копии данных у него нет. Прогон,
запущенный из терминала, виден в браузере с текущей стадией, потому что стадию пишет
оркестратор (`runs.stage`), а не тот, кто его запустил.

Транскрипт и конспект берутся из артефактов последнего успешного прогона, а не из заметки
Obsidian: разбирать Markdown обратно незачем, когда есть JSON, из которого он собран. Поверх
транскрипта — то же, что делает оркестратор: замены терминов, с которыми писался конспект (они в
манифесте артефакта саммари), и пометки детектора выдумок Whisper (#12).

Модуль без FastAPI: данные проверяются тестами напрямую, HTTP — тонкая обёртка в `app.py`.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from summuraze.asr import repetition
from summuraze.asr.terms import Glossary, apply_replacements, load_glossary
from summuraze.config import Settings, llm_destination
from summuraze.pipeline import stages
from summuraze.pipeline.orchestrator import STAGE_TITLES
from summuraze.render.markdown import resolve_at, thought_key
from summuraze.store.artifacts import MANIFEST, ArtifactStore
from summuraze.store.db import Database, LatestRun
from summuraze.summarize.profiles import Claim, LectureDoc, RawDoc, get_profile
from summuraze.summarize.verify import Mark, Report
from summuraze.types import Transcript

Status = Literal["running", "done", "failed", "interrupted"]
# Стадия в stage_runs → каталог её артефакта в кэше.
_ARTIFACT_STAGE = {"audio": "note-audio"}


class Card(BaseModel):
    """Строка списка записей."""

    id: int
    name: str
    path: str
    first_seen: str
    duration: float | None
    pruned: bool
    status: Status | None  # None — запись зарегистрирована, прогонов не было
    stage: str | None
    stage_title: str | None
    stage_started_at: str | None
    started_at: str | None
    finished_at: str | None
    error: str | None
    profile: str | None
    where: str | None
    llm_model: str | None
    cost_usd: float | None


class Line(BaseModel):
    start: float
    end: float
    text: str
    suspect: bool = False
    reason: str | None = None


class Item(BaseModel):
    text: str
    at: float | None
    mark: Mark | None = None
    term: str | None = None  # глоссарий: термин отдельно от определения
    answer: str | None = None  # вопросы: ответ лектора


class Part(BaseModel):
    """Раздел документа: «Коротко», раздел лекции, глоссарий…"""

    kind: Literal["summary", "section", "examples", "glossary", "qa", "reading", "unclear"]
    title: str
    at: float | None = None
    items: list[Item]


class Block(BaseModel):
    """Смысловой блок профиля `raw`: заголовок над строкой транскрипта."""

    title: str
    at: float


class Document(BaseModel):
    profile: str
    title: str
    llm_model: str
    parts: list[Part] = []
    blocks: list[Block] = []


class Detail(BaseModel):
    card: Card
    language: str | None = None
    asr_model: str | None = None
    lines: list[Line] = []
    document: Document | None = None
    verification: str | None = None
    notes: list[str] = []
    audio: bool = False
    # Почему нет транскрипта: прогонов не было, кэш почищен `gc`…
    missing: str | None = None


def cards(db: Database) -> list[Card]:
    return [_card(row) for row in db.latest_runs()]


def detail(
    db: Database, store: ArtifactStore, settings: Settings, recording_id: int
) -> Detail | None:
    row = db.latest_run(recording_id)
    if row is None:
        return None
    result = Detail(
        card=_card(row),
        notes=[str(p) for kind, p in db.outputs(recording_id) if kind == "note"],
        audio=audio_file(db, store, recording_id) is not None,
    )
    run = db.last_done_run(recording_id, "summarize") or db.last_done_run(recording_id, "asr")
    if run is None:
        result.missing = "транскрипта ещё нет: запись не расшифровывалась"
        return result
    done = db.run_stages(run)
    asr_dir = store.located("asr", done["asr"].key)
    if not (asr_dir / MANIFEST).is_file():
        result.missing = "транскрипт удалён из кэша (summuraze gc) — прогони запись заново"
        return result
    source = Path(row.path)
    transcript = stages.load_transcript(asr_dir, source)
    summary_dir = store.located("summarize", done["summarize"].key) if "summarize" in done else None
    if summary_dir is not None and not (summary_dir / MANIFEST).is_file():
        summary_dir = None
    transcript = _as_summarized(transcript, summary_dir, settings)
    result.language = transcript.language
    result.asr_model = transcript.asr_model
    result.lines = [
        Line(
            start=s.start,
            end=s.end,
            text=s.text.strip(),
            suspect=s.suspect,
            reason=s.suspect_reason,
        )
        for s in transcript.segments
        if s.text.strip()
    ]
    if summary_dir is None:
        return result

    payload = json.loads((summary_dir / "summary.json").read_text(encoding="utf-8"))
    try:
        profile = get_profile(payload["profile"])
    except ValueError:
        return result  # профиль из будущей версии — транскрипт всё равно покажем
    summary, document = stages.load_summary(summary_dir, profile)
    report: Report | None = None
    if "verify" in done and (store.located("verify", done["verify"].key) / MANIFEST).is_file():
        report = stages.load_verify(store.located("verify", done["verify"].key))[0]
        result.verification = report.describe()
    result.document = _document(profile.name, summary.model, document, transcript, report, source)
    return result


def audio_file(db: Database, store: ArtifactStore, recording_id: int) -> Path | None:
    """Что играть в плеере: сжатое аудио заметки, wav ингеста или оригинал — что уцелело.
    После `prune` (#49) звука нет нигде, и это честный ответ, а не ошибка."""
    for stage, name in (("audio", stages.NOTE_AUDIO), ("ingest", "audio.wav")):
        for _, key in sorted(db.stage_keys(recording_id, (stage,))):
            path = store.located(_ARTIFACT_STAGE.get(stage, stage), key) / name
            if path.is_file():
                return path
    row = db.latest_run(recording_id)
    if row is not None and row.pruned_at is None and Path(row.path).is_file():
        return Path(row.path)
    return None


def _card(row: LatestRun) -> Card:
    status: Status | None = row.status
    if status == "running" and not _alive(row.pid):
        # Процесс убит (kill, выключили ноутбук) — finish_run не случился, статус так и остался.
        status = "interrupted"
    return Card(
        id=row.id,
        name=Path(row.path).name,
        path=row.path,
        first_seen=row.first_seen,
        duration=row.duration,
        pruned=row.pruned_at is not None,
        status=status,
        stage=row.stage,
        stage_title=STAGE_TITLES.get(row.stage or ""),
        stage_started_at=row.stage_started_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        error=row.error,
        profile=row.profile,
        where=llm_destination(row.llm_provider) if row.llm_provider else None,
        llm_model=row.llm_model,
        cost_usd=row.cost_usd,
    )


def _alive(pid: int | None) -> bool:
    if pid is None:
        return False  # прогон до миграции 4: живым он быть уже не может
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # процесс есть, просто чужой
    return True


def _as_summarized(
    transcript: Transcript, summary_dir: Path | None, settings: Settings
) -> Transcript:
    """Транскрипт таким, каким его видел конспект: с заменами терминов и пометками детектора.
    Замены — из манифеста саммари; без саммари — из общего словаря (`--terms` курса неизвестен)."""
    if summary_dir is not None:
        params = json.loads((summary_dir / MANIFEST).read_text(encoding="utf-8"))["params"]
        rules = params.get("replacements", [])
        glossary = Glossary(replacements=tuple((heard, written) for heard, written in rules))
    else:
        glossary = load_glossary([settings.asr.terms_file])
    segments = apply_replacements(transcript.segments, glossary)
    return transcript.model_copy(update={"segments": repetition.flag_suspects(segments)})


def _document(
    profile: str,
    model: str,
    document: BaseModel,
    transcript: Transcript,
    report: Report | None,
    source: Path,
) -> Document:
    """Документ профиля в виде, который страница рисует без знания профилей: разделы и пункты
    с таймкодом в секундах и пометкой проверки. Порядок и отсев повторов — как в итоговой
    заметке (#50), чтобы браузер и Obsidian показывали одно и то же."""
    seen: set[str] = set()

    def fresh(text: str) -> bool:
        key = thought_key(text)
        if not key or key in seen:
            return False
        seen.add(key)
        return True

    def at(value: str) -> float | None:
        return resolve_at(value, transcript)

    def mark(path: str) -> Mark | None:
        return report.marks.get(path) if report is not None else None

    def claims(items: list[Claim], path: str) -> list[Item]:
        return [
            Item(text=c.text.strip(), at=at(c.at), mark=mark(f"{path}.{i}"))
            for i, c in enumerate(items)
            if fresh(c.text)
        ]

    if isinstance(document, RawDoc):
        blocks = sorted(
            (
                Block(title=b.title.strip(), at=t)
                for b in document.blocks
                if (t := at(b.at)) is not None
            ),
            key=lambda b: b.at,
        )
        return Document(profile=profile, title=source.stem, llm_model=model, blocks=blocks)
    if not isinstance(document, LectureDoc):  # pragma: no cover — новый профиль без вида
        raise TypeError(f"нет вида для документа {type(document).__name__}")

    parts = [Part(kind="summary", title="Коротко", items=claims(document.tldr, "tldr"))]
    for i, sec in enumerate(document.sections):
        parts.append(
            Part(
                kind="section",
                title=sec.title.strip(),
                at=at(sec.at),
                items=claims(sec.theses, f"sections.{i}.theses"),
            )
        )
    parts.append(
        Part(
            kind="examples", title="Формулы и примеры", items=claims(document.examples, "examples")
        )
    )
    glossary = [
        Item(
            text=t.definition.strip(), term=t.term.strip(), at=at(t.at), mark=mark(f"glossary.{i}")
        )
        for i, t in enumerate(document.glossary)
        if fresh(f"термин {t.term}")
    ]
    parts.append(Part(kind="glossary", title="Глоссарий", items=glossary))
    qa = [
        Item(
            text=q.question.strip(),
            answer=q.answer.strip() or None,
            at=at(q.at),
            mark=mark(f"qa.{i}"),
        )
        for i, q in enumerate(document.qa)
        if fresh(q.question)
    ]
    parts.append(Part(kind="qa", title="Вопросы", items=qa))
    parts.append(
        Part(kind="reading", title="Что почитать", items=claims(document.reading, "reading"))
    )
    parts.append(
        Part(kind="unclear", title="Неразборчиво", items=claims(document.unclear, "unclear"))
    )
    return Document(
        profile=profile,
        title=document.title.strip() or source.stem,
        llm_model=model,
        # Пустой раздел не рисуем — как в заметке; раздел лекции без тезисов — рисуем: заголовок
        # с таймкодом сам по себе навигация.
        parts=[p for p in parts if p.items or p.kind == "section"],
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
