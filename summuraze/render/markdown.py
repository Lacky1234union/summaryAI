"""Заметка с саммари и транскриптом.

Документ профиля (#17) приходит от модели JSON-ом и собирается в Markdown здесь: таймкод каждого
утверждения привязывается к строке транскрипта (`resolve_at`). Ссылки таймкодов для Media
Extended, аудио в хранилище и frontmatter целиком — #18.

Имя заметки — от даты записи (mtime исходника) и имени файла, а не от времени прогона:
повторный прогон перезаписывает свою заметку, а не плодит «лекция (1).md».
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from summuraze.summarize.profiles import Claim, LectureDoc, RawDoc
from summuraze.types import AudioInfo, LLMResult, Segment, Transcript


def fmt_timecode(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


DEFAULT_PROFILE = "lecture"


def note_path(notes_dir: Path, source: Path, profile: str = DEFAULT_PROFILE) -> Path:
    """Конспект — основная заметка записи; другой профиль кладётся рядом, а не поверх неё."""
    recorded = datetime.fromtimestamp(source.stat().st_mtime).date().isoformat()
    suffix = "" if profile == DEFAULT_PROFILE else f" ({profile})"
    return notes_dir / f"{recorded} {source.stem}{suffix}.md"


def _transcript_line(s: Segment) -> str:
    text = s.text.strip()
    if s.suspect:
        # Не удаляем (#12): читатель видит, что Whisper тут мог выдумать, и может переслушать.
        text = f"⚠️ *{text}* — {s.suspect_reason}"
    return f"- {_tc(s.start)} {text}"


_TIMECODE = re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}))?")


def resolve_at(at: str, transcript: Transcript) -> float | None:
    """Таймкод из ответа модели → начало строки транскрипта, к которой он относится.

    Модель цитирует таймкоды строк префикса (`[ЧЧ:ММ:СС]`, начало сегмента с точностью до
    секунды) — привязываем к последнему сегменту, начавшемуся не позже. Таймкод за концом
    записи или нечитаемый — `None`: утверждение без проверяемой привязки (guardrail #5).
    """
    match = _TIMECODE.search(at)
    if not match:
        return None
    a, b, c = match.groups()
    t = int(a) * 60 + int(b) if c is None else int(a) * 3600 + int(b) * 60 + int(c)
    if t > transcript.duration + 1:
        return None
    starts = [seg.start for seg in transcript.segments if seg.start <= t + 0.999]
    return starts[-1] if starts else float(t)


def _tc(seconds: float | None) -> str:
    # Единственное место, где таймкод становится текстом заметки: ссылки для Media Extended (#18)
    # меняют только его.
    return "⚠️ без таймкода" if seconds is None else f"`{fmt_timecode(seconds)}`"


def _claims(items: list[Claim], transcript: Transcript) -> list[str]:
    return [f"- {c.text.strip()} {_tc(resolve_at(c.at, transcript))}" for c in items]


def _section(title: str, body: list[str]) -> list[str]:
    return [f"## {title}", "", *body, ""] if body else []


def _lecture(doc: LectureDoc, transcript: Transcript) -> list[str]:
    lines = [*_section("Коротко", _claims(doc.tldr, transcript))]
    for sec in doc.sections:
        lines += [f"## {sec.title.strip()} {_tc(resolve_at(sec.at, transcript))}", ""]
        lines += [*_claims(sec.theses, transcript), ""]
    lines += _section(
        "Глоссарий",
        [
            f"- **{t.term.strip()}** — {t.definition.strip()} {_tc(resolve_at(t.at, transcript))}"
            for t in doc.glossary
        ],
    )
    lines += _section("Формулы и примеры", _claims(doc.examples, transcript))
    qa: list[str] = []
    for q in doc.qa:
        qa.append(f"- **Вопрос:** {q.question.strip()} {_tc(resolve_at(q.at, transcript))}")
        if q.answer.strip():
            qa.append(f"  **Ответ:** {q.answer.strip()}")
    lines += _section("Вопросы и ответы", qa)
    lines += _section("Что почитать", _claims(doc.reading, transcript))
    lines += _section("Неразборчиво", _claims(doc.unclear, transcript))
    lines += ["## Транскрипт", "", *(_transcript_line(s) for s in transcript.segments), ""]
    return lines


def _raw(doc: RawDoc, transcript: Transcript) -> list[str]:
    """Блоки — только заголовки от модели; текст под ними дословно из ASR."""
    blocks = sorted(
        (
            (t, b.title.strip())
            for b in doc.blocks
            if (t := resolve_at(b.at, transcript)) is not None
        ),
        key=lambda x: x[0],
    )
    if not blocks:
        return ["## Транскрипт", "", *(_transcript_line(s) for s in transcript.segments), ""]
    lines: list[str] = []
    current = -1
    for seg in transcript.segments:
        index = max(sum(1 for t, _ in blocks if t <= seg.start) - 1, 0)
        if index != current:
            current = index
            t, title = blocks[index]
            lines += ["", f"## {title} {_tc(t)}", ""]
        lines.append(_transcript_line(seg))
    return [*lines, ""]


def render_note(
    path: Path,
    *,
    audio: AudioInfo,
    transcript: Transcript,
    summary: LLMResult,
    document: BaseModel,
    profile: str,
) -> Path:
    if isinstance(document, LectureDoc):
        title = document.title.strip() or audio.source.stem
        body = _lecture(document, transcript)
    elif isinstance(document, RawDoc):
        title = audio.source.stem
        body = _raw(document, transcript)
    else:  # pragma: no cover — новый профиль без рендера
        raise TypeError(f"нет рендера для документа {type(document).__name__}")
    lines = [
        "---",
        f'source: "{audio.source}"',
        f"duration: {fmt_timecode(audio.duration)}",
        f"profile: {profile}",
        f"asr_model: {transcript.asr_model}",
        f"llm_model: {summary.model}",
        "---",
        "",
        f"# {title}",
        "",
        *body,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    # Атомарно: Obsidian и LiveSync не должны увидеть и разослать половину заметки.
    tmp.replace(path)
    return path
