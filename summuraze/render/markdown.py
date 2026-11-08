"""Заметка Obsidian: документ профиля, таймкоды-ссылки, аудио и субтитры рядом.

Документ профиля (#17) приходит от модели JSON-ом и собирается в Markdown здесь: таймкод каждого
утверждения привязывается к строке транскрипта (`resolve_at`) и становится ссылкой Media Extended
`[[запись.ogg#t=95|00:01:35]]` — клик открывает запись на этой секунде (#18). Проверка
утверждения стоит две секунды — это и есть защита от вымысла модели (guardrail #5).

Раскладка в папке заметок (`SUMMURAZE_OUTPUT_DIR`):

    2026-10-27 лекция.md              конспект (основной профиль)
    2026-10-27 лекция (raw).md        другой профиль — рядом, а не поверх
    Записи/2026-10-27 лекция.ogg      сжатое аудио, общее для профилей
    Записи/2026-10-27 лекция.ru.srt   субтитры: Media Extended находит их по имени

Имена — от даты записи (mtime исходника) и имени файла, а не от времени прогона: повторный
прогон перезаписывает своё, а не плодит «лекция (1).md». Файлы пишутся, только если изменились:
LiveSync не должен заново рассылать 15 МБ аудио из-за повторного прогона.

Два вида заметки (#50). Итоговый — для чтения: короткий frontmatter, TL;DR и вопросы в callout'ах,
транскрипт и «неразборчиво» свёрнуты, таймкоды `▶ 01:37`. Отладочный (`SUMMURAZE_NOTE_DEBUG`,
`--debug`) — всё подряд: модели, стоимость, исходник, полные таймкоды; по нему разбирают, что
сделала модель. Содержание одно и то же — разная только подача.

Пометки проверки (#22) стоят после таймкода утверждения: ❌ — нет в записи, ◐ — частично,
↪ — сказано в другом месте (ссылка туда), ❔ — модель-судья не ответила. Подтверждённое — без
пометки: заметка читается, а не пестрит. Сводка — в шапке.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from summuraze.render.subtitles import to_srt
from summuraze.summarize.profiles import Claim, LectureDoc, RawDoc
from summuraze.types import AudioInfo, LLMResult, Segment, Transcript

if TYPE_CHECKING:
    from summuraze.summarize.verify import Report

DEFAULT_PROFILE = "lecture"
_PROFILE_TAGS = {"lecture": "лекция", "raw": "транскрипт"}
# Символы, которые ломают вики-ссылку Obsidian (`#`, `|`, `^`, `[`, `]`) или имя файла на
# какой-нибудь из ОС, где открывают хранилище.
_UNSAFE = re.compile(r'[\\/:*?"<>|#^\[\]]')
_TIMECODE = re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}))?")


def fmt_timecode(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def short_timecode(seconds: float) -> str:
    """`01:37`, а час и больше — `1:02:05`: в заметке лекции часы почти всегда нулевые."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


@dataclass(frozen=True)
class NoteLayout:
    note: Path
    media_dir: Path
    base: str  # «дата имя» записи: общее для заметок всех профилей и для их аудио

    @property
    def audio(self) -> Path:
        return self.media_dir / f"{self.base}.ogg"

    def subtitles(self, language: str) -> Path:
        lang = f".{language}" if re.fullmatch(r"[a-z]{2}", language or "") else ""
        return self.media_dir / f"{self.base}{lang}.srt"


def note_layout(
    notes_dir: Path, source: Path, profile: str = DEFAULT_PROFILE, media_dir: str = "Записи"
) -> NoteLayout:
    """Конспект — основная заметка записи; другой профиль кладётся рядом, а не поверх неё."""
    recorded = datetime.fromtimestamp(source.stat().st_mtime).date().isoformat()
    base = _UNSAFE.sub("_", f"{recorded} {source.stem}").strip()
    suffix = "" if profile == DEFAULT_PROFILE else f" ({_UNSAFE.sub('_', profile)})"
    return NoteLayout(notes_dir / f"{base}{suffix}.md", notes_dir / media_dir, base)


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


def thought_key(text: str) -> str:
    """Мысль без регистра и пунктуации: повтор по этому ключу в заметку не идёт."""
    return " ".join(re.findall(r"\w+", text.lower()))


@dataclass(frozen=True)
class _Body:
    """Сборка тела заметки: транскрипт для привязки таймкодов и аудио для ссылок.

    Одна мысль попадает в заметку один раз: сбойный провайдер однажды скопировал тезисы во
    все разделы подряд (2026-10-28) — повтор по нормализованному тексту выбрасывается.
    """

    transcript: Transcript
    media: str | None  # имя файла аудио в хранилище; None — таймкоды текстом
    debug: bool = True
    report: Report | None = None
    seen: set[str] = field(default_factory=set)

    def fresh(self, text: str) -> bool:
        key = thought_key(text)
        if not key or key in self.seen:
            return False
        self.seen.add(key)
        return True

    def tc(self, seconds: float | None) -> str:
        # Единственное место, где таймкод становится текстом заметки.
        if seconds is None:
            return "⚠️ без таймкода"
        if self.debug:
            label = fmt_timecode(seconds)
        elif self.media is None:
            label = short_timecode(seconds)
        else:
            label = f"▶ {short_timecode(seconds)}"  # ▶ — кликабельно, откроет запись
        if self.media is None:
            return f"`{label}`"
        return f"[[{self.media}#t={seconds:.1f}|{label}]]"

    def at(self, at: str, path: str | None = None) -> str:
        return self.tc(resolve_at(at, self.transcript)) + self.mark(path)

    def mark(self, path: str | None) -> str:
        """Пометка проверки после таймкода; `path` — адрес утверждения в документе."""
        if self.report is None or path is None or (m := self.report.marks.get(path)) is None:
            return ""
        note = f": {m.note}" if m.note else ""
        if m.status == "unsupported":
            return f" ❌ *нет в записи{note}*"
        if m.status == "partial":
            return f" ◐ *частично{note}*"
        if m.status == "moved":
            return f" ↪ *сказано в* {self.tc(m.evidence)}"
        if m.status == "unchecked":
            return " ❔ *не проверено*"
        return ""

    def claims(self, items: list[Claim], path: str) -> list[str]:
        return [
            f"- {c.text.strip()} {self.at(c.at, f'{path}.{i}')}"
            for i, c in enumerate(items)
            if self.fresh(c.text)
        ]

    def line(self, s: Segment) -> str:
        text = s.text.strip()
        if s.suspect:
            # Не удаляем (#12): читатель видит, что Whisper тут мог выдумать, и может переслушать.
            text = f"⚠️ *{text}* — {s.suspect_reason}"
        return f"- {self.tc(s.start)} {text}"

    def transcript_section(self) -> list[str]:
        lines = [self.line(s) for s in self.transcript.segments]
        if not self.debug:
            return _callout("quote", "Транскрипт", lines, folded=True)
        return ["## Транскрипт", "", *lines, ""]

    def lecture(self, doc: LectureDoc) -> list[str]:
        return self.lecture_debug(doc) if self.debug else self.lecture_clean(doc)

    def lecture_clean(self, doc: LectureDoc) -> list[str]:
        """Итоговый вид: читается сверху вниз, справочное — в callout'ах, сырое — свёрнуто."""
        lines = _callout("summary", "Коротко", self.claims(doc.tldr, "tldr"))
        lines += self.sections(doc)
        lines += _section("Формулы и примеры", self.claims(doc.examples, "examples"))
        lines += _callout("abstract", "Глоссарий", self.glossary(doc))
        for i, q in enumerate(doc.qa):
            if self.fresh(q.question):
                answer = [q.answer.strip()] if q.answer.strip() else []
                title = f"{q.question.strip()} {self.at(q.at, f'qa.{i}')}"
                lines += _callout("question", title, answer)
        lines += _callout("info", "Что почитать", self.claims(doc.reading, "reading"))
        lines += _callout(
            "warning", "Неразборчиво", self.claims(doc.unclear, "unclear"), folded=True
        )
        return lines + self.transcript_section()

    def sections(self, doc: LectureDoc) -> list[str]:
        lines: list[str] = []
        for i, sec in enumerate(doc.sections):
            theses = self.claims(sec.theses, f"sections.{i}.theses")
            lines += [f"## {sec.title.strip()} {self.at(sec.at)}", "", *theses, ""]
        return lines

    def glossary(self, doc: LectureDoc) -> list[str]:
        return [
            f"- **{t.term.strip()}** — {t.definition.strip()} {self.at(t.at, f'glossary.{i}')}"
            for i, t in enumerate(doc.glossary)
            if self.fresh(f"термин {t.term}")
        ]

    def lecture_debug(self, doc: LectureDoc) -> list[str]:
        lines = _section("Коротко", self.claims(doc.tldr, "tldr"))
        lines += self.sections(doc)
        lines += _section("Глоссарий", self.glossary(doc))
        lines += _section("Формулы и примеры", self.claims(doc.examples, "examples"))
        qa: list[str] = []
        for i, q in enumerate(doc.qa):
            if not self.fresh(q.question):
                continue
            qa.append(f"- **Вопрос:** {q.question.strip()} {self.at(q.at, f'qa.{i}')}")
            if q.answer.strip():
                qa.append(f"  **Ответ:** {q.answer.strip()}")
        lines += _section("Вопросы и ответы", qa)
        lines += _section("Что почитать", self.claims(doc.reading, "reading"))
        lines += _section("Неразборчиво", self.claims(doc.unclear, "unclear"))
        return lines + self.transcript_section()

    def raw(self, doc: RawDoc) -> list[str]:
        """Блоки — только заголовки от модели; текст под ними дословно из ASR."""
        blocks = sorted(
            (
                (t, b.title.strip())
                for b in doc.blocks
                if (t := resolve_at(b.at, self.transcript)) is not None
            ),
            key=lambda x: x[0],
        )
        if not blocks:
            return self.transcript_section()
        lines: list[str] = []
        current = -1
        for seg in self.transcript.segments:
            index = max(sum(1 for t, _ in blocks if t <= seg.start) - 1, 0)
            if index != current:
                current = index
                t, title = blocks[index]
                lines += ["", f"## {title} {self.tc(t)}", ""]
            lines.append(self.line(seg))
        return [*lines, ""]


def _section(title: str, body: list[str]) -> list[str]:
    return [f"## {title}", "", *body, ""] if body else []


def _callout(kind: str, title: str, body: list[str], *, folded: bool = False) -> list[str]:
    """Callout Obsidian; пустой — не рисуем, кроме вопроса без ответа (заголовок и есть суть)."""
    if not body and kind != "question":
        return []
    mark = "-" if folded else ""
    return [f"> [!{kind}]{mark} {title}", *(f"> {line}" for line in body), ""]


def publish_media(layout: NoteLayout, audio: Path | None, transcript: Transcript) -> str | None:
    """Положить аудио и субтитры рядом с заметкой; вернуть имя аудио для ссылок."""
    if audio is None:
        return None
    _write_if_changed(layout.audio, audio.read_bytes())
    _write_if_changed(layout.subtitles(transcript.language), to_srt(transcript).encode("utf-8"))
    return layout.audio.name


def note_text(
    layout: NoteLayout,
    *,
    audio: AudioInfo,
    transcript: Transcript,
    summary: LLMResult,
    document: BaseModel,
    profile: str,
    media: str | None = None,
    debug: bool = False,
    report: Report | None = None,
) -> str:
    body = _Body(transcript, media, debug, report)
    if isinstance(document, LectureDoc):
        title = document.title.strip() or audio.source.stem
        lines = body.lecture(document)
    elif isinstance(document, RawDoc):
        title = audio.source.stem
        lines = body.raw(document)
    else:  # pragma: no cover — новый профиль без рендера
        raise TypeError(f"нет рендера для документа {type(document).__name__}")

    recorded = layout.base.split(" ", 1)[0]
    if debug:
        meta: dict[str, object] = {
            "title": title,
            "source": str(audio.source),
            "recorded": recorded,
            "duration": fmt_timecode(audio.duration),
            "profile": profile,
            "asr_model": transcript.asr_model,
            "llm_model": summary.model,
            "cost_usd": summary.cost_usd,
        }
        if report is not None:
            meta["verified"] = report.describe()
    else:
        # Файл назван по дате и имени записи — тема лекции в aliases, чтобы поиск и ссылки
        # Obsidian находили заметку по ней.
        meta = {
            "aliases": [title],
            "recorded": recorded,
            "duration": short_timecode(audio.duration),
        }
        if report is not None and report.marks:
            # «38/45» — сортируется и фильтруется в Dataview, а подробности — в callout'е ниже.
            meta["verified"] = f"{report.confirmed}/{len(report.marks)}"
    if media is not None:
        meta["audio"] = f"[[{media}]]"
    meta["tags"] = ["summuraze"] if debug else ["summuraze", _PROFILE_TAGS.get(profile, profile)]
    head = [
        "---",
        # JSON — подмножество YAML: кавычки и экранирование путей и названий без своей логики.
        *(f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in meta.items()),
        "---",
        "",
        f"# {title}",
        "",
    ]
    if media is not None:
        head += [f"![[{media}]]", ""]  # плеер: слушать можно и там, где нет Media Extended
    if report is not None and report.marks and not debug:
        kind = "success" if report.count("ok") == len(report.marks) else "warning"
        head += _callout(kind, "Проверено по записи", [report.describe()])
    return "\n".join([*head, *lines])


def render_note(
    layout: NoteLayout,
    *,
    audio: AudioInfo,
    transcript: Transcript,
    summary: LLMResult,
    document: BaseModel,
    profile: str,
    media: str | None = None,
    debug: bool = False,
    report: Report | None = None,
) -> Path:
    rewrite_note(
        layout.note,
        note_text(
            layout,
            audio=audio,
            transcript=transcript,
            summary=summary,
            document=document,
            profile=profile,
            media=media,
            debug=debug,
            report=report,
        ),
    )
    return layout.note


def _write_if_changed(path: Path, data: bytes) -> None:
    """Атомарно и только при изменении: Obsidian и LiveSync не должны видеть половину файла
    и не должны заново рассылать неизменившееся."""
    same_size = path.is_file() and path.stat().st_size == len(data)
    if same_size and hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(data).digest():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def detach_audio(note: str, media: str, removed: str) -> str:
    """Заметка без аудио (#49): таймкоды-ссылки становятся текстом, как при выключенном аудио,
    плеер — пометкой, когда и почему аудио удалено. Мёртвая ссылка выглядела бы как поломка."""
    name = re.escape(media)
    note = re.sub(rf"\[\[{name}#t=[\d.]+\|([^\]]+)\]\]", r"`\1`", note)
    note = re.sub(
        rf"^!\[\[{name}\]\]$",
        f"> [!info] Аудио удалено {removed} — истёк срок хранения записи.",
        note,
        flags=re.MULTILINE,
    )
    return re.sub(
        rf'^audio: "\[\[{name}\]\]"$',
        f"audio_removed: {json.dumps(removed)}",
        note,
        flags=re.MULTILINE,
    )


def rewrite_note(path: Path, text: str) -> None:
    _write_if_changed(path, text.encode("utf-8"))
