"""Срок хранения записей (#49): аудио уходит, текст остаётся.

~30 часов записи в неделю — это гигабайты аудио, а транскрипт, конспект и история прогонов —
килобайты. Поэтому по сроку удаляется только звук записи:

- оригинал — если его сделал `summuraze record` (лежит в `SUMMURAZE_RECORD_DIR`); чужой файл
  вроде `~/Downloads/лекция.mp4` — только по явному `--include-foreign`;
- wav ингеста и сжатое аудио заметки в кэше артефактов — по ключам из истории прогонов;
- аудио и субтитры в хранилище Obsidian — по таблице `outputs`, куда их записал рендер.

Заметки остаются, но таймкоды-ссылки в них становятся текстом с пометкой «аудио удалено», а не
ведут в пустоту. Транскрипт и конспект в кэше не трогаются: повторный прогон без аудио невозможен,
но и не нужен — заметка уже есть.

Дата записи — момент, когда конвейер впервые увидел файл (`recordings.first_seen`), а не mtime:
копирование или распаковка файла меняет mtime, а срок хранения сдвигаться от этого не должен.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

from summuraze.render.markdown import detach_audio, rewrite_note
from summuraze.store.artifacts import ArtifactStore, tree_size
from summuraze.store.db import Database, RecordingRow

# Стадия в истории прогонов → каталог её артефактов в кэше. Только то, что содержит звук.
AUDIO_STAGES = {"ingest": "ingest", "audio": "note-audio"}

ItemKind = Literal["original", "cache", "audio", "subtitles"]


@dataclass(frozen=True)
class Item:
    kind: ItemKind
    path: Path
    size: int


@dataclass
class Victim:
    recording: RecordingRow
    recorded: datetime
    items: list[Item] = field(default_factory=list)
    notes: list[Path] = field(default_factory=list)
    media: list[str] = field(default_factory=list)
    foreign_kept: bool = False

    @property
    def size(self) -> int:
        return sum(i.size for i in self.items)

    @property
    def name(self) -> str:
        return Path(self.recording.path).name


@dataclass
class PrunePlan:
    cutoff: datetime
    victims: list[Victim]

    @property
    def size(self) -> int:
        return sum(v.size for v in self.victims)

    @property
    def foreign_kept(self) -> int:
        return sum(v.foreign_kept for v in self.victims)


def cutoff(
    *, before: str | None, older_than: str | None, retention_days: float | None, now: datetime
) -> datetime:
    """Граница «записано раньше» — местное время. `--before` — полночь этой даты."""
    if before is not None and older_than is not None:
        raise ValueError("укажи что-то одно: --before или --older-than")
    if before is not None:
        try:
            day = date.fromisoformat(before)
        except ValueError:
            raise ValueError(f"дата в виде ГГГГ-ММ-ДД, а не {before!r}") from None
        return datetime.combine(day, datetime.min.time()).astimezone()
    if older_than is not None:
        return now - timedelta(days=parse_days(older_than))
    if retention_days is not None:
        return now - timedelta(days=retention_days)
    raise ValueError(
        "не задан срок: --before ГГГГ-ММ-ДД, --older-than 30d или SUMMURAZE_RETENTION_DAYS"
    )


def parse_days(text: str) -> float:
    """`30d`, `30`, `2w` → дни."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([dwдн]?)\s*", text.lower())
    if not m:
        raise ValueError(f"срок в виде 30d или 2w, а не {text!r}")
    value, unit = float(m[1]), m[2]
    return value * 7 if unit in ("w", "н") else value


def plan(
    db: Database,
    artifacts: ArtifactStore,
    record_dir: Path,
    cutoff: datetime,
    *,
    include_foreign: bool = False,
    keep: int | None = None,
) -> PrunePlan:
    """Что удалится; на диске ничего не меняется. `keep` — запись текущего прогона: её аудио
    только что опубликовано, и удалять его в том же прогоне было бы абсурдом."""
    own_dir = record_dir.expanduser().resolve()
    victims: list[Victim] = []
    for rec in db.unpruned_recordings():
        recorded = datetime.fromisoformat(rec.first_seen).astimezone()
        if recorded >= cutoff or rec.id == keep:
            continue
        victim = Victim(rec, recorded)
        original = Path(rec.path)
        if original.is_file():
            if include_foreign or original.resolve().is_relative_to(own_dir):
                victim.items.append(Item("original", original, original.stat().st_size))
            else:
                victim.foreign_kept = True
        for stage, key in sorted(db.stage_keys(rec.id, tuple(AUDIO_STAGES))):
            folder = artifacts.located(AUDIO_STAGES[stage], key)
            if folder.is_dir():
                victim.items.append(Item("cache", folder, tree_size(folder)))
        for kind, path in db.outputs(rec.id):
            if kind == "note":
                if path.is_file():
                    victim.notes.append(path)
                continue
            if kind == "audio":
                victim.media.append(path.name)
            if path.is_file():
                victim.items.append(Item(kind, path, path.stat().st_size))
        if victim.items:
            victims.append(victim)
    return PrunePlan(cutoff, victims)


def apply(pruned: PrunePlan, db: Database, *, today: date) -> int:
    """Удалить по плану; вернуть, сколько заметок переписано."""
    rewritten = 0
    for victim in pruned.victims:
        for item in victim.items:
            if item.path.is_dir():
                shutil.rmtree(item.path, ignore_errors=True)
            else:
                item.path.unlink(missing_ok=True)
        for note in victim.notes:
            text = note.read_text(encoding="utf-8")
            new = text
            for media in victim.media:
                new = detach_audio(new, media, today.isoformat())
            if new != text:
                rewrite_note(note, new)
                rewritten += 1
        db.mark_pruned(victim.recording.id)
    return rewritten


def human_size(size: int) -> str:
    return f"{size / 2**30:.1f} ГБ" if size >= 2**30 else f"{size / 2**20:.0f} МБ"
