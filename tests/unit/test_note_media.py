"""Заметка Obsidian: имена файлов, ссылки Media Extended, субтитры."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from summuraze.render.markdown import NoteLayout, _Body, note_layout
from summuraze.render.subtitles import srt_time, to_srt
from summuraze.types import Segment, Transcript

T = Transcript(
    source=Path("/x/a.mp4"),
    duration=4000,
    language="ru",
    asr_model="m",
    segments=[
        Segment(start=0.0, end=2.5, text=" Привет. "),
        Segment(start=3661.25, end=3661.25, text="Продолжение следует...", suspect=True),
        Segment(start=3670, end=3671, text="   "),
    ],
)


def test_srt_time() -> None:
    assert srt_time(0) == "00:00:00,000"
    assert srt_time(3661.2504) == "01:01:01,250"


def test_srt_cues() -> None:
    assert to_srt(T) == (
        "1\n00:00:00,000 --> 00:00:02,500\nПривет.\n\n"
        "2\n01:01:01,250 --> 01:01:01,750\n[?] Продолжение следует...\n"
    )


def test_layout_names(tmp_path: Path) -> None:
    src = tmp_path / "Лекция #3 [ЦОС] a|b.mp4"
    src.write_bytes(b"")
    os.utime(src, (1_761_500_000, 1_761_500_000))  # 2025-10-26
    main = note_layout(tmp_path, src, "lecture", "Записи")
    raw = note_layout(tmp_path, src, "raw", "Записи")
    assert main.base == "2025-10-26 Лекция _3 _ЦОС_ a_b"
    assert main.note.name == f"{main.base}.md"
    assert raw.note.name == f"{main.base} (raw).md"
    assert raw.audio == main.audio == tmp_path / "Записи" / f"{main.base}.ogg"


@pytest.mark.parametrize(("lang", "name"), [("ru", "x.ru.srt"), ("", "x.srt"), ("rus", "x.srt")])
def test_subtitle_language_suffix(lang: str, name: str) -> None:
    assert NoteLayout(Path("n.md"), Path("m"), "x").subtitles(lang).name == name


def test_timecode_link() -> None:
    body = _Body(T, "2026-10-27 лекция.ogg")
    assert body.tc(95.24) == "[[2026-10-27 лекция.ogg#t=95.2|00:01:35]]"
    assert body.tc(None) == "⚠️ без таймкода"
    assert _Body(T, None).tc(95) == "`00:01:35`"
