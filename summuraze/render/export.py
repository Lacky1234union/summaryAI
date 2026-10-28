"""Экспорт транскрипта в файлы рядом с записью (#14).

Транскрипт полезен и без саммари: его читают, по нему ищут, его подкладывают плееру субтитрами.
Имена — `запись.<формат>` без кода языка: mpv и VLC сами подхватывают `запись.srt` рядом
с `запись.mp4`, а `запись.ru.srt` — уже не все.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path

from summuraze.render.markdown import short_timecode
from summuraze.render.subtitles import SUSPECT_MARK, to_srt, to_vtt
from summuraze.types import Segment, Transcript

# Пауза дольше — новый абзац: лектор переключился на доску, слайд или следующую мысль.
PARAGRAPH_PAUSE = 2.0
# Без пауз абзац всё равно режется — по концу фразы, иначе это стена текста. Whisper на
# разговорной речи нередко пишет вовсе без точек — тогда режем по длине, на границе сегмента.
PARAGRAPH_CHARS = 600
PARAGRAPH_MAX = 1200
_SENTENCE_END = re.compile(r"[.!?…]$")


def _paragraphs(segments: Iterable[Segment]) -> list[list[Segment]]:
    out: list[list[Segment]] = []
    size = 0
    for seg in segments:
        if not seg.text.strip():
            continue
        if out:
            prev = out[-1][-1]
            pause = seg.start - prev.end >= PARAGRAPH_PAUSE
            full = size >= PARAGRAPH_MAX or (
                size >= PARAGRAPH_CHARS and _SENTENCE_END.search(prev.text.strip())
            )
            if pause or full:
                out.append([])
                size = 0
        else:
            out.append([])
        out[-1].append(seg)
        size += len(seg.text.strip()) + 1
    return out


def to_txt(transcript: Transcript) -> str:
    """Абзацы с таймкодом начала; сомнительные куски — с той же пометкой, что в саммари."""
    blocks = []
    for para in _paragraphs(transcript.segments):
        text = " ".join(
            f"{SUSPECT_MARK} {s.text.strip()}" if s.suspect else s.text.strip() for s in para
        )
        blocks.append(f"[{short_timecode(para[0].start)}] {text}")
    return "\n\n".join(blocks) + "\n" if blocks else ""


def to_json(transcript: Transcript) -> str:
    return transcript.model_dump_json(indent=2) + "\n"


FORMATS: dict[str, Callable[[Transcript], str]] = {
    "json": to_json,
    "txt": to_txt,
    "srt": to_srt,
    "vtt": to_vtt,
}


def parse_formats(text: str) -> list[str]:
    """`json,srt` → ['json', 'srt']; `all` — все."""
    names = [n.strip().lower().lstrip(".") for n in text.split(",") if n.strip()]
    if names == ["all"]:
        return list(FORMATS)
    unknown = [n for n in names if n not in FORMATS]
    if unknown or not names:
        raise ValueError(
            f"неизвестный формат {', '.join(unknown) or '(пусто)'}; "
            f"есть {', '.join(FORMATS)} или all"
        )
    return list(dict.fromkeys(names))


def export(
    transcript: Transcript, source: Path, formats: Iterable[str], out_dir: Path | None = None
) -> list[Path]:
    folder = out_dir if out_dir is not None else source.parent
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = folder / f"{source.stem}.{fmt}"
        path.write_text(FORMATS[fmt](transcript), encoding="utf-8")
        written.append(path)
    return written
