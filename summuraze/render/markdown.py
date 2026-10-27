"""Заметка с саммари и транскриптом.

Этап 0 — минимальная заметка, чтобы конвейер заканчивался файлом. Ссылки таймкодов для
Media Extended, аудио в хранилище и frontmatter целиком — #18.

Имя заметки — от даты записи (mtime исходника) и имени файла, а не от времени прогона:
повторный прогон перезаписывает свою заметку, а не плодит «лекция (1).md».
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from summuraze.types import AudioInfo, LLMResult, Segment, Transcript


def fmt_timecode(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def note_path(notes_dir: Path, source: Path) -> Path:
    recorded = datetime.fromtimestamp(source.stat().st_mtime).date().isoformat()
    return notes_dir / f"{recorded} {source.stem}.md"


def _transcript_line(s: Segment) -> str:
    text = s.text.strip()
    if s.suspect:
        # Не удаляем (#12): читатель видит, что Whisper тут мог выдумать, и может переслушать.
        text = f"⚠️ *{text}* — {s.suspect_reason}"
    return f"- `{fmt_timecode(s.start)}` {text}"


def render_note(
    path: Path,
    *,
    audio: AudioInfo,
    transcript: Transcript,
    summary: LLMResult,
    profile: str,
) -> Path:
    lines = [
        "---",
        f'source: "{audio.source}"',
        f"duration: {fmt_timecode(audio.duration)}",
        f"profile: {profile}",
        f"asr_model: {transcript.asr_model}",
        f"llm_model: {summary.model}",
        "---",
        "",
        f"# {audio.source.stem}",
        "",
        summary.text.strip(),
        "",
        "## Транскрипт",
        "",
        *(_transcript_line(s) for s in transcript.segments),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    # Атомарно: Obsidian и LiveSync не должны увидеть и разослать половину заметки.
    tmp.replace(path)
    return path
