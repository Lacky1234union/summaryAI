"""Субтитры .srt из транскрипта — Media Extended показывает их интерактивным транскриптом.

Плагин находит субтитры сам, если они лежат рядом с медиа и называются так же: `запись.ogg` →
`запись.ru.srt` (код языка перед расширением — ISO 639-1).

Refs #18
"""

from __future__ import annotations

from summuraze.types import Transcript


def srt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(transcript: Transcript) -> str:
    cues = []
    for seg in transcript.segments:
        text = seg.text.strip()
        if not text:
            continue
        if seg.suspect:
            text = f"[?] {text}"  # та же пометка, что в промпте саммари (#12)
        end = max(seg.end, seg.start + 0.5)  # нулевая длительность — плеер реплику не покажет
        cues.append(f"{len(cues) + 1}\n{srt_time(seg.start)} --> {srt_time(end)}\n{text}\n")
    return "\n".join(cues)
