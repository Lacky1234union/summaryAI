"""Субтитры .srt/.vtt из транскрипта (#14, #18).

Один и тот же набор реплик идёт и в Obsidian (Media Extended показывает его интерактивным
транскриптом, ищет `запись.ru.srt` рядом с медиа), и в экспорт `summuraze transcribe` (mpv
подхватывает `запись.srt` рядом с `запись.mp4` сам).

Сегмент Whisper — это до 30 секунд речи, и одной репликой он не читается: строка уезжает за экран,
а текст висит дольше, чем звучит. Поэтому сегмент режется на реплики по словам с таймкодами —
не длиннее двух строк по 42 знака и 7 секунд показа, с предпочтением резать на конце фразы.
Реплика не пересекает границу сегмента: так пометка «[?]» остаётся ровно на сомнительном куске.
"""

from __future__ import annotations

from dataclasses import dataclass

from summuraze.types import Segment, Transcript

LINE_CHARS = 42
MAX_LINES = 2
MAX_SECONDS = 7.0
# Короче секунды реплику не успевают прочесть; меньше — только если иначе наедет на следующую.
MIN_SECONDS = 1.0
# Фраза кончилась, а реплика уже заполнена хотя бы наполовину — лучше разрезать здесь.
_SENTENCE_END = (".", "!", "?", "…")
SUSPECT_MARK = "[?]"  # та же пометка, что в промпте саммари (#12)


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def srt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def vtt_time(seconds: float) -> str:
    return srt_time(seconds).replace(",", ".")


def _timed_words(seg: Segment) -> list[tuple[float, float, str]]:
    """Слова сегмента с таймкодами. Без пословных таймкодов (заглушка ASR, старый кэш) время
    делится пропорционально длине слов — грубо, но реплика не уедет за пределы сегмента."""
    if seg.words:
        return [(w.start, w.end, w.text.strip()) for w in seg.words if w.text.strip()]
    tokens = seg.text.split()
    total = sum(len(t) for t in tokens) or 1
    out, t = [], seg.start
    for token in tokens:
        end = t + seg.duration * len(token) / total
        out.append((t, end, token))
        t = end
    return out


def wrap(text: str) -> str:
    """Длинную реплику — в две строки, разрез у пробела ближе к середине."""
    if len(text) <= LINE_CHARS:
        return text
    spaces = [i for i, c in enumerate(text) if c == " "]
    if not spaces:
        return text
    cut = min(spaces, key=lambda i: abs(i - len(text) / 2))
    return f"{text[:cut]}\n{text[cut + 1 :]}"


def _segment_cues(seg: Segment) -> list[Cue]:
    limit = LINE_CHARS * MAX_LINES - (len(SUSPECT_MARK) + 1 if seg.suspect else 0)
    cues: list[Cue] = []
    group: list[tuple[float, float, str]] = []

    def flush() -> None:
        if group:
            text = " ".join(w[2] for w in group)
            cues.append(Cue(group[0][0], group[-1][1], text))
            group.clear()

    for word in _timed_words(seg):
        if group:
            text = " ".join(w[2] for w in group)
            too_long = len(text) + 1 + len(word[2]) > limit
            too_slow = word[1] - group[0][0] > MAX_SECONDS
            phrase_done = text.endswith(_SENTENCE_END) and len(text) >= limit / 2
            if too_long or too_slow or phrase_done:
                flush()
        group.append(word)
    flush()
    if seg.suspect:
        cues = [Cue(c.start, c.end, f"{SUSPECT_MARK} {c.text}") for c in cues]
    return cues


def cues(transcript: Transcript) -> list[Cue]:
    raw = [c for seg in transcript.segments for c in _segment_cues(seg)]
    out: list[Cue] = []
    for i, c in enumerate(raw):
        end = max(c.end, c.start + MIN_SECONDS)
        if i + 1 < len(raw) and end > raw[i + 1].start:
            # Продлили до минимума и наехали на следующую — отступаем, но не короче звучания.
            end = max(raw[i + 1].start, c.end)
        if end <= c.start:
            end = c.start + MIN_SECONDS / 2  # нулевая длительность — плеер реплику не покажет
        out.append(Cue(c.start, end, wrap(c.text)))
    return out


def to_srt(transcript: Transcript) -> str:
    return "\n".join(
        f"{i}\n{srt_time(c.start)} --> {srt_time(c.end)}\n{c.text}\n"
        for i, c in enumerate(cues(transcript), 1)
    )


def to_vtt(transcript: Transcript) -> str:
    body = "\n".join(
        f"{vtt_time(c.start)} --> {vtt_time(c.end)}\n{c.text}\n" for c in cues(transcript)
    )
    return f"WEBVTT\n\n{body}"
