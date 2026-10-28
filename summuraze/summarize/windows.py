"""Окна с перекрытием и двухпроходная схема для записей, которые не влезают в один запрос (#21).

Свёртка по кускам теряет то, ради чего конспект и нужен: вопрос в начале и ответ в конце,
определение, на которое опираются через час, передуманное решение. Поэтому не map-reduce
«пересказ кусков → пересказ пересказов», а два прохода:

1. каждое окно → **карта**: темы по порядку и утверждения с таймкодами, плюс «открытые нити» —
   то, что ждёт продолжения (вопрос без ответа, «вернёмся к этому», введённое обозначение);
2. итоговый документ профиля — по склеенной карте всей записи **и полному транскрипту**, если
   они вместе влезают; если нет — по карте. Модель видит структуру целиком, прежде чем писать.

Окна режутся по границам сегментов (никогда посреди фразы Whisper), с предпочтением конца
предложения, а без точек — самой длинной паузы. Перекрытие ~10 % даёт окну контекст начала, но
утверждение принадлежит одному окну — тому, в чьё «ядро» попал его таймкод: так перекрытие не
даёт дублей, а таймкоды между окнами не противоречат друг другу.

`raw` второго прохода не требует: его блоки — разметка, а не пересказ, и склеиваются как есть.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from summuraze.render.markdown import fmt_timecode
from summuraze.summarize.profiles import AT, Claim, RawDoc
from summuraze.types import Segment

# Меняешь нарезку, промпты карты или склейку — поднимай: версия входит в ключ кэша саммари.
VERSION = 1
OVERLAP = 0.1
# Где искать разрез: в последних 15 % окна. Раньше — окна мельчают, позже — некуда отступить.
CUT_ZONE = 0.15
_SENTENCE_END = (".", "!", "?", "…")


class Topic(BaseModel):
    title: str = Field(description="Тема фрагмента, несколько слов")
    at: str = Field(description="Таймкод начала темы, ЧЧ:ММ:СС — ровно как в транскрипте")
    points: list[Claim] = Field(
        description="Всё существенное по теме: тезисы, определения, формулы, примеры, вопросы "
        "и ответы, советы почитать — каждое своим утверждением со своим таймкодом"
    )


class Thread(BaseModel):
    text: str = Field(
        description="Что осталось открытым: вопрос без ответа, обещание вернуться к теме, "
        "введённое обозначение или допущение, на которое дальше будут опираться"
    )
    at: str = Field(description=AT)


class WindowMap(BaseModel):
    topics: list[Topic] = Field(description="Темы фрагмента по порядку")
    threads: list[Thread] = Field(description="Открытые нити; пусто, если их нет")


MAP_QUESTION = "Составь карту фрагмента по схеме ответа."


def map_system(index: int, total: int, start: float, end: float, suspect_rule: str) -> str:
    return (
        f"Перед тобой фрагмент {index} из {total} длинной записи лекции "
        f"({fmt_timecode(start)}–{fmt_timecode(end)}). Конспект всей лекции потом пишут по "
        "картам всех фрагментов, поэтому карта должна сохранить всё существенное: темы по "
        "порядку, а в них тезисы, определения, формулы, разобранные примеры, прозвучавшие "
        "вопросы и ответы, советы, что почитать.\n\n"
        "Правила:\n"
        "- Каждое утверждение — с таймкодом строки транскрипта, где это сказано.\n"
        "- Только то, что есть в транскрипте фрагмента, без своих знаний.\n"
        "- Начало фрагмента может повторять конец предыдущего — размечай его как обычно.\n"
        "- threads — то, что во фрагменте начато, но не закончено: по ним на втором проходе "
        "связывают начало лекции с её концом.\n\n" + suspect_rule
    )


FINAL_WITH_TRANSCRIPT = (
    "Запись длинная, поэтому её сначала разметили по фрагментам. Карта ниже — оглавление и "
    "опорные утверждения всей лекции с таймкодами. Используй её, чтобы видеть структуру целиком "
    "и связывать начало с концом (открытые нити), а текст и таймкоды сверяй с транскриптом."
)
FINAL_FROM_MAP = (
    "Запись слишком длинная, чтобы прислать транскрипт целиком: вместо него — карта лекции, "
    "составленная по фрагментам. Утверждения карты взяты из транскрипта, таймкоды у них верные — "
    "переноси их как есть. Связывай начало с концом по открытым нитям: если вопрос из начала "
    "получил ответ позже или мысль была пересмотрена, в конспекте это должно быть одним местом."
)


@dataclass(frozen=True)
class Window:
    """Сегменты окна вместе с перекрытием; `core_*` — отрезок, утверждения из которого его."""

    segments: list[Segment]
    core_start: float
    core_end: float

    @property
    def start(self) -> float:
        return self.segments[0].start

    @property
    def end(self) -> float:
        return self.segments[-1].end

    def owns(self, t: float) -> bool:
        return self.core_start <= t < self.core_end


def split(segments: Sequence[Segment], tokens: Sequence[int], limit: int) -> list[Window]:
    """Окна не больше `limit` токенов; `tokens[i]` — токены строки `segments[i]` в запросе."""
    if len(segments) != len(tokens):
        raise ValueError("на каждый сегмент — своё число токенов")
    n = len(segments)
    bounds: list[tuple[int, int, int]] = []  # (начало с перекрытием, начало ядра, конец)
    first, core = 0, 0
    while core < n:
        end, used = first, 0
        while end < n and (used + tokens[end] <= limit or end == first):
            used += tokens[end]
            end += 1
        if end < n:
            end = _cut(segments, tokens, first, end, limit)
        end = max(end, core + 1)  # окно обязано сдвинуться: иначе одна строка зациклит нарезку
        bounds.append((first, core, end))
        if end >= n:
            break
        core = end
        first = _overlap_start(segments, tokens, core, first + 1, limit)
    windows = []
    for i, (lo, c, hi) in enumerate(bounds):
        core_start = 0.0 if i == 0 else segments[c].start
        core_end = segments[bounds[i + 1][1]].start if i + 1 < len(bounds) else math.inf
        windows.append(Window(list(segments[lo:hi]), core_start, core_end))
    return windows


def _cut(
    segments: Sequence[Segment], tokens: Sequence[int], first: int, end: int, limit: int
) -> int:
    """Разрез в зоне конца окна: после конца предложения, а без точек — на самой длинной паузе."""
    zone, used, lo = limit * CUT_ZONE, 0, end
    while lo - 1 > first and used + tokens[lo - 1] <= zone:
        used += tokens[lo - 1]
        lo -= 1
    candidates = range(max(lo, first + 1), end + 1)
    for k in reversed(candidates):
        if segments[k - 1].text.strip().endswith(_SENTENCE_END):
            return k
    return max(candidates, key=lambda k: _gap(segments, k))


def _overlap_start(
    segments: Sequence[Segment], tokens: Sequence[int], core: int, floor: int, limit: int
) -> int:
    """Начало перекрытия: ~10 % окна назад от ядра, с начала предложения — ради него можно
    отступить до 20 %. Не раньше `floor`, иначе следующее окно не сдвинется."""
    marks: list[int] = []  # границы, отступая от ядра: сколько токенов набралось до каждой
    start, used = core, 0
    while start - 1 >= floor and used + tokens[start - 1] <= limit * OVERLAP * 2:
        used += tokens[start - 1]
        start -= 1
        marks.append(used)
    if not marks:
        return core
    candidates = range(core - len(marks), core)
    sentence = [
        k for k in candidates if k == 0 or segments[k - 1].text.strip().endswith(_SENTENCE_END)
    ]
    near = [k for k in candidates if marks[core - k - 1] <= limit * OVERLAP]
    if sentence:
        # Ближайшее к 10 % начало предложения: внутри — самое раннее, за ним — самое позднее.
        inside = [k for k in sentence if k in near]
        return min(inside) if inside else max(sentence)
    return min(near) if near else core - 1


def _gap(segments: Sequence[Segment], k: int) -> float:
    return segments[k].start - segments[k - 1].end if 0 < k < len(segments) else 0.0


def merge_maps(
    maps: Sequence[tuple[Window, WindowMap]], resolve: Callable[[str], float | None]
) -> WindowMap:
    """Склеить карты окон: каждое утверждение — из окна, в чьё ядро попал его таймкод.

    Таймкод, который не читается, остаётся у своего окна: выбросить его — потерять мысль, а
    привязку всё равно проверяет рендер (guardrail #5)."""

    def mine(window: Window, at: str) -> bool:
        t = resolve(at)
        return t is None or window.owns(t)

    topics: list[Topic] = []
    threads: list[Thread] = []
    for window, wmap in maps:
        for topic in wmap.topics:
            points = [p for p in topic.points if mine(window, p.at)]
            if not points and not mine(window, topic.at):
                continue
            if topics and topics[-1].title.casefold() == topic.title.casefold():
                topics[-1].points.extend(points)  # тема продолжилась через границу окна
            else:
                topics.append(topic.model_copy(update={"points": points}))
        threads.extend(t for t in wmap.threads if mine(window, t.at))
    return WindowMap(topics=topics, threads=threads)


def merge_blocks(
    docs: Sequence[tuple[Window, RawDoc]], resolve: Callable[[str], float | None]
) -> RawDoc:
    """`raw` по окнам: блоки склеиваются по ядрам, как утверждения карты."""
    blocks = []
    for window, doc in docs:
        for block in doc.blocks:
            t = resolve(block.at)
            if t is None or window.owns(t):
                blocks.append(block)
    return RawDoc(blocks=blocks)


def map_text(wmap: WindowMap) -> str:
    lines = ["Карта лекции:"]
    for topic in wmap.topics:
        lines.append(f"\n## [{_clean(topic.at)}] {topic.title}")
        lines.extend(f"- [{_clean(p.at)}] {p.text}" for p in topic.points)
    if wmap.threads:
        lines.append("\n## Открытые нити")
        lines.extend(f"- [{_clean(t.at)}] {t.text}" for t in wmap.threads)
    return "\n".join(lines)


def _clean(at: str) -> str:
    return at.strip().strip("[]")
