"""Запись «на пару» (#67): когда остановиться, если запустили на занятии.

Сетка пар — конфиг (`SUMMURAZE_RECORD_PAIRS`), а не константа: у другого кампуса или вуза она
своя. Длительность занятия одна на всю сетку — в ВШЭ любое занятие длится 80 минут.

Стоп — по концу пары из сетки, даже если запись начата посреди неё (опоздал, вспомнил на
десятой минуте): «старт + 80» тогда записал бы десять минут чужой перемены. Вне сетки (доп.
занятие, сдвинутое расписание) — «старт + длительность + запас»: запись до упора хуже.

Refs #67
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta

# Пришёл заранее: пара, которая начнётся не позже чем через столько, считается текущей.
EARLY = timedelta(minutes=30)


@dataclass(frozen=True)
class StopPlan:
    stop: datetime
    # Пара из сетки, по которой посчитан стоп; None — вне сетки или `--until`.
    pair: tuple[datetime, datetime] | None
    reason: str


def parse_hhmm(text: str) -> time:
    try:
        hours, minutes = text.strip().split(":")
        return time(int(hours), int(minutes))
    except ValueError:
        raise ValueError(f"время {text!r} — не ЧЧ:ММ") from None


def plan_stop(now: datetime, starts: list[time], *, minutes: float, margin: float) -> StopPlan:
    length, extra = timedelta(minutes=minutes), timedelta(minutes=margin)
    pairs = sorted(
        (start, start + length) for start in (datetime.combine(now.date(), s) for s in starts)
    )
    for start, end in pairs:
        # Перемена между парами короче EARLY: текущая пара ещё идёт — она и раньше в списке.
        if start - EARLY <= now < end:
            return StopPlan(
                end + extra,
                (start, end),
                f"пара {start:%H:%M}–{end:%H:%M}, стоп в {end + extra:%H:%M} "
                f"(запас {margin:g} мин)",
            )
    upcoming = next((start for start, _ in pairs if start > now), None)
    hint = (
        f"ближайшая пара в {upcoming:%H:%M}" if upcoming is not None else "пары сегодня кончились"
    )
    stop = now + length + extra
    return StopPlan(
        stop,
        None,
        f"сейчас не пара ({hint}) — стоп через {minutes + margin:g} мин, в {stop:%H:%M}",
    )


def until(now: datetime, text: str) -> StopPlan:
    """Явное время остановки. Уже прошедшее сегодня — значит, завтра (запись за полночь)."""
    stop = datetime.combine(now.date(), parse_hhmm(text))
    if stop <= now:
        stop += timedelta(days=1)
    return StopPlan(stop, None, f"стоп в {stop:%H:%M}")
