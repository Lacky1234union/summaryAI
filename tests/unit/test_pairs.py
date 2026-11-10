"""Запись «на пару» (#67): когда остановиться."""

from __future__ import annotations

from datetime import datetime, time

import pytest
from pydantic import ValidationError

from summuraze.audio.pairs import parse_hhmm, plan_stop, until
from summuraze.config import RecordSettings

HSE = [parse_hhmm(t) for t in RecordSettings().pairs]


def at(hhmm: str, day: int = 2) -> datetime:
    h, m = hhmm.split(":")
    return datetime(2026, 11, day, int(h), int(m))


def plan(now: str) -> tuple[str, tuple[str, str] | None, str]:
    p = plan_stop(at(now), HSE, minutes=80, margin=10)
    pair = (f"{p.pair[0]:%H:%M}", f"{p.pair[1]:%H:%M}") if p.pair else None
    return f"{p.stop:%H:%M}", pair, p.reason


@pytest.mark.parametrize(
    ("now", "stop", "pair"),
    [
        # 2026-11-02: дискретка 13:00–14:20, запись начата в 13:10 — стоп по концу пары, не 14:40.
        ("13:10", "14:30", ("13:00", "14:20")),
        ("13:00", "14:30", ("13:00", "14:20")),
        ("12:45", "14:30", ("13:00", "14:20")),  # пришёл заранее
        ("12:35", "14:30", ("13:00", "14:20")),  # перемена после 11:10–12:30
        ("14:19", "14:30", ("13:00", "14:20")),
        ("14:20", "16:10", ("14:40", "16:00")),  # пара кончилась — следующая через 20 мин
        ("09:00", "11:00", ("09:30", "10:50")),
    ],
)
def test_stop_by_pair(now: str, stop: str, pair: tuple[str, str]) -> None:
    got_stop, got_pair, reason = plan(now)
    assert (got_stop, got_pair) == (stop, pair)
    assert f"пара {pair[0]}–{pair[1]}" in reason and "запас 10 мин" in reason


def test_outside_grid_falls_back_to_duration() -> None:
    stop, pair, reason = plan("07:00")
    assert (stop, pair) == ("08:30", None)
    assert "ближайшая пара в 09:30" in reason and "через 90 мин" in reason


def test_after_last_pair() -> None:
    stop, pair, reason = plan("21:30")
    assert (stop, pair) == ("23:00", None) and "пары сегодня кончились" in reason


def test_margin_and_length_from_config() -> None:
    p = plan_stop(at("13:10"), HSE, minutes=90, margin=5)
    assert f"{p.stop:%H:%M}" == "14:35"


def test_until() -> None:
    assert until(at("13:10"), "14:25").stop == at("14:25")
    assert until(at("23:50"), "00:30").stop == at("00:30", day=3), "за полночь — завтра"


@pytest.mark.parametrize("bad", ["14", "25:00", "14:6x", ""])
def test_bad_time(bad: str) -> None:
    with pytest.raises(ValueError, match="ЧЧ:ММ"):
        parse_hhmm(bad)


def test_config_rejects_bad_grid() -> None:
    with pytest.raises(ValidationError, match="ЧЧ:ММ"):
        RecordSettings(pairs=["09:30", "11-10"])
    assert parse_hhmm("9:05") == time(9, 5)
