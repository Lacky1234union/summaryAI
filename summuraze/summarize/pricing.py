"""Цена запроса к LLM — для `summuraze log`, а не для биллинга.

Цены — с https://api-docs.deepseek.com/quick_start/pricing, сверено 2026-10-27 (guardrail #8:
ID и цены сверять в момент реализации). У DeepSeek цена зависит от времени: в часы пик
(01:00–04:00 и 06:00–10:00 UTC, пн–пт) вдвое дороже. Китайские праздники не учитываем —
в праздник оценка выйдет завышенной, не заниженной.

Неизвестная модель — цена `None` («—» в логе), а не ноль: ноль выглядел бы как «бесплатно».
Локальный сервер — честный ноль.

Refs #15
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from summuraze.types import TokenUsage


@dataclass(frozen=True)
class Price:
    """USD за 1M токенов в часы пик."""

    input_hit: float
    input_miss: float
    output: float


DEEPSEEK_HOST = "api.deepseek.com"
DEEPSEEK: dict[str, Price] = {
    "deepseek-flash": Price(input_hit=0.006, input_miss=0.3, output=1.2),
    "deepseek-v4-pro": Price(input_hit=0.044, input_miss=1.32, output=3.96),
}
_PEAK_HOURS_UTC = (range(1, 4), range(6, 10))


def deepseek_peak(at_utc: datetime) -> bool:
    return at_utc.weekday() < 5 and any(at_utc.hour in hours for hours in _PEAK_HOURS_UTC)


def cost_usd(
    base_url: str, model: str, usage: TokenUsage, at_utc: datetime, *, local: bool
) -> float | None:
    if local:
        return 0.0
    price = DEEPSEEK.get(model) if DEEPSEEK_HOST in base_url else None
    if price is None:
        return None
    factor = 1.0 if deepseek_peak(at_utc) else 0.5
    miss = usage.input - usage.cached_input
    total = (
        usage.cached_input * price.input_hit + miss * price.input_miss + usage.output * price.output
    )
    return round(total * factor / 1_000_000, 6)
