"""Бюджет токенов: весь транскрипт одним запросом или окнами (#16).

Single-pass — основной режим (ADR «Single-pass как основной режим»): у DeepSeek V4.1 Flash
контекст 1M, а двухчасовая лекция — ~40 тыс. токенов. Но порог — свойство провайдера, а не
константа: у локальной 8B-модели с контекстом 8–32 тыс. окна понадобятся на обычной лекции.

Как считаем токены:
- локальный сервер (llama.cpp) — его же токенизатором, `POST /tokenize`: там контекст мал и
  ошибка оценки в 10 % решает, влезет ли лекция;
- API — оценкой по символам с коэффициентом, замеренным на реальном токенизаторе провайдера
  (`CHARS_PER_TOKEN`). Эндпоинта подсчёта у OpenRouter и DeepSeek нет, а при контексте в 1M
  ошибка в 10 % ничего не решает. Расхождение оценки с `usage.prompt_tokens` видно в логе.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from loguru import logger

from summuraze.types import LLMProvider

# Замер 2026-10-28, DeepSeek V4.1 Flash через OpenRouter, русская лекция: транскрипт с
# таймкодами — 3.19 символа на токен, сплошной текст — 4.29 (1.55 токена на слово; английский
# ~1.3). Таймкод `[ЧЧ:ММ:СС] ` — около 8 токенов на строку. Берём 3.0: оценка выходит с запасом.
CHARS_PER_TOKEN = 3.0
# Меньше — окно вмещает пару минут речи, и карта из таких окон теряет смысл: конспекта
# на таком контексте не сделать, честнее отказать до ASR.
MIN_CONTEXT = 4096
# Роли, служебные токены шаблона чата, заголовки сообщения.
TEMPLATE_TOKENS = 64
# На ответ — не больше четверти контекста: у малого контекста max_tokens профиля (16k)
# съел бы весь бюджет, и на транскрипт не осталось бы ничего.
OUTPUT_SHARE = 4

Mode = Literal["single", "windows"]


class TranscriptTooLong(RuntimeError):
    """Не влезает даже окнами: окно меньше промпта или карта всей записи больше порога."""


class ContextTooSmall(ValueError):
    """Контекст модели слишком мал для конспекта — выясняем до ASR."""


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def output_reserve(context: int, max_tokens: int) -> int:
    return min(max_tokens, context // OUTPUT_SHARE)


def input_limit(context: int, max_tokens: int, single_pass_max: int | None = None) -> int:
    """Сколько токенов запроса уходит одним проходом: контекст минус ответ и обвязка, но не
    больше `single_pass_max` — на длинном контексте качество падает раньше, чем кончается окно."""
    limit = context - output_reserve(context, max_tokens) - TEMPLATE_TOKENS
    return min(limit, single_pass_max) if single_pass_max else limit


Final = Literal["transcript", "map", "merge"]
_FINAL = {
    "transcript": "итог — по карте и полному транскрипту",
    "map": "итог — по карте: транскрипт целиком не влезает",
    "merge": "блоки окон склеены",
}


@dataclass(frozen=True)
class Budget:
    mode: Mode
    prompt: int
    limit: int
    context: int
    output: int
    exact: bool
    # Заполняет двухпроходная схема (#21), когда запись прошла окнами.
    windows: int = 0
    final: Final | None = None

    def describe(self) -> str:
        how = "токенизатор сервера" if self.exact else "оценка по символам"
        sizes = (
            f"{human_tokens(self.prompt)} из {human_tokens(self.limit)} токенов "
            f"({how}; контекст {human_tokens(self.context)}, на ответ {human_tokens(self.output)})"
        )
        if self.mode == "single":
            return f"один запрос: {sizes}"
        if not self.windows:
            return f"в один запрос не влезает: {sizes}"
        tail = f"; {_FINAL[self.final]}" if self.final else ""
        return f"окнами — {self.windows} шт.: {sizes}{tail}"


def plan(
    llm: LLMProvider,
    parts: Iterable[str],
    *,
    max_tokens: int,
    single_pass_max: int | None = None,
    what: str = "саммари",
) -> Budget:
    """`parts` — всё, что уйдёт в запрос: транскрипт, промпт профиля, вопрос."""
    text = "\n".join(parts)
    counted = llm.count_tokens(text)
    prompt = (counted if counted is not None else estimate_tokens(text)) + TEMPLATE_TOKENS
    context = llm.context_window
    limit = input_limit(context, max_tokens, single_pass_max)
    budget = Budget(
        mode="single" if prompt <= limit else "windows",
        prompt=prompt,
        limit=limit,
        context=context,
        output=output_reserve(context, max_tokens),
        exact=counted is not None,
    )
    logger.info("{} {}: {}", what, llm.model, budget.describe())
    return budget


def check_context(context: int) -> None:
    if context < MIN_CONTEXT:
        raise ContextTooSmall(
            f"контекст модели {human_tokens(context)} токенов — для конспекта нужно хотя бы "
            f"{human_tokens(MIN_CONTEXT)}: задай SUMMURAZE_LLM_CONTEXT по модели"
        )


def too_long(model: str, detail: str) -> str:
    return (
        f"запись не обработать даже окнами на {model} ({detail}). Нужна модель с большим "
        "контекстом (SUMMURAZE_LLM_CONTEXT) или `summuraze transcribe`: транскрипт без саммари"
    )


def human_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.3g}M"
    if n >= 1000:
        return f"{n / 1000:.1f}k".replace(".0k", "k")
    return str(n)
