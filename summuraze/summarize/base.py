"""Фабрика LLM-провайдеров по конфигу.

Здесь же со временем живёт запрет сети под `--local-only` (#20): фабрика — единственное место,
через которое конвейер получает провайдера, значит и единственное, где запрет нельзя обойти.

Refs #7
"""

from __future__ import annotations

from summuraze.config import LLMSettings
from summuraze.types import LLMProvider


def make_llm(settings: LLMSettings) -> LLMProvider:
    if settings.provider == "echo":
        from summuraze.summarize.echo import EchoLLM

        return EchoLLM()
    raise NotImplementedError(
        f"LLM-провайдер {settings.provider!r} ещё не реализован (#15). "
        "Пока доступен только SUMMURAZE_LLM_PROVIDER=echo."
    )
