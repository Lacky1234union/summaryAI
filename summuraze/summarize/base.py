"""Фабрика LLM-провайдеров по конфигу.

Здесь же со временем живёт запрет сети под `--local-only` (#20): фабрика — единственное место,
через которое конвейер получает провайдера, значит и единственное, где запрет нельзя обойти.

Refs #7, #15
"""

from __future__ import annotations

from summuraze.config import LLMSettings
from summuraze.types import LLMProvider


class LLMNotConfigured(ValueError):
    """Саммари не получится — выясняем до транскрибации, а не после часа работы Whisper."""


def check_llm(settings: LLMSettings) -> None:
    remote = settings.provider == "openai-compat" and not settings.is_local
    if remote and not settings.api_key.get_secret_value():
        raise LLMNotConfigured(
            f"для {settings.base_url} нужен ключ: задай OPENROUTER_API_KEY "
            "(или SUMMURAZE_LLM_API_KEY) в .env. "
            "Без ключа — локальный сервер (SUMMURAZE_LLM_BASE_URL=http://127.0.0.1:8080/v1) "
            "или заглушка SUMMURAZE_LLM_PROVIDER=echo"
        )


def make_llm(settings: LLMSettings) -> LLMProvider:
    if settings.provider == "echo":
        from summuraze.summarize.echo import EchoLLM

        return EchoLLM()
    check_llm(settings)
    from summuraze.summarize.openai_compat import OpenAICompatLLM

    return OpenAICompatLLM(settings)
