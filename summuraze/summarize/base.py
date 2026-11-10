"""Фабрика LLM-провайдеров по конфигу.

Здесь же запрет сети под `--local-only` (#20): фабрика — единственное место, через которое
конвейер получает провайдера, значит и единственное, где запрет нельзя обойти.

Refs #7, #15, #20
"""

from __future__ import annotations

from summuraze.config import LLMSettings, local_only_conflict
from summuraze.types import LLMProvider


class LLMNotConfigured(ValueError):
    """Саммари не получится — выясняем до транскрибации, а не после часа работы Whisper."""


class NetworkForbidden(LLMNotConfigured):
    """Под `--local-only` провайдер с внешним адресом не создаётся вовсе."""


def check_llm(settings: LLMSettings) -> None:
    remote = settings.provider == "openai-compat" and not settings.is_local
    if remote and settings.local_only:
        raise NetworkForbidden(local_only_conflict(settings))
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

        return EchoLLM(settings.context)
    check_llm(settings)  # и при прямом вызове фабрики, мимо preflight
    if settings.server is not None:
        from summuraze.summarize.llama_server import LocalLLM

        return LocalLLM(settings)
    from summuraze.summarize.openai_compat import OpenAICompatLLM

    return OpenAICompatLLM(settings)
