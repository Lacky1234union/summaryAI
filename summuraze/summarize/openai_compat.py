"""Единый LLM-адаптер для OpenAI-совместимых серверов: OpenRouter (дефолт), DeepSeek, llama.cpp.

Переключение API ↔ локально — значения `base_url` и `model` в конфиге, а не ветка в коде
(guardrail #2). Протокол простой — один POST `/chat/completions`, поэтому без SDK: `httpx`
уже стоит, а поля кэша у DeepSeek свои (`prompt_cache_hit_tokens`), SDK их всё равно не знает.

**Порядок сообщений — ради кэша префикса** (CLAUDE.md, «Кэш префикса»): у DeepSeek кэш-хит
дешевле промаха в 50 раз, у llama.cpp свой prompt cache — оба работают по общему началу запроса.
Поэтому первым идёт то, что не меняется между прогонами одной записи:

1. `system` — `BASE_SYSTEM`, одинаковый для всех профилей и записей;
2. `user` — транскрипт байт в байт, затем промпт профиля и вопрос.

Промпт профиля нельзя ставить в `system`: он стоит перед транскриптом, и смена профиля обнулила
бы кэш на всём транскрипте. Работает ли кэш — видно по `cached_input` в `summuraze log`.

OpenRouter сам считает, сколько списал за запрос (`usage.cost`), — эта цифра точнее таблицы
`pricing.py` и идёт в лог как есть. Прокси (`proxy_url`) — для хостов в РФ: OpenRouter их режет.

Refs #15
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

from summuraze.summarize.pricing import cost_usd
from summuraze.types import LLMResult, TokenUsage

if TYPE_CHECKING:
    from summuraze.config import LLMSettings

BASE_SYSTEM = (
    "Ты работаешь с транскриптами записей лекций, встреч и созвонов. Транскрипт размечен "
    "таймкодами [ЧЧ:ММ:СС]. Отвечай по-русски, опирайся только на транскрипт."
)
OPENROUTER_HOST = "openrouter.ai"
TRANSCRIPT_HEADER = "Транскрипт записи:\n\n"
INSTRUCTIONS_HEADER = "\n\n---\n\n"

# Ретраи — только на то, что проходит само: перегрузка и сбои сервера. 401/402/400 повтором не
# лечатся, на них — сразу понятная ошибка.
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
RETRIES = 3
BACKOFF_SECONDS = 5.0
# Двухчасовая лекция — 40k токенов на вход и несколько тысяч на выход: ответ идёт минутами.
TIMEOUT = httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=15.0)


class LLMError(RuntimeError):
    """Провайдер не ответил по делу — сообщение для человека."""


class OpenAICompatLLM:
    name = "openai-compat"

    def __init__(self, cfg: LLMSettings, *, client: httpx.Client | None = None) -> None:
        self.model = cfg.model
        self.context_window = cfg.context
        self._base_url = cfg.base_url.rstrip("/")
        self._local = cfg.is_local
        self._openrouter = cfg.host == OPENROUTER_HOST
        self._quantizations = cfg.openrouter_quantizations
        self._reasoning = cfg.openrouter_reasoning
        headers = {"Content-Type": "application/json"}
        key = cfg.api_key.get_secret_value()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if self._openrouter:
            headers["X-Title"] = "Summuraze"
        proxy = cfg.proxy_url.get_secret_value() if cfg.proxy_url else None
        self._client = client or httpx.Client(timeout=TIMEOUT, proxy=proxy)
        self._headers = headers

    def close(self) -> None:
        self._client.close()

    def count_tokens(self, text: str) -> int | None:
        """Точный счёт — только у локального сервера: у llama.cpp есть `POST /tokenize`.
        У OpenRouter и DeepSeek такого эндпоинта нет — там бюджет считает оценкой (#16)."""
        if not self._local:
            return None
        root = self._base_url.removesuffix("/v1")
        try:
            response = self._client.post(
                f"{root}/tokenize", json={"content": text}, headers=self._headers
            )
            tokens = response.json().get("tokens") if response.status_code == 200 else None
        except (httpx.HTTPError, ValueError, AttributeError):
            tokens = None
        if not isinstance(tokens, list):
            logger.info("{} не считает токены (/tokenize) — бюджет по оценке", self._base_url)
            return None
        return len(tokens)

    def complete(
        self,
        *,
        system: str,
        prefix: str,
        question: str,
        max_tokens: int,
        schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": BASE_SYSTEM},
                {"role": "user", "content": _user_message(prefix, system, question)},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "stream": False,
        }
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "document", "strict": True, "schema": schema},
            }
        if self._openrouter:
            body["usage"] = {"include": True}  # вернуть списанную сумму в ответе
            routing: dict[str, Any] = {}
            if schema is not None:
                # Не отдавать запрос провайдеру, который схему молча проигнорирует.
                routing["require_parameters"] = True
            if self._quantizations:
                routing["quantizations"] = self._quantizations
            if routing:
                body["provider"] = routing
            if self._reasoning is not None:
                body["reasoning"] = {"enabled": self._reasoning}
        at = datetime.now(UTC)
        data = self._post(body)
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"{self._base_url} вернул ответ без текста: {str(data)[:300]}") from e
        truncated = choice.get("finish_reason") == "length"
        if truncated:
            logger.warning(
                "ответ модели обрезан лимитом в {} токенов — конец документа может отсутствовать",
                max_tokens,
            )
        raw_usage = data.get("usage") or {}
        usage = _usage(raw_usage)
        billed = raw_usage.get("cost")
        return LLMResult(
            text=text.strip(),
            model=data.get("model") or self.model,
            usage=usage,
            cost_usd=float(billed)
            if billed is not None
            else cost_usd(self._base_url, self.model, usage, at, local=self._local),
            truncated=truncated,
        )

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/chat/completions"
        for attempt in range(1, RETRIES + 1):
            try:
                response = self._client.post(url, json=body, headers=self._headers)
            except httpx.TimeoutException as e:
                error: str = f"нет ответа от {self._base_url} за отведённое время"
                if attempt == RETRIES:
                    raise LLMError(error) from e
            except httpx.TransportError as e:
                raise LLMError(f"не удалось связаться с {self._base_url}: {e}") from e
            else:
                if response.status_code == 200:
                    result: dict[str, Any] = response.json()
                    return result
                error = _explain(response)
                if response.status_code not in RETRY_STATUS or attempt == RETRIES:
                    raise LLMError(error)
            logger.warning("{} — повтор {}/{}", error, attempt, RETRIES - 1)
            time.sleep(BACKOFF_SECONDS * attempt)
        raise AssertionError("unreachable")  # pragma: no cover


def _user_message(prefix: str, system: str, question: str) -> str:
    """Транскрипт первым и байт в байт; всё, что зависит от профиля, — после него."""
    return f"{TRANSCRIPT_HEADER}{prefix}{INSTRUCTIONS_HEADER}{system}\n\n{question}"


def _usage(raw: dict[str, Any]) -> TokenUsage:
    # DeepSeek: prompt_cache_hit_tokens; OpenAI и llama.cpp: prompt_tokens_details.cached_tokens.
    cached = raw.get("prompt_cache_hit_tokens")
    if cached is None:
        cached = (raw.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return TokenUsage(
        input=raw.get("prompt_tokens", 0),
        output=raw.get("completion_tokens", 0),
        cached_input=cached or 0,
    )


def _explain(response: httpx.Response) -> str:
    try:
        detail = response.json().get("error", {}).get("message") or response.text
    except ValueError:
        detail = response.text
    detail = str(detail)[:300]
    hints = {
        401: "ключ не подходит — проверь SUMMURAZE_LLM_API_KEY",
        402: "на счёте провайдера кончились деньги",
        404: "нет такого адреса или модели — проверь SUMMURAZE_LLM_BASE_URL и SUMMURAZE_LLM_MODEL",
        429: "провайдер просит притормозить (лимит запросов)",
        403: "доступ запрещён — OpenRouter режет IP из РФ, задай SUMMURAZE_LLM_PROXY_URL",
    }
    hint = hints.get(response.status_code, "ошибка провайдера")
    return f"{response.status_code}: {hint} ({detail})"
