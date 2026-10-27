"""LLM-адаптер: раскладка запроса под кэш префикса, разбор ответа, цена, ошибки.

Сеть подменена `httpx.MockTransport` — ключ и интернет тестам не нужны.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from summuraze.config import LLMSettings, Settings
from summuraze.summarize import openai_compat
from summuraze.summarize.base import LLMNotConfigured, check_llm, make_llm
from summuraze.summarize.echo import EchoLLM
from summuraze.summarize.openai_compat import (
    BASE_SYSTEM,
    TRANSCRIPT_HEADER,
    LLMError,
    OpenAICompatLLM,
)
from summuraze.summarize.pricing import cost_usd, deepseek_peak
from summuraze.types import LLMProvider, TokenUsage

PREFIX = "[00:00:01] Сегодня о преобразовании Фурье.\n[00:00:05] Теорема Котельникова."

DEEPSEEK_REPLY = {
    "model": "deepseek-flash",
    "choices": [{"message": {"content": "  Конспект [00:00:01].  "}, "finish_reason": "stop"}],
    "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "prompt_cache_hit_tokens": 900,
        "prompt_cache_miss_tokens": 100,
    },
}

Handler = Callable[[httpx.Request], httpx.Response]


DEEPSEEK = "https://api.deepseek.com/v1"


def _llm(handler: Handler, **cfg: Any) -> OpenAICompatLLM:
    cfg = {"base_url": DEEPSEEK, "model": "deepseek-flash", **cfg}
    settings = LLMSettings(provider="openai-compat", api_key=SecretStr("sk-test"), **cfg)
    return OpenAICompatLLM(settings, client=httpx.Client(transport=httpx.MockTransport(handler)))


def _complete(llm: OpenAICompatLLM, system: str = "Профиль лекции.") -> Any:
    return llm.complete(system=system, prefix=PREFIX, question="Составь конспект.", max_tokens=500)


class Recorder:
    def __init__(self, *responses: httpx.Response) -> None:
        self.requests: list[httpx.Request] = []
        self._responses = list(responses)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]

    def body(self, i: int = 0) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(self.requests[i].content)
        return result


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openai_compat, "BACKOFF_SECONDS", 0)


class TestRequest:
    def test_transcript_first_profile_after(self) -> None:
        rec = Recorder(httpx.Response(200, json=DEEPSEEK_REPLY))
        _complete(_llm(rec))
        req = rec.requests[0]
        assert str(req.url) == "https://api.deepseek.com/v1/chat/completions"
        assert req.headers["Authorization"] == "Bearer sk-test"
        body = rec.body()
        assert body["model"] == "deepseek-flash" and body["max_tokens"] == 500
        system, user = body["messages"]
        assert system == {"role": "system", "content": BASE_SYSTEM}
        assert user["content"].startswith(TRANSCRIPT_HEADER + PREFIX)
        assert user["content"].endswith("Профиль лекции.\n\nСоставь конспект.")

    def test_profile_change_keeps_cacheable_prefix(self) -> None:
        """Смена профиля не должна менять ни байта до конца транскрипта (CLAUDE.md, кэш)."""
        rec = Recorder(httpx.Response(200, json=DEEPSEEK_REPLY))
        llm = _llm(rec)
        _complete(llm, system="Профиль лекции.")
        _complete(llm, system="Сырой профиль.")
        a, b = (json.dumps(rec.body(i)["messages"], ensure_ascii=False) for i in (0, 1))
        cut = a.index(PREFIX.replace("\n", "\\n")) + len(PREFIX.replace("\n", "\\n"))
        assert a[:cut] == b[:cut]

    def test_local_server_without_key(self) -> None:
        rec = Recorder(httpx.Response(200, json=DEEPSEEK_REPLY))
        settings = LLMSettings(
            provider="openai-compat", base_url="http://127.0.0.1:8080/v1/", model="qwen"
        )
        llm = OpenAICompatLLM(settings, client=httpx.Client(transport=httpx.MockTransport(rec)))
        result = _complete(llm)
        assert str(rec.requests[0].url) == "http://127.0.0.1:8080/v1/chat/completions"
        assert "Authorization" not in rec.requests[0].headers
        assert result.cost_usd == 0.0, "свой сервер — честный ноль"


class TestResponse:
    def test_deepseek_usage_and_text(self) -> None:
        result = _complete(_llm(Recorder(httpx.Response(200, json=DEEPSEEK_REPLY))))
        assert result.text == "Конспект [00:00:01]."
        assert result.model == "deepseek-flash"
        assert result.usage == TokenUsage(input=1000, output=200, cached_input=900)
        assert result.cost_usd is not None and result.cost_usd > 0

    def test_openai_style_cached_tokens(self) -> None:
        reply = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 40},
            },
        }
        result = _complete(_llm(Recorder(httpx.Response(200, json=reply)), model="other"))
        assert result.usage.cached_input == 40
        assert result.model == "other", "сервер не назвал модель — берём из конфига"

    def test_unknown_model_cost_is_unknown(self) -> None:
        reply = {"choices": [{"message": {"content": "ok"}}]}
        result = _complete(_llm(Recorder(httpx.Response(200, json=reply)), model="mystery"))
        assert result.cost_usd is None and result.usage == TokenUsage(input=0, output=0)

    def test_truncated_answer_warns(self) -> None:
        from loguru import logger

        reply = {
            **DEEPSEEK_REPLY,
            "choices": [{"message": {"content": "…"}, "finish_reason": "length"}],
        }
        warnings: list[str] = []
        sink = logger.add(warnings.append, level="WARNING", format="{message}")
        try:
            _complete(_llm(Recorder(httpx.Response(200, json=reply))))
        finally:
            logger.remove(sink)
        assert warnings and "обрезан" in warnings[0]

    def test_answer_without_text(self) -> None:
        with pytest.raises(LLMError, match="без текста"):
            _complete(_llm(Recorder(httpx.Response(200, json={"choices": []}))))


class TestErrors:
    def test_bad_key_is_not_retried(self) -> None:
        rec = Recorder(httpx.Response(401, json={"error": {"message": "Authentication Fails"}}))
        with pytest.raises(LLMError, match="401: ключ не подходит.*Authentication Fails"):
            _complete(_llm(rec))
        assert len(rec.requests) == 1

    def test_overload_is_retried(self) -> None:
        rec = Recorder(httpx.Response(503, text="busy"), httpx.Response(200, json=DEEPSEEK_REPLY))
        assert _complete(_llm(rec)).text
        assert len(rec.requests) == 2

    def test_gives_up_after_retries(self) -> None:
        rec = Recorder(httpx.Response(429, text="slow down"))
        with pytest.raises(LLMError, match="429: провайдер просит притормозить"):
            _complete(_llm(rec))
        assert len(rec.requests) == openai_compat.RETRIES

    def test_timeout_retried_then_fails(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            raise httpx.ReadTimeout("slow", request=request)

        with pytest.raises(LLMError, match="нет ответа"):
            _complete(_llm(handler))
        assert len(calls) == openai_compat.RETRIES

    def test_network_down(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route", request=request)

        with pytest.raises(LLMError, match="не удалось связаться"):
            _complete(_llm(handler))

    def test_non_json_error_body(self) -> None:
        with pytest.raises(LLMError, match="500: ошибка провайдера \\(<html>"):
            _complete(_llm(Recorder(httpx.Response(500, text="<html>oops</html>"))))


class TestPricing:
    @pytest.mark.parametrize(
        ("when", "peak"),
        [
            (datetime(2026, 10, 27, 2, 30, tzinfo=UTC), True),  # вторник, 02:30
            (datetime(2026, 10, 27, 9, 59, tzinfo=UTC), True),
            (datetime(2026, 10, 27, 4, 0, tzinfo=UTC), False),
            (datetime(2026, 10, 27, 12, 0, tzinfo=UTC), False),
            (datetime(2026, 10, 31, 2, 30, tzinfo=UTC), False),  # суббота
        ],
    )
    def test_peak_hours(self, when: datetime, peak: bool) -> None:
        assert deepseek_peak(when) is peak

    def test_peak_costs_double(self) -> None:
        usage = TokenUsage(input=1_000_000, output=1_000_000, cached_input=0)
        url = "https://api.deepseek.com/v1"
        peak = cost_usd(
            url, "deepseek-flash", usage, datetime(2026, 10, 27, 2, tzinfo=UTC), local=False
        )
        off = cost_usd(
            url, "deepseek-flash", usage, datetime(2026, 10, 27, 12, tzinfo=UTC), local=False
        )
        assert peak == pytest.approx(0.3 + 1.2)
        assert off == pytest.approx((0.3 + 1.2) / 2)

    def test_cache_hit_is_fifty_times_cheaper(self) -> None:
        url, at = "https://api.deepseek.com/v1", datetime(2026, 10, 27, 12, tzinfo=UTC)
        hit = cost_usd(
            url,
            "deepseek-flash",
            TokenUsage(input=10**6, output=0, cached_input=10**6),
            at,
            local=False,
        )
        miss = cost_usd(url, "deepseek-flash", TokenUsage(input=10**6, output=0), at, local=False)
        assert hit is not None and miss is not None and miss / hit == pytest.approx(50)

    def test_other_host_unknown(self) -> None:
        usage = TokenUsage(input=1, output=1)
        at = datetime(2026, 10, 27, tzinfo=UTC)
        assert (
            cost_usd("https://openrouter.ai/api/v1", "deepseek-flash", usage, at, local=False)
            is None
        )


class TestFactory:
    def test_echo(self) -> None:
        assert isinstance(make_llm(LLMSettings(provider="echo")), EchoLLM)

    def test_openai_compat(self) -> None:
        llm = make_llm(LLMSettings(provider="openai-compat", api_key=SecretStr("sk")))
        assert isinstance(llm, LLMProvider) and llm.model == "deepseek/deepseek-v4.1-flash"

    def test_remote_without_key_refused(self) -> None:
        with pytest.raises(LLMNotConfigured, match="SUMMURAZE_LLM_API_KEY"):
            make_llm(LLMSettings(provider="openai-compat"))

    def test_local_without_key_allowed(self) -> None:
        check_llm(LLMSettings(provider="openai-compat", base_url="http://localhost:8080/v1"))


def test_missing_key_fails_before_asr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Без ключа — отказ до транскрибации: иначе час Whisper впустую."""
    from summuraze.pipeline.orchestrator import Pipeline

    rec = tmp_path / "rec.mp3"
    rec.write_bytes(b"\0")
    monkeypatch.setenv("SUMMURAZE_LLM_PROVIDER", "openai-compat")
    created: list[str] = []

    def asr_factory(cfg: object) -> Any:
        created.append("asr")
        raise AssertionError("ASR не должен был создаваться")

    with Pipeline(Settings(), asr_factory=asr_factory) as p, pytest.raises(LLMNotConfigured):
        p.run(rec)
    assert created == []


def test_cli_explains_missing_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from summuraze.cli import app

    rec = tmp_path / "rec.mp3"
    rec.write_bytes(b"\0")
    monkeypatch.setenv("SUMMURAZE_LLM_PROVIDER", "openai-compat")
    result = CliRunner().invoke(app, ["run", str(rec)])
    assert result.exit_code == 1 and "SUMMURAZE_LLM_API_KEY" in result.output


class TestOpenRouter:
    REPLY = {
        "model": "deepseek/deepseek-v4.1-flash",
        "choices": [{"message": {"content": "Конспект."}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 800},
            "cost": 0.000123,
        },
    }

    def _llm(self, handler: Handler) -> OpenAICompatLLM:
        settings = LLMSettings(api_key=SecretStr("sk-or"))  # дефолты — OpenRouter
        return OpenAICompatLLM(
            settings, client=httpx.Client(transport=httpx.MockTransport(handler))
        )

    def test_defaults_and_billed_cost(self) -> None:
        rec = Recorder(httpx.Response(200, json=self.REPLY))
        result = _complete(self._llm(rec))
        req = rec.requests[0]
        assert str(req.url) == "https://openrouter.ai/api/v1/chat/completions"
        assert req.headers["X-Title"] == "Summuraze"
        body = rec.body()
        assert body["model"] == "deepseek/deepseek-v4.1-flash"
        assert body["usage"] == {"include": True}
        assert result.cost_usd == 0.000123, "списанное OpenRouter точнее таблицы"
        assert result.usage.cached_input == 800

    def test_region_block_explained(self) -> None:
        rec = Recorder(httpx.Response(403, json={"error": {"message": "Access denied"}}))
        with pytest.raises(LLMError, match="403: .*SUMMURAZE_LLM_PROXY_URL"):
            _complete(self._llm(rec))


class TestEnv:
    """Имена переменных — как в meditron / vobot / tgit: ключ и прокси не дублируются."""

    def test_openrouter_key_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-shared")
        assert LLMSettings().api_key.get_secret_value() == "sk-or-shared"
        check_llm(LLMSettings())

    def test_own_key_name_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SUMMURAZE_LLM_API_KEY", "sk-own")
        assert LLMSettings().api_key.get_secret_value() == "sk-own"

    @pytest.mark.parametrize(
        "name", ["SUMMURAZE_LLM_PROXY_URL", "LLM_PROXY_URL", "OPENROUTER_PROXY_URL"]
    )
    def test_proxy_names(self, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
        monkeypatch.setenv(name, "socks5://user:pass@1.2.3.4:1080")
        proxy = LLMSettings().proxy_url
        assert proxy is not None and proxy.get_secret_value().startswith("socks5://")
        OpenAICompatLLM(LLMSettings(api_key=SecretStr("k")))  # httpx[socks] на месте

    def test_blank_proxy_is_direct(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SUMMURAZE_LLM_PROXY_URL", " ")
        assert LLMSettings().proxy_url is None

    def test_proxy_not_leaked_in_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROXY_URL", "http://user:secret@1.2.3.4:8080")
        assert "secret" not in repr(LLMSettings())
