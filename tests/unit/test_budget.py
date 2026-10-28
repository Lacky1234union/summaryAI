"""Бюджет токенов: один запрос или окна (#16)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from summuraze.cli import app
from summuraze.config import LLMSettings, get_settings
from summuraze.pipeline import stages
from summuraze.summarize import budget
from summuraze.summarize.budget import (
    TEMPLATE_TOKENS,
    TranscriptTooLong,
    check_speech,
    estimate_tokens,
    human_tokens,
    input_limit,
    output_reserve,
)
from summuraze.summarize.echo import EchoLLM
from summuraze.summarize.openai_compat import OpenAICompatLLM
from summuraze.summarize.profiles import get_profile
from summuraze.types import LLMResult, Segment, Transcript
from tests.conftest import SPEECH_CLIPS

HOUR = 3600.0


def _transcript(seconds: float) -> Transcript:
    """Лекция на `seconds`: строка каждые 5 с — как у Whisper на сплошной речи."""
    line = "сегодня разберём, как сборщик мусора находит недостижимые объекты"
    return Transcript(
        source=Path("/x/лекция.mp4"),
        duration=seconds,
        language="ru",
        asr_model="m",
        segments=[Segment(start=t, end=t + 4.9, text=line) for t in range(0, int(seconds), 5)],
    )


class Counted(EchoLLM):
    """Провайдер с токенизатором — как llama.cpp с /tokenize."""

    def __init__(self, context: int, per_char: float = 0.25) -> None:
        super().__init__(context)
        self.per_char = per_char
        self.sent: list[dict[str, Any]] = []

    def count_tokens(self, text: str) -> int | None:
        return int(len(text) * self.per_char)

    def complete(self, **kw: Any) -> LLMResult:
        self.sent.append(kw)
        return super().complete(**kw)


class TestArithmetic:
    def test_estimate_is_conservative_for_measured_ratio(self) -> None:
        text = "я" * 3190  # замер: 3.19 символа на токен → 1000 токенов
        assert 1000 < estimate_tokens(text) <= 1100

    def test_output_takes_at_most_quarter_of_context(self) -> None:
        assert output_reserve(1_000_000, 16_000) == 16_000
        assert output_reserve(8192, 16_000) == 2048

    def test_input_limit(self) -> None:
        assert input_limit(8192, 16_000) == 8192 - 2048 - TEMPLATE_TOKENS
        assert input_limit(1_000_000, 16_000, single_pass_max=200_000) == 200_000
        assert input_limit(8192, 16_000, single_pass_max=200_000) == 8192 - 2048 - TEMPLATE_TOKENS

    @pytest.mark.parametrize(
        ("n", "text"), [(950, "950"), (1000, "1k"), (23_456, "23.5k"), (1_000_000, "1M")]
    )
    def test_human_tokens(self, n: int, text: str) -> None:
        assert human_tokens(n) == text


class TestPlan:
    def test_same_transcript_different_providers_different_modes(self) -> None:
        """Приёмка #16: один транскрипт, контекст 1M и 8K — разные режимы, и это видно."""
        prefix = stages.transcript_prefix(_transcript(HOUR))
        big = budget.plan(EchoLLM(1_000_000), [prefix], max_tokens=16_000)
        small = budget.plan(EchoLLM(8192), [prefix], max_tokens=16_000)
        assert big.mode == "single" and small.mode == "windows"
        assert big.prompt == small.prompt
        assert 12_000 < big.prompt < 30_000, "час лекции — порядка 20 тыс. токенов"
        assert big.describe().startswith("один запрос: ")
        assert "оценка по символам" in big.describe() and "контекст 1M" in big.describe()
        assert small.describe().startswith("в один запрос не влезает: ")

    def test_server_tokenizer_wins_over_estimate(self) -> None:
        plan = budget.plan(Counted(8192, per_char=0.1), ["я" * 10_000], max_tokens=16_000)
        assert plan.exact and plan.prompt == 1000 + TEMPLATE_TOKENS
        assert "токенизатор сервера" in plan.describe()

    def test_single_pass_max_caps_long_context(self) -> None:
        prefix = stages.transcript_prefix(_transcript(HOUR))
        plan = budget.plan(EchoLLM(), [prefix], max_tokens=16_000, single_pass_max=5000)
        assert plan.mode == "windows" and plan.limit == 5000


class TestStage:
    def test_fits_one_request_with_clipped_answer(self, tmp_path: Path) -> None:
        llm = Counted(32_768)
        plan = stages.build_summary(tmp_path, llm, _transcript(600), get_profile("lecture"))
        assert plan.mode == "single"
        [call] = llm.sent
        assert call["max_tokens"] == 8192, "ответ — четверть контекста, а не 16k профиля"

    def test_too_long_fails_before_sending(self, tmp_path: Path) -> None:
        llm = Counted(8192)
        with pytest.raises(TranscriptTooLong, match="#21") as err:
            stages.build_summary(tmp_path, llm, _transcript(HOUR), get_profile("lecture"))
        assert "в один запрос не влезает" in str(err.value) and "токенизатор сервера" in str(
            err.value
        )
        assert llm.sent == [] and not any(tmp_path.iterdir())


class TestSpeechCheck:
    def test_two_hours_on_small_context_fail_early(self) -> None:
        with pytest.raises(TranscriptTooLong, match="120 мин речи"):
            check_speech(2 * HOUR, model="qwen-8b", context=8192, max_tokens=16_000)

    @pytest.mark.parametrize(("seconds", "context"), [(2 * HOUR, 1_000_000), (600, 8192)])
    def test_passes_when_it_may_fit(self, seconds: float, context: int) -> None:
        check_speech(seconds, model="m", context=context, max_tokens=16_000)


class TestTokenize:
    def _llm(self, base_url: str, handler: Any) -> OpenAICompatLLM:
        cfg = LLMSettings(provider="openai-compat", base_url=base_url, api_key=SecretStr(""))
        return OpenAICompatLLM(cfg, client=httpx.Client(transport=httpx.MockTransport(handler)))

    def test_llama_cpp_tokenize(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"tokens": [1, 2, 3, 4]})

        llm = self._llm("http://127.0.0.1:8080/v1", handler)
        assert llm.count_tokens("привет мир") == 4
        assert str(seen[0].url) == "http://127.0.0.1:8080/tokenize"
        assert json.loads(seen[0].content) == {"content": "привет мир"}

    @pytest.mark.parametrize(
        "response",
        [httpx.Response(404, text="not found"), httpx.Response(200, text="<html>")],
    )
    def test_local_server_without_tokenize_falls_back(self, response: httpx.Response) -> None:
        llm = self._llm("http://localhost:11434/v1", lambda r: response)
        assert llm.count_tokens("текст") is None

    def test_local_server_down_falls_back(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("отказ", request=request)

        assert self._llm("http://127.0.0.1:8080/v1", handler).count_tokens("текст") is None

    def test_remote_api_is_not_asked(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("у API нет /tokenize — и запроса быть не должно")

        llm = self._llm("https://openrouter.ai/api/v1", handler)
        assert llm.count_tokens("текст") is None


@pytest.fixture
def lecture(tmp_path: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("нужен ffmpeg")
    path = tmp_path / "лекция.opus"
    shutil.copy(SPEECH_CLIPS["speech"], path)
    return path


class TestCLI:
    def test_decision_is_printed(self, lecture: Path) -> None:
        out = CliRunner().invoke(app, ["run", str(lecture)])
        assert out.exit_code == 0, out.output
        assert "Саммари: один запрос: " in out.output
        again = CliRunner().invoke(app, ["run", str(lecture)])
        assert "Саммари:" not in again.output, "из кэша — запроса не было"

    def test_small_context_refuses_after_asr(
        self, lecture: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SUMMURAZE_LLM_CONTEXT", "800")
        get_settings.cache_clear()
        out = CliRunner().invoke(app, ["run", str(lecture)])
        assert out.exit_code == 1
        assert "в один запрос не влезает" in out.output and "summuraze transcribe" in out.output

    def test_tiny_context_refuses_before_asr(
        self, lecture: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SUMMURAZE_LLM_CONTEXT", "100")
        get_settings.cache_clear()
        out = CliRunner().invoke(app, ["run", str(lecture)])
        assert out.exit_code == 1 and "мин речи" in out.output
        assert "распознавание" not in out.output, "час Whisper ради отказа не тратим"
