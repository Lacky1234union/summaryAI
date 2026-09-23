"""Проверки конфига: секции читают свои переменные, кривое окружение падает на старте."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from summuraze.config import ASRSettings, LLMSettings, PathsSettings, Settings, get_settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Тесты не должны зависеть от .env разработчика и от его же переменных окружения.

    Без этого прогон на машине с заполненным .env проверяет чужую конфигурацию, а не код.
    """
    for key in list(__import__("os").environ):
        if key.startswith("SUMMURAZE_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()


class TestPaths:
    def test_derived_paths_live_under_data_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SUMMURAZE_DATA_DIR", "/tmp/summuraze-test")
        p = PathsSettings()
        assert p.artifacts_dir == Path("/tmp/summuraze-test/artifacts")
        assert p.db_path == Path("/tmp/summuraze-test/summuraze.sqlite")


class TestASR:
    def test_reads_prefixed_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SUMMURAZE_ASR_MODEL", "medium")
        monkeypatch.setenv("SUMMURAZE_ASR_THREADS", "8")
        a = ASRSettings()
        assert a.model == "medium"
        assert a.threads == 8

    def test_blank_language_means_auto(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`SUMMURAZE_ASR_LANGUAGE=` в .env — это «определяй сам», а не язык с пустым именем."""
        monkeypatch.setenv("SUMMURAZE_ASR_LANGUAGE", "   ")
        assert ASRSettings().language is None

    def test_unknown_device_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SUMMURAZE_ASR_DEVICE", "vulkan")
        with pytest.raises(ValidationError):
            ASRSettings()

    def test_negative_threads_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SUMMURAZE_ASR_THREADS", "-4")
        with pytest.raises(ValidationError):
            ASRSettings()


class TestLLM:
    def test_api_key_is_not_printed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ключ не должен утекать в логи и трейсбеки через repr настроек."""
        monkeypatch.setenv("SUMMURAZE_LLM_API_KEY", "sk-очень-секретный")
        s = LLMSettings()
        assert "очень-секретный" not in repr(s)
        assert s.api_key.get_secret_value() == "sk-очень-секретный"

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("http://127.0.0.1:8080/v1", True),
            ("http://localhost:11434/v1", True),
            ("https://api.deepseek.com/v1", False),
            ("https://openrouter.ai/api/v1", False),
        ],
    )
    def test_is_local_detection(
        self, monkeypatch: pytest.MonkeyPatch, url: str, expected: bool
    ) -> None:
        monkeypatch.setenv("SUMMURAZE_LLM_BASE_URL", url)
        assert LLMSettings().is_local is expected

    def test_zero_context_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Нулевой контекст сломал бы расчёт порога single-pass против иерархии (#16)."""
        monkeypatch.setenv("SUMMURAZE_LLM_CONTEXT", "0")
        with pytest.raises(ValidationError):
            LLMSettings()


class TestLocalOnly:
    def test_local_only_with_remote_llm_fails_on_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Противоречие должно всплыть при старте, а не после часа транскрибации."""
        monkeypatch.setenv("SUMMURAZE_LOCAL_ONLY", "true")
        monkeypatch.setenv("SUMMURAZE_LLM_BASE_URL", "https://api.deepseek.com/v1")
        with pytest.raises(ValidationError, match="смотрит наружу"):
            Settings()

    def test_local_only_with_local_llm_is_fine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SUMMURAZE_LOCAL_ONLY", "true")
        monkeypatch.setenv("SUMMURAZE_LLM_BASE_URL", "http://127.0.0.1:8080/v1")
        assert Settings().local_only is True

    def test_remote_llm_without_local_only_is_fine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert Settings().local_only is False


class TestErrorMessages:
    def test_broken_value_names_the_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Сообщение должно называть поле, иначе искать причину придётся по всему .env."""
        monkeypatch.setenv("SUMMURAZE_LLM_CONTEXT", "не-число")
        with pytest.raises(ValidationError) as exc:
            LLMSettings()
        assert "context" in str(exc.value)


class TestSingleton:
    def test_settings_are_cached(self) -> None:
        assert get_settings() is get_settings()

    def test_cache_clear_rereads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        first = get_settings().asr.model
        monkeypatch.setenv("SUMMURAZE_ASR_MODEL", "tiny")
        get_settings.cache_clear()
        assert get_settings().asr.model == "tiny" != first


def test_dotenv_is_not_loaded_at_import() -> None:
    """`.env` читает сам pydantic-settings через env_file.

    Вызов load_dotenv() на импорте создаёт ловушку порядка: он обязан успеть до создания
    настроек, иначе часть переменных окажется пустой, а ошибка проявится не там, где сделана.
    """
    source = (Path(__file__).parents[2] / "summuraze" / "config.py").read_text(encoding="utf-8")
    assert "load_dotenv()" not in source.replace("`load_dotenv()`", "")
