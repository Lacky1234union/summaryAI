"""Типизированный конфиг: единая точка валидации окружения.

Смысл — уронить процесс на старте с понятным сообщением, а не выстрелить `None` на сороковой
минуте транскрибации. Поэтому `os.getenv` по коду вразброс не заводим (guardrail #3): всё, что
приходит из окружения, объявлено здесь и проверено при создании `Settings`.

Секции соответствуют переменным из `.env.example` один к одному: у каждой свой префикс, так что
`SUMMURAZE_ASR_MODEL` попадает в `settings.asr.model` без ручного маппинга.

`.env` читается самим pydantic-settings через `env_file`, а не вызовом `load_dotenv()` на импорте
модуля. Так исчезает ловушка порядка: при `load_dotenv()` важно, чтобы он успел до создания
настроек, иначе часть переменных окажется пустой — и ошибка проявится не там, где сделана.
`env_file` снимает вопрос совсем, а не описывает его в комментарии.

Refs #3
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Хосты, которые считаем локальными при проверке --local-only.
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "0.0.0.0", "::1", "host.docker.internal")


class PathsSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SUMMURAZE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_dir: Path = Path("./data")

    @property
    def artifacts_dir(self) -> Path:
        """Артефакты стадий. Ключ — хеш входа и версии стадии, поэтому правка промпта
        не перезапускает ASR (ADR «Кэш артефактов по стадиям»)."""
        return self.data_dir / "artifacts"

    @property
    def db_path(self) -> Path:
        """SQLite: состояние задач, история прогонов, стоимость."""
        return self.data_dir / "summuraze.sqlite"


class ASRSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SUMMURAZE_ASR_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Дефолт — предположение до замера. Настоящий выбор делает бенчмарк на этом железе (#9)
    # по правилу «лучшее качество при RTF <= 0.5».
    model: str = "large-v3-turbo"
    compute_type: str = "int8"
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    # Пусто = автоопределение языка.
    language: str | None = "ru"
    # 0 = по числу ядер. На мобильном 15-ваттном CPU больше потоков не значит быстрее.
    threads: int = Field(default=0, ge=0)

    @field_validator("language", mode="before")
    @classmethod
    def _blank_language_means_auto(cls, v: object) -> object:
        """`SUMMURAZE_ASR_LANGUAGE=` в .env — это «определяй сам», а не язык с пустым именем."""
        if isinstance(v, str) and not v.strip():
            return None
        return v


class LLMSettings(BaseSettings):
    """Провайдер саммаризации.

    Переключение локально <-> API живёт ЗДЕСЬ, а не в коде конвейера: DeepSeek, OpenRouter,
    Yandex и локальный llama.cpp говорят по одному OpenAI-совместимому протоколу, поэтому
    разница между режимами — значения `base_url` и `model` (guardrail #2).
    """

    model_config = SettingsConfigDict(
        env_prefix="SUMMURAZE_LLM_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    base_url: str = "https://api.deepseek.com/v1"
    # Алиас deepseek-chat retired 24.07.2026 — не использовать.
    model: str = "deepseek-flash"
    api_key: SecretStr = SecretStr("")
    # Контекст провайдера: от него считается порог single-pass против иерархии (#16).
    # У deepseek-flash — 1M, у локальной 8B-модели на два порядка меньше.
    context: int = Field(default=1_000_000, gt=0)

    @property
    def is_local(self) -> bool:
        return any(host in self.base_url for host in _LOCAL_HOSTS)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SUMMURAZE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    paths: PathsSettings = Field(default_factory=PathsSettings)
    asr: ASRSettings = Field(default_factory=ASRSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    # Локальный STT сам по себе приватности не даёт: саммари по API увозит транскрипт целиком.
    # Полный запрет сети на уровне фабрики провайдеров — #20; здесь ловим противоречие раньше,
    # чтобы оно не всплыло после часа транскрибации.
    local_only: bool = False

    @model_validator(mode="after")
    def _local_only_forbids_remote_llm(self) -> Settings:
        if self.local_only and not self.llm.is_local:
            raise ValueError(
                f"SUMMURAZE_LOCAL_ONLY=true, но SUMMURAZE_LLM_BASE_URL смотрит наружу: "
                f"{self.llm.base_url}. Либо укажи локальный сервер "
                f"(http://127.0.0.1:8080/v1), либо выключи local_only."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Настройки процесса. Кэшируются: окружение за время прогона не меняется.

    В тестах сбрасывается через `get_settings.cache_clear()`.
    """
    return Settings()
