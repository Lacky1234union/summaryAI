"""Профили документа: что спросить у модели и в какой структуре.

Этап 0 — заглушки, чтобы стадия саммари имела вход. Настоящие промпты «лекция» и «сырой» — #17.
`version` профиля входит в ключ кэша саммари: правка промпта перегенерирует саммари, но не ASR.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    version: int
    system: str
    question: str


PROFILES: dict[str, Profile] = {
    "lecture": Profile(
        name="lecture",
        version=0,
        system="Ты составляешь конспект лекции по транскрипту с таймкодами.",
        question="Составь конспект лекции. Каждое утверждение — с таймкодом.",
    ),
}


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise ValueError(f"неизвестный профиль {name!r}; есть: {known}") from None
