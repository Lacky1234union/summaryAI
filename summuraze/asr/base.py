"""Фабрика ASR-провайдеров по конфигу.

Конвейер знает только `ASRProvider`; какой адаптер за ним — решает конфиг (guardrail #1).
Импорт адаптера — внутри фабрики: `faster_whisper` с его рантаймом грузится, только когда
транскрибация действительно нужна, а не на каждый `summuraze --help` и не при попадании в кэш.

Refs #7, #11
"""

from __future__ import annotations

from summuraze.config import ASRSettings
from summuraze.types import ASRProvider


def make_asr(settings: ASRSettings) -> ASRProvider:
    if settings.provider == "echo":
        from summuraze.asr.echo import EchoASR

        return EchoASR()
    from summuraze.asr.faster_whisper import FasterWhisperASR

    return FasterWhisperASR(settings)
