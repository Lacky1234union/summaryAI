"""Общие фикстуры: ни один тест не пишет в ./data репозитория и не читает .env разработчика."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from summuraze.config import ASRSettings, get_settings


@pytest.fixture(autouse=True)
def _isolated_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for key in list(os.environ):
        if key.startswith("SUMMURAZE_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SUMMURAZE_DATA_DIR", str(tmp_path / "data"))
    # Тесты не качают и не грузят Whisper: настоящая модель — только в тестах с маркой `model`.
    monkeypatch.setenv("SUMMURAZE_ASR_PROVIDER", "echo")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


_SYNTHETIC = Path(__file__).parent / "fixtures" / "synthetic"
# Две разные фразы: одинаковый клип дважды Whisper склеивает в одну (принимает за повтор).
SPEECH_CLIPS = {"speech": _SYNTHETIC / "speech_ru.opus", "speech2": _SYNTHETIC / "speech2_ru.opus"}

# Кусок записи для `compose`: ("speech",) / ("speech2",) — фраза; ("silence", сек);
# ("tone", сек) — синус, «музыка», на которой Whisper галлюцинирует не хуже, чем на тишине.
Part = tuple[str] | tuple[str, float]
Composer = Callable[[Path, list[Part]], list[tuple[float, float]]]


@pytest.fixture(scope="session")
def compose() -> Composer:
    """Склеить запись из речи, тишины и тона; вернуть, где на самом деле речь (секунды).

    Речь — синтетика espeak-ng (`tests/fixtures/synthetic`): известное содержимое на известной
    секунде — ровно то, что нужно проверить VAD и смещение таймкодов.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("нужен ffmpeg")
    clips = {
        kind: float(
            subprocess.run(
                [
                    *("ffprobe", "-v", "error", "-show_entries", "format=duration"),
                    *("-of", "csv=p=0", str(path)),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        for kind, path in SPEECH_CLIPS.items()
    }

    def build(path: Path, parts: list[Part]) -> list[tuple[float, float]]:
        inputs: list[str] = []
        spans: list[tuple[float, float]] = []
        t = 0.0
        for part in parts:
            kind = part[0]
            if kind in clips:
                inputs += ["-i", str(SPEECH_CLIPS[kind])]
                spans.append((t, t + clips[kind]))
                t += clips[kind]
                continue
            seconds = part[1]  # type: ignore[misc]
            src = "anullsrc=r=16000:cl=mono" if kind == "silence" else "sine=f=440:r=16000"
            inputs += ["-f", "lavfi", "-t", str(seconds), "-i", src]
            t += seconds
        n = len(parts)
        streams = "".join(
            f"[{i}:a]aresample=16000,aformat=channel_layouts=mono[a{i}];" for i in range(n)
        )
        concat = "".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1"
        subprocess.run(
            [
                *("ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"),
                *inputs,
                *("-filter_complex", streams + concat),
                *("-c:a", "aac", str(path)),
            ],
            check=True,
        )
        return spans

    return build


@pytest.fixture
def cached_whisper() -> ASRSettings:
    """Настройки настоящего Whisper для тестов `slow`; без модели в кэше HuggingFace — skip."""
    from faster_whisper.utils import download_model

    cfg = ASRSettings(provider="faster-whisper")
    try:
        download_model(cfg.model, local_files_only=True)
    except Exception:
        pytest.skip(f"модели {cfg.model} нет в кэше HuggingFace")
    return cfg
