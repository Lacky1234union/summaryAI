"""Общие фикстуры: ни один тест не пишет в ./data репозитория и не читает .env разработчика."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from summuraze.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for key in list(os.environ):
        if key.startswith("SUMMURAZE_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SUMMURAZE_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
