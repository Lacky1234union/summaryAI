"""Выгрузка модели: память возвращается системе и на не-glibc ничего не падает."""

from __future__ import annotations

import ctypes.util

import pytest

from summuraze.asr.faster_whisper import physical_cores
from summuraze.pipeline import memory


def test_release_memory_runs() -> None:
    memory.release_memory()


def test_release_memory_without_libc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: None)
    memory.release_memory()


def test_release_memory_without_malloc_trim(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_libc(name: str) -> object:
        raise OSError("нет такой libc")

    monkeypatch.setattr(memory.ctypes, "CDLL", no_libc)
    memory.release_memory()


def test_physical_cores_is_positive() -> None:
    assert physical_cores() >= 1


def test_physical_cores_without_sysfs(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from summuraze.asr import faster_whisper

    monkeypatch.setattr(faster_whisper, "_CPU_SYS", tmp_path)
    monkeypatch.setattr(faster_whisper.os, "cpu_count", lambda: 3)
    assert physical_cores() == 3
