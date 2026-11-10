"""Кэш артефактов стадий на диске.

Без кэша любая правка промпта саммари стоит десятков минут повторной транскрибации — итерации
по качеству просто не случатся (ADR «Кэш артефактов по стадиям»).

Ключ = sha256(стадия + версия стадии + ключи входов + значимые параметры). Ключ следующей стадии
строится из ключа предыдущей, а не из байтов её выхода: цепочка как в git, и изменение версии
любой стадии выше по течению инвалидирует всё, что ниже. Меняешь логику стадии — меняй версию.

Раскладка: `<root>/<стадия>/<ab>/<ключ>/` — каталог, потому что стадия может отдавать несколько
файлов (wav + метаданные, транскрипт в json/txt/srt). Каталог считается готовым, только если
в нём есть `manifest.json`, а появляется он атомарным переименованием временного каталога:
прерванная на сороковой минуте транскрибация не оставит полуфабрикат, который кэш примет за
готовый результат.

Refs #6
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST = "manifest.json"
_TMP_MARK = ".tmp-"
_STAGE_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
# Временный каталог старше суток — от убитого процесса: живая стадия столько не идёт.
_STALE_TMP_SECONDS = 24 * 3600


@dataclass(frozen=True)
class ArtifactKey:
    stage: str
    version: int
    inputs: tuple[str, ...]
    params: Mapping[str, Any] = field(default_factory=dict)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not _STAGE_RE.match(self.stage):
            raise ValueError(f"имя стадии должно быть [a-z0-9_-]: {self.stage!r}")
        # Без default=str: Path или datetime в параметрах сделали бы ключ зависимым от того,
        # где лежит файл и когда запущено, и кэш перестал бы попадать. Пусть падает сразу.
        payload = json.dumps(
            {
                "stage": self.stage,
                "version": self.version,
                "inputs": list(self.inputs),
                "params": dict(self.params),
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        object.__setattr__(self, "digest", hashlib.sha256(payload.encode()).hexdigest())


@dataclass(frozen=True)
class GCResult:
    removed: int
    freed_bytes: int


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, key: ArtifactKey) -> Path:
        return self.located(key.stage, key.digest)

    def located(self, stage: str, digest: str) -> Path:
        """Каталог артефакта по ключу из истории прогонов, без самого `ArtifactKey`."""
        return self.root / stage / digest[:2] / digest

    def get(self, key: ArtifactKey) -> Path | None:
        """Каталог готового артефакта или `None`.

        Попадание обновляет mtime манифеста: `gc` удаляет давно не использованное, а не давно
        созданное — транскрипт лекции, к которой возвращаешься, не должен пропасть.
        """
        final = self.path(key)
        manifest = final / MANIFEST
        if not manifest.is_file():
            return None
        os.utime(manifest)
        return final

    @contextmanager
    def put(self, key: ArtifactKey) -> Generator[Path, None, None]:
        """Временный каталог для записи артефакта; при успехе он становится готовым.

        Исключение внутри блока удаляет временный каталог — в кэш ничего не попадает.
        """
        final = self.path(key)
        tmp = final.with_name(f"{final.name}{_TMP_MARK}{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            yield tmp
            (tmp / MANIFEST).write_text(
                json.dumps(
                    {
                        "stage": key.stage,
                        "version": key.version,
                        "inputs": list(key.inputs),
                        "params": dict(key.params),
                        "created": datetime.now(UTC).isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            if final.exists():
                shutil.rmtree(final)
            tmp.rename(final)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def get_or_build(self, key: ArtifactKey, build: Callable[[Path], None]) -> tuple[Path, bool]:
        """Готовый артефакт из кэша или собранный `build(каталог)`. Второе значение — попадание."""
        hit = self.get(key)
        if hit is not None:
            return hit, True
        with self.put(key) as tmp:
            build(tmp)
        return self.path(key), False

    def gc(self, max_age_days: float, *, dry_run: bool = False) -> GCResult:
        """Удалить артефакты, не использованные `max_age_days` дней, и брошенные временные."""
        if not self.root.is_dir():
            return GCResult(0, 0)
        now = time.time()
        removed = freed = 0
        for stage_dir in self.root.iterdir():
            for bucket in stage_dir.glob("??"):
                for entry in bucket.iterdir():
                    if _TMP_MARK in entry.name:
                        expired = now - entry.stat().st_mtime > _STALE_TMP_SECONDS
                    else:
                        manifest = entry / MANIFEST
                        # Каталог без манифеста не может появиться штатно — мусор.
                        expired = (
                            not manifest.is_file()
                            or now - manifest.stat().st_mtime > max_age_days * 86400
                        )
                    if not expired:
                        continue
                    removed += 1
                    freed += tree_size(entry)
                    if not dry_run:
                        shutil.rmtree(entry, ignore_errors=True)
                if not dry_run and not any(bucket.iterdir()):
                    bucket.rmdir()
        return GCResult(removed, freed)


def tree_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
