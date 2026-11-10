"""Словарь терминов: подсказка Whisper и карта замен после распознавания.

Файл — обычный текст или заметка Obsidian, по строке на запись:

    # комментарий (и заголовки Markdown — тоже)
    Фурье
    - Котельников               маркеры списка снимаются
    фурии -> Фурье              замена «как слышится -> как пишется»

Два механизма с разной ценой (замеры — BENCHMARKS.md, «Словарь терминов»):

- **Подсказка** (`hotwords` faster-whisper) подставляется в каждое 30-секундное окно и реально
  вытягивает термины. Но место в окне общее с текстом: дальше ~60 токенов (10–12 терминов)
  распознавание начинает терять слова и вставлять термины, которых не говорили. Поэтому порядок
  — приоритет: сначала термины по порядку, затем правые части замен; что не влезло — отрезается.
- **Замены** применяются к готовому транскрипту поверх кэша: ограничений нет, и правка карты не
  перезапускает распознавание.

Refs #13
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from summuraze.types import Segment

_LIST_MARKER = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
_ARROW = "->"


class GlossaryError(ValueError):
    """Словарь не читается — сообщение для человека."""


@dataclass(frozen=True)
class Glossary:
    terms: tuple[str, ...] = ()
    replacements: tuple[tuple[str, str], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.terms or self.replacements)

    @property
    def hint(self) -> tuple[str, ...]:
        """Что подсказывать Whisper, по приоритету: явные термины, затем «как пишется» замен.

        Считается после слияния словарей: замена общего словаря, перекрытая заменой курса,
        не тратит место в подсказке.
        """
        return _unique([*self.terms, *(written for _, written in self.replacements)])

    def merge(self, other: Glossary) -> Glossary:
        """`self` важнее: его термины идут первыми и первыми попадают в подсказку."""
        return Glossary(
            terms=_unique([*self.terms, *other.terms]),
            replacements=tuple(dict([*other.replacements, *self.replacements]).items()),
        )


def load_glossary(paths: Iterable[Path | None]) -> Glossary:
    """Прочитать словари по порядку приоритета; `None` пропускаются."""
    result = Glossary()
    for path in paths:
        if path is not None:
            result = result.merge(parse_glossary(_read(path)))
    return result


def parse_glossary(text: str) -> Glossary:
    terms: list[str] = []
    replacements: dict[str, str] = {}
    for line in _body(text.splitlines()):
        line = _LIST_MARKER.sub("", line.strip()).strip()
        if not line or line.startswith("#"):
            continue
        if _ARROW in line:
            heard, written = (part.strip() for part in line.split(_ARROW, 1))
            if not heard or not written:
                raise GlossaryError(f"замена без одной из частей: «{line}»")
            replacements[heard.lower()] = written
        else:
            terms.append(line)
    return Glossary(terms=_unique(terms), replacements=tuple(replacements.items()))


def apply_replacements(segments: Iterable[Segment], glossary: Glossary) -> list[Segment]:
    """Заменить «как слышится» на «как пишется» целыми словами, без учёта регистра."""
    rules = [
        (re.compile(rf"(?<!\w){re.escape(heard)}(?!\w)", re.IGNORECASE), written)
        for heard, written in glossary.replacements
    ]
    if not rules:
        return list(segments)

    def fix(text: str) -> str:
        for pattern, written in rules:
            text = pattern.sub(written, text)
        return text

    return [
        s.model_copy(
            update={
                "text": fix(s.text),
                # Многословная замена в отдельных словах не ловится — текст сегмента главнее.
                "words": [w.model_copy(update={"text": fix(w.text)}) for w in s.words],
            }
        )
        for s in segments
    ]


def _read(path: Path) -> str:
    try:
        return path.expanduser().read_text(encoding="utf-8")
    except FileNotFoundError:
        raise GlossaryError(f"словарь терминов не найден: {path}") from None
    except (OSError, UnicodeDecodeError) as e:
        raise GlossaryError(f"словарь терминов не читается: {path}: {e}") from None


def _body(lines: list[str]) -> list[str]:
    """Без YAML-шапки заметки Obsidian."""
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                return lines[i + 1 :]
    return lines


def _unique(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out = []
    for item in items:
        if item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return tuple(out)
