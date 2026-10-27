"""Словарь терминов: формат файла, приоритет словарей, замены поверх кэша."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from summuraze.asr.terms import (
    Glossary,
    GlossaryError,
    apply_replacements,
    load_glossary,
    parse_glossary,
)
from summuraze.cli import app
from summuraze.types import Segment, Word

NOTE = """---
tags: [курс, сигналы]
---
# Цифровая обработка сигналов

- Фурье
- Котельников
* окно Хэмминга
1. Найквист
фурии -> Фурье
кули тюрьки -> Кули и Тьюки
# комментарий
фурье
"""


def test_obsidian_note_parses() -> None:
    g = parse_glossary(NOTE)
    assert g.terms == ("Фурье", "Котельников", "окно Хэмминга", "Найквист")
    assert g.hint == (*g.terms, "Кули и Тьюки")
    assert g.replacements == (("фурии", "Фурье"), ("кули тюрьки", "Кули и Тьюки"))


def test_broken_replacement() -> None:
    with pytest.raises(GlossaryError, match="без одной из частей"):
        parse_glossary("фурии ->")


def test_course_glossary_goes_first(tmp_path: Path) -> None:
    course = tmp_path / "course.txt"
    common = tmp_path / "common.md"
    course.write_text("Котельников\nфурии -> Фурье\n", encoding="utf-8")
    common.write_text("Фурье\nКотельников\nфурии -> Фурия\nспектр\n", encoding="utf-8")
    g = load_glossary([course, None, common])
    assert g.terms == ("Котельников", "Фурье", "спектр")
    assert dict(g.replacements)["фурии"] == "Фурье", "замена курса важнее общей"
    assert "Фурия" not in g.hint, "перекрытая замена не тратит место в подсказке"
    assert not Glossary() and g


def test_missing_glossary(tmp_path: Path) -> None:
    with pytest.raises(GlossaryError, match="не найден"):
        load_glossary([tmp_path / "nope.txt"])


def test_unreadable_glossary(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(GlossaryError, match="не читается"):
        load_glossary([bad])


def test_replacements_whole_words_any_case() -> None:
    g = parse_glossary("фурии -> Фурье\nкули тюрьки -> Кули и Тьюки")
    seg = Segment(
        start=0,
        end=3,
        text="Ряд Фурии и метод кули тюрьки, но не фуриист.",
        words=[Word(start=0, end=1, text=" Фурии", prob=0.5)],
    )
    [fixed] = apply_replacements([seg], g)
    assert fixed.text == "Ряд Фурье и метод Кули и Тьюки, но не фуриист."
    assert fixed.words[0].text == " Фурье"
    assert seg.text.startswith("Ряд Фурии"), "исходник не меняется"


def test_no_replacements_is_identity() -> None:
    seg = Segment(start=0, end=1, text="фурии")
    assert apply_replacements([seg], Glossary(terms=("Фурье",))) == [seg]


def test_cli_rejects_missing_terms_file(tmp_path: Path) -> None:
    rec = tmp_path / "rec.mp3"
    rec.write_bytes(b"")
    result = CliRunner().invoke(app, ["run", str(rec), "--terms", str(tmp_path / "nope.txt")])
    assert result.exit_code == 1
    assert "словарь терминов не найден" in result.output
