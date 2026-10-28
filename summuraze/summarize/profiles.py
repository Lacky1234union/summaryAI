"""Профили документа: что спросить у модели и какой формы ответ ждать.

Модель отвечает **JSON по схеме профиля**, Markdown собираем сами (`render/markdown.py`). Так у
каждого утверждения таймкод — отдельное поле, а не кусок текста: рендер делает из него ссылку
(#18), а проверка привязки (#22) — проверку, без разбора свободного текста регулярками.

«Сырой» профиль не пересказывает вовсе: модель только режет транскрипт на смысловые блоки и
называет их, текст под заголовками — дословно из ASR. Интерпретация исключена устройством, а не
просьбой в промпте.

`version` профиля входит в ключ кэша саммари: правка промпта или схемы перегенерирует саммари,
но не ASR. Меняешь текст ниже — поднимай версию.

Refs #17
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

AT = "таймкод из транскрипта в виде ЧЧ:ММ:СС — строка, где это сказано"


class Claim(BaseModel):
    text: str = Field(description="Одно утверждение, своими словами, по-русски")
    at: str = Field(description=AT)


class Section(BaseModel):
    title: str = Field(description="Название раздела лекции")
    at: str = Field(description="Таймкод начала раздела, ЧЧ:ММ:СС")
    theses: list[Claim] = Field(description="Главные тезисы раздела, каждый со своим таймкодом")


class Term(BaseModel):
    term: str = Field(description="Термин, как его пишут")
    definition: str = Field(description="Определение так, как его дал лектор")
    at: str = Field(description=AT)


class Question(BaseModel):
    question: str = Field(description="Вопрос из зала или от лектора к аудитории")
    answer: str = Field(description="Ответ лектора; пусто, если ответа не было")
    at: str = Field(description=AT)


class LectureDoc(BaseModel):
    title: str = Field(description="Тема лекции одной строкой")
    tldr: list[Claim] = Field(description="3–7 главных мыслей лекции")
    sections: list[Section] = Field(description="Разделы лекции по порядку")
    glossary: list[Term] = Field(description="Термины, которые лектор вводил или объяснял")
    examples: list[Claim] = Field(
        description="Формулы и разобранные примеры; формулы — в LaTeX между $…$"
    )
    qa: list[Question] = Field(description="Вопросы и ответы; пусто, если их не было")
    reading: list[Claim] = Field(
        description="Что лектор советовал почитать или посмотреть; пусто, если не советовал"
    )
    unclear: list[Claim] = Field(
        description="Места, где запись неразборчива или помечена [?], и что там, похоже, было"
    )


class Block(BaseModel):
    title: str = Field(description="Короткий заголовок смыслового блока")
    at: str = Field(description="Таймкод первой строки блока, ЧЧ:ММ:СС — ровно как в транскрипте")


class RawDoc(BaseModel):
    blocks: list[Block] = Field(description="Смысловые блоки транскрипта по порядку")


@dataclass(frozen=True)
class Profile:
    name: str
    version: int
    system: str
    question: str
    document: type[BaseModel]
    max_tokens: int = 16_000


_RULES = (
    "Правила:\n"
    "- Каждое утверждение — с таймкодом строки транскрипта, где это сказано. Утверждение, "
    "которое нельзя привязать к строке, не пиши.\n"
    "- Только то, что есть в транскрипте: не дополняй своими знаниями, не исправляй лектора.\n"
    "- Термины, имена и формулы пиши правильно, даже если распознаватель исказил их в "
    "транскрипте, — но смысл бери из транскрипта."
)

PROFILES: dict[str, Profile] = {
    "lecture": Profile(
        name="lecture",
        version=1,
        system=(
            "Ты составляешь конспект лекции, который читают вместо записи. Конспект плотный: "
            "мысли, определения, формулы, примеры, а не пересказ «лектор сказал, что…».\n\n"
            + _RULES
        ),
        question="Составь конспект лекции по схеме ответа.",
        document=LectureDoc,
    ),
    "raw": Profile(
        name="raw",
        version=1,
        system=(
            "Ты размечаешь транскрипт на смысловые блоки, ничего не пересказывая. Блок — "
            "законченная часть разговора на одну тему, обычно от одной до десяти минут. "
            "Заголовок блока — несколько слов о его теме. Таймкод блока — таймкод его первой "
            "строки, ровно как в транскрипте."
        ),
        question="Раздели транскрипт на смысловые блоки по схеме ответа.",
        document=RawDoc,
    ),
}


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise ValueError(f"неизвестный профиль {name!r}; есть: {known}") from None
