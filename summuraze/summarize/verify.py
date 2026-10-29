"""verify: каждое утверждение документа сверяется с транскриптом по своему таймкоду (#22).

Ссылка-таймкод делает проверку дешёвой для человека, но человек проверяет выборочно. Этот проход
проверяет всё: модель-судья получает транскрипт тем же префиксом, что и саммари (кэш провайдера
и слот своей модели попадают), и список утверждений с их таймкодами. По каждому — «да / частично /
нет» и таймкод строки, где это на самом деле сказано.

Неподтверждённое **помечается, а не удаляется**: «модель это придумала» — тоже информация, по
доле таких пометок видно, насколько доверять документу целиком.

Без модели проверяется то, что проверяется без неё:
- утверждение без читаемого таймкода — находка само по себе (guardrail #5);
- «неразборчиво» — по самой записи: рядом строка, помеченная детектором (#12), или слова, в
  которых Whisper не уверен. Иначе пункт выдуман (Qwen3-8B, #19). Судье это не отдаём: Qwen3-4B
  называла «голланд» искажением в причине — и ставила «нет» (2026-10-29).

Судья — та же модель, что писала конспект, и это слабое место: свою ошибку она может и
подтвердить. Поэтому вердикт опирается на таймкод доказательства, который сверяется с таймкодом
утверждения, а доля подтверждённых сверяется вручную (BENCHMARKS.md).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from summuraze.render.markdown import fmt_timecode, resolve_at
from summuraze.summarize.profiles import Claim, Question, Term
from summuraze.types import Transcript

# Меняешь промпт, допуски или разбор вердиктов — поднимай: версия входит в ключ кэша проверки.
VERSION = 1
# Клик по таймкоду открывает запись, и её слушают вперёд: доказательство чуть раньше — уже
# перемотка назад, на пару минут позже — ещё та же мысль.
BEFORE = 30.0
AFTER = 120.0
# «Неразборчиво» подтверждает строка в пределах этого от его таймкода: помеченная детектором или
# с LOW_WORDS словами уверенностью ниже LOW_PROB. Живая лекция: медиана уверенности 0.98, в
# искажённой строке («голланд… осад от самой») — пять слов ниже 0.4, во внятной — ни одного или
# одно (английский термин посреди русской речи).
UNCLEAR_NEAR = 30.0
LOW_PROB = 0.4
LOW_WORDS = 2
# Сколько раз переспросить про утверждения, по которым судья промолчал: YandexGPT-8B ответила
# про 2 из 14 и закончила ответ (2026-10-29).
RETRIES = 2
# Вердикт — номер, слово, таймкод и короткая причина: десятки токенов на утверждение.
TOKENS_PER_CLAIM = 64
# Причина — в схеме ответа, а не только в промпте: Qwen3-4B на «да» писала в неё цитату из
# транскрипта и раз зациклилась на тысячу символов (2026-10-29). Для llama.cpp схема — грамматика,
# лимит жёсткий; API может его не соблюдать, поэтому длину режем и сами, а не валидацией.
NOTE_CHARS = 100
_UNCLEAR = "unclear"

Status = Literal["ok", "moved", "partial", "unsupported", "no_timecode", "unchecked"]


class Verdict(BaseModel):
    # Таймкод доказательства — до вердикта: модель сначала ищет строку, потом судит. Обратный
    # порядок давал «нет» дословно сказанному (Qwen3-4B, 2026-10-29).
    n: int = Field(description="Номер утверждения из списка")
    at: str = Field(
        description="Таймкод строки транскрипта, где это сказано или ближе всего по смыслу, "
        "ЧЧ:ММ:СС ровно как в транскрипте; пустая строка, если такой строки нет"
    )
    verdict: Literal["да", "частично", "нет"] = Field(
        description="да — в транскрипте это сказано; частично — часть добавлена или искажена; "
        "нет — этого в транскрипте нет"
    )
    note: str = Field(
        description="При «частично» и «нет» — чего именно нет в транскрипте, своими словами, "
        "до 12 слов, без цитат; при «да» — пустая строка",
        json_schema_extra={"maxLength": NOTE_CHARS},
    )


class Verdicts(BaseModel):
    items: list[Verdict] = Field(description="Вердикт по каждому утверждению, по номерам")


SYSTEM = (
    "Ты проверяешь конспект по транскрипту, а не пишешь его. Ниже — утверждения из конспекта; у "
    "каждого номер и таймкод строки транскрипта, на которую оно ссылается.\n\n"
    "Для каждого утверждения найди строки транскрипта, где это сказано, и вынеси вердикт:\n"
    "- «да» — лектор это сказал; пересказ своими словами допустим, добавления — нет;\n"
    "- «частично» — часть сказана, а часть добавлена или искажена: число, имя, дата, вывод, "
    "пример, которых в транскрипте нет;\n"
    "- «нет» — в транскрипте этого нет или сказано обратное.\n\n"
    "Проверяй по транскрипту, а не по своим знаниям: верное по сути, но не сказанное лектором — "
    "это «нет». Ищи сначала около указанного таймкода, потом во всём транскрипте.\n\n"
    "Транскрипт — автоматическое распознавание речи, с ошибками: термины, имена и названия в нём "
    "бывают искажены («оси» вместо «C», «голанг» вместо «Go»). Если в утверждении то же слово "
    "написано правильно — это не расхождение. Если лектор только обещает рассказать о "
    "чём-то, утверждение «рассказал» — «частично».\n\n"
    "Ответь про каждое утверждение по порядку номеров, ничего не пропуская."
)


@dataclass(frozen=True)
class ClaimRef:
    """Утверждение документа. `path` — адрес в документе (`sections.1.theses.0`): по нему рендер
    находит пометку для строки заметки."""

    path: str
    text: str
    at: float | None

    @property
    def unclear(self) -> bool:
        return self.path.split(".", 1)[0] == _UNCLEAR


class Mark(BaseModel):
    status: Status
    note: str = ""
    # Где это сказано на самом деле, если не там, куда ведёт таймкод утверждения.
    evidence: float | None = None


_LABELS: dict[Status, str] = {
    "ok": "подтверждены",
    "moved": "таймкод ведёт не туда",
    "partial": "частично",
    "unsupported": "нет в записи",
    "no_timecode": "без таймкода",
    "unchecked": "не проверены",
}


class Report(BaseModel):
    marks: dict[str, Mark] = Field(default_factory=dict)

    def count(self, *statuses: Status) -> int:
        return sum(1 for m in self.marks.values() if m.status in statuses)

    @property
    def confirmed(self) -> int:
        """Сказано в записи — даже если таймкод ведёт не туда: факт верный, ссылка нет."""
        return self.count("ok", "moved")

    def describe(self) -> str:
        if not self.marks:
            return "утверждений нет"
        head = f"{self.confirmed} из {len(self.marks)} утверждений подтверждены транскриптом"
        rest = [
            f"{self.count(s)} — {_LABELS[s]}"
            for s in ("moved", "partial", "unsupported", "no_timecode", "unchecked")
            if self.count(s)
        ]
        return "; ".join([head, *rest])


def claims(document: BaseModel, transcript: Transcript) -> list[ClaimRef]:
    """Все утверждения документа по порядку. Заголовки (разделы, блоки `raw`) — не утверждения:
    их проверяет сам читатель, переходя по ссылке."""
    return [
        ClaimRef(path, text, resolve_at(at, transcript)) for path, text, at in _walk(document, "")
    ]


def _walk(node: BaseModel, path: str) -> Iterator[tuple[str, str, str]]:
    text = _text(node)
    if text is not None:
        yield path, text, getattr(node, "at", "")
    for name, value in node:
        if isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, BaseModel):
                    yield from _walk(item, f"{path}.{name}.{i}".lstrip("."))


def _text(node: BaseModel) -> str | None:
    if isinstance(node, Claim):
        return node.text.strip()
    if isinstance(node, Term):
        return f"{node.term.strip()} — {node.definition.strip()}"
    if isinstance(node, Question):
        answer = node.answer.strip()
        return f"Вопрос: {node.question.strip()}" + (f" Ответ: {answer}" if answer else "")
    return None


def to_judge(items: Sequence[ClaimRef], decided: dict[str, Mark]) -> list[ClaimRef]:
    """Что уходит модели: решённое без неё (`structural`) — не уходит."""
    return [c for c in items if c.path not in decided]


def question(items: Sequence[ClaimRef]) -> str:
    lines = [f"{n}. [{fmt_timecode(c.at or 0)}] {c.text}" for n, c in enumerate(items, 1)]
    return "Утверждения конспекта:\n" + "\n".join(lines) + "\n\nВынеси вердикт по каждому."


def output_tokens(n: int) -> int:
    return TOKENS_PER_CLAIM * n + 256


def judge(items: Sequence[ClaimRef], verdicts: Verdicts, transcript: Transcript) -> dict[str, Mark]:
    """Вердикты модели → пометки. `items` — ровно то, что ушло в `question`, в том же порядке."""
    by_n = {v.n: v for v in verdicts.items}
    marks: dict[str, Mark] = {}
    for n, claim in enumerate(items, 1):
        v = by_n.get(n)
        if v is None or claim.at is None:
            marks[claim.path] = Mark(status="unchecked")
            continue
        evidence = resolve_at(v.at, transcript) if v.at.strip() else None
        note = _short(v.note) if v.verdict != "да" else ""
        if v.verdict == "нет":
            marks[claim.path] = Mark(status="unsupported", note=note)
        elif evidence is not None and not claim.at - BEFORE <= evidence <= claim.at + AFTER:
            # Частично и не там — сначала «не там»: по ссылке человек не найдёт и верную часть.
            marks[claim.path] = Mark(status="moved", note=note, evidence=evidence)
        elif v.verdict == "частично":
            marks[claim.path] = Mark(status="partial", note=note, evidence=evidence)
        else:
            marks[claim.path] = Mark(status="ok")
    return marks


def structural(items: Sequence[ClaimRef], transcript: Transcript) -> dict[str, Mark]:
    """Пометки, для которых модель не нужна: нет таймкода, «неразборчиво»."""
    doubtful = [
        s.start
        for s in transcript.segments
        if s.suspect or sum(w.prob < LOW_PROB for w in s.words) >= LOW_WORDS
    ]
    marks: dict[str, Mark] = {}
    for claim in items:
        if claim.at is None:
            marks[claim.path] = Mark(status="no_timecode")
        elif claim.unclear:
            at = claim.at
            near = any(abs(t - at) <= UNCLEAR_NEAR for t in doubtful)
            marks[claim.path] = (
                Mark(status="ok")
                if near
                else Mark(status="unsupported", note="речь здесь распознана уверенно")
            )
    return marks


def _short(note: str) -> str:
    note = " ".join(note.split())
    return note if len(note) <= NOTE_CHARS else note[: NOTE_CHARS - 1].rstrip() + "…"
