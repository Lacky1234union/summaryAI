"""Профили: схема ответа, разбор JSON модели, сборка заметки с привязкой таймкодов."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from summuraze.config import LLMSettings
from summuraze.pipeline import stages
from summuraze.pipeline.stages import SummaryFormatError
from summuraze.render.markdown import NoteLayout, render_note, resolve_at
from summuraze.summarize.echo import EchoLLM
from summuraze.summarize.openai_compat import OpenAICompatLLM
from summuraze.summarize.profiles import PROFILES, LectureDoc, RawDoc, get_profile
from summuraze.summarize.schema import empty_document, strict_schema
from summuraze.types import AudioInfo, LLMResult, Segment, TokenUsage, Transcript

TRANSCRIPT = Transcript(
    source=Path("/x/lecture.mp4"),
    duration=300,
    language="ru",
    asr_model="large-v3-turbo",
    segments=[
        Segment(start=0.4, end=10, text="Сегодня говорим о преобразовании Фурье."),
        Segment(start=12.7, end=30, text="Фурье раскладывает сигнал на гармоники."),
        Segment(start=95.2, end=110, text="Теперь теорема Котельникова."),
        Segment(
            start=200, end=210, text="Продолжение следует...", suspect=True, suspect_reason="шаблон"
        ),
    ],
)
AUDIO = AudioInfo(
    source=Path("/x/lecture.mp4"),
    wav=Path("/x/audio.wav"),
    duration=300,
    source_duration=300,
    source_codec="aac",
    source_channels=2,
    source_sample_rate=44100,
    audio_streams=1,
)


def _claim(text: str, at: str) -> dict[str, str]:
    return {"text": text, "at": at}


LECTURE = {
    "title": "Цифровая обработка сигналов",
    "tldr": [_claim("Фурье раскладывает сигнал на гармоники", "00:00:12")],
    "sections": [
        {"title": "Фурье", "at": "00:00:00", "theses": [_claim("Гармоники", "00:00:12")]},
        {"title": "Котельников", "at": "00:01:35", "theses": [_claim("Выдумка", "00:59:00")]},
    ],
    "glossary": [{"term": "Гармоника", "definition": "синусоида", "at": "00:00:12"}],
    "examples": [_claim("$x(t) = \\sum a_k e^{ikt}$", "00:00:12")],
    "qa": [{"question": "А если сигнал не периодический?", "answer": "", "at": "1:35"}],
    "reading": [],
    "unclear": [_claim("Похоже, конец записи", "00:03:20")],
}


class TestSchema:
    @pytest.mark.parametrize("name", sorted(PROFILES))
    def test_strict_schema(self, name: str) -> None:
        schema = strict_schema(get_profile(name).document)

        def objects(node: Any) -> list[dict[str, Any]]:
            if isinstance(node, dict):
                found = [node] if node.get("type") == "object" else []
                return found + [o for v in node.values() for o in objects(v)]
            if isinstance(node, list):
                return [o for v in node for o in objects(v)]
            return []

        for obj in objects(schema):
            assert obj["additionalProperties"] is False
            assert set(obj["required"]) == set(obj["properties"])

    @pytest.mark.parametrize("name", sorted(PROFILES))
    def test_empty_document_is_valid(self, name: str) -> None:
        profile = get_profile(name)
        profile.document.model_validate(empty_document(strict_schema(profile.document)))

    def test_empty_document_scalars(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "n": {"type": "integer"},
                "f": {"type": "number"},
                "b": {"type": "boolean"},
                "s": {"type": "string"},
            },
        }
        assert empty_document(schema) == {"n": 0, "f": 0, "b": False, "s": ""}


class TestResolveAt:
    @pytest.mark.parametrize(
        ("at", "expected"),
        [
            ("00:00:12", 12.7),  # строка префикса «[00:00:12]» — сегмент с 12.7 с
            ("00:00:13", 12.7),
            ("[00:01:35]", 95.2),
            ("1:35", 95.2),  # ММ:СС
            ("00:00:13–00:00:16", 12.7),  # диапазон — по началу
            ("00:00:00", 0.4),
            ("00:59:00", None),  # за концом записи
            ("не помню", None),
        ],
    )
    def test_resolve(self, at: str, expected: float | None) -> None:
        assert resolve_at(at, TRANSCRIPT) == expected

    def test_before_first_segment(self) -> None:
        late = TRANSCRIPT.model_copy(update={"segments": TRANSCRIPT.segments[2:]})
        assert resolve_at("00:00:05", late) == 5.0


def _render(tmp_path: Path, profile: str, document: dict[str, Any], *, debug: bool = True) -> str:
    prof = get_profile(profile)
    note = render_note(
        NoteLayout(tmp_path / "n.md", tmp_path / "Записи", "2026-10-27 lecture"),
        audio=AUDIO,
        transcript=TRANSCRIPT,
        summary=LLMResult(text="", model="m", usage=TokenUsage(input=0, output=0)),
        document=prof.document.model_validate(document),
        profile=profile,
        debug=debug,
    )
    return note.read_text(encoding="utf-8")


class TestLectureNote:
    def test_structure_and_timecodes(self, tmp_path: Path) -> None:
        text = _render(tmp_path, "lecture", LECTURE)
        assert "# Цифровая обработка сигналов" in text
        for heading in (
            "## Коротко",
            "## Глоссарий",
            "## Формулы и примеры",
            "## Вопросы и ответы",
        ):
            assert heading in text
        assert "## Фурье `00:00:00`" in text
        assert "## Котельников `00:01:35`" in text
        assert "- Фурье раскладывает сигнал на гармоники `00:00:12`" in text
        assert "- **Гармоника** — синусоида `00:00:12`" in text
        assert "**Вопрос:** А если сигнал не периодический? `00:01:35`" in text
        assert "**Ответ:**" not in text, "пустой ответ не рисуем"
        assert "## Что почитать" not in text, "пустые разделы не рисуем"
        assert "## Неразборчиво" in text and "`00:03:20`" in text

    def test_unverifiable_claim_is_marked(self, tmp_path: Path) -> None:
        """Guardrail #5: утверждение без проверяемого таймкода видно сразу."""
        assert "- Выдумка ⚠️ без таймкода" in _render(tmp_path, "lecture", LECTURE)

    def test_empty_title_falls_back_to_file(self, tmp_path: Path) -> None:
        assert "# lecture" in _render(tmp_path, "lecture", {**LECTURE, "title": " "})


class TestCleanNote:
    """Итоговый вид (#50): то же содержание, другая подача."""

    def test_layout(self, tmp_path: Path) -> None:
        text = _render(tmp_path, "lecture", LECTURE, debug=False)
        head = text.split("---")[1]
        assert 'aliases: ["Цифровая обработка сигналов"]' in head
        assert 'duration: "05:00"' in head and 'tags: ["summuraze", "лекция"]' in head
        assert "llm_model" not in head and "source" not in head and "cost_usd" not in head
        assert "> [!summary] Коротко\n> - Фурье раскладывает сигнал на гармоники `00:12`" in text
        assert "## Котельников `01:35`" in text
        assert "> [!abstract] Глоссарий\n> - **Гармоника** — синусоида `00:12`" in text
        assert "> [!question] А если сигнал не периодический? `01:35`\n\n" in text
        assert "> [!warning]- Неразборчиво" in text, "неразборчивое — свёрнуто"
        assert "> [!quote]- Транскрипт\n> - `00:00` Сегодня говорим" in text
        assert "## Транскрипт" not in text and "Что почитать" not in text
        assert "- Выдумка ⚠️ без таймкода" in text, "guardrail #5 и в чистом виде"

    def test_raw_profile_tag(self, tmp_path: Path) -> None:
        text = _render(tmp_path, "raw", {"blocks": []}, debug=False)
        assert 'tags: ["summuraze", "транскрипт"]' in text


class TestRawNote:
    def test_blocks_hold_verbatim_transcript(self, tmp_path: Path) -> None:
        doc = {
            "blocks": [
                {"title": "Котельников", "at": "00:01:35"},
                {"title": "Фурье", "at": "00:00:00"},
                {"title": "Мимо", "at": "02:00:00"},
            ]
        }
        text = _render(tmp_path, "raw", doc)
        fourier, kotelnikov = (
            text.index("## Фурье `00:00:00`"),
            text.index("## Котельников `00:01:35`"),
        )
        assert fourier < text.index("Фурье раскладывает сигнал") < kotelnikov
        assert kotelnikov < text.index("Теперь теорема Котельникова")
        assert "Мимо" not in text, "блок с таймкодом за концом записи выброшен"
        assert "⚠️ *Продолжение следует...*" in text

    def test_no_valid_blocks_gives_plain_transcript(self, tmp_path: Path) -> None:
        text = _render(tmp_path, "raw", {"blocks": []})
        assert "## Транскрипт" in text and "Сегодня говорим" in text


class FixedLLM(EchoLLM):
    def __init__(self, text: str) -> None:
        self.reply = text

    def complete(self, **kw: Any) -> LLMResult:
        return LLMResult(text=self.reply, model="fixed", usage=TokenUsage(input=1, output=1))


class TestStage:
    def test_document_saved_with_result(self, tmp_path: Path) -> None:
        prof = get_profile("lecture")
        stages.build_summary(tmp_path, FixedLLM(json.dumps(LECTURE)), TRANSCRIPT, prof)
        result, doc = stages.load_summary(tmp_path, prof)
        assert isinstance(doc, LectureDoc) and doc.title == "Цифровая обработка сигналов"
        assert result.model == "fixed"

    @pytest.mark.parametrize("reply", ["Конспект: всё хорошо", '{"title": "обрезано'])
    def test_off_schema_reply_fails(self, tmp_path: Path, reply: str) -> None:
        with pytest.raises(SummaryFormatError, match="не по схеме профиля 'raw'"):
            stages.build_summary(tmp_path, FixedLLM(reply), TRANSCRIPT, get_profile("raw"))

    def test_echo_raw(self, tmp_path: Path) -> None:
        stages.build_summary(tmp_path, EchoLLM(), TRANSCRIPT, get_profile("raw"))
        _, doc = stages.load_summary(tmp_path, get_profile("raw"))
        assert doc == RawDoc(blocks=[])


def test_adapter_sends_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    llm = OpenAICompatLLM(
        LLMSettings(api_key=SecretStr("k")),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    schema = strict_schema(RawDoc)
    llm.complete(system="s", prefix="p", question="q", max_tokens=10, schema=schema)
    llm.complete(system="s", prefix="p", question="q", max_tokens=10)
    with_schema, without = sent
    assert with_schema["response_format"]["json_schema"] == {
        "name": "document",
        "strict": True,
        "schema": schema,
    }
    quant = ["fp8", "bf16", "fp16", "fp32", "unknown"]
    assert with_schema["provider"] == {"require_parameters": True, "quantizations": quant}
    assert "response_format" not in without and without["provider"] == {"quantizations": quant}


def test_quantization_filter_can_be_disabled() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    llm = OpenAICompatLLM(
        LLMSettings(api_key=SecretStr("k"), openrouter_quantizations=[]),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    llm.complete(system="s", prefix="p", question="q", max_tokens=10)
    assert "provider" not in sent[0]


def test_repeated_thoughts_rendered_once(tmp_path: Path) -> None:
    """Сбойный провайдер копировал тезисы во все разделы — в заметке мысль только один раз."""
    same = _claim("Фурье раскладывает сигнал на гармоники.", "00:00:12")
    doc = {
        **LECTURE,
        "tldr": [same, same],
        "sections": [{"title": "Фурье", "at": "00:00:00", "theses": [same]}],
        "examples": [same],
        "reading": [_claim("ФУРЬЕ раскладывает сигнал на гармоники", "00:00:12")],
        "qa": [LECTURE["qa"][0], LECTURE["qa"][0]],
        "glossary": [LECTURE["glossary"][0], LECTURE["glossary"][0]],
        "unclear": [],
    }
    text = _render(tmp_path, "lecture", doc)
    assert text.count("раскладывает сигнал на гармоники") == 2, "тезис + строка транскрипта"
    assert "## Формулы и примеры" not in text and "## Что почитать" not in text
    assert text.count("**Вопрос:**") == 1 and text.count("**Гармоника**") == 1
