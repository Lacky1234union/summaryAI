"""verify: утверждения документа против транскрипта (#22)."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from summuraze.config import ASRSettings, LLMSettings, Settings
from summuraze.pipeline import stages
from summuraze.pipeline.orchestrator import Pipeline
from summuraze.render.markdown import NoteLayout, render_note
from summuraze.summarize import verify
from summuraze.summarize.echo import EchoLLM
from summuraze.summarize.profiles import Claim, LectureDoc, Question, RawDoc, Section, Term
from summuraze.summarize.verify import Mark, Report, Verdict, Verdicts
from summuraze.types import AudioInfo, LLMResult, Segment, TokenUsage, Transcript, Word
from tests.conftest import SPEECH_CLIPS


def _transcript(n: int = 60, *, step: float = 10.0, suspect: set[int] | None = None) -> Transcript:
    segments = [
        Segment(
            start=i * step,
            end=i * step + step - 1,
            text=f"фраза {i}.",
            suspect=i in (suspect or set()),
            suspect_reason="повтор" if i in (suspect or set()) else None,
        )
        for i in range(n)
    ]
    return Transcript(
        source=Path("/x/лекция.mp4"),
        duration=n * step,
        language="ru",
        asr_model="m",
        segments=segments,
    )


def _doc() -> LectureDoc:
    return LectureDoc(
        title="Сборка мусора",
        tldr=[Claim(text="Память освобождает сборщик", at="00:00:10")],
        sections=[
            Section(
                title="Поколения",
                at="00:01:00",
                theses=[
                    Claim(text="Молодое поколение чистится чаще", at="00:01:00"),
                    Claim(text="Старое — реже", at="без таймкода"),
                ],
            )
        ],
        glossary=[Term(term="GC", definition="сборщик мусора", at="00:02:00")],
        examples=[],
        qa=[Question(question="А паузы?", answer="Короткие", at="00:03:00")],
        reading=[],
        unclear=[
            Claim(text="Не слышно название", at="00:05:00"),
            Claim(text="Шум в аудитории", at="00:08:00"),
        ],
    )


class TestClaims:
    def test_paths_texts_and_headings_skipped(self) -> None:
        items = verify.claims(_doc(), _transcript())
        assert [c.path for c in items] == [
            "tldr.0",
            "sections.0.theses.0",
            "sections.0.theses.1",
            "glossary.0",
            "qa.0",
            "unclear.0",
            "unclear.1",
        ], "заголовок раздела — не утверждение"
        by = {c.path: c for c in items}
        assert by["glossary.0"].text == "GC — сборщик мусора"
        assert by["qa.0"].text == "Вопрос: А паузы? Ответ: Короткие"
        assert by["tldr.0"].at == 10.0 and by["sections.0.theses.1"].at is None
        assert by["unclear.0"].unclear and not by["tldr.0"].unclear

    def test_raw_has_nothing_to_check(self) -> None:
        assert verify.claims(RawDoc(blocks=[]), _transcript()) == []

    def test_judge_gets_only_what_needs_a_model(self) -> None:
        t = _transcript(suspect={31})
        items = verify.claims(_doc(), t)
        asked = verify.to_judge(items, verify.structural(items, t))
        assert [c.path for c in asked] == [
            "tldr.0",
            "sections.0.theses.0",
            "glossary.0",
            "qa.0",
        ], "без таймкода и «неразборчиво» — решены без модели"
        q = verify.question(asked)
        assert "1. [00:00:10] Память освобождает сборщик" in q
        assert "4. [00:03:00] Вопрос: А паузы? Ответ: Короткие" in q


class TestStructural:
    def test_no_timecode_and_unclear_against_detector(self) -> None:
        t = _transcript(suspect={31})  # 00:05:10 — рядом с первым «неразборчиво»
        marks = verify.structural(verify.claims(_doc(), t), t)
        assert marks["sections.0.theses.1"].status == "no_timecode"
        assert marks["unclear.0"].status == "ok"
        assert marks["unclear.1"].status == "unsupported", "речь там внятная — пункт выдуман"
        assert "tldr.0" not in marks, "остальное — модели"

    def test_unclear_confirmed_by_whisper_doubt(self) -> None:
        """Whisper исказил без пометки детектора («голланд… осад от самой»): слова с низкой
        уверенностью подтверждают «неразборчиво», одно сомнительное слово — нет."""
        t = _transcript()

        def words(*probs: float) -> list[Word]:
            return [Word(start=0, end=0, text="w", prob=p) for p in probs]

        t.segments[48].words = words(0.04, 0.9, 0.15, 0.34)  # 00:08:00 — «Шум в аудитории»
        t.segments[30].words = words(0.11, 0.95, 0.99)  # 00:05:00 — английский термин
        marks = verify.structural(verify.claims(_doc(), t), t)
        assert marks["unclear.1"].status == "ok"
        assert marks["unclear.0"].status == "unsupported"
        assert marks["unclear.0"].note == "речь здесь распознана уверенно"


def _claim(at: float, path: str = "x.0") -> verify.ClaimRef:
    return verify.ClaimRef(path, "текст", at)


class TestJudge:
    @pytest.mark.parametrize(
        ("verdict", "at", "status"),
        [
            ("да", "00:01:40", "ok"),
            ("да", "", "ok"),
            ("да", "00:01:10", "ok"),  # 30 с назад — ещё в допуске
            ("да", "00:03:40", "ok"),  # 2 минуты вперёд — та же мысль
            ("да", "00:01:00", "moved"),
            ("да", "00:04:00", "moved"),
            ("частично", "00:01:50", "partial"),
            ("частично", "00:09:00", "moved"),
            ("нет", "", "unsupported"),
        ],
    )
    def test_verdicts(self, verdict: Any, at: str, status: str) -> None:
        t = _transcript()
        v = Verdicts(items=[Verdict(n=1, verdict=verdict, at=at, note="причина")])
        mark = verify.judge([_claim(100.0)], v, t)["x.0"]
        assert mark.status == status
        if status == "moved":
            assert mark.evidence is not None

    def test_missing_answer_is_unchecked_not_confirmed(self) -> None:
        v = Verdicts(items=[Verdict(n=2, verdict="да", at="", note="")])
        marks = verify.judge([_claim(10, "a.0"), _claim(20, "b.0")], v, _transcript())
        assert marks["a.0"].status == "unchecked" and marks["b.0"].status == "ok"


class TestReport:
    def test_describe(self) -> None:
        report = Report(
            marks={
                "a": Mark(status="ok"),
                "b": Mark(status="ok"),
                "c": Mark(status="moved", evidence=5),
                "d": Mark(status="unsupported"),
                "e": Mark(status="no_timecode"),
            }
        )
        assert report.confirmed == 3
        assert report.describe() == (
            "3 из 5 утверждений подтверждены транскриптом; 1 — таймкод ведёт не туда; "
            "1 — нет в записи; 1 — без таймкода"
        )
        assert Report().describe() == "утверждений нет"


class Judge(EchoLLM):
    """Отвечает документом на вопрос саммари и вердиктами — на вопрос проверки.

    Вердикт «нет» — утверждениям со словом «выдумка», остальным «да» по их же таймкоду.
    `answers` — сколько первых утверждений судья успевает назвать до конца ответа."""

    def __init__(
        self, context: int = 1_000_000, doc: LectureDoc | None = None, answers: int = 10**6
    ) -> None:
        super().__init__(context)
        self.doc = doc or _doc()
        self.answers = answers
        self.calls: list[dict[str, Any]] = []
        self.closed = 0

    def close(self) -> None:
        self.closed += 1

    def complete(self, **kw: Any) -> LLMResult:
        self.calls.append(kw)
        if "items" not in kw["schema"]["properties"]:
            text = self.doc.model_dump_json()
        else:
            items = []
            for line in kw["question"].splitlines():
                m = re.match(r"(\d+)\. \[(\S+)\] (.*)", line)
                if m:
                    lie = "выдумка" in m[3]
                    items.append(
                        {
                            "n": int(m[1]),
                            "verdict": "нет" if lie else "да",
                            "at": "" if lie else m[2],
                            "note": "не говорил" if lie else "",
                        }
                    )
            text = json.dumps({"items": items[: self.answers]}, ensure_ascii=False)
        return LLMResult(
            text=text, model="judge", usage=TokenUsage(input=100, output=10), cost_usd=0.001
        )


class TestBuild:
    def test_single_request_reuses_summary_prefix(self, tmp_path: Path) -> None:
        t = _transcript()
        doc = _doc()
        doc.tldr.append(Claim(text="Это выдумка модели", at="00:00:20"))
        llm = Judge(doc=doc)
        stages.build_verify(tmp_path, llm, t, doc)
        (call,) = llm.calls
        assert call["prefix"] == stages.transcript_prefix(t), "префикс тот же — кэш тот же"
        assert call["system"].startswith(verify.SYSTEM)
        report, result = stages.load_verify(tmp_path)
        assert report.marks["tldr.1"].status == "unsupported"
        assert report.marks["tldr.1"].note == "не говорил"
        assert report.marks["tldr.0"].status == "ok"
        assert list(report.marks)[:2] == ["tldr.0", "tldr.1"], "порядок — как в документе"
        assert result is not None and result.cost_usd == 0.001

    def test_silent_judge_is_asked_again_about_the_rest(self, tmp_path: Path) -> None:
        t = _transcript()
        llm = Judge(answers=2)
        stages.build_verify(tmp_path, llm, t, _doc())
        assert len(llm.calls) == 2 and "1. [00:00:10]" in llm.calls[0]["question"]
        again = llm.calls[1]["question"]
        assert "Память освобождает" not in again and "1. [00:02:00] GC" in again, (
            "второй раз — только неотвеченные, нумерация заново"
        )
        report, result = stages.load_verify(tmp_path)
        assert report.count("unchecked") == 0, "4 утверждения: 2 + 2 за два захода"
        assert result is not None and result.cost_usd == pytest.approx(0.002)

    def test_judge_that_never_answers_leaves_unchecked(self, tmp_path: Path) -> None:
        llm = Judge(answers=0)
        stages.build_verify(tmp_path, llm, _transcript(), _doc())
        assert len(llm.calls) == 1 + verify.RETRIES
        report, _ = stages.load_verify(tmp_path)
        assert report.count("unchecked") == 4 and report.confirmed == 0

    def test_no_claims_no_request(self, tmp_path: Path) -> None:
        llm = Judge()
        stages.build_verify(tmp_path, llm, _transcript(), RawDoc(blocks=[]))
        assert llm.calls == []
        report, result = stages.load_verify(tmp_path)
        assert report.marks == {} and result is None

    def test_long_transcript_checked_by_windows(self, tmp_path: Path) -> None:
        t = _transcript(n=2000, step=5.0)  # ~2.8 часа, ~23k токенов
        theses = [Claim(text=f"тезис {i}", at=f"{i:02d}:{i:02d}:00") for i in range(3)]
        doc = LectureDoc(
            title="",
            tldr=theses,
            sections=[],
            glossary=[],
            examples=[],
            qa=[],
            reading=[],
            unclear=[],
        )
        llm = Judge(context=8192, doc=doc)
        stages.build_verify(tmp_path, llm, t, doc)
        assert len(llm.calls) >= 2, "транскрипт не влез — окнами"
        asked = []
        for call in llm.calls:
            for line in call["question"].splitlines():
                if m := re.match(r"\d+\. (\[\S+\]) (.*)", line):
                    assert m[1] in call["prefix"], "утверждение проверяется в своём окне"
                    asked.append(m[2])
        assert sorted(asked) == ["тезис 0", "тезис 1", "тезис 2"], "каждое — ровно один раз"
        report, _ = stages.load_verify(tmp_path)
        assert report.confirmed == 3


def _render(tmp_path: Path, report: Report | None, *, debug: bool = False) -> str:
    t = _transcript()
    layout = NoteLayout(tmp_path / "n.md", tmp_path / "Записи", "2026-10-29 лекция")
    audio = AudioInfo(
        source=Path("/x/лекция.mp4"),
        wav=Path("a.wav"),
        duration=t.duration,
        source_codec="opus",
        source_channels=1,
        source_sample_rate=48000,
        audio_streams=1,
    )
    path = render_note(
        layout,
        audio=audio,
        transcript=t,
        summary=LLMResult(text="", model="m", usage=TokenUsage(input=0, output=0)),
        document=_doc(),
        profile="lecture",
        media="a.ogg",
        debug=debug,
        report=report,
    )
    return path.read_text()


class TestRender:
    REPORT = Report(
        marks={
            "tldr.0": Mark(status="ok"),
            "sections.0.theses.0": Mark(status="unsupported", note="лектор этого не говорил"),
            "sections.0.theses.1": Mark(status="no_timecode"),
            "glossary.0": Mark(status="moved", evidence=250.0),
            "qa.0": Mark(status="partial", note="ответа не было"),
            "unclear.0": Mark(status="unchecked"),
        }
    )

    def test_marks_after_timecodes(self, tmp_path: Path) -> None:
        note = _render(tmp_path, self.REPORT)
        assert "Память освобождает сборщик [[a.ogg#t=10.0|▶ 00:10]]\n" in note, "ok — без пометки"
        assert "[[a.ogg#t=60.0|▶ 01:00]] ❌ *нет в записи: лектор этого не говорил*" in note
        assert "сборщик мусора [[a.ogg#t=120.0|▶ 02:00]] ↪ *сказано в* [[a.ogg#t=250.0|" in note
        assert "◐ *частично: ответа не было*" in note
        assert "❔ *не проверено*" in note
        assert 'verified: "2/6"' in note.split("---")[1]
        assert "> [!warning] Проверено по записи\n> 2 из 6 утверждений подтверждены" in note

    def test_all_confirmed_is_success(self, tmp_path: Path) -> None:
        note = _render(tmp_path, Report(marks={"tldr.0": Mark(status="ok")}))
        assert "> [!success] Проверено по записи" in note

    def test_debug_has_full_description_in_frontmatter(self, tmp_path: Path) -> None:
        note = _render(tmp_path, self.REPORT, debug=True)
        head = note.split("---")[1]
        assert 'verified: "2 из 6 утверждений' in head
        assert "❌ *нет в записи" in note and "Проверено по записи" not in note

    def test_without_report_note_is_unchanged(self, tmp_path: Path) -> None:
        note = _render(tmp_path, None)
        assert "verified" not in note and "❌" not in note and "Проверено" not in note


@pytest.fixture
def lecture(tmp_path: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("нужен ffmpeg")
    path = tmp_path / "лекция.opus"
    shutil.copy(SPEECH_CLIPS["speech"], path)
    return path


class TestPipeline:
    def _run(
        self, lecture: Path, judges: list[Judge], monkeypatch: pytest.MonkeyPatch, **env: str
    ) -> Any:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        transcript = _transcript()

        class FixedASR:
            name = "fixed"

            def transcribe(self, audio: Path, **kw: object) -> Transcript:
                return transcript.model_copy(update={"source": audio})

        def make_asr(cfg: ASRSettings) -> FixedASR:
            return FixedASR()

        def make_llm(cfg: LLMSettings) -> Judge:
            judges.append(Judge())
            return judges[-1]

        with Pipeline(Settings(), asr_factory=make_asr, llm_factory=make_llm) as p:  # type: ignore[arg-type]
            return p.run(lecture)

    def test_one_model_for_summary_and_verify(
        self, lecture: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        judges: list[Judge] = []
        result = self._run(lecture, judges, monkeypatch)
        (judge,) = judges
        assert len(judge.calls) == 2 and judge.closed == 1, "своя модель не поднимается дважды"
        assert result.verification is not None and result.verification.confirmed == 4
        assert "verified:" in result.note.read_text()
        assert result.cached["verify"] is False

    def test_verify_added_later_reuses_cached_summary(
        self, lecture: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        judges: list[Judge] = []
        off = self._run(lecture, judges, monkeypatch, SUMMURAZE_VERIFY_ENABLED="false")
        assert off.verification is None and "verify" not in off.cached
        assert "verified" not in off.note.read_text()
        on = self._run(lecture, judges, monkeypatch, SUMMURAZE_VERIFY_ENABLED="true")
        assert on.cached["summarize"] is True and on.cached["verify"] is False
        assert len(judges) == 2 and len(judges[1].calls) == 1, "только запрос проверки"
        again = self._run(lecture, judges, monkeypatch)
        assert again.cached["verify"] is True and len(judges) == 2, "модель не создаётся"
        assert again.verification == on.verification

    def test_failed_verify_keeps_the_note_and_retries_next_time(
        self, lecture: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from summuraze.summarize.openai_compat import LLMError

        class Broken(Judge):
            def complete(self, **kw: Any) -> LLMResult:
                if "items" in kw["schema"]["properties"]:
                    raise LLMError("провайдер отверг схему ответа")
                return super().complete(**kw)

        judges: list[Judge] = []
        events: list[str] = []

        def make_llm(cfg: LLMSettings) -> Judge:
            judges.append(Broken())
            return judges[-1]

        transcript = _transcript()

        class FixedASR:
            name = "fixed"

            def transcribe(self, audio: Path, **kw: object) -> Transcript:
                return transcript.model_copy(update={"source": audio})

        with Pipeline(
            Settings(),
            asr_factory=lambda cfg: FixedASR(),  # type: ignore[arg-type,return-value]
            llm_factory=make_llm,
            on_event=lambda e: events.append(f"{e.stage}:{e.status}"),
        ) as p:
            result = p.run(lecture)
            assert result.verification is None
            assert result.verification_error == "провайдер отверг схему ответа"
            assert "verify:failed" in events and "render:done" in events
            assert result.note is not None and "verified" not in result.note.read_text()
            assert judges[0].closed == 1, "модель погашена и после сбоя"
            again = p.run(lecture)
        assert again.cached is not None and again.cached["summarize"] is True
        assert len(judges) == 2, "сбой не в кэше: модель поднята снова — ради проверки"


def test_cli_reports_failed_verify(capsys: pytest.CaptureFixture[str]) -> None:
    from summuraze import cli
    from summuraze.pipeline.orchestrator import StageEvent

    progress = cli._StageProgress()
    progress(StageEvent("verify", 5, 7, "start"))
    progress(StageEvent("verify", 5, 7, "failed", 3.0))
    assert capsys.readouterr().out == "[5/7] проверка… не удалась\n"


class TestParse:
    FULL = (
        '{"items": [{"n": 1, "at": "00:01:00", "verdict": "да", "note": ""}, '
        '{"n": 2, "at": "", "verdict": "нет", "note": "не говорил"}]}'
    )

    def test_full(self) -> None:
        v = verify.parse(self.FULL)
        assert v is not None and [i.n for i in v.items] == [1, 2]

    def test_truncated_keeps_whole_verdicts(self) -> None:
        cut = self.FULL[: self.FULL.index('"нет"')]
        v = verify.parse(cut)
        assert v is not None and [i.n for i in v.items] == [1]

    @pytest.mark.parametrize("text", ["", "Эхо-саммари", '{"items": [', '{"items": [{"n": "x"}]}'])
    def test_nothing_usable(self, text: str) -> None:
        assert verify.parse(text) is None


def test_truncated_judge_is_asked_about_the_rest(tmp_path: Path) -> None:
    class Cut(Judge):
        def complete(self, **kw: Any) -> LLMResult:
            result = super().complete(**kw)
            if len(self.calls) == 1:  # первый ответ обрезан лимитом посреди второго вердикта
                text = result.text
                return result.model_copy(update={"text": text[: text.index('"n": 2') + 4]})
            return result

    llm = Cut()
    stages.build_verify(tmp_path, llm, _transcript(), _doc())
    report, _ = stages.load_verify(tmp_path)
    assert len(llm.calls) == 2 and report.count("unchecked") == 0


def test_judge_off_schema_fails_cleanly(tmp_path: Path) -> None:
    calls: list[int] = []

    def reply(**kw: Any) -> LLMResult:
        calls.append(1)
        return LLMResult(text="не JSON", model="m", usage=TokenUsage(input=1, output=1))

    llm = EchoLLM()
    llm.complete = reply  # type: ignore[method-assign]
    with pytest.raises(stages.SummaryFormatError, match="раза ответила не по схеме проверки"):
        stages.build_verify(tmp_path, llm, _transcript(), _doc())
    assert len(calls) == 1 + verify.RETRIES
    assert not list(tmp_path.iterdir())


def test_empty_judge_answer_is_asked_again(tmp_path: Path) -> None:
    """DeepSeek-судья ответил пустой строкой (#63, 2026-11-02) — переспросить, а не падать."""

    class Blank(Judge):
        def complete(self, **kw: Any) -> LLMResult:
            result = super().complete(**kw)
            return result.model_copy(update={"text": ""}) if len(self.calls) == 1 else result

    llm = Blank()
    stages.build_verify(tmp_path, llm, _transcript(), _doc())
    report, result = stages.load_verify(tmp_path)
    assert len(llm.calls) == 2 and report.count("unchecked") == 0
    assert result is not None
