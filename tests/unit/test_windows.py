"""Окна с перекрытием и двухпроходная схема (#21)."""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from summuraze.config import ASRSettings, LLMSettings, Settings
from summuraze.pipeline import stages
from summuraze.pipeline.orchestrator import Pipeline
from summuraze.render.markdown import fmt_timecode
from summuraze.summarize.echo import EchoLLM
from summuraze.summarize.profiles import Claim, LectureDoc, get_profile
from summuraze.summarize.windows import (
    Thread,
    Topic,
    Window,
    WindowMap,
    map_text,
    merge_maps,
    split,
)
from summuraze.types import LLMResult, Segment, TokenUsage, Transcript
from tests.conftest import SPEECH_CLIPS

HOUR = 3600.0


def _segments(n: int, *, step: float = 5.0, punct: bool = True) -> list[Segment]:
    out = []
    for i in range(n):
        end = "." if punct and i % 4 == 3 else ""
        out.append(Segment(start=i * step, end=i * step + step - 0.5, text=f"фраза {i}{end}"))
    return out


def _transcript(segments: list[Segment]) -> Transcript:
    return Transcript(
        source=Path("/x/лекция.mp4"),
        duration=segments[-1].end + 1,
        language="ru",
        asr_model="m",
        segments=segments,
    )


class TestSplit:
    def test_windows_cover_everything_in_order_under_limit(self) -> None:
        segs = _segments(200)
        tokens = [10] * 200
        windows = split(segs, tokens, limit=300)
        assert len(windows) > 5
        assert windows[0].segments[0] is segs[0] and windows[-1].segments[-1] is segs[-1]
        for w in windows:
            assert sum(10 for _ in w.segments) <= 300
        # ядра — разбиение записи без дыр и наложений
        assert windows[0].core_start == 0 and windows[-1].core_end == math.inf
        for a, b in zip(windows, windows[1:], strict=False):
            assert a.core_end == b.core_start
            assert b.start < a.end, "перекрытие есть"
            overlap = [s for s in b.segments if s.start < b.core_start]
            assert 0 < len(overlap) * 10 <= 60, "перекрытие 10–20 % окна"

    def test_cut_after_sentence_end(self) -> None:
        windows = split(_segments(200), [10] * 200, limit=300)
        for w in windows[:-1]:
            core_last = [s for s in w.segments if s.start < w.core_end][-1]
            assert core_last.text.endswith("."), "разрез — на конце предложения"
            assert w.segments[-1] is core_last, "окно кончается там же, где ядро"

    def test_without_punctuation_cut_at_longest_pause(self) -> None:
        segs = _segments(100, punct=False)
        pause_at = 26  # длинная пауза перед сегментом 26 — в зоне разреза первого окна
        segs = [
            s.model_copy(update={"start": s.start + 30, "end": s.end + 30}) if i >= pause_at else s
            for i, s in enumerate(segs)
        ]
        first = split(segs, [10] * 100, limit=280)[0]
        assert first.segments[-1] is segs[pause_at - 1]

    def test_overlap_starts_at_sentence_start(self) -> None:
        windows = split(_segments(200), [10] * 200, limit=300)
        for w in windows[1:]:
            i = int(w.segments[0].text.split()[1].rstrip("."))
            assert i % 4 == 0, "перекрытие начинается с начала фразы"

    def test_oversized_line_still_progresses(self) -> None:
        segs = _segments(10)
        tokens = [10, 10, 500, 10, 10, 10, 10, 10, 10, 10]
        windows = split(segs, tokens, limit=100)
        covered = {s.text for w in windows for s in w.segments}
        assert covered == {s.text for s in segs}
        assert any(w.segments == [segs[2]] for w in windows)

    def test_tokens_must_match_segments(self) -> None:
        with pytest.raises(ValueError):
            split(_segments(3), [1, 2], limit=10)


def _resolve(at: str) -> float | None:
    m = re.fullmatch(r"(\d\d):(\d\d):(\d\d)", at)
    return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) if m else None


class TestMerge:
    def test_overlap_gives_no_duplicates_and_topic_continues(self) -> None:
        a = Window(_segments(2), core_start=0, core_end=600)
        b = Window(_segments(2), core_start=600, core_end=math.inf)
        claim = Claim(text="сборщик мусора трёхцветный", at="00:09:50")
        map_a = WindowMap(
            topics=[Topic(title="Сборка мусора", at="00:05:00", points=[claim])],
            threads=[Thread(text="зачем барьер записи — позже", at="00:09:00")],
        )
        map_b = WindowMap(
            topics=[
                Topic(  # перекрытие: та же мысль из начала окна b
                    title="сборка мусора",
                    at="00:09:40",
                    points=[claim, Claim(text="барьер записи нужен для…", at="00:12:00")],
                ),
                Topic(title="Паузы", at="00:15:00", points=[Claim(text="p", at="??")]),
            ],
            threads=[Thread(text="повтор нити", at="00:09:00")],
        )
        merged = merge_maps([(a, map_a), (b, map_b)], _resolve)
        assert [t.title for t in merged.topics] == ["Сборка мусора", "Паузы"]
        assert [p.text for p in merged.topics[0].points] == [
            "сборщик мусора трёхцветный",
            "барьер записи нужен для…",
        ]
        assert merged.topics[1].points[0].at == "??", "нечитаемый таймкод не теряем"
        assert [t.text for t in merged.threads] == ["зачем барьер записи — позже"]

    def test_map_text(self) -> None:
        wmap = WindowMap(
            topics=[Topic(title="Т", at="[00:01:00]", points=[Claim(text="у", at="00:01:05")])],
            threads=[Thread(text="н", at="00:02:00")],
        )
        assert map_text(wmap) == (
            "Карта лекции:\n\n## [00:01:00] Т\n- [00:01:05] у\n\n## Открытые нити\n- [00:02:00] н"
        )


class Scripted(EchoLLM):
    """Модель, которая «конспектирует» честно: в карту — строки с ФАКТ, в итог — всё из карты.

    Итог связывает начало с концом, только если ему передали карту всей записи: это и проверяем.
    """

    def __init__(self, context: int) -> None:
        super().__init__(context)
        self.sent: list[dict[str, Any]] = []

    def complete(self, **kw: Any) -> LLMResult:
        self.sent.append(kw)
        if kw["question"].startswith("Составь карту"):
            points = [
                {"text": line.split("] ", 1)[1], "at": line[1:9]}
                for line in kw["prefix"].splitlines()
                if "ФАКТ" in line
            ]
            body: dict[str, Any] = {
                "topics": [{"title": "Тема", "at": kw["prefix"][1:9], "points": points}],
                "threads": [],
            }
        else:
            claims = re.findall(r"- \[(\d\d:\d\d:\d\d)\] (.*ФАКТ.*)", kw["system"] + kw["prefix"])
            body = {
                "title": "Лекция",
                "tldr": [{"text": text, "at": at} for at, text in dict.fromkeys(claims)],
                "sections": [],
                "glossary": [],
                "examples": [],
                "qa": [],
                "reading": [],
                "unclear": [],
            }
        text = json.dumps(body, ensure_ascii=False)
        return LLMResult(
            text=text, model="scripted", usage=TokenUsage(input=100, output=10), cost_usd=0.001
        )


def _long_lecture() -> Transcript:
    """Час лекции: в начале задан вопрос, в конце лектор на него отвечает — и передумывает."""
    segs = []
    for i in range(720):
        text = f"говорим про сборку мусора, мысль {i}."
        if i == 3:
            text = "ФАКТ: вопрос — нужен ли барьер записи при параллельной сборке?"
        if i == 715:
            text = "ФАКТ: возвращаясь к вопросу из начала — барьер нужен, я ошибался."
        segs.append(Segment(start=i * 5.0, end=i * 5.0 + 4.5, text=text))
    return _transcript(segs)


class TestTwoPass:
    def test_fact_from_first_window_meets_last_one_in_final(self, tmp_path: Path) -> None:
        llm = Scripted(8192)
        t = _long_lecture()
        plan = stages.build_summary(tmp_path, llm, t, get_profile("lecture"))
        assert plan.windows > 2 and plan.final == "map"
        *maps, final = llm.sent
        assert len(maps) == plan.windows
        assert "ФАКТ: вопрос" in maps[0]["prefix"] and "ФАКТ: возвращаясь" in maps[-1]["prefix"]
        assert "ФАКТ: вопрос" in final["prefix"] and "ФАКТ: возвращаясь" in final["prefix"], (
            "второй проход видит начало и конец записи разом"
        )
        payload = json.loads((tmp_path / "summary.json").read_text("utf-8"))
        doc = LectureDoc.model_validate(payload["document"])
        assert [c.at for c in doc.tldr] == [fmt_timecode(15), fmt_timecode(3575)]
        assert payload["mode"] == "windows" and payload["final"] == "map"
        assert payload["result"]["usage"]["input"] == 100 * (plan.windows + 1)
        assert payload["result"]["cost_usd"] == pytest.approx(0.001 * (plan.windows + 1))

    def test_fact_in_overlap_is_mapped_once(self, tmp_path: Path) -> None:
        llm = Scripted(8192)
        segs = [
            Segment(
                start=i * 5.0,
                end=i * 5.0 + 4.5,
                text=f"ФАКТ номер {i}." if i % 6 == 0 else f"рассуждение {i}.",
            )
            for i in range(720)
        ]
        stages.build_summary(tmp_path, llm, _transcript(segs), get_profile("lecture"))
        final = llm.sent[-1]["prefix"]
        stamps = re.findall(r"- \[(\d\d:\d\d:\d\d)\] ФАКТ", final)
        assert len(stamps) == 120 and len(set(stamps)) == 120, "каждый факт ровно один раз"

    def test_single_pass_max_keeps_transcript_in_final(self, tmp_path: Path) -> None:
        llm = Scripted(1_000_000)
        t = _long_lecture()
        plan = stages.build_summary(tmp_path, llm, t, get_profile("lecture"), single_pass_max=6000)
        assert plan.windows > 1 and plan.final == "transcript"
        final = llm.sent[-1]
        assert final["prefix"] == stages.transcript_prefix(t), "транскрипт первым, байт в байт"
        assert "Карта лекции:" in final["system"]
        assert "по карте и полному транскрипту" in plan.describe()

    def test_raw_profile_merges_blocks_without_second_pass(self, tmp_path: Path) -> None:
        class Blocks(EchoLLM):
            calls = 0

            def complete(self, **kw: Any) -> LLMResult:
                Blocks.calls += 1
                first = kw["prefix"].splitlines()[0][1:9]
                blocks = [{"title": f"блок {first}", "at": first}]
                text = json.dumps({"blocks": blocks})
                return LLMResult(text=text, model="b", usage=TokenUsage(input=1, output=1))

        plan = stages.build_summary(tmp_path, Blocks(8192), _long_lecture(), get_profile("raw"))
        assert plan.final == "merge" and Blocks.calls == plan.windows
        payload = json.loads((tmp_path / "summary.json").read_text("utf-8"))
        # Блок каждого следующего окна начат в перекрытии — это ядро предыдущего окна, дубль.
        assert len(payload["document"]["blocks"]) == 1

    def test_map_reply_off_schema_fails_cleanly(self, tmp_path: Path) -> None:
        class Broken(EchoLLM):
            def complete(self, **kw: Any) -> LLMResult:
                return LLMResult(text="не json", model="x", usage=TokenUsage(input=1, output=1))

        with pytest.raises(stages.SummaryFormatError, match="карты окна 1"):
            stages.build_summary(tmp_path, Broken(8192), _long_lecture(), get_profile("lecture"))
        assert not (tmp_path / "summary.json").exists()

    def test_window_smaller_than_prompt_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(stages.budget.TranscriptTooLong, match="на окно остаётся"):
            stages.build_summary(
                tmp_path, EchoLLM(), _long_lecture(), get_profile("lecture"), single_pass_max=900
            )


def test_pipeline_runs_long_recording_in_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Сквозной прогон: длинный транскрипт, контекст 8K — окна, конспект, строка о режиме."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("нужен ffmpeg")
    source = tmp_path / "лекция.opus"
    shutil.copy(SPEECH_CLIPS["speech"], source)
    monkeypatch.setenv("SUMMURAZE_LLM_CONTEXT", "8192")
    long = _long_lecture()

    class LongASR:
        name = "long"

        def transcribe(self, audio: Path, **kw: object) -> Transcript:
            return long.model_copy(update={"source": audio})

    def make_asr(cfg: ASRSettings) -> LongASR:
        return LongASR()

    def make_llm(cfg: LLMSettings) -> EchoLLM:
        return EchoLLM(cfg.context)

    with Pipeline(Settings(), asr_factory=make_asr, llm_factory=make_llm) as p:  # type: ignore[arg-type]
        result = p.run(source, until="summarize")
    assert result.budget is not None and result.budget.windows > 2
    assert result.budget.describe().startswith("окнами — ")
