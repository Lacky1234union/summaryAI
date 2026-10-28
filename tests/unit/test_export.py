"""Экспорт транскрипта: JSON, TXT, SRT/VTT (#14)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from summuraze.cli import app
from summuraze.render.export import PARAGRAPH_MAX, export, parse_formats, to_json, to_txt
from summuraze.render.subtitles import (
    LINE_CHARS,
    MAX_SECONDS,
    MIN_SECONDS,
    cues,
    srt_time,
    to_srt,
    to_vtt,
)
from summuraze.types import Segment, Transcript, Word
from tests.conftest import SPEECH_CLIPS

TEXT = (
    "Сегодня мы разберём дискретное преобразование Фурье. Начнём с того, зачем оно вообще "
    "нужно инженеру, который работает с сигналами, и почему без него не обойтись. Затем "
    "перейдём к быстрому алгоритму и посчитаем его сложность на доске."
)
LOREM = TEXT.split()


def _worded(start: float, words: list[str], step: float = 0.6, suspect: bool = False) -> Segment:
    ws = [
        Word(start=start + i * step, end=start + i * step + step * 0.8, text=f" {w}", prob=0.9)
        for i, w in enumerate(words)
    ]
    return Segment(start=start, end=ws[-1].end, text=" ".join(words), words=ws, suspect=suspect)


def _t(*segments: Segment, duration: float = 600) -> Transcript:
    return Transcript(
        source=Path("/x/лекция.mp4"),
        duration=duration,
        language="ru",
        asr_model="m",
        segments=list(segments),
    )


class TestCues:
    def test_long_segment_is_split_into_readable_cues(self) -> None:
        seg = _worded(10.0, LOREM)
        got = cues(_t(seg))
        assert len(got) > 3
        for c in got:
            lines = c.text.split("\n")
            assert len(lines) <= 2
            assert all(len(line) <= LINE_CHARS + 8 for line in lines), c.text
            assert c.end - c.start <= MAX_SECONDS + 1e-9, c
            assert seg.start <= c.start < c.end <= seg.end + MIN_SECONDS
        assert all(a.end <= b.start for a, b in zip(got, got[1:], strict=False)), "без наложений"
        assert " ".join(c.text.replace("\n", " ") for c in got) == " ".join(LOREM), (
            "ни слова не потеряно"
        )
        assert got[0].text.replace("\n", " ").endswith("Фурье."), "режем на конце фразы"

    def test_cues_follow_word_timecodes(self) -> None:
        seg = _worded(100.0, LOREM)
        first, second = cues(_t(seg))[:2]
        assert first.start == 100.0
        n = len(first.text.split())
        assert second.start == pytest.approx(seg.words[n].start)

    def test_suspect_mark_on_every_cue(self) -> None:
        got = cues(_t(_worded(0, LOREM, suspect=True)))
        assert len(got) > 1 and all(c.text.startswith("[?] ") for c in got)
        assert all(len(c.text) <= 2 * LINE_CHARS + 1 for c in got), "пометка в лимите"

    def test_without_word_timecodes_time_is_shared_by_length(self) -> None:
        seg = Segment(start=0.0, end=30.0, text=" ".join(LOREM))
        got = cues(_t(seg))
        assert len(got) > 3 and got[0].start == 0.0
        assert got[-1].end == pytest.approx(30.0)
        assert all(a.end <= b.start for a, b in zip(got, got[1:], strict=False))

    def test_short_cue_is_extended_but_does_not_overlap(self) -> None:
        a = Segment(start=0.0, end=0.2, text="Да.")
        b = Segment(start=0.5, end=1.5, text="Хорошо.")
        c = Segment(start=5.0, end=5.1, text="Итак.")
        got = cues(_t(a, b, c))
        assert (got[0].start, got[0].end) == (0.0, 0.5), "до следующей реплики, не дальше"
        assert got[2].end == pytest.approx(5.0 + MIN_SECONDS)

    def test_wrap_balances_lines(self) -> None:
        text = cues(_t(_worded(0, LOREM)))[1].text
        top, bottom = text.split("\n")
        assert abs(len(top) - len(bottom)) < 15

    def test_vtt(self) -> None:
        vtt = to_vtt(_t(Segment(start=61.5, end=63.0, text="Привет.")))
        assert vtt == "WEBVTT\n\n00:01:01.500 --> 00:01:03.000\nПривет.\n"


class TestTxt:
    def test_paragraphs_by_pause_with_timecodes(self) -> None:
        t = _t(
            Segment(start=1.0, end=3.0, text=" Первая фраза."),
            Segment(start=3.2, end=5.0, text="Вторая фраза."),
            Segment(start=9.0, end=10.0, text="После паузы.", suspect=True),
            Segment(start=10.0, end=10.0, text="  "),
            Segment(start=3700.0, end=3702.0, text="Через час."),
        )
        assert to_txt(t) == (
            "[00:01] Первая фраза. Вторая фраза.\n\n"
            "[00:09] [?] После паузы.\n\n"
            "[1:01:40] Через час.\n"
        )

    def test_long_monologue_is_cut_at_sentence_end(self) -> None:
        segs = [
            Segment(
                start=i * 5.0,
                end=i * 5.0 + 4.9,
                text=f"Фраза номер {i} про сигналы и спектры" + ("." if i % 3 == 2 else ","),
            )
            for i in range(60)
        ]
        paras = to_txt(_t(*segs)).strip().split("\n\n")
        assert len(paras) > 2
        assert all(p.endswith(".") for p in paras)

    def test_unpunctuated_speech_is_cut_by_length(self) -> None:
        segs = [
            Segment(start=i * 5.0, end=i * 5.0 + 4.9, text=f"фраза номер {i} про сигналы и спектры")
            for i in range(100)
        ]
        paras = to_txt(_t(*segs)).strip().split("\n\n")
        assert len(paras) > 2
        assert all(len(p) <= PARAGRAPH_MAX + 60 for p in paras)

    def test_empty(self) -> None:
        assert to_txt(_t()) == ""


def test_json_round_trip() -> None:
    t = _t(_worded(3.0, LOREM[:5], suspect=True))
    text = to_json(t)
    assert "Сегодня" in text, "кириллица как есть, не \\u-эскейпы"
    assert Transcript.model_validate_json(text) == t


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("json,txt,srt", ["json", "txt", "srt"]),
        (" SRT, .vtt ,srt", ["srt", "vtt"]),
        ("all", ["json", "txt", "srt", "vtt"]),
    ],
)
def test_parse_formats(text: str, expected: list[str]) -> None:
    assert parse_formats(text) == expected


@pytest.mark.parametrize("text", ["docx", "", " , "])
def test_parse_formats_rejects(text: str) -> None:
    with pytest.raises(ValueError, match="есть json"):
        parse_formats(text)


def test_export_names(tmp_path: Path) -> None:
    src = tmp_path / "Лекция 3.mp4"
    written = export(_t(_worded(0, LOREM)), src, ["srt", "txt"], tmp_path / "out" / "x")
    assert [p.name for p in written] == ["Лекция 3.srt", "Лекция 3.txt"]
    assert all(p.parent == tmp_path / "out" / "x" for p in written)


@pytest.fixture
def lecture(tmp_path: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("нужен ffmpeg")
    path = tmp_path / "rec" / "лекция.opus"
    path.parent.mkdir()
    shutil.copy(SPEECH_CLIPS["speech"], path)
    return path


class TestCLI:
    def test_transcribe_writes_files_next_to_recording(self, lecture: Path) -> None:
        result = CliRunner().invoke(app, ["transcribe", str(lecture)])
        assert result.exit_code == 0, result.output
        assert "лекция.json, лекция.txt, лекция.srt" in result.output
        files = sorted(p.name for p in lecture.parent.iterdir())
        assert files == ["лекция.json", "лекция.opus", "лекция.srt", "лекция.txt"]
        t = Transcript.model_validate_json(lecture.with_suffix(".json").read_text("utf-8"))
        assert lecture.with_suffix(".srt").read_text("utf-8") == to_srt(t)

    def test_format_and_out(self, lecture: Path, tmp_path: Path) -> None:
        out = tmp_path / "subs"
        result = CliRunner().invoke(app, ["transcribe", str(lecture), "-f", "vtt", "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert [p.name for p in out.iterdir()] == ["лекция.vtt"]
        assert sorted(p.name for p in lecture.parent.iterdir()) == ["лекция.opus"]

    def test_bad_format_fails_before_transcribing(self, lecture: Path) -> None:
        result = CliRunner().invoke(app, ["transcribe", str(lecture), "-f", "docx"])
        assert result.exit_code == 2 and "неизвестный формат docx" in result.output
        assert "[1/" not in result.output


def test_player_reads_srt_with_same_timecodes(tmp_path: Path) -> None:
    """ffmpeg — тот же демуксер, что у mpv: прочёл реплики с нашими таймкодами — прочтёт и плеер."""
    if shutil.which("ffprobe") is None:
        pytest.skip("нужен ffprobe")
    t = _t(
        _worded(0.5, LOREM[:12]),
        _worded(3725.25, LOREM[12:], suspect=True),
    )
    srt = tmp_path / "x.srt"
    srt.write_text(to_srt(t), encoding="utf-8")
    probe = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-show_entries", "packet=pts_time,duration_time"),
            *("-of", "csv=p=0", str(srt)),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    got = [tuple(float(x) for x in line.split(",")) for line in probe]
    expected = [(c.start, c.end - c.start) for c in cues(t)]
    assert len(got) == len(expected)
    for (start, dur), (exp_start, exp_dur) in zip(got, expected, strict=True):
        assert start == pytest.approx(exp_start, abs=1e-3)
        assert dur == pytest.approx(exp_dur, abs=1e-3)
    assert srt_time(expected[-1][0]).startswith("01:02:")
