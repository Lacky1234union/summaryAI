"""Эталон замера нельзя незаметно подменить пустым или смещённым RTTM."""

import runpy
from pathlib import Path

import pytest

read_rttm = runpy.run_path(str(Path(__file__).parents[1] / "eval" / "diarization.py"))["read_rttm"]


def test_reference_keeps_overlap_and_clips_to_audio(tmp_path: Path) -> None:
    reference = tmp_path / "ref.rttm"
    reference.write_text(
        "SPEAKER meeting 1 0.5 2.0 <NA> <NA> A <NA> <NA>\n"
        "SPEAKER meeting 1 1.0 4.0 <NA> <NA> B <NA> <NA>\n"
        "SPEAKER another 1 0 1 <NA> <NA> C <NA> <NA>\n"
        "SPEAKER meeting 1 7 1 <NA> <NA> C <NA> <NA>\n"
    )
    assert read_rttm(reference, "meeting", 3.0) == [(0.5, 2.5, "A"), (1.0, 3.0, "B")]


@pytest.mark.parametrize("start,length", [("-1", "1"), ("0", "0"), ("nan", "1"), ("0", "inf")])
def test_invalid_reference_interval(tmp_path: Path, start: str, length: str) -> None:
    reference = tmp_path / "ref.rttm"
    reference.write_text(f"SPEAKER meeting 1 {start} {length} <NA> <NA> A <NA> <NA>\n")
    with pytest.raises(ValueError, match="неверный интервал"):
        read_rttm(reference, "meeting", 3.0)


def test_wrong_recording_is_not_silently_scored_as_silence(tmp_path: Path) -> None:
    reference = tmp_path / "ref.rttm"
    reference.write_text("SPEAKER another 1 0 1 <NA> <NA> A <NA> <NA>\n")
    with pytest.raises(ValueError, match="нет эталонной речи"):
        read_rttm(reference, "meeting", 3.0)
