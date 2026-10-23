"""Ингест: форматы, видео, несколько дорожек, понятные ошибки.

Медиа генерируется ffmpeg'ом из lavfi прямо в тесте: в git бинарные записи не кладём, а
синтетика здесь честная — проверяется конвертация, а не распознавание.
"""

from __future__ import annotations

import array
import shutil
import subprocess
import wave
from pathlib import Path

import pytest
from typer.testing import CliRunner

from summuraze.audio.ingest import SAMPLE_RATE, IngestError, ingest
from summuraze.cli import app

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")

DURATION = 3.0
SINE = f"sine=frequency=440:sample_rate=44100:duration={DURATION}"
SILENCE = f"anullsrc=r=44100:cl=stereo:d={DURATION}"
VIDEO = f"testsrc=size=160x120:rate=10:duration={DURATION}"


def _make(path: Path, *args: str) -> Path:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args, str(path)],
        check=True,
    )
    return path


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("media")
    lavfi = "-f", "lavfi", "-i"
    return {
        "m4a": _make(d / "a.m4a", *lavfi, SINE, "-c:a", "aac"),
        "mp4": _make(
            d / "v.mp4", *lavfi, VIDEO, *lavfi, SINE, "-c:v", "mpeg4", "-c:a", "aac", "-shortest"
        ),
        "video_only": _make(d / "mute.mp4", *lavfi, VIDEO, "-c:v", "mpeg4"),
        # Дорожка 0 — тишина, дорожка 1 — тон: по ней видно, свелись ли дорожки.
        "two_tracks": _make(
            d / "two.mkv", *lavfi, SILENCE, *lavfi, SINE, "-map", "0", "-map", "1", "-c:a", "flac"
        ),
    }


def _peak(wav_path: Path) -> int:
    with wave.open(str(wav_path), "rb") as w:
        samples = array.array("h", w.readframes(w.getnframes()))
    return max((abs(s) for s in samples), default=0)


def test_output_is_16k_mono_s16(media: dict[str, Path], tmp_path: Path) -> None:
    info = ingest(media["m4a"], tmp_path / "out.wav")
    with wave.open(str(info.wav), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
    assert info.source_codec == "aac"
    assert info.source_sample_rate == 44100
    assert info.audio_streams == 1
    assert not info.has_video


def test_video_and_audio_give_same_duration(media: dict[str, Path], tmp_path: Path) -> None:
    from_video = ingest(media["mp4"], tmp_path / "v.wav")
    from_audio = ingest(media["m4a"], tmp_path / "a.wav")
    assert from_video.has_video
    assert from_video.duration == pytest.approx(DURATION, abs=0.1)
    assert from_video.duration == pytest.approx(from_audio.duration, abs=0.1)


def test_multiple_tracks_are_mixed_by_default(media: dict[str, Path], tmp_path: Path) -> None:
    mixed = ingest(media["two_tracks"], tmp_path / "mix.wav")
    assert mixed.audio_streams == 2
    assert _peak(mixed.wav) > 1000, "тон со второй дорожки потерялся при сведении"


def test_single_track_can_be_picked(media: dict[str, Path], tmp_path: Path) -> None:
    silent = ingest(media["two_tracks"], tmp_path / "t0.wav", stream=0)
    tone = ingest(media["two_tracks"], tmp_path / "t1.wav", stream=1)
    assert _peak(silent.wav) == 0
    assert _peak(tone.wav) > 1000


def test_output_is_deterministic(media: dict[str, Path], tmp_path: Path) -> None:
    """Одинаковый вход → одинаковые байты: на этом держится ключ кэша следующих стадий (#6)."""
    a = ingest(media["mp4"], tmp_path / "1.wav")
    b = ingest(media["mp4"], tmp_path / "2.wav")
    assert a.wav.read_bytes() == b.wav.read_bytes()


def test_no_audio_track_is_a_clear_error(media: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="нет аудиодорожки"):
        ingest(media["video_only"], tmp_path / "x.wav")


def test_missing_stream_is_a_clear_error(media: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="дорожки 5 нет"):
        ingest(media["m4a"], tmp_path / "x.wav", stream=5)


def test_broken_file_is_a_clear_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"\x00not really audio" * 100)
    with pytest.raises(IngestError, match="битый"):
        ingest(broken, tmp_path / "x.wav")
    assert not list(tmp_path.glob("*.part")), "после ошибки остался недописанный файл"


def test_missing_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="не найден"):
        ingest(tmp_path / "nope.mp4", tmp_path / "x.wav")


def test_cli_ingest(media: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "cli.wav"
    result = CliRunner().invoke(app, ["ingest", str(media["mp4"]), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert "есть видео" in result.output


def test_cli_reports_error_without_traceback(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["ingest", str(tmp_path / "nope.mp4")])
    assert result.exit_code == 1
    assert "не найден" in result.output
