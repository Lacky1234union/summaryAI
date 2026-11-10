"""Ингест: любой медиафайл → wav 16 kHz mono PCM s16 + метаданные исходника.

Первая стадия конвейера; дальше этот wav едят VAD (#10) и ASR (#11). 16 kHz mono — родной формат
и Silero, и Whisper: отдать им сразу его дешевле, чем каждый раз пересэмплировать внутри.

Из видео берётся только аудио (`-vn`): видеопоток не декодируется вовсе, поэтому часовой mp4
обрабатывается за секунды, а не за время перекодирования.

Несколько аудиодорожек (OBS: микрофон и звук системы отдельно) по умолчанию сводятся в одну —
иначе половина разговора молча теряется. Конкретная дорожка выбирается `stream=`. Число дорожек
сохраняется в метаданные: на этапе 6 раздельные дорожки станут бесплатной атрибуцией спикеров.

Выход детерминирован (метаданные контейнера срезаны, bitexact): один и тот же вход даёт те же
байты, иначе хеш wav нельзя будет использовать как ключ кэша следующих стадий (#6).

Refs #5
"""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

from summuraze.types import AudioInfo

SAMPLE_RATE = 16_000


class IngestError(RuntimeError):
    """Файл нельзя превратить в wav. Сообщение — для человека, а не стектрейс ffmpeg."""


def ingest(source: Path, output: Path, *, stream: int | None = None) -> AudioInfo:
    """Нормализовать `source` в wav по пути `output`.

    `stream` — номер аудиодорожки среди аудиодорожек (с нуля); `None` — свести все.
    Запись атомарная: прерванный ffmpeg не оставит недописанный wav, который кэш примет
    за готовый.
    """
    if not source.is_file():
        raise IngestError(f"файл не найден: {source}")

    probe = _probe(source)
    audio = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio:
        raise IngestError(f"в файле нет аудиодорожки: {source}")
    if stream is not None and not 0 <= stream < len(audio):
        raise IngestError(
            f"дорожки {stream} нет: в файле аудиодорожек {len(audio)} (нумерация с нуля)"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    try:
        _run(
            [
                _binary("ffmpeg"),
                *("-nostdin", "-hide_banner", "-loglevel", "error", "-y"),
                *("-i", str(source)),
                *_audio_map(len(audio), stream),
                *("-vn", "-sn", "-dn", "-map_metadata", "-1"),
                *("-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le"),
                *("-fflags", "+bitexact", "-flags:a", "+bitexact"),
                *("-f", "wav", str(partial)),
            ],
            what=f"ffmpeg не смог извлечь аудио из {source}",
        )
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)

    picked = audio[stream if stream is not None else 0]
    fmt = probe.get("format", {})
    return AudioInfo(
        source=source,
        wav=output,
        duration=_wav_duration(output),
        source_duration=_float_or_none(fmt.get("duration")),
        source_codec=str(picked.get("codec_name", "unknown")),
        source_channels=int(picked.get("channels") or 1),
        source_sample_rate=int(picked.get("sample_rate") or SAMPLE_RATE),
        audio_streams=len(audio),
        has_video=any(
            s.get("codec_type") == "video"
            # Обложка альбома в mp3/m4a формально видеопоток, но видео не делает.
            and not s.get("disposition", {}).get("attached_pic")
            for s in probe.get("streams", [])
        ),
    )


def _audio_map(count: int, stream: int | None) -> list[str]:
    if stream is not None:
        return ["-map", f"0:a:{stream}"]
    if count == 1:
        return ["-map", "0:a:0"]
    # normalize=0: при normalize amix делит громкость на число дорожек, и тихий собеседник
    # на раздельных дорожках становится вдвое тише — VAD начинает его терять.
    inputs = "".join(f"[0:a:{i}]" for i in range(count))
    return ["-filter_complex", f"{inputs}amix=inputs={count}:normalize=0[mix]", "-map", "[mix]"]


def duration(source: Path) -> float | None:
    """Длительность по контейнеру — для оценки до ингеста (#23); `None` — не знает и он."""
    return _float_or_none(_probe(source).get("format", {}).get("duration"))


def _probe(source: Path) -> dict[str, Any]:
    out = _run(
        [
            _binary("ffprobe"),
            *("-v", "error", "-of", "json", "-show_format", "-show_streams"),
            str(source),
        ],
        what=f"не удалось прочитать файл как медиа (битый или не аудио/видео): {source}",
    )
    try:
        data: dict[str, Any] = json.loads(out)
    except json.JSONDecodeError as e:
        raise IngestError(f"ffprobe вернул нечитаемый ответ для {source}") from e
    return data


def compress_for_note(wav: Path, output: Path, *, bitrate_kbps: int) -> None:
    """Аудио для заметки Obsidian (#18): Opus в контейнере Ogg, моно.

    Расширение `.ogg`, а не `.opus`: `.opus` Obsidian не считает аудио — ссылка на него не
    резолвится и встроенный плеер не появляется; `.ogg` понимают и Obsidian, и Media Extended.
    Источник — wav ингеста (16 kHz): для речи этого хватает, а исходник не перечитывается.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    try:
        _run(
            [
                _binary("ffmpeg"),
                *("-nostdin", "-hide_banner", "-loglevel", "error", "-y"),
                *("-i", str(wav), "-map_metadata", "-1"),
                *("-c:a", "libopus", "-b:a", f"{bitrate_kbps}k", "-application", "voip"),
                *("-f", "ogg", str(partial)),
            ],
            what=f"ffmpeg не смог сжать аудио для заметки из {wav}",
        )
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)


def _run(cmd: list[str], *, what: str) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        # Из многословного stderr ffmpeg человеку нужна последняя строка — там причина.
        lines = [ln for ln in result.stderr.strip().splitlines() if ln.strip()]
        reason = lines[-1] if lines else f"код выхода {result.returncode}"
        raise IngestError(f"{what}\n  ffmpeg: {reason}")
    return result.stdout


def _binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise IngestError(
            f"не найден {name}: поставь ffmpeg (pacman -S ffmpeg / apt install ffmpeg)"
        )
    return path


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def _float_or_none(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
