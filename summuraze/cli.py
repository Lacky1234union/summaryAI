"""Точка входа CLI.

`ingest`, `transcribe` и `run` — один и тот же конвейер (#7), остановленный на разной стадии:
повторный вызов берёт готовое из кэша, и `run` после `transcribe` не транскрибирует заново.

Импорты тяжёлых зависимостей (torch, faster-whisper) делаются ЛЕНИВО, внутри команд:
`summuraze --help` должен отвечать за ~200 мс, а не грузить 6 секунд рантайм STT.
"""

from __future__ import annotations

import shutil
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from summuraze import __version__

if TYPE_CHECKING:
    from summuraze.audio.record import Source
    from summuraze.config import Settings
    from summuraze.pipeline.orchestrator import RunResult, StageEvent, StageName

app = typer.Typer(
    name="summuraze",
    help="Саммаризатор лекций, семинаров, встреч и созвонов.",
    no_args_is_help=True,
    add_completion=False,
)

SourceArg = Annotated[Path, typer.Argument(help="Аудио или видео: mp3, m4a, mp4, mkv, opus…")]
StreamOpt = Annotated[
    int | None,
    typer.Option(help="Номер аудиодорожки с нуля; по умолчанию все сводятся в одну"),
]
TermsOpt = Annotated[
    Path | None,
    typer.Option(
        "--terms",
        help="Словарь терминов курса: по строке на термин, «слышится -> пишется» — замена",
    ),
]
ProfileOpt = Annotated[
    str, typer.Option(help="Профиль документа: lecture — конспект, raw — транскрипт по блокам")
]
LangOpt = Annotated[
    str | None,
    typer.Option(
        "--lang", help="Язык записи (ru, en…) или auto; по умолчанию SUMMURAZE_ASR_LANGUAGE"
    ),
]

_STAGE_TITLES = {
    "ingest": "ингест",
    "vad": "поиск речи",
    "asr": "распознавание",
    "summarize": "саммари",
    "audio": "аудио для заметки",
    "render": "заметка",
}


@app.callback()
def main() -> None:
    """Саммаризатор лекций, семинаров, встреч и созвонов."""
    import sys

    from loguru import logger

    # Предупреждения — человеку в терминал, без отметок времени и имён модулей. С новой строки:
    # они приходят посреди стадии, когда в строке уже стоит «[3/5] распознавание…».
    logger.remove()
    logger.add(sys.stderr, level="WARNING", format="\n<yellow>Внимание:</yellow> {message}")


@app.command()
def version() -> None:
    """Показать версию."""
    typer.echo(f"summuraze {__version__}")


@app.command()
def ingest(
    source: SourceArg,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Скопировать wav ещё и сюда")
    ] = None,
    stream: StreamOpt = None,
) -> None:
    """Нормализовать запись в wav 16 kHz mono."""
    result = _run(source, until="ingest", stream=stream)
    info = result.audio
    wav = info.wav
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        wav = Path(shutil.copyfile(info.wav, output))
    typer.echo(f"{wav}  ({info.duration:.1f} с)")
    typer.echo(
        f"  исходник: {info.source_codec}, {info.source_channels} кан., "
        f"{info.source_sample_rate} Гц, аудиодорожек: {info.audio_streams}"
        + (", есть видео" if info.has_video else "")
    )


@app.command()
def transcribe(
    source: SourceArg, stream: StreamOpt = None, lang: LangOpt = None, terms: TermsOpt = None
) -> None:
    """Расшифровать запись (без саммари)."""
    result = _run(source, until="asr", stream=stream, lang=lang, terms=terms)
    assert result.transcript is not None
    t = result.transcript
    typer.echo(
        f"{len(t.segments)} сегментов, {t.duration:.1f} с, язык {t.language}, модель {t.asr_model}"
    )
    suspects = sum(s.suspect for s in t.segments)
    if suspects:
        typer.echo(f"  сомнительных сегментов: {suspects} (помечены ⚠️ в заметке)")
    if result.speech is not None:
        speech = sum(i.duration for i in result.speech)
        share = speech / t.duration if t.duration else 0.0
        typer.echo(f"  речь: {speech:.1f} с ({share:.0%}), интервалов: {len(result.speech)}")


@app.command()
def run(
    source: SourceArg,
    profile: ProfileOpt = "lecture",
    stream: StreamOpt = None,
    lang: LangOpt = None,
    terms: TermsOpt = None,
) -> None:
    """Запись → транскрипт → саммари → заметка."""
    result = _run(source, until="render", stream=stream, profile=profile, lang=lang, terms=terms)
    typer.echo(f"Заметка: {result.note}")


@app.command()
def record(
    name: Annotated[
        str | None, typer.Option("--name", "-n", help="Имя записи и заметки, напр. «ЦОС лекция 3»")
    ] = None,
    profile: ProfileOpt = "lecture",
    lang: LangOpt = None,
    terms: TermsOpt = None,
    mic_only: Annotated[
        bool, typer.Option("--mic-only", help="Только микрофон, без звука системы")
    ] = False,
    no_run: Annotated[
        bool, typer.Option("--no-run", help="Только записать, без конспекта")
    ] = False,
) -> None:
    """Записать микрофон и звук системы; по Ctrl+C — конспект в Obsidian."""
    from summuraze.audio import record as rec
    from summuraze.pipeline.orchestrator import Pipeline

    settings = _settings(lang)
    cfg = settings.record
    with _clean_errors():
        if not no_run:
            # Пустой ключ или нет папки хранилища — узнать сейчас, а не после двухчасовой пары.
            with Pipeline(settings) as pipeline:
                pipeline.preflight(profile=profile, terms=terms)
        sources = rec.default_sources(mic=cfg.mic, system="none" if mic_only else cfg.system)
        for src in sources:
            if rec.muted(src.name):
                typer.secho(
                    f"⚠ {src.title} выключен в системе — запись будет немой. Включить: "
                    f"pactl set-source-mute {src.name} 0",
                    fg="yellow",
                    err=True,
                )
        path = rec.record_name(cfg.directory, name)
        typer.echo(f"● Запись: {path}\n  Ctrl+C — остановить и сделать конспект.")
        stopped_by_user = _record_loop(path, sources, cfg.bitrate_kbps, cfg.silence_stop_minutes)
    if stopped_by_user:
        _absorb_echoed_sigint()

    if not path.is_file() or path.stat().st_size < 4096:
        typer.echo("Запись пустая — конспект не делаю.")
        raise typer.Exit(1)
    typer.echo(f"Сохранено: {path} ({path.stat().st_size / 2**20:.1f} МБ)")
    if no_run:
        return
    result = _run(path, until="render", stream=None, profile=profile, lang=lang, terms=terms)
    typer.echo(f"Заметка: {result.note}")


def _absorb_echoed_sigint(window: float = 2.0) -> None:
    """Ctrl+C под `uv run` приходит дважды: от терминала и пересланный uv. Второй убивал конвейер
    сразу после записи — повтор в первые секунды после остановки глотаем. Осознанный Ctrl+C
    позже прерывает конвейер как обычно."""
    import signal
    import time

    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        time.sleep(window)
    finally:
        signal.signal(signal.SIGINT, previous)


def _record_loop(path: Path, sources: list[Source], bitrate: int, silence_minutes: float) -> bool:
    """Записывать до Ctrl+C или авто-стопа; True — остановил человек."""
    import sys

    from summuraze.audio import record as rec

    watch = rec.SilenceWatch(silence_minutes)
    quiet_since: dict[int, float] = {}
    warned: set[int] = set()
    tty = sys.stdout.isatty()
    last_print = 0.0
    try:
        with rec.recording(
            path,
            rec.pulse_inputs(sources),
            [s.title for s in sources],
            bitrate_kbps=bitrate,
        ) as recording:
            for levels in recording.levels():
                # Время потока, а не настенные часы: у живого PipeWire они совпадают, а одна шкала
                # на индикатор, предупреждения и авто-стоп проверяется тестом.
                elapsed = levels.at
                for i, quiet in enumerate(levels.silent()):
                    if not quiet:
                        quiet_since.pop(i, None)
                        continue
                    since = quiet_since.setdefault(i, elapsed)
                    # Немая дорожка 10 с подряд — проверить сейчас, а не узнать после пары.
                    if elapsed - since >= 10 and i not in warned:
                        warned.add(i)
                        hint = (
                            "включён ли микрофон?"
                            if i == 0
                            else "звука в системе нет — это нормально, если лекция не онлайн"
                        )
                        typer.secho(
                            f"\n⚠ {sources[i].title}: тишина 10 с — {hint}", fg="yellow", err=True
                        )
                line = _meter(elapsed, sources, levels.peaks)
                if tty:
                    typer.echo(f"\r{line}", nl=False)
                elif elapsed - last_print >= 30:
                    typer.echo(line)
                    last_print = elapsed
                if watch.update(levels):
                    typer.echo(f"\nТишина {silence_minutes:g} мин — останавливаю запись.")
                    break
    except KeyboardInterrupt:
        # Сигнал SIGINT уже пришёл — до конца остановки ffmpeg повторный не должен её сорвать.
        import signal

        signal.signal(signal.SIGINT, signal.SIG_IGN)
        return True
    finally:
        if tty:
            typer.echo("")
    return False


def _meter(elapsed: float, sources: list[Source], peaks: tuple[float, ...]) -> str:
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    parts = [f"● {h:02d}:{m:02d}:{s:02d}"]
    for src, peak in zip(sources, peaks, strict=True):
        filled = max(0, min(10, round((peak + 60) / 6)))
        level = "тишина" if peak < -50 else f"{peak:4.0f} дБ"
        parts.append(f"{src.title} {'▮' * filled}{'▯' * (10 - filled)} {level:>7}")
    return "   ".join(parts)


@app.command()
def log(
    limit: Annotated[int, typer.Option("--limit", "-n", help="Сколько прогонов показать")] = 20,
) -> None:
    """История прогонов: что, чем, сколько стоило, что взялось из кэша."""
    from summuraze.config import get_settings
    from summuraze.store.db import Database

    with Database(get_settings().paths.db_path) as db:
        runs = db.recent_runs(limit)
    if not runs:
        typer.echo("Прогонов ещё не было.")
        return
    for r in runs:
        cost = f"${r.cost_usd:.4f}" if r.cost_usd is not None else "—"
        typer.echo(
            f"#{r.id}  {_local_time(r.started_at)}  {r.status:<7}  {Path(r.source).name}  "
            f"кэш {r.stages_cached}/{r.stages_total}  {r.llm_model or '—'}  {cost}"
        )
        if r.error:
            typer.echo(f"      {r.error}")


@app.command()
def gc(
    days: Annotated[
        float, typer.Option(help="Удалить артефакты, не использованные столько дней")
    ] = 30,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Только показать, сколько освободится")
    ] = False,
) -> None:
    """Почистить кэш артефактов: wav и транскрипты быстро съедают десятки ГБ."""
    from summuraze.config import get_settings
    from summuraze.store.artifacts import ArtifactStore

    store = ArtifactStore(get_settings().paths.artifacts_dir)
    result = store.gc(days, dry_run=dry_run)
    verb = "будет удалено" if dry_run else "удалено"
    typer.echo(f"{verb}: {result.removed} артефактов, {result.freed_bytes / 2**20:.1f} МБ")


def _local_time(iso_utc: str) -> str:
    # В БД — UTC (сравнимо между машинами), человеку — местное время.
    return datetime.fromisoformat(iso_utc).astimezone().strftime("%Y-%m-%d %H:%M")


def _settings(lang: str | None) -> Settings:
    from summuraze.config import get_settings

    settings = get_settings()
    if lang is not None:
        language = None if lang.strip().lower() in ("", "auto") else lang.strip().lower()
        asr = settings.asr.model_copy(update={"language": language})
        settings = settings.model_copy(update={"asr": asr})
    return settings


@contextmanager
def _clean_errors() -> Generator[None, None, None]:
    """Ожидаемые ошибки — одной строкой для человека и кодом 1, без стектрейса."""
    from summuraze.audio.ingest import IngestError
    from summuraze.audio.record import RecordError
    from summuraze.audio.vad import NoSpeechError
    from summuraze.pipeline.stages import SummaryFormatError
    from summuraze.summarize.openai_compat import LLMError

    try:
        yield
    except (
        IngestError,
        RecordError,
        NoSpeechError,
        LLMError,
        SummaryFormatError,
        NotImplementedError,
        ValueError,
        FileNotFoundError,
    ) as e:
        typer.echo(f"Ошибка: {e}", err=True)
        raise typer.Exit(1) from e


def _run(
    source: Path,
    *,
    until: StageName,
    stream: int | None,
    profile: str = "lecture",
    lang: str | None = None,
    terms: Path | None = None,
) -> RunResult:
    from summuraze.pipeline.orchestrator import Pipeline

    with _clean_errors(), Pipeline(_settings(lang), on_event=_print_event) as pipeline:
        return pipeline.run(source, until=until, stream=stream, profile=profile, terms=terms)


def _print_event(e: StageEvent) -> None:
    title = _STAGE_TITLES[e.stage]
    prefix = f"[{e.index}/{e.total}] {title}"
    if e.status == "start":
        typer.echo(f"{prefix}…", nl=False)
    elif e.status == "cached":
        typer.echo(" из кэша")
    else:
        typer.echo(f" {e.seconds:.1f} с")


if __name__ == "__main__":
    app()
