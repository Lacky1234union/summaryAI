"""Точка входа CLI.

`ingest`, `transcribe` и `run` — один и тот же конвейер (#7), остановленный на разной стадии:
повторный вызов берёт готовое из кэша, и `run` после `transcribe` не транскрибирует заново.

Импорты тяжёлых зависимостей (torch, faster-whisper) делаются ЛЕНИВО, внутри команд:
`summuraze --help` должен отвечать за ~200 мс, а не грузить 6 секунд рантайм STT.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from summuraze import __version__

if TYPE_CHECKING:
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
    "render": "заметка",
}


@app.callback()
def main() -> None:
    """Саммаризатор лекций, семинаров, встреч и созвонов."""


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
def transcribe(source: SourceArg, stream: StreamOpt = None, lang: LangOpt = None) -> None:
    """Расшифровать запись (без саммари)."""
    result = _run(source, until="asr", stream=stream, lang=lang)
    assert result.transcript is not None
    t = result.transcript
    typer.echo(
        f"{len(t.segments)} сегментов, {t.duration:.1f} с, язык {t.language}, модель {t.asr_model}"
    )
    if result.speech is not None:
        speech = sum(i.duration for i in result.speech)
        share = speech / t.duration if t.duration else 0.0
        typer.echo(f"  речь: {speech:.1f} с ({share:.0%}), интервалов: {len(result.speech)}")


@app.command()
def run(
    source: SourceArg,
    profile: Annotated[str, typer.Option(help="Профиль документа")] = "lecture",
    stream: StreamOpt = None,
    lang: LangOpt = None,
) -> None:
    """Запись → транскрипт → саммари → заметка."""
    result = _run(source, until="render", stream=stream, profile=profile, lang=lang)
    typer.echo(f"Заметка: {result.note}")


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


def _run(
    source: Path,
    *,
    until: StageName,
    stream: int | None,
    profile: str = "lecture",
    lang: str | None = None,
) -> RunResult:
    from summuraze.audio.ingest import IngestError
    from summuraze.audio.vad import NoSpeechError
    from summuraze.config import get_settings
    from summuraze.pipeline.orchestrator import Pipeline

    settings = get_settings()
    if lang is not None:
        language = None if lang.strip().lower() in ("", "auto") else lang.strip().lower()
        asr = settings.asr.model_copy(update={"language": language})
        settings = settings.model_copy(update={"asr": asr})
    try:
        with Pipeline(settings, on_event=_print_event) as pipeline:
            return pipeline.run(source, until=until, stream=stream, profile=profile)
    except (
        IngestError,
        NoSpeechError,
        NotImplementedError,
        ValueError,
        FileNotFoundError,
    ) as e:
        typer.echo(f"Ошибка: {e}", err=True)
        raise typer.Exit(1) from e


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
