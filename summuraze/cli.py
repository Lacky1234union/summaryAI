"""Точка входа CLI.

Каркас: команды появляются со своими задачами (#7 — оркестратор и стадии, #11 — transcribe,
#17 — summarize). Здесь только то, что нужно, чтобы entry point резолвился и гейт был зелёным.

Импорты тяжёлых зависимостей (torch, faster-whisper) делаются ЛЕНИВО, внутри команд:
`summuraze --help` должен отвечать за ~200 мс, а не грузить 6 секунд рантайм STT.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from summuraze import __version__

app = typer.Typer(
    name="summuraze",
    help="Саммаризатор лекций, семинаров, встреч и созвонов.",
    no_args_is_help=True,
    add_completion=False,
)


# При единственной зарегистрированной команде Typer схлопывает приложение в неё, и
# `summuraze version` перестаёт быть подкомандой. Пустой callback удерживает структуру
# подкоманд, пока их не набралось несколько (#7).
@app.callback()
def main() -> None:
    """Саммаризатор лекций, семинаров, встреч и созвонов."""


@app.command()
def version() -> None:
    """Показать версию."""
    typer.echo(f"summuraze {__version__}")


@app.command()
def ingest(
    source: Annotated[Path, typer.Argument(help="Аудио или видео: mp3, m4a, mp4, mkv, opus…")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Куда писать wav (по умолчанию data/ingest/<имя>.wav)"),
    ] = None,
    stream: Annotated[
        int | None,
        typer.Option(help="Номер аудиодорожки с нуля; по умолчанию все сводятся в одну"),
    ] = None,
) -> None:
    """Нормализовать запись в wav 16 kHz mono."""
    from summuraze.audio.ingest import IngestError
    from summuraze.audio.ingest import ingest as run_ingest
    from summuraze.config import get_settings

    # Имя по исходнику — временно: ключ по хешу входа появится с кэшем артефактов (#6).
    out = output or get_settings().paths.data_dir / "ingest" / f"{source.stem}.wav"
    try:
        info = run_ingest(source, out, stream=stream)
    except IngestError as e:
        typer.echo(f"Ошибка: {e}", err=True)
        raise typer.Exit(1) from e

    typer.echo(f"{info.wav}  ({info.duration:.1f} с)")
    typer.echo(
        f"  исходник: {info.source_codec}, {info.source_channels} кан., "
        f"{info.source_sample_rate} Гц, аудиодорожек: {info.audio_streams}"
        + (", есть видео" if info.has_video else "")
    )


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


if __name__ == "__main__":
    app()
