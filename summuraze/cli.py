"""Точка входа CLI.

Каркас: команды появляются со своими задачами (#7 — оркестратор и стадии, #11 — transcribe,
#17 — summarize). Здесь только то, что нужно, чтобы entry point резолвился и гейт был зелёным.

Импорты тяжёлых зависимостей (torch, faster-whisper) делаются ЛЕНИВО, внутри команд:
`summuraze --help` должен отвечать за ~200 мс, а не грузить 6 секунд рантайм STT.
"""

from __future__ import annotations

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


if __name__ == "__main__":
    app()
