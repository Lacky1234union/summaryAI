"""Точка входа CLI.

`ingest`, `transcribe` и `run` — один и тот же конвейер (#7), остановленный на разной стадии:
повторный вызов берёт готовое из кэша, и `run` после `transcribe` не транскрибирует заново.

Импорты тяжёлых зависимостей (torch, faster-whisper) делаются ЛЕНИВО, внутри команд:
`summuraze --help` должен отвечать за ~200 мс, а не грузить 6 секунд рантайм STT.

Без аргументов — интерактивное меню (#38): оно зовёт эти же команды, своей логики конвейера
у него нет.
"""

from __future__ import annotations

import shutil
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
from urllib.parse import urlparse

import typer

from summuraze import __version__

if TYPE_CHECKING:
    from summuraze.audio.record import Source
    from summuraze.config import Settings
    from summuraze.pipeline.orchestrator import Pipeline, RunResult, StageEvent, StageName

app = typer.Typer(
    name="summuraze",
    help="Саммаризатор лекций, семинаров, встреч и созвонов.",
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
DebugOpt = Annotated[
    bool, typer.Option("--debug", help="Отладочный вид заметки: модели, стоимость, всё развёрнуто")
]
LocalOnlyOpt = Annotated[
    bool,
    typer.Option(
        "--local-only", help="Без сети: саммари только локальной моделью, Whisper — только с диска"
    ),
]
LocalOpt = Annotated[
    bool,
    typer.Option(
        "--local",
        help="Саммари своей моделью (SUMMURAZE_LOCAL_LLM_MODEL) через llama-server; сеть запрещена",
    ),
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
    "verify": "проверка",
    "audio": "аудио для заметки",
    "render": "заметка",
}


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Саммаризатор лекций, семинаров, встреч и созвонов. Без команды — меню."""
    from loguru import logger

    # Предупреждения — человеку в терминал, без отметок времени и имён модулей. С новой строки:
    # они приходят посреди стадии, когда в строке уже стоит «[3/5] распознавание…».
    logger.remove()
    logger.add(sys.stderr, level="WARNING", format="\n<yellow>Внимание:</yellow> {message}")
    if ctx.invoked_subcommand is None:
        from summuraze.menu import menu

        menu()


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
    source: SourceArg,
    stream: StreamOpt = None,
    lang: LangOpt = None,
    terms: TermsOpt = None,
    local_only: LocalOnlyOpt = False,
    formats: Annotated[
        str,
        typer.Option(
            "--format", "-f", help="Файлы транскрипта через запятую: json, txt, srt, vtt или all"
        ),
    ] = "json,txt,srt",
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Куда положить файлы (по умолчанию — рядом с записью)"),
    ] = None,
) -> None:
    """Расшифровать запись (без саммари) и положить транскрипт рядом: json, txt, srt."""
    from summuraze.render.export import export, parse_formats

    try:
        chosen = parse_formats(formats)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--format") from None
    result = _run(source, until="asr", stream=stream, lang=lang, terms=terms, local_only=local_only)
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
    try:
        written = export(t, source, chosen, out)
    except OSError as e:
        typer.secho(f"Не записать транскрипт: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Транскрипт в {written[0].parent}: {', '.join(p.name for p in written)}")


@app.command()
def run(
    source: SourceArg,
    profile: ProfileOpt = "lecture",
    stream: StreamOpt = None,
    lang: LangOpt = None,
    terms: TermsOpt = None,
    debug: DebugOpt = False,
    local_only: LocalOnlyOpt = False,
    local: LocalOpt = False,
) -> None:
    """Запись → транскрипт → саммари → заметка."""
    result = _run(
        source,
        until="render",
        stream=stream,
        profile=profile,
        lang=lang,
        terms=terms,
        debug=debug,
        local_only=local_only,
        local=local,
    )
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
    debug: DebugOpt = False,
    local_only: LocalOnlyOpt = False,
    local: LocalOpt = False,
) -> None:
    """Записать микрофон и звук системы; по Ctrl+C — конспект в Obsidian."""
    note = _record(
        name,
        profile=profile,
        lang=lang,
        terms=terms,
        mic_only=mic_only,
        no_run=no_run,
        debug=debug,
        local_only=local_only,
        local=local,
    )
    if note is not None:
        typer.echo(f"Заметка: {note}")


def _record(
    name: str | None,
    *,
    profile: str = "lecture",
    lang: str | None = None,
    terms: Path | None = None,
    mic_only: bool = False,
    no_run: bool = False,
    debug: bool = False,
    local_only: bool = False,
    local: bool = False,
) -> Path | None:
    """Записать и, если не `no_run`, прогнать конвейер; вернуть путь заметки."""
    from summuraze.audio import record as rec
    from summuraze.pipeline.orchestrator import Pipeline

    with _clean_errors():
        settings = _settings(lang, debug, local_only, local)
        cfg = settings.record
        if not no_run:
            # Пустой ключ или нет папки хранилища — узнать сейчас, а не после двухчасовой пары.
            with Pipeline(settings) as pipeline:
                pipeline.preflight(profile=profile, terms=terms)
                _confirm_remote(pipeline)
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
        return None
    return _run(
        path,
        until="render",
        stream=None,
        profile=profile,
        lang=lang,
        terms=terms,
        debug=debug,
        local_only=local_only,
        local=local,
    ).note


def _absorb_echoed_sigint(window: float = 2.0) -> None:
    """Ctrl+C под `uv run` приходит дважды: от терминала и пересланный uv. Второй убивал конвейер
    сразу после записи — повтор в первые секунды после остановки глотаем. Осознанный Ctrl+C
    позже прерывает конвейер как обычно."""
    import signal
    import time

    # Вернуть именно обычный обработчик, а не «предыдущий»: `_record_loop` уже поставил SIG_IGN,
    # и с ним конвейер после записи (и меню после неё) не прерывался бы никогда.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        time.sleep(window)
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)


def _record_loop(path: Path, sources: list[Source], bitrate: int, silence_minutes: float) -> bool:
    """Записывать до Ctrl+C или авто-стопа; True — остановил человек."""
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
    from summuraze.config import is_local_url
    from summuraze.render.markdown import short_timecode

    for r in runs:
        took = f"за {_took(r.seconds)}" if r.seconds is not None else "идёт"
        length = short_timecode(r.duration) if r.duration is not None else "—"
        typer.echo(
            f"#{r.id}  {_local_time(r.started_at)}  {_STATUS.get(r.status, r.status)}  "
            f"{Path(r.source).name}  {length}  {took}  кэш {r.stages_cached}/{r.stages_total}"
        )
        if r.llm_model:
            # Что куда уехало (#20): хост провайдера, а не только модель.
            url = r.llm_provider or ""
            where = (
                "локально" if url == "echo" or is_local_url(url) else urlparse(url).hostname or url
            )
            tokens = ""
            if r.input_tokens is not None and r.output_tokens is not None:
                tokens = f" · {_k(r.input_tokens)}→{_k(r.output_tokens)} ток."
                if r.cached_tokens:
                    tokens += f" (кэш {_k(r.cached_tokens)})"
            cost = f"${r.cost_usd:.4f}" if r.cost_usd is not None else "цена —"
            typer.echo(f"      {where} · {r.llm_model}{tokens} · {cost}")
        if r.error:
            typer.echo(f"      {r.error}")
    total = sum(r.cost_usd or 0 for r in runs)
    typer.echo(f"Итого за {len(runs)} прогонов: ${total:.4f}")


_STATUS = {"done": "готово ", "failed": "ошибка ", "running": "идёт   "}


def _took(seconds: float) -> str:
    return f"{seconds:.0f} с" if seconds < 90 else f"{seconds / 60:.0f} мин"


def _k(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


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


@app.command()
def prune(
    before: Annotated[
        str | None, typer.Option(help="Удалить аудио записей, сделанных до этой даты: 2026-10-01")
    ] = None,
    older_than: Annotated[
        str | None, typer.Option("--older-than", help="…или старше срока: 30d, 2w")
    ] = None,
    include_foreign: Annotated[
        bool,
        typer.Option(
            "--include-foreign", help="Удалять и оригиналы не из SUMMURAZE_RECORD_DIR (чужие файлы)"
        ),
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Удалить; без флага — только показать")
    ] = False,
) -> None:
    """Удалить аудио старых записей; заметки, транскрипты и история остаются."""
    from summuraze.config import get_settings
    from summuraze.store import retention
    from summuraze.store.artifacts import ArtifactStore
    from summuraze.store.db import Database

    settings = get_settings()
    with _clean_errors():
        border = retention.cutoff(
            before=before,
            older_than=older_than,
            retention_days=settings.retention_days,
            now=datetime.now().astimezone(),
        )
    with Database(settings.paths.db_path) as db:
        pruned = retention.plan(
            db,
            ArtifactStore(settings.paths.artifacts_dir),
            settings.record.directory,
            border,
            include_foreign=include_foreign,
        )
        if not pruned.victims:
            typer.echo(f"Записей до {border:%Y-%m-%d %H:%M} с аудио нет — удалять нечего.")
            return
        typer.echo(
            f"Записей до {border:%Y-%m-%d %H:%M}: {len(pruned.victims)}, "
            f"аудио {retention.human_size(pruned.size)}"
        )
        for v in pruned.victims:
            kinds = ", ".join(sorted({_PRUNE_KINDS[i.kind] for i in v.items}))
            typer.echo(
                f"  {v.recorded:%Y-%m-%d}  {v.name}  {retention.human_size(v.size)}  ({kinds})"
            )
        if pruned.foreign_kept:
            typer.echo(
                f"Оригиналы не из {settings.record.directory} не трогаю "
                f"({pruned.foreign_kept}) — --include-foreign, чтобы удалить и их."
            )
        if not yes:
            typer.echo("Ничего не удалено. Удалить — тот же вызов с --yes.")
            return
        notes = retention.apply(pruned, db, today=datetime.now().date())
    typer.echo(
        f"Удалено {retention.human_size(pruned.size)}; заметок обновлено: {notes} "
        "(таймкоды теперь текстом)."
    )


_PRUNE_KINDS = {
    "original": "оригинал",
    "cache": "кэш",
    "audio": "аудио в хранилище",
    "subtitles": "субтитры",
}


def _local_time(iso_utc: str) -> str:
    # В БД — UTC (сравнимо между машинами), человеку — местное время.
    return datetime.fromisoformat(iso_utc).astimezone().strftime("%Y-%m-%d %H:%M")


def _settings(
    lang: str | None, debug: bool = False, local_only: bool = False, local: bool = False
) -> Settings:
    from summuraze.config import get_settings

    settings = get_settings()
    if local:
        settings = settings.localized()
    if local_only:
        settings = settings.restricted()
    if debug:
        note = settings.note.model_copy(update={"debug": True})
        settings = settings.model_copy(update={"note": note})
    if lang is not None:
        language = None if lang.strip().lower() in ("", "auto") else lang.strip().lower()
        asr = settings.asr.model_copy(update={"language": language})
        settings = settings.model_copy(update={"asr": asr})
    return settings


@contextmanager
def _clean_errors() -> Generator[None, None, None]:
    """Ожидаемые ошибки — одной строкой для человека и кодом 1, без стектрейса."""
    from summuraze.asr.base import ASRModelMissing
    from summuraze.audio.ingest import IngestError
    from summuraze.audio.record import RecordError
    from summuraze.audio.vad import NoSpeechError
    from summuraze.pipeline.stages import SummaryFormatError
    from summuraze.summarize.budget import TranscriptTooLong
    from summuraze.summarize.openai_compat import LLMError

    try:
        yield
    except (
        ASRModelMissing,
        IngestError,
        RecordError,
        NoSpeechError,
        LLMError,
        SummaryFormatError,
        TranscriptTooLong,
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
    debug: bool = False,
    local_only: bool = False,
    local: bool = False,
) -> RunResult:
    from summuraze.pipeline.orchestrator import Pipeline

    try:
        with (
            _clean_errors(),
            Pipeline(_settings(lang, debug, local_only, local), on_event=_print_event) as pipeline,
        ):
            if until in ("summarize", "render"):
                pipeline.preflight(until=until, profile=profile, terms=terms)
                _confirm_remote(pipeline)
            result = pipeline.run(source, until=until, stream=stream, profile=profile, terms=terms)
            if result.budget is not None:
                typer.echo(f"Саммари: {result.budget.describe()}")
            if result.verification is not None:
                typer.echo(f"Проверка: {result.verification.describe()}")
            if result.verification_error is not None:
                typer.echo(f"Проверка не удалась, заметка — без неё: {result.verification_error}")
            if until == "render":
                _auto_prune(pipeline, keep=result.recording_id)
            return result
    finally:
        # Стадия упала или прервана — секундомер не должен тикать поверх сообщения об ошибке.
        if _print_event.halt():
            typer.echo("")


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _confirm_remote(pipeline: Pipeline) -> None:
    """Транскрипт уезжает наружу — один раз на хост показать, что именно, и запомнить ответ
    (#20). Спрашиваем до транскрибации: после часа Whisper человека у терминала может не быть."""
    llm = pipeline.settings.llm
    if llm.is_local or pipeline.db.has_consent(llm.host):
        return
    typer.echo(
        f"Саммари делает {llm.host} ({llm.model}): туда уйдёт текст транскрипта целиком — "
        "около 20 тыс. токенов на час записи. Аудио наружу не уходит никогда."
    )
    if not _interactive():
        typer.echo(
            f"Ошибка: первую отправку на {llm.host} нужно разрешить в терминале — запусти "
            "команду там один раз. Без сети — --local-only и локальный сервер.",
            err=True,
        )
        raise typer.Exit(1)
    if not typer.confirm(f"Отправлять на {llm.host}? Ответ запомню", default=False):
        typer.echo(
            "Не отправляю. Без сети: --local-only и SUMMURAZE_LLM_BASE_URL=http://127.0.0.1:8080/v1"
        )
        raise typer.Exit(1)
    pipeline.db.grant_consent(llm.host)


def _auto_prune(pipeline: Pipeline, *, keep: int | None) -> None:
    """Срок хранения (#49) — в конце прогона, только свои записи: чужой файл пользователь
    удаляет сам или явным `prune --include-foreign`."""
    from summuraze.store import retention

    settings = pipeline.settings
    if settings.retention_days is None:
        return
    border = retention.cutoff(
        before=None,
        older_than=None,
        retention_days=settings.retention_days,
        now=datetime.now().astimezone(),
    )
    pruned = retention.plan(
        pipeline.db, pipeline.store, settings.record.directory, border, keep=keep
    )
    if not pruned.victims:
        return
    retention.apply(pruned, pipeline.db, today=datetime.now().date())
    typer.echo(
        f"Срок хранения {settings.retention_days:g} дн.: удалено аудио "
        f"{len(pruned.victims)} записей, {retention.human_size(pruned.size)}."
    )


class _StageProgress:
    """Строка стадии. В терминале — с секундомером: распознавание идёт десятки минут, и строка
    без движения выглядит зависшей (#38). В pipe и логе — только итог, без перерисовки."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._ticker: threading.Thread | None = None

    def __call__(self, e: StageEvent) -> None:
        prefix = f"[{e.index}/{e.total}] {_STAGE_TITLES[e.stage]}"
        if e.status == "start":
            typer.echo(f"{prefix}…", nl=False)
            if sys.stdout.isatty():
                self._stop.clear()
                self._ticker = threading.Thread(target=self._tick, args=(prefix,), daemon=True)
                self._ticker.start()
            return
        tty = self.halt()
        tail = {"cached": "из кэша", "failed": "не удалась"}.get(e.status, f"{e.seconds:.1f} с")
        # \033[K — стереть секундомер, если итог короче него.
        typer.echo(f"\r{prefix}… {tail}\033[K" if tty else f" {tail}")

    def _tick(self, prefix: str) -> None:
        elapsed = 0
        while not self._stop.wait(1):
            elapsed += 1
            m, s = divmod(elapsed, 60)
            typer.echo(f"\r{prefix}… {m:02d}:{s:02d}", nl=False)

    def halt(self) -> bool:
        if self._ticker is None:
            return False
        self._stop.set()
        self._ticker.join()
        self._ticker = None
        return True


_print_event = _StageProgress()


if __name__ == "__main__":
    app()
