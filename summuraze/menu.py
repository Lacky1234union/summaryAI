"""Интерактивное меню: `summuraze` без аргументов (#38).

Повседневный запуск без знания подкоманд. Меню только спрашивает и зовёт те же функции, что
и подкоманды CLI: своей логики конвейера у него нет, и «из меню» не может работать иначе,
чем `summuraze run`.

Пункты — нумерованный список через `typer.prompt`, без библиотеки меню: стрелки не стоят
лишней зависимости, а ввод цифрой проверяется в тестах через `CliRunner` как есть.

Настройки пишутся в тот же `.env`, который читает конфиг (#3): второй файл настроек значил бы
второй источник истины. После записи настройки перечитываются — если значение не применилось,
его перекрывает переменная окружения, и меню говорит об этом, а не делает вид, что сохранило.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import typer

from summuraze.config import LLMSettings, Settings, get_settings

ENV_FILE = Path(".env")
LOCAL_URL = "http://127.0.0.1:8080/v1"
# Что ffmpeg ингеста (#5) точно читает и что встречается у записей лекций.
MEDIA_SUFFIXES = frozenset(
    {".mka", ".mkv", ".mp4", ".webm", ".mov", ".avi", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}
    | {".oga", ".wav", ".flac", ".wma"}
)
PROFILE_TITLES = {"lecture": "конспект лекции", "raw": "транскрипт по смысловым блокам"}
RECENT = 9


def interactive() -> bool:
    """Меню только для человека за терминалом; в pipe и CI `prompt` ждал бы ввода вечно."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def menu() -> None:
    if not interactive():
        typer.echo(
            "Меню — только в терминале. Для скриптов — подкоманды: summuraze run <файл>, "
            "summuraze record, summuraze --help.",
            err=True,
        )
        raise typer.Exit(2)

    actions: list[tuple[str, Callable[[], None]]] = [
        ("Записать лекцию", _record),
        ("Из файла", _from_file),
        ("Последние заметки", _recent_notes),
        ("Настройки", _settings_menu),
    ]
    while True:
        typer.echo(f"\n{_status(get_settings())}")
        choice = choose("Что делаем?", [title for title, _ in actions], back="выход")
        if choice is None:
            return
        try:
            actions[choice][1]()
        except typer.Exit:
            pass  # ошибку команда уже напечатала одной строкой — обратно в меню
        except (KeyboardInterrupt, typer.Abort):
            typer.echo("\nПрервано.")


def choose(title: str, options: list[str], *, back: str = "назад") -> int | None:
    """Номер выбранного пункта с нуля; `None` — назад: 0 или Ctrl+D."""
    typer.echo(f"\n{title}")
    for i, option in enumerate(options, start=1):
        typer.echo(f"  {i}. {option}")
    typer.echo(f"  0. {back}")
    while True:
        try:
            raw = typer.prompt(">", prompt_suffix=" ")
        except typer.Abort:
            typer.echo("")
            return None
        if raw.strip().isdigit() and int(raw) <= len(options):
            n = int(raw)
            return None if n == 0 else n - 1
        typer.echo(f"Номер от 0 до {len(options)}.")


# ── Пункты ──────────────────────────────────────────────────────────────────


def _record() -> None:
    from summuraze import cli

    name = typer.prompt("Название (Enter — по дате)", default="", show_default=False).strip()
    # Пара — основной случай (#67): забытая остановка стоит конца лекции или часа тишины.
    pair = typer.confirm("Остановить в конце пары?", default=True)
    note = cli._record(name or None, pair=pair)
    if note is not None:
        _offer_open(note)


def _from_file() -> None:
    from summuraze import cli

    settings = get_settings()
    source = pick_file(settings.record.directory)
    if source is None:
        return
    profile = _pick_profile()
    if profile is None:
        return
    note = cli._run(source, until="render", stream=None, profile=profile).note
    if note is not None:
        _offer_open(note)


def _recent_notes() -> None:
    settings = get_settings()
    notes = _newest(settings.paths.notes_dir.glob("*.md"))
    if not notes:
        typer.echo(f"\nЗаметок пока нет: {settings.paths.notes_dir}")
        return
    choice = choose("Открыть заметку", [n.stem for n in notes])
    if choice is not None:
        open_note(notes[choice], settings)


def _settings_menu() -> None:
    while True:
        settings = get_settings()
        typer.echo(f"\n{_status(settings)}")
        choice = choose(
            "Настройки",
            [
                "LLM по API — OpenRouter",
                "LLM локально — llama.cpp server",
                "Папка заметок (Obsidian)",
            ],
        )
        if choice is None:
            return
        if choice == 0:
            _use_openrouter()
        elif choice == 1:
            _use_local(settings.llm)
        else:
            _set_output_dir(settings)


# ── Выбор файла и профиля ───────────────────────────────────────────────────


def pick_file(record_dir: Path) -> Path | None:
    recordings = _newest(
        p for p in (record_dir.glob("*") if record_dir.is_dir() else []) if _is_media(p)
    )
    options = [f"{p.name}  ({_when(p)}, {p.stat().st_size / 2**20:.0f} МБ)" for p in recordings]
    dialog = _dialog_available()
    if dialog:
        options.append("выбрать в окне…")
    options.append("ввести путь (можно перетащить файл в терминал)")
    choice = choose("Какую запись?", options)
    if choice is None:
        return None
    if choice < len(recordings):
        return recordings[choice]
    path = _dialog() if dialog and choice == len(recordings) else _ask_path()
    if path is None:
        return None
    if not path.is_file():
        typer.echo(f"Файла нет: {path}", err=True)
        return None
    return path


def parse_path(raw: str) -> Path | None:
    """Путь, как его вставляет терминал: в кавычках, с экранированием, `file://` при
    перетаскивании из файлового менеджера."""
    text = raw.strip()
    if not text:
        return None
    try:
        parts = shlex.split(text)
    except ValueError:
        parts = [text]
    text = parts[0] if len(parts) == 1 else text
    if text.startswith("file://"):
        text = unquote(urlparse(text).path)
    return Path(text).expanduser()


def _ask_path() -> Path | None:
    return parse_path(typer.prompt("Путь", default="", show_default=False))


def _dialog_available() -> bool:
    return shutil.which("zenity") is not None and _has_display()


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _dialog() -> Path | None:
    patterns = " ".join(f"*{s}" for s in sorted(MEDIA_SUFFIXES))
    done = subprocess.run(
        [
            "zenity",
            "--file-selection",
            "--title=Запись для конспекта",
            f"--file-filter=Аудио и видео | {patterns}",
            "--file-filter=Все файлы | *",
        ],
        capture_output=True,
        text=True,
    )
    path = done.stdout.strip()
    return Path(path) if done.returncode == 0 and path else None


def _pick_profile() -> str | None:
    from summuraze.summarize.profiles import PROFILES

    names = list(PROFILES)
    choice = choose("Что сделать?", [PROFILE_TITLES.get(n, n) for n in names])
    return None if choice is None else names[choice]


# ── Заметки в Obsidian ──────────────────────────────────────────────────────


def open_note(note: Path, settings: Settings) -> None:
    """В Obsidian — по `obsidian://`, если заметки лежат в хранилище (задана папка вывода);
    иначе — тем, чем система открывает .md."""
    target = (
        f"obsidian://open?path={quote(str(note.resolve()))}"
        if settings.paths.output_dir is not None
        else str(note)
    )
    if not _launch(target):
        typer.echo(f"Нечем открыть (нет xdg-open): {note}")


def _launch(target: str) -> bool:
    opener = shutil.which("xdg-open")
    if opener is None:
        return False
    # Без ожидания и без вывода: xdg-open вернёт управление, а Obsidian живёт своей жизнью.
    subprocess.Popen(
        [opener, target],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return True


def _offer_open(note: Path) -> None:
    typer.echo(f"Заметка: {note}")
    if typer.confirm("Открыть?", default=True):
        open_note(note, get_settings())


# ── Настройки ───────────────────────────────────────────────────────────────


def _use_openrouter() -> None:
    defaults = LLMSettings.model_fields
    _save(
        {
            "SUMMURAZE_LLM_PROVIDER": "openai-compat",
            "SUMMURAZE_LLM_BASE_URL": defaults["base_url"].default,
            "SUMMURAZE_LLM_MODEL": defaults["model"].default,
        }
    )
    if not get_settings().llm.api_key.get_secret_value():
        typer.echo("⚠ Ключа нет: OPENROUTER_API_KEY в .env (https://openrouter.ai/keys).", err=True)


def _use_local(current: LLMSettings) -> None:
    url = typer.prompt("Адрес сервера", default=current.base_url if current.is_local else LOCAL_URL)
    # llama-server отвечает той моделью, что загружена, имя ему почти безразлично; для `log`
    # и ключа кэша оно всё равно нужно — разные модели не должны делить кэш саммари.
    model = typer.prompt("Имя модели", default=current.model if current.is_local else "local")
    if not _save(
        {
            "SUMMURAZE_LLM_PROVIDER": "openai-compat",
            "SUMMURAZE_LLM_BASE_URL": url.strip(),
            "SUMMURAZE_LLM_MODEL": model.strip(),
        }
    ):
        return
    if not _server_alive(url.strip()):
        typer.echo(
            f"⚠ {url} не отвечает — запусти llama-server до прогона: "
            f"llama-server -m <модель.gguf> --port 8080",
            err=True,
        )


def _set_output_dir(settings: Settings) -> None:
    current = settings.paths.output_dir
    path = parse_path(
        typer.prompt(
            "Папка заметок в хранилище Obsidian",
            default=str(current) if current else "",
            show_default=bool(current),
        )
    )
    if path is None:
        return
    # Как и конвейер (#18), папку не создаём: опечатка увела бы заметки мимо хранилища.
    if not path.is_dir():
        typer.echo(f"Папки нет: {path} — создай её в хранилище и выбери снова.", err=True)
        return
    _save({"SUMMURAZE_OUTPUT_DIR": str(path.resolve())})


def _save(updates: dict[str, str]) -> bool:
    """Записать в .env и перечитать конфиг. Не прошло валидацию — вернуть как было."""
    from pydantic import ValidationError

    before = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else None
    write_env(ENV_FILE, updates)
    get_settings.cache_clear()
    try:
        settings = get_settings()
    except ValidationError as e:
        if before is None:
            ENV_FILE.unlink()
        else:
            ENV_FILE.write_text(before, encoding="utf-8")
        get_settings.cache_clear()
        typer.echo(f"Не сохранено: {_first_error(e)}", err=True)
        return False
    applied = {
        "SUMMURAZE_LLM_PROVIDER": settings.llm.provider,
        "SUMMURAZE_LLM_BASE_URL": settings.llm.base_url,
        "SUMMURAZE_LLM_MODEL": settings.llm.model,
        "SUMMURAZE_OUTPUT_DIR": str(settings.paths.output_dir),
    }
    shadowed = [k for k, v in updates.items() if applied.get(k, v) != v]
    if shadowed:
        typer.echo(
            f"⚠ Записано в .env, но перекрыто переменной окружения: {', '.join(shadowed)}. "
            "Убери её из окружения (export / set -a; . ./.env) и запусти заново.",
            err=True,
        )
    else:
        typer.echo("Сохранено в .env.")
    return True


def write_env(path: Path, updates: dict[str, str]) -> None:
    """Заменить значения в .env на месте: комментарии и порядок строк остаются, новые ключи —
    в конец. Закомментированный пример `# KEY=…` не трогаем."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    pending = dict(updates)
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if "=" in line and key in pending and not line.lstrip().startswith("#"):
            lines[i] = f"{key}={_quote_env(pending.pop(key))}"
    lines += [f"{key}={_quote_env(value)}" for key, value in pending.items()]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def _quote_env(value: str) -> str:
    # Пробелы в пути к хранилищу («Obsidian Vault») — обычное дело; без кавычек `. ./.env`
    # в shell разрезал бы значение.
    return f'"{value}"' if any(c in value for c in " #'$`\\") else value


def _server_alive(base_url: str) -> bool:
    import httpx

    try:
        return httpx.get(f"{base_url.rstrip('/')}/models", timeout=2).status_code < 500
    except httpx.HTTPError:
        return False


def _first_error(e: Exception) -> str:
    from pydantic import ValidationError

    if isinstance(e, ValidationError) and e.errors():
        return str(e.errors()[0]["msg"])
    return str(e)


# ── Мелочи ──────────────────────────────────────────────────────────────────


def _status(settings: Settings) -> str:
    llm = settings.llm
    if llm.provider == "echo":
        where = "заглушка echo"
    else:
        host = urlparse(llm.base_url).hostname or llm.base_url
        where = f"{'локально' if llm.is_local else 'API'} · {llm.model} @ {host}"
    return f"summuraze · LLM: {where} · заметки: {settings.paths.notes_dir}"


def _newest(paths: Iterable[Path]) -> list[Path]:
    files = [p for p in paths if p.is_file()]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:RECENT]


def _is_media(path: Path) -> bool:
    return path.suffix.lower() in MEDIA_SUFFIXES


def _when(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
