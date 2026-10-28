"""Своя модель через `llama-server` на время стадии саммари (#19).

Сервер запускается фабрикой — то есть после ASR, когда Whisper уже выгружен, — и гасится сразу
после саммари. Постоянный демон держал бы 5–7 ГБ всё время транскрибации, а Whisper и 8B-модель
вместе в 14 ГБ не влезают (CLAUDE.md, «Память»).

Говорит сервер тем же OpenAI-совместимым протоколом, что DeepSeek и OpenRouter, поэтому адаптер
тот же (`OpenAICompatLLM`); здесь только жизнь процесса. Флаги:

- `--parallel 1` — весь контекст одному запросу: конвейер последовательный;
- `-fa on -ctk q8_0 -ctv q8_0` — KV-кэш в 8 бит: вдвое меньше памяти на тот же контекст,
  на качестве саммари не сказывается (замер — BENCHMARKS.md);
- `--host 127.0.0.1` и свободный порт — наружу сервер не слушает и не мешает чужому на 8080.

**Кэш промпта между прогонами.** Сервер живёт одну стадию, и его prompt cache умирает с ним. А
префикс — транскрипт целиком, на CPU это минуты счёта. Поэтому KV-кэш слота после запроса
сохраняется на диск (`/slots/0?action=save`), а перед запросом с тем же транскриптом —
восстанавливается: повторный прогон той же записи с другим профилем или правленым промптом
считает заново только хвост. Файлы — по sha256 модели, контекста и префикса; держим последние
`slot_cache` штук.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

from summuraze.summarize.openai_compat import LLMError, OpenAICompatLLM

if TYPE_CHECKING:
    from summuraze.config import LLMSettings, LocalLLMSettings
    from summuraze.types import LLMResult

_POLL_SECONDS = 0.5


def physical_cores() -> int:
    """Физические ядра: на SMT генерация в столько потоков быстрее, чем во все логические."""
    try:
        siblings = Path("/proc/cpuinfo").read_text().split("\n\n")
        cores = {
            (_field(block, "physical id"), _field(block, "core id"))
            for block in siblings
            if "core id" in block
        }
        if cores:
            return len(cores)
    except OSError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def _field(block: str, name: str) -> str:
    for line in block.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == name:
            return value.strip()
    return ""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


class LlamaServer:
    def __init__(self, cfg: LocalLLMSettings) -> None:
        self.cfg = cfg
        self.process: subprocess.Popen[bytes] | None = None
        self.log = Path(tempfile.gettempdir()) / f"summuraze-llama-{os.getpid()}.log"

    def command(self, port: int) -> list[str]:
        assert self.cfg.model is not None
        binary = shutil.which(self.cfg.server) or self.cfg.server
        return [
            binary,
            *("-m", str(self.cfg.model)),
            *("-c", str(self.cfg.context)),
            *("-t", str(self.cfg.threads or physical_cores())),
            *("--host", "127.0.0.1", "--port", str(port)),
            *("--parallel", "1"),
            *("-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"),
            # Кэш промптов в ОЗУ (по умолчанию до 8 ГБ) короткоживущему серверу ни к чему.
            *("--cache-ram", "0"),
            "--no-webui",
            # Имя модели в ответах и в `summuraze log` — имя файла, а не путь.
            *("--alias", self.cfg.model.stem),
            *(("--slot-save-path", str(self.cfg.slot_dir)) if self._slots else ()),
            *self.cfg.args,
        ]

    @property
    def _slots(self) -> bool:
        return self.cfg.slot_cache > 0 and self.cfg.slot_dir is not None

    def start(self) -> str:
        """Запустить и дождаться загрузки модели; вернуть base_url."""
        if shutil.which(self.cfg.server) is None and not Path(self.cfg.server).is_file():
            raise LLMError(
                f"не найден {self.cfg.server}: поставь llama.cpp (бинарь llama-server) или укажи "
                "путь в SUMMURAZE_LOCAL_LLM_SERVER"
            )
        port = _free_port()
        if self._slots:
            assert self.cfg.slot_dir is not None
            self.cfg.slot_dir.mkdir(parents=True, exist_ok=True)
        with self.log.open("wb") as log:
            self.process = subprocess.Popen(
                self.command(port), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
            )
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + self.cfg.start_timeout
        with httpx.Client(timeout=5.0) as client:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise LLMError(f"llama-server завершился при запуске: {self._tail()}")
                try:
                    if client.get(f"{url}/health").status_code == 200:
                        logger.info("llama-server готов: {} ({})", url, self.cfg.model)
                        return f"{url}/v1"
                except httpx.HTTPError:
                    pass  # ещё не слушает или грузит модель (503)
                time.sleep(_POLL_SECONDS)
        self.stop()
        raise LLMError(
            f"llama-server не загрузил модель за {self.cfg.start_timeout:.0f} с "
            "(SUMMURAZE_LOCAL_LLM_START_TIMEOUT)"
        )

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def peak_rss(self) -> int | None:
        """Пик RSS сервера в байтах (VmHWM), пока процесс жив."""
        if self.process is None:
            return None
        try:
            status = Path(f"/proc/{self.process.pid}/status").read_text()
        except OSError:
            return None
        for line in status.splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
        return None

    def _tail(self, lines: int = 8) -> str:
        try:
            text = self.log.read_text(errors="replace").strip().splitlines()
        except OSError:
            return "лога нет"
        return " | ".join(text[-lines:]) or "лог пуст"


class LocalLLM(OpenAICompatLLM):
    """`OpenAICompatLLM` со своим сервером: поднимается при создании, гаснет в `close()`."""

    def __init__(self, cfg: LLMSettings) -> None:
        assert cfg.server is not None
        self.server = LlamaServer(cfg.server)
        url = self.server.start()
        try:
            super().__init__(cfg.model_copy(update={"base_url": url}))
        except Exception:
            self.server.stop()
            raise

    def complete(
        self,
        *,
        system: str,
        prefix: str,
        question: str,
        max_tokens: int,
        schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        slot = self._slot_name(prefix)
        if slot is not None:
            self._slot("restore", slot)
        result = super().complete(
            system=system, prefix=prefix, question=question, max_tokens=max_tokens, schema=schema
        )
        if slot is not None and self._slot("save", slot):
            self._trim_slots()
        return result

    def _slot_name(self, prefix: str) -> str | None:
        cfg = self.server.cfg
        if not self.server._slots:
            return None
        key = f"{cfg.model}\0{cfg.context}\0{prefix}".encode()
        return f"{hashlib.sha256(key).hexdigest()[:32]}.slot"

    def _slot(self, action: str, name: str) -> bool:
        """Сохранить или поднять KV-кэш слота. Неудача — не ошибка: просто считаем префикс."""
        assert self.server.cfg.slot_dir is not None
        if action == "restore" and not (self.server.cfg.slot_dir / name).is_file():
            return False
        try:
            response = self._client.post(
                f"{self._base_url.removesuffix('/v1')}/slots/0",
                params={"action": action},
                json={"filename": name},
                headers=self._headers,
            )
        except httpx.HTTPError as e:
            logger.info("слот {} не {}: {}", name, action, e)
            return False
        if response.status_code != 200:
            logger.info("слот {} не {}: {}", name, action, response.text[:200])
            return False
        logger.info("слот {}: {}", name, action)
        if action == "restore":
            (self.server.cfg.slot_dir / name).touch()  # свежий — последним под удаление
        return True

    def _trim_slots(self) -> None:
        folder = self.server.cfg.slot_dir
        assert folder is not None
        slots = sorted(folder.glob("*.slot"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in slots[self.server.cfg.slot_cache :]:
            old.unlink(missing_ok=True)

    def close(self) -> None:
        rss = self.server.peak_rss()
        if rss is not None:
            logger.info("llama-server: пик RSS {:.1f} ГБ", rss / 2**30)
        self.server.stop()
        super().close()
