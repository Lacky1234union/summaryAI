"""Своя модель через llama-server: `--local`, жизнь процесса, порядок выгрузки (#19).

Настоящий llama-server с моделью на 5 ГБ в тестах не поднимаем: вместо него — скрипт с тем же
HTTP-протоколом (`/health`, `/tokenize`, `/v1/chat/completions`), запущенный тем же путём.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import weakref
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from summuraze.cli import app
from summuraze.config import ASRSettings, LLMSettings, LocalLLMSettings, Settings, get_settings
from summuraze.pipeline.orchestrator import Pipeline
from summuraze.summarize.base import make_llm
from summuraze.summarize.echo import EchoLLM
from summuraze.summarize.llama_server import LlamaServer, LocalLLM, physical_cores
from summuraze.summarize.openai_compat import LLMError
from summuraze.types import LLMProvider, Transcript
from tests.conftest import SPEECH_CLIPS

FAKE_SERVER = r"""
import json, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
open(__file__ + ".args", "w").write(json.dumps(args))
if "--die" in args:
    print("error: model is corrupted", flush=True)
    sys.exit(1)
time.sleep(0.3)  # «грузит модель»

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        self._send({"status": "ok"})
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.startswith("/slots/0"):
            folder = args[args.index("--slot-save-path") + 1]
            path = folder + "/" + req["filename"]
            with open(__file__ + ".slots", "a") as log:
                log.write(self.path.split("=")[1] + " " + req["filename"] + "\n")
            if self.path.endswith("save"):
                open(path, "w").write("kv")
            self._send({"n_saved": 1} if self.path.endswith("save") else {"n_restored": 1})
            return
        if self.path == "/tokenize":
            self._send({"tokens": list(range(len(req["content"]) // 4))})
            return
        schema = req["response_format"]["json_schema"]["schema"]
        props = schema["properties"].items()
        doc = {k: ([] if v.get("type") == "array" else "Лекция") for k, v in props}
        if "tldr" in doc:  # пустой конспект конвейер не примет (#63)
            doc["tldr"] = [{"text": "Лекция о сигналах", "at": "00:00:01"}]
        self._send({
            "model": req["model"],
            "choices": [{"message": {"content": json.dumps(doc)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30,
                      "prompt_tokens_details": {"cached_tokens": 0}},
        })

HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


@pytest.fixture
def fake_server(tmp_path: Path) -> Path:
    script = tmp_path / "llama-server"
    script.write_text(f"#!{sys.executable}\n{FAKE_SERVER}")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


@pytest.fixture
def model(tmp_path: Path) -> Path:
    path = tmp_path / "qwen-test.gguf"
    path.write_bytes(b"GGUF")
    return path


def _local(server: Path, model: Path, **kw: Any) -> LocalLLMSettings:
    return LocalLLMSettings(model=model, server=str(server), **{"start_timeout": 10, **kw})


def _args(server: Path) -> list[str]:
    return list(json.loads(Path(str(server) + ".args").read_text()))


class TestServer:
    def test_command(self, model: Path) -> None:
        cmd = LlamaServer(_local(Path("llama-server"), model, threads=6, args=["--x"])).command(9)
        assert cmd[1:] == [
            *("-m", str(model), "-c", "16384", "-t", "6"),
            *("--host", "127.0.0.1", "--port", "9", "--parallel", "1"),
            *("-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--cache-ram", "0"),
            "--no-webui",
            *("--alias", "qwen-test"),
            "--x",
        ]

    def test_threads_default_to_physical_cores(self, model: Path) -> None:
        cmd = LlamaServer(_local(Path("llama-server"), model)).command(9)
        assert cmd[cmd.index("-t") + 1] == str(physical_cores())
        assert 1 <= physical_cores() <= (os.cpu_count() or 1)

    def test_local_llm_round_trip_and_stop(self, fake_server: Path, model: Path) -> None:
        cfg = Settings().model_copy(update={"local_llm": _local(fake_server, model)}).localized()
        llm = make_llm(cfg.llm)
        assert isinstance(llm, LocalLLM)
        try:
            assert llm.count_tokens("я" * 40) == 10, "точный счёт через /tokenize"
            result = llm.complete(
                system="s",
                prefix="p",
                question="q",
                max_tokens=100,
                schema={"type": "object", "properties": {"title": {"type": "string"}}},
            )
            assert json.loads(result.text) == {"title": "Лекция"}
            assert result.cost_usd == 0.0, "своя модель бесплатна"
            process = llm.server.process
            assert process is not None and process.poll() is None
        finally:
            llm.close()
        assert process.poll() is not None, "сервер погашен — память отдана системе"
        assert "-m" in _args(fake_server) and str(model) in _args(fake_server)

    def test_prefix_kv_survives_server_restart(
        self, fake_server: Path, model: Path, tmp_path: Path
    ) -> None:
        slots = tmp_path / "slots"
        cfg = Settings().model_copy(
            update={"local_llm": _local(fake_server, model, slot_dir=slots, slot_cache=2)}
        )
        schema = {"type": "object", "properties": {"title": {"type": "string"}}}

        def ask(prefix: str) -> None:
            llm = make_llm(cfg.localized().llm)
            try:
                llm.complete(system="s", prefix=prefix, question="q", max_tokens=9, schema=schema)
            finally:
                llm.close()

        ask("транскрипт А")
        ask("транскрипт А")  # другой процесс сервера — кэш с диска
        ask("транскрипт Б")
        ask("транскрипт В")
        log = Path(str(fake_server) + ".slots").read_text().split()
        actions = log[::2]
        assert actions == ["save", "restore", "save", "save", "save"], "А поднят из файла"
        assert "--slot-save-path" in _args(fake_server)
        assert len(list(slots.glob("*.slot"))) == 2, "держим последние slot_cache штук"

    def test_slot_cache_off(self, fake_server: Path, model: Path, tmp_path: Path) -> None:
        local = _local(fake_server, model, slot_dir=tmp_path / "s", slot_cache=0)
        assert "--slot-save-path" not in LlamaServer(local).command(9)

    def test_crash_on_start_is_explained(self, fake_server: Path, model: Path) -> None:
        server = LlamaServer(_local(fake_server, model, args=["--die"]))
        with pytest.raises(LLMError, match="model is corrupted"):
            server.start()

    def test_missing_binary(self, model: Path) -> None:
        with pytest.raises(LLMError, match="SUMMURAZE_LOCAL_LLM_SERVER"):
            LlamaServer(_local(Path("/нет/llama-server"), model)).start()

    def test_slow_load_times_out_and_kills(self, tmp_path: Path, model: Path) -> None:
        script = tmp_path / "slow"
        script.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
        script.chmod(0o755)
        server = LlamaServer(_local(script, model, start_timeout=1))
        with pytest.raises(LLMError, match="не загрузил модель за 1 с"):
            server.start()
        assert server.process is not None and server.process.poll() is not None


class TestSettings:
    def test_localized_points_at_loopback_and_forbids_network(self, model: Path) -> None:
        cfg = Settings().model_copy(update={"local_llm": _local(Path("x"), model)}).localized()
        assert cfg.local_only and cfg.llm.local_only and cfg.asr.local_only
        assert cfg.llm.is_local and cfg.llm.model == "qwen-test"
        assert cfg.llm.context == 16384 and cfg.llm.server is not None
        assert not cfg.llm.api_key.get_secret_value(), "ключ API своей модели не нужен"

    def test_localized_needs_model(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="SUMMURAZE_LOCAL_LLM_MODEL"):
            Settings().localized()
        missing = Settings().model_copy(
            update={"local_llm": LocalLLMSettings(model=tmp_path / "нет.gguf")}
        )
        with pytest.raises(ValueError, match="нет файла модели"):
            missing.localized()

    def test_env(self, monkeypatch: pytest.MonkeyPatch, model: Path) -> None:
        monkeypatch.setenv("SUMMURAZE_LOCAL_LLM_MODEL", str(model))
        monkeypatch.setenv("SUMMURAZE_LOCAL_LLM_CONTEXT", "32768")
        monkeypatch.setenv("SUMMURAZE_LOCAL_LLM_ARGS", '["--reasoning-budget", "0"]')
        monkeypatch.setenv("SUMMURAZE_LOCAL_LLM_THREADS", "")
        local = Settings().local_llm
        assert local.model == model and local.context == 32768 and local.threads is None
        assert local.args == ["--reasoning-budget", "0"]


@pytest.fixture
def lecture(tmp_path: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("нужен ffmpeg")
    path = tmp_path / "лекция.opus"
    shutil.copy(SPEECH_CLIPS["speech"], path)
    return path


class TestCLI:
    def test_local_without_model_fails_before_ingest(self, lecture: Path) -> None:
        out = CliRunner().invoke(app, ["run", str(lecture), "--local"])
        assert out.exit_code == 1 and "SUMMURAZE_LOCAL_LLM_MODEL" in out.output
        assert "[1/" not in out.output

    def test_local_run_end_to_end(
        self, lecture: Path, fake_server: Path, model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SUMMURAZE_LLM_PROVIDER", "openai-compat")
        monkeypatch.setenv("SUMMURAZE_LLM_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("SUMMURAZE_LOCAL_LLM_MODEL", str(model))
        monkeypatch.setenv("SUMMURAZE_LOCAL_LLM_SERVER", str(fake_server))
        get_settings.cache_clear()
        out = CliRunner().invoke(app, ["run", str(lecture), "--local"])
        assert out.exit_code == 0, out.output
        assert "токенизатор сервера" in out.output, "бюджет посчитан самим сервером"
        assert "разрешить" not in out.output, "согласие на отправку не спрашивается"
        log = CliRunner().invoke(app, ["log"]).output
        assert "локально · qwen-test" in log


def test_asr_is_released_before_llm_starts(lecture: Path) -> None:
    """Whisper и своя модель вместе в 14 ГБ не влезают: ASR выгружен до запуска сервера."""
    order: list[str] = []

    class TrackedASR:
        name = "tracked"

        def __init__(self) -> None:
            from summuraze.asr.echo import EchoASR

            self.inner = EchoASR()
            weakref.finalize(self, order.append, "asr выгружен")

        def transcribe(self, audio: Path, **kw: Any) -> Transcript:
            return self.inner.transcribe(audio, **kw)

    class TrackedLLM(EchoLLM):
        def close(self) -> None:
            order.append("llm закрыт")

    def make_asr(cfg: ASRSettings) -> TrackedASR:
        return TrackedASR()

    def make_llm_(cfg: LLMSettings) -> LLMProvider:
        order.append("llm запущен")
        return TrackedLLM()

    with Pipeline(Settings(), asr_factory=make_asr, llm_factory=make_llm_) as p:  # type: ignore[arg-type]
        p.run(lecture)
    assert order == ["asr выгружен", "llm запущен", "llm закрыт"]
