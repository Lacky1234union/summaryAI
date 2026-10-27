"""Прогон конвейера: ingest → vad → asr → summarize → render, каждая стадия через кэш.

Последовательность явная, а не «движок DAG»: стадий пять, и прочитать их сверху вниз проще,
чем разбирать граф.

Провайдеры создаются **внутри** сборки стадии, то есть только при промахе кэша. Два следствия:
повторный прогон не грузит модель Whisper даже в память, и модель ASR перестаёт существовать
до того, как создаётся LLM — ASR и локальная LLM вместе в 14 ГБ не влезают (CLAUDE.md, «Память»).

Рендер не кэшируется: это дешёвая запись заметки, и она должна происходить при каждом прогоне —
заметку могли удалить или переместить.

Refs #7, #10
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from summuraze.asr.base import make_asr
from summuraze.audio.ingest import IngestError
from summuraze.audio.vad import NoSpeechError
from summuraze.config import ASRSettings, LLMSettings, Settings
from summuraze.pipeline import stages
from summuraze.render.markdown import note_path, render_note
from summuraze.store.artifacts import ArtifactKey, ArtifactStore
from summuraze.store.db import Database
from summuraze.summarize.base import make_llm
from summuraze.summarize.profiles import get_profile
from summuraze.types import ASRProvider, AudioInfo, Interval, LLMProvider, LLMResult, Transcript

StageName = Literal["ingest", "vad", "asr", "summarize", "render"]
STAGES: tuple[StageName, ...] = ("ingest", "vad", "asr", "summarize", "render")


@dataclass(frozen=True)
class StageEvent:
    stage: StageName
    index: int
    total: int
    status: Literal["start", "done", "cached"]
    seconds: float = 0.0


@dataclass
class RunResult:
    run_id: int
    audio: AudioInfo
    # None — VAD выключен, ASR получил запись целиком.
    speech: list[Interval] | None = None
    transcript: Transcript | None = None
    summary: LLMResult | None = None
    note: Path | None = None
    cached: dict[str, bool] | None = None


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        asr_factory: Callable[[ASRSettings], ASRProvider] = make_asr,
        llm_factory: Callable[[LLMSettings], LLMProvider] = make_llm,
        on_event: Callable[[StageEvent], None] | None = None,
    ) -> None:
        self.settings = settings
        self.store = ArtifactStore(settings.paths.artifacts_dir)
        self.db = Database(settings.paths.db_path)
        self._asr_factory = asr_factory
        self._llm_factory = llm_factory
        self._on_event = on_event or (lambda e: None)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Pipeline:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def run(
        self,
        source: Path,
        *,
        until: StageName = "render",
        stream: int | None = None,
        profile: str = "lecture",
    ) -> RunResult:
        prof = get_profile(profile)  # неизвестный профиль — до часа транскрибации, а не после
        if not source.is_file():
            raise IngestError(f"файл не найден: {source}")
        active = [s for s in STAGES if s != "vad" or self.settings.vad.enabled]
        total = active.index(until) + 1
        recording_id, source_hash = self.db.register_source(source)
        run_id = self.db.start_run(recording_id)
        cached: dict[str, bool] = {}
        llm_fresh: LLMResult | None = None

        def stage(name: StageName, key: ArtifactKey, build: Callable[[Path], None]) -> Path:
            index = active.index(name) + 1
            self._on_event(StageEvent(name, index, total, "start"))
            started = time.monotonic()
            path, hit = self.store.get_or_build(key, build)
            seconds = time.monotonic() - started
            cached[name] = hit
            self.db.record_stage(run_id, name, key.digest, cached=hit, seconds=seconds)
            self._on_event(StageEvent(name, index, total, "cached" if hit else "done", seconds))
            return path

        try:
            ingest_key = ArtifactKey(
                "ingest", stages.INGEST_VERSION, (source_hash,), {"stream": stream}
            )
            audio = stages.load_ingest(
                stage(
                    "ingest",
                    ingest_key,
                    lambda out: stages.build_ingest(out, source, stream=stream),
                ),
                source,
            )
            self.db.set_duration(recording_id, audio.duration)
            result = RunResult(run_id=run_id, audio=audio, cached=cached)
            if until == "ingest":
                self.db.finish_run(run_id, "done")
                return result

            speech: list[Interval] | None = None
            asr_input = ingest_key.digest
            vad_cfg = self.settings.vad
            if vad_cfg.enabled:
                vad_key = ArtifactKey(
                    "vad",
                    stages.VAD_VERSION,
                    (ingest_key.digest,),
                    vad_cfg.model_dump(exclude={"enabled"}),
                )
                speech = stages.load_vad(
                    stage("vad", vad_key, lambda out: stages.build_vad(out, audio, vad_cfg))
                )
                result.speech = speech
                asr_input = vad_key.digest
                if until == "vad":
                    self.db.finish_run(run_id, "done")
                    return result
                if not speech:
                    raise NoSpeechError(
                        "в записи не найдено речи — только тишина или музыка. Если речь там "
                        "есть, понизь SUMMURAZE_VAD_THRESHOLD или выключи VAD"
                    )

            asr_cfg = self.settings.asr
            asr_key = ArtifactKey(
                "asr",
                stages.ASR_VERSION,
                (asr_input,),
                _asr_params(asr_cfg),
            )

            def build_asr(out: Path) -> None:
                asr = self._asr_factory(asr_cfg)  # вне try: упавшая фабрика — не повод для del
                try:
                    stages.build_asr(out, asr, audio, language=asr_cfg.language, speech=speech)
                finally:
                    del asr
                    gc.collect()

            transcript = stages.load_transcript(stage("asr", asr_key, build_asr), source)
            result.transcript = transcript
            if until == "asr":
                self.db.finish_run(run_id, "done", asr_model=transcript.asr_model)
                return result

            llm_cfg = self.settings.llm
            summary_key = ArtifactKey(
                "summarize",
                stages.SUMMARY_VERSION,
                (asr_key.digest,),
                {
                    "provider": llm_cfg.provider,
                    "model": llm_cfg.model,
                    "profile": prof.name,
                    "profile_version": prof.version,
                },
            )

            def build_summary(out: Path) -> None:
                nonlocal llm_fresh
                llm = self._llm_factory(llm_cfg)
                stages.build_summary(out, llm, transcript, prof)
                llm_fresh = stages.load_summary(out)

            summary = stages.load_summary(stage("summarize", summary_key, build_summary))
            result.summary = summary

            if until == "render":
                index = active.index("render") + 1
                self._on_event(StageEvent("render", index, total, "start"))
                started = time.monotonic()
                result.note = render_note(
                    note_path(self.settings.paths.notes_dir, source),
                    audio=audio,
                    transcript=transcript,
                    summary=summary,
                    profile=prof.name,
                )
                seconds = time.monotonic() - started
                cached["render"] = False
                self.db.record_stage(run_id, "render", "-", cached=False, seconds=seconds)
                self._on_event(StageEvent("render", index, total, "done", seconds))

            # Стоимость — только того, что реально ушло в модель в ЭТОМ прогоне: саммари из кэша
            # ничего не стоило, и записать ему цену первого прогона значило бы врать в `log`.
            self.db.finish_run(
                run_id,
                "done",
                asr_model=transcript.asr_model,
                llm_provider=llm_cfg.provider if llm_cfg.provider == "echo" else llm_cfg.base_url,
                llm_model=summary.model,
                usage=llm_fresh.usage if llm_fresh else None,
                cost_usd=llm_fresh.cost_usd if llm_fresh else 0.0,
            )
            return result
        except BaseException as e:
            self.db.finish_run(run_id, "failed", error=f"{type(e).__name__}: {e}")
            raise


def _asr_params(cfg: ASRSettings) -> dict[str, Any]:
    """То, что меняет текст транскрипта. `threads` и `device` — только скорость, в ключ не идут:
    иначе смена числа потоков выкинула бы в мусор час готовой транскрибации."""
    return {
        "provider": cfg.provider,
        "model": cfg.model,
        "compute_type": cfg.compute_type,
        "language": cfg.language,
    }
