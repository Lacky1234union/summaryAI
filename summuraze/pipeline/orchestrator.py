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

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from summuraze.asr import repetition
from summuraze.asr.base import make_asr
from summuraze.asr.terms import Glossary, apply_replacements, load_glossary
from summuraze.audio.ingest import IngestError
from summuraze.audio.vad import NoSpeechError
from summuraze.config import ASRSettings, LLMSettings, Settings
from summuraze.pipeline import stages
from summuraze.pipeline.memory import release_memory
from summuraze.render.markdown import note_layout, publish_media, render_note
from summuraze.store.artifacts import ArtifactKey, ArtifactStore
from summuraze.store.db import Database, OutputKind
from summuraze.summarize.base import check_llm, make_llm
from summuraze.summarize.budget import Budget, check_speech, output_reserve
from summuraze.summarize.profiles import Profile, get_profile
from summuraze.types import ASRProvider, AudioInfo, Interval, LLMProvider, LLMResult, Transcript

StageName = Literal["ingest", "vad", "asr", "summarize", "audio", "render"]
STAGES: tuple[StageName, ...] = ("ingest", "vad", "asr", "summarize", "audio", "render")


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
    document: BaseModel | None = None
    note: Path | None = None
    cached: dict[str, bool] | None = None
    recording_id: int | None = None
    # Решение «один запрос или окна» (#16); None — саммари из кэша, запроса не было.
    budget: Budget | None = None


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

    def preflight(
        self, *, until: StageName = "render", profile: str = "lecture", terms: Path | None = None
    ) -> tuple[Profile, Glossary]:
        """Всё, что может сломать прогон, кроме самого файла, — до часа транскрибации, а не после.
        `summuraze record` зовёт это до начала записи: узнать про пустой ключ после пары поздно."""
        prof = get_profile(profile)
        glossary = load_glossary([terms, self.settings.asr.terms_file])
        if until in ("summarize", "render"):
            check_llm(self.settings.llm)
        if until == "render":
            _check_output_dir(self.settings.paths.output_dir)
        return prof, glossary

    def run(
        self,
        source: Path,
        *,
        until: StageName = "render",
        stream: int | None = None,
        profile: str = "lecture",
        terms: Path | None = None,
    ) -> RunResult:
        """`terms` — словарь курса (#13); он важнее общего `SUMMURAZE_ASR_TERMS_FILE`."""
        prof, glossary = self.preflight(until=until, profile=profile, terms=terms)
        if not source.is_file():
            raise IngestError(f"файл не найден: {source}")
        note_cfg = self.settings.note
        skipped = {"vad": not self.settings.vad.enabled, "audio": not note_cfg.audio}
        active = [s for s in STAGES if not skipped.get(s)]
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
            result = RunResult(run_id=run_id, audio=audio, cached=cached, recording_id=recording_id)
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
            llm_cfg = self.settings.llm
            if until in ("summarize", "render"):
                check_speech(
                    sum(i.duration for i in speech) if speech is not None else audio.duration,
                    model=llm_cfg.model,
                    context=llm_cfg.context,
                    max_tokens=prof.max_tokens,
                    single_pass_max=llm_cfg.single_pass_max,
                )

            asr_cfg = self.settings.asr
            asr_key = ArtifactKey(
                "asr",
                stages.ASR_VERSION,
                (asr_input,),
                # Термины меняют распознавание; без словаря ключ прежний — старый кэш жив.
                _asr_params(asr_cfg) | ({"terms": list(glossary.hint)} if glossary.hint else {}),
            )

            def build_asr(out: Path) -> None:
                asr = self._asr_factory(asr_cfg)  # вне try: упавшая фабрика — не повод для del
                try:
                    stages.build_asr(
                        out,
                        asr,
                        audio,
                        language=asr_cfg.language,
                        speech=speech,
                        terms=list(glossary.hint),
                    )
                finally:
                    del asr
                    release_memory()

            transcript = stages.load_transcript(stage("asr", asr_key, build_asr), source)
            # Замены (#13) и детектор (#12) — поверх кэша, а не внутри ASR: правка карты замен
            # или правил не стоит часа транскрибации. Замены первыми: детектор видит итоговый текст.
            segments = apply_replacements(transcript.segments, glossary)
            transcript = transcript.model_copy(
                update={"segments": repetition.flag_suspects(segments)}
            )
            result.transcript = transcript
            if until == "asr":
                self.db.finish_run(run_id, "done", asr_model=transcript.asr_model)
                return result

            # Ответ урезан малым контекстом — это другой запрос; при обычном ключ прежний.
            output = output_reserve(llm_cfg.context, prof.max_tokens)
            summary_key = ArtifactKey(
                "summarize",
                stages.SUMMARY_VERSION,
                (asr_key.digest,),
                {
                    "provider": llm_cfg.provider,
                    "model": llm_cfg.model,
                    "profile": prof.name,
                    "profile_version": prof.version,
                    "suspects": repetition.VERSION,
                    **({"max_tokens": output} if output != prof.max_tokens else {}),
                    **({"replacements": glossary.replacements} if glossary.replacements else {}),
                },
            )

            def build_summary(out: Path) -> None:
                nonlocal llm_fresh
                llm = self._llm_factory(llm_cfg)
                result.budget = stages.build_summary(
                    out, llm, transcript, prof, single_pass_max=llm_cfg.single_pass_max
                )
                llm_fresh, _ = stages.load_summary(out, prof)

            summary, document = stages.load_summary(
                stage("summarize", summary_key, build_summary), prof
            )
            result.summary = summary
            result.document = document

            if until == "render":
                note_audio: Path | None = None
                if note_cfg.audio:
                    audio_key = ArtifactKey(
                        "note-audio",
                        stages.NOTE_AUDIO_VERSION,
                        (ingest_key.digest,),
                        {"bitrate_kbps": note_cfg.audio_bitrate_kbps},
                    )
                    artifact = stage(
                        "audio",
                        audio_key,
                        lambda out: stages.build_note_audio(
                            out, audio, bitrate_kbps=note_cfg.audio_bitrate_kbps
                        ),
                    )
                    note_audio = artifact / stages.NOTE_AUDIO

                index = active.index("render") + 1
                self._on_event(StageEvent("render", index, total, "start"))
                started = time.monotonic()
                layout = note_layout(
                    self.settings.paths.notes_dir, source, prof.name, note_cfg.media_dir
                )
                media = publish_media(layout, note_audio, transcript)
                result.note = render_note(
                    layout,
                    audio=audio,
                    transcript=transcript,
                    summary=summary,
                    document=document,
                    profile=prof.name,
                    media=media,
                    debug=note_cfg.debug,
                )
                # Срок хранения (#49) найдёт аудио и заметку по БД, а не угадыванием имён.
                outputs: list[tuple[OutputKind, Path]] = [("note", result.note)]
                if media is not None:
                    outputs += [
                        ("audio", layout.audio),
                        ("subtitles", layout.subtitles(transcript.language)),
                    ]
                self.db.record_outputs(recording_id, outputs)
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


class OutputDirError(ValueError):
    """Папка заметок не найдена — выясняем до транскрибации, а не после."""


def _check_output_dir(output_dir: Path | None) -> None:
    # Свою папку по умолчанию (data/notes) создаём сами; заданную в .env — нет: опечатка в пути к
    # хранилищу молча создала бы папки не там, и заметки «пропадали» бы.
    if output_dir is not None and not output_dir.expanduser().is_dir():
        raise OutputDirError(
            f"папки для заметок нет: {output_dir} — создай её в хранилище Obsidian "
            "или поправь SUMMURAZE_OUTPUT_DIR"
        )


def _asr_params(cfg: ASRSettings) -> dict[str, Any]:
    """То, что меняет текст транскрипта. `threads` и `device` — только скорость, в ключ не идут:
    иначе смена числа потоков выкинула бы в мусор час готовой транскрибации."""
    return {
        "provider": cfg.provider,
        "model": cfg.model,
        "compute_type": cfg.compute_type,
        "language": cfg.language,
    }
