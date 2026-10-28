"""Стадии конвейера: что каждая пишет в свой каталог артефакта и как это прочитать обратно.

Версия стадии — часть ключа кэша (guardrail #4). Меняешь, что стадия кладёт в артефакт или как
считает, — поднимай версию, иначе кэш отдаст результат старой логики.

Пути внутри артефактов хранятся относительными: артефакт собирается во временном каталоге и
потом переименовывается, так что абсолютный путь в момент записи был бы уже мёртвым.

Refs #7
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, TypeAdapter, ValidationError

from summuraze.audio.ingest import compress_for_note, ingest
from summuraze.audio.vad import detect_speech
from summuraze.config import VADSettings
from summuraze.render.markdown import fmt_timecode, resolve_at
from summuraze.render.subtitles import SUSPECT_MARK
from summuraze.summarize import budget, windows
from summuraze.summarize.profiles import Profile, RawDoc
from summuraze.summarize.schema import strict_schema
from summuraze.types import (
    ASRProvider,
    AudioInfo,
    Interval,
    LLMProvider,
    LLMResult,
    Segment,
    TokenUsage,
    Transcript,
)

INGEST_VERSION = 1
NOTE_AUDIO_VERSION = 1
VAD_VERSION = 1
# 2: ASR получает интервалы речи от VAD (#10).
ASR_VERSION = 2
# 2: сомнительные сегменты помечены в префиксе, правило про пометку — в system (#12).
# 3: ответ — JSON по схеме профиля, в артефакте разобранный документ (#17).
# Окна (#21) версию не поднимают: у них своя, `windows.VERSION`, в ключе длинных записей.
SUMMARY_VERSION = 3

_AUDIO = "audio.wav"
_AUDIO_INFO = "audio.json"
_SPEECH = "speech.json"
_TRANSCRIPT = "transcript.json"
_SUMMARY = "summary.json"


def build_ingest(out: Path, source: Path, *, stream: int | None) -> None:
    info = ingest(source, out / _AUDIO, stream=stream)
    info = info.model_copy(update={"wav": Path(_AUDIO)})
    (out / _AUDIO_INFO).write_text(info.model_dump_json(indent=2), encoding="utf-8")


def load_ingest(artifact: Path, source: Path) -> AudioInfo:
    info = AudioInfo.model_validate_json((artifact / _AUDIO_INFO).read_text(encoding="utf-8"))
    # Ключ — содержимое, а не путь: тот же файл мог переехать с прошлого прогона.
    return info.model_copy(update={"wav": artifact / info.wav, "source": source})


_Intervals = TypeAdapter(list[Interval])


NOTE_AUDIO = "audio.ogg"


def build_note_audio(out: Path, audio: AudioInfo, *, bitrate_kbps: int) -> None:
    compress_for_note(audio.wav, out / NOTE_AUDIO, bitrate_kbps=bitrate_kbps)


def build_vad(out: Path, audio: AudioInfo, cfg: VADSettings) -> None:
    speech = detect_speech(audio.wav, cfg)
    (out / _SPEECH).write_bytes(_Intervals.dump_json(speech, indent=2))


def load_vad(artifact: Path) -> list[Interval]:
    return _Intervals.validate_json((artifact / _SPEECH).read_bytes())


def build_asr(
    out: Path,
    asr: ASRProvider,
    audio: AudioInfo,
    *,
    language: str | None,
    speech: list[Interval] | None,
    terms: list[str],
) -> None:
    transcript = asr.transcribe(audio.wav, language=language, speech=speech, terms=terms or None)
    (out / _TRANSCRIPT).write_text(transcript.model_dump_json(indent=2), encoding="utf-8")


def load_transcript(artifact: Path, source: Path) -> Transcript:
    t = Transcript.model_validate_json((artifact / _TRANSCRIPT).read_text(encoding="utf-8"))
    return t.model_copy(update={"source": source})


# Уходит в system после промпта профиля: модель должна знать, что значит пометка в транскрипте.
SUSPECT_RULE = (
    f"Строки транскрипта с пометкой {SUSPECT_MARK} распознаны ненадёжно: там тишина, шум или "
    "текст, который распознаватель мог придумать. Не опирайся на них как на факт; если без "
    "них теряется смысл, напиши, что в этом месте запись неразборчива."
)


def transcript_prefix(transcript: Transcript) -> str:
    """Транскрипт в виде, который уходит модели первым.

    Детерминирован байт в байт: никаких дат, путей и ID прогона — иначе кэш префикса у
    провайдера не попадает, и каждый прогон стоит полную цену (CLAUDE.md, «Кэш префикса»).
    Сомнительные сегменты не выкидываются, а помечаются (#12): «здесь плохо слышно» — тоже
    информация для конспекта.
    """
    return "\n".join(_line(s) for s in transcript.segments if s.text.strip())


def _line(s: Segment) -> str:
    return f"[{fmt_timecode(s.start)}] {SUSPECT_MARK + ' ' if s.suspect else ''}{s.text.strip()}"


class SummaryFormatError(RuntimeError):
    """Модель ответила не по схеме профиля — такой ответ в кэш не кладём."""


def build_summary(
    out: Path,
    llm: LLMProvider,
    transcript: Transcript,
    profile: Profile,
    *,
    single_pass_max: int | None = None,
) -> budget.Budget:
    """Один запрос на весь транскрипт, если влезает (#16); иначе — окна и второй проход (#21)."""
    system = f"{profile.system}\n\n{SUSPECT_RULE}"
    prefix = transcript_prefix(transcript)
    # Схему ответа не считаем: llama.cpp делает из неё грамматику, а не промпт, и у DeepSeek
    # в prompt_tokens её почти нет (замер 2026-10-28: 1343 токена против 1.4k оценки без схемы).
    parts = [prefix, system, profile.question]
    plan = budget.plan(llm, parts, max_tokens=profile.max_tokens, single_pass_max=single_pass_max)
    if plan.mode == "single":
        result, document = _ask(
            llm,
            profile.document,
            f"профиля {profile.name!r}",
            system,
            prefix,
            profile.question,
            plan.output,
        )
    else:
        result, document, plan = _windowed(llm, transcript, profile, plan, parts)
    payload = {
        "profile": profile.name,
        "document": document.model_dump(),
        "result": result.model_dump(),
        # Как делали: при разборе плохого конспекта первым делом нужно знать, были ли окна.
        "mode": plan.mode,
        "windows": plan.windows,
        "final": plan.final,
    }
    (out / _SUMMARY).write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    return plan


def _ask(
    llm: LLMProvider,
    document: type[BaseModel],
    what: str,
    system: str,
    prefix: str,
    question: str,
    max_tokens: int,
) -> tuple[LLMResult, BaseModel]:
    result = llm.complete(
        system=system,
        prefix=prefix,
        question=question,
        max_tokens=max_tokens,
        schema=strict_schema(document),
    )
    try:
        return result, document.model_validate_json(result.text)
    except ValidationError as e:
        raise SummaryFormatError(
            f"модель {result.model} ответила не по схеме {what} "
            f"(обрезан ответ или сервер не поддерживает structured outputs): "
            f"{e.errors()[0]['msg']}; начало ответа: {result.text[:200]!r}"
        ) from e


# Окно меньше — карта из него почти пустая, а окон столько, что второй проход утонет в картах.
MIN_WINDOW_TOKENS = 1000


def _windowed(
    llm: LLMProvider,
    transcript: Transcript,
    profile: Profile,
    plan: budget.Budget,
    parts: list[str],
) -> tuple[LLMResult, BaseModel, budget.Budget]:
    """Проход 1 — карта каждого окна, проход 2 — документ профиля по склеенной карте (#21)."""
    lines = [s for s in transcript.segments if s.text.strip()]
    # Токены строк — оценкой, но в масштабе точного счёта, если сервер его дал: один запрос
    # к /tokenize на весь транскрипт вместо тысячи запросов на каждую строку.
    scale = (plan.prompt - budget.TEMPLATE_TOKENS) / max(
        budget.estimate_tokens("\n".join(parts)), 1
    )

    def tokens(text: str) -> int:
        return math.ceil(budget.estimate_tokens(text + "\n") * scale)

    probe = windows.map_system(99, 99, transcript.duration, transcript.duration, SUSPECT_RULE)
    question = windows.MAP_QUESTION if profile.two_pass else profile.question
    system = probe if profile.two_pass else f"{profile.system}\n\n{SUSPECT_RULE}"
    window_limit = plan.limit - tokens(system + question) - budget.TEMPLATE_TOKENS
    if window_limit < MIN_WINDOW_TOKENS:
        raise budget.TranscriptTooLong(
            budget.too_long(llm.model, f"на окно остаётся {window_limit} токенов")
        )
    cut = windows.split(lines, [tokens(_line(s)) for s in lines], window_limit)
    logger.info("саммари {}: {} окон по ≤ {} токенов", llm.model, len(cut), window_limit)

    results: list[LLMResult] = []
    pieces: list[tuple[windows.Window, BaseModel]] = []
    for i, window in enumerate(cut, 1):
        window_prefix = "\n".join(_line(s) for s in window.segments)
        if profile.two_pass:
            system = windows.map_system(i, len(cut), window.start, window.end, SUSPECT_RULE)
            result, doc = _ask(
                llm,
                windows.WindowMap,
                f"карты окна {i}",
                system,
                window_prefix,
                question,
                plan.output,
            )
        else:
            result, doc = _ask(
                llm,
                profile.document,
                f"профиля {profile.name!r}",
                system,
                window_prefix,
                question,
                plan.output,
            )
        results.append(result)
        pieces.append((window, doc))

    def resolve(at: str) -> float | None:
        return resolve_at(at, transcript)

    final: budget.Final
    if not profile.two_pass:
        if profile.document is not RawDoc:
            raise NotImplementedError(f"склейка по окнам для профиля {profile.name!r}")
        document: BaseModel = windows.merge_blocks(
            [(w, RawDoc.model_validate(d.model_dump())) for w, d in pieces], resolve
        )
        final = "merge"
    else:
        merged = windows.merge_maps(
            [(w, windows.WindowMap.model_validate(d.model_dump())) for w, d in pieces], resolve
        )
        outline = windows.map_text(merged)
        base = f"{profile.system}\n\n{SUSPECT_RULE}"
        prefix = transcript_prefix(transcript)
        # Транскрипт первым и байт в байт, как в одном проходе: кэш префикса тот же. Порог —
        # контекст, без SINGLE_PASS_MAX: тот ограничивает транскрипт без опоры, а с картой
        # модель уже видит структуру. Иначе итог с транскриптом не влез бы никогда.
        system = f"{base}\n\n{windows.FINAL_WITH_TRANSCRIPT}\n\n{outline}"
        second = budget.plan(llm, [prefix, system, profile.question], max_tokens=profile.max_tokens)
        final = "transcript"
        if second.mode != "single":
            prefix, system = outline, f"{base}\n\n{windows.FINAL_FROM_MAP}"
            second = budget.plan(
                llm, [prefix, system, profile.question], max_tokens=profile.max_tokens
            )
            final = "map"
            if second.mode != "single":
                raise budget.TranscriptTooLong(
                    budget.too_long(llm.model, f"карта всей записи — {second.describe()}")
                )
        result, document = _ask(
            llm,
            profile.document,
            f"профиля {profile.name!r}",
            system,
            prefix,
            profile.question,
            plan.output,
        )
        results.append(result)
    return _combined(results), document, replace(plan, windows=len(cut), final=final)


def _combined(results: list[LLMResult]) -> LLMResult:
    """Все запросы прогона одной строкой лога: токены и деньги — сумма, текст — итоговый."""
    costs = [r.cost_usd for r in results]
    return LLMResult(
        text=results[-1].text,
        model=results[-1].model,
        usage=TokenUsage(
            input=sum(r.usage.input for r in results),
            output=sum(r.usage.output for r in results),
            cached_input=sum(r.usage.cached_input for r in results),
        ),
        cost_usd=None if None in costs else sum(c for c in costs if c is not None),
    )


def load_summary(artifact: Path, profile: Profile) -> tuple[LLMResult, BaseModel]:
    payload = json.loads((artifact / _SUMMARY).read_text(encoding="utf-8"))
    result = LLMResult.model_validate(payload["result"])
    return result, profile.document.model_validate(payload["document"])
