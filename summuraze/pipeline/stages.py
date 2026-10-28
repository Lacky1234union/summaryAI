"""Стадии конвейера: что каждая пишет в свой каталог артефакта и как это прочитать обратно.

Версия стадии — часть ключа кэша (guardrail #4). Меняешь, что стадия кладёт в артефакт или как
считает, — поднимай версию, иначе кэш отдаст результат старой логики.

Пути внутри артефактов хранятся относительными: артефакт собирается во временном каталоге и
потом переименовывается, так что абсолютный путь в момент записи был бы уже мёртвым.

Refs #7
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, TypeAdapter, ValidationError

from summuraze.audio.ingest import compress_for_note, ingest
from summuraze.audio.vad import detect_speech
from summuraze.config import VADSettings
from summuraze.render.markdown import fmt_timecode
from summuraze.render.subtitles import SUSPECT_MARK
from summuraze.summarize.profiles import Profile
from summuraze.summarize.schema import strict_schema
from summuraze.types import (
    ASRProvider,
    AudioInfo,
    Interval,
    LLMProvider,
    LLMResult,
    Transcript,
)

INGEST_VERSION = 1
NOTE_AUDIO_VERSION = 1
VAD_VERSION = 1
# 2: ASR получает интервалы речи от VAD (#10).
ASR_VERSION = 2
# 2: сомнительные сегменты помечены в префиксе, правило про пометку — в system (#12).
# 3: ответ — JSON по схеме профиля, в артефакте разобранный документ (#17).
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
    return "\n".join(
        f"[{fmt_timecode(s.start)}] {SUSPECT_MARK + ' ' if s.suspect else ''}{s.text.strip()}"
        for s in transcript.segments
        if s.text.strip()
    )


class SummaryFormatError(RuntimeError):
    """Модель ответила не по схеме профиля — такой ответ в кэш не кладём."""


def build_summary(out: Path, llm: LLMProvider, transcript: Transcript, profile: Profile) -> None:
    result = llm.complete(
        system=f"{profile.system}\n\n{SUSPECT_RULE}",
        prefix=transcript_prefix(transcript),
        question=profile.question,
        max_tokens=profile.max_tokens,
        schema=strict_schema(profile.document),
    )
    try:
        document = profile.document.model_validate_json(result.text)
    except ValidationError as e:
        raise SummaryFormatError(
            f"модель {result.model} ответила не по схеме профиля {profile.name!r} "
            f"(обрезан ответ или сервер не поддерживает structured outputs): "
            f"{e.errors()[0]['msg']}; начало ответа: {result.text[:200]!r}"
        ) from e
    payload = {
        "profile": profile.name,
        "document": document.model_dump(),
        "result": result.model_dump(),
    }
    (out / _SUMMARY).write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")


def load_summary(artifact: Path, profile: Profile) -> tuple[LLMResult, BaseModel]:
    payload = json.loads((artifact / _SUMMARY).read_text(encoding="utf-8"))
    result = LLMResult.model_validate(payload["result"])
    return result, profile.document.model_validate(payload["document"])
