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

from summuraze.audio.ingest import ingest
from summuraze.render.markdown import fmt_timecode
from summuraze.summarize.profiles import Profile
from summuraze.types import ASRProvider, AudioInfo, LLMProvider, LLMResult, Transcript

INGEST_VERSION = 1
ASR_VERSION = 1
SUMMARY_VERSION = 1

_AUDIO = "audio.wav"
_AUDIO_INFO = "audio.json"
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


def build_asr(out: Path, asr: ASRProvider, audio: AudioInfo, *, language: str | None) -> None:
    transcript = asr.transcribe(audio.wav, language=language)
    (out / _TRANSCRIPT).write_text(transcript.model_dump_json(indent=2), encoding="utf-8")


def load_transcript(artifact: Path, source: Path) -> Transcript:
    t = Transcript.model_validate_json((artifact / _TRANSCRIPT).read_text(encoding="utf-8"))
    return t.model_copy(update={"source": source})


def transcript_prefix(transcript: Transcript) -> str:
    """Транскрипт в виде, который уходит модели первым.

    Детерминирован байт в байт: никаких дат, путей и ID прогона — иначе кэш префикса у
    провайдера не попадает, и каждый прогон стоит полную цену (CLAUDE.md, «Кэш префикса»).
    """
    return "\n".join(
        f"[{fmt_timecode(s.start)}] {s.text.strip()}" for s in transcript.segments if s.text.strip()
    )


def build_summary(out: Path, llm: LLMProvider, transcript: Transcript, profile: Profile) -> None:
    result = llm.complete(
        system=profile.system,
        prefix=transcript_prefix(transcript),
        question=profile.question,
        max_tokens=8000,
    )
    payload = {"profile": profile.name, "result": result.model_dump()}
    (out / _SUMMARY).write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")


def load_summary(artifact: Path) -> LLMResult:
    payload = json.loads((artifact / _SUMMARY).read_text(encoding="utf-8"))
    return LLMResult.model_validate(payload["result"])
