"""Замер #70: прогон целиком теряет речь — с `condition_on_previous_text` и без.

Не тест, а инструмент замера (`docs/BENCHMARKS.md`); запускается руками:

    uv run python tests/eval/asr_coverage.py run "АиСД" --mode nocond          # вся запись
    uv run python tests/eval/asr_coverage.py run "АиСД" --mode base --from 0 --to 20
    uv run python tests/eval/asr_coverage.py compare "АиСД" --to 20

`run` повторяет путь адаптера (`FasterWhisperASR.transcribe`): речь от VAD на отрезке
`--from`…`--to` минут склеивается и уходит Whisper одним массивом, таймкоды — обратно в исходник.
Отличаются параметры декодера (`MODES`): `base` — с текстом прошлого окна (умолчание
faster-whisper, как до #70), `nocond` — без него, `hst` — без него и с
`hallucination_silence_threshold`. Результат — JSON в `data/bench/asr-70/`.

`compare` кладёт рядом все найденные варианты отрезка: эти JSON, транскрипт из кэша (`cache`) и
журнал кусков #69 (`chunks`), если он есть. Эталона нет, поэтому смотрим не WER, а признаки потери:
покрытие речи сегментами, слова, долю сегментов из 1–2 слов, слова по 5-минуткам, долю помеченного
детектором #12 и средний logprob.

Refs #70
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from live_vs_after import Recording, find_recording, norm_words, wer

from summuraze.asr import repetition
from summuraze.asr.faster_whisper import FasterWhisperASR, _to_source
from summuraze.asr.timeline import SpeechTimeline
from summuraze.audio.vad import detect_speech
from summuraze.audio.wav import SAMPLE_RATE, read_pcm
from summuraze.config import Settings
from summuraze.types import Interval, Segment, Transcript

OUT_DIR = Path("data/bench/asr-70")
CHUNKS_DIR = Path("data/bench/live-69")
# Режим → параметры Whisper поверх тех, что ставит адаптер. `hst` — без прошлого текста и с
# `hallucination_silence_threshold`: выдумка на тишине не уводит окно вперёд вместе с речью.
MODES: dict[str, dict[str, object]] = {
    "base": {"condition_on_previous_text": True},
    "nocond": {"condition_on_previous_text": False},
    "hst": {"condition_on_previous_text": False, "hallucination_silence_threshold": 2.0},
}


def clip(speech: list[Interval], a: float, b: float) -> list[Interval]:
    return [
        Interval(start=max(s.start, a), end=min(s.end, b))
        for s in speech
        if s.end > a and s.start < b
    ]


def out_path(rec: Recording, a: float, b: float | None, mode: str) -> Path:
    span = f"{a:g}-{b:g}" if b is not None else "all"
    return OUT_DIR / f"{rec.name} {span} {mode}.json"


def cmd_run(rec: Recording, args: argparse.Namespace, settings: Settings) -> int:
    a, b = args.start * 60, (args.end * 60 if args.end is not None else rec.duration)
    speech = clip(detect_speech(rec.wav, settings.vad), a, b)
    timeline = SpeechTimeline.from_intervals(speech)
    pcm = read_pcm(rec.wav)
    packed = np.concatenate(
        [pcm[int(c.start * SAMPLE_RATE) : int(c.end * SAMPLE_RATE)] for c in timeline.chunks]
    )
    del pcm
    asr = FasterWhisperASR(settings.asr)
    t0 = time.monotonic()
    raw, _ = asr._model.transcribe(
        packed,
        language=settings.asr.language,
        word_timestamps=True,
        vad_filter=False,
        **MODES[args.mode],  # type: ignore[arg-type]
    )
    segments = [s for seg in raw for s in _to_source(seg, timeline)]
    elapsed = time.monotonic() - t0
    out = out_path(rec, args.start, args.end, args.mode)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "needle": args.recording,
                "a": args.start,
                "b": args.end,
                "mode": args.mode,
                "elapsed": elapsed,
                "speech": timeline.packed_duration,
                "segments": [s.model_dump() for s in segments],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        f"{out}: {len(segments)} сегментов, {elapsed:.0f} с, RTF к речи "
        f"{elapsed / max(timeline.packed_duration, 1):.3f}"
    )
    return 0


def variants(rec: Recording, a: float, b: float | None) -> dict[str, list[Segment]]:
    lo, hi = a * 60, (b * 60 if b is not None else float("inf"))

    def inside(segments: list[Segment]) -> list[Segment]:
        return [s for s in segments if lo <= s.start < hi]

    found: dict[str, list[Segment]] = {}
    for mode in MODES:
        p = out_path(rec, a, b, mode)
        if p.exists():
            found[mode] = inside(
                [Segment.model_validate(s) for s in json.loads(p.read_text())["segments"]]
            )
    cache = Transcript.model_validate_json(rec.transcript.read_text(encoding="utf-8"))
    found["cache"] = inside(cache.segments)
    for journal in sorted(CHUNKS_DIR.glob(f"{rec.name} t60 p600*.jsonl")):
        chunks = [json.loads(ln) for ln in journal.read_text().splitlines()]
        segs = [Segment.model_validate(s) for c in chunks for s in c["segments"]]
        if segs and segs[-1].start >= min(hi, rec.duration) - 120:  # журнал покрывает отрезок
            found["chunks"] = inside(segs)
            break
    return found


def cmd_compare(rec: Recording, args: argparse.Namespace, settings: Settings) -> int:
    a, b = args.start * 60, (args.end * 60 if args.end is not None else rec.duration)
    speech = sum(s.duration for s in clip(detect_speech(rec.wav, settings.vad), a, b))
    found = variants(rec, args.start, args.end)
    bins = range(int(a // 300), int(np.ceil(b / 300)))
    print(f"запись: {rec.name}, {a / 60:g}–{b / 60:.1f} мин, речь по VAD {speech:.0f} с")
    print(
        f"{'вариант':8} {'сегм':>5} {'покрыто':>8} {'слов':>6} {'1–2 сл':>7} "
        f"{'детект':>7} {'logprob':>8}   слова по 5 мин"
    )
    for name, segs in found.items():
        cover = sum(s.duration for s in segs)
        words = [len(norm_words(s.text)) for s in segs]
        short = sum(1 for n in words if n <= 2) / max(len(segs), 1)
        flagged = repetition.flag_suspects(segs)
        suspect = sum(s.duration for s in flagged if s.suspect) / max(cover, 1)
        logprob = sum((s.avg_logprob or 0) * s.duration for s in segs) / max(cover, 1)
        per_bin = [
            sum(n for s, n in zip(segs, words, strict=True) if s.start // 300 == i) for i in bins
        ]
        print(
            f"{name:8} {len(segs):5} {cover / speech * 100:7.1f}% {sum(words):6} "
            f"{short * 100:6.1f}% {suspect * 100:6.1f}% {logprob:8.3f}   {per_bin}"
        )
    names = list(found)
    texts = {n: norm_words(" ".join(s.text for s in found[n])) for n in names}
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            print(f"WER {x} → {y}: {wer(texts[x], texts[y]) * 100:.1f} %")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "compare"])
    parser.add_argument("recording", help="подстрока пути записи в истории")
    parser.add_argument("--mode", choices=list(MODES), default="nocond")
    parser.add_argument("--from", dest="start", type=float, default=0, help="минута начала")
    parser.add_argument("--to", dest="end", type=float, default=None, help="минута конца")
    args = parser.parse_args()
    settings = Settings()
    rec = find_recording(args.recording, settings.paths.data_dir)
    return (cmd_run if args.command == "run" else cmd_compare)(rec, args, settings)


if __name__ == "__main__":
    sys.exit(main())
