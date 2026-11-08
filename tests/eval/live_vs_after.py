"""Замер #69: распознавание кусками по VAD против прогона целиком, на записях из кэша.

Не тест, а инструмент замера (`docs/BENCHMARKS.md`); запускается руками:

    uv run python tests/eval/live_vs_after.py run  "Матан" --target 60 --threads 4
    uv run python tests/eval/live_vs_after.py compare "Матан" --target 60

`run` идёт по готовому wav ингеста так, как шла бы живая запись: шаг по времени, потоковый VAD,
кусок уходит в Whisper после паузы, с хвостом прошлого текста в `initial_prompt`. Каждый кусок
сразу пишется в JSONL — прерванный замер продолжается с места обрыва. Батарея ниже порога — выход
с кодом 3 (запустить снова, когда ноутбук на зарядке).

`compare` сравнивает с транскриптом из кэша. Эталона нет: WER здесь — расхождение двух режимов, а
не ошибка одного из них, поэтому рядом — доля помеченного детектором, термины из глоссария
конспекта и расхождения у стыков.

Refs #69
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import resource
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps

from summuraze.asr import repetition
from summuraze.asr.faster_whisper import FasterWhisperASR, _to_source
from summuraze.asr.timeline import SpeechTimeline
from summuraze.audio.wav import SAMPLE_RATE, read_pcm
from summuraze.config import Settings
from summuraze.types import Interval, Segment, Transcript

POWER_SUPPLY = Path("/sys/class/power_supply")
OUT_DIR = Path("data/bench/live-69")


@dataclass
class Recording:
    name: str
    wav: Path
    transcript: Path
    asr_key: str
    asr_seconds: float
    duration: float


def find_recording(needle: str, data: Path) -> Recording:
    """Запись из истории: последний прогон, в котором ASR реально считался, а не брался из кэша."""
    db = sqlite3.connect(data / "summuraze.sqlite")
    row = db.execute(
        """select r.path, r.duration, s.key, s.seconds, s.run_id from recordings r
           join runs u on u.recording_id = r.id join stage_runs s on s.run_id = u.id
           where r.path like ? and s.stage = 'asr' and s.cached = 0
           order by s.id desc limit 1""",
        (f"%{needle}%",),
    ).fetchone()
    if row is None:
        sys.exit(f"нет готового ASR для записи {needle!r}")
    path, duration, asr_key, seconds, run_id = row
    ingest_key = db.execute(
        """select s.key from stage_runs s join runs u on s.run_id = u.id
           where s.stage = 'ingest' and u.recording_id =
             (select recording_id from runs where id = ?) order by s.id desc limit 1""",
        (run_id,),
    ).fetchone()[0]
    art = data / "artifacts"
    return Recording(
        name=Path(path).stem,
        wav=art / "ingest" / ingest_key[:2] / ingest_key / "audio.wav",
        transcript=art / "asr" / asr_key[:2] / asr_key / "transcript.json",
        asr_key=asr_key,
        asr_seconds=seconds,
        duration=duration,
    )


@dataclass
class StreamingVAD:
    """Режет растущий поток на куски речи по паузам. Запускается заново от прошлого среза:
    Silero на минуте речи — доли секунды, а срез лежит в тишине, холодный старт ничего не портит."""

    pcm: np.ndarray
    opts: VadOptions
    target: float
    committed: int = 0
    pending: list[Interval] = field(default_factory=list)

    def advance(self, now: int, *, final: bool = False) -> list[Interval] | None:
        """Вернуть куски речи готового куска, когда их набралось на `target`, иначе None."""
        window = self.pcm[self.committed : now]
        if not len(window):
            return None
        found = get_speech_timestamps(window, self.opts, sampling_rate=SAMPLE_RATE)
        silence = self.opts.min_silence_duration_ms * SAMPLE_RATE // 1000
        pad = self.opts.speech_pad_ms * SAMPLE_RATE // 1000
        closed = [c for c in found if final or c["end"] - pad + silence <= len(window)]
        if not closed:
            return None
        speech = sum(c["end"] - c["start"] for c in closed) / SAMPLE_RATE
        if not final and speech < self.target:
            return None
        base = self.committed
        self.committed = base + closed[-1]["end"]
        return [
            Interval(start=(base + c["start"]) / SAMPLE_RATE, end=(base + c["end"]) / SAMPLE_RATE)
            for c in closed
        ]


def prompt_tail(segments: list[Segment], chars: int) -> str | None:
    """Хвост уже распознанного текста. Сомнительное не берём: цикл Whisper из прошлого куска не
    должен стать подсказкой для следующего."""
    if chars <= 0:
        return None
    clean = [s.text.strip() for s in repetition.flag_suspects(segments[-40:]) if not s.suspect]
    return " ".join(clean)[-chars:].strip() or None


def batteries() -> list[Path]:
    # Имя зависит от машины (здесь BAT1), поэтому по типу, а не по имени.
    return [d for d in POWER_SUPPLY.glob("*") if (d / "type").read_text().strip() == "Battery"]


def battery_low(floor: int) -> bool:
    return any(
        (d / "status").read_text().strip() == "Discharging"
        and int((d / "capacity").read_text()) < floor
        for d in batteries()
    )


def power_state() -> dict[str, object]:
    """Питание на момент куска: от него скорость зависит сильнее, чем от числа потоков."""
    state: dict[str, object] = {}
    for d in batteries():
        state["status"] = (d / "status").read_text().strip()
        state["capacity"] = int((d / "capacity").read_text())
        if (d / "power_now").exists():
            state["watts"] = int((d / "power_now").read_text()) / 1e6
        elif (d / "current_now").exists():
            amps = int((d / "current_now").read_text()) / 1e6
            state["watts"] = amps * int((d / "voltage_now").read_text()) / 1e6
    # platform_profile есть не везде (здесь нет), а EPP и частота есть и прямо показывают режим.
    cpu = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    if (cpu / "energy_performance_preference").exists():
        state["epp"] = (cpu / "energy_performance_preference").read_text().strip()
    if (cpu / "scaling_cur_freq").exists():
        state["mhz"] = int((cpu / "scaling_cur_freq").read_text()) // 1000
    return state


def out_path(rec: Recording, args: argparse.Namespace) -> Path:
    tag = f" {args.tag}" if args.tag else ""
    return OUT_DIR / f"{rec.name} t{args.target:g} p{args.prompt_chars}{tag}.jsonl"


def cmd_run(rec: Recording, args: argparse.Namespace, settings: Settings) -> int:
    out = out_path(rec, args)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = [json.loads(ln) for ln in out.read_text().splitlines()] if out.exists() else []
    segments = [Segment.model_validate(s) for c in done for s in c["segments"]]

    pcm = read_pcm(rec.wav)
    vad_cfg = settings.vad
    vad = StreamingVAD(
        pcm,
        VadOptions(
            threshold=vad_cfg.threshold,
            min_speech_duration_ms=vad_cfg.min_speech_ms,
            min_silence_duration_ms=vad_cfg.min_silence_ms,
            speech_pad_ms=vad_cfg.speech_pad_ms,
            # Лектор без пауз: кусок всё равно должен закончиться.
            max_speech_duration_s=args.target * 2,
        ),
        args.target,
    )
    asr = FasterWhisperASR(settings.asr.model_copy(update={"threads": args.threads}))
    step = int(args.step * SAMPLE_RATE)
    index = 0
    now = 0
    started = time.monotonic()
    with out.open("a", encoding="utf-8") as journal:
        while True:
            now = min(now + step, len(pcm))
            final = now == len(pcm)
            chunk = vad.advance(now, final=final)
            if chunk:
                if index >= len(done):
                    if battery_low(args.battery_floor):
                        print(
                            f"батарея < {args.battery_floor} % — пауза; продолжить тем же запуском"
                        )
                        return 3
                    timeline = SpeechTimeline.from_intervals(chunk)
                    packed = np.concatenate(
                        [
                            pcm[int(c.start * SAMPLE_RATE) : int(c.end * SAMPLE_RATE)]
                            for c in timeline.chunks
                        ]
                    )
                    t0 = time.monotonic()
                    raw, _ = asr._model.transcribe(
                        packed,
                        language=settings.asr.language,
                        word_timestamps=True,
                        vad_filter=False,
                        initial_prompt=prompt_tail(segments, args.prompt_chars),
                    )
                    fresh = [s for seg in raw for s in _to_source(seg, timeline)]
                    seconds = time.monotonic() - t0
                    power = power_state()
                    segments += fresh
                    entry = {
                        "index": index,
                        "start": chunk[0].start,
                        "end": chunk[-1].end,
                        "speech": timeline.packed_duration,
                        "seconds": seconds,
                        "threads": args.threads,
                        "at_stream": now / SAMPLE_RATE,
                        "power": power,
                        "segments": [s.model_dump() for s in fresh],
                    }
                    journal.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    journal.flush()
                    print(
                        f"кусок {index}: {chunk[0].start:7.0f}–{chunk[-1].end:7.0f} с, речь "
                        f"{timeline.packed_duration:5.1f} с, {seconds:5.1f} с, "
                        f"RTF(к потоку) "
                        f"{(time.monotonic() - started) / max(now, 1) * SAMPLE_RATE:.2f}"
                        f" {power}",
                        flush=True,
                    )
                index += 1
            if final or (args.limit and index >= args.limit):
                break
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"готово: {out}; пик RSS {rss:.2f} ГБ")
    return 0


_WORD = re.compile(r"[\w-]+", re.UNICODE)


def norm_words(text: str) -> list[str]:
    return [w.lower().replace("ё", "е") for w in _WORD.findall(text)]


def wer(ref: list[str], hyp: list[str]) -> float:
    """Левенштейн по словам. Строка ДП — numpy, иначе 12k × 12k в чистом Python не дождаться."""
    vocab = {w: i for i, w in enumerate(set(ref) | set(hyp))}
    h = np.array([vocab[w] for w in hyp], dtype=np.int32)
    idx = np.arange(len(h) + 1, dtype=np.int32)
    prev = idx.copy()
    for i, word in enumerate(ref, 1):
        cur = np.empty_like(prev)
        cur[0] = i
        cur[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (h != vocab[word]))
        # Вставки зависят от левого соседа: d[j] = min(d[k] + j - k) — это накопленный минимум.
        cur = np.minimum.accumulate(cur - idx) + idx
        prev = cur
    return float(prev[-1]) / max(len(ref), 1)


def glossary_terms(rec: Recording, data: Path) -> list[str]:
    for manifest in (data / "artifacts" / "summarize").glob("*/*/manifest.json"):
        if rec.asr_key in json.loads(manifest.read_text())["inputs"]:
            doc = json.loads((manifest.parent / "summary.json").read_text())["document"]
            return [t["term"] for t in doc.get("glossary", [])]
    return []


def cmd_compare(rec: Recording, args: argparse.Namespace, data: Path) -> int:
    base = Transcript.model_validate_json(rec.transcript.read_text(encoding="utf-8"))
    chunks = [json.loads(ln) for ln in out_path(rec, args).read_text().splitlines()]
    live = [Segment.model_validate(s) for c in chunks for s in c["segments"]]
    covered = chunks[-1]["end"]
    # Замер мог быть прерван: сравниваем только покрытую часть записи.
    whole = [s for s in base.segments if s.start < covered]
    ref, hyp = (
        norm_words(" ".join(s.text for s in whole)),
        norm_words(" ".join(s.text for s in live)),
    )

    def suspects(segments: list[Segment]) -> float:
        flagged = repetition.flag_suspects(segments)
        return sum(s.duration for s in flagged if s.suspect) / max(
            sum(s.duration for s in flagged), 1
        )

    def logprob(segments: list[Segment]) -> float:
        total = sum(s.duration for s in segments)
        return sum((s.avg_logprob or 0) * s.duration for s in segments) / max(total, 1)

    speech = sum(c["speech"] for c in chunks)
    spent = sum(c["seconds"] for c in chunks)
    print(f"запись: {rec.name}, покрыто {covered / 60:.1f} мин из {rec.duration / 60:.1f}")
    print(
        f"куски: {len(chunks)}, речь {speech / 60:.1f} мин, в среднем {speech / len(chunks):.0f} с"
    )
    print(f"потоков: {sorted({c['threads'] for c in chunks})}, ASR {spent / 60:.1f} мин")
    print(f"RTF к записи: {spent / covered:.3f}; к речи: {spent / speech:.3f}")
    print(f"отставание: макс {max(c['seconds'] for c in chunks):.0f} с на кусок")
    print(f"целиком: {rec.asr_seconds / rec.duration:.3f} RTF (8 потоков, весь файл)")
    print(f"слов: целиком {len(ref)}, кусками {len(hyp)}")
    print(f"WER между режимами: {wer(ref, hyp) * 100:.2f} %")
    print(
        f"помечено детектором: целиком {suspects(whole) * 100:.1f} %, "
        f"кусками {suspects(live) * 100:.1f} %"
    )
    print(f"avg_logprob: целиком {logprob(whole):.3f}, кусками {logprob(live):.3f}")

    # Где расходятся: у стыка (±5 с) или в глубине куска.
    # Стык — конец каждого куска, кроме последнего.
    seams = [c["end"] for c in chunks[:-1]]
    times = [
        (w, s.start + (s.end - s.start) * i / max(len(ws), 1))
        for s in live
        for ws in [norm_words(s.text)]
        for i, w in enumerate(ws)
    ]
    matcher = difflib.SequenceMatcher(None, [w for w, _ in times], ref, autojunk=False)
    near, far, near_words = 0, 0, 0
    examples: list[str] = []
    near_seam = [any(abs(t - x) <= 5 for x in seams) for _, t in times]
    near_words = sum(near_seam)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        count = max(i2 - i1, j2 - j1)
        at = min(i1, len(times) - 1)
        if near_seam[at]:
            near += count
            if len(examples) < args.examples:
                examples.append(
                    f"  {times[at][1]:7.0f} с: кусками «{' '.join(w for w, _ in times[i1:i2])}»"
                    f" / целиком «{' '.join(ref[j1:j2])}»"
                )
        else:
            far += count
    print(
        f"расхождений у стыков (±5 с): {near} на {near_words} слов "
        f"({near / max(near_words, 1) * 100:.1f} %)"
    )
    print(
        f"в глубине куска: {far} на {len(times) - near_words} слов "
        f"({far / max(len(times) - near_words, 1) * 100:.1f} %)"
    )
    print("примеры у стыков:\n" + "\n".join(examples))

    terms = glossary_terms(rec, data)
    if terms:
        joined_ref, joined_hyp = " ".join(ref), " ".join(hyp)
        rows = []
        for term in terms:
            key = " ".join(norm_words(term))[:-2] or term.lower()  # без окончания: падежи
            rows.append((term, joined_ref.count(key), joined_hyp.count(key)))
        lost = [r for r in rows if r[1] and not r[2]]
        print(
            f"термины глоссария ({len(rows)}): целиком {sum(r[1] for r in rows)} вхождений, "
            f"кусками {sum(r[2] for r in rows)}; потеряны совсем: {lost}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "compare"])
    parser.add_argument("recording", help="подстрока пути записи в истории")
    parser.add_argument("--target", type=float, default=60, help="речи в куске, с")
    parser.add_argument("--step", type=float, default=10, help="шаг потока, с")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prompt-chars", type=int, default=600, help="0 — без подсказки")
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="остановиться после N кусков")
    parser.add_argument("--battery-floor", type=int, default=20, help="пауза ниже этого, %%")
    parser.add_argument("--tag", default="", help="своё имя журнала для опытов")
    args = parser.parse_args()
    settings = Settings()
    data = settings.paths.data_dir
    rec = find_recording(args.recording, data)
    return cmd_run(rec, args, settings) if args.command == "run" else cmd_compare(rec, args, data)


if __name__ == "__main__":
    sys.exit(main())
