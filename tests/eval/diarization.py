"""CPU-замер #25; зависимости из группы diarization-bench, пользовательское аудио не нужно.

Каждый кандидат работает в новом процессе: RSS другого кандидата и вычисление DER
не попадают в его пиковую память. DER считает pyannote.metrics по ручному RTTM,
collar=0, с перекрывающейся речью. Число голосов не подсказывается по эталону.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import subprocess
import sys
import time
import wave
from datetime import UTC, datetime
from pathlib import Path

VERSION = "1"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rttm(path: Path, recording: str, duration: float) -> list[tuple[float, float, str]]:
    turns = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 10 or fields[0] != "SPEAKER":
            raise ValueError(f"{path.name}:{number}: ожидается RTTM SPEAKER из 10 полей")
        if fields[1] != recording:
            continue
        start, length = float(fields[3]), float(fields[4])
        if not math.isfinite(start + length) or start < 0 or length <= 0:
            raise ValueError(f"{path.name}:{number}: неверный интервал")
        end = min(start + length, duration)
        if start < end:
            turns.append((start, end, fields[7]))
    if not turns:
        raise ValueError(f"нет эталонной речи для {recording!r}")
    return turns


def sherpa_worker(args: argparse.Namespace) -> dict:
    """Адаптер кандидата для замера, импорт вендора только внутри него."""
    import numpy as np
    import sherpa_onnx

    started = time.perf_counter()
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(args.segmentation)
            ),
            num_threads=args.threads,
            provider="cpu",
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(args.embedding), num_threads=args.threads, provider="cpu"
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=args.speakers if args.speakers else -1, threshold=args.threshold
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise ValueError("sherpa-onnx: неверная конфигурация")
    model = sherpa_onnx.OfflineSpeakerDiarization(config)
    load_seconds = time.perf_counter() - started
    with wave.open(str(args.audio)) as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise ValueError("нужен PCM WAV: mono, 16-bit, 16000 Hz")
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    samples = pcm.astype(np.float32) / 32768.0
    started = time.perf_counter()
    turns = model.process(samples).sort_by_start_time()
    seconds = time.perf_counter() - started
    return {
        "runtime": "sherpa-onnx",
        "runtime_version": importlib.metadata.version("sherpa-onnx"),
        "load_seconds": load_seconds,
        "inference_seconds": seconds,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "turns": [{"start": t.start, "end": t.end, "speaker": str(t.speaker)} for t in turns],
    }


def score(reference: list[tuple[float, float, str]], turns: list[dict], duration: float) -> dict:
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref, hyp = Annotation(), Annotation()
    for index, (start, end, speaker) in enumerate(reference):
        ref[Segment(start, end), index] = speaker
    for index, turn in enumerate(turns):
        start, end = max(0, turn["start"]), min(duration, turn["end"])
        if start < end:
            hyp[Segment(start, end), index] = turn["speaker"]
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    details = metric(ref, hyp, uem=Timeline([Segment(0, duration)]), detailed=True)
    return {
        "scorer": "pyannote.metrics",
        "scorer_version": importlib.metadata.version("pyannote.metrics"),
        "collar_seconds": 0,
        "skip_overlap": False,
        "der": details["diarization error rate"],
        "missed_speech_seconds": details["missed detection"],
        "false_alarm_seconds": details["false alarm"],
        "confusion_seconds": details["confusion"],
        "reference_speaker_seconds": details["total"],
    }


def run(args: argparse.Namespace) -> None:
    with wave.open(str(args.audio)) as wav:
        duration = wav.getnframes() / wav.getframerate()
    reference = read_rttm(args.reference, args.recording, duration)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    for option in ("audio", "segmentation", "embedding", "threads", "threshold", "speakers"):
        command.extend(["--" + option, str(getattr(args, option))])
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"кандидат завершился с кодом {result.returncode}:\n{result.stderr}")
    measured = json.loads(result.stdout)
    measured.update(
        benchmark_version=VERSION,
        measured_at=datetime.now(UTC).isoformat(),
        context=args.context,
        python=platform.python_version(),
        platform=platform.platform(),
        recording=args.recording,
        duration=duration,
        threads=args.threads,
        threshold=args.threshold,
        supplied_speakers=args.speakers,
        reference_speakers=len({t[2] for t in reference}),
        detected_speakers=len({t["speaker"] for t in measured["turns"]}),
        inputs={
            name: {"file": getattr(args, name).name, "sha256": digest(getattr(args, name))}
            for name in ("audio", "reference", "segmentation", "embedding")
        },
        metrics=score(reference, measured["turns"], duration),
    )
    measured["rtf"] = measured["inference_seconds"] / duration
    measured["cold_rtf"] = (measured["inference_seconds"] + measured["load_seconds"]) / duration
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(measured, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)
    print("| Результат | Голосов | RTF | Cold RTF | DER | RSS, МиБ |")
    print("|---|---:|---:|---:|---:|---:|")
    print(
        f"| {args.output.name} | {measured['detected_speakers']} | "
        f"{measured['rtf']:.3f} | {measured['cold_rtf']:.3f} | "
        f"{measured['metrics']['der']:.2%} | {measured['peak_rss_mib']:.0f} |"
    )
    print(f"Контекст: {args.context}")


def positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("нужно положительное число")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--threads", type=positive, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--speakers", type=int, default=0, help="0 — определить, иначе oracle-замер"
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--recording", default="ES2002a")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--context", default="не указан; RTF предварительный")
    args = parser.parse_args()
    if not 0 < args.threshold <= 1 or args.speakers < 0:
        parser.error("threshold должен быть в (0, 1], speakers >= 0")
    if args.worker:
        print(json.dumps(sherpa_worker(args)))
    else:
        if args.reference is None or args.output is None:
            parser.error("нужны --reference и --output")
        run(args)


if __name__ == "__main__":
    main()
