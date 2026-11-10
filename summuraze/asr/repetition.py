"""Детектор галлюцинаций Whisper — вторая линия после VAD (#10).

Whisper на тишине, музыке и неразборчивой речи пишет текст, которого не было: шаблонные фразы
из субтитров YouTube («Продолжение следует...», «Спасибо за просмотр»), циклы одной фразы,
повтор слова до конца окна. В лекции это выглядит как обычный текст и уезжает в саммари фактом.

Сегменты **помечаются, а не удаляются**: «здесь плохо слышно» и «модель это придумала» — тоже
информация, она должна дойти до промпта саммари и до заметки.

Детектор работает поверх транскрипта из кэша, а не внутри стадии ASR: правка правил не должна
стоить часа повторной транскрибации. Его `VERSION` входит в ключ кэша саммари — флаги меняют
промпт.

Пороги — константы, а не конфиг: это не вкус пользователя, а свойства Whisper. Меняешь порог —
поднимай `VERSION`.

Refs #12
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Iterable

from summuraze.types import Segment

VERSION = 1

# Фразы, которые Whisper выучил из титров видео и вставляет на тишину и музыку. Сравниваются
# после нормализации (нижний регистр, без пунктуации). Сегмент помечается, только если фраза —
# почти весь его текст: «спасибо за внимание» в конце настоящей лекции бывает и правдой.
STOCK_PHRASES = (
    # ru
    "продолжение следует",
    "спасибо за просмотр",
    "спасибо за внимание",
    "подписывайтесь на канал",
    "ставьте лайки",
    "до новых встреч",
    # en
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "like and subscribe",
)
# Короткие фразы, которые Whisper ставит на тишину и шум чаще всего («Thank you.» — на любом
# языке при автоопределении). В живой речи они тоже бывают, поэтому — только когда сегмент
# целиком из них состоит. На созвонах (этап 5) это будет шумно — пересмотреть там.
EXACT_PHRASES = ("thank you", "спасибо")
# Титры субтитровщиков: в речи лекции их не бывает, а Whisper дописывает к ним имена
# («Редактор субтитров А.Семкин Корректор А.Егорова»), так что длину сегмента не проверяем.
CREDIT_PHRASES = (
    "редактор субтитров",
    "субтитры сделал",
    "субтитры создавал",
    "субтитры подготовлены",
    "dimatorzok",
    "amara org",
)
# Сколько своих слов сегмент может иметь сверх шаблонной фразы, чтобы всё ещё считаться ею.
STOCK_SLACK_WORDS = 2

# n-грамма подряд: «да да да да» (n=1) или «и вот это и вот это и вот это» (n=3).
MAX_NGRAM = 6
MIN_REPEATS = {1: 4}  # одно слово — 4 раза подряд (бывает «да, да, да» в живой речи)
DEFAULT_MIN_REPEATS = 3
# Метрика самого Whisper (temperature fallback): текст, который gzip жмёт сильнее 2.4 раза, —
# почти наверняка цикл. Короткие строки жмутся плохо, на них не смотрим.
COMPRESSION_RATIO = 2.4
COMPRESSION_MIN_CHARS = 60

# Пороги уверенности — те же, что Whisper использует внутри для пропуска «тихих» окон.
NO_SPEECH_PROB = 0.6
AVG_LOGPROB = -1.0

# Сосед повторяет сегмент слово в слово — цикл. Короткие реплики («Да.», «Так.») повторяются
# в жизни, поэтому только от трёх слов.
LOOP_MIN_WORDS = 3
LOOP_LOOKBACK = 2

_WORD = re.compile(r"\w+")


def normalize(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("ё", "е"))


def flag_suspects(segments: Iterable[Segment]) -> list[Segment]:
    """Вернуть копии сегментов с `suspect`/`suspect_reason`. Исходные не меняются."""
    result: list[Segment] = []
    recent: list[list[str]] = []
    for seg in segments:
        words = normalize(seg.text)
        reasons = [
            r
            for r in (
                _stock_phrase(words),
                _repeated_ngram(words),
                _compressible(seg.text),
                _loop(words, recent),
                _low_confidence(seg),
            )
            if r
        ]
        recent = [*recent, words][-LOOP_LOOKBACK:]
        result.append(
            seg.model_copy(
                update={"suspect": bool(reasons), "suspect_reason": "; ".join(reasons) or None}
            )
        )
    return result


def _stock_phrase(words: list[str]) -> str | None:
    text = " ".join(words)
    for phrase in CREDIT_PHRASES:
        if phrase in text:
            return f"титры субтитров «{phrase}»"
    if text in EXACT_PHRASES:
        return f"одиночное «{text}» — Whisper часто пишет его на тишине"
    for phrase in STOCK_PHRASES:
        if phrase in text and len(words) <= len(phrase.split()) + STOCK_SLACK_WORDS:
            return f"шаблонная фраза Whisper «{phrase}»"
    return None


def _repeated_ngram(words: list[str]) -> str | None:
    for n in range(1, MAX_NGRAM + 1):
        need = MIN_REPEATS.get(n, DEFAULT_MIN_REPEATS)
        for start in range(len(words) - n * need + 1):
            gram = words[start : start + n]
            repeats = 1
            pos = start + n
            while words[pos : pos + n] == gram:
                repeats += 1
                pos += n
            if repeats >= need:
                return f"повтор «{' '.join(gram)}» ×{repeats}"
    return None


def _compressible(text: str) -> str | None:
    data = text.strip().encode("utf-8")
    if len(data) < COMPRESSION_MIN_CHARS:
        return None
    ratio = len(data) / len(zlib.compress(data))
    return f"текст зациклен (сжатие ×{ratio:.1f})" if ratio > COMPRESSION_RATIO else None


def _loop(words: list[str], recent: list[list[str]]) -> str | None:
    if len(words) >= LOOP_MIN_WORDS and words in recent:
        return "повтор предыдущего сегмента"
    return None


def _low_confidence(seg: Segment) -> str | None:
    reasons = []
    if seg.no_speech_prob is not None and seg.no_speech_prob > NO_SPEECH_PROB:
        reasons.append(f"похоже на тишину (no_speech {seg.no_speech_prob:.2f})")
    if seg.avg_logprob is not None and seg.avg_logprob < AVG_LOGPROB:
        reasons.append(f"низкая уверенность (logprob {seg.avg_logprob:.2f})")
    return "; ".join(reasons) or None
