"""Вернуть системе память выгруженной модели.

`del` модели освобождает её внутри процесса, но glibc держит кучу у себя: замер на
large-v3-turbo — 1.1 ГБ RSS после `del` и 0.24 ГБ после `malloc_trim`. Для 14 ГБ это не мелочь:
следом за ASR может стартовать llama.cpp отдельным процессом, а ему нужна память системы, а не
свободные страницы внутри нашей кучи. Вторая загрузка модели без trim доходила до 1.8 ГБ.

Refs #11
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc


def release_memory() -> None:
    gc.collect()
    libc = ctypes.util.find_library("c")
    if libc is None:  # не Linux/glibc — отдавать нечем, ОС справится сама
        return
    try:
        ctypes.CDLL(libc).malloc_trim(0)
    except (OSError, AttributeError):  # musl и прочие libc без malloc_trim
        return
