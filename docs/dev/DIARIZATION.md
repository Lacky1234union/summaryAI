# Подготовка замера diarization (#25)

Скрипт `tests/eval/diarization.py` сравнивает конфигурации sherpa-onnx на CPU.
Зависимости вынесены в необязательную группу `diarization-bench`; обычный `uv sync`
не устанавливает их. В продукте diarization пока не включена; решение принимается
после измерений на целевом ноутбуке. Эта инструкция сама по себе не означает приёмку #25.

## Эталон и вход

Для первого сравнения — встреча AMI ES2002a с четырьмя участниками:

- [Исходное аудио Mix-Headset](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav).
- [Ручной RTTM only_words](https://github.com/pyannote/AMI-diarization-setup/blob/main/only_words/rttms/train/ES2002a.rttm).
- [Описание получения разметки](https://github.com/pyannote/AMI-diarization-setup): слова
  из ручной аннотации AMI, включая перекрывающуюся речь; отдельные вокализации не включены.

Аудио и результаты держать в `data/bench/diar-25/`, не добавлять в git. Для короткого
предварительного прогона можно взять первые 300 секунд PCM WAV 16 kHz mono:

```bash
ffmpeg -i data/bench/diar-25/ES2002a.Mix-Headset.wav -t 300 \
  -ac 1 -ar 16000 -c:a pcm_s16le data/bench/diar-25/ES2002a-first300.wav
```

Скрипт обрезает эталон по длительности WAV; поддержан только фрагмент **от начала**
исходной записи. Для другого среза сначала нужно сдвинуть RTTM на начало среза.
DER по 5 минутам нельзя выдавать за качество на всей встрече или русском семинаре.

## Модели

Сверены 2026-11-10 по [документации sherpa-onnx](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html):

- [pyannote segmentation-3.0 в ONNX, fp32 и int8](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2).
- [TitaNet small (English)](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx).
- [3D-Speaker ERes2Net (Chinese)](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx).

Языки обучающих данных embeddings различаются: результат на английском AMI не доказывает
качество на русском. Необходима дополнительная вычитанная русская фикстура.

Другой кандидат — [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
Он поддерживает CPU, но требует принятия условий доступа Hugging Face. Без доступа замер
не проводится; чужую таблицу производительности не переносить в наши результаты.

## Запуск и метрики

```bash
make bench-diar ARGS='--audio data/bench/diar-25/ES2002a-first300.wav --reference data/bench/diar-25/ES2002a.rttm --segmentation data/bench/diar-25/sherpa-onnx-pyannote-segmentation-3-0/model.onnx --embedding data/bench/diar-25/nemo_en_titanet_small.onnx --threads 1 --output data/bench/diar-25/titanet-fp32.json'
```

Повторить с `model.int8.onnx` и другим embedding. Для каждого запуска:

- новый процесс кандидата; пиковая RSS не включает подсчёт DER или предыдущую модель;
- RTF инференса и отдельно cold RTF с загрузкой модели (импорт библиотек не включён);
- DER — `pyannote.metrics`, оптимальное сопоставление меток голосов, collar=0,
  перекрывающаяся речь включена, UEM — вся длительность WAV;
- JSON содержит версии, SHA-256 аудио/эталона/весов, параметры, интервалы и компоненты DER.

По умолчанию число голосов определяет модель. `--speakers 4` — отдельный **oracle**-замер,
его нельзя смешивать с автоматическим выбором. Не подбирать threshold по проверочному
фрагменту и затем называть этот же фрагмент независимым тестом.

Режим питания, питание от сети/батареи, число потоков и параллельную нагрузку записывать
в `docs/BENCHMARKS.md`. Во время записи семинара полноценные замеры отложены: они искажают
RTF и расход батареи. Подготовка скрипта и скачивание входов не заменяют замер.
