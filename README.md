# Summuraze

Локальный писатель саммари к созвонам, лекциям, встречам и семинарам.

Запись → транскрипт с таймкодами (STT локально, на CPU, без GPU) → структурированный документ
(DeepSeek по API или локальная модель — переключается флагом) → Markdown, который можно читать
вместо записи.

```
summuraze run lecture.mp4 --profile lecture          # по API, ~$0.01 за двухчасовую запись
summuraze run lecture.mp4 --profile lecture --local  # локальная модель, ничего не уходит наружу
```

Статус: **проектирование.** Кода пока нет — есть план.

- [`docs/PLAN.md`](docs/PLAN.md) — этапы и приёмка
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — конвейер и контракты
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — почему сделано именно так
- [`docs/QUESTIONS.md`](docs/QUESTIONS.md) — что ещё не решено
