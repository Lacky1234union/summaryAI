# CLAUDE.md — Руководство для Claude Code

Аналог для Codex — [`AGENTS.md`](AGENTS.md). Общие правила поддерживаем согласованными.

## Что мы строим

**Summuraze** — локальный писатель саммари к созвонам, лекциям, встречам и семинарам.
Аудио/видео → транскрипт с таймкодами (STT локально на CPU) → структурированный документ
(LLM **переключаемо**: DeepSeek по API или локальная модель) → Markdown со ссылками на таймкоды.

Порядок: сначала **лекции с одним говорящим**, многоголосые записи (созвоны, встречи, семинары)
— этап 5 вместе с атрибуцией спикеров.

Язык общения — русский. Железо известно и жёсткое: ноутбук без GPU, 14 ГБ RAM —
см. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), раздел «Железо».

---

## ⚡ Первое действие в каждой сессии

Из корня проекта (после `uv sync`). `.env` читаем как dotenv, а не shell-код.
В окружение экспортируем **только** Gitea-переменные: сначала окружение процесса, затем
`.env` проекта, недостающие URL и токен — из `../vobot/.env`. Соседний файл целиком не
подключаем: его `GITEA_REPO` относится к другому проекту.

```bash
load_summuraze_gitea() {
  local exports
  exports="$(uv run --no-sync python - <<'PYTHON'
import os
import shlex
from dotenv import dotenv_values

project = dotenv_values(".env", interpolate=False)
shared = dotenv_values("../vobot/.env", interpolate=False)
values = {}
for key in ("GITEA_URL", "GITEA_TOKEN"):
    value = os.environ.get(key) or project.get(key) or shared.get(key)
    if not value:
        raise SystemExit(f"Нет {key}: задай в окружении или .env проекта")
    values[key] = value
values["GITEA_REPO"] = "HSE/SummaryAI"
for key, value in values.items():
    print(f"export {key}={shlex.quote(value)}")
PYTHON
  )" || return
  eval "$exports"
}
load_summuraze_gitea
```

`GITEA_TOKEN` — секрет: не печатать в чат, логи или git; не включать `set -x`.
Команда захватывает вывод Python внутри оболочки, а не выводит токен в терминал.
`GITEA_REPO` всегда задаётся явно: **HSE/SummaryAI**, независимо от окружения и соседнего `.env`.
Проверка доступа без вывода секрета:

```bash
uv run --no-sync python - <<'PYTHON'
import json
import os
import urllib.request

url = f"{os.environ['GITEA_URL'].rstrip('/')}/api/v1/repos/{os.environ['GITEA_REPO']}"
request = urllib.request.Request(
    url, headers={"Authorization": "token " + os.environ["GITEA_TOKEN"]}
)
with urllib.request.urlopen(request, timeout=30) as response:
    repo = json.load(response)
assert repo["full_name"] == "HSE/SummaryAI", "Ответ от другого репозитория"
print("Gitea: доступ к HSE/SummaryAI проверен")
PYTHON
```

**Задачи — в Gitea Issues, не в файлах:** https://gitea.24studbank.ru/HSE/SummaryAI/issues
Этапы плана = вехи (milestones), одна задача = один issue. Нет issue — нет кода.
curl-шпаргалка по API (создание issue, комментарии, PR, кириллица в JSON) —
[`../vobot/docs/dev/GITEA_API.md`](../vobot/docs/dev/GITEA_API.md).

> **Кириллица в JSON:** `curl -d '{...}'` ломает кодировку. Только через `json.dumps` / `JSON.stringify`
> + `--data-binary @-` + `Content-Type: application/json; charset=utf-8`.

Лог выполнения дописывается в тело issue секцией `### YYYY-MM-DD`; архитектурные решения —
в `docs/DECISIONS.md`.

### Ветки

| Ветка | Что в ней |
|---|---|
| `main` | **только работающее.** Сюда мержим, когда этап принят и проверен на живой записи |
| `dev` | рабочая ветка: вся разработка идёт здесь, PR из задач — сюда |
| `feat/<issue#>-<slug>` | одна задача = одна ветка от `dev` |

Порядок: ветка от `dev` → работа → push → PR в `dev`. Промоушн `dev` → `main` — только осознанно,
когда этап принят; никогда не автоматом за завершением задачи.

---

## Карта документации

| Файл | О чём |
|------|-------|
| [`docs/PLAN.md`](docs/PLAN.md) | **Этапы и приёмка** — что делаем сейчас и что считается готовым |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Конвейер, контракты, раскладка репозитория, железо |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | ADR: что решили и почему (дописываем сверху) |
| [`docs/QUESTIONS.md`](docs/QUESTIONS.md) | Открытые вопросы по этапам + бэклог идей |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Замеры моделей на **этом** железе: RTF, WER, RSS |
| [`docs/dev/CI_RUNNER.md`](docs/dev/CI_RUNNER.md) | CI: раннер act_runner, защита веток, разбор падений |

---

## Процесс работы

- Один этап = одна фокус-сессия; следующий не начинаем, пока текущий не работает у пользователя.
- Перед кодом — подход в 2–3 предложениях на архитектурных развилках.
- Код по модулям; пользователь тестирует после каждого; комментарии — только когда WHY неочевиден.
- Значимое решение → запись в `DECISIONS.md` (контекст / решение / причина / последствия).
- Решённый вопрос переезжает из `QUESTIONS.md` в `DECISIONS.md`, а не удаляется.

---

## Guardrails

Мнемоника: **провайдер за протоколом, конфиг типизирован, стадия кэширована, утверждение с
таймкодом, локально — значит локально, гейт зелёный.**

Полная таблица с обоснованиями — [`docs/PLAN.md`](docs/PLAN.md), раздел «Guardrails».
Коротко, что нарушать нельзя:

1. Вендорский импорт (`faster_whisper`, `anthropic`, `pyannote`) — только внутри своего адаптера,
   и лениво. Конвейер знает только Protocol.
2. Переключение локально ↔ API — это конфиг (`base_url` + `model`), а не ветка в конвейере.
   DeepSeek, OpenRouter, Yandex и llama.cpp говорят по одному протоколу — адаптер один.
3. Конфиг — `pydantic-settings`. `os.getenv` по коду вразброс не заводим.
4. Стадия пишет артефакт на диск; ключ кэша включает версию стадии. Меняешь логику — меняй версию.
5. Любое утверждение в саммари привязано к таймкоду. Непроверяемое утверждение — баг.
6. `--local-only` режет сетевых провайдеров в фабрике, а не предупреждением в доках.
7. Модель (STT или LLM) выбирается **замером на этом железе**; результат — в `BENCHMARKS.md`
   с датой и версиями. Чужие бенчмарки на 4090 здесь не значат ничего.
8. ID моделей провайдера сверять в момент реализации: `deepseek-chat` retired 24.07.2026.
9. `make lint` + `make test` — зелёные перед коммитом; `make ci` = то, что гоняет CI (с покрытием ≥ 85 %).

---

## Что легко сломать

- **Таймкоды после VAD.** ASR получает только речевые интервалы, но таймкоды должны считаться от
  начала исходного файла. Ошибка здесь ломает всю навигацию по саммари и заметна не сразу.
- **Память.** ASR-модель и локальная LLM вместе в 14 ГБ не влезают. Конвейер последовательный;
  модель стадии выгружается до загрузки следующей.
- **Кэш префикса.** Любой байт в префиксе рушит кэш ниже — а у DeepSeek кэш-хит дешевле промаха
  в 50 раз. Транскрипт идёт первым и байт в байт; timestamp'ы, ID прогона и профиль — после него.
  Проверка, что работает — поле usage в ответе, а не вера.
- **Галлюцинации Whisper на тишине.** Без VAD в лекции появляется «Спасибо за просмотр».
  Сомнительные сегменты помечаем, не удаляем — пометка должна дойти до промпта саммари.
