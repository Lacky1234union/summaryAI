# Раннер для CI (act_runner)

CI (`.gitea/workflows/ci.yml`, #35) выполняется на self-hosted раннере Gitea. Пока раннера с
меткой `ubuntu-latest` нет, задания висят в статусе «Ожидание» вечно — это не ошибка workflow.

Раннеры vobot этому репозиторию **не видны**: они зарегистрированы не на HSE. Нужен свой.

## Где запускать

**Рекомендуется — на том же VPS, где живёт раннер vobot.** Docker там уже есть, машина включена
всегда. Ноутбук годится только как временный вариант: закрыл крышку — CI встал, а 14 ГБ RAM
нужны под ASR.

Регистрировать лучше **на организацию HSE**, а не на репозиторий: тогда тот же раннер подхватит
и другие репозитории HSE без повторной регистрации.

## 1. Токен регистрации

Gitea → организация **HSE** → «Настройки» → «Actions» → «Раннеры» → «Создать новый раннер» →
скопировать токен.

*(Только для этого репозитория: HSE/SummaryAI → «Настройки» → «Actions» → «Раннеры».)*

Токен — секрет: в чат, в issue и в git не вставлять.

## 2. Запуск на VPS

```bash
sudo mkdir -p /opt/act_runner-hse && cd /opt/act_runner-hse

# Версию сверить на https://gitea.com/gitea/act_runner/releases и прибить точную, не latest.
RUNNER_IMAGE=gitea/act_runner:<версия>

docker run --rm --entrypoint act_runner "$RUNNER_IMAGE" generate-config > config.yaml
```

В `config.yaml` поправить:

```yaml
runner:
  capacity: 1          # один джоб за раз: зависимости тянут torch, параллель забьёт диск и сеть
cache:
  enabled: true        # кэш uv между прогонами: без него каждый CI заново качает ~300 МБ
```

`docker-compose.yml` рядом:

```yaml
services:
  runner:
    image: gitea/act_runner:<версия>
    restart: unless-stopped
    environment:
      CONFIG_FILE: /config.yaml
      GITEA_INSTANCE_URL: https://gitea.24studbank.ru
      GITEA_RUNNER_REGISTRATION_TOKEN: ${HSE_RUNNER_TOKEN}
      GITEA_RUNNER_NAME: hse-vps
      # Метка = образ, в котором крутятся джобы. ci.yml просит ubuntu-latest.
      GITEA_RUNNER_LABELS: ubuntu-latest:docker://gitea/runner-images:ubuntu-latest
    volumes:
      - ./config.yaml:/config.yaml
      - ./data:/data                                # здесь .runner — регистрация переживает рестарт
      - /var/run/docker.sock:/var/run/docker.sock
```

```bash
HSE_RUNNER_TOKEN='<токен из шага 1>' docker compose up -d
docker compose logs -f runner     # ждём «runner registered successfully» и «declare successfully»
```

Токен нужен только при первой регистрации; дальше раннер живёт по `./data/.runner`.

## 3. Проверка

1. Gitea → HSE → «Настройки» → «Actions» → «Раннеры»: `hse-vps` в статусе «Свободен».
2. Открыть любой PR → вкладка «Actions» → перезапустить CI. Ожидаемо: джоб `lint-test` зелёный,
   в логе шага «Тесты и покрытие» — `Required test coverage of 85.0% reached`.
3. Первый прогон дольше (качаются Python 3.14 и зависимости), следующие — из кэша uv.

## 4. Защита веток — без неё CI только советует

HSE/SummaryAI → «Настройки» → «Ветки» → «Добавить правило» — отдельно для `dev` и для `main`:

- **Включить проверку статуса** → шаблон контекста: `CI / lint-test (pull_request)`
- Запретить прямой push (только через PR) — для `main` обязательно, для `dev` по желанию.

Имя контекста строится как `<name workflow> / <name джоба> (<событие>)`. Если переименовать
джоб в `ci.yml`, слияние заблокируется навсегда «ожидающей» проверкой — правило надо править
вместе с файлом.

## Если не работает

| Симптом | Причина |
|---|---|
| Джоб вечно «Ожидание» | нет раннера с меткой `ubuntu-latest`, или он зарегистрирован не на HSE |
| Падает на `actions/checkout` / `setup-uv` | раннер не достаёт до github.com — оттуда Gitea по умолчанию берёт actions |
| Падает шаг `ffmpeg` | в образе нет apt (не Debian/Ubuntu) — сменить образ в метке |
| `uv sync --locked` падает | `uv.lock` не обновлён в PR: локально `uv lock` и закоммитить |
| Покрытие ниже 85 % | добавлен код без тестов; `make cov` локально покажет строки |
