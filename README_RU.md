# Local Whisper Docker Service

HTTP-сервис транскрипции аудио на `faster-whisper` для Docker-сети `ai`.

Сервис принимает аудиофайл от других контейнеров и возвращает текст. Desktop-диктовки, хоткеев, микрофона и вставки текста здесь нет.

По умолчанию:

- модель: `large-v3-turbo`
- устройство: `cpu`
- вычисления: `int8`
- язык: `auto`
- адрес внутри сети `ai`: `http://local-whisper:8000`

## Запуск

Создайте внешнюю сеть, если ее еще нет:

```bash
docker network create ai
```

Соберите и запустите сервис:

```bash
docker compose up -d --build
```

Первый запуск скачает модель в Docker volume `whisper-models`.

## API

Проверка:

```bash
curl http://local-whisper:8000/health
```

Транскрипция JSON-ответом:

```bash
curl -s \
  -F "file=@audio.wav" \
  -F "language=auto" \
  http://local-whisper:8000/transcribe
```

Ответ:

```json
{
  "text": "распознанный текст",
  "language": "ru",
  "duration": 12.34
}
```

Только текст:

```bash
curl -s \
  -F "file=@audio.wav" \
  http://local-whisper:8000/transcribe.txt
```

## Вызов из другого контейнера

Другой контейнер должен быть подключен к сети `ai`.

Пример в `docker-compose.yml` другого проекта:

```yaml
services:
  app:
    image: your-image
    networks:
      - ai

networks:
  ai:
    external: true
```

После этого сервис доступен по DNS-имени:

```text
http://local-whisper:8000/transcribe
```

## Настройки

Переменные окружения в `docker-compose.yml`:

- `WHISPER_MODEL=large-v3-turbo`
- `WHISPER_DEVICE=cpu`
- `WHISPER_COMPUTE_TYPE=int8`
- `WHISPER_LANGUAGE=auto`
- `WHISPER_MODEL_DIR=/models`
- `WHISPER_BEAM_SIZE=5`

## Поддерживаемые форматы

`faster-whisper` через PyAV обычно принимает `wav`, `mp3`, `m4a`, `ogg`, `flac`, `webm` и другие распространенные аудиоформаты.
