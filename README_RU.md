# Local Whisper для Linux

Локальная диктовка на `faster-whisper`: удерживаете `F13`, говорите на русском или английском, отпускаете клавишу, текст вставляется в активное поле через буфер обмена и `Ctrl+V`.

По умолчанию используется:

- `large-v3-turbo`
- `language auto`
- `cuda`
- `float16`

## Системные пакеты

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip portaudio19-dev xclip
```

Fedora:

```bash
sudo dnf install -y python3 python3-pip python3-devel portaudio-devel xclip
```

Для Wayland часто нужен `wl-clipboard`, но глобальные хоткеи через `pynput` надежнее работают в X11-сессии. Если кнопка не ловится в Wayland, переключитесь на Xorg/X11 или назначьте в системе запуск внешней команды.

## Установка

```bash
chmod +x install.sh run_dictation.sh run_dictation_cpu.sh
./install.sh
```

Первый запуск скачает модель в папку `models`. Установщик также ставит CUDA runtime wheels из `requirements-cuda.txt`, чтобы `faster-whisper` мог найти `cuBLAS`/`cuDNN` без ручной установки CUDA Toolkit.

## Запуск на GPU

```bash
./run_dictation.sh
```

## Запуск на CPU

```bash
./run_dictation_cpu.sh
```

CPU-режим использует `int8`, чтобы снизить нагрузку:

```bash
python dictate_hold.py --device cpu --compute-type int8 --language auto
```

## Автозапуск через systemd user

Сервис в репозитории предполагает, что проект лежит в `~/Local Whisper`.

```bash
mkdir -p ~/.config/systemd/user
cp local-whisper.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now local-whisper.service
```

Проверка логов:

```bash
journalctl --user -u local-whisper.service -f
```

Отключение:

```bash
systemctl --user disable --now local-whisper.service
```

## Хоткей

Назначьте кнопку мыши или клавиатуры на `F13`. На Linux это можно сделать средствами вашей DE/WM, `input-remapper`, `keyd`, `xbindkeys` или настройками производителя, если они доступны.

Логи приложения пишутся в `logs/dictate_hold.log`
