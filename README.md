# YouTube Video Downloader and Transcriber

Проект для скачивания YouTube видео и их транскрипции с помощью Whisper.

## 🚀 Быстрая установка

### Вариант 1: Автоматическая установка (рекомендуется)
```bash
# Запустить автоматическую установку
python install_all.py
```

### Вариант 2: Ручная установка
```bash
# Установить Python пакеты
pip install -r requirements.txt

# Создать папки проекта
mkdir downloads transcripts
```

### Вариант 3: Windows (batch файл)
```cmd
install.bat
```

## 📋 Требования

- Python 3.8+
- FFmpeg (устанавливается автоматически через `install_all.py`)
- Интернет соединение

## 🎬 Установка FFmpeg

FFmpeg устанавливается автоматически через `install_all.py`. Если нужна ручная установка:

1. Скачайте FFmpeg с [официального сайта](https://ffmpeg.org/download.html)
2. Распакуйте в папку (например, `C:\ffmpeg`)
3. Добавьте `C:\ffmpeg\bin` в переменную PATH

## 📦 Установленные пакеты

- `whisper-openai` - транскрипция аудио/видео
- `yt-dlp` - скачивание YouTube видео
- `mcp` - Model Context Protocol
- `fastmcp` - быстрый MCP сервер
- `smolagents` - агенты для обработки
- `pyyaml` - работа с YAML файлами
- `python-dotenv` - переменные окружения

## 🚀 Использование

### Основной скрипт
```bash
python main.py
```

### Тестирование
```bash
# Тест транскрипции
python 123.py

# Тест FFmpeg
python test_ffmpeg_simple.py
```

## 📁 Структура проекта

```
agents_shorts/
├── downloads/          # Скачанные видео
├── transcripts/        # Результаты транскрипции
├── mcp_tools.py        # Основные инструменты
├── main.py            # Главный скрипт
├── install_all.py     # Автоматическая установка
├── install.bat        # Установка для Windows
├── requirements.txt   # Python зависимости
└── README.md         # Этот файл
```

## 🔧 Функции

### `get_video(url)`
Скачивает YouTube видео и возвращает путь к файлу.

### `transcribe(path)`
Транскрибирует аудио/видео файл и сохраняет результат в JSON.

## 🐛 Устранение проблем

### Ошибка "FFmpeg not found"
1. Убедитесь, что FFmpeg установлен
2. Проверьте переменную PATH
3. Запустите `python install_all.py`

### Ошибка "Bad file descriptor"
1. Проверьте права доступа к файлам
2. Убедитесь, что файл не заблокирован другим процессом
3. Попробуйте перезапустить скрипт

### Ошибка импорта пакетов
1. Установите зависимости: `pip install -r requirements.txt`
2. Проверьте версию Python (должна быть 3.8+)
3. Активируйте виртуальное окружение, если используете

## 📝 Примеры

```python
from mcp_tools import get_video, transcribe

# Скачать видео
video_path = get_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

# Транскрибировать
transcript_path = transcribe(video_path)
```

## 🤝 Поддержка

Если возникли проблемы:
1. Проверьте логи ошибок
2. Убедитесь, что все зависимости установлены
3. Проверьте права доступа к папкам
4. Убедитесь, что FFmpeg работает: `ffmpeg -version`

