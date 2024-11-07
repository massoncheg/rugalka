import whisper
import re

# Загрузите модель Whisper (например, "base" или "small" для русской речи)
model = whisper.load_model("base")

# Транскрипция аудио
result = model.transcribe("your_file.mp3")

# Определите список матерных слов
mat_words = ["слово1", "слово2", "слово3"]  # Добавьте сюда матерные слова

# Ищем матерные слова с таймкодом
for segment in result['segments']:
    text = segment['text']
    for word in mat_words:
        if re.search(rf"\b{word}\b", text, re.IGNORECASE):
            print(f"{segment['start']} - {word}")
