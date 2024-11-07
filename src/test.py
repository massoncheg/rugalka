import os
import wget
import torch
import torchaudio
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

# Путь к аудиофайлу
audio_path = 'путь_к_вашему_аудиофайлу.mp3'

# Скачивание модели GigaAM-RNNT и конфигурационного файла
model_url = 'https://n-ws-q0bez.s3pd12.sbercloud.ru/b-ws-q0bez-jpv/GigaAM/rnnt_model_weights.ckpt'
config_url = 'https://n-ws-q0bez.s3pd12.sbercloud.ru/b-ws-q0bez-jpv/GigaAM/rnnt_model_config.yaml'
model_path = 'rnnt_model_weights.ckpt'
config_path = 'rnnt_model_config.yaml'

if not os.path.exists(model_path):
    wget.download(model_url, model_path)
if not os.path.exists(config_path):
    wget.download(config_url, config_path)

# Загрузка конфигурации модели
config = OmegaConf.load(config_path)

# Инициализация модели
asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(restore_path=model_path, override_config_path=config_path)

# Загрузка и преобразование аудиофайла
audio, sr = torchaudio.load(audio_path)
if sr != asr_model.cfg.sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=asr_model.cfg.sample_rate)
    audio = resampler(audio)

# Распознавание речи
transcription = asr_model.transcribe([audio])

# Словарь нецензурных слов
swear_words = ['список', 'нецензурных', 'слов']

# Поиск нецензурных слов с таймкодами
detected_swears = []
for segment in transcription:
    for word in swear_words:
        if word in segment.lower():
            # Получение таймкода (примерный, так как модель не возвращает точные таймкоды)
            timestamp = audio.size(1) / asr_model.cfg.sample_rate
            detected_swears.append({
                "timestamp": timestamp,
                "text": segment
            })

# Вывод результатов
for entry in detected_swears:
    print(f"Время: {entry['timestamp']} секунда, Текст: {entry['text']}")
