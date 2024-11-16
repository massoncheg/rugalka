import locale
from typing import List, Tuple
import numpy as np
import logging
from silero_vad import read_audio  # Убрали импорт get_speech_timestamps
import torch
import torchaudio
from nemo.collections.asr.models import EncDecCTCModel
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
import gc
import hashlib
import string
import os
import json
import sys
import tkinter as tk
from tkinter import filedialog

logger = logging.getLogger(__name__)
BATCH_SIZE = 4

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")

path_to_conf_weights = os.path.join(CONFIG_DIR, "ctc_model_weights.ckpt")
path_to_conf_config = os.path.join(CONFIG_DIR, "ctc_model_config.yaml")


def read_large_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка чтения JSON: {e}")


def file_exist(in_file_path):
    return os.path.exists(in_file_path)


def segment_audio(
    wav: torch.Tensor,
    sample_rate: int,
    vad_model,
    get_speech_timestamps,
    max_duration: float = 26.0,
    min_duration: float = 10.0,
    new_chunk_threshold: float = 0.1,
) -> Tuple[List[np.ndarray], List[List[float]]]:
    logger.info("Начало сегментации аудио...")
    try:
        speech_timestamps = get_speech_timestamps(
            wav, vad_model, sampling_rate=sample_rate
        )
    except Exception as e:
        logger.error(f"Ошибка при получении меток активности речи: {e}")
        raise

    segments = []
    boundaries = []
    curr_duration = 0.0
    curr_start = None
    curr_end = None

    for segment in speech_timestamps:
        start = segment["start"] / sample_rate  # Время в секундах
        end = segment["end"] / sample_rate

        if curr_start is None:
            curr_start = start
            curr_end = end
            curr_duration = curr_end - curr_start
            continue

        if (
            start - curr_end > new_chunk_threshold and curr_duration >= min_duration
        ) or (curr_duration + (end - curr_end) > max_duration):
            start_sample = int(curr_start * sample_rate)
            end_sample = int(curr_end * sample_rate)
            audio_chunk = wav[start_sample:end_sample]
            audio_chunk_np = audio_chunk.numpy()
            segments.append(audio_chunk_np)
            boundaries.append([curr_start, curr_end])

            curr_start = start
            curr_end = end
            curr_duration = curr_end - curr_start
        else:
            curr_end = end
            curr_duration = curr_end - curr_start

    if curr_start is not None:
        start_sample = int(curr_start * sample_rate)
        end_sample = int(curr_end * sample_rate)
        audio_chunk = wav[start_sample:end_sample]
        audio_chunk_np = audio_chunk.numpy()
        segments.append(audio_chunk_np)
        boundaries.append([curr_start, curr_end])

    logger.info("Сегментация аудио завершена.")
    return segments, boundaries


def init_process():
    global local_model
    from nemo.collections.asr.models import EncDecCTCModel
    import torch

    cuda_av = torch.cuda.is_available()
    device = "cuda" if cuda_av else "cpu"

    try:
        # Загрузка конфигурации и весов модели
        local_model = EncDecCTCModel.from_config_file(path_to_conf_config)
        ckpt = torch.load(path_to_conf_weights, map_location=device)
        local_model.load_state_dict(ckpt, strict=False)
        local_model.eval()
        local_model = local_model.to(device)
        logger.info("Модель успешно инициализирована.")
    except FileNotFoundError as e:
        logger.error(
            f"Файл не найден: {e}. Проверьте наличие конфигурации и весов модели."
        )
        raise
    except Exception as e:
        logger.error(f"Ошибка при инициализации модели: {e}")
        raise


def transcribe_batch(batch):
    return local_model.transcribe(batch, batch_size=BATCH_SIZE)


def generate_code_from_file(file_path):
    # Генерация MD5 хеша для файла
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    file_hash = hasher.hexdigest()

    # Преобразование хеша в 4-буквенный код
    letters = string.ascii_uppercase  # Алфавит для кода
    short_code = "".join(
        letters[int(file_hash[i : i + 2], 16) % len(letters)] for i in range(0, 8, 2)
    )
    return short_code
