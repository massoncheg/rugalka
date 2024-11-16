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
import msgpack

from utils.utils import (
    file_exist,
    segment_audio,
    init_process,
    transcribe_batch,
    format_time,
    generate_code_from_file,
    read_large_json,
)

locale.getpreferredencoding = lambda: "UTF-8"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

root = tk.Tk()
root.withdraw()

def get_user_inputs():
    logger.info("get_user_inputs !!!!")

    PATH_TO_INPUT_FILE = filedialog.askopenfilename(
        title="Выберите исходный файл",
        filetypes=[
            ("Media Files", "*.mp3 *.mp4 *.wav *.webm *.opus"),
            ("All Files", "*.*"),
        ],
    )
    if not PATH_TO_INPUT_FILE:
        logger.error("Файл не был выбран.")
        sys.exit(1)

    SEL_OUTPUT_DIR = filedialog.askdirectory(title="Выберите папку для выгрузки")
    if not SEL_OUTPUT_DIR:
        logger.error("Папка для выгрузки не была выбрана.")
        sys.exit(1)

    return PATH_TO_INPUT_FILE, SEL_OUTPUT_DIR

def main():
    PATH_TO_INPUT_FILE, SEL_OUTPUT_DIR = get_user_inputs()
    # Сгенерировать уникальное имя
    hashName = generate_code_from_file(PATH_TO_INPUT_FILE)
    # Может быть как видео, так и аудио
    input_file_extension = os.path.splitext(PATH_TO_INPUT_FILE)[1]

    OUTPUT_DIR = os.path.join(SEL_OUTPUT_DIR, hashName)
    BATCH_SIZE = 4
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    temp_segments = False
    output_tmp_dir = os.path.join(OUTPUT_DIR, "tmp")
    force_re_segmentation = False
    search_words = {"бля", "сук", "пид"}

    os.makedirs(output_tmp_dir, exist_ok=True)

    path_to_audio_file = os.path.join(output_tmp_dir, f"{hashName}_extracted_audio.wav")

    if not file_exist(path_to_audio_file):
        if input_file_extension in [".wav"]:
            path_to_audio_file = PATH_TO_INPUT_FILE
        elif input_file_extension in [".mp4", ".webm", ".mp3", ".opus"]:
            # Извлечение аудио с помощью ffmpeg
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    PATH_TO_INPUT_FILE,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    path_to_audio_file,
                ],
                check=True,
            )
        else:
            logger.error(f"Ошибка: файлы {input_file_extension} не поддерживаются")
            sys.exit(1)

    if not file_exist(path_to_audio_file):
        logger.error(f"Ошибка: исходный файл {path_to_audio_file} не существует")
        sys.exit(1)
    # Прочитать аудиофайл
    fileW = read_audio(path_to_audio_file, sampling_rate=16000)

    segm_output_file_path = os.path.join(
        output_tmp_dir, f"{hashName}_segmentation_result.msgpack"
    )

    need_to_segment = True

    try:
        # Загрузить файл
        with open(segm_output_file_path, "rb") as f:
            data = msgpack.load(f)
            segments = [np.array(segment) for segment in data["segments"]]
            boundaries = data["boundaries"]
        need_to_segment = False

    except FileNotFoundError:
        logger.error(f"Ошибка: Файл '{segm_output_file_path}' не найден.")
    except msgpack.exceptions.FormatError:
        logger.error(
            f"Ошибка: Файл '{segm_output_file_path}' не является корректным MessagePack."
        )

    if need_to_segment:
        logger.info("Будет выполнена сегментация")
        # Загрузка модели VAD
        vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
        (get_speech_timestamps, _, _, _, _) = utils
        torch.cuda.empty_cache()
        gc.collect()
        # Выполнить сегментацию
        segments, boundaries = segment_audio(
            fileW,
            sample_rate=16000,
            vad_model=vad_model,
            get_speech_timestamps=get_speech_timestamps,
        )
        # Очистка памяти после сегментации
        torch.cuda.empty_cache()
        gc.collect()

        result_data_to_save = {
            "segments": [
                segment.tolist() for segment in segments
            ],  # Преобразуем массивы NumPy в списки
            "boundaries": boundaries,
        }

        with open(segm_output_file_path, "wb") as f:
            msgpack.dump(result_data_to_save, f)

        logger.info(f"Результат сохранен в файл {segm_output_file_path}")

    file_paths = []
    if temp_segments:
        for i, segment in enumerate(segments):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                torchaudio.save(
                    temp_file.name, torch.tensor(segment).unsqueeze(0), 16000
                )
                file_paths.append(temp_file.name)
    else:
        for i, segment in enumerate(segments):
            file_path = os.path.join(output_tmp_dir, f"{hashName}_{i}.wav")
            if not file_exist(file_path):
                torchaudio.save(file_path, torch.tensor(segment).unsqueeze(0), 16000)
                logger.info(f"Создан файл '{file_path}'")
            file_paths.append(file_path)

    trans_result_file_path = os.path.join(
        output_tmp_dir, f"{hashName}_transcription.json"
    )
    if not file_exist(trans_result_file_path):
        # Параллельная транскрипция
        # max_workers - количество потоков
        with ProcessPoolExecutor(max_workers=1, initializer=init_process) as executor:
            results = executor.map(
                transcribe_batch,
                [
                    file_paths[i : i + BATCH_SIZE]
                    for i in range(0, len(file_paths), BATCH_SIZE)
                ],
            )
            # Сегменты текста
            transcriptions = [trans for result in results for trans in result]
        with open(trans_result_file_path, "w", encoding="utf-8") as f:
            json.dump(transcriptions, f, ensure_ascii=False, indent=4)
    else:
        with open(trans_result_file_path, "r", encoding="utf-8") as f:
            transcriptions = json.load(f)

    out_trans_file_path = os.path.join(OUTPUT_DIR, f"{hashName}_transcription.txt")
    if not file_exist(out_trans_file_path):
        with open(out_trans_file_path, "w", encoding="utf-8") as f:
            for transcription in transcriptions:
                f.write(f"{transcription}\n")
        logger.info(f"Чистая транскрипция сохранена в файл {out_trans_file_path}")
    else:
        logger.info(f"Чистая транскрипция {out_trans_file_path} уже существует")

    out_trans_time_file_path = os.path.join(
        OUTPUT_DIR, f"{hashName}_transcription_time.txt"
    )
    if not file_exist(out_trans_time_file_path):
        with open(out_trans_time_file_path, "w", encoding="utf-8") as f:
            for transcription, boundary in zip(transcriptions, boundaries):
                boundary_0 = format_time(boundary[0])
                boundary_1 = format_time(boundary[1])
                f.write(f"[{boundary_0} - {boundary_1}]: {transcription}\n")
        logger.info(
            f"Транскрипция с временными метками сохранена в файл {out_trans_time_file_path}"
        )
    else:
        logger.info(
            f"Транскрипция с временными метками {out_trans_time_file_path} уже существует"
        )
        

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        logger.error(f"Ошибка: {e}")
        sys.exit(1)  # Завершить
