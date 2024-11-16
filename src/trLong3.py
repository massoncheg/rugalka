import locale
from typing import List, Tuple
import numpy as np
import logging
from silero_vad import read_audio  # Убрали импорт get_speech_timestamps
from sympy import false
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
import numpy as np

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

# сгенерировать уникальное имя
hashName = generate_code_from_file(PATH_TO_INPUT_FILE)

input_file_extension = os.path.splitext(PATH_TO_INPUT_FILE)[1]

BATCH_SIZE = 4
SEL_OUTPUT_DIR = filedialog.askdirectory(title="Выберите папку для выгрузки")
OUTPUT_DIR = f"{SEL_OUTPUT_DIR}\\{hashName}\\"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# может быть как видео так и аудио


temp_segments = False
output_tmp_dir = f"{OUTPUT_DIR}tmp\\"
force_re_segmentation = False
search_words = {"бля", "сук", "пид"}


def main():
    os.makedirs(output_tmp_dir, exist_ok=True)
    path_to_audio_file = f"{output_tmp_dir}_extracted_audio.wav"

    if not file_exist(path_to_audio_file):
        if input_file_extension in [".wav"]:
            #
            path_to_audio_file = PATH_TO_INPUT_FILE

        elif input_file_extension in [".mp4", ".webm", ".mp3", ".opus"]:

            # Временный путь для сохранения аудио в формате WAV

            # Извлечение аудио из MP4 с помощью ffmpeg
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
            #
            print(f"Ошибка: файлы {input_file_extension} не поддерживаются")
            sys.exit(1)

    if not file_exist(path_to_audio_file):
        print(f"Ошибка: исходно файла {path_to_audio_file} не существует")
        sys.exit(1)

    # Прочитать аудиофайл
    fileW = read_audio(path_to_audio_file, sampling_rate=16000)

    segm_output_file_path = f"{output_tmp_dir}{hashName}_segmentation_result.msgpack"

    need_to_segment = True

    try:
        # загрузить файл
        with open(segm_output_file_path, "rb") as f:
            data = msgpack.unpack(f)
            segments = data["segments"]
            boundaries = data["boundaries"]
        need_to_segment = False

    except FileNotFoundError:
        print(f"Ошибка: Файл '{segm_output_file_path}' не найден.")
    except msgpack.exceptions.FormatError:
        print(
            f"Ошибка: Файл '{segm_output_file_path}' не является корректным MessagePack."
        )

    if need_to_segment:
        print("будет выполнена сегментация")
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
            msgpack.pack(result_data_to_save, f)

        print(f"Результат сохранен в файл {segm_output_file_path}")

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
                # print(f"Файл '{file_path}' существует. Пропуска")
                torchaudio.save(file_path, torch.tensor(segment).unsqueeze(0), 16000)
                print(f"Создан Файл '{file_path}'")
            file_paths.append(file_path)

    trans_result_file_path = f"{output_tmp_dir}{hashName}_transcription.json"
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
            # сегменты текста
            transcriptions = [trans for result in results for trans in result]
        with open(trans_result_file_path, "w", encoding="utf-8") as f:
            json.dump(transcriptions, f, ensure_ascii=False, indent=4)
    else:
        with open(trans_result_file_path, "r", encoding="utf-8") as f:
            transcriptions = json.load(f)

    out_trans_file_path = f"{OUTPUT_DIR}{hashName}_transcription.txt"
    if not file_exist(out_trans_file_path):
        with open(out_trans_file_path, "w", encoding="utf-8") as f:
            for transcription in transcriptions:
                f.write(f"{transcription}\n")
        print(f"Чистая транскрипция сохранена в файл {out_trans_file_path}")
    else:
        print(f"Чистая транскрипция {out_trans_file_path} \nуже существует")

    out_trans_time_file_path = f"{OUTPUT_DIR}{hashName}_transcription_time.txt"
    if not file_exist(out_trans_time_file_path):
        with open(out_trans_time_file_path, "w", encoding="utf-8") as f:
            for transcription, boundary in zip(transcriptions, boundaries):
                boundary_0 = format_time(boundary[0])
                boundary_1 = format_time(boundary[1])
                f.write(f"[{boundary_0} - {boundary_1}]: {transcription}\n")
        print(
            f"Транскрипция с временными метками сохранена в файл {out_trans_time_file_path}\n\n"
        )
    else:
        print(
            f"Транскрипция с временными метками {out_trans_time_file_path} \nуже существует"
        )

    # search_results = []
    # for transcription, boundary in zip(transcriptions, boundaries):
    #     words = transcription.split()
    #     start_time = boundary[0]

    #     for word in words:
    #         normalized_word = word.lower().strip(",.?!")  # Убираем знаки препинания
    #         if normalized_word in search_words:
    #             search_results.append((normalized_word, start_time))

    # with open(f"{outputDir}{hashName}_searchResult.txt", "w", encoding="utf-8") as f:
    #     for word, start_time in search_results:
    #         formatted_time = format_time(start_time)
    #         f.write(f"{word}: {formatted_time}\n")


if __name__ == "__main__":
    try:
        main()
        raise RuntimeError("Произошла ошибка!")

    except RuntimeError as e:
        print(f"Ошибка: {e}")
        sys.exit(1)  # Завершить
