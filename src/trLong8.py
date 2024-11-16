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
import os
import json
import string
import sys
import tkinter as tk
from tkinter import filedialog
import msgpack
import csv

from utils.utils import (
    file_exist,
    segment_audio,
    init_process,
    transcribe_batch,
    generate_code_from_file,
    read_large_json,
)

locale.getpreferredencoding = lambda: "UTF-8"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

root = tk.Tk()
root.withdraw()


# fps проекта (частота кадров)
proj_fps = 30  # Замените 30 на частоту кадров вашего проекта, если она отличается
markers_with_duration = False
add_words_to_markers = False


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


def format_time_to_timecode(seconds, fps=(proj_fps - 1)):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    frames = int(round((seconds % 1) * fps))  # Округляем кадры до ближайшего целого
    return f"{hours:02}:{minutes:02}:{secs:02}:{frames:02}"


def main():
    # можно быстро использовать из консоли
    if len(sys.argv) < 3:
        PATH_TO_INPUT_FILE, SEL_OUTPUT_DIR = get_user_inputs()
    else:
        PATH_TO_INPUT_FILE = sys.argv[1]
        SEL_OUTPUT_DIR = sys.argv[2]

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
    CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
    # путь к словарю
    path_to_dict = os.path.join(CONFIG_DIR, "search_words.txt")
    if not file_exist(path_to_dict):
        logger.error(f"Словарь {path_to_dict} не существует")
        sys.exit(1)

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
            transcriptions = []
            for result in results:
                transcriptions.extend(result)
                torch.cuda.empty_cache()
                gc.collect()
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
    #
    if not file_exist(out_trans_time_file_path):
        with open(out_trans_time_file_path, "w", encoding="utf-8") as f:
            for transcription, boundary in zip(transcriptions, boundaries):
                boundary_0 = format_time_to_timecode(boundary[0])
                boundary_1 = format_time_to_timecode(boundary[1])
                f.write(f"[{boundary_0} - {boundary_1}]: {transcription}\n")
        logger.info(
            f"Транскрипция с временными метками сохранена в файл {out_trans_time_file_path}"
        )
    else:
        logger.info(
            f"Транскрипция с временными метками {out_trans_time_file_path} уже существует"
        )
        transcriptions = []
        with open(out_trans_time_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if "]:" in line:
                    transcription = line.split("]:", 1)[1].strip()
                    transcriptions.append(transcription)
    # ---- отсюда начинается поиск
    logger.info("Начинается поиск слов:")

    # Загрузка словаря с логированием
    search_words_set = set()
    with open(path_to_dict, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.strip().lower()
            if clean_line:
                search_words_set.add(clean_line)

    logger.info(
        f"Загружено {len(search_words_set)} слов из словаря: {', '.join(search_words_set)}"
    )

    search_results = []

    for i, transcription in enumerate(transcriptions):
        segment_start = boundaries[i][0]  # Начало сегмента
        segment_duration = boundaries[i][1] - boundaries[i][0]  # Длительность сегмента
        words = transcription.split()  # Разбиение на слова
        word_duration = (
            segment_duration / len(words) if words else 0
        )  # Длительность одного слова

        for word_index, word in enumerate(words):
            clean_word = word.strip(string.punctuation).lower()

            found_search_words = set()
            for search_word in search_words_set:
                if search_word in clean_word:
                    found_search_words.add(search_word)

            if found_search_words:
                word_start_time = segment_start + word_index * word_duration
                word_end_time = word_start_time + word_duration

                for found_word in found_search_words:
                    logger.info(
                        f"Найдено слово '{found_word}' в слове '{clean_word}' в сегменте {i} "
                        f"(начало: {format_time_to_timecode(word_start_time)}, "
                        f"конец: {format_time_to_timecode(word_end_time)})"
                    )
                    search_results.append((clean_word, word_start_time, word_end_time))

    # Логирование итогов поиска
    logger.info(f"Найдено {len(search_results)} совпадений.")
    if search_results:
        found_words_summary = ", ".join([word for word, _, _ in search_results])
        logger.info(f"Список найденных слов: {found_words_summary}")

    # Сохранение в CSV
    csv_file_path = os.path.join(OUTPUT_DIR, f"{hashName}_markers.csv")
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(
            ["Marker Name", "Description", "In", "Out", "Duration", "Marker Type"]
        )

        for word, start_time, end_time in search_results:
            formatted_start = format_time_to_timecode(start_time)
            formatted_end = format_time_to_timecode(
                end_time if markers_with_duration else start_time
            )
            duration = (
                format_time_to_timecode(end_time - start_time)
                if markers_with_duration
                else "00:00:00:00"
            )

            writer.writerow(
                [word if add_words_to_markers else "", "", formatted_start, formatted_end, duration, "Comment"]
            )

    logger.info(f"Результаты сохранены в файл: {csv_file_path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        logger.error(f"Ошибка: {e}")
        sys.exit(1)  # Завершить
