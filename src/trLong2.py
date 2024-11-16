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


locale.getpreferredencoding = lambda: "UTF-8"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

root = tk.Tk()
root.withdraw()

PATH_TO_INPUT_FILE = filedialog.askopenfilename(
    title="Выберите исходный файл",
    filetypes=[
        ("Media Files", "*.mp3 *.mp4 *.wav *.webm"),
        ("All Files", "*.*"),
    ],
)

if not PATH_TO_INPUT_FILE:
    logger.error("Файл не был выбран.")
    sys.exit(1)



BATCH_SIZE = 4
OUTPUT_DIR = f"{filedialog.askdirectory(title="Выберите папку для выгрузки")}\\{()}"
# может быть как видео так и аудио
path_to_input_file = filedialog.askopenfilename(
    title="Выберите исходный файл",
    filetypes=[
        ("Media Files", "*.mp3 *.mp4 *.wav *.webm"),
        ("All Files", "*.*"),
    ],
)
input_file_extension = os.path.splitext(path_to_input_file)[1]

OUTPUT_DIR = filedialog.askdirectory(title="Выберите папку для выгрузки")
temp_segments = False
output_tmp_dir = "../output/tmp/"
force_re_segmentation = False
search_words = {"бля", "сук", "пид"}


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

    cudda_av = torch.cuda.is_available()
    logger.info("cuda is ", cudda_av)
    device = "cuda" if cudda_av else "cpu"
    local_model = EncDecCTCModel.from_config_file("./ctc_model_config.yaml")
    ckpt = torch.load("./ctc_model_weights.ckpt", map_location=device)
    local_model.load_state_dict(ckpt, strict=False)
    local_model.eval()
    local_model = local_model.to(device)


def transcribe_batch(batch):
    return local_model.transcribe(batch, batch_size=BATCH_SIZE)


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    full_seconds = int(seconds)
    milliseconds = int((seconds - full_seconds) * 1000)

    if hours > 0:
        return f"{hours:02}:{minutes:02}:{full_seconds:02}:{milliseconds:03}"
    else:
        return f"{minutes:02}:{full_seconds:02}:{milliseconds:03}"


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


def main():
    os.makedirs(output_tmp_dir, exist_ok=True)
    if input_file_extension in [".mp3", ".wav"]:
        #
        path_to_audio_file = path_to_input_file

    elif input_file_extension in [".mp4", ".webm"]:

        # Временный путь для сохранения аудио в формате WAV
        path_to_audio_file = f"{output_tmp_dir}extracted_audio.wav"

        # Извлечение аудио из MP4 с помощью ffmpeg
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                path_to_input_file,
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
    # сгенерировать уникальное имя
    hashName = generate_code_from_file(path_to_audio_file)
    # Прочитать аудиофайл
    fileW = read_audio(path_to_audio_file, sampling_rate=16000)

    segm_output_file_path = f"{output_tmp_dir}{hashName}_segmentation_result.json"

    need_to_segment = True

    try:
        # загрузить файл
        with open(segm_output_file_path, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)

        # Восстановление сегментов и границ
        segments = [
            np.array(segment) for segment in loaded_data["segments"]
        ]  # Преобразуем списки обратно в массивы NumPy
        boundaries = loaded_data["boundaries"]
        need_to_segment = False
    except FileNotFoundError:
        print(f"Ошибка: Файл '{segm_output_file_path}' не найден.")
    except json.JSONDecodeError:
        print(f"Ошибка: Файл '{segm_output_file_path}' не является корректным JSON.")

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

        with open(segm_output_file_path, "w", encoding="utf-8") as f:
            json.dump(result_data_to_save, f, ensure_ascii=False, indent=4)
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
            if file_exist(file_path):
                print(f"Файл '{file_path}' существует.")
            else:
                print(f"Файл '{file_path}' не найден.")
                torchaudio.save(file_path, torch.tensor(segment).unsqueeze(0), 16000)
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
