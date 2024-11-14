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

locale.getpreferredencoding = lambda: "UTF-8"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BATCH_SIZE = 4


def segment_audio(
    wav: torch.Tensor,
    sample_rate: int,
    vad_model,
    get_speech_timestamps,
    max_duration: float = 22.0,
    min_duration: float = 15.0,
    new_chunk_threshold: float = 0.2,
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


if __name__ == "__main__":
    logger.info("cuda is ", torch.cuda.is_available())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Инициализация модели ASR
    model = EncDecCTCModel.from_config_file("./ctc_model_config.yaml")
    ckpt = torch.load("./ctc_model_weights.ckpt", map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    model = model.to(device)

    torch.cuda.empty_cache()
    gc.collect()


    # Загрузка модели VAD
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )
    (get_speech_timestamps, _, _, _, _) = utils

    torch.cuda.empty_cache()
    gc.collect()


    # # Путь к вашему MP4 файлу
    # video_path = "../input/input.mp4"

    # # Временный путь для сохранения аудио в формате WAV
    # audio_path = "../output/extracted_audio.wav"

    # # Извлечение аудио из MP4 с помощью ffmpeg
    # subprocess.run(
    #     [
    #         "ffmpeg",
    #         "-i",
    #         video_path,
    #         "-vn",
    #         "-acodec",
    #         "pcm_s16le",
    #         "-ar",
    #         "16000",
    #         "-ac",
    #         "1",
    #         audio_path,
    #     ],
    #     check=True,
    # )
    audio_path = "../input/input.wav"

    # Прочитать аудиофайл
    fileW = read_audio(audio_path, sampling_rate=16000)

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


    file_paths = []
    for i, segment in enumerate(segments):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            torchaudio.save(temp_file.name, torch.tensor(segment).unsqueeze(0), 16000)
            file_paths.append(temp_file.name)

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
        transcriptions = [trans for result in results for trans in result]

    # Сохранение результатов
    with open("../output/transcription.txt", "w", encoding="utf-8") as f:
        for transcription, boundary in zip(transcriptions, boundaries):
            boundary_0 = format_time(boundary[0])
            boundary_1 = format_time(boundary[1])
            f.write(f"[{boundary_0} - {boundary_1}]: {transcription}\n")
