from typing import List, Tuple
import numpy as np
import torch
import torchaudio
import whisper
import logging
from silero_vad import VoiceActivityDetector
from silero_vad.utils import get_speech_timestamps, read_audio

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Установка числа потоков для PyTorch
torch.set_num_threads(1)

# Загрузка модели Silero VAD
logger.info("Загрузка модели Silero VAD...")
try:
    vad_model = VoiceActivityDetector("silero_vad")  # Используем установленную модель
except Exception as e:
    logger.error(f"Ошибка при загрузке модели Silero VAD: {e}")
    raise

# Загрузка модели ASR (Whisper)
logger.info("Загрузка модели ASR (Whisper)...")
try:
    asr_model = whisper.load_model('base')
except Exception as e:
    logger.error(f"Ошибка при загрузке модели ASR: {e}")
    raise

def segment_audio(
    wav: torch.Tensor,
    sample_rate: int,
    model: VoiceActivityDetector,
    max_duration: float = 22.0,
    min_duration: float = 15.0,
    new_chunk_threshold: float = 0.2,
) -> Tuple[List[np.ndarray], List[List[float]]]:
    """Сегментация аудио на основе активности речи."""
    logger.info("Начало сегментации аудио...")
    try:
        # Получение меток активности речи
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sample_rate)
    except Exception as e:
        logger.error(f"Ошибка при получении меток активности речи: {e}")
        raise

    segments = []
    boundaries = []
    curr_duration = 0.0
    curr_start = None
    curr_end = None

    for segment in speech_timestamps:
        start = segment['start'] / sample_rate  # Время в секундах
        end = segment['end'] / sample_rate

        if curr_start is None:
            curr_start = start
            curr_end = end
            curr_duration = curr_end - curr_start
            continue

        if (
            (start - curr_end > new_chunk_threshold and curr_duration >= min_duration)
            or (curr_duration + (end - curr_end) > max_duration)
        ):
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

def format_time(seconds: float) -> str:
    """Форматирование времени в часы:минуты:секунды:миллисекунды."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    full_seconds = int(seconds_remainder)
    milliseconds = int((seconds_remainder - full_seconds) * 100)

    if hours > 0:
        return f"{hours:02}:{minutes:02}:{full_seconds:02}:{milliseconds:02}"
    else:
        return f"{minutes:02}:{full_seconds:02}:{milliseconds:02}"

def transcribe_segments(segments: List[np.ndarray]) -> List[str]:
    """Транскрибирование сегментов аудио."""
    logger.info("Начало транскрибирования сегментов...")
    transcriptions = []
    for i, segment in enumerate(segments):
        try:
            # Преобразование numpy array в формат, совместимый с Whisper
            audio_tensor = torch.from_numpy(segment).float()
            result = asr_model.transcribe(audio_tensor, language='en')
            transcriptions.append(result['text'])
            logger.debug(f"Сегмент {i+1} транскрибирован.")
        except Exception as e:
            logger.error(f"Ошибка при транскрибировании сегмента {i+1}: {e}")
            transcriptions.append("")
    logger.info("Транскрибирование сегментов завершено.")
    return transcriptions

def main():
    wav_path = './long_example.wav'

    logger.info(f"Чтение аудио файла {wav_path}...")
    try:
        wav = read_audio(wav_path, sampling_rate=16000)
    except Exception as e:
        logger.error(f"Ошибка при чтении аудио файла: {e}")
        return

    segments, boundaries = segment_audio(
        wav,
        sample_rate=16000,
        model=vad_model
    )

    transcriptions = transcribe_segments(segments)

    logger.info("Результаты транскрибирования:")
    for transcription, boundary in zip(transcriptions, boundaries):
        boundary_start = format_time(boundary[0])
        boundary_end = format_time(boundary[1])
        print(f"[{boundary_start} - {boundary_end}]: {transcription}\n")

if __name__ == "__main__":
    main()
