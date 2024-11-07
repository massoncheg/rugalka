# транскрипция коротких роликов 
import locale

locale.getpreferredencoding = lambda: "UTF-8"

from typing import List, Tuple
import numpy as np
import whisper
import logging

from silero_vad import get_speech_timestamps, read_audio

import torch
import torchaudio
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
)
from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FilterbankFeaturesTA(NeMoFilterbankFeaturesTA):
    def __init__(self, mel_scale: str = "htk", wkwargs=None, **kwargs):
        if "window_size" in kwargs:
            del kwargs["window_size"]
        if "window_stride" in kwargs:
            del kwargs["window_stride"]

        super().__init__(**kwargs)

        self._mel_spec_extractor: torchaudio.transforms.MelSpectrogram = (
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self._sample_rate,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=kwargs["nfilt"],
                window_fn=self.torch_windows[kwargs["window"]],
                mel_scale=mel_scale,
                norm=kwargs["mel_norm"],
                n_fft=kwargs["n_fft"],
                f_max=kwargs.get("highfreq", None),
                f_min=kwargs.get("lowfreq", 0),
                wkwargs=wkwargs,
            )
        )


class AudioToMelSpectrogramPreprocessor(NeMoAudioToMelSpectrogramPreprocessor):
    def __init__(self, mel_scale: str = "htk", **kwargs):
        super().__init__(**kwargs)
        kwargs["nfilt"] = kwargs["features"]
        del kwargs["features"]
        self.featurizer = (
            FilterbankFeaturesTA(  # Deprecated arguments; kept for config compatibility
                mel_scale=mel_scale,
                **kwargs,
            )
        )

device = "cuda" if torch.cuda.is_available() else "cpu"

model = EncDecCTCModel.from_config_file("./ctc_model_config.yaml")
ckpt = torch.load("./ctc_model_weights.ckpt", map_location="cpu")
model.load_state_dict(ckpt, strict=False)
model.eval()
model = model.to(device)

# ------------
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    trust_repo=True
)
(get_speech_timestamps, _, _, _, _) = utils



def segment_audio(
    wav: torch.Tensor,
    sample_rate: int,
    vad_model,
    max_duration: float = 22.0,
    min_duration: float = 15.0,
    new_chunk_threshold: float = 0.2,
) -> Tuple[List[np.ndarray], List[List[float]]]:
    """Сегментация аудио на основе активности речи."""
    logger.info("Начало сегментации аудио...")
    try:
        # Получение меток активности речи
        speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=sample_rate)
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


import subprocess
import torchaudio

# Путь к вашему MP4 файлу
video_path = "./input1.mp4"

# Временный путь для сохранения аудио в формате WAV
audio_path = "./extracted_audio.wav"

# Извлечение аудио из MP4 с помощью ffmpeg
# subprocess.run(["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path])


# Прочитать аудиофайл
fileW = read_audio(audio_path, sampling_rate=16000)

import tempfile

# Выполнить сегментацию
segments, boundaries = segment_audio(
    fileW,
    sample_rate=16000,
    vad_model=vad_model
)
BATCH_SIZE = 10

file_paths = []
for i, segment in enumerate(segments):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        torchaudio.save(temp_file.name, torch.tensor(segment).unsqueeze(0), 16000)  # Сохранение сегмента во временный файл

        file_paths.append(temp_file.name)

# Передаем пути к файлам вместо массивов данных
transcriptions = model.transcribe(file_paths, batch_size=BATCH_SIZE)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    full_seconds = int(seconds)
    milliseconds = int((seconds - full_seconds) * 100)

    if hours > 0:
        return f"{hours:02}:{minutes:02}:{full_seconds:02}:{milliseconds:02}"
    else:
        return f"{minutes:02}:{full_seconds:02}:{milliseconds:02}"

with open("transcription.txt", "w", encoding="utf-8") as f:

    for transcription, boundary in zip(transcriptions, boundaries):
        boundary_0 = format_time(boundary[0])
        boundary_1 = format_time(boundary[1])
        f.write(f"[{boundary_0} - {boundary_1}]: {transcription}\n")