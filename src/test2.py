import locale

locale.getpreferredencoding = lambda: "UTF-8"

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

# Список нецензурных слов
bad_words = ["слово1", "слово2", "аборт"]  # Замените на реальные слова

# Функция для поиска нецензурных слов
def find_bad_words(text, bad_words):
    found_words = []
    for word in bad_words:
        if re.search(rf"\b{word}\b", text, re.IGNORECASE):
            found_words.append(word)
    return found_words

# Обработка каждого сегмента
for start, end in segments:
    segment_waveform = waveform[:, start:end]
    with torch.no_grad():
        logits = model.forward(input_signal=segment_waveform, input_signal_length=torch.tensor([segment_waveform.size(1)]).to(segment_waveform.device))
        decoded = model.decoding.ctc_decoder_predictions_tensor(logits)
        transcription = decoded[0][0]

    # Поиск нецензурных слов
    found_words = find_bad_words(transcription, bad_words)
    if found_words:
        start_time = start / sample_rate
        end_time = end / sample_rate
        print(f"Время: {start_time:.2f} - {end_time:.2f} сек. | Слова: {', '.join(found_words)}")
