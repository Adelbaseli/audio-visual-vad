"""Audio front-end: log-mel spectrogram features and a classic DSP VAD baseline."""
import numpy as np
import librosa

SAMPLE_RATE = 25_000
N_FFT = 625          # 25ms at 25kHz
HOP_LENGTH = 250      # 10ms at 25kHz
N_MELS = 64


def log_mel_spectrogram(wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return (n_mels, n_frames) log-mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max)
