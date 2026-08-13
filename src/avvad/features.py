"""Audio front-end: log-mel spectrogram features and a classic DSP VAD baseline."""
import numpy as np
import librosa

SAMPLE_RATE = 25_000
N_FFT = 625          # 25ms at 25kHz
HOP_LENGTH = 250      # 10ms at 25kHz
N_MELS = 64


def log_mel_spectrogram(wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return (n_mels, n_frames) log-mel spectrogram.

    Uses a fixed reference (ref=1.0), not ref=np.max. Per-clip max
    normalization breaks down on short streaming windows that contain no
    genuinely loud sound (e.g. a silent room) -- silence divided by
    near-silence degenerates to a flat, artificial 0dB pattern the model
    never saw in training. A fixed reference keeps silence looking like
    silence regardless of what else is (or isn't) in the window.
    """
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=1.0)
