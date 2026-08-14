"""Audio front-end: log-mel spectrogram features and a classic DSP VAD baseline."""
import numpy as np
import librosa
import torch
import torchaudio

SAMPLE_RATE = 25_000
N_FFT = 625          # 25ms at 25kHz
HOP_LENGTH = 250      # 10ms at 25kHz
N_MELS = 64


def n_frames_for(n_samples: int, hop_length: int = HOP_LENGTH) -> int:
    """Number of STFT frames librosa/torchaudio produce for n_samples (center=True).

    This is ceil(n_samples / hop_length), NOT 1 + n_samples // hop_length -- the
    latter overshoots by one whenever n_samples is an exact multiple of hop_length.
    """
    return -(-n_samples // hop_length)


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


def build_mel_transform(device=None):
    """GPU-capable batched equivalent of log_mel_spectrogram, for training throughput.

    Parameters are matched to librosa's defaults (pad_mode="constant", slaney mel
    scale + norm) -- empirically verified to within ~0.1dB mean absolute difference
    against log_mel_spectrogram on real GRID clips. realtime_vad.py and the ONNX
    export stay on the CPU/librosa path unchanged: a single ~2s window recomputed
    every 0.2s was never the bottleneck, only batched training throughput is.
    """
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
        power=2.0, center=True, pad_mode="constant", norm="slaney", mel_scale="slaney",
    )
    db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    if device is not None:
        mel, db = mel.to(device), db.to(device)
    return mel, db


def log_mel_spectrogram_batch(wavs: torch.Tensor, mel_transform, db_transform) -> torch.Tensor:
    """wavs: (batch, n_samples) -> (batch, n_mels, n_frames), fixed ref=1.0.

    AmplitudeToDB's top_db floor is computed per-leading-batch-dim only when the
    input is 4D+; a bare 3D (batch, n_mels, n_frames) tensor gets a single floor
    for the whole batch instead of librosa's per-clip floor. The unsqueeze/squeeze
    here forces the per-sample behavior that matches log_mel_spectrogram.
    """
    mel = mel_transform(wavs)
    return db_transform(mel.unsqueeze(1)).squeeze(1)
