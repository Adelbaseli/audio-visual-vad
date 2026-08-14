"""PyTorch Dataset over GRID audio + word alignments."""
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from avvad.features import SAMPLE_RATE, HOP_LENGTH, n_frames_for
from avvad.labels import frame_labels


MAX_SILENCE_PAD_S = 3.0
DEFAULT_NOISE_DIR = Path(__file__).resolve().parents[2] / "data" / "noise"


class GridVadDataset(Dataset):
    def __init__(self, data_dir: Path, speakers: list[int], augment_silence: bool = True, noise_dir: Path = DEFAULT_NOISE_DIR):
        self.samples: list[tuple[Path, Path]] = []
        for sid in speakers:
            speaker_dir = Path(data_dir) / f"s{sid}"
            audio_dir = speaker_dir / "audio"
            align_dir = speaker_dir / "align"
            for wav_path in sorted(audio_dir.glob("*.wav")):
                align_path = align_dir / f"{wav_path.stem}.align"
                if align_path.exists():
                    self.samples.append((wav_path, align_path))

        if not self.samples:
            raise RuntimeError(f"No (audio, align) pairs found under {data_dir} for speakers {speakers}")

        self.augment_silence = augment_silence

        # Real recorded ambient noise (mic hum, fan, room tone) for silence
        # augmentation, alongside synthetic Gaussian noise. Gaussian noise is
        # spectrally flat and only teaches "reject broadband noise" -- real
        # ambient noise has energy concentrated in specific bands that can
        # resemble speech formants, and a model never exposed to that during
        # training has no reason to reject it at inference time.
        self.noise_clips: list[np.ndarray] = []
        if noise_dir is not None and Path(noise_dir).is_dir():
            for noise_path in sorted(Path(noise_dir).glob("*.wav")):
                clip, clip_sr = sf.read(noise_path, dtype="float32")
                if clip_sr != SAMPLE_RATE:
                    raise ValueError(f"{noise_path} has sample rate {clip_sr}, expected {SAMPLE_RATE}")
                self.noise_clips.append(clip)

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_pad(self, pad_n: int) -> np.ndarray:
        if self.noise_clips and np.random.random() < 0.5:
            clip = self.noise_clips[np.random.randint(len(self.noise_clips))]
            start = np.random.randint(0, len(clip) - min(pad_n, len(clip)) + 1)
            crop = clip[start:start + pad_n]
            if len(crop) < pad_n:  # pad_n longer than the clip: tile it
                reps = pad_n // len(crop) + 1
                crop = np.tile(crop, reps)[:pad_n]
            gain = np.random.uniform(0.5, 2.0)
            return (crop * gain).astype(np.float32)

        noise_sigma = np.random.uniform(0.001, 0.02)  # realistic mic noise floor
        return (np.random.standard_normal(pad_n) * noise_sigma).astype(np.float32)

    def __getitem__(self, idx: int):
        wav_path, align_path = self.samples[idx]
        wav, sr = sf.read(wav_path, dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError(f"{wav_path} has sample rate {sr}, expected {SAMPLE_RATE}")

        orig_frames = n_frames_for(len(wav))

        if self.augment_silence:
            # GRID utterances are short and structured -- brief lead-in
            # silence, then mostly speech to the end. A model trained only
            # on that shape learns "elapsed time" as a proxy for "probably
            # speech by now", which falls apart on a real streaming window
            # that can sit in silence indefinitely. Prepending a random
            # span of noise-floor-level silence (up to and past a realistic
            # streaming window) decorrelates position from label so the
            # model has to use actual spectral content instead.
            pad_s = np.random.uniform(0, MAX_SILENCE_PAD_S)
            pad_n = int(pad_s * sr)
            if pad_n > 0:
                pad = self._sample_pad(pad_n)
                wav = np.concatenate([pad, wav])

        # Mel-spectrogram extraction happens batched on GPU in train.py, not here --
        # this just returns raw waveform + labels so DataLoader workers stay cheap.
        pad_frames = n_frames_for(len(wav)) - orig_frames
        labels = np.concatenate([
            np.zeros(pad_frames, dtype=np.uint8),
            frame_labels(align_path, orig_frames, HOP_LENGTH, sr),
        ])

        return torch.from_numpy(wav.copy()), torch.from_numpy(labels.astype(np.float32))


def collate_pad(batch):
    """Pad variable-length (raw waveform, frame labels) pairs to the batch max."""
    wavs, labels = zip(*batch)
    max_wav_len = max(w.shape[0] for w in wavs)
    max_frames = n_frames_for(max_wav_len)

    padded_wavs = torch.zeros(len(wavs), max_wav_len)
    padded_labels = torch.zeros(len(labels), max_frames)
    mask = torch.zeros(len(labels), max_frames, dtype=torch.bool)

    for i, (w, l) in enumerate(zip(wavs, labels)):
        padded_wavs[i, :w.shape[0]] = w
        n = l.shape[0]
        padded_labels[i, :n] = l
        mask[i, :n] = True

    return padded_wavs, padded_labels, mask
