"""PyTorch Dataset over GRID audio + word alignments."""
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from avvad.features import SAMPLE_RATE, HOP_LENGTH, log_mel_spectrogram
from avvad.labels import frame_labels


class GridVadDataset(Dataset):
    def __init__(self, data_dir: Path, speakers: list[int]):
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        wav_path, align_path = self.samples[idx]
        wav, sr = sf.read(wav_path, dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError(f"{wav_path} has sample rate {sr}, expected {SAMPLE_RATE}")

        feats = log_mel_spectrogram(wav, sr)          # (n_mels, n_frames)
        labels = frame_labels(align_path, feats.shape[1], HOP_LENGTH, sr)

        return torch.from_numpy(feats.T.copy()), torch.from_numpy(labels.astype(np.float32))


def collate_pad(batch):
    """Pad variable-length (n_frames, n_mels) sequences to the max length in the batch."""
    feats, labels = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in feats])
    max_len = int(lengths.max())

    n_mels = feats[0].shape[1]
    padded_feats = torch.zeros(len(feats), max_len, n_mels)
    padded_labels = torch.zeros(len(feats), max_len)
    mask = torch.zeros(len(feats), max_len, dtype=torch.bool)

    for i, (f, l) in enumerate(zip(feats, labels)):
        n = f.shape[0]
        padded_feats[i, :n] = f
        padded_labels[i, :n] = l
        mask[i, :n] = True   
        

    return padded_feats, padded_labels, mask
