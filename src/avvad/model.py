"""Small CRNN for per-frame speech/silence classification."""
import torch
import torch.nn as nn


class CRNNVad(nn.Module):
    def __init__(self, n_mels: int = 64, gru_hidden: int = 64):
        super().__init__()
        # Input to conv is (batch, 1, n_frames, n_mels) -- pool kernel is (H, W) =
        # (time, mels), so (1, 2) halves the mel axis only and preserves
        # per-frame time resolution (needed for frame-level predictions).
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
        )
        conv_out_mels = n_mels // 4
        self.gru = nn.GRU(
            input_size=32 * conv_out_mels,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(gru_hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_frames, n_mels)
        x = x.unsqueeze(1)                       # (batch, 1, n_frames, n_mels)
        x = self.conv(x)                         # (batch, C, n_frames, n_mels')
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        x, _ = self.gru(x)                       # (batch, n_frames, 2*hidden)
        logits = self.classifier(x).squeeze(-1)  # (batch, n_frames)
        return logits
