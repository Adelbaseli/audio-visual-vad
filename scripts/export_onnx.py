"""Export the trained CRNN VAD checkpoint to ONNX.

Usage (from audio-visual-vad/):
    python scripts/export_onnx.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avvad.model import CRNNVad  # noqa: E402

CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "crnn_vad.pt"
OUT_PATH = Path(__file__).resolve().parents[1] / "checkpoints" / "vad.onnx"


def main():
    model = CRNNVad()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 100, 64)  # (batch, n_frames, n_mels)

    # dynamo=True (the new default) decomposes nn.GRU in a way that bakes the
    # dummy input's n_frames into an internal reshape, breaking on any other
    # sequence length. The legacy TorchScript exporter (dynamo=False) traces
    # nn.GRU natively as an ONNX GRU op and handles dynamic_axes correctly.
    torch.onnx.export(
        model,
        dummy,
        str(OUT_PATH),
        input_names=["mel_spectrogram"],
        output_names=["logits"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch", 1: "n_frames"},
            "logits": {0: "batch", 1: "n_frames"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported ONNX model to {OUT_PATH}")


if __name__ == "__main__":
    main()
