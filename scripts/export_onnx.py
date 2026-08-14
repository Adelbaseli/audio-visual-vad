"""Export the trained VAD checkpoint to an ONNX file."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avvad.model import CRNNVad

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "crnn_vad.pt"
ONNX_PATH = REPO_ROOT / "checkpoints" / "vad.onnx"


def main():
    model = CRNNVad()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    # A fake input just to show torch the expected shape: (batch, n_frames, n_mels).
    dummy_input = torch.randn(1, 100, 64)

    # dynamo=False: the newer exporter breaks this GRU model on variable-length
    # input, so we use the older, more reliable exporter instead.
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        input_names=["mel_spectrogram"],
        output_names=["logits"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch", 1: "n_frames"},
            "logits": {0: "batch", 1: "n_frames"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Saved ONNX model to {ONNX_PATH}")


if __name__ == "__main__":
    main()
