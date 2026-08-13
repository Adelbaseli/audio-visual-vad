"""Real-time VAD prototype: mic -> rolling buffer -> ONNX model -> console.

Sliding-window approach: every UPDATE_INTERVAL seconds, re-runs the model on
the last WINDOW_SECONDS of audio and reports only the newest frame's
prediction. This deliberately avoids needing the model to expose/carry
hidden state between calls -- the model itself is untouched, this script
just keeps re-feeding it a recent window of audio.

Requires libportaudio2 (sounddevice's backend):
    sudo apt install libportaudio2
    pip install sounddevice
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avvad.features import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE, log_mel_spectrogram  # noqa: E402

WINDOW_SECONDS = 2.0
UPDATE_INTERVAL = 0.2
DEFAULT_ONNX = Path(__file__).resolve().parents[1] / "checkpoints" / "vad.onnx"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX)
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.onnx_path))

    window_len = int(WINDOW_SECONDS * SAMPLE_RATE)
    buffer = np.zeros(window_len, dtype=np.float32)

    def callback(indata, frames, time_info, status):
        nonlocal buffer
        chunk = indata[:, 0]
        buffer = np.concatenate([buffer, chunk])[-window_len:]

    print(f"Loaded {args.onnx_path}")
    print("Listening... (Ctrl+C to stop)\n")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        while True:
            feats = log_mel_spectrogram(buffer, SAMPLE_RATE).T.astype(np.float32)
            feats = feats[np.newaxis, :, :]  # (1, n_frames, n_mels)

            logits = session.run(None, {"mel_spectrogram": feats})[0][0]
            prob = sigmoid(logits[-1])  # newest frame only

            label = "SPEAKING" if prob > 0.5 else "silence "
            bar = "#" * int(prob * 30)
            print(f"\r[{label}] p={prob:.2f} {bar:<30}", end="", flush=True)

            time.sleep(UPDATE_INTERVAL)


if __name__ == "__main__":
    main()
