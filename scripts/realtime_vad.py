"""Listen to the microphone and print live speech/silence predictions.

Requires libportaudio2 (sounddevice's backend):
    sudo apt install libportaudio2
    pip install sounddevice
"""
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avvad.features import SAMPLE_RATE, log_mel_spectrogram

ONNX_PATH = Path(__file__).resolve().parents[1] / "checkpoints" / "vad.onnx"
WINDOW_SECONDS = 2.0   # how much recent audio we look at each time
UPDATE_SECONDS = 0.2   # how often we make a new prediction
WINDOW_SIZE = int(WINDOW_SECONDS * SAMPLE_RATE)

# Rolling buffer of the most recent audio, updated by the mic callback below.
audio_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)


def on_audio(indata, frames, time_info, status):
    global audio_buffer
    new_audio = indata[:, 0]
    audio_buffer = np.concatenate([audio_buffer, new_audio])[-WINDOW_SIZE:]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    session = ort.InferenceSession(str(ONNX_PATH))
    print(f"Loaded model from {ONNX_PATH}")
    print("Listening... press Ctrl+C to stop\n")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=on_audio):
        while True:
            features = log_mel_spectrogram(audio_buffer, SAMPLE_RATE)
            features = features.T[np.newaxis, :, :].astype(np.float32)  # (1, n_frames, n_mels)

            logits = session.run(None, {"mel_spectrogram": features})[0][0]
            probability = sigmoid(logits[-1])  # only the newest frame matters

            status_text = "SPEAKING" if probability > 0.5 else "silence "
            bar = "#" * int(probability * 30)
            print(f"\r[{status_text}] p={probability:.2f} {bar:<30}", end="", flush=True)

            time.sleep(UPDATE_SECONDS)


if __name__ == "__main__":
    main()
