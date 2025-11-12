# Audio-Visual VAD

Frame-level Voice Activity Detection (speech / silence), starting audio-only
and extending to audio-visual fusion (lip motion) for robustness in noisy,
multi-speaker rooms.

## Checkpoints

1. **Audio-only VAD (offline)** — log-mel CRNN trained on GRID corpus,
   compared against a classic energy/ZCR DSP baseline.
2. **Browser demo (audio-only)** — ONNX export + real-time mic-driven
   demo in the browser via onnxruntime-web.
3. **Lip-fusion (stretch)** — add MediaPipe Face Landmarker lip motion as a
   second input stream, fused with the audio branch.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Checkpoint 1

```bash
python scripts/download_grid.py --speakers 1 2
python -m avvad.train --train-speakers 1 --val-speakers 2
```

Trains on speaker 1, validates on held-out speaker 2, and prints CRNN vs.
DSP-baseline accuracy/F1 on the validation split.
