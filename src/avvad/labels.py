"""Parse GRID .align files into frame-level speech/silence labels.

GRID alignment timestamps are sample indices at 25000 Hz -- the same rate
as the "25kHz endpointed" audio distribution, so no resampling of the
timestamps themselves is needed.
"""
from pathlib import Path

import numpy as np

ALIGN_SAMPLE_RATE = 25_000
SILENCE_TOKENS = {"sil", "sp"}


def parse_align(align_path: Path) -> list[tuple[int, int, str]]:
    intervals = []
    for line in Path(align_path).read_text().splitlines():
        start, end, word = line.split()
        intervals.append((int(start), int(end), word))
    return intervals


def frame_labels(
    align_path: Path,
    num_frames: int,
    hop_length: int,
    sample_rate: int,
) -> np.ndarray:
    """Return a (num_frames,) uint8 array: 1 = speech, 0 = silence.

    Frame i is labeled speech if its center sample falls inside a
    non-silence interval.
    """
    intervals = parse_align(align_path)
    scale = sample_rate / ALIGN_SAMPLE_RATE

    labels = np.zeros(num_frames, dtype=np.uint8)
    frame_centers = (np.arange(num_frames) * hop_length + hop_length // 2)

    for start, end, word in intervals:
        if word in SILENCE_TOKENS:
            continue
        s, e = start * scale, end * scale
        mask = (frame_centers >= s) & (frame_centers < e)
        labels[mask] = 1

    return labels
