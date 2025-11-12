"""Download a subset of the GRID corpus (25kHz endpointed audio + word alignments).

Video download is intentionally left out of Checkpoint 1 (audio-only VAD).
It gets added in Checkpoint 3 when we build the lip-fusion branch.
"""
import argparse
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

BASE_URL = "https://spandh.dcs.shef.ac.uk/gridcorpus"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download_speaker(speaker_id: int, data_dir: Path) -> None:
    speaker = f"s{speaker_id}"
    speaker_dir = data_dir / speaker
    speaker_dir.mkdir(parents=True, exist_ok=True)

    for kind, tar_name in (("audio", f"{speaker}.tar"), ("align", f"{speaker}.tar")):
        dest_tar = speaker_dir / f"{kind}.tar"
        extract_dir = speaker_dir / kind
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"[{speaker}] {kind} already present, skipping")
            continue

        url = f"{BASE_URL}/{speaker}/{kind}/{tar_name}"
        print(f"[{speaker}] downloading {url}")
        urlretrieve(url, dest_tar)

        print(f"[{speaker}] extracting {kind}")
        with tarfile.open(dest_tar) as tf:
            tf.extractall(speaker_dir)
        dest_tar.unlink()

        # The audio tar's top-level dir is named after the speaker (e.g. "s1"),
        # not "audio" -- normalize so downstream code can rely on kind-named dirs.
        speaker_named_dir = speaker_dir / speaker
        if speaker_named_dir.exists() and not extract_dir.exists():
            speaker_named_dir.rename(extract_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--speakers", type=int, nargs="+", default=list(range(22,35)),
        help="GRID speaker IDs to download (1-34, no 21)",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    for sid in args.speakers:
        download_speaker(sid, args.data_dir)


if __name__ == "__main__":
    main()
