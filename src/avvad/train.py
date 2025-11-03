"""Train the CRNN VAD and compare it against the energy/ZCR DSP baseline.

Usage:
    python -m avvad.train --data-dir ../data/raw --train-speakers 1 --val-speakers 2
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from avvad.dataset import GridVadDataset, collate_pad
from avvad.model import CRNNVad


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for feats, labels, mask in loader:
        feats, labels, mask = feats.to(device), labels.to(device), mask.to(device)
        optimizer.zero_grad()
        logits = model(feats)
        loss = criterion(logits, labels)
        loss = (loss * mask).sum() / mask.sum()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * feats.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for feats, labels, mask in loader:
        feats = feats.to(device)
        logits = model(feats)
        probs = torch.sigmoid(logits).cpu().numpy()
        mask_np = mask.numpy()
        labels_np = labels.numpy()
        for i in range(feats.size(0)):
            n = mask_np[i].sum()
            all_preds.append((probs[i, :n] > 0.5).astype(np.uint8))
            all_labels.append(labels_np[i, :n].astype(np.uint8))
    return np.concatenate(all_preds), np.concatenate(all_labels)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "raw")
    parser.add_argument("--train-speakers", type=int, nargs="+", default=[1,2,3,4,5])
    parser.add_argument("--val-speakers", type=int, nargs="+", default=[6])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parents[2] / "checkpoints" / "crnn_vad.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = GridVadDataset(args.data_dir, args.train_speakers)
    val_ds = GridVadDataset(args.data_dir, args.val_speakers)
    print(f"Train utterances: {len(train_ds)} (speakers {args.train_speakers})")
    print(f"Val utterances:   {len(val_ds)} (speakers {args.val_speakers})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad)

    model = CRNNVad().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"epoch {epoch:02d}  train_loss={loss:.4f}")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)
    print(f"Saved checkpoint to {args.checkpoint}")

    model_preds, model_labels = predict_model(model, val_loader, device)

    print("\n=== Validation results (held-out speaker) ===")
    print(f"CRNN     acc={accuracy_score(model_labels, model_preds):.4f}  f1={f1_score(model_labels, model_preds):.4f}")


if __name__ == "__main__":
    main()
