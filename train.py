"""
train.py

Trains WhisperSLU on preprocessed SLURP data (output of preprocess.py).

Usage:
  python train.py \
      --train_file train_processed.jsonl \
      --val_file val_processed.jsonl \
      --vocab vocab.json \
      --epochs 20 --batch_size 16 --lr 3e-4
"""

import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperFeatureExtractor

from model import WhisperSLU, build_full_vocab, ids_to_json


class SLUDataset(Dataset):
    """Loads (audio_file, target_ids) pairs and computes Whisper log-mel features on the fly."""

    def __init__(self, jsonl_path, whisper_model_name="openai/whisper-small"):
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))
        self.feat_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        import torchaudio
        wav, sr = torchaudio.load(ex["audio_file"])
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        feats = self.feat_extractor(
            wav.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt"
        )["input_features"][0]  # (80, 3000)
        target_ids = torch.tensor(ex["target_ids"], dtype=torch.long)
        return feats, target_ids


def make_collate_fn(pad_id):
    def collate(batch):
        feats, targets = zip(*batch)
        feats = torch.stack(feats, dim=0)  # all same length (3000) by construction
        max_len = max(t.size(0) for t in targets)
        padded = torch.full((len(targets), max_len), pad_id, dtype=torch.long)
        for i, t in enumerate(targets):
            padded[i, : t.size(0)] = t
        return feats, padded
    return collate


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for feats, targets in loader:
        feats, targets = feats.to(device), targets.to(device)
        decoder_in = targets[:, :-1]
        decoder_out = targets[:, 1:]

        logits = model(feats, decoder_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * feats.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for feats, targets in loader:
        feats, targets = feats.to(device), targets.to(device)
        decoder_in = targets[:, :-1]
        decoder_out = targets[:, 1:]
        logits = model(feats, decoder_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_out.reshape(-1))
        total_loss += loss.item() * feats.size(0)
    return total_loss / len(loader.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--val_file", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--whisper_model_name", default="openai/whisper-small")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--freeze_encoder", action="store_true", default=True)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--out_dir", default="checkpoints")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_info = build_full_vocab(args.vocab, args.whisper_model_name)
    pad_id = vocab_info["closed_vocab"]["token2id"]["<pad>"]

    train_ds = SLUDataset(args.train_file, args.whisper_model_name)
    val_ds = SLUDataset(args.val_file, args.whisper_model_name)
    collate = make_collate_fn(pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = WhisperSLU(vocab_info, args.whisper_model_name, freeze_encoder=args.freeze_encoder).to(device)

    # only optimize params that require grad (decoder + any unfrozen encoder layers)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=args.label_smoothing)

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{args.out_dir}/best_model.pt")
            print(f"  saved new best checkpoint (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    main()
