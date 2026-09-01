"""
lora_finetune.py

Stage-3 fine-tuning: load a decoder that was already trained with a FROZEN
Whisper encoder (train.py, freeze_encoder=True), then attach LoRA adapters
to the encoder's attention projections and continue training end-to-end at
a low learning rate. This adapts Whisper's acoustic features to SLURP's
domain (far-field, accented, noisy home-assistant speech) without the cost
or forgetting risk of full encoder fine-tuning.

Requires: pip install peft

Usage:
  python lora_finetune.py \
      --train_file train_processed.jsonl --val_file val_processed.jsonl \
      --vocab vocab.json --init_checkpoint checkpoints/best_model.pt \
      --epochs 5 --batch_size 8 --lr 1e-5 --lora_r 16
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from peft import LoraConfig, get_peft_model

from model import WhisperSLU, build_full_vocab
from train import SLUDataset, make_collate_fn, train_one_epoch, evaluate


def attach_lora(model, r=16, alpha=32, dropout=0.05):
    """
    Attach LoRA adapters to the Whisper encoder's attention projections
    (q_proj, v_proj — the standard LoRA targets for transformer attention).
    Everything else (decoder, embeddings, out_proj) stays fully trainable
    as-is since it was randomly initialized and needs full-rank updates.
    """
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=dropout,
        bias="none",
    )
    model.encoder = get_peft_model(model.encoder, lora_config)
    model.encoder.print_trainable_parameters()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--val_file", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--whisper_model_name", default="openai/whisper-small")
    ap.add_argument("--init_checkpoint", required=True,
                     help="decoder checkpoint from stage-2 (frozen-encoder) training")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--out_dir", default="checkpoints_lora")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_info = build_full_vocab(args.vocab, args.whisper_model_name)
    pad_id = vocab_info["closed_vocab"]["token2id"]["<pad>"]

    # build model with encoder UNFROZEN before wrapping in LoRA, so peft
    # controls which params are trainable via the adapter config
    model = WhisperSLU(vocab_info, args.whisper_model_name, freeze_encoder=False)

    state_dict = torch.load(args.init_checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded stage-2 checkpoint. missing={len(missing)} unexpected={len(unexpected)}")

    # freeze raw encoder weights, LoRA adapters will be the only trainable
    # part of the encoder; decoder remains fully trainable
    for p in model.encoder.parameters():
        p.requires_grad = False

    model = attach_lora(model, r=args.lora_r, alpha=args.lora_alpha)
    model.to(device)

    train_ds = SLUDataset(args.train_file, args.whisper_model_name)
    val_ds = SLUDataset(args.val_file, args.whisper_model_name)
    collate = make_collate_fn(pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=args.label_smoothing)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"[LoRA] epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            # save full model (LoRA adapters merged into state dict) + decoder
            torch.save(model.state_dict(), f"{args.out_dir}/best_model_lora.pt")
            # also save just the small LoRA adapter weights for portability
            model.encoder.save_pretrained(f"{args.out_dir}/lora_encoder_adapter")
            print(f"  saved new best LoRA checkpoint (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    main()
