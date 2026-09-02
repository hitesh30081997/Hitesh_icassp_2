"""
Trains the FastConformer (pretrained, frozen/lightly-finetuned) encoder +
Transformer (untrained) decoder end-to-end on SLURP to autoregressively
generate a serialized "intent + slots" string.

Example:
    python train.py \
        --train_jsonl slurp/dataset/slurp/train.jsonl slurp/dataset/slurp/train_synthetic.jsonl \
        --devel_jsonl slurp/dataset/slurp/devel.jsonl \
        --audio_dirs slurp/audio/slurp_real slurp/audio/slurp_synth \
        --tokenizer_path tokenizer/slu_tokenizer.json \
        --pretrained_encoder stt_en_fastconformer_transducer_large \
        --out_dir checkpoints/ \
        --epochs 30 --batch_size 16 --lr 3e-4
"""

import argparse
import functools
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import FastConformerSLU
from slurp_dataset import SlurpSLUDataset, collate_fn
from tokenizers import Tokenizer


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", nargs="+", required=True)
    ap.add_argument("--devel_jsonl", nargs="+", required=True)
    ap.add_argument("--audio_dirs", nargs="+", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--pretrained_encoder", default="stt_en_fastconformer_transducer_large")
    ap.add_argument("--freeze_encoder", action="store_true", default=True)
    ap.add_argument("--unfreeze_last_n_layers", type=int, default=0)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num_decoder_layers", type=int, default=6)
    ap.add_argument("--dim_feedforward", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_steps", type=int, default=2000)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_audio_sec", type=float, default=20.0)
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", default=True)
    return ap.parse_args()


class WarmupInverseSqrtLR:
    """LR schedule: linear warmup then inverse sqrt decay (Transformer-style)."""

    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5, self.step_num * (self.warmup_steps ** -1.5)
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


def run_epoch(model, loader, optimizer, scheduler, criterion, device, pad_id, train, scaler, grad_accum):
    model.train(mode=train)
    total_loss, total_tokens = 0.0, 0
    torch.set_grad_enabled(train)

    pbar = tqdm(loader, desc="train" if train else "eval")
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(pbar):
        audio = batch["audio"].to(device)
        audio_lens = batch["audio_lens"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)

        with torch.autocast(device_type="cuda", enabled=(scaler is not None and device == "cuda")):
            logits = model(audio, audio_lens, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        n_tokens = tgt_out.ne(pad_id).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        if train:
            loss_to_backward = loss / grad_accum
            if scaler is not None:
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            if (step + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 5.0
                )
                lr = scheduler.step()
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr:.2e}")

    return total_loss / max(total_tokens, 1)


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    tok = Tokenizer.from_file(args.tokenizer_path)
    pad_id = tok.token_to_id("<pad>")
    sos_id = tok.token_to_id("<sos>")
    eos_id = tok.token_to_id("<eos>")
    vocab_size = tok.get_vocab_size()

    train_ds = SlurpSLUDataset(
        args.train_jsonl, args.audio_dirs, args.tokenizer_path, max_audio_sec=args.max_audio_sec
    )
    devel_ds = SlurpSLUDataset(
        args.devel_jsonl, args.audio_dirs, args.tokenizer_path, max_audio_sec=args.max_audio_sec
    )

    collate = functools.partial(collate_fn, pad_id=pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, drop_last=True,
    )
    devel_loader = DataLoader(
        devel_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )

    model = FastConformerSLU(
        vocab_size=vocab_size,
        pad_id=pad_id, sos_id=sos_id, eos_id=eos_id,
        pretrained_encoder_name=args.pretrained_encoder,
        freeze_encoder=args.freeze_encoder,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
        d_model=args.d_model, nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-2)
    scheduler = WarmupInverseSqrtLR(optimizer, args.d_model, args.warmup_steps)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device == "cuda"))

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss = run_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, pad_id,
            train=True, scaler=scaler, grad_accum=args.grad_accum_steps,
        )
        val_loss = run_epoch(
            model, devel_loader, optimizer, scheduler, criterion, device, pad_id,
            train=False, scaler=None, grad_accum=1,
        )
        print(f"epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        ckpt = {
            "model_state": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "val_loss": val_loss,
        }
        torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
            print(f"  -> new best (val_loss={val_loss:.4f}), saved best.pt")


if __name__ == "__main__":
    main()
