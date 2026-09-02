"""
Runs inference with a trained FastConformerSLU checkpoint and (optionally)
computes intent accuracy + slot micro-F1 on a SLURP jsonl split.

Usage (single file):
    python infer.py --ckpt checkpoints/best.pt --tokenizer_path tokenizer/slu_tokenizer.json \
        --wav path/to/utterance.wav

Usage (evaluate a split):
    python infer.py --ckpt checkpoints/best.pt --tokenizer_path tokenizer/slu_tokenizer.json \
        --eval_jsonl slurp/dataset/slurp/test.jsonl --audio_dirs slurp/audio/slurp_real slurp/audio/slurp_synth
"""

import argparse
import functools

import torch
import torchaudio
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from model import FastConformerSLU
from serialization import parse_target
from slurp_dataset import SlurpSLUDataset, collate_fn, SAMPLE_RATE


def load_model(ckpt_path, tokenizer_path, device):
    tok = Tokenizer.from_file(tokenizer_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt["args"]
    model = FastConformerSLU(
        vocab_size=tok.get_vocab_size(),
        pad_id=tok.token_to_id("<pad>"),
        sos_id=tok.token_to_id("<sos>"),
        eos_id=tok.token_to_id("<eos>"),
        pretrained_encoder_name=a["pretrained_encoder"],
        freeze_encoder=a["freeze_encoder"],
        unfreeze_last_n_layers=a["unfreeze_last_n_layers"],
        d_model=a["d_model"], nhead=a["nhead"],
        num_decoder_layers=a["num_decoder_layers"],
        dim_feedforward=a["dim_feedforward"], dropout=a["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tok


@torch.no_grad()
def transcribe_file(model, tok, wav_path, device, beam_size=1, max_len=128):
    wav, sr = torchaudio.load(wav_path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav.to(device)
    length = torch.tensor([wav.size(1)], device=device)

    ids = model.generate(wav, length, max_len=max_len, beam_size=beam_size)[0].tolist()
    eos_id = tok.token_to_id("<eos>")
    if eos_id in ids:
        ids = ids[: ids.index(eos_id) + 1]
    text = tok.decode(ids, skip_special_tokens=True)
    return text, parse_target(text)


def slot_f1(pred_entities, gold_entities):
    """Micro-F1 over (type, filler) tuples, exact match."""
    pred_set = {(e["type"], e["filler"]) for e in pred_entities}
    gold_set = {(e["type"], e["filler"]) for e in gold_entities}
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--wav")
    ap.add_argument("--eval_jsonl", nargs="+")
    ap.add_argument("--audio_dirs", nargs="+")
    ap.add_argument("--beam_size", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, tok = load_model(args.ckpt, args.tokenizer_path, args.device)

    if args.wav:
        text, (intent, entities) = transcribe_file(model, tok, args.wav, args.device, args.beam_size)
        print("raw decoded string:", text)
        print("intent:", intent)
        print("slots:", entities)
        return

    if args.eval_jsonl:
        assert args.audio_dirs, "--audio_dirs required with --eval_jsonl"
        ds = SlurpSLUDataset(args.eval_jsonl, args.audio_dirs, args.tokenizer_path)
        pad_id = tok.token_to_id("<pad>")
        eos_id = tok.token_to_id("<eos>")
        loader = DataLoader(
            ds, batch_size=8, shuffle=False,
            collate_fn=functools.partial(collate_fn, pad_id=pad_id),
        )

        n_correct_intent, n_total = 0, 0
        tp_all, fp_all, fn_all = 0, 0, 0

        for batch in loader:
            audio = batch["audio"].to(args.device)
            audio_lens = batch["audio_lens"].to(args.device)
            gen_ids = model.generate(audio, audio_lens, max_len=128, beam_size=args.beam_size)

            for i in range(gen_ids.size(0)):
                ids = gen_ids[i].tolist()
                if eos_id in ids:
                    ids = ids[: ids.index(eos_id) + 1]
                pred_text = tok.decode(ids, skip_special_tokens=True)
                pred_intent, pred_entities = parse_target(pred_text)
                gold_intent, gold_entities = parse_target(batch["target_strs"][i])

                n_total += 1
                n_correct_intent += int(pred_intent == gold_intent)
                tp, fp, fn = slot_f1(pred_entities, gold_entities)
                tp_all += tp
                fp_all += fp
                fn_all += fn

        intent_acc = n_correct_intent / max(n_total, 1)
        precision = tp_all / max(tp_all + fp_all, 1)
        recall = tp_all / max(tp_all + fn_all, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        print(f"Intent accuracy: {intent_acc:.4f}")
        print(f"Slot precision:  {precision:.4f}")
        print(f"Slot recall:     {recall:.4f}")
        print(f"Slot F1:         {f1:.4f}")
        return

    ap.error("Provide either --wav for a single file or --eval_jsonl/--audio_dirs for evaluation.")


if __name__ == "__main__":
    main()
