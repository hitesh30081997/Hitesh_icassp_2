"""
Builds a byte-level BPE tokenizer over the serialized SLURP target strings
(intent + slots), used by the decoder's output vocabulary.

Usage:
    python build_tokenizer.py \
        --slurp_jsonl /path/to/slurp/dataset/slurp/train.jsonl \
                      /path/to/slurp/dataset/slurp/train_synthetic.jsonl \
        --out_dir tokenizer/ \
        --vocab_size 1000
"""

import argparse
import json
import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from serialization import serialize_target

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]  # ids 0,1,2,3 respectively


def iter_target_strings(jsonl_paths):
    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                yield serialize_target(rec["scenario"], rec["action"], rec.get("entities", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slurp_jsonl", nargs="+", required=True,
                     help="One or more SLURP jsonl annotation files (e.g. train.jsonl, "
                          "train_synthetic.jsonl) to build vocabulary from.")
    ap.add_argument("--out_dir", default="tokenizer")
    ap.add_argument("--vocab_size", type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # write target strings to a temp corpus file for the trainer
    corpus_path = os.path.join(args.out_dir, "_target_corpus.txt")
    n = 0
    with open(corpus_path, "w", encoding="utf-8") as out_f:
        for s in iter_target_strings(args.slurp_jsonl):
            out_f.write(s + "\n")
            n += 1
    print(f"Wrote {n} serialized target strings to {corpus_path}")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=args.vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train([corpus_path], trainer)

    save_path = os.path.join(args.out_dir, "slu_tokenizer.json")
    tokenizer.save(save_path)
    print(f"Saved tokenizer ({tokenizer.get_vocab_size()} tokens) to {save_path}")

    os.remove(corpus_path)


if __name__ == "__main__":
    main()
