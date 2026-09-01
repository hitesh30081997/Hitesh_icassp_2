"""
build_vocab.py

Scans SLURP train.jsonl to build:
  - intent vocabulary   (scenario_action, e.g. "alarm_set")
  - slot-type vocabulary (entity types, e.g. "date", "person")

SLURP annotation format (per line), roughly:
{
  "slurp_id": 1234,
  "sentence": "wake me up at nine am on friday",
  "scenario": "alarm",
  "action": "set",
  "intent": "alarm_set",
  "entities": [
      {"type": "time", "filler": "nine am"},
      {"type": "date", "filler": "friday"}
  ],
  "recordings": [{"file": "audio-1234.flac"}, ...]
}

Output: vocab.json containing all special tokens + intents + slot types,
mapped to integer ids. This vocab is shared by preprocess.py and model.py.
"""

import json
import argparse
from collections import Counter
from pathlib import Path


SPECIAL_TOKENS = [
    "<pad>", "<s>", "</s>", "<unk>",
    "<intent>", "<slots>", "<slot_type>", "<slot_value>",
]


def scan_slurp_file(path):
    intents = Counter()
    slot_types = Counter()
    n_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            n_lines += 1

            intent = ex.get("intent")
            if intent is None:
                scenario = ex.get("scenario", "unknown")
                action = ex.get("action", "unknown")
                intent = f"{scenario}_{action}"
            intents[intent] += 1

            for ent in ex.get("entities", []):
                slot_types[ent["type"]] += 1

    return intents, slot_types, n_lines


def build_vocab(train_files, out_path, bpe_placeholder=True):
    all_intents = Counter()
    all_slots = Counter()
    total = 0

    for tf in train_files:
        intents, slots, n = scan_slurp_file(tf)
        all_intents.update(intents)
        all_slots.update(slots)
        total += n
        print(f"[{tf}] {n} examples, {len(intents)} intents, {len(slots)} slot types")

    intent_list = sorted(all_intents.keys())
    slot_list = sorted(all_slots.keys())

    tokens = list(SPECIAL_TOKENS)
    tokens += [f"[intent:{i}]" for i in intent_list]
    tokens += [f"[slot:{s}]" for s in slot_list]

    token2id = {tok: idx for idx, tok in enumerate(tokens)}

    vocab = {
        "special_tokens": SPECIAL_TOKENS,
        "intents": intent_list,
        "slot_types": slot_list,
        "token2id": token2id,
        "note": (
            "Slot VALUES are not enumerated here - they are open vocabulary "
            "and should be tokenized with a subword tokenizer (BPE/WordPiece), "
            "e.g. reuse Whisper's own tokenizer or train a small one on SLURP "
            "sentence_annotation fields. This file only covers the closed "
            "vocabulary: structural tokens + intents + slot types."
        ),
    }

    Path(out_path).write_text(json.dumps(vocab, indent=2))
    print(f"\nTotal examples scanned: {total}")
    print(f"Closed vocab size (structural+intent+slot_type): {len(tokens)}")
    print(f"Wrote vocab to {out_path}")
    return vocab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_files", nargs="+", required=True,
        help="One or more SLURP jsonl files (e.g. train.jsonl train_synthetic.jsonl)",
    )
    ap.add_argument("--out", default="vocab.json")
    args = ap.parse_args()

    build_vocab(args.train_files, args.out)
