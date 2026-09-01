"""
preprocess.py

Converts SLURP (audio, annotation) pairs into training examples of the form:

  input:  log-mel spectrogram (80, 3000) via WhisperFeatureExtractor
  target: token id sequence, e.g.

  <s> <intent> [intent:alarm_set] <slots>
     <slot_type> [slot:time] <slot_value> n ##ine am
     <slot_type> [slot:date] <slot_value> friday
  </s>

Slot VALUES are tokenized with Whisper's own BPE tokenizer (open vocabulary),
while everything else (special tokens, intent, slot type) comes from the
closed vocab built by build_vocab.py. The two vocabularies are concatenated
into one final vocabulary at model-build time (see model.py:build_full_vocab).

Usage:
  python preprocess.py \
      --jsonl slurp/train.jsonl \
      --audio_dir slurp/audio/slurp_real \
      --vocab vocab.json \
      --out train_processed.jsonl
"""

import json
import argparse
from pathlib import Path

import torch
import torchaudio
from transformers import WhisperFeatureExtractor, WhisperTokenizer


def load_audio_16k(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0)


def build_target_string(example):
    """Build the flat target string described in the module docstring."""
    intent = example.get("intent")
    if intent is None:
        intent = f"{example.get('scenario', 'unknown')}_{example.get('action', 'unknown')}"

    parts = ["<s>", "<intent>", f"[intent:{intent}]", "<slots>"]
    for ent in example.get("entities", []):
        parts.append("<slot_type>")
        parts.append(f"[slot:{ent['type']}]")
        parts.append("<slot_value>")
        parts.append(ent["filler"])  # will be subword-tokenized separately
    parts.append("</s>")
    return parts  # list of "words"; slot fillers get BPE-split later


def encode_target(parts, closed_token2id, bpe_tokenizer, unk_id):
    """
    parts: list of strings, some are closed-vocab tokens like "[slot:date]",
           others (slot fillers) are natural language phrases to be BPE-split.
    Returns: list[int] token ids in the *combined* vocabulary space, where
             closed-vocab ids come first [0, len(closed)) and BPE ids are
             offset by len(closed_token2id).
    """
    ids = []
    closed_set = set(closed_token2id.keys())
    offset = len(closed_token2id)

    i = 0
    while i < len(parts):
        p = parts[i]
        if p in closed_set:
            ids.append(closed_token2id[p])
            i += 1
        else:
            # this is a slot filler phrase -> subword tokenize
            bpe_ids = bpe_tokenizer.encode(p, add_special_tokens=False)
            ids.extend([b + offset for b in bpe_ids])
            i += 1
    return ids


def process_file(jsonl_path, audio_dir, vocab_path, out_path,
                  whisper_model_name="openai/whisper-small"):
    vocab = json.loads(Path(vocab_path).read_text())
    closed_token2id = vocab["token2id"]

    feat_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)
    bpe_tokenizer = WhisperTokenizer.from_pretrained(whisper_model_name)
    unk_id = closed_token2id.get("<unk>")

    n_ok, n_skip = 0, 0
    with open(jsonl_path, "r", encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)

            recordings = ex.get("recordings", [])
            if not recordings:
                n_skip += 1
                continue
            audio_file = Path(audio_dir) / recordings[0]["file"]
            if not audio_file.exists():
                n_skip += 1
                continue

            try:
                wav = load_audio_16k(str(audio_file))
            except Exception as e:
                print(f"skip {audio_file}: {e}")
                n_skip += 1
                continue

            feats = feat_extractor(
                wav.numpy(), sampling_rate=16000, return_tensors="np"
            )["input_features"][0]  # (80, 3000)

            target_parts = build_target_string(ex)
            target_ids = encode_target(target_parts, closed_token2id, bpe_tokenizer, unk_id)

            record = {
                "slurp_id": ex.get("slurp_id"),
                "audio_file": str(audio_file),
                "target_ids": target_ids,
                "target_readable": " ".join(target_parts),
            }
            f_out.write(json.dumps(record) + "\n")
            n_ok += 1

    print(f"Done. {n_ok} examples written, {n_skip} skipped -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--whisper_model_name", default="openai/whisper-small")
    args = ap.parse_args()

    process_file(args.jsonl, args.audio_dir, args.vocab, args.out, args.whisper_model_name)
