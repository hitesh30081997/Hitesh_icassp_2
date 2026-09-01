"""
model.py

WhisperEncoder (frozen or LoRA-finetuned) + a custom Transformer decoder
that generates the flat structural sequence described in preprocess.py, e.g.:

  <s> <intent> [intent:alarm_set] <slots>
     <slot_type> [slot:time] <slot_value> n ##ine am
     <slot_type> [slot:date] <slot_value> friday
  </s>

which is deterministically converted back into:

  {"intent": "alarm_set", "slots": {"time": "nine am", "date": "friday"}}

Combined vocabulary layout:
  ids [0, N_closed)                -> structural / intent / slot-type tokens
  ids [N_closed, N_closed+N_bpe)   -> Whisper BPE subword tokens (slot values)
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperTokenizer


# --------------------------------------------------------------------------
# Vocab helpers
# --------------------------------------------------------------------------

def build_full_vocab(closed_vocab_path, whisper_model_name="openai/whisper-small"):
    closed_vocab = json.loads(Path(closed_vocab_path).read_text())
    bpe_tokenizer = WhisperTokenizer.from_pretrained(whisper_model_name)

    n_closed = len(closed_vocab["token2id"])
    n_bpe = bpe_tokenizer.vocab_size
    total = n_closed + n_bpe

    info = {
        "closed_vocab": closed_vocab,
        "n_closed": n_closed,
        "n_bpe": n_bpe,
        "total_vocab_size": total,
        "bpe_tokenizer": bpe_tokenizer,
    }
    return info


# --------------------------------------------------------------------------
# Decoder
# --------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SLUDecoder(nn.Module):
    """Standard Transformer decoder with cross-attention into Whisper encoder states."""

    def __init__(self, vocab_size, d_model=768, n_heads=8, n_layers=6,
                 d_ff=3072, dropout=0.1, max_len=256, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_ids, encoder_states, encoder_padding_mask=None):
        """
        tgt_ids: (B, T) target token ids, teacher-forced (shifted right by caller)
        encoder_states: (B, S, d_model) Whisper encoder output
        encoder_padding_mask: (B, S) bool, True = pad
        """
        B, T = tgt_ids.shape
        x = self.tok_emb(tgt_ids) * math.sqrt(self.tok_emb.embedding_dim)
        x = self.pos_enc(x)
        x = self.dropout(x)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        tgt_key_padding_mask = tgt_ids.eq(self.pad_id)

        h = self.decoder(
            tgt=x,
            memory=encoder_states,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=encoder_padding_mask,
        )
        logits = self.out_proj(h)  # (B, T, vocab_size)
        return logits


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------

class WhisperSLU(nn.Module):
    def __init__(self, vocab_info, whisper_model_name="openai/whisper-small",
                 freeze_encoder=True, decoder_layers=6, decoder_heads=8):
        super().__init__()
        whisper = WhisperModel.from_pretrained(whisper_model_name)
        self.encoder = whisper.encoder
        d_model = self.encoder.config.d_model  # 768 for whisper-small

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.vocab_info = vocab_info
        self.pad_id = vocab_info["closed_vocab"]["token2id"]["<pad>"]

        self.decoder = SLUDecoder(
            vocab_size=vocab_info["total_vocab_size"],
            d_model=d_model,
            n_heads=decoder_heads,
            n_layers=decoder_layers,
            pad_id=self.pad_id,
        )

    def forward(self, input_features, decoder_input_ids):
        """
        input_features: (B, 80, 3000) log-mel from WhisperFeatureExtractor
        decoder_input_ids: (B, T) right-shifted target ids (teacher forcing)
        """
        enc_out = self.encoder(input_features).last_hidden_state  # (B, 1500, d_model)
        logits = self.decoder(decoder_input_ids, enc_out)
        return logits

    @torch.no_grad()
    def generate(self, input_features, max_len=128, temperature=1.0):
        """
        Greedy / constrained decoding. Enforces grammar at each step:
          <s> -> must emit <intent>
          <intent> -> must emit an [intent:*] token
          [intent:*] -> must emit <slots>
          <slots> -> may emit <slot_type> or </s>
          <slot_type> -> must emit a [slot:*] token
          [slot:*] -> must emit <slot_value>
          <slot_value> -> free-form BPE tokens until next <slot_type> or </s>
                          (model chooses when to stop this span by emitting
                          <slot_type> or </s> directly, since those are the
                          only closed-vocab tokens allowed to follow a value)
        """
        cv = self.vocab_info["closed_vocab"]["token2id"]
        n_closed = self.vocab_info["n_closed"]
        device = input_features.device
        B = input_features.size(0)

        enc_out = self.encoder(input_features).last_hidden_state

        intent_ids = [i for tok, i in cv.items() if tok.startswith("[intent:")]
        slot_ids = [i for tok, i in cv.items() if tok.startswith("[slot:")]
        bpe_range = list(range(n_closed, self.vocab_info["total_vocab_size"]))

        seqs = torch.full((B, 1), cv["<s>"], dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        # simple finite-state grammar per sequence
        states = ["expect_intent_kw"] * B

        for step in range(max_len):
            logits = self.decoder(seqs, enc_out)[:, -1, :] / temperature  # (B, V)
            mask = torch.full_like(logits, float("-inf"))

            for b in range(B):
                if finished[b]:
                    mask[b, cv["<pad>"]] = 0
                    continue
                st = states[b]
                if st == "expect_intent_kw":
                    mask[b, cv["<intent>"]] = 0
                elif st == "expect_intent_val":
                    mask[b, intent_ids] = 0
                elif st == "expect_slots_kw":
                    mask[b, cv["<slots>"]] = 0
                elif st == "in_slots":
                    mask[b, cv["<slot_type>"]] = 0
                    mask[b, cv["</s>"]] = 0
                elif st == "expect_slot_type_val":
                    mask[b, slot_ids] = 0
                elif st == "expect_slot_value_kw":
                    mask[b, cv["<slot_value>"]] = 0
                elif st == "in_slot_value":
                    # free BPE tokens, or close the value / whole sequence
                    mask[b, bpe_range] = 0
                    mask[b, cv["<slot_type>"]] = 0
                    mask[b, cv["</s>"]] = 0

            next_id = (logits + mask).argmax(dim=-1)  # (B,)

            for b in range(B):
                if finished[b]:
                    continue
                tid = next_id[b].item()
                st = states[b]
                if st == "expect_intent_kw":
                    states[b] = "expect_intent_val"
                elif st == "expect_intent_val":
                    states[b] = "expect_slots_kw"
                elif st == "expect_slots_kw":
                    states[b] = "in_slots"
                elif st == "in_slots":
                    if tid == cv["</s>"]:
                        finished[b] = True
                    else:
                        states[b] = "expect_slot_type_val"
                elif st == "expect_slot_type_val":
                    states[b] = "expect_slot_value_kw"
                elif st == "expect_slot_value_kw":
                    states[b] = "in_slot_value"
                elif st == "in_slot_value":
                    if tid == cv["</s>"]:
                        finished[b] = True
                    elif tid == cv["<slot_type>"]:
                        states[b] = "expect_slot_type_val"
                    # else stay in_slot_value

            seqs = torch.cat([seqs, next_id.unsqueeze(1)], dim=1)
            if finished.all():
                break

        return seqs


# --------------------------------------------------------------------------
# Decode token ids -> JSON
# --------------------------------------------------------------------------

def ids_to_json(ids, vocab_info):
    """Deterministically reconstruct the JSON string from a generated id sequence."""
    cv = vocab_info["closed_vocab"]["token2id"]
    id2tok = {v: k for k, v in cv.items()}
    n_closed = vocab_info["n_closed"]
    bpe_tok = vocab_info["bpe_tokenizer"]

    intent = None
    slots = {}
    cur_slot_type = None
    cur_value_ids = []

    def flush_value():
        nonlocal cur_slot_type, cur_value_ids
        if cur_slot_type is not None:
            text = bpe_tok.decode(cur_value_ids).strip() if cur_value_ids else ""
            slots[cur_slot_type] = text
        cur_slot_type = None
        cur_value_ids = []

    for tid in ids:
        tid = int(tid)
        if tid < n_closed:
            tok = id2tok.get(tid, "<unk>")
            if tok.startswith("[intent:"):
                intent = tok[len("[intent:"):-1]
            elif tok.startswith("[slot:"):
                flush_value()
                cur_slot_type = tok[len("[slot:"):-1]
            elif tok == "</s>":
                flush_value()
                break
            # <s>, <intent>, <slots>, <slot_type>, <slot_value> are structural, skip
        else:
            cur_value_ids.append(tid - n_closed)

    flush_value()
    return json.dumps({"intent": intent, "slots": slots}, ensure_ascii=False)


if __name__ == "__main__":
    # smoke test with random weights / dummy input, no real SLURP data needed
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="vocab.json")
    args = ap.parse_args()

    if not Path(args.vocab).exists():
        print(f"{args.vocab} not found - run build_vocab.py first. Skipping smoke test.")
    else:
        vocab_info = build_full_vocab(args.vocab)
        model = WhisperSLU(vocab_info)
        dummy_feats = torch.randn(1, 80, 3000)
        out_ids = model.generate(dummy_feats, max_len=32)
        print("Generated ids:", out_ids)
        print("JSON:", ids_to_json(out_ids[0].tolist(), vocab_info))
