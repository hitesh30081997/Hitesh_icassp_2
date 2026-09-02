"""
FastConformer-based End-to-End Spoken Language Understanding (SLU) model.

Architecture
------------
    audio waveform
        -> [PRETRAINED NeMo FastConformer encoder]   (frozen or lightly fine-tuned)
        -> encoder states (B, T, D)
        -> [UNTRAINED Transformer decoder]            (trained from scratch)
        -> autoregressively generated token sequence encoding "intent + slots"

The decoder is trained with teacher forcing on serialized target strings such as:

    <sos> intent : alarm_set | date = tomorrow ; time = six am <eos>

See serialization.py for the exact string format used for SLURP.

Requires:
    pip install "nemo_toolkit[asr]"
"""

import math
import torch
import torch.nn as nn

import nemo.collections.asr as nemo_asr


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class FastConformerEncoderWrapper(nn.Module):
    """
    Loads a pretrained NeMo FastConformer ASR model and exposes only the
    preprocessor + encoder stack (discarding the ASR decoder/joint head).
    """

    def __init__(
        self,
        pretrained_model_name: str = "stt_en_fastconformer_transducer_large",
        freeze: bool = True,
        unfreeze_last_n_layers: int = 0,
    ):
        super().__init__()
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=pretrained_model_name)

        self.preprocessor = asr_model.preprocessor  # waveform -> log-mel features
        self.encoder = asr_model.encoder             # FastConformer conv+attn stack
        self.enc_out_dim = asr_model.cfg.encoder.d_model

        del asr_model  # drop decoder/joint, we only needed the encoder path

        if freeze:
            for p in self.preprocessor.parameters():
                p.requires_grad = False
            for p in self.encoder.parameters():
                p.requires_grad = False
            if unfreeze_last_n_layers > 0:
                for layer in self.encoder.layers[-unfreeze_last_n_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True

    def forward(self, audio_signal: torch.Tensor, audio_lengths: torch.Tensor):
        """
        audio_signal: (B, num_samples) raw 16kHz waveform
        audio_lengths: (B,) number of valid samples per utterance
        returns: enc_out (B, T, D), enc_lengths (B,)
        """
        feats, feat_lengths = self.preprocessor(input_signal=audio_signal, length=audio_lengths)
        enc_out, enc_lengths = self.encoder(audio_signal=feats, length=feat_lengths)
        enc_out = enc_out.transpose(1, 2)  # NeMo returns (B, D, T) -> (B, T, D)
        return enc_out, enc_lengths


class TransformerSLUDecoder(nn.Module):
    """Randomly initialized autoregressive Transformer decoder over sub-word tokens."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pad_id: int = 0,
        max_len: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _causal_mask(sz: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(self, tgt_ids: torch.Tensor, memory: torch.Tensor, memory_key_padding_mask=None):
        """
        tgt_ids: (B, L) right-shifted target ids (teacher forcing input)
        memory:  (B, T, D) encoder states
        """
        device = tgt_ids.device
        tgt_pad_mask = tgt_ids.eq(self.pad_id)

        x = self.tok_emb(tgt_ids) * math.sqrt(self.d_model)
        x = self.dropout(self.pos_enc(x))

        h = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self._causal_mask(tgt_ids.size(1), device),
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out_proj(h)

    @torch.no_grad()
    def greedy_decode(self, memory, memory_key_padding_mask, sos_id, eos_id, max_len=128):
        device = memory.device
        B = memory.size(0)
        ys = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(ys, memory, memory_key_padding_mask)
            next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            finished |= next_tok.squeeze(1).eq(eos_id)
            if finished.all():
                break
        return ys

    @torch.no_grad()
    def beam_search_decode(self, memory, memory_key_padding_mask, sos_id, eos_id,
                            beam_size=4, max_len=128, length_penalty=0.6):
        """Simple batch-size-1 beam search (call per-utterance)."""
        assert memory.size(0) == 1, "beam_search_decode expects a single utterance at a time"
        device = memory.device
        beams = [(torch.tensor([[sos_id]], device=device), 0.0, False)]

        for _ in range(max_len - 1):
            candidates = []
            for seq, score, done in beams:
                if done:
                    candidates.append((seq, score, True))
                    continue
                mem = memory.expand(seq.size(0), -1, -1)
                mask = memory_key_padding_mask.expand(seq.size(0), -1)
                logits = self.forward(seq, mem, mask)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
                topk_lp, topk_idx = log_probs.topk(beam_size)
                for lp, idx in zip(topk_lp.tolist(), topk_idx.tolist()):
                    new_seq = torch.cat([seq, torch.tensor([[idx]], device=device)], dim=1)
                    candidates.append((new_seq, score + lp, idx == eos_id))
            lp_norm = lambda c: c[1] / (c[0].size(1) ** length_penalty)
            beams = sorted(candidates, key=lp_norm, reverse=True)[:beam_size]
            if all(done for _, _, done in beams):
                break

        best_seq = sorted(beams, key=lp_norm, reverse=True)[0][0]
        return best_seq


class FastConformerSLU(nn.Module):
    """Full pretrained-encoder / untrained-decoder E2E SLU model."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        pretrained_encoder_name: str = "stt_en_fastconformer_transducer_large",
        freeze_encoder: bool = True,
        unfreeze_last_n_layers: int = 0,
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()
        self.pad_id, self.sos_id, self.eos_id = pad_id, sos_id, eos_id

        self.encoder = FastConformerEncoderWrapper(
            pretrained_encoder_name,
            freeze=freeze_encoder,
            unfreeze_last_n_layers=unfreeze_last_n_layers,
        )
        enc_dim = self.encoder.enc_out_dim
        self.enc_proj = nn.Linear(enc_dim, d_model) if enc_dim != d_model else nn.Identity()

        self.decoder = TransformerSLUDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pad_id=pad_id,
            max_len=max_len,
        )

    @staticmethod
    def _padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return idx >= lengths.unsqueeze(1)  # True at padded positions

    def encode(self, audio_signal, audio_lengths):
        enc_out, enc_lengths = self.encoder(audio_signal, audio_lengths)
        enc_out = self.enc_proj(enc_out)
        mem_pad_mask = self._padding_mask(enc_lengths, enc_out.size(1))
        return enc_out, mem_pad_mask

    def forward(self, audio_signal, audio_lengths, tgt_in_ids):
        """Teacher-forced training forward pass. Returns logits (B, L, V)."""
        enc_out, mem_pad_mask = self.encode(audio_signal, audio_lengths)
        return self.decoder(tgt_in_ids, enc_out, memory_key_padding_mask=mem_pad_mask)

    @torch.no_grad()
    def generate(self, audio_signal, audio_lengths, max_len=128, beam_size=1):
        enc_out, mem_pad_mask = self.encode(audio_signal, audio_lengths)
        if beam_size <= 1:
            return self.decoder.greedy_decode(
                enc_out, mem_pad_mask, self.sos_id, self.eos_id, max_len=max_len
            )
        # beam search: loop per-utterance
        outs = []
        for i in range(enc_out.size(0)):
            seq = self.decoder.beam_search_decode(
                enc_out[i : i + 1], mem_pad_mask[i : i + 1],
                self.sos_id, self.eos_id, beam_size=beam_size, max_len=max_len,
            )
            outs.append(seq.squeeze(0))
        max_l = max(o.size(0) for o in outs)
        padded = torch.full((len(outs), max_l), self.pad_id, dtype=torch.long, device=enc_out.device)
        for i, o in enumerate(outs):
            padded[i, : o.size(0)] = o
        return padded
