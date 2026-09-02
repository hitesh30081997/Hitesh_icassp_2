"""
PyTorch Dataset for SLURP end-to-end SLU.

Expected layout (matches the official SLURP repo: github.com/pswietojanski/slurp):

    slurp/
      dataset/slurp/{train,train_synthetic,devel,test}.jsonl
      audio/slurp_real/*.wav
      audio/slurp_synth/*.wav

Each jsonl line has "recordings": a list of dicts with a "file" field; SLURP
utterances are often recorded multiple times by different speakers, so one
annotation can map to several audio files (we expand one dataset item per
recording by default).
"""

import json
import os
from typing import List, Optional

import torch
import torchaudio
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from serialization import serialize_target

SAMPLE_RATE = 16000


class SlurpSLUDataset(Dataset):
    def __init__(
        self,
        jsonl_paths: List[str],
        audio_dirs: List[str],
        tokenizer_path: str,
        max_audio_sec: float = 20.0,
        max_target_tokens: int = 128,
    ):
        """
        jsonl_paths: e.g. ["slurp/dataset/slurp/train.jsonl"]
        audio_dirs: directories to search for each recording's wav file,
                    e.g. ["slurp/audio/slurp_real", "slurp/audio/slurp_synth"]
        """
        self.audio_dirs = audio_dirs
        self.max_audio_samples = int(max_audio_sec * SAMPLE_RATE)
        self.max_target_tokens = max_target_tokens

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.sos_id = self.tokenizer.token_to_id("<sos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")

        self.items = []  # list of (wav_path, target_string)
        for jp in jsonl_paths:
            with open(jp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    target = serialize_target(rec["scenario"], rec["action"], rec.get("entities", []))
                    for recording in rec.get("recordings", []):
                        wav_path = self._resolve_audio(recording["file"])
                        if wav_path is not None:
                            self.items.append((wav_path, target))

        if not self.items:
            raise RuntimeError(
                "No (audio, target) pairs found. Check jsonl_paths / audio_dirs."
            )

    def _resolve_audio(self, filename: str) -> Optional[str]:
        for d in self.audio_dirs:
            candidate = os.path.join(d, filename)
            if os.path.exists(candidate):
                return candidate
        return None

    def __len__(self):
        return len(self.items)

    def _load_wav(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:  # downmix to mono
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.squeeze(0)
        if wav.numel() > self.max_audio_samples:
            wav = wav[: self.max_audio_samples]
        return wav

    def __getitem__(self, idx):
        wav_path, target_str = self.items[idx]
        wav = self._load_wav(wav_path)

        ids = self.tokenizer.encode(target_str).ids
        ids = ids[: self.max_target_tokens - 2]
        ids = [self.sos_id] + ids + [self.eos_id]

        return {
            "audio": wav,
            "audio_len": wav.numel(),
            "target_ids": torch.tensor(ids, dtype=torch.long),
            "target_str": target_str,
        }


def collate_fn(batch, pad_id: int):
    audio_lens = torch.tensor([b["audio_len"] for b in batch], dtype=torch.long)
    max_audio = audio_lens.max().item()
    audio = torch.zeros(len(batch), max_audio)
    for i, b in enumerate(batch):
        audio[i, : b["audio_len"]] = b["audio"]

    tgt_lens = torch.tensor([b["target_ids"].numel() for b in batch], dtype=torch.long)
    max_tgt = tgt_lens.max().item()
    target = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        target[i, : b["target_ids"].numel()] = b["target_ids"]

    # decoder input = target[:, :-1], loss target = target[:, 1:]
    tgt_in = target[:, :-1]
    tgt_out = target[:, 1:]

    return {
        "audio": audio,
        "audio_lens": audio_lens,
        "tgt_in": tgt_in,
        "tgt_out": tgt_out,
        "target_strs": [b["target_str"] for b in batch],
    }
