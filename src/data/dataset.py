"""
src/data/dataset.py
Stage 2 & 3 — Tokenizer, BIO tagger, SlotDataset, DataLoader

What this file does end-to-end:
  utterances.json
    → Tokenizer   : builds vocab, converts text → token ID sequences
    → BIOTagger   : converts slot dicts → per-token BIO label sequences
    → SlotDataset : torch Dataset that pairs (token_ids, bio_label_ids)
    → collate_fn  : pads sequences to the longest in a batch
    → get_loaders : returns train / val DataLoaders ready for the model

BIO format recap:
  "I want something spicy and warm"
   O  O     O         B-craving O  B-temp_preference
  B = Beginning of a slot span
  I = Inside a slot span (for multi-word values)
  O = Outside any slot
"""

import json
import re
import random
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


# ── 1. TOKENIZER ─────────────────────────────────────────────────────────────

class Tokenizer:
    """
    Character-free, word-level tokenizer built from scratch.

    Special tokens:
      <PAD>  id=0  padding (ignored by attention mask)
      <UNK>  id=1  out-of-vocabulary words
      <CLS>  id=2  prepended to every sequence (the "sentence vector" token)
    """

    PAD, PAD_ID = "<PAD>", 0
    UNK, UNK_ID = "<UNK>", 1
    CLS, CLS_ID = "<CLS>", 2
    SPECIAL = [PAD, UNK, CLS]

    def __init__(self, min_freq: int = 2):
        """
        min_freq: words appearing fewer than this many times are mapped to <UNK>.
                  Keeps vocabulary tight and forces the model to generalise.
        """
        self.min_freq = min_freq
        self.word2id: dict[str, int] = {}
        self.id2word: dict[int, str] = {}
        self._built = False

    # ── text cleaning (same logic as generation script) ──────────────────────

    @staticmethod
    def clean(text: str) -> str:
        """Lowercase and keep only letters, digits, apostrophes, spaces."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9' ]", " ", text)
        return " ".join(text.split())   # collapse whitespace

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Split cleaned text into word tokens."""
        return Tokenizer.clean(text).split()

    # ── vocabulary building ───────────────────────────────────────────────────

    def build_vocab(self, texts: list[str]) -> None:
        """
        Count word frequencies across all texts, keep those >= min_freq,
        assign integer IDs starting after the special tokens.
        """
        counts: Counter = Counter()
        for text in texts:
            counts.update(self.tokenize(text))

        # Start IDs after specials (0, 1, 2 are reserved)
        self.word2id = {tok: idx for idx, tok in enumerate(self.SPECIAL)}
        next_id = len(self.SPECIAL)
        for word, freq in counts.most_common():     # most common first
            if freq >= self.min_freq:
                self.word2id[word] = next_id
                next_id += 1

        self.id2word = {v: k for k, v in self.word2id.items()}
        self._built = True
        print(f"[Tokenizer] vocab size: {len(self.word2id):,}  "
              f"(min_freq={self.min_freq}, {len(counts):,} unique words seen)")

    # ── encoding ─────────────────────────────────────────────────────────────

    def encode(self, text: str, add_cls: bool = True) -> list[int]:
        """
        text → list of integer IDs.
        Prepends CLS token by default so position 0 is always <CLS>.
        Unknown words get UNK_ID.
        """
        if not self._built:
            raise RuntimeError("Call build_vocab() before encode()")
        ids = [self.word2id.get(w, self.UNK_ID) for w in self.tokenize(text)]
        if add_cls:
            ids = [self.CLS_ID] + ids
        return ids

    def decode(self, ids: list[int]) -> str:
        """Integer IDs → space-joined words."""
        return " ".join(self.id2word.get(i, self.UNK) for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.word2id)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"min_freq": self.min_freq, "word2id": self.word2id}, f)
        print(f"[Tokenizer] saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        with open(path) as f:
            obj = json.load(f)
        tok = cls(min_freq=obj["min_freq"])
        tok.word2id = {k: int(v) for k, v in obj["word2id"].items()}
        tok.id2word = {v: k for k, v in tok.word2id.items()}
        tok._built = True
        print(f"[Tokenizer] loaded from {path}  (vocab={tok.vocab_size:,})")
        return tok


# ── 2. BIO TAGGER ─────────────────────────────────────────────────────────────

class BIOTagger:
    """
    Converts slot dicts into per-token BIO label sequences.

    Label set (built dynamically from slot names):
      O          → outside any slot
      B-<slot>   → beginning of a slot value span
      I-<slot>   → continuation of a slot value span

    Example:
      text  : "I want something spicy and warm"
      slots : {"craving": "spicy", "temp_preference": "warm"}
      tokens: ["<CLS>", "i", "want", "something", "spicy", "and", "warm"]
      labels: ["O",     "O", "O",    "O",          "B-craving", "O", "B-temp_preference"]
    """

    O = "O"

    def __init__(self, slot_names: list[str]):
        self.slot_names = slot_names

        # Build label vocabulary
        labels = [self.O]
        for slot in slot_names:
            labels.append(f"B-{slot}")
            labels.append(f"I-{slot}")

        self.label2id: dict[str, int] = {l: i for i, l in enumerate(labels)}
        self.id2label: dict[int, str] = {i: l for l, i in self.label2id.items()}
        self.O_ID = self.label2id[self.O]

        print(f"[BIOTagger] {len(self.label2id)} labels: {labels}")

    @property
    def num_labels(self) -> int:
        return len(self.label2id)

    # ── core tagging logic ────────────────────────────────────────────────────

    def tag(self, tokens: list[str], slots: dict[str, str]) -> list[int]:
        """
        Given a token list (already cleaned+split, WITH <CLS> prepended)
        and a slots dict, return a list of label IDs of the same length.

        Strategy: for each slot value, find where its tokens appear
        in the sequence and assign B / I tags.
        CLS token always gets O.
        """
        n = len(tokens)
        label_ids = [self.O_ID] * n     # default everything to O

        for slot_name, slot_value in slots.items():
            if slot_name not in [s for s in self.slot_names]:
                continue

            # Tokenize the slot value the same way the text was tokenized
            value_tokens = Tokenizer.clean(slot_value).split()
            if not value_tokens:
                continue

            vlen = len(value_tokens)
            b_id = self.label2id.get(f"B-{slot_name}", self.O_ID)
            i_id = self.label2id.get(f"I-{slot_name}", self.O_ID)

            # Scan tokens for the value span (skip position 0 = <CLS>)
            for start in range(1, n - vlen + 1):
                if tokens[start: start + vlen] == value_tokens:
                    label_ids[start] = b_id
                    for k in range(1, vlen):
                        label_ids[start + k] = i_id
                    break   # tag first occurrence only

        return label_ids

    def decode_labels(self, label_ids: list[int]) -> dict[str, str]:
        """
        Convert a list of label IDs back into a slot dict.
        Reconstructs multi-token values by joining I- continuations.
        Useful for inference.
        """
        slots: dict[str, list[str]] = {}
        current_slot: Optional[str] = None

        for lid in label_ids:
            label = self.id2label.get(lid, self.O)
            if label.startswith("B-"):
                current_slot = label[2:]
                slots[current_slot] = []
            elif label.startswith("I-") and current_slot == label[2:]:
                pass    # value token will be added from token sequence
            else:
                current_slot = None

        return slots   # caller pairs with token strings for final values

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"slot_names": self.slot_names, "label2id": self.label2id}, f)
        print(f"[BIOTagger] saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BIOTagger":
        with open(path) as f:
            obj = json.load(f)
        tagger = cls(slot_names=obj["slot_names"])
        tagger.label2id = {k: int(v) for k, v in obj["label2id"].items()}
        tagger.id2label = {v: k for k, v in tagger.label2id.items()}
        return tagger


# ── 3. DATASET ────────────────────────────────────────────────────────────────

class SlotDataset(Dataset):
    """
    PyTorch Dataset that wraps encoded examples.

    Each item is a dict:
      token_ids    : LongTensor [seq_len]   — input token IDs
      label_ids    : LongTensor [seq_len]   — BIO label IDs per token
      attention_mask: LongTensor [seq_len]  — 1 for real tokens, 0 for PAD
    """

    def __init__(
        self,
        examples: list[dict],          # raw {"text": ..., "slots": ...}
        tokenizer: Tokenizer,
        tagger: BIOTagger,
        max_len: int = 64,
    ):
        self.tokenizer = tokenizer
        self.tagger    = tagger
        self.max_len   = max_len
        self.data      = self._process(examples)

    def _process(self, examples: list[dict]) -> list[dict]:
        processed = []
        skipped   = 0
        for ex in examples:
            text  = ex["text"]
            slots = ex["slots"]

            # Tokenize: clean text, split, prepend CLS
            tokens = ["<CLS>"] + Tokenizer.tokenize(text)

            # Truncate to max_len
            tokens = tokens[:self.max_len]

            # Encode tokens → IDs
            token_ids = [self.tokenizer.CLS_ID] + [
                self.tokenizer.word2id.get(t, self.tokenizer.UNK_ID)
                for t in tokens[1:]     # skip the <CLS> string, already handled
            ]

            # BIO tag
            label_ids = self.tagger.tag(tokens, slots)

            # Sanity check
            if len(token_ids) != len(label_ids):
                skipped += 1
                continue

            processed.append({
                "token_ids": token_ids,
                "label_ids": label_ids,
                "length":    len(token_ids),
            })

        print(f"[SlotDataset] {len(processed):,} examples  "
              f"({skipped} skipped due to length mismatch)")
        return processed

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        return {
            "token_ids":      torch.tensor(item["token_ids"], dtype=torch.long),
            "label_ids":      torch.tensor(item["label_ids"], dtype=torch.long),
            "attention_mask": torch.ones(item["length"],       dtype=torch.long),
        }


# ── 4. COLLATE (padding) ──────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Pads all sequences in a batch to the length of the longest one.
    token_ids    padded with PAD_ID (0)
    label_ids    padded with O_ID  (0) — O is safe to ignore in loss
    attention_mask padded with 0
    """
    max_len = max(item["token_ids"].size(0) for item in batch)

    token_ids_b      = []
    label_ids_b      = []
    attention_mask_b = []

    for item in batch:
        seq_len = item["token_ids"].size(0)
        pad_len = max_len - seq_len

        token_ids_b.append(
            torch.cat([item["token_ids"],
                       torch.zeros(pad_len, dtype=torch.long)])
        )
        label_ids_b.append(
            torch.cat([item["label_ids"],
                       torch.zeros(pad_len, dtype=torch.long)])
        )
        attention_mask_b.append(
            torch.cat([item["attention_mask"],
                       torch.zeros(pad_len, dtype=torch.long)])
        )

    return {
        "token_ids":      torch.stack(token_ids_b),       # [B, max_len]
        "label_ids":      torch.stack(label_ids_b),       # [B, max_len]
        "attention_mask": torch.stack(attention_mask_b),  # [B, max_len]
    }


# ── 5. GET LOADERS ────────────────────────────────────────────────────────────

SLOT_NAMES = ["craving", "temp_preference", "mood", "cuisine", "portion_size", "dietary"]

def get_loaders(
    data_path:  str | Path = "data/raw/utterances.json",
    save_dir:   str | Path = "data/processed",
    batch_size: int        = 32,
    val_split:  float      = 0.1,
    max_len:    int        = 64,
    seed:       int        = 42,
) -> tuple[DataLoader, DataLoader, Tokenizer, BIOTagger]:
    """
    Full pipeline: load JSON → build vocab → BIO tag → split → DataLoaders.

    Returns: train_loader, val_loader, tokenizer, tagger
    """
    random.seed(seed)
    save_dir = Path(save_dir)

    # ── load raw data ─────────────────────────────────────────────────────────
    with open(data_path) as f:
        examples = json.load(f)
    print(f"[get_loaders] loaded {len(examples):,} examples from {data_path}")

    # ── shuffle before split ─────────────────────────────────────────────────
    random.shuffle(examples)

    # ── build tokenizer ───────────────────────────────────────────────────────
    tokenizer = Tokenizer(min_freq=2)
    tokenizer.build_vocab([ex["text"] for ex in examples])
    tokenizer.save(save_dir / "tokenizer.json")

    # ── build BIO tagger ──────────────────────────────────────────────────────
    tagger = BIOTagger(slot_names=SLOT_NAMES)
    tagger.save(save_dir / "tagger.json")

    # ── train / val split ────────────────────────────────────────────────────
    n_val   = int(len(examples) * val_split)
    val_ex  = examples[:n_val]
    train_ex = examples[n_val:]
    print(f"[get_loaders] train={len(train_ex):,}  val={len(val_ex):,}")

    # ── datasets ─────────────────────────────────────────────────────────────
    train_ds = SlotDataset(train_ex, tokenizer, tagger, max_len=max_len)
    val_ds   = SlotDataset(val_ex,   tokenizer, tagger, max_len=max_len)

    # ── loaders ───────────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,      # 0 = main process (safe on all platforms incl M4)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print(f"[get_loaders] train batches={len(train_loader)}  "
          f"val batches={len(val_loader)}  batch_size={batch_size}")

    return train_loader, val_loader, tokenizer, tagger


# ── 6. SANITY CHECK ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/utterances.json"

    train_loader, val_loader, tokenizer, tagger = get_loaders(
        data_path=data_path,
        save_dir="data/processed",
        batch_size=32,
    )

    # Inspect one batch
    batch = next(iter(train_loader))
    print("\n── Batch shapes ──────────────────────────────")
    for k, v in batch.items():
        print(f"  {k:<20} {tuple(v.shape)}")

    # Decode a single example so we can visually verify BIO tags
    print("\n── Example decode ────────────────────────────")
    idx = 0
    ids    = batch["token_ids"][idx].tolist()
    labels = batch["label_ids"][idx].tolist()
    mask   = batch["attention_mask"][idx].tolist()

    print(f"  {'token':<20} {'label':<25} mask")
    print(f"  {'-'*20} {'-'*25} ----")
    for tid, lid, m in zip(ids, labels, mask):
        if tid == 0 and m == 0:
            break   # stop at padding
        word  = tokenizer.id2word.get(tid, "<UNK>")
        label = tagger.id2label.get(lid, "O")
        print(f"  {word:<20} {label:<25} {m}")

    print("\n✓ Stage 2 & 3 complete — dataset pipeline working")
