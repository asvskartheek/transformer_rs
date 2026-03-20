"""
Dataset: roneneldan/TinyStories — pre-split train / validation.
Trains a BPE tokenizer on a sample of the training set, then tokenises
the full dataset into fixed-length chunks of `seq_len` tokens.
"""
import os
import json
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

DATASET_CACHE = Path(__file__).parent / ".cache"


def build_tokenizer(texts: list[str], vocab_size: int, save_path: Path) -> Tokenizer:
    """Train a BPE tokenizer and save it."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        min_frequency=2,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(str(save_path))
    print(f"Tokenizer saved to {save_path}  (vocab={vocab_size})")
    return tokenizer


def load_or_train_tokenizer(vocab_size: int, train_texts: list[str]) -> Tokenizer:
    path = DATASET_CACHE / f"tokenizer_v{vocab_size}.json"
    if path.exists():
        print(f"Loading cached tokenizer from {path}")
        return Tokenizer.from_file(str(path))
    # Sample up to 50k stories to keep tokenizer training fast
    sample = train_texts[:50_000]
    return build_tokenizer(sample, vocab_size, path)


def tokenise_texts(tokenizer: Tokenizer, texts: list[str], seq_len: int) -> np.ndarray:
    """Encode all texts, concatenate, chunk into (N, seq_len+1) windows."""
    bos = tokenizer.token_to_id("[BOS]")
    eos = tokenizer.token_to_id("[EOS]")

    all_ids: list[int] = []
    for enc in tokenizer.encode_batch(texts, add_special_tokens=False):
        all_ids.append(bos)
        all_ids.extend(enc.ids)
        all_ids.append(eos)

    arr = np.array(all_ids, dtype=np.uint16)
    # Drop the tail that doesn't fill a complete window
    n_chunks = len(arr) // (seq_len + 1)
    arr = arr[: n_chunks * (seq_len + 1)]
    return arr.reshape(n_chunks, seq_len + 1)   # each row: input(seq_len) + 1 target token


def load_dataset(vocab_size: int, seq_len: int, max_train: int = 200_000):
    """
    Returns:
        tokenizer  — trained BPE tokenizer
        train_data — np.ndarray [N_train, seq_len+1] uint16
        eval_data  — np.ndarray [N_eval,  seq_len+1] uint16
    """
    from datasets import load_dataset as hf_load

    cache_train = DATASET_CACHE / f"train_v{vocab_size}_s{seq_len}.npy"
    cache_eval  = DATASET_CACHE / f"eval_v{vocab_size}_s{seq_len}.npy"
    tok_path    = DATASET_CACHE / f"tokenizer_v{vocab_size}.json"

    DATASET_CACHE.mkdir(parents=True, exist_ok=True)

    print("Loading TinyStories …")
    ds = hf_load("roneneldan/TinyStories", trust_remote_code=True)
    train_texts = ds["train"]["text"][:max_train]
    eval_texts  = ds["validation"]["text"]

    tokenizer = load_or_train_tokenizer(vocab_size, train_texts)

    if cache_train.exists() and cache_eval.exists():
        print("Loading cached tokenised splits …")
        train_data = np.load(cache_train)
        eval_data  = np.load(cache_eval)
    else:
        print(f"Tokenising {len(train_texts):,} train stories …")
        train_data = tokenise_texts(tokenizer, train_texts, seq_len)
        print(f"Tokenising {len(eval_texts):,} eval stories …")
        eval_data  = tokenise_texts(tokenizer, eval_texts, seq_len)
        np.save(cache_train, train_data)
        np.save(cache_eval,  eval_data)
        print(f"Train chunks: {len(train_data):,}  |  Eval chunks: {len(eval_data):,}")

    return tokenizer, train_data, eval_data


class DataLoader:
    """Infinite random-batch iterator over pre-tokenised chunks."""

    def __init__(self, data: np.ndarray, batch_size: int, shuffle: bool = True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idx = 0
        self._order = np.arange(len(data))
        if shuffle:
            np.random.shuffle(self._order)

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self.idx + self.batch_size > len(self._order):
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self._order)
        batch_idx = self._order[self.idx : self.idx + self.batch_size]
        self.idx += self.batch_size
        return self.data[batch_idx].astype(np.int32)

    def __len__(self) -> int:
        return len(self.data) // self.batch_size
