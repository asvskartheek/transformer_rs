# transformer-rs

A pure-Rust transformer with a companion MLX training pipeline for Apple Silicon.

## W&B Training Dashboard

**https://wandb.ai/asvskartheek/transformer-rs-mlx**

All training runs (tiny → small → medium → large) traced live: loss, perplexity, LR, tokens/sec, eval metrics every 200 steps.

---

## Rust — inference / benchmarks

Pure-Rust forward pass: RMSNorm, multi-head attention with RoPE, GELU feed-forward. No dependencies beyond `rand`.

```bash
cargo run --release   # throughput demo
cargo bench           # Criterion statistical benchmark
```

### Architecture

- Pre-norm residual blocks (RMSNorm → Attention → RMSNorm → FFN)
- Rotary Position Embeddings (RoPE) on Q and K
- Causal self-attention, GELU feed-forward
- i-k-j matmul loop — auto-vectorises to NEON on Apple Silicon

Default config (matches the `medium` training run):

| param | value |
|---|---|
| layers | 4 |
| d\_model | 256 |
| heads | 8 |
| head\_dim | 32 |
| d\_ff | 1024 |
| vocab\_size | 8192 |
| max\_seq\_len | 512 |

### Rust inference benchmark (Apple M2, 8 GB)

CPU-only. Throughput scales with model size — smaller models are faster.

```bash
cargo run --release -- --model <size> [--checkpoint <path>]
```

| model | seq\_len=64 | seq\_len=128 | seq\_len=256 | seq\_len=512 |
|---|---|---|---|---|
| tiny (0.6M) | ~19 600 tok/s | ~28 100 tok/s | ~26 100 tok/s | ~21 900 tok/s |
| medium (7.3M) | ~2 250 tok/s | ~2 190 tok/s | ~2 010 tok/s | ~1 840 tok/s |

Trained weights load the same speed as random — weight loading doesn't change throughput, only model size does.

---

## Python/MLX — training

Trains on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) using [MLX](https://github.com/ml-explore/mlx) (Apple Silicon GPU). BPE tokenizer, AdamW + cosine-decay LR, full WandB tracing.

### Setup

```bash
cd train
uv sync
```

### Run

```bash
uv run python train.py --model tiny
uv run python train.py --model small
uv run python train.py --model medium   # matches Rust default config
uv run python train.py --model large
uv run python train.py --all            # all sizes sequentially
uv run python train.py --model medium --steps 10000
```

### Model sizes

| name | layers | d\_model | heads | d\_ff | vocab | ~params |
|---|---|---|---|---|---|---|
| tiny | 2 | 64 | 2 | 256 | 4096 | 0.6M |
| small | 4 | 128 | 4 | 512 | 4096 | 1.8M |
| **medium** | **4** | **256** | **8** | **1024** | **8192** | **7.3M** |
| large | 6 | 512 | 8 | 2048 | 8192 | 27M |

### Project layout

```
train/
├── configs.py    — all four ModelConfig definitions
├── model.py      — MLX transformer (RMSNorm, RoPE, MHA, FFN, LM head)
├── data.py       — TinyStories loader, BPE tokenizer trainer, DataLoader
├── train.py      — training loop, WandB logging, checkpointing
└── pyproject.toml / uv.lock
```

### WandB metrics

| metric | frequency |
|---|---|
| `train/loss`, `train/perplexity` | every 20 steps |
| `train/lr`, `train/tokens_seen` | every 20 steps |
| `perf/tokens_per_sec` | every 20 steps |
| `eval/loss`, `eval/perplexity` | every 200 steps |
| checkpoint artifact | every 200 steps |
