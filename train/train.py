"""
Train a small transformer on TinyStories using MLX (Apple Silicon).
Traces every metric to Weights & Biases.

Usage (from the train/ directory):
    uv run python train.py --model tiny
    uv run python train.py --model small
    uv run python train.py --model medium   # matches Rust default config
    uv run python train.py --model large
    uv run python train.py --all            # run all sizes sequentially
"""
import argparse
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import wandb
from mlx.utils import tree_flatten

from configs import CONFIGS, ModelConfig
from data import DataLoader, load_dataset
from model import Transformer


# ── hyper-parameters ──────────────────────────────────────────────────────────

BATCH_SIZE     = 16
MAX_STEPS      = 5_000          # per model size; crank up for real runs
EVAL_INTERVAL  = 200
EVAL_BATCHES   = 50
LOG_INTERVAL   = 20
LR_MAX         = 3e-4
LR_MIN         = 3e-5
WARMUP_STEPS   = 200
GRAD_CLIP      = 1.0
WEIGHT_DECAY   = 0.1
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


# ── lr schedule ───────────────────────────────────────────────────────────────

def lr_schedule(step: int, max_steps: int) -> float:
    if step < WARMUP_STEPS:
        return LR_MAX * step / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(max_steps - WARMUP_STEPS, 1)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LR_MIN + (LR_MAX - LR_MIN) * cosine


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: Transformer, loader: DataLoader, n_batches: int) -> dict:
    losses = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        tokens = mx.array(batch)
        loss   = model.loss(tokens)
        mx.eval(loss)
        losses.append(loss.item())
    mean_loss = float(np.mean(losses))
    return {"eval/loss": mean_loss, "eval/perplexity": math.exp(mean_loss)}


# ── training loop ─────────────────────────────────────────────────────────────

def train_model(cfg: ModelConfig):
    print(f"\n{'='*60}")
    print(f"  Model: {cfg.name}  |  ~{cfg.n_params/1e6:.1f}M params")
    print(f"  {cfg.n_layers} layers, d_model={cfg.d_model}, "
          f"{cfg.n_heads} heads, d_ff={cfg.d_ff}, vocab={cfg.vocab_size}")
    print(f"{'='*60}\n")

    # ── data ──────────────────────────────────────────────────────────────────
    tokenizer, train_data, eval_data = load_dataset(cfg.vocab_size, cfg.max_seq_len)
    train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True)
    eval_loader  = DataLoader(eval_data,  BATCH_SIZE, shuffle=False)
    print(f"Train batches available: {len(train_loader):,}")
    print(f"Eval  batches available: {len(eval_loader):,}")

    # ── model + optimiser ─────────────────────────────────────────────────────
    model     = Transformer(cfg)
    mx.eval(model.parameters())

    optimizer = optim.AdamW(learning_rate=LR_MAX, weight_decay=WEIGHT_DECAY)

    loss_and_grad = nn.value_and_grad(model, model.loss)

    # ── wandb ─────────────────────────────────────────────────────────────────
    run = wandb.init(
        project="transformer-rs-mlx",
        name=f"{cfg.name}_s{MAX_STEPS}_b{BATCH_SIZE}_lr{LR_MAX:.0e}",
        config={
            "model_size":  cfg.name,
            "d_model":     cfg.d_model,
            "n_heads":     cfg.n_heads,
            "n_layers":    cfg.n_layers,
            "d_ff":        cfg.d_ff,
            "max_seq_len": cfg.max_seq_len,
            "vocab_size":  cfg.vocab_size,
            "head_dim":    cfg.head_dim,
            "n_params":    cfg.n_params,
            "batch_size":  BATCH_SIZE,
            "max_steps":   MAX_STEPS,
            "lr_max":      LR_MAX,
            "lr_min":      LR_MIN,
            "warmup_steps": WARMUP_STEPS,
            "grad_clip":   GRAD_CLIP,
            "weight_decay": WEIGHT_DECAY,
            "dataset":     "roneneldan/TinyStories",
        },
        reinit=True,
    )

    # ── training ──────────────────────────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    step         = 0
    tokens_seen  = 0
    t0           = time.perf_counter()
    running_loss = 0.0

    train_iter = iter(train_loader)

    for step in range(1, MAX_STEPS + 1):
        # lr update
        lr = lr_schedule(step, MAX_STEPS)
        optimizer.learning_rate = lr

        batch  = next(train_iter)
        tokens = mx.array(batch)

        loss, grads = loss_and_grad(tokens)
        grads = optim.clip_grad_norm(grads, GRAD_CLIP)[0]
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())

        tokens_seen  += batch.shape[0] * batch.shape[1]
        running_loss += loss.item()

        # ── periodic logging ──────────────────────────────────────────────────
        if step % LOG_INTERVAL == 0:
            t1       = time.perf_counter()
            elapsed  = t1 - t0
            tok_s    = (LOG_INTERVAL * BATCH_SIZE * cfg.max_seq_len) / elapsed
            avg_loss = running_loss / LOG_INTERVAL

            wandb.log({
                "train/loss":       avg_loss,
                "train/perplexity": math.exp(avg_loss),
                "train/lr":         lr,
                "train/tokens_seen": tokens_seen,
                "perf/tokens_per_sec": tok_s,
                "step": step,
            }, step=step)

            print(f"step {step:5d}/{MAX_STEPS}  loss={avg_loss:.4f}  "
                  f"ppl={math.exp(avg_loss):.2f}  lr={lr:.2e}  "
                  f"{tok_s:,.0f} tok/s")

            running_loss = 0.0
            t0 = t1

        # ── evaluation ────────────────────────────────────────────────────────
        if step % EVAL_INTERVAL == 0:
            model.eval()
            metrics = evaluate(model, eval_loader, EVAL_BATCHES)
            model.train()

            wandb.log({**metrics, "step": step}, step=step)
            print(f"  → eval loss={metrics['eval/loss']:.4f}  "
                  f"ppl={metrics['eval/perplexity']:.2f}")

            # checkpoint
            ckpt = CHECKPOINT_DIR / f"{cfg.name}_step{step}.npz"
            mx.savez(str(ckpt), **dict(tree_flatten(model.parameters())))
            wandb.save(str(ckpt))

    # final eval
    model.eval()
    final = evaluate(model, eval_loader, len(eval_loader))
    wandb.log({**final, "step": MAX_STEPS}, step=MAX_STEPS)
    print(f"\nFinal eval — loss={final['eval/loss']:.4f}  "
          f"ppl={final['eval/perplexity']:.2f}")

    run.finish()
    return final


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    global MAX_STEPS, BATCH_SIZE, LR_MAX, LR_MIN, WARMUP_STEPS, GRAD_CLIP, WEIGHT_DECAY  # noqa: PLW0603
    parser = argparse.ArgumentParser(description="Train MLX transformer on TinyStories")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=list(CONFIGS.keys()),
                       help="Single model size to train")
    group.add_argument("--all",   action="store_true",
                       help="Train all model sizes sequentially")
    parser.add_argument("--steps",        type=int,   default=MAX_STEPS,    help=f"Training steps (default {MAX_STEPS})")
    parser.add_argument("--batch-size",   type=int,   default=BATCH_SIZE,   help=f"Batch size (default {BATCH_SIZE})")
    parser.add_argument("--lr-max",       type=float, default=LR_MAX,       help=f"Peak learning rate (default {LR_MAX})")
    parser.add_argument("--lr-min",       type=float, default=LR_MIN,       help=f"Final learning rate (default {LR_MIN})")
    parser.add_argument("--warmup",       type=int,   default=WARMUP_STEPS, help=f"Warmup steps (default {WARMUP_STEPS})")
    parser.add_argument("--grad-clip",    type=float, default=GRAD_CLIP,    help=f"Gradient clip norm (default {GRAD_CLIP})")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help=f"AdamW weight decay (default {WEIGHT_DECAY})")
    args = parser.parse_args()

    MAX_STEPS    = args.steps
    BATCH_SIZE   = args.batch_size
    LR_MAX       = args.lr_max
    LR_MIN       = args.lr_min
    WARMUP_STEPS = args.warmup
    GRAD_CLIP    = args.grad_clip
    WEIGHT_DECAY = args.weight_decay

    sizes = list(CONFIGS.keys()) if args.all else [args.model]
    for name in sizes:
        train_model(CONFIGS[name])


if __name__ == "__main__":
    main()
