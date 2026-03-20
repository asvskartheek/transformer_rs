"""
Inference benchmark for a trained MLX model checkpoint.

Usage:
    uv run python benchmark.py --model tiny --checkpoint checkpoints/tiny_step5000.npz
"""
import argparse
import math
import time

import mlx.core as mx
from mlx.utils import tree_unflatten
import numpy as np

from configs import CONFIGS
from model import Transformer


SEQ_LENS = [64, 128, 256, 512]
RUNS = 20


def load_model(cfg, ckpt_path: str) -> Transformer:
    model = Transformer(cfg)
    weights = list(mx.load(ckpt_path).items())
    model.load_weights(weights)
    mx.eval(model.parameters())
    return model


def bench(model: Transformer, seq_len: int, runs: int = RUNS):
    tokens = mx.array(np.random.randint(0, model.cfg.vocab_size, (1, seq_len)))

    # warm-up (let Metal compile the graph)
    for _ in range(3):
        out = model(tokens)
        mx.eval(out)

    t0 = time.perf_counter()
    for _ in range(runs):
        out = model(tokens)
        mx.eval(out)
    elapsed = time.perf_counter() - t0

    total_tokens = runs * seq_len
    tok_s = total_tokens / elapsed
    ms_per_run = elapsed / runs * 1000
    return ms_per_run, tok_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(CONFIGS.keys()), required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = CONFIGS[args.model]
    print(f"\nModel : {cfg.name}  (~{cfg.n_params/1e6:.2f}M params)")
    print(f"Ckpt  : {args.checkpoint}")
    print(f"Device: {mx.default_device()}\n")

    model = load_model(cfg, args.checkpoint)
    model.eval()

    print(f"{'seq_len':>8}  {'ms/run':>8}  {'tok/s':>10}")
    print("-" * 32)
    results = {}
    for seq_len in SEQ_LENS:
        ms, tok_s = bench(model, seq_len)
        print(f"{seq_len:>8}  {ms:>8.1f}  {tok_s:>10,.0f}")
        results[seq_len] = (ms, tok_s)

    return results


if __name__ == "__main__":
    main()
