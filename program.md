# Optimization Program: Minimize Tiny Model Eval Loss

## Objective
Minimize `eval/loss` on the **tiny** model (0.6M params, 2 layers, d_model=64) trained on TinyStories.

## Workflow (repeat after every experiment)

1. **Train** — Run `uv run python train.py --model tiny [flags]` from the `train/` directory.
2. **Compare runs** — Use `/wandb-compare-runs`. Judge solely by minimum `eval/loss`. Record result in Experiment Log.
3. **Load best checkpoint in Rust** — Copy the best checkpoint `.npz` and run Rust inference:
   ```bash
   # find the step with best eval/loss from W&B, then:
   cargo run --release -- --model tiny --checkpoint train/checkpoints/tiny_step<N>.npz
   ```
4. **Benchmark throughput** — Run Criterion bench for all sequence lengths:
   ```bash
   cargo bench
   ```
   Capture tokens/sec for seq=64, 128, 256, 512.
5. **Update README** — Edit the benchmark table: eval/loss, throughput numbers, trained step count, ✅ status.
6. **Commit** — Stage `README.md`, `program.md`, `train/train.py`, and any changed files. Commit with message describing the experiment and result.
7. **Push** — `git push`.
8. **Iterate** — If improvement > 0.01, continue in same direction. Otherwise try a different hyperparameter axis.

## Experiment Log

| # | Steps | Batch | LR_max | LR_min | Warmup | Notes | Min eval/loss |
|---|-------|-------|--------|--------|--------|-------|---------------|
| 1 | 5,000 | 16 | 3e-4 | 3e-5 | 200 | Baseline (W&B: 07xllm1m) | **2.871** |
| 2 | 10,000 | 16 | 3e-4 | 3e-5 | 200 | 2× steps (W&B: flesu7iu) | **2.678** |
| 3 | 10,000 | 32 | 6e-4 | 6e-5 | 400 | Linear scaling rule (W&B: d40klm1z) | **2.452** |
| 4 | 20,000 | 32 | 6e-4 | 6e-5 | 400 | 4× steps from baseline | TBD |

## Hyperparameter Strategy

### What moves the needle most (in order of expected impact)
1. **More steps** — underfitting is the primary bottleneck for a 0.6M param model on TinyStories
2. **Batch size + LR scaling** — double batch → double LR (linear scaling rule)
3. **Warmup** — scale warmup proportionally with batch size
4. **LR_MIN** — keep ratio LR_MIN/LR_MAX constant (~0.1×)

### Things to try if stuck
- Weight decay: try 0.05 or 0.2 (currently 0.1)
- Gradient clipping: try 0.5 or 2.0 (currently 1.0)
- More eval batches for lower-variance estimates

## Commands

```bash
# 1. Train
cd train
uv run python train.py --model tiny --steps 10000
uv run python train.py --model tiny --steps 10000 --batch-size 32 --lr-max 6e-4 --lr-min 6e-5 --warmup 400

# 2. Compare (invoke skill)
# /wandb-compare-runs

# 3. Load best checkpoint in Rust + quick sanity check
cargo run --release -- --model tiny --checkpoint train/checkpoints/tiny_step<N>.npz

# 4. Full Criterion benchmark
cargo bench

# 5. Commit & push
git add README.md program.md train/train.py train/checkpoints/
git commit -m "exp<N>: tiny eval/loss <X> (steps=<S>, batch=<B>, lr=<LR>)"
git push
```

## Decision Rule
- If new min eval/loss < best so far → accept, continue in same direction
- If new min eval/loss ≥ best so far → revert, try a different hyperparameter axis
- Stop when improvement < 0.01 across two consecutive experiments

## README Update Checklist (after each experiment)
- [ ] Update `eval/loss` in the benchmark table for tiny
- [ ] Update throughput numbers (seq=64/128/256/512) from `cargo bench`
- [ ] Update trained step count and mark ✅
- [ ] Note best W&B run ID
