# transformer_rs

A transformer architecture implemented in pure Rust — no Python, no C++ bindings, no ML frameworks.

## Architecture

- **4 transformer layers** with pre-norm residual connections
- **Rotary Position Embeddings (RoPE)** applied per-position and per-head to Q and K
- **Multi-Head Self-Attention** with causal masking
- **Feed-Forward Network** with GELU activation
- **RMS LayerNorm**

Default config: `d_model=256`, `8 heads`, `head_dim=32`, `d_ff=1024`, `vocab_size=8192`.

## Project structure

```
src/
├── config.rs       — TransformerConfig
├── matrix.rs       — row-major f32 Matrix with all required ops
├── rope.rs         — RoPE cache and apply()
├── layernorm.rs    — RMSNorm
├── attention.rs    — MultiHeadAttention
├── feedforward.rs  — FFN (GELU)
└── transformer.rs  — TransformerBlock × 4 + full Transformer
benches/
└── throughput.rs   — Criterion benchmark (tokens/second)
```

## Usage

```bash
# Quick demo — prints throughput for several sequence lengths
cargo run --release

# Criterion benchmark with statistical analysis
cargo bench
```

### Example output (`cargo run --release`)

```
Building transformer: 4 layers, d_model=256, 8 heads, head_dim=32
seq_len=  64  3 runs  total=84ms    throughput=2279 tokens/s
seq_len= 128  3 runs  total=174ms   throughput=2205 tokens/s
seq_len= 256  3 runs  total=371ms   throughput=2070 tokens/s
seq_len= 512  3 runs  total=842ms   throughput=1824 tokens/s
```

### Criterion benchmark (`cargo bench`)

**Hardware:** Apple M2, 8 GB unified memory, macOS 15

| seq\_len | time/iter | throughput |
|----------|-----------|------------|
| 64       | ~28 ms    | ~2 270 tok/s |
| 128      | ~58 ms    | ~2 200 tok/s |
| 256      | ~124 ms   | ~2 070 tok/s |
| 512      | ~277 ms   | ~1 850 tok/s |

The throughput drop with longer sequences is expected — self-attention is O(n²) in sequence length.

## Implementation notes

- All matrix operations are implemented from scratch using `Vec<f32>` (row-major).
- RoPE rotates each `(x[2i], x[2i+1])` pair using precomputed `cos`/`sin` tables stored in a flat `Vec<f32>` indexed by `pos * half + i`.
- Weights are initialised with Kaiming-normal (Box-Muller transform, no extra crates).
- The only runtime dependency is `rand` (weight init + seeding).
- Compiled with `target-cpu=native` (via `.cargo/config.toml`) so LLVM emits NEON SIMD on Apple Silicon.

## Performance notes

The `matmul` kernel uses an i-k-j loop order: the inner loop is a SAXPY over contiguous `f32` slices, which auto-vectorises to NEON on ARM and AVX2 on x86. Per-head attention avoids all intermediate `col_slice`/`transpose` allocations by computing dot-products directly from strided views into Q, K, and V. The causal softmax and V-accumulation only iterate over the lower-triangular window, halving the O(n²) attention work.

Swapping the matmul for a BLAS backend (`blas-src` + `cblas`) or `faer` would yield a further 5–20× speedup.
