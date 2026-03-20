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
seq_len=  64  3 runs  total=104ms   throughput=1841 tokens/s
seq_len= 128  3 runs  total=207ms   throughput=1849 tokens/s
seq_len= 256  3 runs  total=470ms   throughput=1632 tokens/s
seq_len= 512  3 runs  total=1.118s  throughput=1373 tokens/s
```

### Criterion benchmark (`cargo bench`)

| seq\_len | time/iter | throughput |
|----------|-----------|------------|
| 64       | ~33 ms    | ~1 900 tok/s |
| 128      | ~69 ms    | ~1 850 tok/s |
| 256      | ~156 ms   | ~1 640 tok/s |
| 512      | ~373 ms   | ~1 370 tok/s |

The throughput drop with longer sequences is expected — self-attention is O(n²) in sequence length.

## Implementation notes

- All matrix operations are implemented from scratch using `Vec<f32>` (row-major).
- RoPE rotates each `(x[2i], x[2i+1])` pair using precomputed `cos`/`sin` tables indexed by position.
- Weights are initialised with Kaiming-normal (Box-Muller transform, no extra crates).
- The only runtime dependency is `rand` (weight init + seeding).

## Performance headroom

The matmul in `matrix.rs` is a naive O(n³) loop. Swapping it for a BLAS backend (`blas-src` + `cblas`) or `faer` would yield a 10–50× speedup.
