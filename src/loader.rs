/// Load trained MLX weights from a `.npz` file into a Rust `Transformer`.
///
/// MLX `nn.Linear` stores weights as `[out, in]` (same as PyTorch).
/// The Rust forward pass computes `x.matmul(&w)` where x is `[seq, in]`,
/// so every linear weight must be transposed to `[in, out]` on load.
/// Embeddings and RMSNorm vectors are 1-D / row-major and need no transpose.
use std::{fs::File, io::BufReader};

use npyz::NpyFile;
use zip::ZipArchive;

use crate::{
    attention::MultiHeadAttention,
    config::Config,
    feedforward::FeedForward,
    layernorm::RmsNorm,
    matrix::Matrix,
    transformer::{Transformer, TransformerBlock},
    rope::RopeCache,
};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Read a named array from the zip archive as `Vec<f32>`.
fn read_array(archive: &mut ZipArchive<BufReader<File>>, key: &str) -> Vec<f32> {
    let entry_name = format!("{key}.npy");
    let entry = archive
        .by_name(&entry_name)
        .unwrap_or_else(|_| panic!("key '{key}' not found in npz"));
    let npy = NpyFile::new(entry).expect("failed to parse npy");
    npy.into_vec::<f32>().expect("failed to read f32 array")
}

/// Transpose a flat row-major `[rows, cols]` buffer → `[cols, rows]`.
fn transpose(data: Vec<f32>, rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

// ── public entry point ────────────────────────────────────────────────────────

impl Transformer {
    /// Build a `Transformer` by loading weights saved by the Python/MLX trainer.
    ///
    /// The `.npz` file must have been produced by:
    /// ```python
    /// mx.savez(path, **dict(tree_flatten(model.parameters())))
    /// ```
    pub fn from_npz(path: &str, config: &Config) -> Self {
        let file = File::open(path)
            .unwrap_or_else(|e| panic!("cannot open '{path}': {e}"));
        let reader = BufReader::new(file);
        let mut archive = ZipArchive::new(reader)
            .unwrap_or_else(|e| panic!("not a valid zip/npz: {e}"));

        // ── embedding [vocab_size, d_model] — same orientation in Rust ────────
        let embed_data = read_array(&mut archive, "embed.weight");
        let embed = Matrix::from_vec(config.vocab_size, config.d_model, embed_data);

        // ── transformer layers ────────────────────────────────────────────────
        let layers = (0..config.n_layers)
            .map(|i| load_block(&mut archive, config, i))
            .collect::<Vec<_>>();

        // ── final norm + lm_head ─────────────────────────────────────────────
        let norm_w = read_array(&mut archive, "norm.weight");
        let norm = RmsNorm::from_weight(norm_w);

        // lm_head: MLX stores [vocab_size, d_model]; Rust needs [d_model, vocab_size]
        let lm_data = read_array(&mut archive, "lm_head.weight");
        let lm_head = Matrix::from_vec(
            config.d_model,
            config.vocab_size,
            transpose(lm_data, config.vocab_size, config.d_model),
        );

        let rope = RopeCache::new(config.max_seq_len, config.head_dim, config.rope_base);

        Transformer { embed, layers, norm, lm_head, rope, config: config.clone() }
    }
}

fn load_block(
    archive: &mut ZipArchive<BufReader<File>>,
    config: &Config,
    i: usize,
) -> TransformerBlock {
    let d = config.d_model;
    let ff = config.d_ff;

    // ── attention norm [d_model] ──────────────────────────────────────────────
    let attn_norm = RmsNorm::from_weight(
        read_array(archive, &format!("layers.{i}.attn_norm.weight")),
    );

    // ── attention projections — MLX: [out, in] → Rust: [in, out] via transpose
    let load_t = |archive: &mut ZipArchive<BufReader<File>>, key: &str, out_dim: usize, in_dim: usize| {
        let raw = read_array(archive, key);
        Matrix::from_vec(in_dim, out_dim, transpose(raw, out_dim, in_dim))
    };

    let wq = load_t(archive, &format!("layers.{i}.attn.wq.weight"), d, d);
    let wk = load_t(archive, &format!("layers.{i}.attn.wk.weight"), d, d);
    let wv = load_t(archive, &format!("layers.{i}.attn.wv.weight"), d, d);
    let wo = load_t(archive, &format!("layers.{i}.attn.wo.weight"), d, d);
    let attention = MultiHeadAttention::from_weights(config, wq, wk, wv, wo);

    // ── ffn norm [d_model] ────────────────────────────────────────────────────
    let ffn_norm = RmsNorm::from_weight(
        read_array(archive, &format!("layers.{i}.ffn_norm.weight")),
    );

    // w1: MLX [d_ff, d_model] → Rust [d_model, d_ff]
    let w1 = load_t(archive, &format!("layers.{i}.ffn.w1.weight"), ff, d);
    // w2: MLX [d_model, d_ff] → Rust [d_ff, d_model]
    let w2 = load_t(archive, &format!("layers.{i}.ffn.w2.weight"), d, ff);
    let ffn = FeedForward::from_weights(w1, w2);

    TransformerBlock { attn_norm, attention, ffn_norm, ffn }
}
