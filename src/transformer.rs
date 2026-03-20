use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

use crate::{
    attention::MultiHeadAttention,
    config::Config,
    feedforward::FeedForward,
    layernorm::RmsNorm,
    matrix::Matrix,
    rope::RopeCache,
};

// ── single transformer block ──────────────────────────────────────────────────

pub struct TransformerBlock {
    attn_norm: RmsNorm,
    attention: MultiHeadAttention,
    ffn_norm: RmsNorm,
    ffn: FeedForward,
}

impl TransformerBlock {
    pub fn new(config: &Config, rng: &mut impl Rng) -> Self {
        Self {
            attn_norm: RmsNorm::new(config.d_model),
            attention: MultiHeadAttention::new(config, rng),
            ffn_norm: RmsNorm::new(config.d_model),
            ffn: FeedForward::new(config, rng),
        }
    }

    /// Pre-norm residual block:
    ///   x = x + Attention(RMSNorm(x))
    ///   x = x + FFN(RMSNorm(x))
    pub fn forward(&self, x: &Matrix, rope: &RopeCache) -> Matrix {
        let normed = self.attn_norm.forward(x);
        let attn_out = self.attention.forward(&normed, rope);
        let x = x.add(&attn_out);

        let normed = self.ffn_norm.forward(&x);
        let ffn_out = self.ffn.forward(&normed);
        x.add(&ffn_out)
    }
}

// ── full transformer ──────────────────────────────────────────────────────────

pub struct Transformer {
    embed: Matrix,           // [vocab_size, d_model]
    layers: Vec<TransformerBlock>, // 4 layers
    norm: RmsNorm,
    lm_head: Matrix,         // [d_model, vocab_size]
    rope: RopeCache,
    pub config: Config,
}

impl Transformer {
    pub fn new(config: &Config) -> Self {
        Self::with_seed(config, 42)
    }

    pub fn with_seed(config: &Config, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);

        let embed = Matrix::random(config.vocab_size, config.d_model, &mut rng);
        let layers = (0..config.n_layers)
            .map(|_| TransformerBlock::new(config, &mut rng))
            .collect();
        let norm = RmsNorm::new(config.d_model);
        let lm_head = Matrix::random(config.d_model, config.vocab_size, &mut rng);
        let rope = RopeCache::new(config.max_seq_len, config.head_dim, config.rope_base);

        Self { embed, layers, norm, lm_head, rope, config: config.clone() }
    }

    /// Forward pass: token ids → logits [seq_len, vocab_size].
    pub fn forward(&self, tokens: &[usize]) -> Matrix {
        let seq_len = tokens.len();
        let d = self.config.d_model;

        // Token embedding lookup → [seq_len, d_model]
        let mut x = Matrix::zeros(seq_len, d);
        for (i, &tok) in tokens.iter().enumerate() {
            let src = self.embed.row(tok % self.config.vocab_size);
            x.row_mut(i).copy_from_slice(src);
        }

        // 4 transformer layers
        for layer in &self.layers {
            x = layer.forward(&x, &self.rope);
        }

        // Final norm + language-model head
        let x = self.norm.forward(&x);
        x.matmul(&self.lm_head)
    }
}
