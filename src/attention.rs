use crate::{config::Config, matrix::Matrix, rope::RopeCache};
use rand::Rng;

/// Multi-Head Self-Attention with Rotary Position Embeddings.
pub struct MultiHeadAttention {
    wq: Matrix, // [d_model, d_model]
    wk: Matrix,
    wv: Matrix,
    wo: Matrix,
    n_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl MultiHeadAttention {
    pub fn new(config: &Config, rng: &mut impl Rng) -> Self {
        let d = config.d_model;
        Self {
            wq: Matrix::random(d, d, rng),
            wk: Matrix::random(d, d, rng),
            wv: Matrix::random(d, d, rng),
            wo: Matrix::random(d, d, rng),
            n_heads: config.n_heads,
            head_dim: config.head_dim,
            scale: (config.head_dim as f32).sqrt().recip(),
        }
    }

    pub fn forward(&self, x: &Matrix, rope: &RopeCache) -> Matrix {
        let seq_len = x.rows;

        // Linear projections → [seq_len, d_model]
        let mut q = x.matmul(&self.wq);
        let mut k = x.matmul(&self.wk);
        let v = x.matmul(&self.wv);

        // Apply RoPE to every position and every head in Q and K
        for pos in 0..seq_len {
            let qrow = q.row_mut(pos);
            for h in 0..self.n_heads {
                let s = h * self.head_dim;
                rope.apply(&mut qrow[s..s + self.head_dim], pos);
            }
            let krow = k.row_mut(pos);
            for h in 0..self.n_heads {
                let s = h * self.head_dim;
                rope.apply(&mut krow[s..s + self.head_dim], pos);
            }
        }

        // Compute attention per head, accumulate into output
        let mut output = Matrix::zeros(seq_len, q.cols);

        for h in 0..self.n_heads {
            let s = h * self.head_dim;
            let e = s + self.head_dim;

            let q_h = q.col_slice(s, e); // [seq_len, head_dim]
            let k_h = k.col_slice(s, e);
            let v_h = v.col_slice(s, e);

            // scores = Q_h @ K_h^T · scale   [seq_len, seq_len]
            let mut scores = q_h.matmul(&k_h.transpose());
            scores.scale_inplace(self.scale);
            scores.causal_mask();
            scores.softmax_rows();

            // context = scores @ V_h          [seq_len, head_dim]
            let ctx = scores.matmul(&v_h);
            output.write_col_slice(s, &ctx);
        }

        output.matmul(&self.wo)
    }
}
