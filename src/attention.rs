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
        let d_model = x.cols;
        let hd = self.head_dim;

        // Linear projections → [seq_len, d_model]
        let mut q = x.matmul(&self.wq);
        let mut k = x.matmul(&self.wk);
        let v = x.matmul(&self.wv);

        // Apply RoPE to every position and every head in Q and K
        for pos in 0..seq_len {
            let qrow = q.row_mut(pos);
            for h in 0..self.n_heads {
                let s = h * hd;
                rope.apply(&mut qrow[s..s + hd], pos);
            }
            let krow = k.row_mut(pos);
            for h in 0..self.n_heads {
                let s = h * hd;
                rope.apply(&mut krow[s..s + hd], pos);
            }
        }

        // Per-head attention without col_slice or transpose allocations.
        // Reuse a single scores buffer across all heads.
        let mut output = Matrix::zeros(seq_len, d_model);
        let mut scores = vec![0.0f32; seq_len * seq_len];

        for h in 0..self.n_heads {
            let hs = h * hd;  // head start column offset

            // scores[i,j] = dot(q[i, hs..hs+hd], k[j, hs..hs+hd]) * scale
            // With causal masking: only compute j <= i, set rest to 0.
            for i in 0..seq_len {
                let qi = &q.data[i * d_model + hs..i * d_model + hs + hd];
                let row = &mut scores[i * seq_len..(i + 1) * seq_len];

                for j in 0..=i {
                    let kj = &k.data[j * d_model + hs..j * d_model + hs + hd];
                    let mut dot = 0.0f32;
                    for p in 0..hd {
                        dot += qi[p] * kj[p];
                    }
                    row[j] = dot * self.scale;
                }

                // Softmax over the causal window [0..=i] in-place
                let window = &mut row[..=i];
                let max = window.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for x in window.iter_mut() {
                    *x = (*x - max).exp();
                    sum += *x;
                }
                let inv = sum.recip();
                for x in &mut row[..=i] {
                    *x *= inv;
                }
                // Zero out the upper triangle so the V accumulation is clean
                for x in &mut row[i + 1..seq_len] {
                    *x = 0.0;
                }
            }

            // context = scores @ V_h, accumulated into output[:, hs..hs+hd]
            for i in 0..seq_len {
                let out_row = &mut output.data[i * d_model + hs..i * d_model + hs + hd];
                // Only j <= i is non-zero (causal)
                for j in 0..=i {
                    let s = scores[i * seq_len + j];
                    let vj = &v.data[j * d_model + hs..j * d_model + hs + hd];
                    for p in 0..hd {
                        out_row[p] += s * vj[p];
                    }
                }
            }
        }

        output.matmul(&self.wo)
    }
}
