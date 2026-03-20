use crate::{config::Config, matrix::Matrix};
use rand::Rng;

/// Position-wise feed-forward network with GELU activation.
///   FFN(x) = GELU(x W1) W2
pub struct FeedForward {
    w1: Matrix, // [d_model, d_ff]
    w2: Matrix, // [d_ff, d_model]
}

impl FeedForward {
    pub fn new(config: &Config, rng: &mut impl Rng) -> Self {
        Self {
            w1: Matrix::random(config.d_model, config.d_ff, rng),
            w2: Matrix::random(config.d_ff, config.d_model, rng),
        }
    }

    /// Construct from pre-loaded weight matrices (already transposed to Rust layout).
    pub fn from_weights(w1: Matrix, w2: Matrix) -> Self {
        Self { w1, w2 }
    }

    pub fn forward(&self, x: &Matrix) -> Matrix {
        let mut hidden = x.matmul(&self.w1);
        gelu_inplace(&mut hidden);
        hidden.matmul(&self.w2)
    }
}

/// Gaussian Error Linear Unit (tanh approximation).
fn gelu_inplace(m: &mut Matrix) {
    const C: f32 = 0.797_884_6; // sqrt(2/π)
    for v in &mut m.data {
        let x = *v;
        *v = 0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh());
    }
}
