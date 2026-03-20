/// Rotary Position Embeddings (RoPE).
///
/// Precomputes cos/sin tables of shape [max_seq_len, head_dim/2].
/// At inference time, `apply` rotates each (x[2i], x[2i+1]) pair
/// using the angle `pos * θ_i` where `θ_i = base^(-2i / head_dim)`.
pub struct RopeCache {
    /// cos[pos][i] = cos(pos · θ_i)
    cos: Vec<Vec<f32>>,
    /// sin[pos][i] = sin(pos · θ_i)
    sin: Vec<Vec<f32>>,
}

impl RopeCache {
    pub fn new(max_seq_len: usize, head_dim: usize, base: f32) -> Self {
        let half = head_dim / 2;
        let mut cos = vec![vec![0.0f32; half]; max_seq_len];
        let mut sin = vec![vec![0.0f32; half]; max_seq_len];

        for pos in 0..max_seq_len {
            for i in 0..half {
                let theta = base.powf(-(2.0 * i as f32) / head_dim as f32);
                let angle = pos as f32 * theta;
                cos[pos][i] = angle.cos();
                sin[pos][i] = angle.sin();
            }
        }

        Self { cos, sin }
    }

    /// Rotate a head vector `x` (length = head_dim) in-place at `pos`.
    ///
    /// For each pair (x[2i], x[2i+1]):
    ///   x'[2i]   = x[2i]   · cos - x[2i+1] · sin
    ///   x'[2i+1] = x[2i]   · sin + x[2i+1] · cos
    #[inline]
    pub fn apply(&self, x: &mut [f32], pos: usize) {
        let half = x.len() / 2;
        for i in 0..half {
            let c = self.cos[pos][i];
            let s = self.sin[pos][i];
            let x0 = x[2 * i];
            let x1 = x[2 * i + 1];
            x[2 * i]     = x0 * c - x1 * s;
            x[2 * i + 1] = x0 * s + x1 * c;
        }
    }
}
