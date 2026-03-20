/// Rotary Position Embeddings (RoPE).
///
/// Precomputes cos/sin tables stored flat as [max_seq_len * half].
/// At inference time, `apply` rotates each (x[2i], x[2i+1]) pair
/// using the angle `pos * θ_i` where `θ_i = base^(-2i / head_dim)`.
pub struct RopeCache {
    /// cos[pos * half + i] = cos(pos · θ_i)
    cos: Vec<f32>,
    /// sin[pos * half + i] = sin(pos · θ_i)
    sin: Vec<f32>,
    half: usize,
}

impl RopeCache {
    pub fn new(max_seq_len: usize, head_dim: usize, base: f32) -> Self {
        let half = head_dim / 2;
        let mut cos = vec![0.0f32; max_seq_len * half];
        let mut sin = vec![0.0f32; max_seq_len * half];

        for pos in 0..max_seq_len {
            for i in 0..half {
                let theta = base.powf(-(2.0 * i as f32) / head_dim as f32);
                let angle = pos as f32 * theta;
                cos[pos * half + i] = angle.cos();
                sin[pos * half + i] = angle.sin();
            }
        }

        Self { cos, sin, half }
    }

    /// Rotate a head vector `x` (length = head_dim) in-place at `pos`.
    ///
    /// For each pair (x[2i], x[2i+1]):
    ///   x'[2i]   = x[2i]   · cos - x[2i+1] · sin
    ///   x'[2i+1] = x[2i]   · sin + x[2i+1] · cos
    #[inline]
    pub fn apply(&self, x: &mut [f32], pos: usize) {
        let half = self.half;
        let cos = &self.cos[pos * half..(pos + 1) * half];
        let sin = &self.sin[pos * half..(pos + 1) * half];
        for i in 0..half {
            let x0 = x[2 * i];
            let x1 = x[2 * i + 1];
            x[2 * i]     = x0 * cos[i] - x1 * sin[i];
            x[2 * i + 1] = x0 * sin[i] + x1 * cos[i];
        }
    }
}
