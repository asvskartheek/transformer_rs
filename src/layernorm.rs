use crate::matrix::Matrix;

/// RMS LayerNorm: y = x / rms(x) · weight
///   rms(x) = sqrt(mean(x²) + ε)
pub struct RmsNorm {
    weight: Vec<f32>,
    eps: f32,
}

impl RmsNorm {
    pub fn new(d_model: usize) -> Self {
        Self { weight: vec![1.0; d_model], eps: 1e-6 }
    }

    /// Normalise each row of `x` independently.
    pub fn forward(&self, x: &Matrix) -> Matrix {
        let mut out = Matrix::zeros(x.rows, x.cols);
        for i in 0..x.rows {
            let row = x.row(i);
            let ms: f32 = row.iter().map(|&v| v * v).sum::<f32>() / row.len() as f32;
            let inv_rms = (ms + self.eps).sqrt().recip();
            for j in 0..x.cols {
                out.set(i, j, x.get(i, j) * inv_rms * self.weight[j]);
            }
        }
        out
    }
}
