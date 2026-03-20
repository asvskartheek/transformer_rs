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
        let cols = x.cols;
        let inv_cols = (cols as f32).recip();
        let mut out = Matrix::zeros(x.rows, cols);
        for i in 0..x.rows {
            let in_row = &x.data[i * cols..(i + 1) * cols];
            let out_row = &mut out.data[i * cols..(i + 1) * cols];
            let ms: f32 = in_row.iter().map(|&v| v * v).sum::<f32>() * inv_cols;
            let inv_rms = (ms + self.eps).sqrt().recip();
            for j in 0..cols {
                out_row[j] = in_row[j] * inv_rms * self.weight[j];
            }
        }
        out
    }
}
