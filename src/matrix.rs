/// Row-major dense f32 matrix.
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn randn(rng: &mut impl rand::Rng) -> f32 {
    // Box-Muller transform
    let u1: f32 = rng.gen::<f32>().max(1e-10);
    let u2: f32 = rng.gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

// ── Matrix ────────────────────────────────────────────────────────────────────

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    /// Kaiming-normal initialisation: std = sqrt(2 / fan_in)
    pub fn random(rows: usize, cols: usize, rng: &mut impl rand::Rng) -> Self {
        let std = (2.0 / rows as f32).sqrt();
        let data = (0..rows * cols).map(|_| randn(rng) * std).collect();
        Self { rows, cols, data }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.data[i * self.cols + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f32) {
        self.data[i * self.cols + j] = v;
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        &mut self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// Extract columns `[start, end)` as a new matrix `[rows, end-start]`.
    pub fn col_slice(&self, start: usize, end: usize) -> Matrix {
        let ncols = end - start;
        let mut out = Matrix::zeros(self.rows, ncols);
        for i in 0..self.rows {
            for j in 0..ncols {
                out.set(i, j, self.get(i, start + j));
            }
        }
        out
    }

    /// Write `src` into columns starting at `col_start`.
    pub fn write_col_slice(&mut self, col_start: usize, src: &Matrix) {
        assert_eq!(self.rows, src.rows);
        for i in 0..src.rows {
            for j in 0..src.cols {
                self.set(i, col_start + j, src.get(i, j));
            }
        }
    }

    /// self [m,k] × other [k,n] → [m,n]
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "matmul shape mismatch");
        let mut out = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                for j in 0..other.cols {
                    let v = out.get(i, j) + a * other.get(k, j);
                    out.set(i, j, v);
                }
            }
        }
        out
    }

    pub fn transpose(&self) -> Matrix {
        let mut out = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.set(j, i, self.get(i, j));
            }
        }
        out
    }

    /// Element-wise addition (returns new matrix).
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn scale_inplace(&mut self, s: f32) {
        for v in &mut self.data {
            *v *= s;
        }
    }

    /// Set upper-triangular entries (j > i) to -∞ for causal masking.
    pub fn causal_mask(&mut self) {
        assert_eq!(self.rows, self.cols);
        for i in 0..self.rows {
            for j in (i + 1)..self.cols {
                self.set(i, j, f32::NEG_INFINITY);
            }
        }
    }

    /// Numerically stable row-wise softmax (in-place).
    pub fn softmax_rows(&mut self) {
        for i in 0..self.rows {
            let row = self.row_mut(i);
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for x in row.iter_mut() {
                *x = (*x - max).exp();
                sum += *x;
            }
            let inv = sum.recip();
            for x in row.iter_mut() {
                *x *= inv;
            }
        }
    }
}
