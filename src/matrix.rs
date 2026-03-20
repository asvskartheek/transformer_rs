/// Row-major dense f32 matrix.
#[derive(Clone)]
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

    /// Construct from an existing flat buffer (row-major).
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols, "data length mismatch");
        Self { rows, cols, data }
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
    ///
    /// Uses the i-k-j loop order: the inner loop is a SAXPY over contiguous
    /// slices of `b` and `c`, which the compiler auto-vectorises with SIMD.
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "matmul shape mismatch");
        let m = self.rows;
        let k = self.cols;
        let n = other.cols;
        let mut out = Matrix::zeros(m, n);
        let a = &self.data;
        let b = &other.data;
        let c = &mut out.data;
        for i in 0..m {
            for p in 0..k {
                let a_ip = a[i * k + p];
                let b_row = &b[p * n..(p + 1) * n];
                let c_row = &mut c[i * n..(i + 1) * n];
                for j in 0..n {
                    c_row[j] += a_ip * b_row[j];
                }
            }
        }
        out
    }

    /// self [m,k] × other^T where other is [n,k] → [m,n]
    ///
    /// Avoids materialising the transpose matrix.  Each output entry is a
    /// dot-product of two rows, which the compiler auto-vectorises.
    pub fn matmul_t(&self, other: &Matrix) -> Matrix {
        // self: [m, k], other: [n, k]
        assert_eq!(self.cols, other.cols, "matmul_t shape mismatch");
        let m = self.rows;
        let k = self.cols;
        let n = other.rows;
        let mut out = Matrix::zeros(m, n);
        let a = &self.data;
        let b = &other.data;
        let c = &mut out.data;
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            let c_row = &mut c[i * n..(i + 1) * n];
            for j in 0..n {
                let b_row = &b[j * k..(j + 1) * k];
                let mut dot = 0.0f32;
                for p in 0..k {
                    dot += a_row[p] * b_row[p];
                }
                c_row[j] = dot;
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

    /// In-place element-wise addition.
    pub fn add_inplace(&mut self, other: &Matrix) {
        debug_assert_eq!(self.rows, other.rows);
        debug_assert_eq!(self.cols, other.cols);
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += b;
        }
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
