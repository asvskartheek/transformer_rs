#[derive(Debug, Clone)]
pub struct Config {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub rope_base: f32,
}

impl Config {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        d_ff: usize,
        max_seq_len: usize,
        vocab_size: usize,
    ) -> Self {
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");
        Self {
            d_model,
            n_heads,
            n_layers,
            d_ff,
            max_seq_len,
            vocab_size,
            head_dim: d_model / n_heads,
            rope_base: 10_000.0,
        }
    }
}

impl Default for Config {
    /// 4-layer transformer, d_model=256, 8 heads, head_dim=32
    fn default() -> Self {
        Self::new(256, 8, 4, 1024, 512, 8192)
    }
}
