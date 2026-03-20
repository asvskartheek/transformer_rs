use std::time::Instant;
use transformer_rs::{config::Config, transformer::Transformer};

fn main() {
    let config = Config::default(); // 4-layer, d_model=256, 8 heads
    println!(
        "Building transformer: {} layers, d_model={}, {} heads, head_dim={}",
        config.n_layers, config.d_model, config.n_heads, config.head_dim
    );

    let model = Transformer::new(&config);

    for &seq_len in &[64usize, 128, 256, 512] {
        let tokens: Vec<usize> = (0..seq_len).collect();

        // Warm-up
        let _ = model.forward(&tokens);

        let runs = 3usize;
        let start = Instant::now();
        for _ in 0..runs {
            let _ = model.forward(&tokens);
        }
        let elapsed = start.elapsed();

        let total_tokens = (runs * seq_len) as f64;
        let secs = elapsed.as_secs_f64();
        let throughput = total_tokens / secs;

        println!(
            "seq_len={seq_len:4}  {runs} runs  total={elapsed:.3?}  \
             throughput={throughput:.0} tokens/s"
        );
    }
}
