use std::time::Instant;
use transformer_rs::{config::Config, transformer::Transformer};

/// Model size presets — mirror the Python training configs.
fn config_for(name: &str) -> Config {
    match name {
        "tiny"   => Config::new(64,  2, 2, 256,  512, 4096),
        "small"  => Config::new(128, 4, 4, 512,  512, 4096),
        "medium" => Config::new(256, 8, 4, 1024, 512, 8192),
        "large"  => Config::new(512, 8, 6, 2048, 512, 8192),
        other    => panic!("unknown model size '{other}'; choose tiny/small/medium/large"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Simple arg parsing: [--model <size>] [--checkpoint <path>]
    let mut model_name = "medium".to_string();
    let mut checkpoint: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model"      => { i += 1; model_name  = args[i].clone(); }
            "--checkpoint" => { i += 1; checkpoint  = Some(args[i].clone()); }
            other          => eprintln!("warning: unknown arg '{other}'"),
        }
        i += 1;
    }

    let config = config_for(&model_name);
    println!(
        "Model: {}  ({} layers, d_model={}, {} heads)",
        model_name, config.n_layers, config.d_model, config.n_heads
    );

    let model = match &checkpoint {
        Some(path) => {
            println!("Loading weights from {path}");
            Transformer::from_npz(path, &config)
        }
        None => {
            println!("No checkpoint — using random weights");
            Transformer::new(&config)
        }
    };

    println!();
    println!("{:>8}  {:>6}  {:>12}", "seq_len", "runs", "throughput");
    println!("{}", "-".repeat(32));

    for &seq_len in &[64usize, 128, 256, 512] {
        let tokens: Vec<usize> = (0..seq_len).map(|t| t % config.vocab_size).collect();

        // Warm-up
        let _ = model.forward(&tokens);

        let runs = 10usize;
        let start = Instant::now();
        for _ in 0..runs {
            let _ = model.forward(&tokens);
        }
        let elapsed = start.elapsed();

        let throughput = (runs * seq_len) as f64 / elapsed.as_secs_f64();
        println!(
            "{:>8}  {:>6}  {:>9.0} tok/s",
            seq_len, runs, throughput
        );
    }
}
