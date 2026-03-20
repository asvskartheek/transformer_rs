use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use transformer_rs::{config::Config, transformer::Transformer};

fn bench_throughput(c: &mut Criterion) {
    let config = Config::default(); // 4-layer, d_model=256, 8 heads
    let model = Transformer::new(&config);

    let mut group = c.benchmark_group("transformer_throughput");

    for &seq_len in &[64usize, 128, 256, 512] {
        let tokens: Vec<usize> = (0..seq_len).collect();

        // Tell Criterion how many "elements" (tokens) each iteration processes
        // so it can report tokens/second automatically.
        group.throughput(Throughput::Elements(seq_len as u64));

        group.bench_with_input(
            BenchmarkId::new("forward_pass", seq_len),
            &tokens,
            |b, toks| b.iter(|| model.forward(toks)),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_throughput);
criterion_main!(benches);
