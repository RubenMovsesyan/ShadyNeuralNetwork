use criterion::{criterion_group, criterion_main};

mod matrix_benchmarks;

criterion_group!(
    benches,
    matrix_benchmarks::dotting_benchmarks::bench_matrix_dot
);
criterion_main!(benches);
