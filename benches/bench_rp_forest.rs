use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use eonn_accel::{Auto, Vector, X1024};
use rann::rp_trees::make_forest;

mod utils;

const N_TREES: usize = 32;
const N_VECTORS: usize = 1_000;

fn benchmark_rp_forest(c: &mut Criterion) {
    fastrand::seed(2352356346463346);

    for angular in [true, false] {
        for n in [10, 32, 64] {
            c.bench_function(
                &format!("make_forest angular={angular} x1024 {n} neighbors"),
                |b| {
                    let mut data = Vec::with_capacity(N_VECTORS);
                    for _ in 0..N_VECTORS {
                        let v: Vector<X1024, Auto> = utils::random_vector();
                        data.push(v);
                    }

                    b.iter(|| {
                        make_forest(
                            black_box(&data),
                            black_box(N_TREES),
                            black_box(n),
                            black_box(angular),
                            black_box(200),
                        )
                    })
                },
            );
        }
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = benchmark_rp_forest,
);
criterion_main!(benches);
