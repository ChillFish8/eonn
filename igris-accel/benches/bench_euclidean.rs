#[cfg(unix)]
extern crate blas_src;

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use igris_accel::distance_ops::*;
use simsimd::SpatialSimilarity;

fn euclidean<T: DistanceOps>(a: &T, b: &T) -> f32 {
    unsafe { a.euclidean(b) }
}

fn simsimd_euclidean(a: &[f32], b: &[f32]) -> f32 {
    f32::sqeuclidean(a, b).unwrap_or_default() as f32
}

macro_rules! repeat {
    ($n:expr, $val:block) => {{
        for _ in 0..$n {
            black_box($val);
        }
    }};
}

fn criterion_benchmark(c: &mut Criterion) {
    #[cfg(any(
        feature = "bypass-arch-flags",
        all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "avx2",
        )
    ))]
    c.bench_function("euclidean avx2 1024 nofma", |b| unsafe {
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }

        let v1 = Vector::<Avx2, X1024, f32, NoFma>::from_vec_unchecked(v1);
        let v2 = Vector::<Avx2, X1024, f32, NoFma>::from_vec_unchecked(v2);

        b.iter(|| repeat!(1000, { euclidean(black_box(&v1), black_box(&v2)) }))
    });
    #[cfg(any(
        feature = "bypass-arch-flags",
        all(
            any(target_arch = "x86_64", target_arch = "x86"),
            all(target_feature = "avx2", target_feature = "fma"),
        )
    ))]
    c.bench_function("euclidean avx2 1024 fma", |b| unsafe {
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }

        let v1 = Vector::<Avx2, X1024, f32, Fma>::from_vec_unchecked(v1);
        let v2 = Vector::<Avx2, X1024, f32, Fma>::from_vec_unchecked(v2);

        b.iter(|| repeat!(1000, { euclidean(black_box(&v1), black_box(&v2)) }))
    });
    c.bench_function("euclidean autovec 1024 nofma", |b| unsafe {
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }

        let v1 = Vector::<Fallback, X1024, f32, NoFma>::from_vec_unchecked(v1);
        let v2 = Vector::<Fallback, X1024, f32, NoFma>::from_vec_unchecked(v2);

        b.iter(|| repeat!(1000, { euclidean(black_box(&v1), black_box(&v2)) }))
    });
    #[cfg(any(
        feature = "bypass-arch-flags",
        all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "fma",
        )
    ))]
    c.bench_function("euclidean autovec 1024 fma", |b| unsafe {
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }

        let v1 = Vector::<Fallback, X1024, f32, Fma>::from_vec_unchecked(v1);
        let v2 = Vector::<Fallback, X1024, f32, Fma>::from_vec_unchecked(v2);

        b.iter(|| repeat!(1000, { euclidean(black_box(&v1), black_box(&v2)) }))
    });
    c.bench_function("euclidean simsimd 1024 auto", |b| {
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }

        b.iter(|| repeat!(1000, { simsimd_euclidean(black_box(&v1), black_box(&v2)) }))
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(75))
        .sample_size(500);
    targets = criterion_benchmark
);
criterion_main!(benches);
