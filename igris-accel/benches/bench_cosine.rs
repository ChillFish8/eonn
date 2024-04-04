extern crate blas_src;

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use igris_accel::vector_ops::{Avx2, DistanceOps, Fallback, Fma, NoFma, Vector, X1024};
use simsimd::SpatialSimilarity;

fn cosine<T: DistanceOps>(a: &T, b: &T) -> f32 {
    unsafe { a.cosine(b) }
}

fn simsimd_cosine(a: &[f32], b: &[f32]) -> f32 {
    f32::cosine(a, b).unwrap_or_default() as f32
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("cosine autovec 1024 nofma", |b| unsafe {
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }

        let v1 = Vector::<Fallback, X1024, f32, NoFma>::from_vec_unchecked(v1);
        let v2 = Vector::<Fallback, X1024, f32, NoFma>::from_vec_unchecked(v2);

        b.iter(|| cosine(black_box(&v1), black_box(&v2)))
    });
    c.bench_function("cosine autovec 1024 fma", |b| unsafe {
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }

        let v1 = Vector::<Fallback, X1024, f32, Fma>::from_vec_unchecked(v1);
        let v2 = Vector::<Fallback, X1024, f32, Fma>::from_vec_unchecked(v2);

        b.iter(|| cosine(black_box(&v1), black_box(&v2)))
    });
    c.bench_function("cosine simsimd 1024 auto", |b| {
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }

        b.iter(|| simsimd_cosine(black_box(&v1), black_box(&v2)))
    });
    c.bench_function("cosine avx2 1024 nofma", |b| unsafe {
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }

        let v1 = Vector::<Avx2, X1024, f32, NoFma>::from_vec_unchecked(v1);
        let v2 = Vector::<Avx2, X1024, f32, NoFma>::from_vec_unchecked(v2);

        b.iter(|| cosine(black_box(&v1), black_box(&v2)))
    });
    c.bench_function("cosine avx2 1024 fma", |b| unsafe {
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }

        let v1 = Vector::<Avx2, X1024, f32, Fma>::from_vec_unchecked(v1);
        let v2 = Vector::<Avx2, X1024, f32, Fma>::from_vec_unchecked(v2);

        b.iter(|| cosine(black_box(&v1), black_box(&v2)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);