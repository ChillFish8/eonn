extern crate blas_src;

use std::hint::black_box;
use std::marker::PhantomData;

use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array1, ArrayView1};
use igris_accel::vector_ops::{Avx2, DistanceOps, Fallback, Fma, NoFma, Vector, X1024};
use simsimd::SpatialSimilarity;

fn dot<T: DistanceOps>(a: &T, b: &T) -> f32 {
    unsafe { a.dot(b) }
}

fn simsimd_dot(a: &[f32], b: &[f32]) -> f32 {
    f32::dot(a, b).unwrap_or_default() as f32
}

fn ndarray_dot(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    a.dot(b)
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("dot ndarray 1024 auto", |b| {
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }
        
        let v1 = Array1::from_shape_vec((1024,), v1).unwrap();
        let v2 = Array1::from_shape_vec((1024,), v2).unwrap();

        b.iter(|| ndarray_dot(black_box(&v1), black_box(&v2)))
    });
    c.bench_function("dot simsimd 1024 auto", |b| {
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }

        b.iter(|| simsimd_dot(black_box(&v1), black_box(&v2)))
    });
    c.bench_function("dot fallback 1024 nofma", |b| {
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }

        let v1 = Vector::<Fallback, X1024, f32, NoFma>(v1, PhantomData);
        let v2 = Vector::<Fallback, X1024, f32, NoFma>(v2, PhantomData);

        b.iter(|| dot(black_box(&v1), black_box(&v2)))
    });
    c.bench_function("dot fallback 1024 fma", |b| {
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }

        let v1 = Vector::<Fallback, X1024, f32, Fma>(v1, PhantomData);
        let v2 = Vector::<Fallback, X1024, f32, Fma>(v2, PhantomData);

        b.iter(|| dot(black_box(&v1), black_box(&v2)))
    });
    c.bench_function("dot avx2 1024 nofma", |b| {
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }

        let v1 = Vector::<Avx2, X1024, f32, NoFma>(v1, PhantomData);
        let v2 = Vector::<Avx2, X1024, f32, NoFma>(v2, PhantomData);

        b.iter(|| dot(black_box(&v1), black_box(&v2)))
    });
    c.bench_function("dot avx2 1024 fma", |b| {
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }

        let v1 = Vector::<Avx2, X1024, f32, Fma>(v1, PhantomData);
        let v2 = Vector::<Avx2, X1024, f32, Fma>(v2, PhantomData);

        b.iter(|| dot(black_box(&v1), black_box(&v2)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
