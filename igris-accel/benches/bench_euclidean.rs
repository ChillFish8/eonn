#[cfg(unix)]
extern crate blas_src;

use std::hint::black_box;
use std::ops::Sub;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use igris_accel::danger::*;
use simsimd::SpatialSimilarity;

mod utils;

fn benchmark_3rd_party_impls(c: &mut Criterion) {
    c.bench_function("euclidean ndarray x1024 auto", |b| {
        let (x, y) = utils::get_sample_vectors(1024);
        let x = ndarray::Array1::from_shape_vec((1024,), x).unwrap();
        let y = ndarray::Array1::from_shape_vec((1024,), y).unwrap();
        b.iter(|| repeat!(1000, ndarray_euclidean, &x, &y));
    });
    c.bench_function("euclidean ndarray x768 auto", |b| {
        let (x, y) = utils::get_sample_vectors(768);
        let x = ndarray::Array1::from_shape_vec((768,), x).unwrap();
        let y = ndarray::Array1::from_shape_vec((768,), y).unwrap();
        b.iter(|| repeat!(1000, ndarray_euclidean, &x, &y));
    });
    c.bench_function("euclidean ndarray x512 auto", |b| {
        let (x, y) = utils::get_sample_vectors(512);
        let x = ndarray::Array1::from_shape_vec((512,), x).unwrap();
        let y = ndarray::Array1::from_shape_vec((512,), y).unwrap();
        b.iter(|| repeat!(1000, ndarray_euclidean, &x, &y));
    });
    c.bench_function("euclidean simsimd x1024 auto", |b| {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, simsimd_euclidean, &x, &y));
    });
    c.bench_function("euclidean simsimd x768 auto", |b| {
        let (x, y) = utils::get_sample_vectors(768);
        b.iter(|| repeat!(1000, simsimd_euclidean, &x, &y));
    });
    c.bench_function("euclidean simsimd x512 auto", |b| {
        let (x, y) = utils::get_sample_vectors(512);
        b.iter(|| repeat!(1000, simsimd_euclidean, &x, &y));
    });
}

fn benchmark_dangerous_avx2_nofma_impls(c: &mut Criterion) {
    c.bench_function("euclidean avx2 x1024 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, f32_x1024_avx2_nofma_euclidean, &x, &y));
    });
    c.bench_function("euclidean avx2 x768 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(768);
        b.iter(|| repeat!(1000, f32_x768_avx2_nofma_euclidean, &x, &y));
    });
    c.bench_function("euclidean avx2 x512 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(512);
        b.iter(|| repeat!(1000, f32_x512_avx2_nofma_euclidean, &x, &y));
    });
}

fn benchmark_dangerous_avx2_fma_impls(c: &mut Criterion) {
    c.bench_function("euclidean avx2 x1024 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, f32_x1024_avx2_fma_euclidean, &x, &y));
    });
    c.bench_function("euclidean avx2 x768 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(768);
        b.iter(|| repeat!(1000, f32_x768_avx2_fma_euclidean, &x, &y));
    });
    c.bench_function("euclidean avx2 x512 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(512);
        b.iter(|| repeat!(1000, f32_x512_avx2_fma_euclidean, &x, &y));
    });
}

fn benchmark_dangerous_avx512_nofma_impls(c: &mut Criterion) {
    c.bench_function("euclidean avx512 x1024 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, f32_x1024_avx512_nofma_euclidean, &x, &y));
    });
    c.bench_function("euclidean avx512 x768 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(768);
        b.iter(|| repeat!(1000, f32_x768_avx512_nofma_euclidean, &x, &y));
    });
    c.bench_function("euclidean avx512 x512 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(512);
        b.iter(|| repeat!(1000, f32_x512_avx512_nofma_euclidean, &x, &y));
    });
}

fn benchmark_dangerous_avx512_fma_impls(c: &mut Criterion) {
    c.bench_function("euclidean avx512 x1024 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, f32_x1024_avx512_fma_euclidean, &x, &y));
    });
    c.bench_function("euclidean avx512 x768 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(768);
        b.iter(|| repeat!(1000, f32_x768_avx512_fma_euclidean, &x, &y));
    });
    c.bench_function("euclidean avx512 x512 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(512);
        b.iter(|| repeat!(1000, f32_x512_avx512_fma_euclidean, &x, &y));
    });
}

fn benchmark_fallback_nofma_impls(c: &mut Criterion) {
    c.bench_function("euclidean fallback x1024 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, f32_x1024_fallback_nofma_euclidean, &x, &y));
    });
    c.bench_function("euclidean fallback x768 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(768);
        b.iter(|| repeat!(1000, f32_x768_fallback_nofma_euclidean, &x, &y));
    });
    c.bench_function("euclidean fallback x512 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(512);
        b.iter(|| repeat!(1000, f32_x512_fallback_nofma_euclidean, &x, &y));
    });
}

fn benchmark_fallback_fma_impls(c: &mut Criterion) {
    c.bench_function("euclidean fallback x1024 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, f32_x1024_fallback_fma_euclidean, &x, &y));
    });
    c.bench_function("euclidean fallback x768 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(768);
        b.iter(|| repeat!(1000, f32_x768_fallback_fma_euclidean, &x, &y));
    });
    c.bench_function("euclidean fallback x512 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(512);
        b.iter(|| repeat!(1000, f32_x512_fallback_fma_euclidean, &x, &y));
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(60))
        .sample_size(500)
        .warm_up_time(Duration::from_secs(10));
    targets =
        benchmark_3rd_party_impls,
        benchmark_dangerous_avx2_fma_impls,
        benchmark_dangerous_avx2_nofma_impls,
        benchmark_dangerous_avx512_fma_impls,
        benchmark_dangerous_avx512_nofma_impls,
        benchmark_fallback_fma_impls,
        benchmark_fallback_nofma_impls,
);
criterion_main!(benches);

fn ndarray_euclidean(a: &ndarray::Array1<f32>, b: &ndarray::Array1<f32>) -> f32 {
    let diff = a.sub(b);
    diff.dot(&diff)
}

fn simsimd_euclidean(a: &[f32], b: &[f32]) -> f32 {
    f32::sqeuclidean(a, b).unwrap_or_default() as f32
}
