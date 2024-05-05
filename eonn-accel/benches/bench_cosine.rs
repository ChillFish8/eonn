#[cfg(unix)]
extern crate blas_src;

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use eonn_accel::danger::*;
use simsimd::SpatialSimilarity;

mod utils;

fn benchmark_3rd_party_impls(c: &mut Criterion) {
    c.bench_function("cosine ndarray x1024 auto", |b| {
        let (x, y) = utils::get_sample_vectors(1024);
        let x = ndarray::Array1::from_shape_vec((1024,), x).unwrap();
        let y = ndarray::Array1::from_shape_vec((1024,), y).unwrap();
        b.iter(|| repeat!(1000, ndarray_cosine, &x, &y));
    });
    c.bench_function("cosine ndarray x768 auto", |b| {
        let (x, y) = utils::get_sample_vectors(768);
        let x = ndarray::Array1::from_shape_vec((768,), x).unwrap();
        let y = ndarray::Array1::from_shape_vec((768,), y).unwrap();
        b.iter(|| repeat!(1000, ndarray_cosine, &x, &y));
    });
    c.bench_function("cosine ndarray x512 auto", |b| {
        let (x, y) = utils::get_sample_vectors(512);
        let x = ndarray::Array1::from_shape_vec((512,), x).unwrap();
        let y = ndarray::Array1::from_shape_vec((512,), y).unwrap();
        b.iter(|| repeat!(1000, ndarray_cosine, &x, &y));
    });
    c.bench_function("cosine simsimd x1024 auto", |b| {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, simsimd_cosine, &x, &y));
    });
    c.bench_function("cosine simsimd x768 auto", |b| {
        let (x, y) = utils::get_sample_vectors(768);
        b.iter(|| repeat!(1000, simsimd_cosine, &x, &y));
    });
    c.bench_function("cosine simsimd x512 auto", |b| {
        let (x, y) = utils::get_sample_vectors(512);
        b.iter(|| repeat!(1000, simsimd_cosine, &x, &y));
    });
}

fn benchmark_dangerous_avx2_nofma_impls(c: &mut Criterion) {
    c.bench_function("cosine avx2 x1024 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, f32_x1024_avx2_nofma_cosine, &x, &y));
    });
    c.bench_function("cosine avx2 x768 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(768);
        b.iter(|| repeat!(1000, f32_x768_avx2_nofma_cosine, &x, &y));
    });
    c.bench_function("cosine avx2 x512 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(512);
        b.iter(|| repeat!(1000, f32_x512_avx2_nofma_cosine, &x, &y));
    });
}

#[cfg(feature = "nightly")]
fn benchmark_dangerous_avx2_fma_impls(c: &mut Criterion) {
    c.bench_function("cosine avx2 x1024 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, f32_x1024_avx2_fma_cosine, &x, &y));
    });
    c.bench_function("cosine avx2 x768 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(768);
        b.iter(|| repeat!(1000, f32_x768_avx2_fma_cosine, &x, &y));
    });
    c.bench_function("cosine avx2 x512 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(512);
        b.iter(|| repeat!(1000, f32_x512_avx2_fma_cosine, &x, &y));
    });
}

#[cfg(feature = "nightly")]
fn benchmark_dangerous_avx512_nofma_impls(c: &mut Criterion) {
    c.bench_function("cosine avx512 x1024 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, f32_x1024_avx512_nofma_cosine, &x, &y));
    });
    c.bench_function("cosine avx512 x768 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(768);
        b.iter(|| repeat!(1000, f32_x768_avx512_nofma_cosine, &x, &y));
    });
    c.bench_function("cosine avx512 x512 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(512);
        b.iter(|| repeat!(1000, f32_x512_avx512_nofma_cosine, &x, &y));
    });
}

#[cfg(feature = "nightly")]
fn benchmark_dangerous_avx512_fma_impls(c: &mut Criterion) {
    c.bench_function("cosine avx512 x1024 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, f32_x1024_avx512_fma_cosine, &x, &y));
    });
    c.bench_function("cosine avx512 x768 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(768);
        b.iter(|| repeat!(1000, f32_x768_avx512_fma_cosine, &x, &y));
    });
    c.bench_function("cosine avx512 x512 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(512);
        b.iter(|| repeat!(1000, f32_x512_avx512_fma_cosine, &x, &y));
    });
}

fn benchmark_dangerous_fallback_nofma_impls(c: &mut Criterion) {
    c.bench_function("cosine fallback x1024 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, f32_x1024_fallback_nofma_cosine, &x, &y));
    });
    c.bench_function("cosine fallback x768 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(768);
        b.iter(|| repeat!(1000, f32_x768_fallback_nofma_cosine, &x, &y));
    });
    c.bench_function("cosine fallback x512 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(512);
        b.iter(|| repeat!(1000, f32_x512_fallback_nofma_cosine, &x, &y));
    });
}

#[cfg(feature = "nightly")]
fn benchmark_dangerous_fallback_fma_impls(c: &mut Criterion) {
    c.bench_function("cosine fallback x1024 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, f32_x1024_fallback_fma_cosine, &x, &y));
    });
    c.bench_function("cosine fallback x768 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(768);
        b.iter(|| repeat!(1000, f32_x768_fallback_fma_cosine, &x, &y));
    });
    c.bench_function("cosine fallback x512 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(512);
        b.iter(|| repeat!(1000, f32_x512_fallback_fma_cosine, &x, &y));
    });
}

#[cfg(feature = "nightly")]
criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(30))
        .sample_size(250)
        .warm_up_time(Duration::from_secs(10));
    targets =
        benchmark_3rd_party_impls,
        benchmark_dangerous_avx2_fma_impls,
        benchmark_dangerous_avx2_nofma_impls,
        benchmark_dangerous_avx512_fma_impls,
        benchmark_dangerous_avx512_nofma_impls,
        benchmark_dangerous_fallback_fma_impls,
        benchmark_dangerous_fallback_nofma_impls,
);
#[cfg(not(feature = "nightly"))]
criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(30))
        .sample_size(250)
        .warm_up_time(Duration::from_secs(10));
    targets =
        benchmark_3rd_party_impls,
        benchmark_dangerous_avx2_nofma_impls,
        benchmark_dangerous_fallback_nofma_impls,
);
criterion_main!(benches);

fn ndarray_cosine(a: &ndarray::Array1<f32>, b: &ndarray::Array1<f32>) -> f32 {
    let dot_product = a.dot(b);
    let norm_x = a.dot(a);
    let norm_y = b.dot(b);

    if norm_x == 0.0 && norm_y == 0.0 {
        0.0
    } else if norm_x == 0.0 || norm_y == 0.0 {
        1.0
    } else {
        1.0 - (dot_product / (norm_x * norm_y).sqrt())
    }
}

fn simsimd_cosine(a: &[f32], b: &[f32]) -> f32 {
    f32::cosine(a, b).unwrap_or_default() as f32
}
