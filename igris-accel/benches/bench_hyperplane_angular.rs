use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use igris_accel::danger::*;

mod utils;

fn benchmark_dangerous_avx2_nofma_impls(c: &mut Criterion) {
    c.bench_function("angular_hyperplane avx2 x1024 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, f32_x1024_avx2_nofma_angular_hyperplane, &x, &y));
    });
    c.bench_function("angular_hyperplane avx2 x768 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(768);
        b.iter(|| repeat!(1000, f32_x768_avx2_nofma_angular_hyperplane, &x, &y));
    });
    c.bench_function("angular_hyperplane avx2 x512 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(512);
        b.iter(|| repeat!(1000, f32_x512_avx2_nofma_angular_hyperplane, &x, &y));
    });
}

#[cfg(feature = "nightly")]
fn benchmark_dangerous_avx2_fma_impls(c: &mut Criterion) {
    c.bench_function("angular_hyperplane avx2 x1024 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, f32_x1024_avx2_fma_angular_hyperplane, &x, &y));
    });
    c.bench_function("angular_hyperplane avx2 x768 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(768);
        b.iter(|| repeat!(1000, f32_x768_avx2_fma_angular_hyperplane, &x, &y));
    });
    c.bench_function("angular_hyperplane avx2 x512 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(512);
        b.iter(|| repeat!(1000, f32_x512_avx2_fma_angular_hyperplane, &x, &y));
    });
}

#[cfg(feature = "nightly")]
fn benchmark_dangerous_avx512_nofma_impls(c: &mut Criterion) {
    c.bench_function("angular_hyperplane avx512 x1024 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, f32_x1024_avx512_nofma_angular_hyperplane, &x, &y));
    });
    c.bench_function("angular_hyperplane avx512 x768 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(768);
        b.iter(|| repeat!(1000, f32_x768_avx512_nofma_angular_hyperplane, &x, &y));
    });
    c.bench_function("angular_hyperplane avx512 x512 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(512);
        b.iter(|| repeat!(1000, f32_x512_avx512_nofma_angular_hyperplane, &x, &y));
    });
}

#[cfg(feature = "nightly")]
fn benchmark_dangerous_avx512_fma_impls(c: &mut Criterion) {
    c.bench_function("angular_hyperplane avx512 x1024 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, f32_x1024_avx512_fma_angular_hyperplane, &x, &y));
    });
    c.bench_function("angular_hyperplane avx512 x768 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(768);
        b.iter(|| repeat!(1000, f32_x768_avx512_fma_angular_hyperplane, &x, &y));
    });
    c.bench_function("angular_hyperplane avx512 x512 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(512);
        b.iter(|| repeat!(1000, f32_x512_avx512_fma_angular_hyperplane, &x, &y));
    });
}

fn benchmark_dangerous_fallback_nofma_impls(c: &mut Criterion) {
    c.bench_function("angular_hyperplane fallback x1024 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, f32_x1024_fallback_nofma_angular_hyperplane, &x, &y));
    });
    c.bench_function("angular_hyperplane fallback x768 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(768);
        b.iter(|| repeat!(1000, f32_x768_fallback_nofma_angular_hyperplane, &x, &y));
    });
    c.bench_function("angular_hyperplane fallback x512 nofma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(512);
        b.iter(|| repeat!(1000, f32_x512_fallback_nofma_angular_hyperplane, &x, &y));
    });
}

#[cfg(feature = "nightly")]
fn benchmark_dangerous_fallback_fma_impls(c: &mut Criterion) {
    c.bench_function("angular_hyperplane fallback x1024 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, f32_x1024_fallback_fma_angular_hyperplane, &x, &y));
    });
    c.bench_function("angular_hyperplane fallback x768 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(768);
        b.iter(|| repeat!(1000, f32_x768_fallback_fma_angular_hyperplane, &x, &y));
    });
    c.bench_function("angular_hyperplane fallback x512 fma", |b| unsafe {
        let (x, y) = utils::get_sample_vectors(512);
        b.iter(|| repeat!(1000, f32_x512_fallback_fma_angular_hyperplane, &x, &y));
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
        benchmark_dangerous_avx2_nofma_impls,
        benchmark_dangerous_fallback_nofma_impls,
);
criterion_main!(benches);
