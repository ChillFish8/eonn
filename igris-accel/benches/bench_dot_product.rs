use std::hint::black_box;
use std::marker::PhantomData;
use simsimd::SpatialSimilarity;
use igris_accel::math::{FastMath, Math};
use igris_accel::vector_ops::{Vector, Avx2, X1024, DistanceOps, NoFma, Fma};
use divan::Bencher;

fn main() {
    divan::main();
}

#[divan::bench]
fn bench_llvmauto_x1024_auto(bencher: Bencher) {
    let [v1, v2] = random_vectors();

    bencher.bench(|| basic_dot(black_box(&v1), black_box(&v2)));
}


#[divan::bench]
fn bench_simsimd_x1024_auto(bencher: Bencher) {
    let [v1, v2] = random_vectors();

    bencher.bench(|| simsimd_dot(black_box(&v1), black_box(&v2)));
}

#[divan::bench]
fn bench_avx2_x1024_nofma(bencher: Bencher) {
    let [v1, v2] = random_vectors();
    
    let v1 = Vector::<Avx2, X1024, f32, NoFma>(v1, PhantomData);
    let v2 = Vector::<Avx2, X1024, f32, NoFma>(v2, PhantomData);
    
    bencher.bench(|| dot(black_box(&v1), black_box(&v2)));
}

#[divan::bench]
fn bench_avx2_x1024_fma(bencher: Bencher) {
    let [v1, v2] = random_vectors();

    let v1 = Vector::<Avx2, X1024, f32, Fma>(v1, PhantomData);
    let v2 = Vector::<Avx2, X1024, f32, Fma>(v2, PhantomData);

    bencher.bench(|| dot(black_box(&v1), black_box(&v2)));
}



fn dot<T: DistanceOps>(a: &T, b: &T) -> f32 {
    unsafe { a.dot(b) }
}

fn basic_dot(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut acc = 0.0;

    for i in 0..len {
        let a = unsafe { a.get_unchecked(i) };
        let b = unsafe { b.get_unchecked(i) };
        let r = FastMath::mul(*a, *b);
        acc = FastMath::add(acc, r)
    }

    acc
}

fn simsimd_dot(a: &[f32], b: &[f32]) -> f32 {
    f32::dot(a, b).unwrap_or_default() as f32
}

fn random_vectors() -> [Vec<f32>; 2] {
    let mut v1 = Vec::with_capacity(1024);
    let mut v2 = Vec::with_capacity(1024);
    for _ in 0..1024 {
        v1.push(rand::random());
        v2.push(rand::random());
    }

    [v1, v2]
}
