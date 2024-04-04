use std::hint::black_box;
use std::marker::PhantomData;
use simsimd::SpatialSimilarity;
use igris_accel::math::{FastMath, Math};
use igris_accel::vector_ops::{Vector, Avx2, X1024, DistanceOps, NoFma, Fma};
use criterion::{criterion_main, Criterion, criterion_group};

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

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("dot autovec 1024 auto", |b| {
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }
        
        b.iter(|| basic_dot(black_box(&v1), black_box(&v2)))
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