use std::hint::black_box;
use std::marker::PhantomData;
use igris_accel::vector_ops::{Vector, Avx2, X1024, DistanceOps};
use criterion::{criterion_group, criterion_main, Criterion};

fn dot<T: DistanceOps>(a: &T, b: &T) -> f32 {
    unsafe { a.dot(b) }
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("dot avx2 1024 nofma", |b| { 
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }
        
        let v1 = Vector::<Avx2, X1024, f32>(v1, PhantomData);
        let v2 = Vector::<Avx2, X1024, f32>(v2, PhantomData);
        
        b.iter(|| dot(black_box(&v1), black_box(&v2)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);