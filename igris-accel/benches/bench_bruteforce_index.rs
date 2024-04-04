extern crate blas_src;

use std::hint::black_box;
use std::marker::PhantomData;

use ndarray::{Array1, Array2};
use criterion::{criterion_group, criterion_main, Criterion};
use igris_accel::vector_ops::{Avx2, DistanceOps, Fma, Matrix, Vector, X1024};


fn criterion_benchmark(c: &mut Criterion) {
    let mut query = Vec::new();
    for _ in 0..1024 {
        query.push(rand::random::<f32>());
    }

    let mut vectors = Vec::new();
    for _ in 0..128 {
        let mut v = Vec::new();
        for _ in 0..1024 {
            v.push(rand::random::<f32>());
        }
        vectors.push(v);
    }
    
    let mut ndarray_index = Vec::new();
    for v in vectors.iter() {
        ndarray_index.extend_from_slice(v);
    }
    let ndarray_index = Array2::from_shape_vec((128, 1024), ndarray_index).unwrap();
    
    let mut accel_index = Vec::new();
    for v in vectors {
        let v = Vector::<Avx2, X1024, f32, Fma>(v, PhantomData);
        accel_index.push(v);
    }
    let accel_index = Matrix(accel_index);

    c.bench_function("dot ndarray 128x1024", |b| {
        let query = Array1::from_shape_vec((1024,), query.clone()).unwrap();
        let ndarray_index = ndarray_index.clone();

        b.iter(|| {
            for _ in 0..1000 {
                let query = black_box(&query);
                let index = black_box(&ndarray_index);
                black_box(index.dot(query));
            }
        })
    });
    c.bench_function("dot igris-accel 128x1024", |b| {
        let query = Vector::<Avx2, X1024, f32, Fma>(query.clone(), PhantomData);
        let accel_index = accel_index.clone();
        
        b.iter(|| {
            for _ in 0..1000 {
                let query = black_box(&query);
                let index = black_box(&accel_index);
                black_box(index.dot(query));
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
