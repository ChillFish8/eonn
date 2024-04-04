//! Specialized size-aware vector operations.

use std::marker::PhantomData;
use std::ops::Deref;

mod arch;
mod float32;

pub use arch::{Avx2, Avx512, Fallback, Fma, NoFma};

/// A view of a given vector.
pub struct VectorView<'a, Arch, Dims, T = f32, Fma = NoFma>(pub &'a [T], pub PhantomData<(Arch, Dims, Fma)>);

/// A specialised vector wrapper
pub struct Vector<Arch, Dims, T = f32, Fma = NoFma>(pub Vec<T>, pub PhantomData<(Arch, Dims, Fma)>);  // TODO: Make this data private

impl<A, D, T: Clone, F> Clone for Vector<A, D, T, F> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<A, D, T, F> Vector<A, D, T, F> {
    #[inline]
    /// Get an immutable view of the vector data.
    pub fn view(&self) -> VectorView<A, D, T, F> {
        VectorView(&self.0, PhantomData)
    }
}


#[derive(Clone)]
/// A specialised matrix wrapper for optimized matrix ops.
pub struct Matrix<V>(pub Vec<V>);

impl<V: DistanceOps> Matrix<V> {
    #[inline]
    /// Computes the dot products of all the matrix vertices and the given query.
    pub fn dot(&self, q: &V) -> Vec<f32> {
        unsafe { self.dot_inner(q) }
    }
    
    unsafe fn dot_inner(&self, q: &V) -> Vec<f32> {
        let mut results = Vec::with_capacity(self.0.len());
        let chunks = self.0.chunks_exact(8);
        
        for chunk in chunks {
            let v1 = chunk.get_unchecked(0);
            let v2 = chunk.get_unchecked(1);
            let v3 = chunk.get_unchecked(2);
            let v4 = chunk.get_unchecked(3);
            let v5 = chunk.get_unchecked(4);
            let v6 = chunk.get_unchecked(5);
            let v7 = chunk.get_unchecked(6);
            let v8 = chunk.get_unchecked(7);

            let r1 = q.dot(v1);
            let r2 = q.dot(v2);
            let r3 = q.dot(v3);
            let r4 = q.dot(v4);
            let r5 = q.dot(v5);
            let r6 = q.dot(v6);
            let r7 = q.dot(v7);
            let r8 = q.dot(v8);

            results.push(r1);
            results.push(r2);
            results.push(r3);
            results.push(r4);
            results.push(r5);
            results.push(r6);
            results.push(r7);
            results.push(r8);
        }

        let remainder = self.0.chunks_exact(8).remainder();
        for vector in remainder {
            results.push(q.dot(vector));
        }

        results        
    }
}


/// Vector dimensions of 1024
pub struct X1024;
/// Vector dimensions of 768
pub struct X768;
/// Vector dimensions of 512
pub struct X512;

/// Core vector space distance calculations
pub trait DistanceOps {
    /// Calculates the dot product distance.
    unsafe fn dot(&self, other: &Self) -> f32;
    /// Calculates the cosine distance.
    unsafe fn cosine(&self, other: &Self) -> f32;
    /// Calculates the **squared** Euclidean distance.
    unsafe fn euclidean(&self, other: &Self) -> f32;
}
