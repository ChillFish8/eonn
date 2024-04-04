//! Specialized size-aware vector operations.

use std::marker::PhantomData;

mod arch;
mod dims;
mod float32;

pub use arch::{Avx2, Avx512, Fallback, Fma, NoFma};
pub use dims::{X1024, X512, X768};

/// Core vector space distance calculations
pub trait DistanceOps {
    /// Calculates the dot product distance.
    unsafe fn dot(&self, other: &Self) -> f32;
    /// Calculates the cosine distance.
    unsafe fn cosine(&self, other: &Self) -> f32;
    /// Calculates the **squared** Euclidean distance.
    unsafe fn euclidean(&self, other: &Self) -> f32;
}

#[doc(hidden)]
pub trait DimSlice<D, T> {
    fn as_slice(&self) -> &[T];
}

/// A view of a given vector.
pub struct VectorView<'a, Arch, Dims, T = f32, Fma = NoFma>(
    pub &'a [T],
    pub PhantomData<(Arch, Dims, Fma)>,
);

impl<'a, A, D, T, F> DimSlice<D, T> for VectorView<'a, A, D, T, F> {
    fn as_slice(&self) -> &[T] {
        self.0
    }
}

/// A specialised vector wrapper
pub struct Vector<Arch, Dims, T = f32, Fma = NoFma>(
    pub Vec<T>,
    pub PhantomData<(Arch, Dims, Fma)>,
); // TODO: Make this data private

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

impl<A, D, T, F> DimSlice<D, T> for Vector<A, D, T, F> {
    fn as_slice(&self) -> &[T] {
        &self.0
    }
}
