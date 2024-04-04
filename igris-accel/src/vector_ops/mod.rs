//! Specialized size-aware vector operations.

use std::marker::PhantomData;
use std::ops::Deref;

mod arch;
mod dims;
mod float32;

pub use arch::*;
pub use dims::{X1024, X512, X768};

use crate::vector_ops::dims::Dim;

/// Core vector space distance calculations
pub trait DistanceOps {
    /// Calculates the dot product distance.
    ///
    /// # Safety
    /// These operations can all assume that all values within are finite.
    /// These methods may also perform unsafe intrinsics like SIMD instructions.
    unsafe fn dot(&self, other: &Self) -> f32;
    /// Calculates the cosine distance.
    ///
    /// # Safety
    /// These operations can all assume that all values within are finite.
    /// These methods may also perform unsafe intrinsics like SIMD instructions.
    unsafe fn cosine(&self, other: &Self) -> f32;
    /// Calculates the **squared** Euclidean distance.
    ///
    /// # Safety
    /// These operations can all assume that all values within are finite.
    /// These methods may also perform unsafe intrinsics like SIMD instructions.
    unsafe fn euclidean(&self, other: &Self) -> f32;
}

/// A view of a given vector.
pub struct VectorView<'a, Arch, Dims, T = f32, Fma = NoFma>(
    &'a [T],
    PhantomData<(Arch, Dims, Fma)>,
);

/// A specialised vector wrapper
pub struct Vector<Arch, Dims, T = f32, Fma = NoFma>(
    Vec<T>,
    PhantomData<(Arch, Dims, Fma)>,
);

impl<Arch, Dims: Dim, T, Fma> Vector<Arch, Dims, T, Fma> {
    #[inline]
    /// Creates a new [Vector] using the given data.
    ///
    /// # Safety
    ///
    /// The length of the provided vec must equal the number of dimensions
    /// / `Dim::size()` otherwise out of bounds access and other numerical
    /// errors can occur during operations leading to immediate UB.    ///
    pub unsafe fn from_vec_unchecked(data: Vec<T>) -> Self {
        Self(data, PhantomData)
    }

    #[inline]
    /// Get an immutable view of the vector data.
    pub fn view(&self) -> VectorView<Arch, Dims, T, Fma> {
        VectorView(&self.0, PhantomData)
    }
}

impl<Arch, Dims, T: Clone, Fma> Clone for Vector<Arch, Dims, T, Fma> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<Arch, Dims: Dim, T, Fma> AsRef<[T]> for Vector<Arch, Dims, T, Fma> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<Arch, Dims: Dim, T, Fma> Deref for Vector<Arch, Dims, T, Fma> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.0.as_ref()
    }
}
