//! Specialized size-aware vector operations.

use std::marker::PhantomData;

mod arch;
mod float32;

pub use arch::{Avx2, Avx512, Fallback, Fma, NoFma};

/// A specialised vector wrapper
pub struct Vector<Arch, Dims, T = f32, Fma = NoFma>(pub Vec<T>, pub PhantomData<(Arch, Dims, Fma)>);

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
    /// Calculates the Euclidean distance.
    unsafe fn euclidean(&self, other: &Self) -> f32;
}
