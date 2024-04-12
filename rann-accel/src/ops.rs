use crate::arch::*;
use crate::dims::*;

/// Safe spacial type operations.
pub trait SpacialOps: Sized {
    /// Returns the length of the vector.
    fn len() -> usize;
    /// Computes the dot product between self and another vector.
    fn dot(&self, other: &Self) -> f32;
    /// Computes the squared norm of the vector.
    fn squared_norm(&self) -> f32;
    /// Normalizes the vector.
    fn normalize(&mut self);
    /// Computes the dot product distance between self and another vector.
    fn dist_dot(&self, other: &Self) -> f32;
    /// Computes the cosine distance between self and another vector.
    fn dist_cosine(&self, other: &Self) -> f32;
    /// Computes the cosine distance between self and another vector.
    fn dist_squared_euclidean(&self, other: &Self) -> f32;
    /// Computes the angular hyperplane vector between self and another vector.
    fn angular_hyperplane(&self, other: &Self) -> Self;
    /// Computes the Euclidean hyperplane vector between self and another vector and
    /// returns the offset.
    fn euclidean_hyperplane(&self, other: &Self) -> (Self, f32);
}

/// A set of compute ops various archs and dims implement.
///
/// # Safety
/// All vectors must contain only finite values.
pub trait DangerousOps {
    /// Computes the dot product of the two provided vectors.
    /// 
    /// # Safety
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32;
    /// Computes the cosine distance of the two provided vectors.
    /// 
    /// # Safety
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32;
    /// Computes the squared Euclidean distance of the two provided vectors.
    /// 
    /// # Safety
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32;
    /// Computes the angular hyperplane to the two vector points.
    /// 
    /// # Safety
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32>;
    /// Computes the Euclidean hyperplane and hyperplane offset.
    /// 
    /// # Safety
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32);
    /// Computes the squared norm of the given vector.
    /// 
    /// # Safety
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn squared_norm(&self, x: &[f32]) -> f32;
}

impl DangerousOps for (X1024, Fallback) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x1024_fallback_nofma_dot(x, y)
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x1024_fallback_nofma_cosine(x, y)
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x1024_fallback_nofma_euclidean(x, y)
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        crate::danger::f32_x1024_fallback_nofma_angular_hyperplane(x, y)
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        crate::danger::f32_x1024_fallback_nofma_euclidean_hyperplane(x, y)
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        crate::danger::f32_x1024_fallback_nofma_dot(x, x)
    }
}

impl DangerousOps for (X768, Fallback) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x768_fallback_nofma_dot(x, y)
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x768_fallback_nofma_cosine(x, y)
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x768_fallback_nofma_euclidean(x, y)
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        crate::danger::f32_x768_fallback_nofma_angular_hyperplane(x, y)
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        crate::danger::f32_x768_fallback_nofma_euclidean_hyperplane(x, y)
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        crate::danger::f32_x768_fallback_nofma_dot(x, x)
    }
}

impl DangerousOps for (X512, Fallback) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x512_fallback_nofma_dot(x, y)
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x512_fallback_nofma_cosine(x, y)
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x512_fallback_nofma_euclidean(x, y)
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        crate::danger::f32_x512_fallback_nofma_angular_hyperplane(x, y)
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        crate::danger::f32_x512_fallback_nofma_euclidean_hyperplane(x, y)
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        crate::danger::f32_x512_fallback_nofma_dot(x, x)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl DangerousOps for (X1024, Avx2) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x1024_avx2_nofma_dot(x, y)
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x1024_avx2_nofma_cosine(x, y)
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x1024_avx2_nofma_euclidean(x, y)
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        crate::danger::f32_x1024_avx2_nofma_angular_hyperplane(x, y)
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        crate::danger::f32_x1024_avx2_nofma_euclidean_hyperplane(x, y)
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        crate::danger::f32_x1024_avx2_nofma_norm(x)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl DangerousOps for (X768, Avx2) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x768_avx2_nofma_dot(x, y)
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x768_avx2_nofma_cosine(x, y)
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x768_avx2_nofma_euclidean(x, y)
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        crate::danger::f32_x768_avx2_nofma_angular_hyperplane(x, y)
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        crate::danger::f32_x768_avx2_nofma_euclidean_hyperplane(x, y)
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        crate::danger::f32_x768_avx2_nofma_norm(x)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl DangerousOps for (X512, Avx2) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x512_avx2_nofma_dot(x, y)
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x512_avx2_nofma_cosine(x, y)
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x512_avx2_nofma_euclidean(x, y)
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        crate::danger::f32_x512_avx2_nofma_angular_hyperplane(x, y)
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        crate::danger::f32_x512_avx2_nofma_euclidean_hyperplane(x, y)
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        crate::danger::f32_x512_avx2_nofma_norm(x)
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
impl DangerousOps for (X1024, Avx512) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x1024_avx512_nofma_dot(x, y)
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x1024_avx512_nofma_cosine(x, y)
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x1024_avx512_nofma_euclidean(x, y)
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        crate::danger::f32_x1024_avx512_nofma_angular_hyperplane(x, y)
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        crate::danger::f32_x1024_avx512_nofma_euclidean_hyperplane(x, y)
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        crate::danger::f32_x1024_avx512_nofma_norm(x)
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
impl DangerousOps for (X768, Avx512) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x768_avx512_nofma_dot(x, y)
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x768_avx512_nofma_cosine(x, y)
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x768_avx512_nofma_euclidean(x, y)
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        crate::danger::f32_x768_avx512_nofma_angular_hyperplane(x, y)
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        crate::danger::f32_x768_avx512_nofma_euclidean_hyperplane(x, y)
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        crate::danger::f32_x768_avx512_nofma_norm(x)
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
impl DangerousOps for (X512, Avx512) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x512_avx512_nofma_dot(x, y)
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x512_avx512_nofma_cosine(x, y)
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x512_avx512_nofma_euclidean(x, y)
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        crate::danger::f32_x512_avx512_nofma_angular_hyperplane(x, y)
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        crate::danger::f32_x512_avx512_nofma_euclidean_hyperplane(x, y)
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        crate::danger::f32_x512_avx512_nofma_norm(x)
    }
}

#[cfg(feature = "nightly")]
impl DangerousOps for (X1024, (Fallback, Fma)) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x1024_fallback_fma_dot(x, y)
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x1024_fallback_fma_cosine(x, y)
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x1024_fallback_fma_euclidean(x, y)
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        crate::danger::f32_x1024_fallback_fma_angular_hyperplane(x, y)
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        crate::danger::f32_x1024_fallback_fma_euclidean_hyperplane(x, y)
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        crate::danger::f32_x1024_fallback_fma_dot(x, x)
    }
}

#[cfg(feature = "nightly")]
impl DangerousOps for (X768, (Fallback, Fma)) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x768_fallback_fma_dot(x, y)
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x768_fallback_fma_cosine(x, y)
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x768_fallback_fma_euclidean(x, y)
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        crate::danger::f32_x768_fallback_fma_angular_hyperplane(x, y)
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        crate::danger::f32_x768_fallback_fma_euclidean_hyperplane(x, y)
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        crate::danger::f32_x768_fallback_fma_dot(x, x)
    }
}

#[cfg(feature = "nightly")]
impl DangerousOps for (X512, (Fallback, Fma)) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x512_fallback_fma_dot(x, y)
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x512_fallback_fma_cosine(x, y)
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x512_fallback_fma_euclidean(x, y)
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        crate::danger::f32_x512_fallback_fma_angular_hyperplane(x, y)
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        crate::danger::f32_x512_fallback_fma_euclidean_hyperplane(x, y)
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        crate::danger::f32_x512_fallback_fma_dot(x, x)
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
impl DangerousOps for (X1024, (Avx2, Fma)) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x1024_avx2_fma_dot(x, y)
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x1024_avx2_fma_cosine(x, y)
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x1024_avx2_fma_euclidean(x, y)
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        crate::danger::f32_x1024_avx2_fma_angular_hyperplane(x, y)
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        crate::danger::f32_x1024_avx2_fma_euclidean_hyperplane(x, y)
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        crate::danger::f32_x1024_avx2_fma_norm(x)
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
impl DangerousOps for (X768, (Avx2, Fma)) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x768_avx2_fma_dot(x, y)
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x768_avx2_fma_cosine(x, y)
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x768_avx2_fma_euclidean(x, y)
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        crate::danger::f32_x768_avx2_fma_angular_hyperplane(x, y)
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        crate::danger::f32_x768_avx2_fma_euclidean_hyperplane(x, y)
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        crate::danger::f32_x768_avx2_fma_norm(x)
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
impl DangerousOps for (X512, (Avx2, Fma)) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x512_avx2_fma_dot(x, y)
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x512_avx2_fma_cosine(x, y)
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x512_avx2_fma_euclidean(x, y)
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        crate::danger::f32_x512_avx2_fma_angular_hyperplane(x, y)
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        crate::danger::f32_x512_avx2_fma_euclidean_hyperplane(x, y)
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        crate::danger::f32_x512_avx2_fma_norm(x)
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
impl DangerousOps for (X1024, (Avx512, Fma)) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x1024_avx512_fma_dot(x, y)
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x1024_avx512_fma_cosine(x, y)
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x1024_avx512_fma_euclidean(x, y)
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        crate::danger::f32_x1024_avx512_fma_angular_hyperplane(x, y)
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        crate::danger::f32_x1024_avx512_fma_euclidean_hyperplane(x, y)
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        crate::danger::f32_x1024_avx512_fma_norm(x)
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
impl DangerousOps for (X768, (Avx512, Fma)) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x768_avx512_fma_dot(x, y)
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x768_avx512_fma_cosine(x, y)
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x768_avx512_fma_euclidean(x, y)
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        crate::danger::f32_x768_avx512_fma_angular_hyperplane(x, y)
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        crate::danger::f32_x768_avx512_fma_euclidean_hyperplane(x, y)
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        crate::danger::f32_x768_avx512_fma_norm(x)
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
impl DangerousOps for (X512, (Avx512, Fma)) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x512_avx512_fma_dot(x, y)
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x512_avx512_fma_cosine(x, y)
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        crate::danger::f32_x512_avx512_fma_euclidean(x, y)
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        crate::danger::f32_x512_avx512_fma_angular_hyperplane(x, y)
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        crate::danger::f32_x512_avx512_fma_euclidean_hyperplane(x, y)
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        crate::danger::f32_x512_avx512_fma_norm(x)
    }
}

impl DangerousOps for (X1024, Auto) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x1024_avx2_nofma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x1024_avx2_fma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x1024_avx512_nofma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_x1024_avx512_fma_dot(x, y),
            SelectedArch::Fallback => crate::danger::f32_x1024_fallback_nofma_dot(x, y),
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => crate::danger::f32_x1024_fallback_fma_dot(x, y),
        }
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x1024_avx2_nofma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x1024_avx2_fma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x1024_avx512_nofma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_x1024_avx512_fma_cosine(x, y),
            SelectedArch::Fallback => {
                crate::danger::f32_x1024_fallback_nofma_cosine(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_x1024_fallback_fma_cosine(x, y)
            },
        }
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x1024_avx2_nofma_euclidean(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x1024_avx2_fma_euclidean(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_x1024_avx512_nofma_euclidean(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_x1024_avx512_fma_euclidean(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_x1024_fallback_nofma_euclidean(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_x1024_fallback_fma_euclidean(x, y)
            },
        }
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_x1024_avx2_nofma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_x1024_avx2_fma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_x1024_avx512_nofma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_x1024_avx512_fma_angular_hyperplane(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_x1024_fallback_nofma_angular_hyperplane(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_x1024_fallback_fma_angular_hyperplane(x, y)
            },
        }
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_x1024_avx2_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_x1024_avx2_fma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_x1024_avx512_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_x1024_avx512_fma_euclidean_hyperplane(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_x1024_fallback_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_x1024_fallback_fma_euclidean_hyperplane(x, y)
            },
        }
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x1024_avx2_nofma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x1024_avx2_fma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x1024_avx512_nofma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_x1024_avx512_fma_norm(x),
            SelectedArch::Fallback => crate::danger::f32_x1024_fallback_nofma_dot(x, x),
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => crate::danger::f32_x1024_fallback_fma_dot(x, x),
        }
    }
}

impl DangerousOps for (X768, Auto) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x768_avx2_nofma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x768_avx2_fma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x768_avx512_nofma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_x768_avx512_fma_dot(x, y),
            SelectedArch::Fallback => crate::danger::f32_x768_fallback_nofma_dot(x, y),
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => crate::danger::f32_x768_fallback_fma_dot(x, y),
        }
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x768_avx2_nofma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x768_avx2_fma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x768_avx512_nofma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_x768_avx512_fma_cosine(x, y),
            SelectedArch::Fallback => {
                crate::danger::f32_x768_fallback_nofma_cosine(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_x768_fallback_fma_cosine(x, y)
            },
        }
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x768_avx2_nofma_euclidean(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x768_avx2_fma_euclidean(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x768_avx512_nofma_euclidean(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_x768_avx512_fma_euclidean(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_x768_fallback_nofma_euclidean(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_x768_fallback_fma_euclidean(x, y)
            },
        }
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_x768_avx2_nofma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_x768_avx2_fma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_x768_avx512_nofma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_x768_avx512_fma_angular_hyperplane(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_x768_fallback_nofma_angular_hyperplane(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_x768_fallback_fma_angular_hyperplane(x, y)
            },
        }
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_x768_avx2_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_x768_avx2_fma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_x768_avx512_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_x768_avx512_fma_euclidean_hyperplane(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_x768_fallback_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_x768_fallback_fma_euclidean_hyperplane(x, y)
            },
        }
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x768_avx2_nofma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x768_avx2_fma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x768_avx512_nofma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_x768_avx512_fma_norm(x),
            SelectedArch::Fallback => crate::danger::f32_x768_fallback_nofma_dot(x, x),
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => crate::danger::f32_x768_fallback_fma_dot(x, x),
        }
    }
}

impl DangerousOps for (X512, Auto) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x512_avx2_nofma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x512_avx2_fma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x512_avx512_nofma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_x512_avx512_fma_dot(x, y),
            SelectedArch::Fallback => crate::danger::f32_x512_fallback_nofma_dot(x, y),
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => crate::danger::f32_x512_fallback_fma_dot(x, y),
        }
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x512_avx2_nofma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x512_avx2_fma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x512_avx512_nofma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_x512_avx512_fma_cosine(x, y),
            SelectedArch::Fallback => {
                crate::danger::f32_x512_fallback_nofma_cosine(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_x512_fallback_fma_cosine(x, y)
            },
        }
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x512_avx2_nofma_euclidean(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x512_avx2_fma_euclidean(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x512_avx512_nofma_euclidean(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_x512_avx512_fma_euclidean(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_x512_fallback_nofma_euclidean(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_x512_fallback_fma_euclidean(x, y)
            },
        }
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_x512_avx2_nofma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_x512_avx2_fma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_x512_avx512_nofma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_x512_avx512_fma_angular_hyperplane(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_x512_fallback_nofma_angular_hyperplane(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_x512_fallback_fma_angular_hyperplane(x, y)
            },
        }
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_x512_avx2_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_x512_avx2_fma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_x512_avx512_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_x512_avx512_fma_euclidean_hyperplane(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_x512_fallback_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_x512_fallback_fma_euclidean_hyperplane(x, y)
            },
        }
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x512_avx2_nofma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x512_avx2_fma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x512_avx512_nofma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_x512_avx512_fma_norm(x),
            SelectedArch::Fallback => crate::danger::f32_x512_fallback_nofma_dot(x, x),
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => crate::danger::f32_x512_fallback_fma_dot(x, x),
        }
    }
}
