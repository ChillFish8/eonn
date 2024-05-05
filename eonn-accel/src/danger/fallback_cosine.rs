#[cfg(feature = "nightly")]
use crate::danger::f32_xany_fallback_fma_dot;
use crate::danger::{cosine, f32_xany_fallback_nofma_dot};
use crate::math::*;

#[inline]
/// Computes the cosine distance of two `f32` vectors.
///
/// These are fallback routines, they are designed to be optimized
/// by the compiler only, in areas where manually optimized routines
/// are unable to run due to lack of CPU features.
///
/// # Safety
///
/// Vectors **MUST** be equal length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xany_fallback_nofma_cosine(x: &[f32], y: &[f32]) -> f32 {
    let norm_x = f32_xany_fallback_nofma_dot(x, x);
    let norm_y = f32_xany_fallback_nofma_dot(y, y);
    let dot_product = f32_xany_fallback_nofma_dot(x, y);
    cosine::<StdMath>(dot_product, norm_x, norm_y)
}

#[cfg(feature = "nightly")]
#[inline]
/// Computes the cosine distance of two `f32` vectors.
///
/// These are fallback routines, they are designed to be optimized
/// by the compiler only, in areas where manually optimized routines
/// are unable to run due to lack of CPU features.
///
/// # Safety
///
/// Vectors **MUST** be equal length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xany_fallback_fma_cosine(x: &[f32], y: &[f32]) -> f32 {
    let norm_x = f32_xany_fallback_fma_dot(x, x);
    let norm_y = f32_xany_fallback_fma_dot(y, y);
    let dot_product = f32_xany_fallback_fma_dot(x, y);
    cosine::<FastMath>(dot_product, norm_x, norm_y)
}
