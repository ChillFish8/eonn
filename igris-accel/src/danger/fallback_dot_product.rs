use crate::danger::utils::rollup_scalar_x8;
use crate::math::*;

#[inline]
/// Computes the dot product of two `[f32; 1024]` vectors.
///
/// These are fallback routines, they are designed to be optimized
/// by the compiler only, in areas where manually optimized routines
/// are unable to run due to lack of CPU features.
///
/// # Safety
///
/// Vectors **MUST** be `1024` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x1024_fallback_nofma_dot(x: &[f32], y: &[f32]) -> f32 {
    fallback_dot::<StdMath, 1024>(x, y)
}

#[inline]
/// Computes the dot product of two `[f32; 768]` vectors.
///
/// These are fallback routines, they are designed to be optimized
/// by the compiler only, in areas where manually optimized routines
/// are unable to run due to lack of CPU features.
///
/// # Safety
///
/// Vectors **MUST** be `768` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x768_fallback_nofma_dot(x: &[f32], y: &[f32]) -> f32 {
    fallback_dot::<StdMath, 768>(x, y)
}

#[inline]
/// Computes the dot product of two `[f32; 512]` vectors.
///
/// These are fallback routines, they are designed to be optimized
/// by the compiler only, in areas where manually optimized routines
/// are unable to run due to lack of CPU features.
///
/// # Safety
///
/// Vectors **MUST** be `512` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x512_fallback_nofma_dot(x: &[f32], y: &[f32]) -> f32 {
    fallback_dot::<StdMath, 512>(x, y)
}

#[cfg(feature = "nightly")]
#[inline]
/// Computes the dot product of two `[f32; 1024]` vectors.
///
/// These are fallback routines, they are designed to be optimized
/// by the compiler only, in areas where manually optimized routines
/// are unable to run due to lack of CPU features.
///
/// # Safety
///
/// Vectors **MUST** be `1024` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x1024_fallback_fma_dot(x: &[f32], y: &[f32]) -> f32 {
    fallback_dot::<FastMath, 1024>(x, y)
}

#[cfg(feature = "nightly")]
#[inline]
/// Computes the dot product of two `[f32; 768]` vectors.
///
/// These are fallback routines, they are designed to be optimized
/// by the compiler only, in areas where manually optimized routines
/// are unable to run due to lack of CPU features.
///
/// # Safety
///
/// Vectors **MUST** be `768` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x768_fallback_fma_dot(x: &[f32], y: &[f32]) -> f32 {
    fallback_dot::<FastMath, 768>(x, y)
}

#[cfg(feature = "nightly")]
#[inline]
/// Computes the dot product of two `[f32; 512]` vectors.
///
/// These are fallback routines, they are designed to be optimized
/// by the compiler only, in areas where manually optimized routines
/// are unable to run due to lack of CPU features.
///
/// # Safety
///
/// Vectors **MUST** be `512` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x512_fallback_fma_dot(x: &[f32], y: &[f32]) -> f32 {
    fallback_dot::<FastMath, 512>(x, y)
}

#[inline]
pub(super) unsafe fn fallback_dot<M: Math, const DIMS: usize>(
    x: &[f32],
    y: &[f32],
) -> f32 {
    debug_assert_eq!(
        y.len(),
        DIMS,
        "Improper implementation detected, vectors must match constant"
    );
    debug_assert_eq!(
        x.len(),
        DIMS,
        "Improper implementation detected, vectors must match constant"
    );
    debug_assert_eq!(
        DIMS % 8,
        0,
        "DIMS must be able to fit entirely into chunks of 8 lanes."
    );

    let mut i = 0;

    // We do this manual unrolling to allow the compiler to vectorize
    // the loop and avoid some branching even if we're not doing it explicitly.
    // This made a significant difference in benchmarking ~8x
    let mut acc1 = 0.0;
    let mut acc2 = 0.0;
    let mut acc3 = 0.0;
    let mut acc4 = 0.0;
    let mut acc5 = 0.0;
    let mut acc6 = 0.0;
    let mut acc7 = 0.0;
    let mut acc8 = 0.0;

    while i < x.len() {
        let x1 = *x.get_unchecked(i);
        let x2 = *x.get_unchecked(i + 1);
        let x3 = *x.get_unchecked(i + 2);
        let x4 = *x.get_unchecked(i + 3);
        let x5 = *x.get_unchecked(i + 4);
        let x6 = *x.get_unchecked(i + 5);
        let x7 = *x.get_unchecked(i + 6);
        let x8 = *x.get_unchecked(i + 7);

        let y1 = *y.get_unchecked(i);
        let y2 = *y.get_unchecked(i + 1);
        let y3 = *y.get_unchecked(i + 2);
        let y4 = *y.get_unchecked(i + 3);
        let y5 = *y.get_unchecked(i + 4);
        let y6 = *y.get_unchecked(i + 5);
        let y7 = *y.get_unchecked(i + 6);
        let y8 = *y.get_unchecked(i + 7);

        acc1 = M::add(acc1, M::mul(x1, y1));
        acc2 = M::add(acc2, M::mul(x2, y2));
        acc3 = M::add(acc3, M::mul(x3, y3));
        acc4 = M::add(acc4, M::mul(x4, y4));
        acc5 = M::add(acc5, M::mul(x5, y5));
        acc6 = M::add(acc6, M::mul(x6, y6));
        acc7 = M::add(acc7, M::mul(x7, y7));
        acc8 = M::add(acc8, M::mul(x8, y8));

        i += 8;
    }

    rollup_scalar_x8::<M>(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}
