use crate::math::{FastMath, Math, StdMath};
use crate::danger::utils::rollup_scalar_x8;


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
pub unsafe fn f32_x1024_fallback_nofma_dot(x: &[f32], y: &[f32]) -> f32 {
    fallback_dot::<StdMath, 1024>(x, y)
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
pub unsafe fn f32_x1024_fallback_fma_dot(x: &[f32], y: &[f32]) -> f32 {
    fallback_dot::<FastMath, 1024>(x, y)
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
pub unsafe fn f32_x768_fallback_fma_dot(x: &[f32], y: &[f32]) -> f32 {
    fallback_dot::<FastMath, 768>(x, y)
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
pub unsafe fn f32_x512_fallback_fma_dot(x: &[f32], y: &[f32]) -> f32 {
    fallback_dot::<FastMath, 512>(x, y)
}

#[inline]
unsafe fn fallback_dot<M: Math, const DIMS: usize>(
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
        let a1 = *x.get_unchecked(i);
        let a2 = *x.get_unchecked(i + 1);
        let a3 = *x.get_unchecked(i + 2);
        let a4 = *x.get_unchecked(i + 3);
        let a5 = *x.get_unchecked(i + 4);
        let a6 = *x.get_unchecked(i + 5);
        let a7 = *x.get_unchecked(i + 6);
        let a8 = *x.get_unchecked(i + 7);

        let b1 = *y.get_unchecked(i);
        let b2 = *y.get_unchecked(i + 1);
        let b3 = *y.get_unchecked(i + 2);
        let b4 = *y.get_unchecked(i + 3);
        let b5 = *y.get_unchecked(i + 4);
        let b6 = *y.get_unchecked(i + 5);
        let b7 = *y.get_unchecked(i + 6);
        let b8 = *y.get_unchecked(i + 7);

        acc1 = M::add(acc1, M::mul(a1, b1));
        acc2 = M::add(acc2, M::mul(a2, b2));
        acc3 = M::add(acc3, M::mul(a3, b3));
        acc4 = M::add(acc4, M::mul(a4, b4));
        acc5 = M::add(acc5, M::mul(a5, b5));
        acc6 = M::add(acc6, M::mul(a6, b6));
        acc7 = M::add(acc7, M::mul(a7, b7));
        acc8 = M::add(acc8, M::mul(a8, b8));

        i += 8;
    }

    rollup_scalar_x8::<M>(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}