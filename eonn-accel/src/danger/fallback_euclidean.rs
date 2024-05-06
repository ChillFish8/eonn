use crate::danger::utils::rollup_scalar_x8;
use crate::math::*;

#[inline]
/// Computes the squared Euclidean distance of two `f32` vectors.
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
pub unsafe fn f32_xany_fallback_nofma_euclidean(x: &[f32], y: &[f32]) -> f32 {
    fallback_euclidean::<StdMath>(x, y)
}

#[cfg(feature = "nightly")]
#[inline]
/// Computes the squared Euclidean distance of two `f32` vectors.
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
pub unsafe fn f32_xany_fallback_fma_euclidean(x: &[f32], y: &[f32]) -> f32 {
    fallback_euclidean::<FastMath>(x, y)
}

#[inline]
unsafe fn fallback_euclidean<M: Math>(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(
        y.len(),
        x.len(),
        "Improper implementation detected, vectors must match in size"
    );

    let mut offset_from = x.len() % 8;

    // We do this manual unrolling to allow the compiler to vectorize
    // the loop and avoid some branching even if we're not doing it explicitly.
    // This made a significant difference in benchmarking ~4-8x
    let mut acc1 = 0.0;
    let mut acc2 = 0.0;
    let mut acc3 = 0.0;
    let mut acc4 = 0.0;
    let mut acc5 = 0.0;
    let mut acc6 = 0.0;
    let mut acc7 = 0.0;
    let mut acc8 = 0.0;

    if offset_from != 0 {
        for i in 0..offset_from {
            let x = *x.get_unchecked(i);
            let y = *y.get_unchecked(i);

            let diff = M::sub(x, y);
            acc1 = M::add(acc1, M::mul(diff, diff));
        }
    }

    while offset_from < x.len() {
        let x1 = *x.get_unchecked(offset_from);
        let x2 = *x.get_unchecked(offset_from + 1);
        let x3 = *x.get_unchecked(offset_from + 2);
        let x4 = *x.get_unchecked(offset_from + 3);
        let x5 = *x.get_unchecked(offset_from + 4);
        let x6 = *x.get_unchecked(offset_from + 5);
        let x7 = *x.get_unchecked(offset_from + 6);
        let x8 = *x.get_unchecked(offset_from + 7);

        let y1 = *y.get_unchecked(offset_from);
        let y2 = *y.get_unchecked(offset_from + 1);
        let y3 = *y.get_unchecked(offset_from + 2);
        let y4 = *y.get_unchecked(offset_from + 3);
        let y5 = *y.get_unchecked(offset_from + 4);
        let y6 = *y.get_unchecked(offset_from + 5);
        let y7 = *y.get_unchecked(offset_from + 6);
        let y8 = *y.get_unchecked(offset_from + 7);

        let diff1 = M::sub(x1, y1);
        let diff2 = M::sub(x2, y2);
        let diff3 = M::sub(x3, y3);
        let diff4 = M::sub(x4, y4);
        let diff5 = M::sub(x5, y5);
        let diff6 = M::sub(x6, y6);
        let diff7 = M::sub(x7, y7);
        let diff8 = M::sub(x8, y8);

        acc1 = M::add(acc1, M::mul(diff1, diff1));
        acc2 = M::add(acc2, M::mul(diff2, diff2));
        acc3 = M::add(acc3, M::mul(diff3, diff3));
        acc4 = M::add(acc4, M::mul(diff4, diff4));
        acc5 = M::add(acc5, M::mul(diff5, diff5));
        acc6 = M::add(acc6, M::mul(diff6, diff6));
        acc7 = M::add(acc7, M::mul(diff7, diff7));
        acc8 = M::add(acc8, M::mul(diff8, diff8));

        offset_from += 8;
    }

    rollup_scalar_x8::<M>(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_is_close, get_sample_vectors, simple_euclidean};

    #[test]
    fn test_xany_nofma_euclidean() {
        let (x, y) = get_sample_vectors(127);
        let dist = unsafe { f32_xany_fallback_nofma_euclidean(&x, &y) };
        assert_is_close(dist, simple_euclidean(&x, &y));
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_xany_fma_euclidean() {
        let (x, y) = get_sample_vectors(127);
        let dist = unsafe { f32_xany_fallback_fma_euclidean(&x, &y) };
        assert_is_close(dist, simple_euclidean(&x, &y));
    }
}
