use crate::danger::rollup_scalar_x8;
use crate::math::*;

#[inline]
/// Computes the Euclidean hyperplane of two `[f32; 1024]` vectors.
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
pub unsafe fn f32_x1024_fallback_nofma_euclidean_hyperplane(
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, f32) {
    fallback_euclidean_hyperplane::<StdMath, 1024>(x, y)
}

#[inline]
/// Computes the Euclidean hyperplane of two `[f32; 768]` vectors
/// and the offset from origin.
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
pub unsafe fn f32_x768_fallback_nofma_euclidean_hyperplane(
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, f32) {
    fallback_euclidean_hyperplane::<StdMath, 768>(x, y)
}

#[inline]
/// Computes theeuclidean hyperplane of two `[f32; 512]` vectors
/// and the offset from origin.
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
pub unsafe fn f32_x512_fallback_nofma_euclidean_hyperplane(
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, f32) {
    fallback_euclidean_hyperplane::<StdMath, 512>(x, y)
}

#[cfg(feature = "nightly")]
#[inline]
/// Computes the Euclidean hyperplane of two `[f32; 1024]` vectors
/// and the offset from origin.
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
pub unsafe fn f32_x1024_fallback_fma_euclidean_hyperplane(
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, f32) {
    fallback_euclidean_hyperplane::<FastMath, 1024>(x, y)
}

#[cfg(feature = "nightly")]
#[inline]
/// Computes the Euclidean hyperplane of two `[f32; 768]` vectors
/// and the offset from origin.
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
pub unsafe fn f32_x768_fallback_fma_euclidean_hyperplane(
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, f32) {
    fallback_euclidean_hyperplane::<FastMath, 768>(x, y)
}

#[cfg(feature = "nightly")]
#[inline]
/// Computes the Euclidean hyperplane of two `[f32; 512]` vectors
/// and the offset from origin.
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
pub unsafe fn f32_x512_fallback_fma_euclidean_hyperplane(
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, f32) {
    fallback_euclidean_hyperplane::<FastMath, 512>(x, y)
}

unsafe fn fallback_euclidean_hyperplane<M: Math, const DIMS: usize>(
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, f32) {
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

    let mut hyperplane = Vec::with_capacity(DIMS);

    let mut offset_acc1 = 0.0;
    let mut offset_acc2 = 0.0;
    let mut offset_acc3 = 0.0;
    let mut offset_acc4 = 0.0;
    let mut offset_acc5 = 0.0;
    let mut offset_acc6 = 0.0;
    let mut offset_acc7 = 0.0;
    let mut offset_acc8 = 0.0;

    let mut i = 0;
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

        let diff1 = M::sub(x1, y1);
        let diff2 = M::sub(x2, y2);
        let diff3 = M::sub(x3, y3);
        let diff4 = M::sub(x4, y4);
        let diff5 = M::sub(x5, y5);
        let diff6 = M::sub(x6, y6);
        let diff7 = M::sub(x7, y7);
        let diff8 = M::sub(x8, y8);

        offset_acc1 = M::add(offset_acc1, M::mul(diff1, M::mul(M::add(x1, y1), 0.5)));
        offset_acc2 = M::add(offset_acc2, M::mul(diff2, M::mul(M::add(x2, y2), 0.5)));
        offset_acc3 = M::add(offset_acc3, M::mul(diff3, M::mul(M::add(x3, y3), 0.5)));
        offset_acc4 = M::add(offset_acc4, M::mul(diff4, M::mul(M::add(x4, y4), 0.5)));
        offset_acc5 = M::add(offset_acc5, M::mul(diff5, M::mul(M::add(x5, y5), 0.5)));
        offset_acc6 = M::add(offset_acc6, M::mul(diff6, M::mul(M::add(x6, y6), 0.5)));
        offset_acc7 = M::add(offset_acc7, M::mul(diff7, M::mul(M::add(x7, y7), 0.5)));
        offset_acc8 = M::add(offset_acc8, M::mul(diff8, M::mul(M::add(x8, y8), 0.5)));

        hyperplane.extend_from_slice(&[
            diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8,
        ]);

        i += 8;
    }

    let hyperplane_offset = -rollup_scalar_x8::<M>(
        offset_acc1,
        offset_acc2,
        offset_acc3,
        offset_acc4,
        offset_acc5,
        offset_acc6,
        offset_acc7,
        offset_acc8,
    );

    (hyperplane, hyperplane_offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::danger::test_utils::{
        assert_is_close,
        assert_is_close_vector,
        get_sample_vectors,
        simple_euclidean_hyperplane,
    };

    #[cfg(feature = "nightly")]
    #[test]
    fn test_x1024_fma_euclidean_hyperplane() {
        let (x, y) = get_sample_vectors(1024);
        let (hyperplane, offset) =
            unsafe { f32_x1024_fallback_fma_euclidean_hyperplane(&x, &y) };
        let (expected, expected_offset) = simple_euclidean_hyperplane(&x, &y);
        assert_is_close(offset, expected_offset);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_x1024_nofma_euclidean_hyperplane() {
        let (x, y) = get_sample_vectors(1024);
        let (hyperplane, offset) =
            unsafe { f32_x1024_fallback_nofma_euclidean_hyperplane(&x, &y) };
        let (expected, expected_offset) = simple_euclidean_hyperplane(&x, &y);
        assert_is_close(offset, expected_offset);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_x768_fma_euclidean_hyperplane() {
        let (x, y) = get_sample_vectors(768);
        let (hyperplane, offset) =
            unsafe { f32_x768_fallback_fma_euclidean_hyperplane(&x, &y) };
        let (expected, expected_offset) = simple_euclidean_hyperplane(&x, &y);
        assert_is_close(offset, expected_offset);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_x768_nofma_euclidean_hyperplane() {
        let (x, y) = get_sample_vectors(768);
        let (hyperplane, offset) =
            unsafe { f32_x768_fallback_nofma_euclidean_hyperplane(&x, &y) };
        let (expected, expected_offset) = simple_euclidean_hyperplane(&x, &y);
        assert_is_close(offset, expected_offset);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_x512_fma_euclidean_hyperplane() {
        let (x, y) = get_sample_vectors(512);
        let (hyperplane, offset) =
            unsafe { f32_x512_fallback_fma_euclidean_hyperplane(&x, &y) };
        let (expected, expected_offset) = simple_euclidean_hyperplane(&x, &y);
        assert_is_close(offset, expected_offset);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_x512_nofma_euclidean_hyperplane() {
        let (x, y) = get_sample_vectors(512);
        let (hyperplane, offset) =
            unsafe { f32_x512_fallback_nofma_euclidean_hyperplane(&x, &y) };
        let (expected, expected_offset) = simple_euclidean_hyperplane(&x, &y);
        assert_is_close(offset, expected_offset);
        assert_is_close_vector(&hyperplane, &expected);
    }
}
