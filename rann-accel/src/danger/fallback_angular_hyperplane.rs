use crate::danger::fallback_dot_product::fallback_dot;
use crate::math::*;

#[inline]
/// Computes the angular hyperplane of two `[f32; 1024]` vectors.
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
pub unsafe fn f32_x1024_fallback_nofma_angular_hyperplane(
    x: &[f32],
    y: &[f32],
) -> Vec<f32> {
    fallback_angular_hyperplane::<StdMath, 1024>(x, y)
}

#[inline]
/// Computes the angular hyperplane of two `[f32; 768]` vectors.
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
pub unsafe fn f32_x768_fallback_nofma_angular_hyperplane(
    x: &[f32],
    y: &[f32],
) -> Vec<f32> {
    fallback_angular_hyperplane::<StdMath, 768>(x, y)
}

#[inline]
/// Computes the angular hyperplane of two `[f32; 512]` vectors.
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
pub unsafe fn f32_x512_fallback_nofma_angular_hyperplane(
    x: &[f32],
    y: &[f32],
) -> Vec<f32> {
    fallback_angular_hyperplane::<StdMath, 512>(x, y)
}

#[cfg(feature = "nightly")]
#[inline]
/// Computes the angular hyperplane of two `[f32; 1024]` vectors.
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
pub unsafe fn f32_x1024_fallback_fma_angular_hyperplane(
    x: &[f32],
    y: &[f32],
) -> Vec<f32> {
    fallback_angular_hyperplane::<FastMath, 1024>(x, y)
}

#[cfg(feature = "nightly")]
#[inline]
/// Computes the angular hyperplane of two `[f32; 68]` vectors.
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
pub unsafe fn f32_x768_fallback_fma_angular_hyperplane(
    x: &[f32],
    y: &[f32],
) -> Vec<f32> {
    fallback_angular_hyperplane::<FastMath, 768>(x, y)
}

#[cfg(feature = "nightly")]
#[inline]
/// Computes the angular hyperplane of two `[f32; 512]` vectors.
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
pub unsafe fn f32_x512_fallback_fma_angular_hyperplane(
    x: &[f32],
    y: &[f32],
) -> Vec<f32> {
    fallback_angular_hyperplane::<FastMath, 512>(x, y)
}

unsafe fn fallback_angular_hyperplane<M: Math, const DIMS: usize>(
    x: &[f32],
    y: &[f32],
) -> Vec<f32> {
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

    let mut norm_x = fallback_dot::<M, DIMS>(x, x).sqrt();
    if norm_x.abs() < f32::EPSILON {
        norm_x = 1.0;
    }

    let mut norm_y = fallback_dot::<M, DIMS>(y, y).sqrt();
    if norm_y.abs() < f32::EPSILON {
        norm_y = 1.0;
    }

    let mut hyperplane = Vec::with_capacity(DIMS);

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

        hyperplane.extend_from_slice(&[
            M::sub(M::div(x1, norm_x), M::div(y1, norm_y)),
            M::sub(M::div(x2, norm_x), M::div(y2, norm_y)),
            M::sub(M::div(x3, norm_x), M::div(y3, norm_y)),
            M::sub(M::div(x4, norm_x), M::div(y4, norm_y)),
            M::sub(M::div(x5, norm_x), M::div(y5, norm_y)),
            M::sub(M::div(x6, norm_x), M::div(y6, norm_y)),
            M::sub(M::div(x7, norm_x), M::div(y7, norm_y)),
            M::sub(M::div(x8, norm_x), M::div(y8, norm_y)),
        ]);

        i += 8;
    }

    let mut norm_hyperplane = fallback_dot::<M, DIMS>(&hyperplane, &hyperplane).sqrt();
    if norm_hyperplane.abs() < f32::EPSILON {
        norm_hyperplane = 1.0;
    }

    let mut i = 0;
    while i < hyperplane.len() {
        let x1 = *hyperplane.get_unchecked(i);
        let x2 = *hyperplane.get_unchecked(i + 1);
        let x3 = *hyperplane.get_unchecked(i + 2);
        let x4 = *hyperplane.get_unchecked(i + 3);
        let x5 = *hyperplane.get_unchecked(i + 4);
        let x6 = *hyperplane.get_unchecked(i + 5);
        let x7 = *hyperplane.get_unchecked(i + 6);
        let x8 = *hyperplane.get_unchecked(i + 7);

        *hyperplane.get_unchecked_mut(i) = M::div(x1, norm_hyperplane);
        *hyperplane.get_unchecked_mut(i + 1) = M::div(x2, norm_hyperplane);
        *hyperplane.get_unchecked_mut(i + 2) = M::div(x3, norm_hyperplane);
        *hyperplane.get_unchecked_mut(i + 3) = M::div(x4, norm_hyperplane);
        *hyperplane.get_unchecked_mut(i + 4) = M::div(x5, norm_hyperplane);
        *hyperplane.get_unchecked_mut(i + 5) = M::div(x6, norm_hyperplane);
        *hyperplane.get_unchecked_mut(i + 6) = M::div(x7, norm_hyperplane);
        *hyperplane.get_unchecked_mut(i + 7) = M::div(x8, norm_hyperplane);

        i += 8;
    }

    hyperplane
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::danger::test_utils::{
        assert_is_close_vector,
        get_sample_vectors,
        simple_angular_hyperplane,
    };

    #[cfg(feature = "nightly")]
    #[test]
    fn test_x1024_fma_angular_hyperplane() {
        let (x, y) = get_sample_vectors(1024);
        let hyperplane = unsafe { f32_x1024_fallback_fma_angular_hyperplane(&x, &y) };
        let expected = simple_angular_hyperplane(&x, &y);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_x1024_nofma_angular_hyperplane() {
        let (x, y) = get_sample_vectors(1024);
        let hyperplane = unsafe { f32_x1024_fallback_nofma_angular_hyperplane(&x, &y) };
        let expected = simple_angular_hyperplane(&x, &y);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_x768_fma_angular_hyperplane() {
        let (x, y) = get_sample_vectors(768);
        let hyperplane = unsafe { f32_x768_fallback_fma_angular_hyperplane(&x, &y) };
        let expected = simple_angular_hyperplane(&x, &y);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_x768_nofma_angular_hyperplane() {
        let (x, y) = get_sample_vectors(768);
        let hyperplane = unsafe { f32_x768_fallback_nofma_angular_hyperplane(&x, &y) };
        let expected = simple_angular_hyperplane(&x, &y);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_x512_fma_angular_hyperplane() {
        let (x, y) = get_sample_vectors(512);
        let hyperplane = unsafe { f32_x512_fallback_fma_angular_hyperplane(&x, &y) };
        let expected = simple_angular_hyperplane(&x, &y);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_x512_nofma_angular_hyperplane() {
        let (x, y) = get_sample_vectors(512);
        let hyperplane = unsafe { f32_x512_fallback_nofma_angular_hyperplane(&x, &y) };
        let expected = simple_angular_hyperplane(&x, &y);
        assert_is_close_vector(&hyperplane, &expected);
    }
}
