use std::arch::x86_64::*;
use std::{mem, ptr};

use crate::danger::{
    f32_x1024_avx512_fma_norm,
    f32_x1024_avx512_nofma_norm,
    f32_x512_avx512_fma_norm,
    f32_x512_avx512_nofma_norm,
    f32_x768_avx512_fma_norm,
    f32_x768_avx512_nofma_norm,
    offsets_avx512,
    CHUNK_0,
    CHUNK_1,
};

macro_rules! compute_normal_vector_unrolled {
    (
        $hyperplane_ptr:expr,
        $x:ident,
        $y:ident,
        $norm_x:ident,
        $norm_y:ident,
        offsets => $($offset:expr $(,)?)*
    ) => {{
        $(
            let results = execute_f32_x128_block_normal_vector(
                $x.add($offset),
                $y.add($offset),
                $norm_x,
                $norm_y,
            );
            ptr::copy_nonoverlapping(results.as_ptr(), $hyperplane_ptr.add($offset), results.len());
        )*
    }};
}

macro_rules! normalize_hyperplane_unrolled {
    (
        $hyperplane_ptr:expr,
        $norm_hyperplane:ident,
        offsets => $($offset:expr $(,)?)*
    ) => {{
        $(
            let results = execute_f32_x128_block_apply_norm(
                $hyperplane_ptr.add($offset),
                $norm_hyperplane,
            );
            ptr::copy_nonoverlapping(results.as_ptr(), $hyperplane_ptr.add($offset), results.len());
        )*
    }};
}

macro_rules! compute_angular_hyperplane {
    (
        $x:expr,
        $y:expr,
        dims = $dims:expr,
        norm = $norm_func:ident,
        offsets => $($offset:expr $(,)?)*
    ) => {{
        debug_assert_eq!($x.len(), $dims);
        debug_assert_eq!($y.len(), $dims);

        let mut hyperplane = vec![0.0; $dims];

        let mut norm_x = $norm_func(&$x).sqrt();
        let mut norm_y = $norm_func(&$y).sqrt();

        if norm_x.abs() < f32::EPSILON {
            norm_x = 1.0;
        }

        if norm_y.abs() < f32::EPSILON {
            norm_y = 1.0;
        }

        let x = $x.as_ptr();
        let y = $y.as_ptr();

        let norm_x = _mm512_set1_ps(norm_x);
        let norm_y = _mm512_set1_ps(norm_y);

        //  Compute the normal vector to the hyperplane (the vector between the two points)
        compute_normal_vector_unrolled!(
            hyperplane.as_mut_ptr(),
            x,
            y,
            norm_x,
            norm_y,
            offsets => $($offset,)*
        );

        let mut norm_hyperplane = $norm_func(&hyperplane).sqrt();
        if norm_hyperplane.abs() < f32::EPSILON {
            norm_hyperplane = 1.0;
        }

        let norm_hyperplane = _mm512_set1_ps(norm_hyperplane);
        normalize_hyperplane_unrolled!(
            hyperplane.as_mut_ptr(),
            norm_hyperplane,
            offsets => $($offset,)*
        );

        hyperplane
    }};
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the angular hyperplane of two `[f32; 1024]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `1024` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x1024_avx512_nofma_angular_hyperplane(
    x: &[f32],
    y: &[f32],
) -> Vec<f32> {
    compute_angular_hyperplane!(
        x,
        y,
        dims = 1024,
        norm = f32_x1024_avx512_nofma_norm,
        offsets => 0, 128, 256, 384, 512, 640, 768, 896,
    )
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the angular hyperplane of two `[f32; 768]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `768` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x768_avx512_nofma_angular_hyperplane(
    x: &[f32],
    y: &[f32],
) -> Vec<f32> {
    compute_angular_hyperplane!(
        x,
        y,
        dims = 768,
        norm = f32_x768_avx512_nofma_norm,
        offsets => 0, 128, 256, 384, 512, 640,
    )
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the angular hyperplane of two `[f32; 512]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `512` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x512_avx512_nofma_angular_hyperplane(
    x: &[f32],
    y: &[f32],
) -> Vec<f32> {
    compute_angular_hyperplane!(
        x,
        y,
        dims = 512,
        norm = f32_x512_avx512_nofma_norm,
        offsets => 0, 128, 256, 384,
    )
}

#[target_feature(enable = "avx512f", enable = "fma")]
#[inline]
/// Computes the angular hyperplane of two `[f32; 1024]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `1024` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x1024_avx512_fma_angular_hyperplane(x: &[f32], y: &[f32]) -> Vec<f32> {
    compute_angular_hyperplane!(
        x,
        y,
        dims = 1024,
        norm = f32_x1024_avx512_fma_norm,
        offsets => 0, 128, 256, 384, 512, 640, 768, 896,
    )
}

#[target_feature(enable = "avx512f", enable = "fma")]
#[inline]
/// Computes the angular hyperplane of two `[f32; 768]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `768` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x768_avx512_fma_angular_hyperplane(x: &[f32], y: &[f32]) -> Vec<f32> {
    compute_angular_hyperplane!(
        x,
        y,
        dims = 768,
        norm = f32_x768_avx512_fma_norm,
        offsets => 0, 128, 256, 384, 512, 640,
    )
}

#[target_feature(enable = "avx512f", enable = "fma")]
#[inline]
/// Computes the angular hyperplane of two `[f32; 512]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `512` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x512_avx512_fma_angular_hyperplane(x: &[f32], y: &[f32]) -> Vec<f32> {
    compute_angular_hyperplane!(
        x,
        y,
        dims = 512,
        norm = f32_x512_avx512_fma_norm,
        offsets => 0, 128, 256, 384,
    )
}

#[inline(always)]
unsafe fn execute_f32_x128_block_normal_vector(
    x: *const f32,
    y: *const f32,
    norm_x: __m512,
    norm_y: __m512,
) -> [f32; 128] {
    let [x1, x2, x3, x4] = offsets_avx512::<CHUNK_0>(x);
    let [x5, x6, x7, x8] = offsets_avx512::<CHUNK_1>(x);

    let [y1, y2, y3, y4] = offsets_avx512::<CHUNK_0>(y);
    let [y5, y6, y7, y8] = offsets_avx512::<CHUNK_1>(y);

    let x1 = _mm512_loadu_ps(x1);
    let x2 = _mm512_loadu_ps(x2);
    let x3 = _mm512_loadu_ps(x3);
    let x4 = _mm512_loadu_ps(x4);
    let x5 = _mm512_loadu_ps(x5);
    let x6 = _mm512_loadu_ps(x6);
    let x7 = _mm512_loadu_ps(x7);
    let x8 = _mm512_loadu_ps(x8);

    let y1 = _mm512_loadu_ps(y1);
    let y2 = _mm512_loadu_ps(y2);
    let y3 = _mm512_loadu_ps(y3);
    let y4 = _mm512_loadu_ps(y4);
    let y5 = _mm512_loadu_ps(y5);
    let y6 = _mm512_loadu_ps(y6);
    let y7 = _mm512_loadu_ps(y7);
    let y8 = _mm512_loadu_ps(y8);

    let normalized_x1 = _mm512_div_ps(x1, norm_x);
    let normalized_x2 = _mm512_div_ps(x2, norm_x);
    let normalized_x3 = _mm512_div_ps(x3, norm_x);
    let normalized_x4 = _mm512_div_ps(x4, norm_x);
    let normalized_x5 = _mm512_div_ps(x5, norm_x);
    let normalized_x6 = _mm512_div_ps(x6, norm_x);
    let normalized_x7 = _mm512_div_ps(x7, norm_x);
    let normalized_x8 = _mm512_div_ps(x8, norm_x);

    let normalized_y1 = _mm512_div_ps(y1, norm_y);
    let normalized_y2 = _mm512_div_ps(y2, norm_y);
    let normalized_y3 = _mm512_div_ps(y3, norm_y);
    let normalized_y4 = _mm512_div_ps(y4, norm_y);
    let normalized_y5 = _mm512_div_ps(y5, norm_y);
    let normalized_y6 = _mm512_div_ps(y6, norm_y);
    let normalized_y7 = _mm512_div_ps(y7, norm_y);
    let normalized_y8 = _mm512_div_ps(y8, norm_y);

    let diff1 = _mm512_sub_ps(normalized_x1, normalized_y1);
    let diff2 = _mm512_sub_ps(normalized_x2, normalized_y2);
    let diff3 = _mm512_sub_ps(normalized_x3, normalized_y3);
    let diff4 = _mm512_sub_ps(normalized_x4, normalized_y4);
    let diff5 = _mm512_sub_ps(normalized_x5, normalized_y5);
    let diff6 = _mm512_sub_ps(normalized_x6, normalized_y6);
    let diff7 = _mm512_sub_ps(normalized_x7, normalized_y7);
    let diff8 = _mm512_sub_ps(normalized_x8, normalized_y8);

    let lanes = [diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8];

    mem::transmute(lanes)
}

#[inline(always)]
unsafe fn execute_f32_x128_block_apply_norm(
    x: *const f32,
    norm_x: __m512,
) -> [f32; 128] {
    let [x1, x2, x3, x4] = offsets_avx512::<CHUNK_0>(x);
    let [x5, x6, x7, x8] = offsets_avx512::<CHUNK_1>(x);

    let x1 = _mm512_loadu_ps(x1);
    let x2 = _mm512_loadu_ps(x2);
    let x3 = _mm512_loadu_ps(x3);
    let x4 = _mm512_loadu_ps(x4);
    let x5 = _mm512_loadu_ps(x5);
    let x6 = _mm512_loadu_ps(x6);
    let x7 = _mm512_loadu_ps(x7);
    let x8 = _mm512_loadu_ps(x8);

    let normalized_x1 = _mm512_div_ps(x1, norm_x);
    let normalized_x2 = _mm512_div_ps(x2, norm_x);
    let normalized_x3 = _mm512_div_ps(x3, norm_x);
    let normalized_x4 = _mm512_div_ps(x4, norm_x);
    let normalized_x5 = _mm512_div_ps(x5, norm_x);
    let normalized_x6 = _mm512_div_ps(x6, norm_x);
    let normalized_x7 = _mm512_div_ps(x7, norm_x);
    let normalized_x8 = _mm512_div_ps(x8, norm_x);

    let lanes = [
        normalized_x1,
        normalized_x2,
        normalized_x3,
        normalized_x4,
        normalized_x5,
        normalized_x6,
        normalized_x7,
        normalized_x8,
    ];

    mem::transmute(lanes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::danger::test_utils::{
        assert_is_close_vector,
        get_sample_vectors,
        simple_angular_hyperplane,
    };

    #[test]
    fn test_x1024_fma_angular_hyperplane() {
        let (x, y) = get_sample_vectors(1024);
        let hyperplane = unsafe { f32_x1024_avx512_fma_angular_hyperplane(&x, &y) };
        let expected = simple_angular_hyperplane(&x, &y);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_x1024_nofma_angular_hyperplane() {
        let (x, y) = get_sample_vectors(1024);
        let hyperplane = unsafe { f32_x1024_avx512_nofma_angular_hyperplane(&x, &y) };
        let expected = simple_angular_hyperplane(&x, &y);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_x768_fma_angular_hyperplane() {
        let (x, y) = get_sample_vectors(768);
        let hyperplane = unsafe { f32_x768_avx512_fma_angular_hyperplane(&x, &y) };
        let expected = simple_angular_hyperplane(&x, &y);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_x768_nofma_angular_hyperplane() {
        let (x, y) = get_sample_vectors(768);
        let hyperplane = unsafe { f32_x768_avx512_nofma_angular_hyperplane(&x, &y) };
        let expected = simple_angular_hyperplane(&x, &y);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_x512_fma_angular_hyperplane() {
        let (x, y) = get_sample_vectors(512);
        let hyperplane = unsafe { f32_x512_avx512_fma_angular_hyperplane(&x, &y) };
        let expected = simple_angular_hyperplane(&x, &y);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_x512_nofma_angular_hyperplane() {
        let (x, y) = get_sample_vectors(512);
        let hyperplane = unsafe { f32_x512_avx512_nofma_angular_hyperplane(&x, &y) };
        let expected = simple_angular_hyperplane(&x, &y);
        assert_is_close_vector(&hyperplane, &expected);
    }
}
