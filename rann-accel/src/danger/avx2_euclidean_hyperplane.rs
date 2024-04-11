use std::arch::x86_64::*;
use std::{mem, ptr};

use crate::danger::{offsets_avx2, rollup_x8, sum_avx2, CHUNK_0, CHUNK_1};

macro_rules! compute_euclidean_hyperplane {
    (
        $executor:ident,
        $x:ident,
        $y:ident,
        dims = $dims:expr,
        offsets => $($offset:expr $(,)?)*
    ) => {{
        debug_assert_eq!($x.len(), $dims);
        debug_assert_eq!($y.len(), $dims);

        let x = $x.as_ptr();
        let y = $y.as_ptr();

        let mut hyperplane = vec![0.0; $dims];

        let mut offset_acc1 = _mm256_setzero_ps();
        let mut offset_acc2 = _mm256_setzero_ps();
        let mut offset_acc3 = _mm256_setzero_ps();
        let mut offset_acc4 = _mm256_setzero_ps();
        let mut offset_acc5 = _mm256_setzero_ps();
        let mut offset_acc6 = _mm256_setzero_ps();
        let mut offset_acc7 = _mm256_setzero_ps();
        let mut offset_acc8 = _mm256_setzero_ps();

        $(
            let results = $executor(
                x.add($offset),
                y.add($offset),
                &mut offset_acc1,
                &mut offset_acc2,
                &mut offset_acc3,
                &mut offset_acc4,
                &mut offset_acc5,
                &mut offset_acc6,
                &mut offset_acc7,
                &mut offset_acc8,
            );
            ptr::copy_nonoverlapping(
                results.as_ptr(),
                hyperplane.as_mut_ptr().add($offset),
                results.len(),
            );
        )*

        let hyperplane_offset = sub_reduce_x8(
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
    }};
}

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the Euclidean hyperplane of two `[f32; 1024]` vectors
/// and the offset from origin.
///
/// # Safety
///
/// Vectors **MUST** be `1024` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x1024_avx2_nofma_euclidean_hyperplane(
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, f32) {
    compute_euclidean_hyperplane!(
        execute_f32_x64_block_nofma_hyperplane,
        x,
        y,
        dims = 1024,
        offsets =>
            0, 64, 128, 192,
            256, 320, 384, 448,
            512, 576, 640, 704,
            768, 832, 896, 960,
    )
}

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the Euclidean hyperplane of two `[f32; 768]` vectors
/// and the offset from origin.
///
/// # Safety
///
/// Vectors **MUST** be `768` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x768_avx2_nofma_euclidean_hyperplane(
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, f32) {
    compute_euclidean_hyperplane!(
        execute_f32_x64_block_nofma_hyperplane,
        x,
        y,
        dims = 768,
        offsets =>
            0, 64, 128, 192,
            256, 320, 384, 448,
            512, 576, 640, 704,
    )
}

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the Euclidean hyperplane of two `[f32; 512]` vectors
/// and the offset from origin.
///
/// # Safety
///
/// Vectors **MUST** be `512` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x512_avx2_nofma_euclidean_hyperplane(
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, f32) {
    compute_euclidean_hyperplane!(
        execute_f32_x64_block_nofma_hyperplane,
        x,
        y,
        dims = 512,
        offsets =>
            0, 64, 128, 192,
            256, 320, 384, 448,
    )
}

#[cfg(feature = "nightly")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the Euclidean hyperplane of two `[f32; 1024]` vectors
/// and the offset from origin.
///
/// # Safety
///
/// Vectors **MUST** be `1024` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x1024_avx2_fma_euclidean_hyperplane(
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, f32) {
    compute_euclidean_hyperplane!(
        execute_f32_x64_block_fma_hyperplane,
        x,
        y,
        dims = 1024,
        offsets =>
            0, 64, 128, 192,
            256, 320, 384, 448,
            512, 576, 640, 704,
            768, 832, 896, 960,
    )
}

#[cfg(feature = "nightly")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the Euclidean hyperplane of two `[f32; 768]` vectors
/// and the offset from origin.
///
/// # Safety
///
/// Vectors **MUST** be `768` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x768_avx2_fma_euclidean_hyperplane(
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, f32) {
    compute_euclidean_hyperplane!(
        execute_f32_x64_block_fma_hyperplane,
        x,
        y,
        dims = 768,
        offsets =>
            0, 64, 128, 192,
            256, 320, 384, 448,
            512, 576, 640, 704,
    )
}

#[cfg(feature = "nightly")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the Euclidean hyperplane of two `[f32; 512]` vectors
/// and the offset from origin.
///
/// # Safety
///
/// Vectors **MUST** be `512` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x512_avx2_fma_euclidean_hyperplane(
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, f32) {
    compute_euclidean_hyperplane!(
        execute_f32_x64_block_fma_hyperplane,
        x,
        y,
        dims = 512,
        offsets =>
            0, 64, 128, 192,
            256, 320, 384, 448,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn execute_f32_x64_block_nofma_hyperplane(
    x: *const f32,
    y: *const f32,
    offset_acc1: &mut __m256,
    offset_acc2: &mut __m256,
    offset_acc3: &mut __m256,
    offset_acc4: &mut __m256,
    offset_acc5: &mut __m256,
    offset_acc6: &mut __m256,
    offset_acc7: &mut __m256,
    offset_acc8: &mut __m256,
) -> [f32; 64] {
    // TODO: Hopefully LLVM is smart enough to optimize this out, but we should
    //       double check that we don't reset the register each time.
    let div_by_2 = _mm256_set1_ps(0.5);

    let [x1, x2, x3, x4] = offsets_avx2::<CHUNK_0>(x);
    let [x5, x6, x7, x8] = offsets_avx2::<CHUNK_1>(x);

    let [y1, y2, y3, y4] = offsets_avx2::<CHUNK_0>(y);
    let [y5, y6, y7, y8] = offsets_avx2::<CHUNK_1>(y);

    let x1 = _mm256_loadu_ps(x1);
    let x2 = _mm256_loadu_ps(x2);
    let x3 = _mm256_loadu_ps(x3);
    let x4 = _mm256_loadu_ps(x4);
    let x5 = _mm256_loadu_ps(x5);
    let x6 = _mm256_loadu_ps(x6);
    let x7 = _mm256_loadu_ps(x7);
    let x8 = _mm256_loadu_ps(x8);

    let y1 = _mm256_loadu_ps(y1);
    let y2 = _mm256_loadu_ps(y2);
    let y3 = _mm256_loadu_ps(y3);
    let y4 = _mm256_loadu_ps(y4);
    let y5 = _mm256_loadu_ps(y5);
    let y6 = _mm256_loadu_ps(y6);
    let y7 = _mm256_loadu_ps(y7);
    let y8 = _mm256_loadu_ps(y8);

    let diff1 = _mm256_sub_ps(x1, y1);
    let diff2 = _mm256_sub_ps(x2, y2);
    let diff3 = _mm256_sub_ps(x3, y3);
    let diff4 = _mm256_sub_ps(x4, y4);
    let diff5 = _mm256_sub_ps(x5, y5);
    let diff6 = _mm256_sub_ps(x6, y6);
    let diff7 = _mm256_sub_ps(x7, y7);
    let diff8 = _mm256_sub_ps(x8, y8);

    let sum1 = _mm256_add_ps(x1, y1);
    let sum2 = _mm256_add_ps(x2, y2);
    let sum3 = _mm256_add_ps(x3, y3);
    let sum4 = _mm256_add_ps(x4, y4);
    let sum5 = _mm256_add_ps(x5, y5);
    let sum6 = _mm256_add_ps(x6, y6);
    let sum7 = _mm256_add_ps(x7, y7);
    let sum8 = _mm256_add_ps(x8, y8);

    let mean1 = _mm256_mul_ps(sum1, div_by_2);
    let mean2 = _mm256_mul_ps(sum2, div_by_2);
    let mean3 = _mm256_mul_ps(sum3, div_by_2);
    let mean4 = _mm256_mul_ps(sum4, div_by_2);
    let mean5 = _mm256_mul_ps(sum5, div_by_2);
    let mean6 = _mm256_mul_ps(sum6, div_by_2);
    let mean7 = _mm256_mul_ps(sum7, div_by_2);
    let mean8 = _mm256_mul_ps(sum8, div_by_2);

    let r1 = _mm256_mul_ps(diff1, mean1);
    let r2 = _mm256_mul_ps(diff2, mean2);
    let r3 = _mm256_mul_ps(diff3, mean3);
    let r4 = _mm256_mul_ps(diff4, mean4);
    let r5 = _mm256_mul_ps(diff5, mean5);
    let r6 = _mm256_mul_ps(diff6, mean6);
    let r7 = _mm256_mul_ps(diff7, mean7);
    let r8 = _mm256_mul_ps(diff8, mean8);

    *offset_acc1 = _mm256_add_ps(*offset_acc1, r1);
    *offset_acc2 = _mm256_add_ps(*offset_acc2, r2);
    *offset_acc3 = _mm256_add_ps(*offset_acc3, r3);
    *offset_acc4 = _mm256_add_ps(*offset_acc4, r4);
    *offset_acc5 = _mm256_add_ps(*offset_acc5, r5);
    *offset_acc6 = _mm256_add_ps(*offset_acc6, r6);
    *offset_acc7 = _mm256_add_ps(*offset_acc7, r7);
    *offset_acc8 = _mm256_add_ps(*offset_acc8, r8);

    let plane = [diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8];

    mem::transmute(plane)
}

#[cfg(feature = "nightly")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn execute_f32_x64_block_fma_hyperplane(
    x: *const f32,
    y: *const f32,
    offset_acc1: &mut __m256,
    offset_acc2: &mut __m256,
    offset_acc3: &mut __m256,
    offset_acc4: &mut __m256,
    offset_acc5: &mut __m256,
    offset_acc6: &mut __m256,
    offset_acc7: &mut __m256,
    offset_acc8: &mut __m256,
) -> [f32; 64] {
    // TODO: Hopefully LLVM is smart enough to optimize this out, but we should
    //       double check that we don't reset the register each time.
    let div_by_2 = _mm256_set1_ps(0.5);

    let [x1, x2, x3, x4] = offsets_avx2::<CHUNK_0>(x);
    let [x5, x6, x7, x8] = offsets_avx2::<CHUNK_1>(x);

    let [y1, y2, y3, y4] = offsets_avx2::<CHUNK_0>(y);
    let [y5, y6, y7, y8] = offsets_avx2::<CHUNK_1>(y);

    let x1 = _mm256_loadu_ps(x1);
    let x2 = _mm256_loadu_ps(x2);
    let x3 = _mm256_loadu_ps(x3);
    let x4 = _mm256_loadu_ps(x4);
    let x5 = _mm256_loadu_ps(x5);
    let x6 = _mm256_loadu_ps(x6);
    let x7 = _mm256_loadu_ps(x7);
    let x8 = _mm256_loadu_ps(x8);

    let y1 = _mm256_loadu_ps(y1);
    let y2 = _mm256_loadu_ps(y2);
    let y3 = _mm256_loadu_ps(y3);
    let y4 = _mm256_loadu_ps(y4);
    let y5 = _mm256_loadu_ps(y5);
    let y6 = _mm256_loadu_ps(y6);
    let y7 = _mm256_loadu_ps(y7);
    let y8 = _mm256_loadu_ps(y8);

    let diff1 = _mm256_sub_ps(x1, y1);
    let diff2 = _mm256_sub_ps(x2, y2);
    let diff3 = _mm256_sub_ps(x3, y3);
    let diff4 = _mm256_sub_ps(x4, y4);
    let diff5 = _mm256_sub_ps(x5, y5);
    let diff6 = _mm256_sub_ps(x6, y6);
    let diff7 = _mm256_sub_ps(x7, y7);
    let diff8 = _mm256_sub_ps(x8, y8);

    let sum1 = _mm256_add_ps(x1, y1);
    let sum2 = _mm256_add_ps(x2, y2);
    let sum3 = _mm256_add_ps(x3, y3);
    let sum4 = _mm256_add_ps(x4, y4);
    let sum5 = _mm256_add_ps(x5, y5);
    let sum6 = _mm256_add_ps(x6, y6);
    let sum7 = _mm256_add_ps(x7, y7);
    let sum8 = _mm256_add_ps(x8, y8);

    let mean1 = _mm256_mul_ps(sum1, div_by_2);
    let mean2 = _mm256_mul_ps(sum2, div_by_2);
    let mean3 = _mm256_mul_ps(sum3, div_by_2);
    let mean4 = _mm256_mul_ps(sum4, div_by_2);
    let mean5 = _mm256_mul_ps(sum5, div_by_2);
    let mean6 = _mm256_mul_ps(sum6, div_by_2);
    let mean7 = _mm256_mul_ps(sum7, div_by_2);
    let mean8 = _mm256_mul_ps(sum8, div_by_2);

    *offset_acc1 = _mm256_fmadd_ps(diff1, mean1, *offset_acc1);
    *offset_acc2 = _mm256_fmadd_ps(diff2, mean2, *offset_acc2);
    *offset_acc3 = _mm256_fmadd_ps(diff3, mean3, *offset_acc3);
    *offset_acc4 = _mm256_fmadd_ps(diff4, mean4, *offset_acc4);
    *offset_acc5 = _mm256_fmadd_ps(diff5, mean5, *offset_acc5);
    *offset_acc6 = _mm256_fmadd_ps(diff6, mean6, *offset_acc6);
    *offset_acc7 = _mm256_fmadd_ps(diff7, mean7, *offset_acc7);
    *offset_acc8 = _mm256_fmadd_ps(diff8, mean8, *offset_acc8);

    let plane = [diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8];

    mem::transmute(plane)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn sub_reduce_x8(
    mut acc1: __m256,
    acc2: __m256,
    acc3: __m256,
    acc4: __m256,
    acc5: __m256,
    acc6: __m256,
    acc7: __m256,
    acc8: __m256,
) -> f32 {
    acc1 = rollup_x8(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
    -sum_avx2(acc1)
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
            unsafe { f32_x1024_avx2_fma_euclidean_hyperplane(&x, &y) };
        let (expected, expected_offset) = simple_euclidean_hyperplane(&x, &y);
        assert_is_close(offset, expected_offset);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_x1024_nofma_euclidean_hyperplane() {
        let (x, y) = get_sample_vectors(1024);
        let (hyperplane, offset) =
            unsafe { f32_x1024_avx2_nofma_euclidean_hyperplane(&x, &y) };
        let (expected, expected_offset) = simple_euclidean_hyperplane(&x, &y);
        assert_is_close(offset, expected_offset);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_x768_fma_euclidean_hyperplane() {
        let (x, y) = get_sample_vectors(768);
        let (hyperplane, offset) =
            unsafe { f32_x768_avx2_fma_euclidean_hyperplane(&x, &y) };
        let (expected, expected_offset) = simple_euclidean_hyperplane(&x, &y);
        assert_is_close(offset, expected_offset);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_x768_nofma_euclidean_hyperplane() {
        let (x, y) = get_sample_vectors(768);
        let (hyperplane, offset) =
            unsafe { f32_x768_avx2_nofma_euclidean_hyperplane(&x, &y) };
        let (expected, expected_offset) = simple_euclidean_hyperplane(&x, &y);
        assert_is_close(offset, expected_offset);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_x512_fma_euclidean_hyperplane() {
        let (x, y) = get_sample_vectors(512);
        let (hyperplane, offset) =
            unsafe { f32_x512_avx2_fma_euclidean_hyperplane(&x, &y) };
        let (expected, expected_offset) = simple_euclidean_hyperplane(&x, &y);
        assert_is_close(offset, expected_offset);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_x512_nofma_euclidean_hyperplane() {
        let (x, y) = get_sample_vectors(512);
        let (hyperplane, offset) =
            unsafe { f32_x512_avx2_nofma_euclidean_hyperplane(&x, &y) };
        let (expected, expected_offset) = simple_euclidean_hyperplane(&x, &y);
        assert_is_close(offset, expected_offset);
        assert_is_close_vector(&hyperplane, &expected);
    }
}
