use std::arch::x86_64::*;

use crate::danger::{offsets_avx512, sum_avx512_x8, CHUNK_0, CHUNK_1};

macro_rules! unrolled_loop {
    (
        $executor:ident,
        $x:ident,
        $y:ident,
        $acc1:expr,
        $acc2:expr,
        $acc3:expr,
        $acc4:expr,
        $acc5:expr,
        $acc6:expr,
        $acc7:expr,
        $acc8:expr,
        offsets => $($offset:expr $(,)?)*
    ) => {{
        $(
            $executor(
                $x.add($offset),
                $y.add($offset),
                $acc1,
                $acc2,
                $acc3,
                $acc4,
                $acc5,
                $acc6,
                $acc7,
                $acc8,
            );
        )*
    }};
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the squared Euclidean distance of two `[f32; 1024]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `1024` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x1024_avx512_nofma_euclidean(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 1024);
    debug_assert_eq!(y.len(), 1024);

    let x = x.as_ptr();
    let y = y.as_ptr();

    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();
    let mut acc8 = _mm512_setzero_ps();

    unrolled_loop!(
        execute_f32_x128_nofma_block_euclidean,
        x,
        y,
        &mut acc1,
        &mut acc2,
        &mut acc3,
        &mut acc4,
        &mut acc5,
        &mut acc6,
        &mut acc7,
        &mut acc8,
        offsets => 0, 128, 256, 384, 512, 640, 768, 896,
    );

    sum_avx512_x8(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the squared Euclidean distance of two `[f32; 768]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `768` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x768_avx512_nofma_euclidean(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 768);
    debug_assert_eq!(y.len(), 768);

    let x = x.as_ptr();
    let y = y.as_ptr();

    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();
    let mut acc8 = _mm512_setzero_ps();

    unrolled_loop!(
        execute_f32_x128_nofma_block_euclidean,
        x,
        y,
        &mut acc1,
        &mut acc2,
        &mut acc3,
        &mut acc4,
        &mut acc5,
        &mut acc6,
        &mut acc7,
        &mut acc8,
        offsets => 0, 128, 256, 384, 512, 640
    );

    sum_avx512_x8(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the squared Euclidean distance of two `[f32; 512]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `512` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x512_avx512_nofma_euclidean(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 512);
    debug_assert_eq!(y.len(), 512);

    let x = x.as_ptr();
    let y = y.as_ptr();

    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();
    let mut acc8 = _mm512_setzero_ps();

    unrolled_loop!(
        execute_f32_x128_nofma_block_euclidean,
        x,
        y,
        &mut acc1,
        &mut acc2,
        &mut acc3,
        &mut acc4,
        &mut acc5,
        &mut acc6,
        &mut acc7,
        &mut acc8,
        offsets => 0, 128, 256, 384
    );

    sum_avx512_x8(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the squared Euclidean distance of two `[f32; 1024]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `1024` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x1024_avx512_fma_euclidean(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 1024);
    debug_assert_eq!(y.len(), 1024);

    let x = x.as_ptr();
    let y = y.as_ptr();

    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();
    let mut acc8 = _mm512_setzero_ps();

    unrolled_loop!(
        execute_f32_x128_fma_block_euclidean,
        x,
        y,
        &mut acc1,
        &mut acc2,
        &mut acc3,
        &mut acc4,
        &mut acc5,
        &mut acc6,
        &mut acc7,
        &mut acc8,
        offsets => 0, 128, 256, 384, 512, 640, 768, 896,
    );

    sum_avx512_x8(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the squared Euclidean distance of two `[f32; 768]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `768` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x768_avx512_fma_euclidean(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 768);
    debug_assert_eq!(y.len(), 768);

    let x = x.as_ptr();
    let y = y.as_ptr();

    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();
    let mut acc8 = _mm512_setzero_ps();

    unrolled_loop!(
        execute_f32_x128_fma_block_euclidean,
        x,
        y,
        &mut acc1,
        &mut acc2,
        &mut acc3,
        &mut acc4,
        &mut acc5,
        &mut acc6,
        &mut acc7,
        &mut acc8,
        offsets => 0, 128, 256, 384, 512, 640
    );

    sum_avx512_x8(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the squared Euclidean distance of two `[f32; 512]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `512` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x512_avx512_fma_euclidean(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 512);
    debug_assert_eq!(y.len(), 512);

    let x = x.as_ptr();
    let y = y.as_ptr();

    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();
    let mut acc8 = _mm512_setzero_ps();

    unrolled_loop!(
        execute_f32_x128_fma_block_euclidean,
        x,
        y,
        &mut acc1,
        &mut acc2,
        &mut acc3,
        &mut acc4,
        &mut acc5,
        &mut acc6,
        &mut acc7,
        &mut acc8,
        offsets => 0, 128, 256, 384
    );

    sum_avx512_x8(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn execute_f32_x128_nofma_block_euclidean(
    x: *const f32,
    y: *const f32,
    acc1: &mut __m512,
    acc2: &mut __m512,
    acc3: &mut __m512,
    acc4: &mut __m512,
    acc5: &mut __m512,
    acc6: &mut __m512,
    acc7: &mut __m512,
    acc8: &mut __m512,
) {
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

    let diff1 = _mm512_sub_ps(x1, y1);
    let diff2 = _mm512_sub_ps(x2, y2);
    let diff3 = _mm512_sub_ps(x3, y3);
    let diff4 = _mm512_sub_ps(x4, y4);
    let diff5 = _mm512_sub_ps(x5, y5);
    let diff6 = _mm512_sub_ps(x6, y6);
    let diff7 = _mm512_sub_ps(x7, y7);
    let diff8 = _mm512_sub_ps(x8, y8);

    let r1 = _mm512_mul_ps(diff1, diff1);
    let r2 = _mm512_mul_ps(diff2, diff2);
    let r3 = _mm512_mul_ps(diff3, diff3);
    let r4 = _mm512_mul_ps(diff4, diff4);
    let r5 = _mm512_mul_ps(diff5, diff5);
    let r6 = _mm512_mul_ps(diff6, diff6);
    let r7 = _mm512_mul_ps(diff7, diff7);
    let r8 = _mm512_mul_ps(diff8, diff8);

    *acc1 = _mm512_add_ps(*acc1, r1);
    *acc2 = _mm512_add_ps(*acc2, r2);
    *acc3 = _mm512_add_ps(*acc3, r3);
    *acc4 = _mm512_add_ps(*acc4, r4);
    *acc5 = _mm512_add_ps(*acc5, r5);
    *acc6 = _mm512_add_ps(*acc6, r6);
    *acc7 = _mm512_add_ps(*acc7, r7);
    *acc8 = _mm512_add_ps(*acc8, r8);
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn execute_f32_x128_fma_block_euclidean(
    x: *const f32,
    y: *const f32,
    acc1: &mut __m512,
    acc2: &mut __m512,
    acc3: &mut __m512,
    acc4: &mut __m512,
    acc5: &mut __m512,
    acc6: &mut __m512,
    acc7: &mut __m512,
    acc8: &mut __m512,
) {
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

    let diff1 = _mm512_sub_ps(x1, y1);
    let diff2 = _mm512_sub_ps(x2, y2);
    let diff3 = _mm512_sub_ps(x3, y3);
    let diff4 = _mm512_sub_ps(x4, y4);
    let diff5 = _mm512_sub_ps(x5, y5);
    let diff6 = _mm512_sub_ps(x6, y6);
    let diff7 = _mm512_sub_ps(x7, y7);
    let diff8 = _mm512_sub_ps(x8, y8);

    *acc1 = _mm512_fmadd_ps(diff1, diff1, *acc1);
    *acc2 = _mm512_fmadd_ps(diff2, diff2, *acc2);
    *acc3 = _mm512_fmadd_ps(diff3, diff3, *acc3);
    *acc4 = _mm512_fmadd_ps(diff4, diff4, *acc4);
    *acc5 = _mm512_fmadd_ps(diff5, diff5, *acc5);
    *acc6 = _mm512_fmadd_ps(diff6, diff6, *acc6);
    *acc7 = _mm512_fmadd_ps(diff7, diff7, *acc7);
    *acc8 = _mm512_fmadd_ps(diff8, diff8, *acc8);
}

#[cfg(all(test, target_feature = "avx512f"))]
mod tests {
    use super::*;
    use crate::danger::test_utils::{get_sample_vectors, is_close, simple_euclidean};

    #[test]
    fn test_x1024_fma_euclidean() {
        let (x, y) = get_sample_vectors(1024);
        let dist = unsafe { f32_x1024_avx512_fma_euclidean(&x, &y) };
        assert!(is_close(dist, simple_euclidean(&x, &y)));
    }

    #[test]
    fn test_x1024_nofma_euclidean() {
        let (x, y) = get_sample_vectors(1024);
        let dist = unsafe { f32_x1024_avx512_nofma_euclidean(&x, &y) };
        assert!(is_close(dist, simple_euclidean(&x, &y)));
    }

    #[test]
    fn test_x768_fma_euclidean() {
        let (x, y) = get_sample_vectors(768);
        let dist = unsafe { f32_x768_avx512_fma_euclidean(&x, &y) };
        assert!(is_close(dist, simple_euclidean(&x, &y)));
    }

    #[test]
    fn test_x768_nofma_euclidean() {
        let (x, y) = get_sample_vectors(768);
        let dist = unsafe { f32_x768_avx512_nofma_euclidean(&x, &y) };
        assert!(is_close(dist, simple_euclidean(&x, &y)));
    }

    #[test]
    fn test_x512_fma_euclidean() {
        let (x, y) = get_sample_vectors(512);
        let dist = unsafe { f32_x512_avx512_fma_euclidean(&x, &y) };
        assert!(is_close(dist, simple_euclidean(&x, &y)));
    }

    #[test]
    fn test_x512_nofma_euclidean() {
        let (x, y) = get_sample_vectors(512);
        let dist = unsafe { f32_x512_avx512_nofma_euclidean(&x, &y) };
        assert!(is_close(dist, simple_euclidean(&x, &y)));
    }
}
