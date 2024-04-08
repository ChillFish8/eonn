use std::arch::x86_64::*;

use crate::danger::{offsets, rollup_x4, sum_avx2};
use crate::math::{FastMath, Math, StdMath};

macro_rules! unrolled_loop {
    (
        $executor:ident,
        $x:ident,
        $y:ident,
        $acc1:expr,
        $acc2:expr,
        $acc3:expr,
        $acc4:expr,
        $norm_x_acc1:expr,
        $norm_x_acc2:expr,
        $norm_x_acc3:expr,
        $norm_x_acc4:expr,
        $norm_y_acc1:expr,
        $norm_y_acc2:expr,
        $norm_y_acc3:expr,
        $norm_y_acc4:expr,
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
                $norm_x_acc1,
                $norm_x_acc2,
                $norm_x_acc3,
                $norm_x_acc4,
                $norm_y_acc1,
                $norm_y_acc2,
                $norm_y_acc3,
                $norm_y_acc4,
            );
        )*
    }};
}

#[inline]
/// Computes the cosine distance of two `[f32; 1024]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `1024` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x1024_avx2_nofma_cosine(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 1024);
    debug_assert_eq!(y.len(), 1024);

    let x = x.as_ptr();
    let y = y.as_ptr();

    _mm_prefetch::<_MM_HINT_T1>(x.cast());
    _mm_prefetch::<_MM_HINT_T1>(y.cast());

    let mut acc1 = _mm256_set1_ps(0.0);
    let mut acc2 = _mm256_set1_ps(0.0);
    let mut acc3 = _mm256_set1_ps(0.0);
    let mut acc4 = _mm256_set1_ps(0.0);

    let mut norm_x_acc1 = _mm256_set1_ps(0.0);
    let mut norm_x_acc2 = _mm256_set1_ps(0.0);
    let mut norm_x_acc3 = _mm256_set1_ps(0.0);
    let mut norm_x_acc4 = _mm256_set1_ps(0.0);

    let mut norm_y_acc1 = _mm256_set1_ps(0.0);
    let mut norm_y_acc2 = _mm256_set1_ps(0.0);
    let mut norm_y_acc3 = _mm256_set1_ps(0.0);
    let mut norm_y_acc4 = _mm256_set1_ps(0.0);

    unrolled_loop!(
        execute_f32_x64_nofma_block_cosine,
        x,
        y,
        &mut acc1,
        &mut acc2,
        &mut acc3,
        &mut acc4,
        &mut norm_x_acc1,
        &mut norm_x_acc2,
        &mut norm_x_acc3,
        &mut norm_x_acc4,
        &mut norm_y_acc1,
        &mut norm_y_acc2,
        &mut norm_y_acc3,
        &mut norm_y_acc4,
        offsets =>
            0, 64, 128, 192,
            256, 320, 384, 448,
            512, 576, 640, 704,
            768, 832, 896, 960,
    );

    let acc = rollup_x4(acc1, acc2, acc3, acc4);
    let norm_x_acc = rollup_x4(norm_x_acc1, norm_x_acc2, norm_x_acc3, norm_x_acc4);
    let norm_y_acc = rollup_x4(norm_y_acc1, norm_y_acc2, norm_y_acc3, norm_y_acc4);

    cosine::<StdMath>(acc, norm_x_acc, norm_y_acc)
}

#[inline]
/// Computes the cosine distance of two `[f32; 768]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `768` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x768_avx2_nofma_cosine(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 768);
    debug_assert_eq!(y.len(), 768);

    let x = x.as_ptr();
    let y = y.as_ptr();

    _mm_prefetch::<_MM_HINT_T1>(x.cast());
    _mm_prefetch::<_MM_HINT_T1>(y.cast());

    let mut acc1 = _mm256_set1_ps(0.0);
    let mut acc2 = _mm256_set1_ps(0.0);
    let mut acc3 = _mm256_set1_ps(0.0);
    let mut acc4 = _mm256_set1_ps(0.0);

    let mut norm_x_acc1 = _mm256_set1_ps(0.0);
    let mut norm_x_acc2 = _mm256_set1_ps(0.0);
    let mut norm_x_acc3 = _mm256_set1_ps(0.0);
    let mut norm_x_acc4 = _mm256_set1_ps(0.0);

    let mut norm_y_acc1 = _mm256_set1_ps(0.0);
    let mut norm_y_acc2 = _mm256_set1_ps(0.0);
    let mut norm_y_acc3 = _mm256_set1_ps(0.0);
    let mut norm_y_acc4 = _mm256_set1_ps(0.0);

    unrolled_loop!(
        execute_f32_x64_nofma_block_cosine,
        x,
        y,
        &mut acc1,
        &mut acc2,
        &mut acc3,
        &mut acc4,
        &mut norm_x_acc1,
        &mut norm_x_acc2,
        &mut norm_x_acc3,
        &mut norm_x_acc4,
        &mut norm_y_acc1,
        &mut norm_y_acc2,
        &mut norm_y_acc3,
        &mut norm_y_acc4,
        offsets =>
            0, 64, 128, 192,
            256, 320, 384, 448,
            512, 576, 640, 704
    );

    let acc = rollup_x4(acc1, acc2, acc3, acc4);
    let norm_x_acc = rollup_x4(norm_x_acc1, norm_x_acc2, norm_x_acc3, norm_x_acc4);
    let norm_y_acc = rollup_x4(norm_y_acc1, norm_y_acc2, norm_y_acc3, norm_y_acc4);

    cosine::<StdMath>(acc, norm_x_acc, norm_y_acc)
}

#[inline]
/// Computes the cosine distance of two `[f32; 512]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `512` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x512_avx2_nofma_cosine(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 512);
    debug_assert_eq!(y.len(), 512);

    let x = x.as_ptr();
    let y = y.as_ptr();

    _mm_prefetch::<_MM_HINT_T1>(x.cast());
    _mm_prefetch::<_MM_HINT_T1>(y.cast());

    let mut acc1 = _mm256_set1_ps(0.0);
    let mut acc2 = _mm256_set1_ps(0.0);
    let mut acc3 = _mm256_set1_ps(0.0);
    let mut acc4 = _mm256_set1_ps(0.0);

    let mut norm_x_acc1 = _mm256_set1_ps(0.0);
    let mut norm_x_acc2 = _mm256_set1_ps(0.0);
    let mut norm_x_acc3 = _mm256_set1_ps(0.0);
    let mut norm_x_acc4 = _mm256_set1_ps(0.0);

    let mut norm_y_acc1 = _mm256_set1_ps(0.0);
    let mut norm_y_acc2 = _mm256_set1_ps(0.0);
    let mut norm_y_acc3 = _mm256_set1_ps(0.0);
    let mut norm_y_acc4 = _mm256_set1_ps(0.0);

    unrolled_loop!(
        execute_f32_x64_nofma_block_cosine,
        x,
        y,
        &mut acc1,
        &mut acc2,
        &mut acc3,
        &mut acc4,
        &mut norm_x_acc1,
        &mut norm_x_acc2,
        &mut norm_x_acc3,
        &mut norm_x_acc4,
        &mut norm_y_acc1,
        &mut norm_y_acc2,
        &mut norm_y_acc3,
        &mut norm_y_acc4,
        offsets =>
            0, 64, 128, 192,
            256, 320, 384, 448
    );

    let acc = rollup_x4(acc1, acc2, acc3, acc4);
    let norm_x_acc = rollup_x4(norm_x_acc1, norm_x_acc2, norm_x_acc3, norm_x_acc4);
    let norm_y_acc = rollup_x4(norm_y_acc1, norm_y_acc2, norm_y_acc3, norm_y_acc4);

    cosine::<StdMath>(acc, norm_x_acc, norm_y_acc)
}

#[inline]
/// Computes the cosine distance of two `[f32; 1024]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `1024` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x1024_avx2_fma_cosine(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 1024);
    debug_assert_eq!(y.len(), 1024);

    let x = x.as_ptr();
    let y = y.as_ptr();

    _mm_prefetch::<_MM_HINT_T1>(x.cast());
    _mm_prefetch::<_MM_HINT_T1>(y.cast());

    let mut acc1 = _mm256_set1_ps(0.0);
    let mut acc2 = _mm256_set1_ps(0.0);
    let mut acc3 = _mm256_set1_ps(0.0);
    let mut acc4 = _mm256_set1_ps(0.0);

    let mut norm_x_acc1 = _mm256_set1_ps(0.0);
    let mut norm_x_acc2 = _mm256_set1_ps(0.0);
    let mut norm_x_acc3 = _mm256_set1_ps(0.0);
    let mut norm_x_acc4 = _mm256_set1_ps(0.0);

    let mut norm_y_acc1 = _mm256_set1_ps(0.0);
    let mut norm_y_acc2 = _mm256_set1_ps(0.0);
    let mut norm_y_acc3 = _mm256_set1_ps(0.0);
    let mut norm_y_acc4 = _mm256_set1_ps(0.0);

    unrolled_loop!(
        execute_f32_x64_fma_block_cosine,
        x,
        y,
        &mut acc1,
        &mut acc2,
        &mut acc3,
        &mut acc4,
        &mut norm_x_acc1,
        &mut norm_x_acc2,
        &mut norm_x_acc3,
        &mut norm_x_acc4,
        &mut norm_y_acc1,
        &mut norm_y_acc2,
        &mut norm_y_acc3,
        &mut norm_y_acc4,
        offsets =>
            0, 64, 128, 192,
            256, 320, 384, 448,
            512, 576, 640, 704,
            768, 832, 896, 960,
    );

    let acc = rollup_x4(acc1, acc2, acc3, acc4);
    let norm_x_acc = rollup_x4(norm_x_acc1, norm_x_acc2, norm_x_acc3, norm_x_acc4);
    let norm_y_acc = rollup_x4(norm_y_acc1, norm_y_acc2, norm_y_acc3, norm_y_acc4);

    cosine::<FastMath>(acc, norm_x_acc, norm_y_acc)
}

#[inline]
/// Computes the cosine distance of two `[f32; 768]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `768` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x768_avx2_fma_cosine(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 768);
    debug_assert_eq!(y.len(), 768);

    let x = x.as_ptr();
    let y = y.as_ptr();

    _mm_prefetch::<_MM_HINT_T1>(x.cast());
    _mm_prefetch::<_MM_HINT_T1>(y.cast());

    let mut acc1 = _mm256_set1_ps(0.0);
    let mut acc2 = _mm256_set1_ps(0.0);
    let mut acc3 = _mm256_set1_ps(0.0);
    let mut acc4 = _mm256_set1_ps(0.0);

    let mut norm_x_acc1 = _mm256_set1_ps(0.0);
    let mut norm_x_acc2 = _mm256_set1_ps(0.0);
    let mut norm_x_acc3 = _mm256_set1_ps(0.0);
    let mut norm_x_acc4 = _mm256_set1_ps(0.0);

    let mut norm_y_acc1 = _mm256_set1_ps(0.0);
    let mut norm_y_acc2 = _mm256_set1_ps(0.0);
    let mut norm_y_acc3 = _mm256_set1_ps(0.0);
    let mut norm_y_acc4 = _mm256_set1_ps(0.0);

    unrolled_loop!(
        execute_f32_x64_fma_block_cosine,
        x,
        y,
        &mut acc1,
        &mut acc2,
        &mut acc3,
        &mut acc4,
        &mut norm_x_acc1,
        &mut norm_x_acc2,
        &mut norm_x_acc3,
        &mut norm_x_acc4,
        &mut norm_y_acc1,
        &mut norm_y_acc2,
        &mut norm_y_acc3,
        &mut norm_y_acc4,
        offsets =>
            0, 64, 128, 192,
            256, 320, 384, 448,
            512, 576, 640, 704
    );

    let acc = rollup_x4(acc1, acc2, acc3, acc4);
    let norm_x_acc = rollup_x4(norm_x_acc1, norm_x_acc2, norm_x_acc3, norm_x_acc4);
    let norm_y_acc = rollup_x4(norm_y_acc1, norm_y_acc2, norm_y_acc3, norm_y_acc4);

    cosine::<FastMath>(acc, norm_x_acc, norm_y_acc)
}

#[inline]
/// Computes the cosine distance of two `[f32; 512]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `512` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x512_avx2_fma_cosine(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 512);
    debug_assert_eq!(y.len(), 512);

    let x = x.as_ptr();
    let y = y.as_ptr();

    _mm_prefetch::<_MM_HINT_T1>(x.cast());
    _mm_prefetch::<_MM_HINT_T1>(y.cast());

    let mut acc1 = _mm256_set1_ps(0.0);
    let mut acc2 = _mm256_set1_ps(0.0);
    let mut acc3 = _mm256_set1_ps(0.0);
    let mut acc4 = _mm256_set1_ps(0.0);

    let mut norm_x_acc1 = _mm256_set1_ps(0.0);
    let mut norm_x_acc2 = _mm256_set1_ps(0.0);
    let mut norm_x_acc3 = _mm256_set1_ps(0.0);
    let mut norm_x_acc4 = _mm256_set1_ps(0.0);

    let mut norm_y_acc1 = _mm256_set1_ps(0.0);
    let mut norm_y_acc2 = _mm256_set1_ps(0.0);
    let mut norm_y_acc3 = _mm256_set1_ps(0.0);
    let mut norm_y_acc4 = _mm256_set1_ps(0.0);

    unrolled_loop!(
        execute_f32_x64_fma_block_cosine,
        x,
        y,
        &mut acc1,
        &mut acc2,
        &mut acc3,
        &mut acc4,
        &mut norm_x_acc1,
        &mut norm_x_acc2,
        &mut norm_x_acc3,
        &mut norm_x_acc4,
        &mut norm_y_acc1,
        &mut norm_y_acc2,
        &mut norm_y_acc3,
        &mut norm_y_acc4,
        offsets =>
            0, 64, 128, 192,
            256, 320, 384, 448
    );

    let acc = rollup_x4(acc1, acc2, acc3, acc4);
    let norm_x_acc = rollup_x4(norm_x_acc1, norm_x_acc2, norm_x_acc3, norm_x_acc4);
    let norm_y_acc = rollup_x4(norm_y_acc1, norm_y_acc2, norm_y_acc3, norm_y_acc4);

    cosine::<FastMath>(acc, norm_x_acc, norm_y_acc)
}

#[inline(always)]
unsafe fn cosine<M: Math>(acc: __m256, norm_x_acc: __m256, norm_y_acc: __m256) -> f32 {
    let dot_product = sum_avx2(acc);
    let norm_x = sum_avx2(norm_x_acc);
    let norm_y = sum_avx2(norm_y_acc);

    if norm_x == 0.0 && norm_y == 0.0 {
        0.0
    } else if norm_x == 0.0 || norm_y == 0.0 {
        1.0
    } else {
        M::sub(1.0, M::div(dot_product, M::mul(norm_x, norm_y).sqrt()))
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn execute_f32_x64_nofma_block_cosine(
    x: *const f32,
    y: *const f32,
    acc1: &mut __m256,
    acc2: &mut __m256,
    acc3: &mut __m256,
    acc4: &mut __m256,
    norm_x_acc1: &mut __m256,
    norm_x_acc2: &mut __m256,
    norm_x_acc3: &mut __m256,
    norm_x_acc4: &mut __m256,
    norm_y_acc1: &mut __m256,
    norm_y_acc2: &mut __m256,
    norm_y_acc3: &mut __m256,
    norm_y_acc4: &mut __m256,
) {
    let [x1, x2, x3, x4] = offsets(x, 0);
    let [x5, x6, x7, x8] = offsets(x, 32);

    let [y1, y2, y3, y4] = offsets(y, 0);
    let [y5, y6, y7, y8] = offsets(y, 32);

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

    let r1 = _mm256_mul_ps(x1, y1);
    let r2 = _mm256_mul_ps(x2, y2);
    let r3 = _mm256_mul_ps(x3, y3);
    let r4 = _mm256_mul_ps(x4, y4);
    let r5 = _mm256_mul_ps(x5, y5);
    let r6 = _mm256_mul_ps(x6, y6);
    let r7 = _mm256_mul_ps(x7, y7);
    let r8 = _mm256_mul_ps(x8, y8);

    let norm_x_r1 = _mm256_mul_ps(x1, x1);
    let norm_x_r2 = _mm256_mul_ps(x2, x2);
    let norm_x_r3 = _mm256_mul_ps(x3, x3);
    let norm_x_r4 = _mm256_mul_ps(x4, x4);
    let norm_x_r5 = _mm256_mul_ps(x5, x5);
    let norm_x_r6 = _mm256_mul_ps(x6, x6);
    let norm_x_r7 = _mm256_mul_ps(x7, x7);
    let norm_x_r8 = _mm256_mul_ps(x8, x8);

    let norm_y_r1 = _mm256_mul_ps(y1, y1);
    let norm_y_r2 = _mm256_mul_ps(y2, y2);
    let norm_y_r3 = _mm256_mul_ps(y3, y3);
    let norm_y_r4 = _mm256_mul_ps(y4, y4);
    let norm_y_r5 = _mm256_mul_ps(y5, y5);
    let norm_y_r6 = _mm256_mul_ps(y6, y6);
    let norm_y_r7 = _mm256_mul_ps(y7, y7);
    let norm_y_r8 = _mm256_mul_ps(y8, y8);

    *acc1 = _mm256_add_ps(*acc1, r1);
    *acc2 = _mm256_add_ps(*acc2, r2);
    *acc3 = _mm256_add_ps(*acc3, r3);
    *acc4 = _mm256_add_ps(*acc4, r4);
    *acc1 = _mm256_add_ps(*acc1, r5);
    *acc2 = _mm256_add_ps(*acc2, r6);
    *acc3 = _mm256_add_ps(*acc3, r7);
    *acc4 = _mm256_add_ps(*acc4, r8);

    *norm_x_acc1 = _mm256_add_ps(*norm_x_acc1, norm_x_r1);
    *norm_x_acc2 = _mm256_add_ps(*norm_x_acc2, norm_x_r2);
    *norm_x_acc3 = _mm256_add_ps(*norm_x_acc3, norm_x_r3);
    *norm_x_acc4 = _mm256_add_ps(*norm_x_acc4, norm_x_r4);
    *norm_x_acc1 = _mm256_add_ps(*norm_x_acc1, norm_x_r5);
    *norm_x_acc2 = _mm256_add_ps(*norm_x_acc2, norm_x_r6);
    *norm_x_acc3 = _mm256_add_ps(*norm_x_acc3, norm_x_r7);
    *norm_x_acc4 = _mm256_add_ps(*norm_x_acc4, norm_x_r8);

    *norm_y_acc1 = _mm256_add_ps(*norm_y_acc1, norm_y_r1);
    *norm_y_acc2 = _mm256_add_ps(*norm_y_acc2, norm_y_r2);
    *norm_y_acc3 = _mm256_add_ps(*norm_y_acc3, norm_y_r3);
    *norm_y_acc4 = _mm256_add_ps(*norm_y_acc4, norm_y_r4);
    *norm_y_acc1 = _mm256_add_ps(*norm_y_acc1, norm_y_r5);
    *norm_y_acc2 = _mm256_add_ps(*norm_y_acc2, norm_y_r6);
    *norm_y_acc3 = _mm256_add_ps(*norm_y_acc3, norm_y_r7);
    *norm_y_acc4 = _mm256_add_ps(*norm_y_acc4, norm_y_r8);
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn execute_f32_x64_fma_block_cosine(
    x: *const f32,
    y: *const f32,
    acc1: &mut __m256,
    acc2: &mut __m256,
    acc3: &mut __m256,
    acc4: &mut __m256,
    norm_x_acc1: &mut __m256,
    norm_x_acc2: &mut __m256,
    norm_x_acc3: &mut __m256,
    norm_x_acc4: &mut __m256,
    norm_y_acc1: &mut __m256,
    norm_y_acc2: &mut __m256,
    norm_y_acc3: &mut __m256,
    norm_y_acc4: &mut __m256,
) {
    let [x1, x2, x3, x4] = offsets(x, 0);
    let [x5, x6, x7, x8] = offsets(x, 32);

    let [y1, y2, y3, y4] = offsets(y, 0);
    let [y5, y6, y7, y8] = offsets(y, 32);

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

    *acc1 = _mm256_fmadd_ps(x1, y1, *acc1);
    *acc2 = _mm256_fmadd_ps(x2, y2, *acc2);
    *acc3 = _mm256_fmadd_ps(x3, y3, *acc3);
    *acc4 = _mm256_fmadd_ps(x4, y4, *acc4);
    *acc1 = _mm256_fmadd_ps(x5, y5, *acc1);
    *acc2 = _mm256_fmadd_ps(x6, y6, *acc2);
    *acc3 = _mm256_fmadd_ps(x7, y7, *acc3);
    *acc4 = _mm256_fmadd_ps(x8, y8, *acc4);

    *norm_x_acc1 = _mm256_fmadd_ps(x1, x1, *norm_x_acc1);
    *norm_x_acc2 = _mm256_fmadd_ps(x2, x2, *norm_x_acc2);
    *norm_x_acc3 = _mm256_fmadd_ps(x3, x3, *norm_x_acc3);
    *norm_x_acc4 = _mm256_fmadd_ps(x4, x4, *norm_x_acc4);
    *norm_x_acc1 = _mm256_fmadd_ps(x5, x5, *norm_x_acc1);
    *norm_x_acc2 = _mm256_fmadd_ps(x6, x6, *norm_x_acc2);
    *norm_x_acc3 = _mm256_fmadd_ps(x7, x7, *norm_x_acc3);
    *norm_x_acc4 = _mm256_fmadd_ps(x8, x8, *norm_x_acc4);

    *norm_y_acc1 = _mm256_fmadd_ps(y1, y1, *norm_y_acc1);
    *norm_y_acc2 = _mm256_fmadd_ps(y2, y2, *norm_y_acc2);
    *norm_y_acc3 = _mm256_fmadd_ps(y3, y3, *norm_y_acc3);
    *norm_y_acc4 = _mm256_fmadd_ps(y4, y4, *norm_y_acc4);
    *norm_y_acc1 = _mm256_fmadd_ps(y5, y5, *norm_y_acc1);
    *norm_y_acc2 = _mm256_fmadd_ps(y6, y6, *norm_y_acc2);
    *norm_y_acc3 = _mm256_fmadd_ps(y7, y7, *norm_y_acc3);
    *norm_y_acc4 = _mm256_fmadd_ps(y8, y8, *norm_y_acc4);
}
