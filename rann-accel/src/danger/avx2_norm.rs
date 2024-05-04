use std::arch::x86_64::*;

use crate::danger::utils::{CHUNK_0, CHUNK_1};
use crate::danger::{offsets_avx2, rollup_x8, sum_avx2};
use crate::math::*;

macro_rules! unrolled_loop {
    (
        $executor:ident,
        $x:ident,
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

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the squared norm of one `[f32; 1024]` vector.
///
/// # Safety
///
/// Vectors **MUST** be `1024` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x1024_avx2_nofma_norm(x: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 1024);

    let x = x.as_ptr();

    _mm_prefetch::<_MM_HINT_T1>(x.cast());

    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();
    let mut acc8 = _mm256_setzero_ps();

    unrolled_loop!(
        execute_f32_x64_nofma_block_norm,
        x,
        &mut acc1,
        &mut acc2,
        &mut acc3,
        &mut acc4,
        &mut acc5,
        &mut acc6,
        &mut acc7,
        &mut acc8,
        offsets =>
            0, 64, 128, 192,
            256, 320, 384, 448,
            512, 576, 640, 704,
            768, 832, 896, 960,
    );

    let acc = rollup_x8(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
    sum_avx2(acc)
}

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the squared norm of one `[f32; 768]` vector.
///
/// # Safety
///
/// Vectors **MUST** be `768` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x768_avx2_nofma_norm(x: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 768);

    let x = x.as_ptr();

    _mm_prefetch::<_MM_HINT_T1>(x.cast());

    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();
    let mut acc8 = _mm256_setzero_ps();

    unrolled_loop!(
        execute_f32_x64_nofma_block_norm,
        x,
        &mut acc1,
        &mut acc2,
        &mut acc3,
        &mut acc4,
        &mut acc5,
        &mut acc6,
        &mut acc7,
        &mut acc8,
        offsets =>
            0, 64, 128, 192,
            256, 320, 384, 448,
            512, 576, 640, 704
    );

    let acc = rollup_x8(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
    sum_avx2(acc)
}

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the squared norm of one `[f32; 512]` vector.
///
/// # Safety
///
/// Vectors **MUST** be `512` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x512_avx2_nofma_norm(x: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 512);

    let x = x.as_ptr();

    _mm_prefetch::<_MM_HINT_T1>(x.cast());

    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();
    let mut acc8 = _mm256_setzero_ps();

    unrolled_loop!(
        execute_f32_x64_nofma_block_norm,
        x,
        &mut acc1,
        &mut acc2,
        &mut acc3,
        &mut acc4,
        &mut acc5,
        &mut acc6,
        &mut acc7,
        &mut acc8,
        offsets =>
            0, 64, 128, 192,
            256, 320, 384, 448
    );

    let acc = rollup_x8(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
    sum_avx2(acc)
}

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the squared norm of one f32 vector.
///
/// This vector can be any size although it may perform worse than
/// the specialized size handling.
///
/// # Safety
///
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xany_avx2_nofma_norm(x: &[f32]) -> f32 {
    let len = x.len();
    let mut offset_from = len % 64;
    let mut total = 0.0;

    if offset_from != 0 {
        total = linear_norm::<StdMath>(x, offset_from);
    }

    let x = x.as_ptr();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();
    let mut acc8 = _mm256_setzero_ps();

    while offset_from < len {
        execute_f32_x64_nofma_block_norm(
            x.add(offset_from),
            &mut acc1,
            &mut acc2,
            &mut acc3,
            &mut acc4,
            &mut acc5,
            &mut acc6,
            &mut acc7,
            &mut acc8,
        );

        offset_from += 64;
    }

    let acc = rollup_x8(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
    total + sum_avx2(acc)
}

#[cfg(feature = "nightly")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the squared norm of one `[f32; 1024]` vector.
///
/// # Safety
///
/// Vectors **MUST** be `1024` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x1024_avx2_fma_norm(x: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 1024);

    let x = x.as_ptr();

    _mm_prefetch::<_MM_HINT_T1>(x.cast());

    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();
    let mut acc8 = _mm256_setzero_ps();

    unrolled_loop!(
        execute_f32_x64_fma_block_norm,
        x,
        &mut acc1,
        &mut acc2,
        &mut acc3,
        &mut acc4,
        &mut acc5,
        &mut acc6,
        &mut acc7,
        &mut acc8,
        offsets =>
            0, 64, 128, 192,
            256, 320, 384, 448,
            512, 576, 640, 704,
            768, 832, 896, 960,
    );

    let acc = rollup_x8(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
    sum_avx2(acc)
}

#[cfg(feature = "nightly")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the squared norm of one `[f32; 768]` vector.
///
/// # Safety
///
/// Vectors **MUST** be `768` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x768_avx2_fma_norm(x: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 768);

    let x = x.as_ptr();

    _mm_prefetch::<_MM_HINT_T1>(x.cast());

    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();
    let mut acc8 = _mm256_setzero_ps();

    unrolled_loop!(
        execute_f32_x64_fma_block_norm,
        x,
        &mut acc1,
        &mut acc2,
        &mut acc3,
        &mut acc4,
        &mut acc5,
        &mut acc6,
        &mut acc7,
        &mut acc8,
        offsets =>
            0, 64, 128, 192,
            256, 320, 384, 448,
            512, 576, 640, 704
    );

    let acc = rollup_x8(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
    sum_avx2(acc)
}

#[cfg(feature = "nightly")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the squared norm of one `[f32; 512]` vector.
///
/// # Safety
///
/// Vectors **MUST** be `512` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x512_avx2_fma_norm(x: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 512);

    let x = x.as_ptr();

    _mm_prefetch::<_MM_HINT_T1>(x.cast());

    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();
    let mut acc8 = _mm256_setzero_ps();

    unrolled_loop!(
        execute_f32_x64_fma_block_norm,
        x,
        &mut acc1,
        &mut acc2,
        &mut acc3,
        &mut acc4,
        &mut acc5,
        &mut acc6,
        &mut acc7,
        &mut acc8,
        offsets =>
            0, 64, 128, 192,
            256, 320, 384, 448
    );

    let acc = rollup_x8(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
    sum_avx2(acc)
}

#[cfg(feature = "nightly")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the squared norm of one f32 vector.
///
/// This vector can be any size although it may perform worse than
/// the specialized size handling.
///
/// # Safety
///
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xany_avx2_fma_norm(x: &[f32]) -> f32 {
    let len = x.len();
    let mut offset_from = len % 64;
    let mut total = 0.0;

    if offset_from != 0 {
        total = linear_norm::<FastMath>(x, offset_from);
    }

    let x = x.as_ptr();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();
    let mut acc8 = _mm256_setzero_ps();

    while offset_from < len {
        execute_f32_x64_fma_block_norm(
            x.add(offset_from),
            &mut acc1,
            &mut acc2,
            &mut acc3,
            &mut acc4,
            &mut acc5,
            &mut acc6,
            &mut acc7,
            &mut acc8,
        );

        offset_from += 64;
    }

    let acc = rollup_x8(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
    total + sum_avx2(acc)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn execute_f32_x64_nofma_block_norm(
    x: *const f32,
    acc1: &mut __m256,
    acc2: &mut __m256,
    acc3: &mut __m256,
    acc4: &mut __m256,
    acc5: &mut __m256,
    acc6: &mut __m256,
    acc7: &mut __m256,
    acc8: &mut __m256,
) {
    let [x1, x2, x3, x4] = offsets_avx2::<CHUNK_0>(x);
    let [x5, x6, x7, x8] = offsets_avx2::<CHUNK_1>(x);

    let x1 = _mm256_loadu_ps(x1);
    let x2 = _mm256_loadu_ps(x2);
    let x3 = _mm256_loadu_ps(x3);
    let x4 = _mm256_loadu_ps(x4);
    let x5 = _mm256_loadu_ps(x5);
    let x6 = _mm256_loadu_ps(x6);
    let x7 = _mm256_loadu_ps(x7);
    let x8 = _mm256_loadu_ps(x8);

    let r1 = _mm256_mul_ps(x1, x1);
    let r2 = _mm256_mul_ps(x2, x2);
    let r3 = _mm256_mul_ps(x3, x3);
    let r4 = _mm256_mul_ps(x4, x4);
    let r5 = _mm256_mul_ps(x5, x5);
    let r6 = _mm256_mul_ps(x6, x6);
    let r7 = _mm256_mul_ps(x7, x7);
    let r8 = _mm256_mul_ps(x8, x8);

    *acc1 = _mm256_add_ps(*acc1, r1);
    *acc2 = _mm256_add_ps(*acc2, r2);
    *acc3 = _mm256_add_ps(*acc3, r3);
    *acc4 = _mm256_add_ps(*acc4, r4);
    *acc5 = _mm256_add_ps(*acc5, r5);
    *acc6 = _mm256_add_ps(*acc6, r6);
    *acc7 = _mm256_add_ps(*acc7, r7);
    *acc8 = _mm256_add_ps(*acc8, r8);
}

#[cfg(feature = "nightly")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn execute_f32_x64_fma_block_norm(
    x: *const f32,
    acc1: &mut __m256,
    acc2: &mut __m256,
    acc3: &mut __m256,
    acc4: &mut __m256,
    acc5: &mut __m256,
    acc6: &mut __m256,
    acc7: &mut __m256,
    acc8: &mut __m256,
) {
    let [x1, x2, x3, x4] = offsets_avx2::<CHUNK_0>(x);
    let [x5, x6, x7, x8] = offsets_avx2::<CHUNK_1>(x);

    let x1 = _mm256_loadu_ps(x1);
    let x2 = _mm256_loadu_ps(x2);
    let x3 = _mm256_loadu_ps(x3);
    let x4 = _mm256_loadu_ps(x4);
    let x5 = _mm256_loadu_ps(x5);
    let x6 = _mm256_loadu_ps(x6);
    let x7 = _mm256_loadu_ps(x7);
    let x8 = _mm256_loadu_ps(x8);

    *acc1 = _mm256_fmadd_ps(x1, x1, *acc1);
    *acc2 = _mm256_fmadd_ps(x2, x2, *acc2);
    *acc3 = _mm256_fmadd_ps(x3, x3, *acc3);
    *acc4 = _mm256_fmadd_ps(x4, x4, *acc4);
    *acc5 = _mm256_fmadd_ps(x5, x5, *acc5);
    *acc6 = _mm256_fmadd_ps(x6, x6, *acc6);
    *acc7 = _mm256_fmadd_ps(x7, x7, *acc7);
    *acc8 = _mm256_fmadd_ps(x8, x8, *acc8);
}

#[inline]
unsafe fn linear_norm<M: Math>(
    x: &[f32],
    n: usize,
) -> f32 {
    let mut total = 0.0;

    for i in 0..n {
        let x = *x.get_unchecked(i);
        total = M::add(total, M::mul(x, x))
    }

    total
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::danger::test_utils::get_sample_vectors;

    #[cfg(feature = "nightly")]
    #[test]
    fn test_x1024_fma_norm() {
        let (x, _) = get_sample_vectors(1024);
        let dist = unsafe { f32_x1024_avx2_fma_norm(&x) };
        let expected = 337.62814;
        assert_eq!(dist, expected);
    }

    #[test]
    fn test_x1024_nofma_norm() {
        let (x, _) = get_sample_vectors(1024);
        let dist = unsafe { f32_x1024_avx2_nofma_norm(&x) };
        assert_eq!(dist, 337.62814);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_x768_fma_norm() {
        let (x, _) = get_sample_vectors(768);
        let dist = unsafe { f32_x768_avx2_fma_norm(&x) };
        assert_eq!(dist, 254.10095);
    }

    #[test]
    fn test_x768_nofma_norm() {
        let (x, _) = get_sample_vectors(768);
        let dist = unsafe { f32_x768_avx2_nofma_norm(&x) };
        assert_eq!(dist, 254.10095);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_x512_fma_norm() {
        let (x, _) = get_sample_vectors(512);
        let dist = unsafe { f32_x512_avx2_fma_norm(&x) };
        assert_eq!(dist, 161.06982);
    }

    #[test]
    fn test_x512_nofma_norm() {
        let (x, _) = get_sample_vectors(512);
        let dist = unsafe { f32_x512_avx2_nofma_norm(&x) };
        assert_eq!(dist, 161.06982);
    }
}
