use std::arch::x86_64::*;

#[cfg(feature = "nightly")]
use crate::danger::f32_xany_fallback_fma_dot;
use crate::danger::utils::{CHUNK_0, CHUNK_1};
use crate::danger::{f32_xany_fallback_nofma_dot, offsets_avx2, rollup_x8, sum_avx2};

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the squared norm of one `[f32; DIMS]` vector.
///
/// # Safety
///
/// DIMS **MUST** be a multiple of `64` and vector must be `DIMS` in length,
/// otherwise this routine will become immediately UB due to out of bounds pointer accesses.
///
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xconst_avx2_nofma_norm<const DIMS: usize>(x: &[f32]) -> f32 {
    let x = x.as_ptr();

    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();
    let mut acc8 = _mm256_setzero_ps();

    let mut i = 0;
    while i < DIMS {
        execute_f32_x64_nofma_block_norm(
            x.add(i),
            &mut acc1,
            &mut acc2,
            &mut acc3,
            &mut acc4,
            &mut acc5,
            &mut acc6,
            &mut acc7,
            &mut acc8,
        );

        i += 64;
    }

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
        let subsection = &x[..offset_from];
        total = f32_xany_fallback_nofma_dot(subsection, subsection);
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
/// Computes the squared norm of one `[f32; DIMS]` vector.
///
/// # Safety
///
/// DIMS **MUST** be a multiple of `64` and vector must be `DIMS` in length,
/// otherwise this routine will become immediately UB due to out of bounds pointer accesses.
///
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xconst_avx2_fma_norm<const DIMS: usize>(x: &[f32]) -> f32 {
    let x = x.as_ptr();

    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();
    let mut acc8 = _mm256_setzero_ps();

    let mut i = 0;
    while i < DIMS {
        execute_f32_x64_fma_block_norm(
            x.add(i),
            &mut acc1,
            &mut acc2,
            &mut acc3,
            &mut acc4,
            &mut acc5,
            &mut acc6,
            &mut acc7,
            &mut acc8,
        );

        i += 64;
    }

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
        let subsection = &x[..offset_from];
        total = f32_xany_fallback_fma_dot(subsection, subsection);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_is_close, get_sample_vectors, simple_dot};

    #[cfg(feature = "nightly")]
    #[test]
    fn test_xany_fma_norm() {
        let (x, _) = get_sample_vectors(127);
        let dist = unsafe { f32_xany_avx2_fma_norm(&x) };
        assert_is_close(dist, simple_dot(&x, &x));
    }

    #[test]
    fn test_xany_nofma_norm() {
        let (x, _) = get_sample_vectors(127);
        let dist = unsafe { f32_xany_avx2_nofma_norm(&x) };
        assert_is_close(dist, simple_dot(&x, &x));
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_xconst_fma_norm() {
        let (x, _) = get_sample_vectors(1024);
        let dist = unsafe { f32_xconst_avx2_fma_norm::<1024>(&x) };
        assert_is_close(dist, simple_dot(&x, &x));
    }

    #[test]
    fn test_xconst_nofma_norm() {
        let (x, _) = get_sample_vectors(1024);
        let dist = unsafe { f32_xconst_avx2_nofma_norm::<1024>(&x) };
        assert_is_close(dist, simple_dot(&x, &x));
    }
}
