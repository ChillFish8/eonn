use std::arch::x86_64::*;
use std::array;

use crate::math::Math;

#[inline(always)]
/// Performs a sum of all packed values in the provided [__m256] register
/// returning the resulting f32 value.
pub(crate) unsafe fn sum_avx2(v: __m256) -> f32 {
    let left_half = _mm256_extractf128_ps::<1>(v);
    let right_half = _mm256_castps256_ps128(v);
    let sum_quad = _mm_add_ps(left_half, right_half);

    let left_half = sum_quad;
    let right_half = _mm_movehl_ps(sum_quad, sum_quad);
    let sum_dual = _mm_add_ps(left_half, right_half);

    let left_half = sum_dual;
    let right_half = _mm_shuffle_ps::<0x1>(sum_dual, sum_dual);
    let sum = _mm_add_ss(left_half, right_half);

    _mm_cvtss_f32(sum)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
/// Rolls up 8 [__m256] registers into 1 summing them together.
pub(crate) unsafe fn rollup_x8(
    mut acc1: __m256,
    acc2: __m256,
    mut acc3: __m256,
    acc4: __m256,
    mut acc5: __m256,
    acc6: __m256,
    mut acc7: __m256,
    acc8: __m256,
) -> __m256 {
    acc1 = _mm256_add_ps(acc1, acc2);
    acc3 = _mm256_add_ps(acc3, acc4);
    acc5 = _mm256_add_ps(acc5, acc6);
    acc7 = _mm256_add_ps(acc7, acc8);

    acc1 = _mm256_add_ps(acc1, acc3);
    acc5 = _mm256_add_ps(acc5, acc7);

    _mm256_add_ps(acc1, acc5)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
/// Rolls up 4 [__m256] registers into 1 summing them together.
pub(crate) unsafe fn rollup_x4(
    mut acc1: __m256,
    acc2: __m256,
    mut acc3: __m256,
    acc4: __m256,
) -> __m256 {
    acc1 = _mm256_add_ps(acc1, acc2);
    acc3 = _mm256_add_ps(acc3, acc4);

    _mm256_add_ps(acc1, acc3)
}

#[inline(always)]
pub(crate) unsafe fn offsets(ptr: *const f32, offset: usize) -> [*const f32; 4] {
    [
        ptr.add(offset),
        ptr.add(offset + 8),
        ptr.add(offset + 16),
        ptr.add(offset + 24),
    ]
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
/// Sums 8 scalar accumulators into one f32 value.
pub fn rollup_scalar_x8<M: Math>(
    mut acc1: f32,
    acc2: f32,
    mut acc3: f32,
    acc4: f32,
    mut acc5: f32,
    acc6: f32,
    mut acc7: f32,
    acc8: f32,
) -> f32 {
    acc1 = M::add(acc1, acc2);
    acc3 = M::add(acc3, acc4);
    acc5 = M::add(acc5, acc6);
    acc7 = M::add(acc7, acc8);

    acc1 = M::add(acc1, acc3);
    acc5 = M::add(acc5, acc7);

    M::add(acc1, acc5)
}
