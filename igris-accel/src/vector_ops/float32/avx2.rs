use crate::math::{FastMath, Math, StdMath};
use crate::vector_ops::{Avx2, DistanceOps, Fma, NoFma, Vector, X1024, X512, X768};
use std::arch::x86_64::*;

impl DistanceOps for Vector<Avx2, X1024, f32, NoFma> {
    #[inline]
    unsafe fn dot(&self, other: &Self) -> f32 {
        f32_avx2_dot::<1024>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn cosine(&self, other: &Self) -> f32 {
        f32_avx2_cosine::<StdMath, 1024>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn euclidean(&self, other: &Self) -> f32 {
        f32_avx2_euclidean::<1024>(&self.0, &other.0)
    }
}

impl DistanceOps for Vector<Avx2, X768, f32, NoFma> {
    #[inline]
    unsafe fn dot(&self, other: &Self) -> f32 {
        f32_avx2_dot::<768>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn cosine(&self, other: &Self) -> f32 {
        f32_avx2_cosine::<StdMath, 768>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn euclidean(&self, other: &Self) -> f32 {
        f32_avx2_euclidean::<768>(&self.0, &other.0)
    }
}

impl DistanceOps for Vector<Avx2, X512, f32, NoFma> {
    #[inline]
    unsafe fn dot(&self, other: &Self) -> f32 {
        f32_avx2_dot::<512>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn cosine(&self, other: &Self) -> f32 {
        f32_avx2_cosine::<StdMath, 512>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn euclidean(&self, other: &Self) -> f32 {
        f32_avx2_euclidean::<512>(&self.0, &other.0)
    }
}

impl DistanceOps for Vector<Avx2, X1024, f32, Fma> {
    #[inline]
    unsafe fn dot(&self, other: &Self) -> f32 {
        f32_avx2_fma_dot_x1024(&self.0, &other.0)
    }

    #[inline]
    unsafe fn cosine(&self, other: &Self) -> f32 {
        f32_avx2_fma_cosine::<FastMath, 1024>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn euclidean(&self, other: &Self) -> f32 {
        f32_avx2_fma_euclidean::<1024>(&self.0, &other.0)
    }
}

impl DistanceOps for Vector<Avx2, X768, f32, Fma> {
    #[inline]
    unsafe fn dot(&self, other: &Self) -> f32 {
        f32_avx2_fma_dot_x768(&self.0, &other.0)
    }

    #[inline]
    unsafe fn cosine(&self, other: &Self) -> f32 {
        f32_avx2_fma_cosine::<FastMath, 768>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn euclidean(&self, other: &Self) -> f32 {
        f32_avx2_fma_euclidean::<768>(&self.0, &other.0)
    }
}

impl DistanceOps for Vector<Avx2, X512, f32, Fma> {
    #[inline]
    unsafe fn dot(&self, other: &Self) -> f32 {
        f32_avx2_fma_dot_x512(&self.0, &other.0)
    }

    #[inline]
    unsafe fn cosine(&self, other: &Self) -> f32 {
        f32_avx2_fma_cosine::<FastMath, 512>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn euclidean(&self, other: &Self) -> f32 {
        f32_avx2_fma_euclidean::<512>(&self.0, &other.0)
    }
}

#[inline]
/// AVX2 cosine distance implementation for f32 vectors.
///
/// This implementation assumes no FMA is enabled.
///
/// Since it is more likely than not that an FMA enabled CPU is available now days,
/// this implementation is not as heavily inlined or unrolled to save some space.
unsafe fn f32_avx2_cosine<M: Math, const DIMS: usize>(a: &[f32], b: &[f32]) -> f32 {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut acc1 = _mm256_set1_ps(0.0);
    let mut acc2 = _mm256_set1_ps(0.0);
    let mut acc3 = _mm256_set1_ps(0.0);
    let mut acc4 = _mm256_set1_ps(0.0);

    let mut norm_acc_a1 = _mm256_set1_ps(0.0);
    let mut norm_acc_a2 = _mm256_set1_ps(0.0);
    let mut norm_acc_a3 = _mm256_set1_ps(0.0);
    let mut norm_acc_a4 = _mm256_set1_ps(0.0);

    let mut norm_acc_b1 = _mm256_set1_ps(0.0);
    let mut norm_acc_b2 = _mm256_set1_ps(0.0);
    let mut norm_acc_b3 = _mm256_set1_ps(0.0);
    let mut norm_acc_b4 = _mm256_set1_ps(0.0);

    // iters = num_dims / num_elements computer per pass
    for i in 0..DIMS / 64 {
        // Step * num per lane * groups of
        let base_offset = i * 64;

        let [a1, a2, a3, a4] = offsets(a_ptr, base_offset);
        let [a5, a6, a7, a8] = offsets(a_ptr, base_offset + 32);

        let [b1, b2, b3, b4] = offsets(b_ptr, base_offset);
        let [b5, b6, b7, b8] = offsets(b_ptr, base_offset + 32);

        let a1 = _mm256_loadu_ps(a1);
        let a2 = _mm256_loadu_ps(a2);
        let a3 = _mm256_loadu_ps(a3);
        let a4 = _mm256_loadu_ps(a4);
        let a5 = _mm256_loadu_ps(a5);
        let a6 = _mm256_loadu_ps(a6);
        let a7 = _mm256_loadu_ps(a7);
        let a8 = _mm256_loadu_ps(a8);

        let b1 = _mm256_loadu_ps(b1);
        let b2 = _mm256_loadu_ps(b2);
        let b3 = _mm256_loadu_ps(b3);
        let b4 = _mm256_loadu_ps(b4);
        let b5 = _mm256_loadu_ps(b5);
        let b6 = _mm256_loadu_ps(b6);
        let b7 = _mm256_loadu_ps(b7);
        let b8 = _mm256_loadu_ps(b8);

        // Compute the dot product for lanes 1-8
        let r1 = _mm256_mul_ps(a1, b1);
        let r2 = _mm256_mul_ps(a2, b2);
        let r3 = _mm256_mul_ps(a3, b3);
        let r4 = _mm256_mul_ps(a4, b4);
        let r5 = _mm256_mul_ps(a5, b5);
        let r6 = _mm256_mul_ps(a6, b6);
        let r7 = _mm256_mul_ps(a7, b7);
        let r8 = _mm256_mul_ps(a8, b8);

        // Compute the squared norm for lanes A 1-8
        let norm_a1 = _mm256_mul_ps(a1, a1);
        let norm_a2 = _mm256_mul_ps(a2, a2);
        let norm_a3 = _mm256_mul_ps(a3, a3);
        let norm_a4 = _mm256_mul_ps(a4, a4);
        let norm_a5 = _mm256_mul_ps(a5, a5);
        let norm_a6 = _mm256_mul_ps(a6, a6);
        let norm_a7 = _mm256_mul_ps(a7, a7);
        let norm_a8 = _mm256_mul_ps(a8, a8);

        // Compute the squared norm for lanes B 1-8
        let norm_b1 = _mm256_mul_ps(b1, b1);
        let norm_b2 = _mm256_mul_ps(b2, b2);
        let norm_b3 = _mm256_mul_ps(b3, b3);
        let norm_b4 = _mm256_mul_ps(b4, b4);
        let norm_b5 = _mm256_mul_ps(b5, b5);
        let norm_b6 = _mm256_mul_ps(b6, b6);
        let norm_b7 = _mm256_mul_ps(b7, b7);
        let norm_b8 = _mm256_mul_ps(b8, b8);

        // Accumulate dot product
        acc1 = _mm256_add_ps(acc1, r1);
        acc2 = _mm256_add_ps(acc2, r2);
        acc3 = _mm256_add_ps(acc3, r3);
        acc4 = _mm256_add_ps(acc4, r4);
        acc1 = _mm256_add_ps(acc1, r5);
        acc2 = _mm256_add_ps(acc2, r6);
        acc3 = _mm256_add_ps(acc3, r7);
        acc4 = _mm256_add_ps(acc4, r8);

        // Accumulate norm A
        norm_acc_a1 = _mm256_add_ps(norm_acc_a1, norm_a1);
        norm_acc_a2 = _mm256_add_ps(norm_acc_a2, norm_a2);
        norm_acc_a3 = _mm256_add_ps(norm_acc_a3, norm_a3);
        norm_acc_a4 = _mm256_add_ps(norm_acc_a4, norm_a4);
        norm_acc_a1 = _mm256_add_ps(norm_acc_a1, norm_a5);
        norm_acc_a2 = _mm256_add_ps(norm_acc_a2, norm_a6);
        norm_acc_a3 = _mm256_add_ps(norm_acc_a3, norm_a7);
        norm_acc_a4 = _mm256_add_ps(norm_acc_a4, norm_a8);

        // Accumulate norm B
        norm_acc_b1 = _mm256_add_ps(norm_acc_b1, norm_b1);
        norm_acc_b2 = _mm256_add_ps(norm_acc_b2, norm_b2);
        norm_acc_b3 = _mm256_add_ps(norm_acc_b3, norm_b3);
        norm_acc_b4 = _mm256_add_ps(norm_acc_b4, norm_b4);
        norm_acc_b1 = _mm256_add_ps(norm_acc_b1, norm_b5);
        norm_acc_b2 = _mm256_add_ps(norm_acc_b2, norm_b6);
        norm_acc_b3 = _mm256_add_ps(norm_acc_b3, norm_b7);
        norm_acc_b4 = _mm256_add_ps(norm_acc_b4, norm_b8);
    }

    acc1 = _mm256_add_ps(acc1, acc2);
    acc3 = _mm256_add_ps(acc3, acc4);

    norm_acc_a1 = _mm256_add_ps(norm_acc_a1, norm_acc_a2);
    norm_acc_a3 = _mm256_add_ps(norm_acc_a3, norm_acc_a4);

    norm_acc_b1 = _mm256_add_ps(norm_acc_b1, norm_acc_b2);
    norm_acc_b3 = _mm256_add_ps(norm_acc_b3, norm_acc_b4);

    acc1 = _mm256_add_ps(acc1, acc3);
    norm_acc_a1 = _mm256_add_ps(norm_acc_a1, norm_acc_a3);
    norm_acc_b1 = _mm256_add_ps(norm_acc_b1, norm_acc_b3);

    let result = sum_avx2(acc1);
    let norm_a = sum_avx2(norm_acc_a1);
    let norm_b = sum_avx2(norm_acc_b1);

    if norm_a == 0.0 && norm_b == 0.0 {
        0.0
    } else if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        M::sub(1.0, M::div(result, M::mul(norm_a, norm_b).sqrt()))
    }
}

#[inline]
/// AVX2 cosine distance implementation for f32 vectors.
///
/// This implementation assumes FMA is enabled.
unsafe fn f32_avx2_fma_cosine<M: Math, const DIMS: usize>(a: &[f32], b: &[f32]) -> f32 {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut acc1 = _mm256_set1_ps(0.0);
    let mut acc2 = _mm256_set1_ps(0.0);
    let mut acc3 = _mm256_set1_ps(0.0);
    let mut acc4 = _mm256_set1_ps(0.0);

    let mut norm_acc_a1 = _mm256_set1_ps(0.0);
    let mut norm_acc_a2 = _mm256_set1_ps(0.0);
    let mut norm_acc_a3 = _mm256_set1_ps(0.0);
    let mut norm_acc_a4 = _mm256_set1_ps(0.0);

    let mut norm_acc_b1 = _mm256_set1_ps(0.0);
    let mut norm_acc_b2 = _mm256_set1_ps(0.0);
    let mut norm_acc_b3 = _mm256_set1_ps(0.0);
    let mut norm_acc_b4 = _mm256_set1_ps(0.0);

    // iters = num_dims / num_elements computer per pass
    for i in 0..DIMS / 64 {
        // Step * num per lane * groups of
        let base_offset = i * 64;

        let [a1, a2, a3, a4] = offsets(a_ptr, base_offset);
        let [a5, a6, a7, a8] = offsets(a_ptr, base_offset + 32);

        let [b1, b2, b3, b4] = offsets(b_ptr, base_offset);
        let [b5, b6, b7, b8] = offsets(b_ptr, base_offset + 32);

        let a1 = _mm256_loadu_ps(a1);
        let a2 = _mm256_loadu_ps(a2);
        let a3 = _mm256_loadu_ps(a3);
        let a4 = _mm256_loadu_ps(a4);
        let a5 = _mm256_loadu_ps(a5);
        let a6 = _mm256_loadu_ps(a6);
        let a7 = _mm256_loadu_ps(a7);
        let a8 = _mm256_loadu_ps(a8);

        let b1 = _mm256_loadu_ps(b1);
        let b2 = _mm256_loadu_ps(b2);
        let b3 = _mm256_loadu_ps(b3);
        let b4 = _mm256_loadu_ps(b4);
        let b5 = _mm256_loadu_ps(b5);
        let b6 = _mm256_loadu_ps(b6);
        let b7 = _mm256_loadu_ps(b7);
        let b8 = _mm256_loadu_ps(b8);

        // Accumulate dot product
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        acc2 = _mm256_fmadd_ps(a2, b2, acc2);
        acc3 = _mm256_fmadd_ps(a3, b3, acc3);
        acc4 = _mm256_fmadd_ps(a4, b4, acc4);
        acc1 = _mm256_fmadd_ps(a5, b5, acc1);
        acc2 = _mm256_fmadd_ps(a6, b6, acc2);
        acc3 = _mm256_fmadd_ps(a7, b7, acc3);
        acc4 = _mm256_fmadd_ps(a8, b8, acc4);

        // Accumulate norm A
        norm_acc_a1 = _mm256_fmadd_ps(a1, a1, norm_acc_a1);
        norm_acc_a2 = _mm256_fmadd_ps(a2, a2, norm_acc_a2);
        norm_acc_a3 = _mm256_fmadd_ps(a3, a3, norm_acc_a3);
        norm_acc_a4 = _mm256_fmadd_ps(a4, a4, norm_acc_a4);
        norm_acc_a1 = _mm256_fmadd_ps(a5, a5, norm_acc_a1);
        norm_acc_a2 = _mm256_fmadd_ps(a6, a6, norm_acc_a2);
        norm_acc_a3 = _mm256_fmadd_ps(a7, a7, norm_acc_a3);
        norm_acc_a4 = _mm256_fmadd_ps(a8, a8, norm_acc_a4);

        // Accumulate norm B
        norm_acc_b1 = _mm256_fmadd_ps(b1, b1, norm_acc_b1);
        norm_acc_b2 = _mm256_fmadd_ps(b2, b2, norm_acc_b2);
        norm_acc_b3 = _mm256_fmadd_ps(b3, b3, norm_acc_b3);
        norm_acc_b4 = _mm256_fmadd_ps(b4, b4, norm_acc_b4);
        norm_acc_b1 = _mm256_fmadd_ps(b5, b5, norm_acc_b1);
        norm_acc_b2 = _mm256_fmadd_ps(b6, b6, norm_acc_b2);
        norm_acc_b3 = _mm256_fmadd_ps(b7, b7, norm_acc_b3);
        norm_acc_b4 = _mm256_fmadd_ps(b8, b8, norm_acc_b4);
    }

    acc1 = _mm256_add_ps(acc1, acc2);
    acc3 = _mm256_add_ps(acc3, acc4);

    norm_acc_a1 = _mm256_add_ps(norm_acc_a1, norm_acc_a2);
    norm_acc_a3 = _mm256_add_ps(norm_acc_a3, norm_acc_a4);

    norm_acc_b1 = _mm256_add_ps(norm_acc_b1, norm_acc_b2);
    norm_acc_b3 = _mm256_add_ps(norm_acc_b3, norm_acc_b4);

    acc1 = _mm256_add_ps(acc1, acc3);
    norm_acc_a1 = _mm256_add_ps(norm_acc_a1, norm_acc_a3);
    norm_acc_b1 = _mm256_add_ps(norm_acc_b1, norm_acc_b3);

    let result = sum_avx2(acc1);
    let norm_a = sum_avx2(norm_acc_a1);
    let norm_b = sum_avx2(norm_acc_b1);

    if norm_a == 0.0 && norm_b == 0.0 {
        0.0
    } else if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        M::sub(1.0, M::div(result, M::mul(norm_a, norm_b).sqrt()))
    }
}

#[inline]
/// AVX2 Euclidean distance implementation for f32 vectors.
///
/// This implementation assumes no FMA is enabled.
///
/// Since it is more likely than not that an FMA enabled CPU is available now days,
/// this implementation is not as heavily inlined or unrolled to save some space.
unsafe fn f32_avx2_euclidean<const DIMS: usize>(a: &[f32], b: &[f32]) -> f32 {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut acc1 = _mm256_set1_ps(0.0);
    let mut acc2 = _mm256_set1_ps(0.0);
    let mut acc3 = _mm256_set1_ps(0.0);
    let mut acc4 = _mm256_set1_ps(0.0);

    // iters = num_dims / num_elements computer per pass
    for i in 0..DIMS / 64 {
        // Step * num per lane * groups of
        let base_offset = i * 64;

        let [a1, a2, a3, a4] = offsets(a_ptr, base_offset);
        let [a5, a6, a7, a8] = offsets(a_ptr, base_offset + 32);

        let [b1, b2, b3, b4] = offsets(b_ptr, base_offset);
        let [b5, b6, b7, b8] = offsets(b_ptr, base_offset + 32);

        let a1 = _mm256_loadu_ps(a1);
        let a2 = _mm256_loadu_ps(a2);
        let a3 = _mm256_loadu_ps(a3);
        let a4 = _mm256_loadu_ps(a4);

        let a5 = _mm256_loadu_ps(a5);
        let a6 = _mm256_loadu_ps(a6);
        let a7 = _mm256_loadu_ps(a7);
        let a8 = _mm256_loadu_ps(a8);

        let b1 = _mm256_loadu_ps(b1);
        let b2 = _mm256_loadu_ps(b2);
        let b3 = _mm256_loadu_ps(b3);
        let b4 = _mm256_loadu_ps(b4);

        let b5 = _mm256_loadu_ps(b5);
        let b6 = _mm256_loadu_ps(b6);
        let b7 = _mm256_loadu_ps(b7);
        let b8 = _mm256_loadu_ps(b8);

        // Calculate the diff between the lanes 1-8
        let diff1 = _mm256_sub_ps(a1, b1);
        let diff2 = _mm256_sub_ps(a2, b2);
        let diff3 = _mm256_sub_ps(a3, b3);
        let diff4 = _mm256_sub_ps(a4, b4);

        let diff5 = _mm256_sub_ps(a5, b5);
        let diff6 = _mm256_sub_ps(a6, b6);
        let diff7 = _mm256_sub_ps(a7, b7);
        let diff8 = _mm256_sub_ps(a8, b8);

        // Calculated the squared difference between points a1-a8 and b1-b8
        let r1 = _mm256_mul_ps(diff1, diff1);
        let r2 = _mm256_mul_ps(diff2, diff2);
        let r3 = _mm256_mul_ps(diff3, diff3);
        let r4 = _mm256_mul_ps(diff4, diff4);

        let r5 = _mm256_mul_ps(diff5, diff5);
        let r6 = _mm256_mul_ps(diff6, diff6);
        let r7 = _mm256_mul_ps(diff7, diff7);
        let r8 = _mm256_mul_ps(diff8, diff8);

        acc1 = _mm256_add_ps(acc1, r1);
        acc2 = _mm256_add_ps(acc2, r2);
        acc3 = _mm256_add_ps(acc3, r3);
        acc4 = _mm256_add_ps(acc4, r4);

        acc1 = _mm256_add_ps(acc1, r5);
        acc2 = _mm256_add_ps(acc2, r6);
        acc3 = _mm256_add_ps(acc3, r7);
        acc4 = _mm256_add_ps(acc4, r8);
    }

    acc1 = _mm256_add_ps(acc1, acc2);
    acc3 = _mm256_add_ps(acc3, acc4);

    acc1 = _mm256_add_ps(acc1, acc3);

    sum_avx2(acc1)
}

#[inline]
/// AVX2 Euclidean distance implementation for f32 vectors.
///
/// This implementation assumes FMA is enabled.
unsafe fn f32_avx2_fma_euclidean<const DIMS: usize>(a: &[f32], b: &[f32]) -> f32 {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut acc1 = _mm256_set1_ps(0.0);
    let mut acc2 = _mm256_set1_ps(0.0);
    let mut acc3 = _mm256_set1_ps(0.0);
    let mut acc4 = _mm256_set1_ps(0.0);

    // iters = num_dims / num_elements computer per pass
    for i in 0..DIMS / 64 {
        // Step * num per lane * groups of
        let base_offset = i * 64;

        let [a1, a2, a3, a4] = offsets(a_ptr, base_offset);
        let [a5, a6, a7, a8] = offsets(a_ptr, base_offset + 32);

        let [b1, b2, b3, b4] = offsets(b_ptr, base_offset);
        let [b5, b6, b7, b8] = offsets(b_ptr, base_offset + 32);

        let a1 = _mm256_loadu_ps(a1);
        let a2 = _mm256_loadu_ps(a2);
        let a3 = _mm256_loadu_ps(a3);
        let a4 = _mm256_loadu_ps(a4);

        let a5 = _mm256_loadu_ps(a5);
        let a6 = _mm256_loadu_ps(a6);
        let a7 = _mm256_loadu_ps(a7);
        let a8 = _mm256_loadu_ps(a8);

        let b1 = _mm256_loadu_ps(b1);
        let b2 = _mm256_loadu_ps(b2);
        let b3 = _mm256_loadu_ps(b3);
        let b4 = _mm256_loadu_ps(b4);

        let b5 = _mm256_loadu_ps(b5);
        let b6 = _mm256_loadu_ps(b6);
        let b7 = _mm256_loadu_ps(b7);
        let b8 = _mm256_loadu_ps(b8);

        // Calculate the diff between the lanes 1-8
        let diff1 = _mm256_sub_ps(a1, b1);
        let diff2 = _mm256_sub_ps(a2, b2);
        let diff3 = _mm256_sub_ps(a3, b3);
        let diff4 = _mm256_sub_ps(a4, b4);

        let diff5 = _mm256_sub_ps(a5, b5);
        let diff6 = _mm256_sub_ps(a6, b6);
        let diff7 = _mm256_sub_ps(a7, b7);
        let diff8 = _mm256_sub_ps(a8, b8);

        // Calculated the squared difference between points a1-a8 and b1-b8
        // add them to the accumulator in 1 instruction.
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
        acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);
        acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);
        acc4 = _mm256_fmadd_ps(diff4, diff4, acc4);

        acc1 = _mm256_fmadd_ps(diff5, diff5, acc1);
        acc2 = _mm256_fmadd_ps(diff6, diff6, acc2);
        acc3 = _mm256_fmadd_ps(diff7, diff7, acc3);
        acc4 = _mm256_fmadd_ps(diff8, diff8, acc4);
    }

    acc1 = _mm256_add_ps(acc1, acc2);
    acc3 = _mm256_add_ps(acc3, acc4);

    acc1 = _mm256_add_ps(acc1, acc3);

    sum_avx2(acc1)
}

#[inline]
/// AVX2 dot product implementation for f32 vectors.
///
/// This implementation assumes no FMA is enabled.
///
/// Since it is more likely than not that an FMA enabled CPU is available now days,
/// this implementation is not as heavily inlined or unrolled to save some space.
unsafe fn f32_avx2_dot<const DIMS: usize>(a: &[f32], b: &[f32]) -> f32 {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut acc1 = _mm256_set1_ps(0.0);
    let mut acc2 = _mm256_set1_ps(0.0);
    let mut acc3 = _mm256_set1_ps(0.0);
    let mut acc4 = _mm256_set1_ps(0.0);

    // iters = num_dims / num_elements computer per pass
    for i in 0..DIMS / 64 {
        // Step * num per lane * groups of
        let base_offset = i * 64;

        let [a1, a2, a3, a4] = offsets(a_ptr, base_offset);
        let [a5, a6, a7, a8] = offsets(a_ptr, base_offset + 32);

        let [b1, b2, b3, b4] = offsets(b_ptr, base_offset);
        let [b5, b6, b7, b8] = offsets(b_ptr, base_offset + 32);

        let a1 = _mm256_loadu_ps(a1);
        let a2 = _mm256_loadu_ps(a2);
        let a3 = _mm256_loadu_ps(a3);
        let a4 = _mm256_loadu_ps(a4);

        let a5 = _mm256_loadu_ps(a5);
        let a6 = _mm256_loadu_ps(a6);
        let a7 = _mm256_loadu_ps(a7);
        let a8 = _mm256_loadu_ps(a8);

        let b1 = _mm256_loadu_ps(b1);
        let b2 = _mm256_loadu_ps(b2);
        let b3 = _mm256_loadu_ps(b3);
        let b4 = _mm256_loadu_ps(b4);

        let b5 = _mm256_loadu_ps(b5);
        let b6 = _mm256_loadu_ps(b6);
        let b7 = _mm256_loadu_ps(b7);
        let b8 = _mm256_loadu_ps(b8);

        let r1 = _mm256_mul_ps(a1, b1);
        let r2 = _mm256_mul_ps(a2, b2);
        let r3 = _mm256_mul_ps(a3, b3);
        let r4 = _mm256_mul_ps(a4, b4);

        let r5 = _mm256_mul_ps(a5, b5);
        let r6 = _mm256_mul_ps(a6, b6);
        let r7 = _mm256_mul_ps(a7, b7);
        let r8 = _mm256_mul_ps(a8, b8);

        acc1 = _mm256_add_ps(acc1, r1);
        acc2 = _mm256_add_ps(acc2, r2);
        acc3 = _mm256_add_ps(acc3, r3);
        acc4 = _mm256_add_ps(acc4, r4);

        acc1 = _mm256_add_ps(acc1, r5);
        acc2 = _mm256_add_ps(acc2, r6);
        acc3 = _mm256_add_ps(acc3, r7);
        acc4 = _mm256_add_ps(acc4, r8);
    }

    acc1 = _mm256_add_ps(acc1, acc2);
    acc3 = _mm256_add_ps(acc3, acc4);

    acc1 = _mm256_add_ps(acc1, acc3);

    sum_avx2(acc1)
}

macro_rules! f32_ax2_fma_dot_128_el {
    ($offset:expr, $a_ptr:expr, $b_ptr:expr, $acc1:expr, $acc2:expr, $acc3:expr, $acc4:expr) => {{
        let [a1, a2, a3, a4] = offsets($a_ptr, $offset);
        let [a5, a6, a7, a8] = offsets($a_ptr, $offset + 32);
        let [a9, a10, a11, a12] = offsets($a_ptr, $offset + 64);
        let [a13, a14, a15, a16] = offsets($a_ptr, $offset + 96);

        let [b1, b2, b3, b4] = offsets($b_ptr, $offset);
        let [b5, b6, b7, b8] = offsets($b_ptr, $offset + 32);
        let [b9, b10, b11, b12] = offsets($b_ptr, $offset + 96);
        let [b13, b14, b15, b16] = offsets($b_ptr, $offset + 96);

        let a1 = _mm256_loadu_ps(a1);
        let a2 = _mm256_loadu_ps(a2);
        let a3 = _mm256_loadu_ps(a3);
        let a4 = _mm256_loadu_ps(a4);

        let a5 = _mm256_loadu_ps(a5);
        let a6 = _mm256_loadu_ps(a6);
        let a7 = _mm256_loadu_ps(a7);
        let a8 = _mm256_loadu_ps(a8);

        let a9 = _mm256_loadu_ps(a9);
        let a10 = _mm256_loadu_ps(a10);
        let a11 = _mm256_loadu_ps(a11);
        let a12 = _mm256_loadu_ps(a12);

        let a13 = _mm256_loadu_ps(a13);
        let a14 = _mm256_loadu_ps(a14);
        let a15 = _mm256_loadu_ps(a15);
        let a16 = _mm256_loadu_ps(a16);

        let b1 = _mm256_loadu_ps(b1);
        let b2 = _mm256_loadu_ps(b2);
        let b3 = _mm256_loadu_ps(b3);
        let b4 = _mm256_loadu_ps(b4);

        let b5 = _mm256_loadu_ps(b5);
        let b6 = _mm256_loadu_ps(b6);
        let b7 = _mm256_loadu_ps(b7);
        let b8 = _mm256_loadu_ps(b8);

        let b9 = _mm256_loadu_ps(b9);
        let b10 = _mm256_loadu_ps(b10);
        let b11 = _mm256_loadu_ps(b11);
        let b12 = _mm256_loadu_ps(b12);

        let b13 = _mm256_loadu_ps(b13);
        let b14 = _mm256_loadu_ps(b14);
        let b15 = _mm256_loadu_ps(b15);
        let b16 = _mm256_loadu_ps(b16);

        $acc1 = _mm256_fmadd_ps(a1, b1, $acc1);
        $acc2 = _mm256_fmadd_ps(a2, b2, $acc2);
        $acc3 = _mm256_fmadd_ps(a3, b3, $acc3);
        $acc4 = _mm256_fmadd_ps(a4, b4, $acc4);

        $acc1 = _mm256_fmadd_ps(a5, b5, $acc1);
        $acc2 = _mm256_fmadd_ps(a6, b6, $acc2);
        $acc3 = _mm256_fmadd_ps(a7, b7, $acc3);
        $acc4 = _mm256_fmadd_ps(a8, b8, $acc4);

        $acc1 = _mm256_fmadd_ps(a9, b9, $acc1);
        $acc2 = _mm256_fmadd_ps(a10, b10, $acc2);
        $acc3 = _mm256_fmadd_ps(a11, b11, $acc3);
        $acc4 = _mm256_fmadd_ps(a12, b12, $acc4);

        $acc1 = _mm256_fmadd_ps(a13, b13, $acc1);
        $acc2 = _mm256_fmadd_ps(a14, b14, $acc2);
        $acc3 = _mm256_fmadd_ps(a15, b15, $acc3);
        $acc4 = _mm256_fmadd_ps(a16, b16, $acc4);
    }};
}

/// Effectively used as a manual loop unroll.
///
/// This produces `n_offsets` numbers of `f32_ax2_fma_128_el` calls.
macro_rules! f32_ax2_fma_dot_lanes_each {
    ($a_ptr:expr, $b_ptr:expr, $acc1:expr, $acc2:expr, $acc3:expr, $acc4:expr, offsets => $($offset:expr $(,)?)*) => {{
        $(
            f32_ax2_fma_dot_128_el!(
                $offset,
                $a_ptr,
                $b_ptr,
                $acc1,
                $acc2,
                $acc3,
                $acc4
            );
        )*
    }};
}

#[inline]
/// AVX2 dot product implementation for f32 vectors.
///
/// This implementation assumes FMA is enabled.
unsafe fn f32_avx2_fma_dot_x1024(a: &[f32], b: &[f32]) -> f32 {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut acc1 = _mm256_set1_ps(0.0);
    let mut acc2 = _mm256_set1_ps(0.0);
    let mut acc3 = _mm256_set1_ps(0.0);
    let mut acc4 = _mm256_set1_ps(0.0);

    f32_ax2_fma_dot_lanes_each!(
        a_ptr,
        b_ptr,
        acc1,
        acc2,
        acc3,
        acc4,
        offsets => 0, 128, 256, 384, 512, 640, 768, 896
    );

    acc1 = _mm256_add_ps(acc1, acc2);
    acc3 = _mm256_add_ps(acc3, acc4);

    acc1 = _mm256_add_ps(acc1, acc3);

    sum_avx2(acc1)
}

#[inline]
/// AVX2 dot product implementation for f32 vectors.
///
/// This implementation assumes FMA is enabled.
unsafe fn f32_avx2_fma_dot_x768(a: &[f32], b: &[f32]) -> f32 {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut acc1 = _mm256_set1_ps(0.0);
    let mut acc2 = _mm256_set1_ps(0.0);
    let mut acc3 = _mm256_set1_ps(0.0);
    let mut acc4 = _mm256_set1_ps(0.0);

    f32_ax2_fma_dot_lanes_each!(
        a_ptr,
        b_ptr,
        acc1,
        acc2,
        acc3,
        acc4,
        offsets => 0, 128, 256, 384, 512, 640
    );

    acc1 = _mm256_add_ps(acc1, acc2);
    acc3 = _mm256_add_ps(acc3, acc4);

    acc1 = _mm256_add_ps(acc1, acc3);

    sum_avx2(acc1)
}

#[inline]
/// AVX2 dot product implementation for f32 vectors.
///
/// This implementation assumes FMA is enabled.
unsafe fn f32_avx2_fma_dot_x512(a: &[f32], b: &[f32]) -> f32 {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut acc1 = _mm256_set1_ps(0.0);
    let mut acc2 = _mm256_set1_ps(0.0);
    let mut acc3 = _mm256_set1_ps(0.0);
    let mut acc4 = _mm256_set1_ps(0.0);

    f32_ax2_fma_dot_lanes_each!(
        a_ptr,
        b_ptr,
        acc1,
        acc2,
        acc3,
        acc4,
        offsets => 0, 128, 256, 384
    );

    acc1 = _mm256_add_ps(acc1, acc2);
    acc3 = _mm256_add_ps(acc3, acc4);

    acc1 = _mm256_add_ps(acc1, acc3);

    sum_avx2(acc1)
}

#[inline]
unsafe fn sum_avx2(v: __m256) -> f32 {
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

#[inline(always)]
unsafe fn offsets(ptr: *const f32, offset: usize) -> [*const f32; 4] {
    [
        ptr.add(offset),
        ptr.add(offset + 8),
        ptr.add(offset + 16),
        ptr.add(offset + 24),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{FastMath, StdMath};
    use crate::vector_ops::float32::fallback::{fallback_cosine, fallback_dot_product};

    #[test]
    fn test_avx2_rollup_sum() {
        unsafe {
            let acc1 = _mm256_set1_ps(0.0);
            let acc2 = _mm256_set1_ps(1.0);
            let acc = _mm256_add_ps(acc1, acc2);
            let sum = sum_avx2(acc);
            assert_eq!(sum, 8.0);
        }
    }

    #[test]
    fn test_avx2_dot_product() {
        let mut v1 = Vec::with_capacity(1024);
        let mut v2 = Vec::with_capacity(1024);
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }

        let v = unsafe { f32_avx2_dot::<1024>(&v1, &v2) };
        let expected = unsafe { fallback_dot_product::<StdMath, 1024>(&v1, &v2) };
        println!("Av2 DOT: {v} vs {expected}");
    }

    #[test]
    fn test_avx2_fma_dot_product() {
        let mut v1 = Vec::with_capacity(1024);
        let mut v2 = Vec::with_capacity(1024);
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }

        let v = unsafe { f32_avx2_fma_dot_x1024(&v1, &v2) };
        let expected = unsafe { fallback_dot_product::<StdMath, 1024>(&v1, &v2) };
        println!("FMA DOT: {v} vs {expected}");
    }

    #[test]
    fn test_avx2_cosine() {
        let mut v1 = Vec::with_capacity(1024);
        let mut v2 = Vec::with_capacity(1024);
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }

        let v = unsafe { f32_avx2_cosine::<StdMath, 1024>(&v1, &v2) };
        let expected = unsafe { fallback_cosine::<StdMath, 1024>(&v1, &v2) };
        println!("Av2 COSINE: {v} vs {expected}");
    }

    #[test]
    fn test_avx2_fma_cosine() {
        let mut v1 = Vec::with_capacity(1024);
        let mut v2 = Vec::with_capacity(1024);
        for _ in 0..1024 {
            v1.push(rand::random());
            v2.push(rand::random());
        }

        let v = unsafe { f32_avx2_fma_cosine::<FastMath, 1024>(&v1, &v2) };
        let expected = unsafe { fallback_cosine::<FastMath, 1024>(&v1, &v2) };
        println!("FMA COSINE: {v} vs {expected}");
    }
}
