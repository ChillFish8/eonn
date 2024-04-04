use std::arch::x86_64::*;

use super::arch::Avx2;
use super::{DistanceOps, Fma, NoFma, Vector};

/// Float 32 vectors with 512 dimensions.
pub struct X1024;
/// Float 32 vectors with 512 dimensions.
pub struct X768;
/// Float 32 vectors with 512 dimensions.
pub struct X512;

impl DistanceOps for Vector<Avx2, X1024, f32, NoFma> {
    /// AVX2 enabled dot product.
    ///
    /// Since we have a 1024 element vector, we end up with 128 steps to complete.
    unsafe fn dot(&self, other: &Self) -> f32 {
        f32_avx2_dot::<1024>(&self.0, &other.0)
    }

    unsafe fn cosine(&self, other: &Self) -> f32 {
        todo!()
    }

    unsafe fn euclidean(&self, other: &Self) -> f32 {
        todo!()
    }
}

impl DistanceOps for Vector<Avx2, X768, f32, NoFma> {
    /// AVX2 enabled dot product.
    ///
    /// Since we have a 1024 element vector, we end up with 128 steps to complete.
    unsafe fn dot(&self, other: &Self) -> f32 {
        f32_avx2_dot::<768>(&self.0, &other.0)
    }

    unsafe fn cosine(&self, other: &Self) -> f32 {
        todo!()
    }

    unsafe fn euclidean(&self, other: &Self) -> f32 {
        todo!()
    }
}

impl DistanceOps for Vector<Avx2, X512, f32, NoFma> {
    /// AVX2 enabled dot product.
    ///
    /// Since we have a 1024 element vector, we end up with 128 steps to complete.
    unsafe fn dot(&self, other: &Self) -> f32 {
        f32_avx2_dot::<512>(&self.0, &other.0)
    }

    unsafe fn cosine(&self, other: &Self) -> f32 {
        todo!()
    }

    unsafe fn euclidean(&self, other: &Self) -> f32 {
        todo!()
    }
}

impl DistanceOps for Vector<Avx2, X1024, f32, Fma> {
    /// AVX2 enabled dot product.
    ///
    /// Since we have a 1024 element vector, we end up with 128 steps to complete.
    unsafe fn dot(&self, other: &Self) -> f32 {
        f32_avx2_fma_dot_x1024(&self.0, &other.0)
    }

    unsafe fn cosine(&self, other: &Self) -> f32 {
        todo!()
    }

    unsafe fn euclidean(&self, other: &Self) -> f32 {
        todo!()
    }
}

impl DistanceOps for Vector<Avx2, X768, f32, Fma> {
    /// AVX2 enabled dot product.
    ///
    /// Since we have a 1024 element vector, we end up with 128 steps to complete.
    unsafe fn dot(&self, other: &Self) -> f32 {
        f32_avx2_fma_dot_x768(&self.0, &other.0)
    }

    unsafe fn cosine(&self, other: &Self) -> f32 {
        todo!()
    }

    unsafe fn euclidean(&self, other: &Self) -> f32 {
        todo!()
    }
}

impl DistanceOps for Vector<Avx2, X512, f32, Fma> {
    /// AVX2 enabled dot product.
    ///
    /// Since we have a 1024 element vector, we end up with 128 steps to complete.
    unsafe fn dot(&self, other: &Self) -> f32 {
        f32_avx2_fma_dot_x512(&self.0, &other.0)
    }

    unsafe fn cosine(&self, other: &Self) -> f32 {
        todo!()
    }

    unsafe fn euclidean(&self, other: &Self) -> f32 {
        todo!()
    }
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

    fn basic_impl(a: &[f32], b: &[f32]) -> f32 {
        let mut res = 0.0;
        for i in 0..a.len() {
            res += a[i] * b[i];
        }
        res
    }

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
        println!("Av2: {v} vs {}", basic_impl(&v1, &v2));
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
        println!("FMA: {v} vs {}", basic_impl(&v1, &v2));
    }
}
