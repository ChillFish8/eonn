use std::arch::x86_64::*;

use super::arch::Avx2;
use super::{DistanceOps, Vector};

/// Float 32 vectors with 512 dimensions.
pub struct X1024;
/// Float 32 vectors with 512 dimensions.
pub struct X768;
/// Float 32 vectors with 512 dimensions.
pub struct X512;


impl DistanceOps for Vector<Avx2, X1024, f32> {    
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

#[inline]
unsafe fn f32_avx2_dot<const DIMS: usize>(a: &[f32], b: &[f32]) -> f32 {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut acc1 = _mm256_set1_ps(0.0);
    let mut acc2 = _mm256_set1_ps(0.0);
    let mut acc3 = _mm256_set1_ps(0.0);
    let mut acc4 = _mm256_set1_ps(0.0);
    
    // iters = num_dims / (lane_size_for_f32 * chunk size)
    for i in 0..DIMS / (8 * 8) {
        // Step * num per lane * groups of 
        let base_offset = i * 8 * 8;

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
        println!("{v}");
    }
}