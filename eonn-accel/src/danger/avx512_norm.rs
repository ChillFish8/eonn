use std::arch::x86_64::*;

use crate::danger::{offsets_avx512, sum_avx512_x8, CHUNK_0, CHUNK_1};

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

#[target_feature(enable = "avx512f")]
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
pub unsafe fn f32_x1024_avx512_nofma_norm(x: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 1024);

    let x = x.as_ptr();

    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();
    let mut acc8 = _mm512_setzero_ps();

    unrolled_loop!(
        execute_f32_x128_nofma_block_norm,
        x,
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
pub unsafe fn f32_x768_avx512_nofma_norm(x: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 768);

    let x = x.as_ptr();

    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();
    let mut acc8 = _mm512_setzero_ps();

    unrolled_loop!(
        execute_f32_x128_nofma_block_norm,
        x,
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
pub unsafe fn f32_x512_avx512_nofma_norm(x: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 512);

    let x = x.as_ptr();

    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();
    let mut acc8 = _mm512_setzero_ps();

    unrolled_loop!(
        execute_f32_x128_nofma_block_norm,
        x,
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
/// Computes the squared norm of one `f32` vector.
///
/// # Safety
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xany_avx512_nofma_norm(x: &[f32]) -> f32 {
    let len = x.len();
    let mut offset_from = len % 128;
    let x = x.as_ptr();

    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();
    let mut acc8 = _mm512_setzero_ps();

    if offset_from != 0 {
        execute_f32_xany_nofma_block_norm(x, offset_from, &mut acc1);
    }

    while offset_from < len {
        execute_f32_x128_nofma_block_norm(
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

        offset_from += 128;
    }

    sum_avx512_x8(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[target_feature(enable = "avx512f")]
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
pub unsafe fn f32_x1024_avx512_fma_norm(x: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 1024);

    let x = x.as_ptr();

    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();
    let mut acc8 = _mm512_setzero_ps();

    unrolled_loop!(
        execute_f32_x128_fma_block_norm,
        x,
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
pub unsafe fn f32_x768_avx512_fma_norm(x: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 768);

    let x = x.as_ptr();

    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();
    let mut acc8 = _mm512_setzero_ps();

    unrolled_loop!(
        execute_f32_x128_fma_block_norm,
        x,
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
pub unsafe fn f32_x512_avx512_fma_norm(x: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 512);

    let x = x.as_ptr();

    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();
    let mut acc8 = _mm512_setzero_ps();

    unrolled_loop!(
        execute_f32_x128_fma_block_norm,
        x,
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
/// Computes the squared norm of one `f32` vector.
///
/// # Safety
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xany_avx512_fma_norm(x: &[f32]) -> f32 {
    let len = x.len();
    let mut offset_from = len % 128;
    let x = x.as_ptr();

    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();
    let mut acc8 = _mm512_setzero_ps();

    if offset_from != 0 {
        execute_f32_xany_fma_block_norm(x, offset_from, &mut acc1);
    }

    while offset_from < len {
        execute_f32_x128_fma_block_norm(
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

        offset_from += 128;
    }

    sum_avx512_x8(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn execute_f32_x128_nofma_block_norm(
    x: *const f32,
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

    let x1 = _mm512_loadu_ps(x1);
    let x2 = _mm512_loadu_ps(x2);
    let x3 = _mm512_loadu_ps(x3);
    let x4 = _mm512_loadu_ps(x4);
    let x5 = _mm512_loadu_ps(x5);
    let x6 = _mm512_loadu_ps(x6);
    let x7 = _mm512_loadu_ps(x7);
    let x8 = _mm512_loadu_ps(x8);

    let r1 = _mm512_mul_ps(x1, x1);
    let r2 = _mm512_mul_ps(x2, x2);
    let r3 = _mm512_mul_ps(x3, x3);
    let r4 = _mm512_mul_ps(x4, x4);
    let r5 = _mm512_mul_ps(x5, x5);
    let r6 = _mm512_mul_ps(x6, x6);
    let r7 = _mm512_mul_ps(x7, x7);
    let r8 = _mm512_mul_ps(x8, x8);

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
unsafe fn execute_f32_x128_fma_block_norm(
    x: *const f32,
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

    let x1 = _mm512_loadu_ps(x1);
    let x2 = _mm512_loadu_ps(x2);
    let x3 = _mm512_loadu_ps(x3);
    let x4 = _mm512_loadu_ps(x4);
    let x5 = _mm512_loadu_ps(x5);
    let x6 = _mm512_loadu_ps(x6);
    let x7 = _mm512_loadu_ps(x7);
    let x8 = _mm512_loadu_ps(x8);

    *acc1 = _mm512_fmadd_ps(x1, x1, *acc1);
    *acc2 = _mm512_fmadd_ps(x2, x2, *acc2);
    *acc3 = _mm512_fmadd_ps(x3, x3, *acc3);
    *acc4 = _mm512_fmadd_ps(x4, x4, *acc4);
    *acc5 = _mm512_fmadd_ps(x5, x5, *acc5);
    *acc6 = _mm512_fmadd_ps(x6, x6, *acc6);
    *acc7 = _mm512_fmadd_ps(x7, x7, *acc7);
    *acc8 = _mm512_fmadd_ps(x8, x8, *acc8);
}

#[inline(always)]
unsafe fn execute_f32_xany_fma_block_norm(x: *const f32, n: usize, acc: &mut __m512) {
    let mut i = 0;
    while i < n {
        let cap = n - i;
        let addr = x.add(i);

        let x = if cap < 16 {
            let mask = _bzhi_u32(0xFFFFFFFF, cap as u32) as _;
            _mm512_maskz_loadu_ps(mask, addr)
        } else {
            _mm512_loadu_ps(addr)
        };

        *acc = _mm512_fmadd_ps(x, x, *acc);

        i += 16
    }
}

#[inline(always)]
unsafe fn execute_f32_xany_nofma_block_norm(x: *const f32, n: usize, acc: &mut __m512) {
    let mut i = 0;
    while i < n {
        let cap = n - i;
        let addr = x.add(i);

        let x = if cap < 16 {
            let mask = _bzhi_u32(0xFFFFFFFF, cap as u32) as _;
            _mm512_maskz_loadu_ps(mask, addr)
        } else {
            _mm512_loadu_ps(addr)
        };

        let r = _mm512_mul_ps(x, x);
        *acc = _mm512_add_ps(*acc, r);

        i += 16
    }
}

#[cfg(all(test, target_feature = "avx512f"))]
mod tests {
    use super::*;
    use crate::danger::test_utils::{assert_is_close, get_sample_vectors, simple_dot};

    #[test]
    fn test_x1024_fma_norm() {
        let (x, _) = get_sample_vectors(1024);
        let dist = unsafe { f32_x1024_avx512_fma_norm(&x) };
        assert_eq!(dist, 337.62817);
    }

    #[test]
    fn test_x1024_nofma_norm() {
        let (x, _) = get_sample_vectors(1024);
        let dist = unsafe { f32_x1024_avx512_nofma_norm(&x) };
        assert_eq!(dist, 337.62817);
    }

    #[test]
    fn test_x768_fma_norm() {
        let (x, _) = get_sample_vectors(768);
        let dist = unsafe { f32_x768_avx512_fma_norm(&x) };
        assert_eq!(dist, 254.10095);
    }

    #[test]
    fn test_x768_nofma_norm() {
        let (x, _) = get_sample_vectors(768);
        let dist = unsafe { f32_x768_avx512_nofma_norm(&x) };
        assert_eq!(dist, 254.10095);
    }

    #[test]
    fn test_x512_fma_norm() {
        let (x, _) = get_sample_vectors(512);
        let dist = unsafe { f32_x512_avx512_fma_norm(&x) };
        assert_eq!(dist, 161.06982);
    }

    #[test]
    fn test_x512_nofma_norm() {
        let (x, _) = get_sample_vectors(512);
        let dist = unsafe { f32_x512_avx512_nofma_norm(&x) };
        assert_eq!(dist, 161.06982);
    }

    #[test]
    fn test_xany_fma_norm() {
        let (x, _) = get_sample_vectors(547);
        let dist = unsafe { f32_xany_avx512_fma_norm(&x) };
        assert_is_close(dist, simple_dot(&x, &x));
    }

    #[test]
    fn test_xany_nofma_norm() {
        let (x, _) = get_sample_vectors(547);
        let dist = unsafe { f32_xany_avx512_nofma_norm(&x) };
        assert_is_close(dist, simple_dot(&x, &x));
    }
}
