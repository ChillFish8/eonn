use std::arch::x86_64::*;
use std::{mem, ptr};

use crate::danger::{
    copy_masked_avx512_register_to,
    load_one_variable_size_avx512,
    offsets_avx512,
    CHUNK_0,
    CHUNK_1,
};

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the horizontal maximum of the given vector that is `[f32; DIMS]`.
///
/// # Safety
///
/// `DIMS` **MUST** be a multiple of `128`, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xconst_avx512_nofma_max_horizontal<const DIMS: usize>(
    arr: &[f32],
) -> f32 {
    debug_assert_eq!(arr.len(), DIMS, "Array length must match DIMS");
    debug_assert_eq!(DIMS % 128, 0, "DIMS must be a multiple of 128");

    let arr = arr.as_ptr();

    _mm_prefetch::<_MM_HINT_T1>(arr.cast());

    let mut acc1 = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut acc2 = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut acc3 = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut acc4 = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut acc5 = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut acc6 = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut acc7 = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut acc8 = _mm512_set1_ps(f32::NEG_INFINITY);

    let mut i = 0;
    while i < DIMS {
        max_by_x128_horizontal(
            arr.add(i),
            &mut acc1,
            &mut acc2,
            &mut acc3,
            &mut acc4,
            &mut acc5,
            &mut acc6,
            &mut acc7,
            &mut acc8,
        );

        i += 128;
    }

    rollup_max_acc(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the vertical maximum of the given vector that is `[[f32; DIMS]; N]`.
///
/// # Safety
///
/// `DIMS` **MUST** be a multiple of `128`, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xconst_avx512_nofma_max_vertical<const DIMS: usize>(
    matrix: &[&[f32]],
) -> Vec<f32> {
    debug_assert_eq!(DIMS % 128, 0, "DIMS must be a multiple of 128");

    let mut max_values = vec![0.0; DIMS];
    let max_values_ptr = max_values.as_mut_ptr();

    // We work our way horizontally by taking steps of 128 and finding
    // the max of for each of the lanes vertically through the matrix.
    // TODO: I am unsure how hard this is on the cache or if there is a smarter
    //       way to write this.
    let mut i = 0;
    while i < DIMS {
        let mut acc1 = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut acc2 = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut acc3 = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut acc4 = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut acc5 = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut acc6 = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut acc7 = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut acc8 = _mm512_set1_ps(f32::NEG_INFINITY);

        // Vertical max of the 128 elements.
        for m in 0..matrix.len() {
            let arr = *matrix.get_unchecked(m);
            let arr = arr.as_ptr();

            max_by_x128_horizontal(
                arr.add(i),
                &mut acc1,
                &mut acc2,
                &mut acc3,
                &mut acc4,
                &mut acc5,
                &mut acc6,
                &mut acc7,
                &mut acc8,
            );
        }

        let merged = [acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8];

        let result = mem::transmute::<[__m512; 8], [f32; 128]>(merged);
        ptr::copy_nonoverlapping(result.as_ptr(), max_values_ptr.add(i), result.len());

        i += 128;
    }

    max_values
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the horizontal maximum of the given vector that is `[f32; N]`.
///
/// # Safety
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xany_avx512_nofma_max_horizontal(arr: &[f32]) -> f32 {
    let dims = arr.len();
    let arr = arr.as_ptr();

    let mut acc1 = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut acc2 = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut acc3 = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut acc4 = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut acc5 = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut acc6 = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut acc7 = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut acc8 = _mm512_set1_ps(f32::NEG_INFINITY);

    let mut offset_from = dims % 128;
    if offset_from != 0 {
        let mut i = 0;
        while i < offset_from {
            let n = offset_from - i;

            let x = load_one_variable_size_avx512(arr.add(i), n);
            acc1 = _mm512_max_ps(acc1, x);

            i += 16;
        }
    }

    while offset_from < dims {
        max_by_x128_horizontal(
            arr.add(offset_from),
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

    rollup_max_acc(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the vertical maximum of the given vector that is `[[f32; N]; N2]`.
///
/// # Safety
///
/// The size of each array in the matrix must be equal otherwise out of bounds
/// access can occur.
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xany_avx512_nofma_max_vertical(matrix: &[&[f32]]) -> Vec<f32> {
    let dims = matrix[0].len();

    let mut max_values = vec![0.0; dims];
    let max_values_ptr = max_values.as_mut_ptr();
    let mut offset_from = dims % 128;

    if offset_from != 128 {
        let mut i = 0;
        while i < offset_from {
            let n = offset_from - i;

            let mut acc = _mm512_set1_ps(f32::NEG_INFINITY);

            for m in 0..matrix.len() {
                let arr = *matrix.get_unchecked(m);
                debug_assert_eq!(arr.len(), dims);

                let arr = arr.as_ptr();
                let x = load_one_variable_size_avx512(arr.add(i), n);
                acc = _mm512_max_ps(acc, x);
            }

            copy_masked_avx512_register_to(max_values_ptr.add(i), acc, n);

            i += 16;
        }
    }

    // We work our way horizontally by taking steps of 128 and finding
    // the max of for each of the lanes vertically through the matrix.
    // TODO: I am unsure how hard this is on the cache or if there is a smarter
    //       way to write this.
    while offset_from < dims {
        let mut acc1 = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut acc2 = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut acc3 = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut acc4 = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut acc5 = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut acc6 = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut acc7 = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut acc8 = _mm512_set1_ps(f32::NEG_INFINITY);

        // Vertical max of the 128 elements.
        for m in 0..matrix.len() {
            let arr = *matrix.get_unchecked(m);
            debug_assert_eq!(arr.len(), dims);

            let arr = arr.as_ptr();
            max_by_x128_horizontal(
                arr.add(offset_from),
                &mut acc1,
                &mut acc2,
                &mut acc3,
                &mut acc4,
                &mut acc5,
                &mut acc6,
                &mut acc7,
                &mut acc8,
            );
        }

        let merged = [acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8];

        let result = mem::transmute::<[__m512; 8], [f32; 128]>(merged);
        ptr::copy_nonoverlapping(
            result.as_ptr(),
            max_values_ptr.add(offset_from),
            result.len(),
        );

        offset_from += 128;
    }

    max_values
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn max_by_x128_horizontal(
    arr: *const f32,
    acc1: &mut __m512,
    acc2: &mut __m512,
    acc3: &mut __m512,
    acc4: &mut __m512,
    acc5: &mut __m512,
    acc6: &mut __m512,
    acc7: &mut __m512,
    acc8: &mut __m512,
) {
    let [x1, x2, x3, x4] = offsets_avx512::<CHUNK_0>(arr);
    let [x5, x6, x7, x8] = offsets_avx512::<CHUNK_1>(arr);

    let x1 = _mm512_loadu_ps(x1);
    let x2 = _mm512_loadu_ps(x2);
    let x3 = _mm512_loadu_ps(x3);
    let x4 = _mm512_loadu_ps(x4);
    let x5 = _mm512_loadu_ps(x5);
    let x6 = _mm512_loadu_ps(x6);
    let x7 = _mm512_loadu_ps(x7);
    let x8 = _mm512_loadu_ps(x8);

    *acc1 = _mm512_max_ps(*acc1, x1);
    *acc2 = _mm512_max_ps(*acc2, x2);
    *acc3 = _mm512_max_ps(*acc3, x3);
    *acc4 = _mm512_max_ps(*acc4, x4);
    *acc5 = _mm512_max_ps(*acc5, x5);
    *acc6 = _mm512_max_ps(*acc6, x6);
    *acc7 = _mm512_max_ps(*acc7, x7);
    *acc8 = _mm512_max_ps(*acc8, x8);
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn rollup_max_acc(
    mut acc1: __m512,
    acc2: __m512,
    mut acc3: __m512,
    acc4: __m512,
    mut acc5: __m512,
    acc6: __m512,
    mut acc7: __m512,
    acc8: __m512,
) -> f32 {
    acc1 = _mm512_max_ps(acc1, acc2);
    acc3 = _mm512_max_ps(acc3, acc4);
    acc5 = _mm512_max_ps(acc5, acc6);
    acc7 = _mm512_max_ps(acc7, acc8);

    acc1 = _mm512_max_ps(acc1, acc3);
    acc5 = _mm512_max_ps(acc5, acc7);

    acc1 = _mm512_max_ps(acc1, acc5);

    _mm512_reduce_max_ps(acc1)
}

#[cfg(all(test, target_feature = "avx512f"))]
mod tests {
    use super::*;
    use crate::test_utils::get_sample_vectors;

    #[test]
    fn test_xconst_nofma_max_horizontal() {
        let (x, _) = get_sample_vectors(512);
        let max = unsafe { f32_xconst_avx512_nofma_max_horizontal::<512>(&x) };
        assert_eq!(max, x.iter().fold(f32::NEG_INFINITY, |acc, v| acc.max(*v)));
    }

    #[test]
    fn test_xconst_nofma_max_vertical() {
        let mut matrix = Vec::new();
        for _ in 0..25 {
            let (x, _) = get_sample_vectors(512);
            matrix.push(x);
        }

        let matrix_view = matrix.iter().map(|v| v.as_ref()).collect::<Vec<&[f32]>>();

        let mut expected_vertical_max = vec![f32::NEG_INFINITY; 512];
        for i in 0..512 {
            let mut max = f32::NEG_INFINITY;
            for arr in matrix.iter() {
                max = max.max(arr[i]);
            }
            expected_vertical_max[i] = max;
        }

        let max = unsafe { f32_xconst_avx512_nofma_max_vertical::<512>(&matrix_view) };
        assert_eq!(max, expected_vertical_max);
    }

    #[test]
    fn test_xany_nofma_max_horizontal() {
        let (x, _) = get_sample_vectors(537);
        let max = unsafe { f32_xany_avx512_nofma_max_horizontal(&x) };
        assert_eq!(max, x.iter().fold(f32::NEG_INFINITY, |acc, v| acc.max(*v)));
    }

    #[test]
    fn test_xany_nofma_max_vertical() {
        let mut matrix = Vec::new();
        for _ in 0..25 {
            let (x, _) = get_sample_vectors(537);
            matrix.push(x);
        }

        let matrix_view = matrix.iter().map(|v| v.as_ref()).collect::<Vec<&[f32]>>();

        let mut expected_vertical_max = vec![f32::NEG_INFINITY; 537];
        for i in 0..537 {
            let mut max = f32::NEG_INFINITY;
            for arr in matrix.iter() {
                max = max.max(arr[i]);
            }
            expected_vertical_max[i] = max;
        }

        let max = unsafe { f32_xany_avx512_nofma_max_vertical(&matrix_view) };
        assert_eq!(max, expected_vertical_max);
    }
}