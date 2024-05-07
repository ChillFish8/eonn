#[inline]
/// Computes the horizontal minimum of the given vector that is `[f32; N]`.
///
/// # Safety
///
/// This method in theory is safe, but like the rest of the dangerous API, makes
/// no guarantee that it will always remain safe with no strings attached.
pub unsafe fn f32_xany_fallback_nofma_min_horizontal(arr: &[f32]) -> f32 {
    let mut acc1 = f32::INFINITY;
    let mut acc2 = f32::INFINITY;
    let mut acc3 = f32::INFINITY;
    let mut acc4 = f32::INFINITY;
    let mut acc5 = f32::INFINITY;
    let mut acc6 = f32::INFINITY;
    let mut acc7 = f32::INFINITY;
    let mut acc8 = f32::INFINITY;

    let mut offset_from = arr.len() % 8;

    if offset_from != 0 {
        for i in 0..offset_from {
            let x = *arr.get_unchecked(i);
            acc1 = acc1.min(x);
        }
    }

    while offset_from < arr.len() {
        let x1 = *arr.get_unchecked(offset_from);
        let x2 = *arr.get_unchecked(offset_from + 1);
        let x3 = *arr.get_unchecked(offset_from + 2);
        let x4 = *arr.get_unchecked(offset_from + 3);
        let x5 = *arr.get_unchecked(offset_from + 4);
        let x6 = *arr.get_unchecked(offset_from + 5);
        let x7 = *arr.get_unchecked(offset_from + 6);
        let x8 = *arr.get_unchecked(offset_from + 7);

        acc1 = acc1.min(x1);
        acc2 = acc2.min(x2);
        acc3 = acc3.min(x3);
        acc4 = acc4.min(x4);
        acc5 = acc5.min(x5);
        acc6 = acc6.min(x6);
        acc7 = acc7.min(x7);
        acc8 = acc8.min(x8);

        offset_from += 8;
    }

    acc1 = acc1.min(acc2);
    acc3 = acc3.min(acc4);
    acc5 = acc5.min(acc6);
    acc7 = acc7.min(acc8);

    acc1 = acc1.min(acc3);
    acc5 = acc5.min(acc7);

    acc1.min(acc5)
}

#[inline]
/// Computes the horizontal minimum of the given vector that is `[[f32; DIMS]; N]`.
///
/// # Safety
///
/// Each vector in the matrix must be the same size, this routine assumes the dimensions
/// of all vectors in the matrix are equal to the dimensions of the first vector in
/// the matrix.
pub unsafe fn f32_xany_fallback_nofma_min_vertical(matrix: &[&[f32]]) -> Vec<f32> {
    let dims = matrix[0].len();

    let mut min_values = vec![0.0; dims];
    let mut offset_from = dims % 8;

    if offset_from != 0 {
        for i in 0..offset_from {
            let mut acc = f32::INFINITY;
            for m in 0..matrix.len() {
                let arr = *matrix.get_unchecked(m);
                debug_assert_eq!(arr.len(), dims);

                let x = *arr.get_unchecked(i);
                acc = acc.min(x);
            }

            *min_values.get_unchecked_mut(i) = acc;
        }
    }

    // We work our way horizontally by taking steps of 8 and finding
    // the min of for each of the lanes vertically through the matrix.
    while offset_from < dims {
        let mut acc1 = f32::INFINITY;
        let mut acc2 = f32::INFINITY;
        let mut acc3 = f32::INFINITY;
        let mut acc4 = f32::INFINITY;
        let mut acc5 = f32::INFINITY;
        let mut acc6 = f32::INFINITY;
        let mut acc7 = f32::INFINITY;
        let mut acc8 = f32::INFINITY;

        // Vertical min of the 8 elements.
        for m in 0..matrix.len() {
            let arr = *matrix.get_unchecked(m);
            debug_assert_eq!(arr.len(), dims);

            let x1 = *arr.get_unchecked(offset_from);
            let x2 = *arr.get_unchecked(offset_from + 1);
            let x3 = *arr.get_unchecked(offset_from + 2);
            let x4 = *arr.get_unchecked(offset_from + 3);
            let x5 = *arr.get_unchecked(offset_from + 4);
            let x6 = *arr.get_unchecked(offset_from + 5);
            let x7 = *arr.get_unchecked(offset_from + 6);
            let x8 = *arr.get_unchecked(offset_from + 7);

            acc1 = acc1.min(x1);
            acc2 = acc2.min(x2);
            acc3 = acc3.min(x3);
            acc4 = acc4.min(x4);
            acc5 = acc5.min(x5);
            acc6 = acc6.min(x6);
            acc7 = acc7.min(x7);
            acc8 = acc8.min(x8);
        }

        *min_values.get_unchecked_mut(offset_from) = acc1;
        *min_values.get_unchecked_mut(offset_from + 1) = acc2;
        *min_values.get_unchecked_mut(offset_from + 2) = acc3;
        *min_values.get_unchecked_mut(offset_from + 3) = acc4;
        *min_values.get_unchecked_mut(offset_from + 4) = acc5;
        *min_values.get_unchecked_mut(offset_from + 5) = acc6;
        *min_values.get_unchecked_mut(offset_from + 6) = acc7;
        *min_values.get_unchecked_mut(offset_from + 7) = acc8;

        offset_from += 8;
    }

    min_values
}

#[cfg(all(test, target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::get_sample_vectors;

    #[test]
    fn test_xany_nofma_min_horizontal() {
        let (x, _) = get_sample_vectors(131);
        let min = unsafe { f32_xany_fallback_nofma_min_horizontal(&x) };
        assert_eq!(min, x.iter().fold(f32::INFINITY, |acc, v| acc.min(*v)));
    }

    #[test]
    fn test_xany_nofma_min_vertical() {
        let mut matrix = Vec::new();
        for _ in 0..25 {
            let (x, _) = get_sample_vectors(537);
            matrix.push(x);
        }

        let matrix_view = matrix.iter().map(|v| v.as_ref()).collect::<Vec<&[f32]>>();

        let mut expected_vertical_min = vec![f32::INFINITY; 537];
        for i in 0..537 {
            let mut min = f32::INFINITY;
            for arr in matrix.iter() {
                min = min.min(arr[i]);
            }
            expected_vertical_min[i] = min;
        }

        let min = unsafe { f32_xany_fallback_nofma_min_vertical(&matrix_view) };
        assert_eq!(min, expected_vertical_min);
    }
}
