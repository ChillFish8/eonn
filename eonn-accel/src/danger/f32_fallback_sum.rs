use crate::math::*;

#[inline]
/// Sums all elements of the vector.
///
/// ```py
/// D: int
/// total: f32
/// x: [f32; D]
///
/// for i in 0..D:
///     total = total + x[i]
/// ```
///
/// # Safety
///
/// This method in theory is safe, but like the rest of the dangerous API, makes
/// no guarantee that it will always remain safe with no strings attached.
pub unsafe fn f32_xany_fallback_nofma_sum(x: &[f32]) -> f32 {
    sum::<StdMath>(x)
}

#[cfg(feature = "nightly")]
#[inline]
/// Sums all elements of the vector.
///
/// ```py
/// D: int
/// total: f32
/// x: [f32; D]
///
/// for i in 0..D:
///     total = total + x[i]
/// ```
///
/// # Safety
///
/// All values within the array must be finite and not `NaN` otherwise this
/// function can be UB.
pub unsafe fn f32_xany_fallback_fma_sum(x: &[f32]) -> f32 {
    sum::<FastMath>(x)
}

#[inline(always)]
unsafe fn sum<M: Math>(arr: &[f32]) -> f32 {
    let mut acc1 = 0.0;
    let mut acc2 = 0.0;
    let mut acc3 = 0.0;
    let mut acc4 = 0.0;
    let mut acc5 = 0.0;
    let mut acc6 = 0.0;
    let mut acc7 = 0.0;
    let mut acc8 = 0.0;

    let mut offset_from = arr.len() % 8;

    if offset_from != 0 {
        for i in 0..offset_from {
            let x = *arr.get_unchecked(i);
            acc1 = M::add(acc1, x);
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

        acc1 = M::add(acc1, x1);
        acc2 = M::add(acc2, x2);
        acc3 = M::add(acc3, x3);
        acc4 = M::add(acc4, x4);
        acc5 = M::add(acc5, x5);
        acc6 = M::add(acc6, x6);
        acc7 = M::add(acc7, x7);
        acc8 = M::add(acc8, x8);

        offset_from += 8;
    }

    acc1 = M::add(acc1, acc2);
    acc3 = M::add(acc3, acc4);
    acc5 = M::add(acc5, acc6);
    acc7 = M::add(acc7, acc8);

    acc1 = M::add(acc1, acc3);
    acc5 = M::add(acc5, acc7);

    M::add(acc1, acc5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_is_close, get_sample_vectors};

    #[test]
    fn test_xany_nofma_sum() {
        let (x, _) = get_sample_vectors(131);
        let sum = unsafe { f32_xany_fallback_nofma_sum(&x) };
        assert_is_close(sum, x.iter().sum::<f32>());
    }

    #[test]
    fn test_xany_fma_sum() {
        let (x, _) = get_sample_vectors(131);
        let sum = unsafe { f32_xany_fallback_fma_sum(&x) };
        assert_is_close(sum, x.iter().sum::<f32>());
    }
}
