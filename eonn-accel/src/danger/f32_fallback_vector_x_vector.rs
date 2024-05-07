use crate::math::{FastMath, Math, StdMath};

#[inline]
/// Divides each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f32; D]
/// y: [f32; D]
///
/// for i in 0..D:
///     x[i] = x[i] / y[i]
/// ```
///
/// # Safety
///
/// Lengths of `x` and `y` **MUST** be equal.
pub unsafe fn f32_xany_fallback_nofma_div_vertical(x: &mut [f32], y: &[f32]) {
    f32_op_vertical(StdMath::div, x, y)
}

#[cfg(feature = "nightly")]
#[inline]
/// Divides each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f32; D]
/// y: [f32; D]
///
/// for i in 0..D:
///     x[i] = x[i] / y[i]
/// ```
///
/// # Safety
///
/// Lengths of `x` and `y` **MUST** be equal.
pub unsafe fn f32_xany_fallback_fma_div_vertical(x: &mut [f32], y: &[f32]) {
    f32_op_vertical(FastMath::div, x, y)
}

#[inline]
/// Multiplies each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f32; D]
/// y: [f32; D]
///
/// for i in 0..D:
///     x[i] = x[i] * y[i]
/// ```
///
/// # Safety
///
/// Lengths of `x` and `y` **MUST** be equal.
pub unsafe fn f32_xany_fallback_nofma_mul_vertical(x: &mut [f32], y: &[f32]) {
    f32_op_vertical(StdMath::mul, x, y)
}

#[cfg(feature = "nightly")]
#[inline]
/// Multiplies each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f32; D]
/// y: [f32; D]
///
/// for i in 0..D:
///     x[i] = x[i] * y[i]
/// ```
///
/// # Safety
///
/// Lengths of `x` and `y` **MUST** be equal.
pub unsafe fn f32_xany_fallback_fma_mul_vertical(x: &mut [f32], y: &[f32]) {
    f32_op_vertical(FastMath::mul, x, y)
}

#[inline]
/// Adds each the input mutable vector `x` with the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f32; D]
/// y: [f32; D]
///
/// for i in 0..D:
///     x[i] = x[i] + y[i]
/// ```
///
/// # Safety
///
/// Lengths of `x` and `y` **MUST** be equal.
pub unsafe fn f32_xany_fallback_nofma_add_vertical(x: &mut [f32], y: &[f32]) {
    f32_op_vertical(StdMath::add, x, y)
}

#[cfg(feature = "nightly")]
#[inline]
/// Adds each the input mutable vector `x` with the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f32; D]
/// y: [f32; D]
///
/// for i in 0..D:
///     x[i] = x[i] + y[i]
/// ```
///
/// # Safety
///
/// Lengths of `x` and `y` **MUST** be equal.
pub unsafe fn f32_xany_fallback_fma_add_vertical(x: &mut [f32], y: &[f32]) {
    f32_op_vertical(FastMath::add, x, y)
}

#[inline]
/// Subtracts each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f32; D]
/// y: [f32; D]
///
/// for i in 0..D:
///     x[i] = x[i] + y[i]
/// ```
///
/// # Safety
///
/// Lengths of `x` and `y` **MUST** be equal.
pub unsafe fn f32_xany_fallback_nofma_sub_vertical(x: &mut [f32], y: &[f32]) {
    f32_op_vertical(StdMath::sub, x, y)
}

#[cfg(feature = "nightly")]
#[inline]
/// Subtracts each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f32; D]
/// y: [f32; D]
///
/// for i in 0..D:
///     x[i] = x[i] + y[i]
/// ```
///
/// # Safety
///
/// Lengths of `x` and `y` **MUST** be equal.
pub unsafe fn f32_xany_fallback_fma_sub_vertical(x: &mut [f32], y: &[f32]) {
    f32_op_vertical(FastMath::sub, x, y)
}

#[inline(always)]
unsafe fn f32_op_vertical<O>(op: O, x: &mut [f32], y: &[f32])
where
    O: Fn(f32, f32) -> f32,
{
    debug_assert_eq!(x.len(), y.len());

    let mut offset_from = x.len() % 8;

    if offset_from != 0 {
        for i in 0..offset_from {
            let x = x.get_unchecked_mut(i);
            let y = *y.get_unchecked(i);

            *x = op(*x, y);
        }
    }

    while offset_from < x.len() {
        let x1 = *x.get_unchecked(offset_from);
        let x2 = *x.get_unchecked(offset_from + 1);
        let x3 = *x.get_unchecked(offset_from + 2);
        let x4 = *x.get_unchecked(offset_from + 3);
        let x5 = *x.get_unchecked(offset_from + 4);
        let x6 = *x.get_unchecked(offset_from + 5);
        let x7 = *x.get_unchecked(offset_from + 6);
        let x8 = *x.get_unchecked(offset_from + 7);

        let y1 = *y.get_unchecked(offset_from);
        let y2 = *y.get_unchecked(offset_from + 1);
        let y3 = *y.get_unchecked(offset_from + 2);
        let y4 = *y.get_unchecked(offset_from + 3);
        let y5 = *y.get_unchecked(offset_from + 4);
        let y6 = *y.get_unchecked(offset_from + 5);
        let y7 = *y.get_unchecked(offset_from + 6);
        let y8 = *y.get_unchecked(offset_from + 7);

        *x.get_unchecked_mut(offset_from) = op(x1, y1);
        *x.get_unchecked_mut(offset_from + 1) = op(x2, y2);
        *x.get_unchecked_mut(offset_from + 2) = op(x3, y3);
        *x.get_unchecked_mut(offset_from + 3) = op(x4, y4);
        *x.get_unchecked_mut(offset_from + 4) = op(x5, y5);
        *x.get_unchecked_mut(offset_from + 5) = op(x6, y6);
        *x.get_unchecked_mut(offset_from + 6) = op(x7, y7);
        *x.get_unchecked_mut(offset_from + 7) = op(x8, y8);

        offset_from += 8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_is_close_vector, get_sample_vectors};

    #[test]
    fn test_xany_nofma_div_vertical() {
        let (mut x, y) = get_sample_vectors(537);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x / y)
            .collect::<Vec<f32>>();

        unsafe { f32_xany_fallback_nofma_div_vertical(&mut x, &y) };

        assert_is_close_vector(&x, &expected_res);
    }

    #[test]
    fn test_xany_nofma_mul_vertical() {
        let (mut x, y) = get_sample_vectors(537);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x * y)
            .collect::<Vec<f32>>();

        unsafe { f32_xany_fallback_nofma_mul_vertical(&mut x, &y) };

        assert_is_close_vector(&x, &expected_res);
    }

    #[test]
    fn test_xany_nofma_add_vertical() {
        let (mut x, y) = get_sample_vectors(537);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x + y)
            .collect::<Vec<f32>>();

        unsafe { f32_xany_fallback_nofma_add_vertical(&mut x, &y) };

        assert_is_close_vector(&x, &expected_res);
    }

    #[test]
    fn test_xany_nofma_sub_vertical() {
        let (mut x, y) = get_sample_vectors(537);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x - y)
            .collect::<Vec<f32>>();

        unsafe { f32_xany_fallback_nofma_sub_vertical(&mut x, &y) };

        assert_is_close_vector(&x, &expected_res);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_xany_fma_div_vertical() {
        let (mut x, y) = get_sample_vectors(537);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x / y)
            .collect::<Vec<f32>>();

        unsafe { f32_xany_fallback_fma_div_vertical(&mut x, &y) };

        assert_is_close_vector(&x, &expected_res);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_xany_fma_mul_vertical() {
        let (mut x, y) = get_sample_vectors(537);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x * y)
            .collect::<Vec<f32>>();

        unsafe { f32_xany_fallback_fma_mul_vertical(&mut x, &y) };

        assert_is_close_vector(&x, &expected_res);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_xany_fma_add_vertical() {
        let (mut x, y) = get_sample_vectors(537);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x + y)
            .collect::<Vec<f32>>();

        unsafe { f32_xany_fallback_fma_add_vertical(&mut x, &y) };

        assert_is_close_vector(&x, &expected_res);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_xany_fma_sub_vertical() {
        let (mut x, y) = get_sample_vectors(537);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x - y)
            .collect::<Vec<f32>>();

        unsafe { f32_xany_fallback_fma_sub_vertical(&mut x, &y) };

        assert_is_close_vector(&x, &expected_res);
    }
}
