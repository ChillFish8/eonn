use crate::math::*;

#[inline]
/// Divides each element in the provided mutable `[f32; DIMS]` vector by `value`.
///
/// # Safety
///
/// Vectors **MUST** be `DIMS` elements in length and divisible by 128,
/// otherwise this function becomes immediately UB due to out of bounds
/// access.
pub unsafe fn f32_xany_fallback_nofma_div_value(arr: &mut [f32], divider: f32) {
    f32_xany_fallback_nofma_mul_value(arr, 1.0 / divider)
}

#[inline]
/// Multiplies each element in the provided mutable `[f32; DIMS]` vector by `value`.
///
/// # Safety
///
/// Vectors **MUST** be `DIMS` elements in length and divisible by 128,
/// otherwise this function becomes immediately UB due to out of bounds
/// access.
pub unsafe fn f32_xany_fallback_nofma_mul_value(arr: &mut [f32], multiplier: f32) {
    f32_xany_fallback_mul_impl::<AutoMath>(arr, multiplier)
}

#[inline]
/// Multiplies each element in the provided mutable `[f32; DIMS]` vector by `value`.
///
/// # Safety
///
/// Vectors **MUST** be `DIMS` elements in length and divisible by 128,
/// otherwise this function becomes immediately UB due to out of bounds
/// access.
pub unsafe fn f32_xany_fallback_nofma_add_value(arr: &mut [f32], value: f32) {
    f32_xany_fallback_add_impl::<AutoMath>(arr, value)
}

#[inline]
/// Multiplies each element in the provided mutable `[f32; DIMS]` vector by `value`.
///
/// # Safety
///
/// Vectors **MUST** be `DIMS` elements in length and divisible by 128,
/// otherwise this function becomes immediately UB due to out of bounds
/// access.
pub unsafe fn f32_xany_fallback_nofma_sub_value(arr: &mut [f32], value: f32) {
    f32_xany_fallback_sub_impl::<AutoMath>(arr, value)
}

#[inline(always)]
unsafe fn f32_xany_fallback_mul_impl<M: Math>(arr: &mut [f32], multiplier: f32) {
    let mut offset_from = arr.len() % 8;

    if offset_from != 0 {
        for i in 0..offset_from {
            let x = arr.get_unchecked_mut(i);
            *x = M::mul(*x, multiplier);
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

        *arr.get_unchecked_mut(offset_from) = M::mul(x1, multiplier);
        *arr.get_unchecked_mut(offset_from + 1) = M::mul(x2, multiplier);
        *arr.get_unchecked_mut(offset_from + 2) = M::mul(x3, multiplier);
        *arr.get_unchecked_mut(offset_from + 3) = M::mul(x4, multiplier);
        *arr.get_unchecked_mut(offset_from + 4) = M::mul(x5, multiplier);
        *arr.get_unchecked_mut(offset_from + 5) = M::mul(x6, multiplier);
        *arr.get_unchecked_mut(offset_from + 6) = M::mul(x7, multiplier);
        *arr.get_unchecked_mut(offset_from + 7) = M::mul(x8, multiplier);

        offset_from += 8;
    }
}

#[inline(always)]
unsafe fn f32_xany_fallback_add_impl<M: Math>(arr: &mut [f32], value: f32) {
    let mut offset_from = arr.len() % 8;

    if offset_from != 0 {
        for i in 0..offset_from {
            let x = arr.get_unchecked_mut(i);
            *x = M::add(*x, value);
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

        *arr.get_unchecked_mut(offset_from) = M::add(x1, value);
        *arr.get_unchecked_mut(offset_from + 1) = M::add(x2, value);
        *arr.get_unchecked_mut(offset_from + 2) = M::add(x3, value);
        *arr.get_unchecked_mut(offset_from + 3) = M::add(x4, value);
        *arr.get_unchecked_mut(offset_from + 4) = M::add(x5, value);
        *arr.get_unchecked_mut(offset_from + 5) = M::add(x6, value);
        *arr.get_unchecked_mut(offset_from + 6) = M::add(x7, value);
        *arr.get_unchecked_mut(offset_from + 7) = M::add(x8, value);

        offset_from += 8;
    }
}

#[inline(always)]
unsafe fn f32_xany_fallback_sub_impl<M: Math>(arr: &mut [f32], value: f32) {
    let mut offset_from = arr.len() % 8;

    if offset_from != 0 {
        for i in 0..offset_from {
            let x = arr.get_unchecked_mut(i);
            *x = M::sub(*x, value);
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

        *arr.get_unchecked_mut(offset_from) = M::sub(x1, value);
        *arr.get_unchecked_mut(offset_from + 1) = M::sub(x2, value);
        *arr.get_unchecked_mut(offset_from + 2) = M::sub(x3, value);
        *arr.get_unchecked_mut(offset_from + 3) = M::sub(x4, value);
        *arr.get_unchecked_mut(offset_from + 4) = M::sub(x5, value);
        *arr.get_unchecked_mut(offset_from + 5) = M::sub(x6, value);
        *arr.get_unchecked_mut(offset_from + 6) = M::sub(x7, value);
        *arr.get_unchecked_mut(offset_from + 7) = M::sub(x8, value);

        offset_from += 8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_is_close_vector, get_sample_vectors};

    #[test]
    fn test_f32_xany_nofma_div() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(557);
        let expected = x.iter().copied().map(|v| v / value).collect::<Vec<_>>();
        unsafe { f32_xany_fallback_nofma_div_value(&mut x, value) };
        assert_is_close_vector(&x, &expected);
    }

    #[test]
    fn test_f32_xany_nofma_mul() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(557);
        let expected = x.iter().copied().map(|v| v * value).collect::<Vec<_>>();
        unsafe { f32_xany_fallback_nofma_mul_value(&mut x, value) };
        assert_is_close_vector(&x, &expected);
    }

    #[test]
    fn test_f32_xany_nofma_add() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(557);
        let expected = x.iter().copied().map(|v| v + value).collect::<Vec<_>>();
        unsafe { f32_xany_fallback_nofma_add_value(&mut x, value) };
        assert_is_close_vector(&x, &expected);
    }

    #[test]
    fn test_f32_xany_nofma_sub() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(557);
        let expected = x.iter().copied().map(|v| v - value).collect::<Vec<_>>();
        unsafe { f32_xany_fallback_nofma_sub_value(&mut x, value) };
        assert_is_close_vector(&x, &expected);
    }
}
