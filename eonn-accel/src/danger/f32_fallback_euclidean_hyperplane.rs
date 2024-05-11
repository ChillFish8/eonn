use crate::danger::rollup_scalar_x8;
use crate::math::*;

#[inline]
/// Computes the Euclidean hyperplane of two `f32` vectors
/// and the offset from origin.
///
/// These are fallback routines, they are designed to be optimized
/// by the compiler only, in areas where manually optimized routines
/// are unable to run due to lack of CPU features.
///
/// # Safety
///
/// Vectors **MUST** be equal length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
pub unsafe fn f32_xany_fallback_nofma_euclidean_hyperplane(
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, f32) {
    let mut hyperplane = vec![0.0; x.len()];
    let offset = fallback_euclidean_hyperplane::<AutoMath>(x, y, &mut hyperplane);
    (hyperplane, offset)
}

pub(crate) unsafe fn fallback_euclidean_hyperplane<M: Math>(
    x: &[f32],
    y: &[f32],
    hyperplane: &mut [f32],
) -> f32 {
    debug_assert_eq!(
        y.len(),
        x.len(),
        "Improper implementation detected, vectors must match in size"
    );
    debug_assert!(
        hyperplane.len() >= x.len(),
        "Input hyperplane must match length of input vectors"
    );

    let mut offset_acc1 = 0.0;
    let mut offset_acc2 = 0.0;
    let mut offset_acc3 = 0.0;
    let mut offset_acc4 = 0.0;
    let mut offset_acc5 = 0.0;
    let mut offset_acc6 = 0.0;
    let mut offset_acc7 = 0.0;
    let mut offset_acc8 = 0.0;

    let mut offset_from = x.len() % 8;
    if offset_from != 0 {
        for i in 0..offset_from {
            let x = *x.get_unchecked(i);
            let y = *y.get_unchecked(i);

            let diff = M::sub(x, y);
            let mean = M::mul(M::add(x, y), 0.5);

            offset_acc1 = M::add(offset_acc1, M::mul(diff, mean));
            *hyperplane.get_unchecked_mut(i) = diff;
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

        let diff1 = M::sub(x1, y1);
        let diff2 = M::sub(x2, y2);
        let diff3 = M::sub(x3, y3);
        let diff4 = M::sub(x4, y4);
        let diff5 = M::sub(x5, y5);
        let diff6 = M::sub(x6, y6);
        let diff7 = M::sub(x7, y7);
        let diff8 = M::sub(x8, y8);

        offset_acc1 = M::add(offset_acc1, M::mul(diff1, M::mul(M::add(x1, y1), 0.5)));
        offset_acc2 = M::add(offset_acc2, M::mul(diff2, M::mul(M::add(x2, y2), 0.5)));
        offset_acc3 = M::add(offset_acc3, M::mul(diff3, M::mul(M::add(x3, y3), 0.5)));
        offset_acc4 = M::add(offset_acc4, M::mul(diff4, M::mul(M::add(x4, y4), 0.5)));
        offset_acc5 = M::add(offset_acc5, M::mul(diff5, M::mul(M::add(x5, y5), 0.5)));
        offset_acc6 = M::add(offset_acc6, M::mul(diff6, M::mul(M::add(x6, y6), 0.5)));
        offset_acc7 = M::add(offset_acc7, M::mul(diff7, M::mul(M::add(x7, y7), 0.5)));
        offset_acc8 = M::add(offset_acc8, M::mul(diff8, M::mul(M::add(x8, y8), 0.5)));

        *hyperplane.get_unchecked_mut(offset_from) = diff1;
        *hyperplane.get_unchecked_mut(offset_from + 1) = diff2;
        *hyperplane.get_unchecked_mut(offset_from + 2) = diff3;
        *hyperplane.get_unchecked_mut(offset_from + 3) = diff4;
        *hyperplane.get_unchecked_mut(offset_from + 4) = diff5;
        *hyperplane.get_unchecked_mut(offset_from + 5) = diff6;
        *hyperplane.get_unchecked_mut(offset_from + 6) = diff7;
        *hyperplane.get_unchecked_mut(offset_from + 7) = diff8;

        offset_from += 8;
    }

    let hyperplane_offset = -rollup_scalar_x8::<M>(
        offset_acc1,
        offset_acc2,
        offset_acc3,
        offset_acc4,
        offset_acc5,
        offset_acc6,
        offset_acc7,
        offset_acc8,
    );

    hyperplane_offset
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{
        assert_is_close,
        assert_is_close_vector,
        get_sample_vectors,
        simple_euclidean_hyperplane,
    };

    #[test]
    fn test_xany_nofma_euclidean_hyperplane() {
        let (x, y) = get_sample_vectors(514);
        let (hyperplane, offset) =
            unsafe { f32_xany_fallback_nofma_euclidean_hyperplane(&x, &y) };
        let (expected, expected_offset) = simple_euclidean_hyperplane(&x, &y);
        assert_is_close(offset, expected_offset);
        assert_is_close_vector(&hyperplane, &expected);
    }
}
