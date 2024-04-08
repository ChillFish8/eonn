use crate::distance_ops::arch::*;
use crate::distance_ops::{
    DistanceOps,
    Fallback,
    Vector,
    VectorView,
    X1024,
    X512,
    X768,
};
use crate::math::*;

impl DistanceOps for Vector<Fallback, X1024, f32, NoFma> {
    #[inline]
    unsafe fn dot(&self, other: &Self) -> f32 {
        fallback_dot_product::<StdMath, 1024>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn cosine(&self, other: &Self) -> f32 {
        fallback_cosine::<StdMath, 1024>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn euclidean(&self, other: &Self) -> f32 {
        fallback_euclidean::<StdMath, 1024>(&self.0, &other.0)
    }
}

impl DistanceOps for Vector<Fallback, X768, f32, NoFma> {
    #[inline]
    unsafe fn dot(&self, other: &Self) -> f32 {
        fallback_dot_product::<StdMath, 768>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn cosine(&self, other: &Self) -> f32 {
        fallback_cosine::<StdMath, 768>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn euclidean(&self, other: &Self) -> f32 {
        fallback_euclidean::<StdMath, 768>(&self.0, &other.0)
    }
}

impl DistanceOps for Vector<Fallback, X512, f32, NoFma> {
    #[inline]
    unsafe fn dot(&self, other: &Self) -> f32 {
        fallback_dot_product::<StdMath, 512>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn cosine(&self, other: &Self) -> f32 {
        fallback_cosine::<StdMath, 512>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn euclidean(&self, other: &Self) -> f32 {
        fallback_euclidean::<StdMath, 512>(&self.0, &other.0)
    }
}

#[cfg(any(
    feature = "bypass-arch-flags",
    all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "fma",
    )
))]
impl DistanceOps for Vector<Fallback, X1024, f32, Fma> {
    #[inline]
    unsafe fn dot(&self, other: &Self) -> f32 {
        fallback_dot_product::<FastMath, 1024>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn cosine(&self, other: &Self) -> f32 {
        fallback_cosine::<FastMath, 1024>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn euclidean(&self, other: &Self) -> f32 {
        fallback_euclidean::<FastMath, 1024>(&self.0, &other.0)
    }
}

#[cfg(any(
    feature = "bypass-arch-flags",
    all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "fma",
    )
))]
impl DistanceOps for Vector<Fallback, X768, f32, Fma> {
    #[inline]
    unsafe fn dot(&self, other: &Self) -> f32 {
        fallback_dot_product::<FastMath, 768>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn cosine(&self, other: &Self) -> f32 {
        fallback_cosine::<FastMath, 768>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn euclidean(&self, other: &Self) -> f32 {
        fallback_euclidean::<FastMath, 768>(&self.0, &other.0)
    }
}

#[cfg(any(
    feature = "bypass-arch-flags",
    all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "fma",
    )
))]
impl DistanceOps for Vector<Fallback, X512, f32, Fma> {
    #[inline]
    unsafe fn dot(&self, other: &Self) -> f32 {
        fallback_dot_product::<FastMath, 512>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn cosine(&self, other: &Self) -> f32 {
        fallback_cosine::<FastMath, 512>(&self.0, &other.0)
    }

    #[inline]
    unsafe fn euclidean(&self, other: &Self) -> f32 {
        fallback_euclidean::<FastMath, 512>(&self.0, &other.0)
    }
}

impl<'a> DistanceOps for VectorView<'a, Fallback, X1024, f32, NoFma> {
    #[inline]
    unsafe fn dot(&self, other: &Self) -> f32 {
        fallback_dot_product::<StdMath, 1024>(self.0, other.0)
    }

    #[inline]
    unsafe fn cosine(&self, other: &Self) -> f32 {
        fallback_cosine::<StdMath, 1024>(self.0, other.0)
    }

    #[inline]
    unsafe fn euclidean(&self, other: &Self) -> f32 {
        fallback_euclidean::<StdMath, 1024>(self.0, other.0)
    }
}

impl<'a> DistanceOps for VectorView<'a, Fallback, X768, f32, NoFma> {
    #[inline]
    unsafe fn dot(&self, other: &Self) -> f32 {
        fallback_dot_product::<StdMath, 768>(self.0, other.0)
    }

    #[inline]
    unsafe fn cosine(&self, other: &Self) -> f32 {
        fallback_cosine::<StdMath, 768>(self.0, other.0)
    }

    #[inline]
    unsafe fn euclidean(&self, other: &Self) -> f32 {
        fallback_euclidean::<StdMath, 768>(self.0, other.0)
    }
}

impl<'a> DistanceOps for VectorView<'a, Fallback, X512, f32, NoFma> {
    #[inline]
    unsafe fn dot(&self, other: &Self) -> f32 {
        fallback_dot_product::<StdMath, 512>(self.0, other.0)
    }

    #[inline]
    unsafe fn cosine(&self, other: &Self) -> f32 {
        fallback_cosine::<StdMath, 512>(self.0, other.0)
    }

    #[inline]
    unsafe fn euclidean(&self, other: &Self) -> f32 {
        fallback_euclidean::<StdMath, 512>(self.0, other.0)
    }
}

#[cfg(any(
    feature = "bypass-arch-flags",
    all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "fma",
    )
))]
impl<'a> DistanceOps for VectorView<'a, Fallback, X1024, f32, Fma> {
    #[inline]
    unsafe fn dot(&self, other: &Self) -> f32 {
        fallback_dot_product::<FastMath, 1024>(self.0, other.0)
    }

    #[inline]
    unsafe fn cosine(&self, other: &Self) -> f32 {
        fallback_cosine::<FastMath, 1024>(self.0, other.0)
    }

    #[inline]
    unsafe fn euclidean(&self, other: &Self) -> f32 {
        fallback_euclidean::<FastMath, 1024>(self.0, other.0)
    }
}

#[cfg(any(
    feature = "bypass-arch-flags",
    all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "fma",
    )
))]
impl<'a> DistanceOps for VectorView<'a, Fallback, X768, f32, Fma> {
    #[inline]
    unsafe fn dot(&self, other: &Self) -> f32 {
        fallback_dot_product::<FastMath, 768>(self.0, other.0)
    }

    #[inline]
    unsafe fn cosine(&self, other: &Self) -> f32 {
        fallback_cosine::<FastMath, 768>(self.0, other.0)
    }

    #[inline]
    unsafe fn euclidean(&self, other: &Self) -> f32 {
        fallback_euclidean::<FastMath, 768>(self.0, other.0)
    }
}

#[cfg(any(
    feature = "bypass-arch-flags",
    all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "fma",
    )
))]
impl<'a> DistanceOps for VectorView<'a, Fallback, X512, f32, Fma> {
    #[inline]
    unsafe fn dot(&self, other: &Self) -> f32 {
        fallback_dot_product::<FastMath, 512>(self.0, other.0)
    }

    #[inline]
    unsafe fn cosine(&self, other: &Self) -> f32 {
        fallback_cosine::<FastMath, 512>(self.0, other.0)
    }

    #[inline]
    unsafe fn euclidean(&self, other: &Self) -> f32 {
        fallback_euclidean::<FastMath, 512>(self.0, other.0)
    }
}

#[inline]
pub(super) unsafe fn fallback_dot_product<M: Math, const DIMS: usize>(
    a: &[f32],
    b: &[f32],
) -> f32 {
    debug_assert_eq!(
        b.len(),
        DIMS,
        "Improper implementation detected, vectors must match constant"
    );
    debug_assert_eq!(
        a.len(),
        DIMS,
        "Improper implementation detected, vectors must match constant"
    );
    debug_assert_eq!(
        DIMS % 8,
        0,
        "DIMS must be able to fit entirely into chunks of 8 lanes."
    );

    let mut i = 0;

    // We do this manual unrolling to allow the compiler to vectorize
    // the loop and avoid some branching even if we're not doing it explicitly.
    // This made a significant difference in benchmarking ~8x
    let mut acc1 = 0.0;
    let mut acc2 = 0.0;
    let mut acc3 = 0.0;
    let mut acc4 = 0.0;
    let mut acc5 = 0.0;
    let mut acc6 = 0.0;
    let mut acc7 = 0.0;
    let mut acc8 = 0.0;

    while i < a.len() {
        let a1 = *a.get_unchecked(i);
        let a2 = *a.get_unchecked(i + 1);
        let a3 = *a.get_unchecked(i + 2);
        let a4 = *a.get_unchecked(i + 3);
        let a5 = *a.get_unchecked(i + 4);
        let a6 = *a.get_unchecked(i + 5);
        let a7 = *a.get_unchecked(i + 6);
        let a8 = *a.get_unchecked(i + 7);

        let b1 = *b.get_unchecked(i);
        let b2 = *b.get_unchecked(i + 1);
        let b3 = *b.get_unchecked(i + 2);
        let b4 = *b.get_unchecked(i + 3);
        let b5 = *b.get_unchecked(i + 4);
        let b6 = *b.get_unchecked(i + 5);
        let b7 = *b.get_unchecked(i + 6);
        let b8 = *b.get_unchecked(i + 7);

        acc1 = M::add(acc1, M::mul(a1, b1));
        acc2 = M::add(acc2, M::mul(a2, b2));
        acc3 = M::add(acc3, M::mul(a3, b3));
        acc4 = M::add(acc4, M::mul(a4, b4));
        acc5 = M::add(acc5, M::mul(a5, b5));
        acc6 = M::add(acc6, M::mul(a6, b6));
        acc7 = M::add(acc7, M::mul(a7, b7));
        acc8 = M::add(acc8, M::mul(a8, b8));

        i += 8;
    }

    rollup_x8::<M>(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[inline]
pub(super) unsafe fn fallback_cosine<M: Math, const DIMS: usize>(
    a: &[f32],
    b: &[f32],
) -> f32 {
    debug_assert_eq!(
        b.len(),
        DIMS,
        "Improper implementation detected, vectors must match constant"
    );
    debug_assert_eq!(
        a.len(),
        DIMS,
        "Improper implementation detected, vectors must match constant"
    );
    debug_assert_eq!(
        DIMS % 8,
        0,
        "DIMS must be able to fit entirely into chunks of 8 lanes."
    );

    let mut i = 0;

    // We do this manual unrolling to allow the compiler to vectorize
    // the loop and avoid some branching even if we're not doing it explicitly.
    // This made a significant difference in benchmarking ~4-8x
    let mut acc1 = 0.0;
    let mut acc2 = 0.0;
    let mut acc3 = 0.0;
    let mut acc4 = 0.0;
    let mut acc5 = 0.0;
    let mut acc6 = 0.0;
    let mut acc7 = 0.0;
    let mut acc8 = 0.0;

    let mut norm_a_acc1 = 0.0;
    let mut norm_a_acc2 = 0.0;
    let mut norm_a_acc3 = 0.0;
    let mut norm_a_acc4 = 0.0;
    let mut norm_a_acc5 = 0.0;
    let mut norm_a_acc6 = 0.0;
    let mut norm_a_acc7 = 0.0;
    let mut norm_a_acc8 = 0.0;

    let mut norm_b_acc1 = 0.0;
    let mut norm_b_acc2 = 0.0;
    let mut norm_b_acc3 = 0.0;
    let mut norm_b_acc4 = 0.0;
    let mut norm_b_acc5 = 0.0;
    let mut norm_b_acc6 = 0.0;
    let mut norm_b_acc7 = 0.0;
    let mut norm_b_acc8 = 0.0;

    while i < a.len() {
        let a1 = *a.get_unchecked(i);
        let a2 = *a.get_unchecked(i + 1);
        let a3 = *a.get_unchecked(i + 2);
        let a4 = *a.get_unchecked(i + 3);
        let a5 = *a.get_unchecked(i + 4);
        let a6 = *a.get_unchecked(i + 5);
        let a7 = *a.get_unchecked(i + 6);
        let a8 = *a.get_unchecked(i + 7);

        let b1 = *b.get_unchecked(i);
        let b2 = *b.get_unchecked(i + 1);
        let b3 = *b.get_unchecked(i + 2);
        let b4 = *b.get_unchecked(i + 3);
        let b5 = *b.get_unchecked(i + 4);
        let b6 = *b.get_unchecked(i + 5);
        let b7 = *b.get_unchecked(i + 6);
        let b8 = *b.get_unchecked(i + 7);

        acc1 = M::add(acc1, M::mul(a1, b1));
        acc2 = M::add(acc2, M::mul(a2, b2));
        acc3 = M::add(acc3, M::mul(a3, b3));
        acc4 = M::add(acc4, M::mul(a4, b4));
        acc5 = M::add(acc5, M::mul(a5, b5));
        acc6 = M::add(acc6, M::mul(a6, b6));
        acc7 = M::add(acc7, M::mul(a7, b7));
        acc8 = M::add(acc8, M::mul(a8, b8));

        norm_a_acc1 = M::add(norm_a_acc1, M::mul(a1, a1));
        norm_a_acc2 = M::add(norm_a_acc2, M::mul(a2, a2));
        norm_a_acc3 = M::add(norm_a_acc3, M::mul(a3, a3));
        norm_a_acc4 = M::add(norm_a_acc4, M::mul(a4, a4));
        norm_a_acc5 = M::add(norm_a_acc5, M::mul(a5, a5));
        norm_a_acc6 = M::add(norm_a_acc6, M::mul(a6, a6));
        norm_a_acc7 = M::add(norm_a_acc7, M::mul(a7, a7));
        norm_a_acc8 = M::add(norm_a_acc8, M::mul(a8, a8));

        norm_b_acc1 = M::add(norm_b_acc1, M::mul(b1, b1));
        norm_b_acc2 = M::add(norm_b_acc2, M::mul(b2, b2));
        norm_b_acc3 = M::add(norm_b_acc3, M::mul(b3, b3));
        norm_b_acc4 = M::add(norm_b_acc4, M::mul(b4, b4));
        norm_b_acc5 = M::add(norm_b_acc5, M::mul(b5, b5));
        norm_b_acc6 = M::add(norm_b_acc6, M::mul(b6, b6));
        norm_b_acc7 = M::add(norm_b_acc7, M::mul(b7, b7));
        norm_b_acc8 = M::add(norm_b_acc8, M::mul(b8, b8));

        i += 8;
    }

    let result = rollup_x8::<M>(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
    let norm_a = rollup_x8::<M>(
        norm_a_acc1,
        norm_a_acc2,
        norm_a_acc3,
        norm_a_acc4,
        norm_a_acc5,
        norm_a_acc6,
        norm_a_acc7,
        norm_a_acc8,
    );
    let norm_b = rollup_x8::<M>(
        norm_b_acc1,
        norm_b_acc2,
        norm_b_acc3,
        norm_b_acc4,
        norm_b_acc5,
        norm_b_acc6,
        norm_b_acc7,
        norm_b_acc8,
    );

    if norm_a == 0.0 && norm_b == 0.0 {
        0.0
    } else if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        M::sub(1.0, M::div(result, M::mul(norm_a, norm_b).sqrt()))
    }
}

#[inline]
pub(super) unsafe fn fallback_euclidean<M: Math, const DIMS: usize>(
    a: &[f32],
    b: &[f32],
) -> f32 {
    debug_assert_eq!(
        b.len(),
        DIMS,
        "Improper implementation detected, vectors must match constant"
    );
    debug_assert_eq!(
        a.len(),
        DIMS,
        "Improper implementation detected, vectors must match constant"
    );
    debug_assert_eq!(
        DIMS % 8,
        0,
        "DIMS must be able to fit entirely into chunks of 8 lanes."
    );

    let mut i = 0;

    // We do this manual unrolling to allow the compiler to vectorize
    // the loop and avoid some branching even if we're not doing it explicitly.
    // This made a significant difference in benchmarking ~4-8x
    let mut acc1 = 0.0;
    let mut acc2 = 0.0;
    let mut acc3 = 0.0;
    let mut acc4 = 0.0;
    let mut acc5 = 0.0;
    let mut acc6 = 0.0;
    let mut acc7 = 0.0;
    let mut acc8 = 0.0;

    while i < a.len() {
        let a1 = *a.get_unchecked(i);
        let a2 = *a.get_unchecked(i + 1);
        let a3 = *a.get_unchecked(i + 2);
        let a4 = *a.get_unchecked(i + 3);
        let a5 = *a.get_unchecked(i + 4);
        let a6 = *a.get_unchecked(i + 5);
        let a7 = *a.get_unchecked(i + 6);
        let a8 = *a.get_unchecked(i + 7);

        let b1 = *b.get_unchecked(i);
        let b2 = *b.get_unchecked(i + 1);
        let b3 = *b.get_unchecked(i + 2);
        let b4 = *b.get_unchecked(i + 3);
        let b5 = *b.get_unchecked(i + 4);
        let b6 = *b.get_unchecked(i + 5);
        let b7 = *b.get_unchecked(i + 6);
        let b8 = *b.get_unchecked(i + 7);

        let diff1 = M::sub(a1, b1);
        let diff2 = M::sub(a2, b2);
        let diff3 = M::sub(a3, b3);
        let diff4 = M::sub(a4, b4);
        let diff5 = M::sub(a5, b5);
        let diff6 = M::sub(a6, b6);
        let diff7 = M::sub(a7, b7);
        let diff8 = M::sub(a8, b8);

        acc1 = M::add(acc1, M::mul(diff1, diff1));
        acc2 = M::add(acc2, M::mul(diff2, diff2));
        acc3 = M::add(acc3, M::mul(diff3, diff3));
        acc4 = M::add(acc4, M::mul(diff4, diff4));
        acc5 = M::add(acc5, M::mul(diff5, diff5));
        acc6 = M::add(acc6, M::mul(diff6, diff6));
        acc7 = M::add(acc7, M::mul(diff7, diff7));
        acc8 = M::add(acc8, M::mul(diff8, diff8));

        i += 8;
    }

    rollup_x8::<M>(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn rollup_x8<M: Math>(
    mut acc1: f32,
    acc2: f32,
    mut acc3: f32,
    acc4: f32,
    mut acc5: f32,
    acc6: f32,
    mut acc7: f32,
    acc8: f32,
) -> f32 {
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
    use crate::math::StdMath;

    #[test]
    fn test_fallback_dot_product() {
        let v1 = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.5, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let v = unsafe { fallback_dot_product::<StdMath, 8>(&v1, &v2) };
        assert_eq!(v, 18.5);
    }

    #[test]
    fn test_fallback_cosine() {
        let v1 = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.8, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let v = unsafe { fallback_cosine::<StdMath, 8>(&v1, &v2) };
        assert_eq!(v, 0.021929622);
    }

    #[test]
    fn test_fallback_euclidean() {
        let v1 = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2 = vec![2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let v = unsafe { fallback_euclidean::<StdMath, 8>(&v1, &v2) };
        assert_eq!(v, 3.0);
    }
}
