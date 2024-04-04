use crate::math::*;
use crate::vector_ops::arch::*;
use crate::vector_ops::{DistanceOps, Fallback, Vector, VectorView, X1024, X512, X768};

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

    let mut result = 0.0;
    for i in 0..a.len() {
        let a = a.get_unchecked(i);
        let b = b.get_unchecked(i);

        result = M::add(result, M::mul(*a, *b));
    }

    result
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

    let mut result = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..a.len() {
        let a = a.get_unchecked(i);
        let b = b.get_unchecked(i);

        result = M::add(result, M::mul(*a, *b));
        norm_a = M::add(norm_a, M::mul(*a, *a));
        norm_b = M::add(norm_b, M::mul(*b, *b));
    }

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

    let mut result = 0.0;
    for i in 0..a.len() {
        let a = a.get_unchecked(i);
        let b = b.get_unchecked(i);

        let diff = M::sub(*a, *b);
        result = M::add(result, M::mul(diff, diff));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::StdMath;

    #[test]
    fn test_fallback_dot_product() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![0.5, 3.0, 4.0];

        let v = unsafe { fallback_dot_product::<StdMath, 3>(&v1, &v2) };
        assert_eq!(v, 18.5);
    }

    #[test]
    fn test_fallback_cosine() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![0.8, 3.0, 3.0];

        let v = unsafe { fallback_cosine::<StdMath, 3>(&v1, &v2) };
        assert_eq!(v, 0.021929622);
    }

    #[test]
    fn test_fallback_euclidean() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![2.0, 3.0, 4.0];

        let v = unsafe { fallback_euclidean::<StdMath, 3>(&v1, &v2) };
        assert_eq!(v, 3.0);
    }
}
