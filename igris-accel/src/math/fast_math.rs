use super::Math;
use core::intrinsics;

/// Basic math operations backed by fast-math intrinsics.
///
/// WARNING:
/// These operations can assume all values are finite, there is no
/// guaranteed behaviour if this condition is breached.
pub struct FastMath;

impl Math for FastMath {
    #[inline(always)]
    fn add(a: f32, b: f32) -> f32 {
        debug_assert!(a.is_finite());
        debug_assert!(b.is_finite());
        unsafe { intrinsics::fadd_fast(a, b) }
    }

    #[inline(always)]
    fn sub(a: f32, b: f32) -> f32 {
        debug_assert!(a.is_finite());
        debug_assert!(b.is_finite());
        unsafe { intrinsics::fsub_fast(a, b) }
    }

    #[inline(always)]
    fn mul(a: f32, b: f32) -> f32 {
        debug_assert!(a.is_finite());
        debug_assert!(b.is_finite());
        unsafe { intrinsics::fmul_fast(a, b) }
    }

    #[inline(always)]
    fn div(a: f32, b: f32) -> f32 {
        debug_assert!(a.is_finite());
        debug_assert!(b.is_finite());
        unsafe { intrinsics::fdiv_fast(a, b) }
    }
}
