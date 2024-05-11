use core::intrinsics;

use super::Math;

/// Basic math operations backed by fast-math intrinsics.
pub struct FastMath;

impl Math for FastMath {
    #[inline(always)]
    fn add(a: f32, b: f32) -> f32 {
        if cfg!(miri) {
            a + b
        } else {
            intrinsics::fadd_algebraic(a, b)
        }
    }

    #[inline(always)]
    fn sub(a: f32, b: f32) -> f32 {
        if cfg!(miri) {
            a - b
        } else {
            intrinsics::fsub_algebraic(a, b)
        }
    }

    #[inline(always)]
    fn mul(a: f32, b: f32) -> f32 {
        if cfg!(miri) {
            a * b
        } else {
            intrinsics::fmul_algebraic(a, b)
        }
    }

    #[inline(always)]
    fn div(a: f32, b: f32) -> f32 {
        if cfg!(miri) {
            a / b
        } else {
            intrinsics::fdiv_algebraic(a, b)
        }
    }
}
