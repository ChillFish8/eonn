use super::Math;

/// Standard math operations that apply no specialised handling.
pub struct StdMath;

impl Math for StdMath {
    #[inline(always)]
    fn add(a: f32, b: f32) -> f32 {
        a + b
    }

    #[inline(always)]
    fn sub(a: f32, b: f32) -> f32 {
        a - b
    }

    #[inline(always)]
    fn mul(a: f32, b: f32) -> f32 {
        a * b
    }

    #[inline(always)]
    fn div(a: f32, b: f32) -> f32 {
        a / b
    }
}
