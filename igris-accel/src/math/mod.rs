mod default;
mod fast_math;

pub use default::StdMath;
pub use fast_math::FastMath;

/// Core simple math operations that can be adjusted for certain features
/// or architectures.
///
/// NOTE:
/// All operations assume floating point values are finite.
pub trait Math {
    /// `a + b`
    fn add(a: f32, b: f32) -> f32;

    /// `a - b`
    fn sub(a: f32, b: f32) -> f32;

    /// `a * b`
    fn mul(a: f32, b: f32) -> f32;

    /// `a / b`
    fn div(a: f32, b: f32) -> f32;
}
