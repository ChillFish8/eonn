mod default;
#[cfg(feature = "nightly")]
mod fast_math;

pub use default::StdMath;
#[cfg(feature = "nightly")]
pub use fast_math::FastMath;

#[cfg(not(feature = "nightly"))]
pub type AutoMath = StdMath;
#[cfg(feature = "nightly")]
pub type AutoMath = FastMath;

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
