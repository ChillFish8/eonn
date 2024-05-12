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
pub trait Math<T> {
    /// Returns the equivalent zero value.
    fn zero() -> T;
    
    /// Returns the equivalent 1.0 value.
    fn one() -> T;

    /// Returns the equivalent 1.0 value.
    fn sqrt(a: T) -> T;
    
    /// Returns if the two values are equal.
    fn eq(a: T, b: T) -> bool;
    
    /// `a + b`
    fn add(a: T, b: T) -> T;

    /// `a - b`
    fn sub(a: T, b: T) -> T;

    /// `a * b`
    fn mul(a: T, b: T) -> T;

    /// `a / b`
    fn div(a: T, b: T) -> T;
}
