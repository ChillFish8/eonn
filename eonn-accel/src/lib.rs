#![allow(internal_features)]
#![cfg_attr(feature = "nightly", feature(core_intrinsics))]
#![cfg_attr(
    all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"),
    feature(avx512_target_feature)
)]
#![cfg_attr(
    all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"),
    feature(stdarch_x86_avx512)
)]

mod arch;
mod dims;
pub(crate) mod math;

#[cfg(feature = "dangerous-access")]
pub mod danger;
#[cfg(not(feature = "dangerous-access"))]
pub(crate) mod danger;
mod ops;
#[cfg(test)]
mod test_ops;
#[cfg(test)]
mod test_utils;
mod types;
mod vector;

pub use self::arch::*;
pub use self::dims::{Dim, X1024, X512, X768};
pub use self::ops::{DangerousOps, SpacialOps};
pub use self::types::VectorType;
pub use self::vector::{Vector, VectorCreateError};
