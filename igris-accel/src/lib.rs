#![allow(internal_features)]
#![feature(core_intrinsics)]
#![feature(stdarch_x86_avx512)]

pub mod distance_ops;
pub mod math;

#[cfg(feature = "dangerous-access")]
pub mod danger;
#[cfg(not(feature = "dangerous-access"))]
pub(crate) mod danger;
