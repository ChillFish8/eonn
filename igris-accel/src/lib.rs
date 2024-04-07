#![allow(internal_features)]
#![feature(core_intrinsics)]

pub mod math;
pub mod distance_ops;

#[cfg(feature = "dangerous-access")]
pub mod danger;
#[cfg(not(feature = "dangerous-access"))]
pub(crate) mod danger;
pub mod bindings;

