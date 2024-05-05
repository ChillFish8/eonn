#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2_angular_hyperplane;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2_cosine;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2_dot_product;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2_euclidean;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2_euclidean_hyperplane;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2_norm;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod avx512_angular_hyperplane;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod avx512_cosine;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod avx512_dot_product;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod avx512_euclidean;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod avx512_euclidean_hyperplane;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod avx512_norm;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod avx512_vector_x_value;
mod fallback_angular_hyperplane;
mod fallback_cosine;
mod fallback_dot_product;
mod fallback_euclidean;
mod fallback_euclidean_hyperplane;
#[cfg(test)]
mod test_utils;
mod utils;

pub(crate) use utils::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::avx2_angular_hyperplane::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::avx2_cosine::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::avx2_dot_product::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::avx2_euclidean::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::avx2_euclidean_hyperplane::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::avx2_norm::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::avx512_angular_hyperplane::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::avx512_cosine::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::avx512_dot_product::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::avx512_euclidean::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::avx512_euclidean_hyperplane::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::avx512_norm::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::avx512_vector_x_value::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::fallback_angular_hyperplane::*;
pub use self::fallback_cosine::*;
pub use self::fallback_dot_product::*;
pub use self::fallback_euclidean::*;
pub use self::fallback_euclidean_hyperplane::*;
