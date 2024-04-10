mod avx2_angular_hyperplane;
mod avx2_cosine;
mod avx2_dot_product;
mod avx2_euclidean;
mod avx2_euclidean_hyperplane;
mod avx2_norm;
mod avx512_angular_hyperplane;
mod avx512_cosine;
mod avx512_dot_product;
mod avx512_euclidean;
mod avx512_euclidean_hyperplane;
mod avx512_norm;
mod fallback_cosine;
mod fallback_dot_product;
mod fallback_euclidean;
#[cfg(test)]
mod test_utils;
mod utils;

pub(crate) use utils::*;

pub use self::avx2_angular_hyperplane::*;
pub use self::avx2_cosine::*;
pub use self::avx2_dot_product::*;
pub use self::avx2_euclidean::*;
pub use self::avx2_euclidean_hyperplane::*;
pub use self::avx2_norm::*;
pub use self::avx512_angular_hyperplane::*;
pub use self::avx512_cosine::*;
pub use self::avx512_dot_product::*;
pub use self::avx512_euclidean::*;
pub use self::avx512_euclidean_hyperplane::*;
pub use self::avx512_norm::*;
pub use self::fallback_cosine::*;
pub use self::fallback_dot_product::*;
pub use self::fallback_euclidean::*;
