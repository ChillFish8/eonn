mod avx2_cosine;
mod avx2_dot_product;
mod avx2_euclidean;
mod fallback_cosine;
mod fallback_dot_product;
mod fallback_euclidean;
mod utils;

pub(crate) use utils::*;

pub use self::avx2_cosine::*;
pub use self::avx2_dot_product::*;
pub use self::avx2_euclidean::*;
pub use self::fallback_cosine::*;
pub use self::fallback_dot_product::*;
pub use self::fallback_euclidean::*;
