mod avx2_cosine;
mod avx2_dot_product;
mod avx2_euclidean;
mod utils;
mod fallback_dot_product;

pub(crate) use utils::*;

pub use self::avx2_dot_product::*;
pub use self::avx2_cosine::*;
pub use self::avx2_euclidean::*;
pub use self::fallback_dot_product::*;