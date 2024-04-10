use crate::danger::avx2_dot_product::{
    f32_x1024_avx2_fma_dot,
    f32_x1024_avx2_nofma_dot,
    f32_x512_avx2_fma_dot,
    f32_x512_avx2_nofma_dot,
    f32_x768_avx2_fma_dot,
    f32_x768_avx2_nofma_dot,
};
use crate::danger::utils::cosine;
use crate::danger::{
    f32_x1024_avx2_fma_norm,
    f32_x1024_avx2_nofma_norm,
    f32_x512_avx2_fma_norm,
    f32_x512_avx2_nofma_norm,
    f32_x768_avx2_fma_norm,
    f32_x768_avx2_nofma_norm,
};
use crate::math::{FastMath, StdMath};

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the cosine distance of two `[f32; 1024]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `1024` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x1024_avx2_nofma_cosine(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 1024);
    debug_assert_eq!(y.len(), 1024);

    let norm_x = f32_x1024_avx2_nofma_norm(x);
    let norm_y = f32_x1024_avx2_nofma_norm(y);
    let dot_product = f32_x1024_avx2_nofma_dot(x, y);

    cosine::<StdMath>(dot_product, norm_x, norm_y)
}

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the cosine distance of two `[f32; 768]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `768` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x768_avx2_nofma_cosine(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 768);
    debug_assert_eq!(y.len(), 768);

    let norm_x = f32_x768_avx2_nofma_norm(x);
    let norm_y = f32_x768_avx2_nofma_norm(y);
    let dot_product = f32_x768_avx2_nofma_dot(x, y);

    cosine::<StdMath>(dot_product, norm_x, norm_y)
}

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the cosine distance of two `[f32; 512]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `512` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x512_avx2_nofma_cosine(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 512);
    debug_assert_eq!(y.len(), 512);

    let norm_x = f32_x512_avx2_nofma_norm(x);
    let norm_y = f32_x512_avx2_nofma_norm(y);
    let dot_product = f32_x512_avx2_nofma_dot(x, y);

    cosine::<StdMath>(dot_product, norm_x, norm_y)
}

#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the cosine distance of two `[f32; 1024]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `1024` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x1024_avx2_fma_cosine(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 1024);
    debug_assert_eq!(y.len(), 1024);

    let norm_x = f32_x1024_avx2_fma_norm(x);
    let norm_y = f32_x1024_avx2_fma_norm(y);
    let dot_product = f32_x1024_avx2_fma_dot(x, y);

    cosine::<FastMath>(dot_product, norm_x, norm_y)
}

#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the cosine distance of two `[f32; 768]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `768` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x768_avx2_fma_cosine(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 768);
    debug_assert_eq!(y.len(), 768);

    let norm_x = f32_x768_avx2_fma_norm(x);
    let norm_y = f32_x768_avx2_fma_norm(y);
    let dot_product = f32_x768_avx2_fma_dot(x, y);

    cosine::<FastMath>(dot_product, norm_x, norm_y)
}

#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the cosine distance of two `[f32; 512]` vectors.
///
/// # Safety
///
/// Vectors **MUST** be `512` elements in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_x512_avx2_fma_cosine(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), 512);
    debug_assert_eq!(y.len(), 512);

    let norm_x = f32_x512_avx2_fma_norm(x);
    let norm_y = f32_x512_avx2_fma_norm(y);
    let dot_product = f32_x512_avx2_fma_dot(x, y);

    cosine::<FastMath>(dot_product, norm_x, norm_y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::danger::test_utils::{get_sample_vectors, is_close, simple_cosine};

    #[test]
    fn test_x1024_fma_cosine() {
        let (x, y) = get_sample_vectors(1024);
        let dist = unsafe { f32_x1024_avx2_fma_cosine(&x, &y) };
        assert!(is_close(dist, simple_cosine(&x, &y)))
    }

    #[test]
    fn test_x1024_nofma_cosine() {
        let (x, y) = get_sample_vectors(1024);
        let dist = unsafe { f32_x1024_avx2_nofma_cosine(&x, &y) };
        assert!(is_close(dist, simple_cosine(&x, &y)))
    }

    #[test]
    fn test_x768_fma_cosine() {
        let (x, y) = get_sample_vectors(768);
        let dist = unsafe { f32_x768_avx2_fma_cosine(&x, &y) };
        assert!(is_close(dist, simple_cosine(&x, &y)))
    }

    #[test]
    fn test_x768_nofma_cosine() {
        let (x, y) = get_sample_vectors(768);
        let dist = unsafe { f32_x768_avx2_nofma_cosine(&x, &y) };
        assert!(is_close(dist, simple_cosine(&x, &y)))
    }

    #[test]
    fn test_x512_fma_cosine() {
        let (x, y) = get_sample_vectors(512);
        let dist = unsafe { f32_x512_avx2_fma_cosine(&x, &y) };
        assert!(is_close(dist, simple_cosine(&x, &y)))
    }

    #[test]
    fn test_x512_nofma_cosine() {
        let (x, y) = get_sample_vectors(512);
        let dist = unsafe { f32_x512_avx2_nofma_cosine(&x, &y) };
        assert!(is_close(dist, simple_cosine(&x, &y)))
    }
}
