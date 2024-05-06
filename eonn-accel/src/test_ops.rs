use crate::ops::*;
use crate::arch::*;
use crate::dims::*;
use crate::vector::{Vector, VectorCreateError};
use crate::test_utils::get_sample_vectors;


#[test]
fn test_vector_f32_x1024_fallback_nofma_construct_ok() {
    let (x, y) = get_sample_vectors(1024);
    let x = Vector::<X1024, Fallback, f32>::try_from_vec(x)
        .expect("Create vector");
    let y = Vector::<X1024, Fallback, f32>::try_from_vec(y)
        .expect("Create vector");
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), 1024);
}

#[test]
fn test_vector_f32_x1024_fallback_nofma_length_miss_match() {
    let (x, _) = get_sample_vectors(768);
    let x = Vector::<X1024, Fallback, f32>::try_from_vec(x)
        .expect_err("Length should be marked as invalid");
    assert!(matches!(x, VectorCreateError::BadDimensions { .. }));
}

#[test]
fn test_vector_f32_x1024_fallback_nofma_non_finite() {
    let (mut x, _) = get_sample_vectors(1024);
    x[0] = f32::INFINITY;
    let x = Vector::<X1024, Fallback, f32>::try_from_vec(x)
        .expect_err("Length should be marked as invalid");
    assert!(matches!(x, VectorCreateError::NonFinite));
}

#[test]
fn test_vector_f32_x1024_fallback_nofma_nan() {
    let (mut x, _) = get_sample_vectors(1024);
    x[0] = f32::NAN;
    let x = Vector::<X1024, Fallback, f32>::try_from_vec(x)
        .expect_err("Length should be marked as invalid");
    assert!(matches!(x, VectorCreateError::NonFinite));
}

