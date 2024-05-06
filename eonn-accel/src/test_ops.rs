use crate::arch::*;
use crate::dims::*;
use crate::ops::*;
use crate::test_utils::{
    assert_is_close,
    assert_is_close_vector,
    get_sample_vectors,
    simple_angular_hyperplane,
    simple_cosine,
    simple_dot,
    simple_euclidean,
    simple_euclidean_hyperplane,
};
use crate::vector::{Vector, VectorCreateError};

#[test]
fn test_vector_f32_non_finite() {
    let (mut x, _) = get_sample_vectors(1024);
    x[0] = f32::INFINITY;
    let x = Vector::<X1024, Fallback, f32>::try_from_vec(x)
        .expect_err("Should reject non-finite value");
    assert!(matches!(x, VectorCreateError::NonFinite));
}

#[test]
fn test_vector_f32_nan() {
    let (mut x, _) = get_sample_vectors(1024);
    x[0] = f32::NAN;
    let x =
        Vector::<X1024, Fallback, f32>::try_from_vec(x).expect_err("Should reject Nan");
    assert!(matches!(x, VectorCreateError::NonFinite));
}

#[test]
fn test_vector_construct_ok() {
    let (x, y) = get_sample_vectors(1024);
    let x = Vector::<X1024, Fallback, f32>::try_from_vec(x).expect("Create vector");
    let y = Vector::<X1024, Fallback, f32>::try_from_vec(y).expect("Create vector");
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), 1024);
}

#[test]
fn test_vector_length_miss_match() {
    let (x, _) = get_sample_vectors(768);
    let x = Vector::<X1024, Fallback, f32>::try_from_vec(x)
        .expect_err("Length should be marked as invalid");
    assert!(matches!(x, VectorCreateError::BadDimensions { .. }));
}

macro_rules! define_vector_op_test_suite {
    (
        suite_name = $name:ident,
        dim = $dim:ident,
        len = $len:expr,
        arch = $arch:ident,
        tp = $tp:ident,
    ) => {
            paste::paste! {
                #[test]
                fn [<test_vector_ $name _dot>]() {
                    let (x, y) = get_sample_vectors($len);
                    let x = Vector::<$dim, $arch, $tp>::try_from_vec(x)
                        .expect("Create vector");
                    let y = Vector::<$dim, $arch, $tp>::try_from_vec(y)
                        .expect("Create vector");
                    let res = x.dot(&y);
                    assert_is_close(res, simple_dot(&x, &y));
                }

                #[test]
                fn [<test_vector_ $name _dist_dot>]() {
                    let (x, y) = get_sample_vectors($len);
                    let x = Vector::<$dim, $arch, $tp>::try_from_vec(x)
                        .expect("Create vector");
                    let y = Vector::<$dim, $arch, $tp>::try_from_vec(y)
                        .expect("Create vector");
                    let res = x.dist_dot(&y);
                    // Technically there is the if/else logic, but this test covers the rest.
                    assert_is_close(res, 1.0 - simple_dot(&x, &y));
                }

                #[test]
                fn [<test_vector_ $name _dist_cosine>]() {
                    let (x, y) = get_sample_vectors($len);
                    let x = Vector::<$dim, $arch, $tp>::try_from_vec(x)
                        .expect("Create vector");
                    let y = Vector::<$dim, $arch, $tp>::try_from_vec(y)
                        .expect("Create vector");
                    let res = x.dist_cosine(&y);
                    assert_is_close(res, simple_cosine(&x, &y));
                }

                #[test]
                fn [<test_vector_ $name _dist_squared_euclidean>]() {
                    let (x, y) = get_sample_vectors($len);
                    let x = Vector::<$dim, $arch, $tp>::try_from_vec(x)
                        .expect("Create vector");
                    let y = Vector::<$dim, $arch, $tp>::try_from_vec(y)
                        .expect("Create vector");
                    let res = x.dist_squared_euclidean(&y);
                    assert_is_close(res, simple_euclidean(&x, &y));
                }

                #[test]
                fn [<test_vector_ $name _squared_norm>]() {
                    let (x, _) = get_sample_vectors($len);
                    let x = Vector::<$dim, $arch, $tp>::try_from_vec(x)
                        .expect("Create vector");
                    let res = x.squared_norm();
                    assert_is_close(res, simple_dot(&x, &x));
                }

                #[test]
                fn [<test_vector_ $name _normalize>]() {
                    let (mut expected, _) = get_sample_vectors($len);
                    let mut sample = Vector::<$dim, $arch, $tp>::try_from_vec(expected.clone())
                        .expect("Create vector");
                    sample.normalize();

                    let norm = simple_dot(&expected, &expected).sqrt();
                    for v in expected.iter_mut() {
                        *v /= norm;
                    }

                    assert_is_close_vector(&sample, &expected);
                }

                #[test]
                fn [<test_vector_ $name _angular_hyperplane>]() {
                    let (x, y) = get_sample_vectors($len);
                    let x = Vector::<$dim, $arch, $tp>::try_from_vec(x)
                        .expect("Create vector");
                    let y = Vector::<$dim, $arch, $tp>::try_from_vec(y)
                        .expect("Create vector");

                    let hyperplane = x.angular_hyperplane(&y);
                    let expected_hyperplane = simple_angular_hyperplane(&x, &y);

                    assert_is_close_vector(&hyperplane, &expected_hyperplane);
                }

                #[test]
                fn [<test_vector_ $name _euclidean_hyperplane>]() {
                    let (x, y) = get_sample_vectors($len);
                    let x = Vector::<$dim, $arch, $tp>::try_from_vec(x)
                        .expect("Create vector");
                    let y = Vector::<$dim, $arch, $tp>::try_from_vec(y)
                        .expect("Create vector");

                    let (hyperplane, offset) = x.euclidean_hyperplane(&y);
                    let (expected_hyperplane, expected_offset) = simple_euclidean_hyperplane(&x, &y);

                    assert_is_close(offset, expected_offset);
                    assert_is_close_vector(&hyperplane, &expected_hyperplane);
                }

                #[test]
                fn [<test_vector_ $name _add_value>]() {
                    let (mut expected, _) = get_sample_vectors($len);
                    let mut sample = Vector::<$dim, $arch, $tp>::try_from_vec(expected.clone())
                        .expect("Create vector");
                    sample += 2.0;

                    for v in expected.iter_mut() {
                        *v += 2.0;
                    }

                    assert_is_close_vector(&sample, &expected);
                }

                #[test]
                fn [<test_vector_ $name _sub_value>]() {
                    let (mut expected, _) = get_sample_vectors($len);
                    let mut sample = Vector::<$dim, $arch, $tp>::try_from_vec(expected.clone())
                        .expect("Create vector");
                    sample -= 2.0;

                    for v in expected.iter_mut() {
                        *v -= 2.0;
                    }

                    assert_is_close_vector(&sample, &expected);
                }

                #[test]
                fn [<test_vector_ $name _div_value>]() {
                    let (mut expected, _) = get_sample_vectors($len);
                    let mut sample = Vector::<$dim, $arch, $tp>::try_from_vec(expected.clone())
                        .expect("Create vector");
                    sample /= 2.0;

                    for v in expected.iter_mut() {
                        *v /= 2.0;
                    }

                    assert_is_close_vector(&sample, &expected);
                }

                #[test]
                fn [<test_vector_ $name _mul_value>]() {
                    let (mut expected, _) = get_sample_vectors($len);
                    let mut sample = Vector::<$dim, $arch, $tp>::try_from_vec(expected.clone())
                        .expect("Create vector");
                    sample *= 2.0;

                    for v in expected.iter_mut() {
                        *v *= 2.0;
                    }

                    assert_is_close_vector(&sample, &expected);
                }
        }
    };
}

// Fallback routines wo/ffast-math
define_vector_op_test_suite!(
    suite_name = f32_x1024_fallback_nofma,
    dim = X1024,
    len = 1024,
    arch = Fallback,
    tp = f32,
);
define_vector_op_test_suite!(
    suite_name = f32_x768_fallback_nofma,
    dim = X768,
    len = 768,
    arch = Fallback,
    tp = f32,
);
define_vector_op_test_suite!(
    suite_name = f32_x512_fallback_nofma,
    dim = X512,
    len = 512,
    arch = Fallback,
    tp = f32,
);
define_vector_op_test_suite!(
    suite_name = f32_xany_fallback_nofma,
    dim = XAny,
    len = 537,
    arch = Fallback,
    tp = f32,
);

// Fallback routines w/ffast-math
#[cfg(feature = "nightly")]
define_vector_op_test_suite!(
    suite_name = f32_x1024_fallback_fma,
    dim = X1024,
    len = 1024,
    arch = FallbackFma,
    tp = f32,
);
#[cfg(feature = "nightly")]
define_vector_op_test_suite!(
    suite_name = f32_x768_fallback_fma,
    dim = X768,
    len = 768,
    arch = FallbackFma,
    tp = f32,
);
#[cfg(feature = "nightly")]
define_vector_op_test_suite!(
    suite_name = f32_x512_fallback_fma,
    dim = X512,
    len = 512,
    arch = FallbackFma,
    tp = f32,
);
#[cfg(feature = "nightly")]
define_vector_op_test_suite!(
    suite_name = f32_xany_fallback_fma,
    dim = XAny,
    len = 537,
    arch = FallbackFma,
    tp = f32,
);

// AVX2 routines wo/fma
define_vector_op_test_suite!(
    suite_name = f32_x1024_avx2_nofma,
    dim = X1024,
    len = 1024,
    arch = Avx2,
    tp = f32,
);
define_vector_op_test_suite!(
    suite_name = f32_x768_avx2_nofma,
    dim = X768,
    len = 768,
    arch = Avx2,
    tp = f32,
);
define_vector_op_test_suite!(
    suite_name = f32_x512_avx2_nofma,
    dim = X512,
    len = 512,
    arch = Avx2,
    tp = f32,
);
define_vector_op_test_suite!(
    suite_name = f32_xany_avx2_nofma,
    dim = XAny,
    len = 537,
    arch = Avx2,
    tp = f32,
);

// AVX2 routines w/fma
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_vector_op_test_suite!(
    suite_name = f32_x1024_avx2_fma,
    dim = X1024,
    len = 1024,
    arch = Avx2Fma,
    tp = f32,
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_vector_op_test_suite!(
    suite_name = f32_x768_avx2fma,
    dim = X768,
    len = 768,
    arch = Avx2Fma,
    tp = f32,
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_vector_op_test_suite!(
    suite_name = f32_x512_avx2_fma,
    dim = X512,
    len = 512,
    arch = Avx2Fma,
    tp = f32,
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_vector_op_test_suite!(
    suite_name = f32_xany_avx2_fma,
    dim = XAny,
    len = 537,
    arch = Avx2Fma,
    tp = f32,
);

// AVX512 routines wo/fma
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly",
    target_feature = "avx512f"
))]
define_vector_op_test_suite!(
    suite_name = f32_x1024_avx512_nofma,
    dim = X1024,
    len = 1024,
    arch = Avx512Fma,
    tp = f32,
);
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly",
    target_feature = "avx512f"
))]
define_vector_op_test_suite!(
    suite_name = f32_x768_avx512_nofma,
    dim = X768,
    len = 768,
    arch = Avx512Fma,
    tp = f32,
);
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly",
    target_feature = "avx512f"
))]
define_vector_op_test_suite!(
    suite_name = f32_x512_avx512_nofma,
    dim = X512,
    len = 512,
    arch = Avx512Fma,
    tp = f32,
);
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly",
    target_feature = "avx512f"
))]
define_vector_op_test_suite!(
    suite_name = f32_xany_avx512_nofma,
    dim = XAny,
    len = 537,
    arch = Avx512Fma,
    tp = f32,
);

// AVX512 routines w/fma
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly",
    target_feature = "avx512f"
))]
define_vector_op_test_suite!(
    suite_name = f32_x1024_avx512_fma,
    dim = X1024,
    len = 1024,
    arch = Avx512Fma,
    tp = f32,
);
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly",
    target_feature = "avx512f"
))]
define_vector_op_test_suite!(
    suite_name = f32_x768_avx512_fma,
    dim = X768,
    len = 768,
    arch = Avx512Fma,
    tp = f32,
);
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly",
    target_feature = "avx512f"
))]
define_vector_op_test_suite!(
    suite_name = f32_x512_avx512_fma,
    dim = X512,
    len = 512,
    arch = Avx512Fma,
    tp = f32,
);
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly",
    target_feature = "avx512f"
))]
define_vector_op_test_suite!(
    suite_name = f32_xany_avx512_fma,
    dim = XAny,
    len = 537,
    arch = Avx512Fma,
    tp = f32,
);
