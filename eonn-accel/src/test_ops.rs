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
                    assert_is_close(res, simple_dot(x.as_ref(), y.as_ref()));
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
                    assert_is_close(res, 1.0 - simple_dot(x.as_ref(), y.as_ref()));
                }

                #[test]
                fn [<test_vector_ $name _dist_cosine>]() {
                    let (x, y) = get_sample_vectors($len);
                    let x = Vector::<$dim, $arch, $tp>::try_from_vec(x)
                        .expect("Create vector");
                    let y = Vector::<$dim, $arch, $tp>::try_from_vec(y)
                        .expect("Create vector");
                    let res = x.dist_cosine(&y);
                    assert_is_close(res, simple_cosine(x.as_ref(), y.as_ref()));
                }

                #[test]
                fn [<test_vector_ $name _dist_squared_euclidean>]() {
                    let (x, y) = get_sample_vectors($len);
                    let x = Vector::<$dim, $arch, $tp>::try_from_vec(x)
                        .expect("Create vector");
                    let y = Vector::<$dim, $arch, $tp>::try_from_vec(y)
                        .expect("Create vector");
                    let res = x.dist_squared_euclidean(&y);
                    assert_is_close(res, simple_euclidean(x.as_ref(), y.as_ref()));
                }

                #[test]
                fn [<test_vector_ $name _squared_norm>]() {
                    let (x, _) = get_sample_vectors($len);
                    let x = Vector::<$dim, $arch, $tp>::try_from_vec(x)
                        .expect("Create vector");
                    let res = x.squared_norm();
                    assert_is_close(res, simple_dot(x.as_ref(), x.as_ref()));
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

                    assert_is_close_vector(sample.as_ref(), expected.as_ref());
                }

                #[test]
                fn [<test_vector_ $name _angular_hyperplane>]() {
                    let (x, y) = get_sample_vectors($len);
                    let x = Vector::<$dim, $arch, $tp>::try_from_vec(x)
                        .expect("Create vector");
                    let y = Vector::<$dim, $arch, $tp>::try_from_vec(y)
                        .expect("Create vector");

                    let hyperplane = x.angular_hyperplane(&y);
                    let expected_hyperplane = simple_angular_hyperplane(x.as_ref(), y.as_ref());

                    assert_is_close_vector(hyperplane.as_ref(), expected_hyperplane.as_ref());
                }

                #[test]
                fn [<test_vector_ $name _euclidean_hyperplane>]() {
                    let (x, y) = get_sample_vectors($len);
                    let x = Vector::<$dim, $arch, $tp>::try_from_vec(x)
                        .expect("Create vector");
                    let y = Vector::<$dim, $arch, $tp>::try_from_vec(y)
                        .expect("Create vector");

                    let (hyperplane, offset) = x.euclidean_hyperplane(&y);
                    let (expected_hyperplane, expected_offset) = simple_euclidean_hyperplane(x.as_ref(), y.as_ref());

                    assert_is_close(offset, expected_offset);
                    assert_is_close_vector(hyperplane.as_ref(), expected_hyperplane.as_ref());
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

                    assert_is_close_vector(sample.as_ref(), expected.as_ref());
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

                    assert_is_close_vector(sample.as_ref(), expected.as_ref());
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

                    assert_is_close_vector(sample.as_ref(), expected.as_ref());
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

                    assert_is_close_vector(sample.as_ref(), expected.as_ref());
                }

                #[test]
                fn [<test_vector_ $name _min>]() {
                    let (x, _) = get_sample_vectors($len);
                    let x = Vector::<$dim, $arch, $tp>::try_from_vec(x)
                        .expect("Create vector");

                    let expected_min = x.as_ref()
                        .iter()
                        .fold(f32::INFINITY, |acc, v| acc.min(*v));
                    assert_eq!(x.min(), expected_min);
                }

                #[test]
                fn [<test_vector_ $name _max>]() {
                    let (x, _) = get_sample_vectors($len);
                    let x = Vector::<$dim, $arch, $tp>::try_from_vec(x)
                        .expect("Create vector");

                    let expected_max = x.as_ref()
                        .iter()
                        .fold(f32::NEG_INFINITY, |acc, v| acc.max(*v));
                    assert_eq!(x.max(), expected_max);
                }

                #[test]
                fn [<test_vector_ $name _sum>]() {
                    let (_, x) = get_sample_vectors($len);
                    let x = Vector::<$dim, $arch, $tp>::try_from_vec(x)
                        .expect("Create vector");

                    let expected_sum = x.as_ref()
                        .iter()
                        .fold(0.0, |acc, v| acc + *v);
                    assert_is_close(x.sum(), expected_sum);
                }

                #[test]
                fn [<test_vector_ $name _mean>]() {
                    let (x, _) = get_sample_vectors($len);
                    let x = Vector::<$dim, $arch, $tp>::try_from_vec(x)
                        .expect("Create vector");

                    let expected_mean = x.as_ref()
                        .iter()
                        .fold(0.0, |acc, v| acc + *v) / x.len() as f32;
                    assert_is_close(x.mean(), expected_mean);
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
    arch = Avx512,
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
    arch = Avx512,
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
    arch = Avx512,
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
    arch = Avx512,
    tp = f32,
);

macro_rules! define_vector_op_auto_arch_test_suite {
    (
        suite_name = $name:ident,
        dim = $dim:ident,
        len = $len:expr,
        variant = $variant:ident,
        tp = $tp:ident,
    ) => {
            paste::paste! {
                #[test]
                fn [<test_vector_ $name _dot>]() {
                    let (x, y) = get_sample_vectors($len);
                    let mut x = Vector::<$dim, Auto, $tp>::try_from_vec(x)
                        .expect("Create vector");
                    x.set_ops(Auto(SelectedArch::$variant));
                    let mut y = Vector::<$dim, Auto, $tp>::try_from_vec(y)
                        .expect("Create vector");
                    y.set_ops(Auto(SelectedArch::$variant));
                    let res = x.dot(&y);
                    assert_is_close(res, simple_dot(x.as_ref(), y.as_ref()));
                }

                #[test]
                fn [<test_vector_ $name _dist_dot>]() {
                    let (x, y) = get_sample_vectors($len);
                    let mut x = Vector::<$dim, Auto, $tp>::try_from_vec(x)
                        .expect("Create vector");
                    x.set_ops(Auto(SelectedArch::$variant));
                    let mut y = Vector::<$dim, Auto, $tp>::try_from_vec(y)
                        .expect("Create vector");
                    y.set_ops(Auto(SelectedArch::$variant));
                    let res = x.dist_dot(&y);
                    // Technically there is the if/else logic, but this test covers the rest.
                    assert_is_close(res, 1.0 - simple_dot(x.as_ref(), y.as_ref()));
                }

                #[test]
                fn [<test_vector_ $name _dist_cosine>]() {
                    let (x, y) = get_sample_vectors($len);
                    let mut x = Vector::<$dim, Auto, $tp>::try_from_vec(x)
                        .expect("Create vector");
                    x.set_ops(Auto(SelectedArch::$variant));
                    let mut y = Vector::<$dim, Auto, $tp>::try_from_vec(y)
                        .expect("Create vector");
                    y.set_ops(Auto(SelectedArch::$variant));
                    let res = x.dist_cosine(&y);
                    assert_is_close(res, simple_cosine(x.as_ref(), y.as_ref()));
                }

                #[test]
                fn [<test_vector_ $name _dist_squared_euclidean>]() {
                    let (x, y) = get_sample_vectors($len);
                    let mut x = Vector::<$dim, Auto, $tp>::try_from_vec(x)
                        .expect("Create vector");
                    x.set_ops(Auto(SelectedArch::$variant));
                    let mut y = Vector::<$dim, Auto, $tp>::try_from_vec(y)
                        .expect("Create vector");
                    y.set_ops(Auto(SelectedArch::$variant));
                    let res = x.dist_squared_euclidean(&y);
                    assert_is_close(res, simple_euclidean(x.as_ref(), y.as_ref()));
                }

                #[test]
                fn [<test_vector_ $name _squared_norm>]() {
                    let (x, _) = get_sample_vectors($len);
                    let mut x = Vector::<$dim, Auto, $tp>::try_from_vec(x)
                        .expect("Create vector");
                    x.set_ops(Auto(SelectedArch::$variant));
                    let res = x.squared_norm();
                    assert_is_close(res, simple_dot(x.as_ref(), x.as_ref()));
                }

                #[test]
                fn [<test_vector_ $name _normalize>]() {
                    let (mut expected, _) = get_sample_vectors($len);
                    let mut sample = Vector::<$dim, Auto, $tp>::try_from_vec(expected.clone())
                        .expect("Create vector");
                    sample.set_ops(Auto(SelectedArch::$variant));
                    sample.normalize();

                    let norm = simple_dot(&expected, &expected).sqrt();
                    for v in expected.iter_mut() {
                        *v /= norm;
                    }

                    assert_is_close_vector(sample.as_ref(), expected.as_ref());
                }

                #[test]
                fn [<test_vector_ $name _angular_hyperplane>]() {
                    let (x, y) = get_sample_vectors($len);
                    let mut x = Vector::<$dim, Auto, $tp>::try_from_vec(x)
                        .expect("Create vector");
                    x.set_ops(Auto(SelectedArch::$variant));
                    let mut y = Vector::<$dim, Auto, $tp>::try_from_vec(y)
                        .expect("Create vector");
                    y.set_ops(Auto(SelectedArch::$variant));

                    let hyperplane = x.angular_hyperplane(&y);
                    let expected_hyperplane = simple_angular_hyperplane(x.as_ref(), y.as_ref());

                    assert_is_close_vector(hyperplane.as_ref(), expected_hyperplane.as_ref());
                }

                #[test]
                fn [<test_vector_ $name _euclidean_hyperplane>]() {
                    let (x, y) = get_sample_vectors($len);
                    let mut x = Vector::<$dim, Auto, $tp>::try_from_vec(x)
                        .expect("Create vector");
                    x.set_ops(Auto(SelectedArch::$variant));
                    let mut y = Vector::<$dim, Auto, $tp>::try_from_vec(y)
                        .expect("Create vector");
                    y.set_ops(Auto(SelectedArch::$variant));

                    let (hyperplane, offset) = x.euclidean_hyperplane(&y);
                    let (expected_hyperplane, expected_offset) = simple_euclidean_hyperplane(x.as_ref(), y.as_ref());

                    assert_is_close(offset, expected_offset);
                    assert_is_close_vector(hyperplane.as_ref(), expected_hyperplane.as_ref());
                }

                #[test]
                fn [<test_vector_ $name _add_value>]() {
                    let (mut expected, _) = get_sample_vectors($len);
                    let mut sample = Vector::<$dim, Auto, $tp>::try_from_vec(expected.clone())
                        .expect("Create vector");
                    sample.set_ops(Auto(SelectedArch::$variant));
                    sample += 2.0;

                    for v in expected.iter_mut() {
                        *v += 2.0;
                    }

                    assert_is_close_vector(sample.as_ref(), expected.as_ref());
                }

                #[test]
                fn [<test_vector_ $name _sub_value>]() {
                    let (mut expected, _) = get_sample_vectors($len);
                    let mut sample = Vector::<$dim, Auto, $tp>::try_from_vec(expected.clone())
                        .expect("Create vector");
                    sample.set_ops(Auto(SelectedArch::$variant));
                    sample -= 2.0;

                    for v in expected.iter_mut() {
                        *v -= 2.0;
                    }

                    assert_is_close_vector(sample.as_ref(), expected.as_ref());
                }

                #[test]
                fn [<test_vector_ $name _div_value>]() {
                    let (mut expected, _) = get_sample_vectors($len);
                    let mut sample = Vector::<$dim, Auto, $tp>::try_from_vec(expected.clone())
                        .expect("Create vector");
                    sample.set_ops(Auto(SelectedArch::$variant));
                    sample /= 2.0;

                    for v in expected.iter_mut() {
                        *v /= 2.0;
                    }

                    assert_is_close_vector(sample.as_ref(), expected.as_ref());
                }

                #[test]
                fn [<test_vector_ $name _mul_value>]() {
                    let (mut expected, _) = get_sample_vectors($len);
                    let mut sample = Vector::<$dim, Auto, $tp>::try_from_vec(expected.clone())
                        .expect("Create vector");
                    sample.set_ops(Auto(SelectedArch::$variant));
                    sample *= 2.0;

                    for v in expected.iter_mut() {
                        *v *= 2.0;
                    }

                    assert_is_close_vector(sample.as_ref(), expected.as_ref());
                }
        }
    };
}

// Auto select routines w/ Fallback enabled
define_vector_op_auto_arch_test_suite!(
    suite_name = f32_x1024_auto_fallback_variant,
    dim = X1024,
    len = 1024,
    variant = Fallback,
    tp = f32,
);
define_vector_op_auto_arch_test_suite!(
    suite_name = f32_x768_auto_fallback_variant,
    dim = X768,
    len = 768,
    variant = Fallback,
    tp = f32,
);
define_vector_op_auto_arch_test_suite!(
    suite_name = f32_x512_auto_fallback_variant,
    dim = X512,
    len = 512,
    variant = Fallback,
    tp = f32,
);
define_vector_op_auto_arch_test_suite!(
    suite_name = f32_xauto_auto_fallback_variant,
    dim = XAny,
    len = 537,
    variant = Fallback,
    tp = f32,
);

// Auto select routines w/ FallbackFma enabled
#[cfg(feature = "nightly")]
define_vector_op_auto_arch_test_suite!(
    suite_name = f32_x1024_auto_fallbackfma_variant,
    dim = X1024,
    len = 1024,
    variant = FallbackFma,
    tp = f32,
);
#[cfg(feature = "nightly")]
define_vector_op_auto_arch_test_suite!(
    suite_name = f32_x768_auto_fallbackfma_variant,
    dim = X768,
    len = 768,
    variant = FallbackFma,
    tp = f32,
);
#[cfg(feature = "nightly")]
define_vector_op_auto_arch_test_suite!(
    suite_name = f32_x512_auto_fallbackfma_variant,
    dim = X512,
    len = 512,
    variant = FallbackFma,
    tp = f32,
);
#[cfg(feature = "nightly")]
define_vector_op_auto_arch_test_suite!(
    suite_name = f32_xauto_auto_fallbackfma_variant,
    dim = XAny,
    len = 537,
    variant = FallbackFma,
    tp = f32,
);

// Auto select routines w/ AVX2 enabled
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_vector_op_auto_arch_test_suite!(
    suite_name = f32_x1024_auto_avx2_variant,
    dim = X1024,
    len = 1024,
    variant = Avx2,
    tp = f32,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_vector_op_auto_arch_test_suite!(
    suite_name = f32_x768_auto_avx2_variant,
    dim = X768,
    len = 768,
    variant = Avx2,
    tp = f32,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_vector_op_auto_arch_test_suite!(
    suite_name = f32_x512_auto_avx2_variant,
    dim = X512,
    len = 512,
    variant = Avx2,
    tp = f32,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_vector_op_auto_arch_test_suite!(
    suite_name = f32_xauto_auto_avx2_variant,
    dim = XAny,
    len = 537,
    variant = Avx2,
    tp = f32,
);

// Auto select routines w/ AVX2Fma enabled
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_vector_op_auto_arch_test_suite!(
    suite_name = f32_x1024_auto_avx2fma_variant,
    dim = X1024,
    len = 1024,
    variant = Avx2Fma,
    tp = f32,
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_vector_op_auto_arch_test_suite!(
    suite_name = f32_x768_auto_avx2fma_variant,
    dim = X768,
    len = 768,
    variant = Avx2Fma,
    tp = f32,
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_vector_op_auto_arch_test_suite!(
    suite_name = f32_x512_auto_avx2fma_variant,
    dim = X512,
    len = 512,
    variant = Avx2Fma,
    tp = f32,
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_vector_op_auto_arch_test_suite!(
    suite_name = f32_xauto_auto_avx2fma_variant,
    dim = XAny,
    len = 537,
    variant = Avx2Fma,
    tp = f32,
);

// Auto select routines w/ AVX512 enabled
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly",
    target_feature = "avx512f"
))]
define_vector_op_auto_arch_test_suite!(
    suite_name = f32_x1024_auto_avx512_variant,
    dim = X1024,
    len = 1024,
    variant = Avx512,
    tp = f32,
);
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly",
    target_feature = "avx512f"
))]
define_vector_op_auto_arch_test_suite!(
    suite_name = f32_x768_auto_avx512_variant,
    dim = X768,
    len = 768,
    variant = Avx512,
    tp = f32,
);
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly",
    target_feature = "avx512f"
))]
define_vector_op_auto_arch_test_suite!(
    suite_name = f32_x512_auto_avx512_variant,
    dim = X512,
    len = 512,
    variant = Avx512,
    tp = f32,
);
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly",
    target_feature = "avx512f"
))]
define_vector_op_auto_arch_test_suite!(
    suite_name = f32_xauto_auto_avx512_variant,
    dim = XAny,
    len = 537,
    variant = Avx512,
    tp = f32,
);
