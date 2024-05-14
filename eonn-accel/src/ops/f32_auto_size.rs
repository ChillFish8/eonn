use super::DangerousOps;
use crate::arch::*;
use crate::dims::*;

macro_rules! impl_dangerous_auto_ops {
    ($dim:ident) => {
        impl DangerousOps for ($dim, Auto) {
            #[inline]
            unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
                match self.1 .0 {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2 => crate::danger::f32_xconst_avx2_nofma_dot::<{$dim::DIMS}>(x, y),
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2Fma => crate::danger::f32_xconst_avx2_fma_dot::<{$dim::DIMS}>(x, y),
                    #[cfg(all(
                        any(target_arch = "x86", target_arch = "x86_64"),
                        feature = "nightly"
                    ))]
                    SelectedArch::Avx512 => crate::danger::f32_xconst_avx512_fma_dot::<{$dim::DIMS}>(x, y),
                    SelectedArch::Fallback => crate::danger::generic_xany_fallback_nofma_dot(x, y),
                }
            }

            #[inline]
            unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
                match self.1 .0 {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2 => crate::danger::f32_xconst_avx2_nofma_cosine::<{$dim::DIMS}>(x, y),
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2Fma => crate::danger::f32_xconst_avx2_fma_cosine::<{$dim::DIMS}>(x, y),
                    #[cfg(all(
                        any(target_arch = "x86", target_arch = "x86_64"),
                        feature = "nightly"
                    ))]
                    SelectedArch::Avx512 => crate::danger::f32_xconst_avx512_fma_cosine::<{$dim::DIMS}>(x, y),
                    SelectedArch::Fallback => {
                        crate::danger::generic_xany_fallback_nofma_cosine(x, y)
                    },
                }
            }

            #[inline]
            unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
                match self.1 .0 {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2 => crate::danger::f32_xconst_avx2_nofma_euclidean::<{$dim::DIMS}>(x, y),
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2Fma => crate::danger::f32_xconst_avx2_fma_euclidean::<{$dim::DIMS}>(x, y),
                    #[cfg(all(
                        any(target_arch = "x86", target_arch = "x86_64"),
                        feature = "nightly"
                    ))]
                    SelectedArch::Avx512 => {
                        crate::danger::f32_xconst_avx512_fma_euclidean::<{$dim::DIMS}>(x, y)
                    },
                    SelectedArch::Fallback => {
                        crate::danger::generic_xany_fallback_nofma_euclidean(x, y)
                    },
                }
            }

            #[inline]
            unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
                match self.1 .0 {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2 => {
                        crate::danger::f32_xconst_avx2_nofma_angular_hyperplane::<{$dim::DIMS}>(x, y)
                    },
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2Fma => {
                        crate::danger::f32_xconst_avx2_fma_angular_hyperplane::<{$dim::DIMS}>(x, y)
                    },
                    #[cfg(all(
                        any(target_arch = "x86", target_arch = "x86_64"),
                        feature = "nightly"
                    ))]
                    SelectedArch::Avx512 => {
                        crate::danger::f32_xconst_avx512_fma_angular_hyperplane::<{$dim::DIMS}>(x, y)
                    },
                    SelectedArch::Fallback => {
                        crate::danger::f32_xany_fallback_nofma_angular_hyperplane(x, y)
                    },
                }
            }

            #[inline]
            unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
                match self.1 .0 {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2 => {
                        crate::danger::f32_xconst_avx2_nofma_euclidean_hyperplane::<{$dim::DIMS}>(x, y)
                    },
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2Fma => {
                        crate::danger::f32_xconst_avx2_fma_euclidean_hyperplane::<{$dim::DIMS}>(x, y)
                    },
                    #[cfg(all(
                        any(target_arch = "x86", target_arch = "x86_64"),
                        feature = "nightly"
                    ))]
                    SelectedArch::Avx512 => {
                        crate::danger::f32_xconst_avx512_fma_euclidean_hyperplane::<{$dim::DIMS}>(x, y)
                    },
                    SelectedArch::Fallback => {
                        crate::danger::f32_xany_fallback_nofma_euclidean_hyperplane(x, y)
                    },
                }
            }

            #[inline]
            unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
                match self.1 .0 {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2 => crate::danger::f32_xconst_avx2_nofma_norm::<{$dim::DIMS}>(x),
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2Fma => crate::danger::f32_xconst_avx2_fma_norm::<{$dim::DIMS}>(x),
                    #[cfg(all(
                        any(target_arch = "x86", target_arch = "x86_64"),
                        feature = "nightly"
                    ))]
                    SelectedArch::Avx512 => crate::danger::f32_xconst_avx512_fma_norm::<{$dim::DIMS}>(x),
                    SelectedArch::Fallback => crate::danger::generic_xany_fallback_nofma_dot(x, x),
                }
            }

            #[inline]
            unsafe fn add_value(&self, x: &mut [f32], val: f32) {
                match self.1 .0 {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2 => {
                        crate::danger::f32_xconst_avx2_nofma_add_value::<{$dim::DIMS}>(x, val)
                    },
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2Fma => {
                        crate::danger::f32_xconst_avx2_nofma_add_value::<{$dim::DIMS}>(x, val)
                    },
                    #[cfg(all(
                        any(target_arch = "x86", target_arch = "x86_64"),
                        feature = "nightly"
                    ))]
                    SelectedArch::Avx512 => {
                        crate::danger::f32_xconst_avx512_nofma_add_value::<{$dim::DIMS}>(x, val)
                    },
                    SelectedArch::Fallback => {
                        crate::danger::generic_xany_fallback_nofma_add_value(x, val)
                    },
                }
            }

            #[inline]
            unsafe fn sub_value(&self, x: &mut [f32], val: f32) {
                match self.1 .0 {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2 => {
                        crate::danger::f32_xconst_avx2_nofma_sub_value::<{$dim::DIMS}>(x, val)
                    },
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2Fma => {
                        crate::danger::f32_xconst_avx2_nofma_sub_value::<{$dim::DIMS}>(x, val)
                    },
                    #[cfg(all(
                        any(target_arch = "x86", target_arch = "x86_64"),
                        feature = "nightly"
                    ))]
                    SelectedArch::Avx512 => {
                        crate::danger::f32_xconst_avx512_nofma_sub_value::<{$dim::DIMS}>(x, val)
                    },
                    SelectedArch::Fallback => {
                        crate::danger::generic_xany_fallback_nofma_sub_value(x, val)
                    },
                }
            }

            #[inline]
            unsafe fn mul_value(&self, x: &mut [f32], val: f32) {
                match self.1 .0 {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2 => {
                        crate::danger::f32_xconst_avx2_nofma_mul_value::<{$dim::DIMS}>(x, val)
                    },
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2Fma => {
                        crate::danger::f32_xconst_avx2_nofma_mul_value::<{$dim::DIMS}>(x, val)
                    },
                    #[cfg(all(
                        any(target_arch = "x86", target_arch = "x86_64"),
                        feature = "nightly"
                    ))]
                    SelectedArch::Avx512 => {
                        crate::danger::f32_xconst_avx512_nofma_mul_value::<{$dim::DIMS}>(x, val)
                    },
                    SelectedArch::Fallback => {
                        crate::danger::generic_xany_fallback_nofma_mul_value(x, val)
                    },
                }
            }

            #[inline]
            unsafe fn div_value(&self, x: &mut [f32], val: f32) {
                match self.1 .0 {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2 => {
                        crate::danger::f32_xconst_avx2_nofma_div_value::<{$dim::DIMS}>(x, val)
                    },
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2Fma => {
                        crate::danger::f32_xconst_avx2_nofma_div_value::<{$dim::DIMS}>(x, val)
                    },
                    #[cfg(all(
                        any(target_arch = "x86", target_arch = "x86_64"),
                        feature = "nightly"
                    ))]
                    SelectedArch::Avx512 => {
                        crate::danger::f32_xconst_avx512_nofma_div_value::<{$dim::DIMS}>(x, val)
                    },
                    SelectedArch::Fallback => {
                        crate::danger::generic_xany_fallback_nofma_div_value(x, val)
                    },
                }
            }

            #[inline]
            unsafe fn add_vertical(&self, x: &mut [f32], y: &[f32]) {
                match self.1 .0 {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2 => {
                        crate::danger::f32_xconst_avx2_nofma_add_vertical::<{$dim::DIMS}>(x, y)
                    },
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2Fma => {
                        crate::danger::f32_xconst_avx2_nofma_add_vertical::<{$dim::DIMS}>(x, y)
                    },
                    #[cfg(all(
                        any(target_arch = "x86", target_arch = "x86_64"),
                        feature = "nightly"
                    ))]
                    SelectedArch::Avx512 => {
                        crate::danger::f32_xconst_avx512_nofma_add_vertical::<{$dim::DIMS}>(x, y)
                    },
                    SelectedArch::Fallback => {
                        crate::danger::generic_xany_fallback_nofma_add_vertical(x, y)
                    },
                }
            }

            #[inline]
            unsafe fn sub_vertical(&self, x: &mut [f32], y: &[f32]) {
                match self.1 .0 {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2 => {
                        crate::danger::f32_xconst_avx2_nofma_sub_vertical::<{$dim::DIMS}>(x, y)
                    },
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2Fma => {
                        crate::danger::f32_xconst_avx2_nofma_sub_vertical::<{$dim::DIMS}>(x, y)
                    },
                    #[cfg(all(
                        any(target_arch = "x86", target_arch = "x86_64"),
                        feature = "nightly"
                    ))]
                    SelectedArch::Avx512 => {
                        crate::danger::f32_xconst_avx512_nofma_sub_vertical::<{$dim::DIMS}>(x, y)
                    },
                    SelectedArch::Fallback => {
                        crate::danger::generic_xany_fallback_nofma_sub_vertical(x, y)
                    },
                }
            }

            #[inline]
            unsafe fn mul_vertical(&self, x: &mut [f32], y: &[f32]) {
                match self.1 .0 {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2 => {
                        crate::danger::f32_xconst_avx2_nofma_mul_vertical::<{$dim::DIMS}>(x, y)
                    },
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2Fma => {
                        crate::danger::f32_xconst_avx2_nofma_mul_vertical::<{$dim::DIMS}>(x, y)
                    },
                    #[cfg(all(
                        any(target_arch = "x86", target_arch = "x86_64"),
                        feature = "nightly"
                    ))]
                    SelectedArch::Avx512 => {
                        crate::danger::f32_xconst_avx512_nofma_mul_vertical::<{$dim::DIMS}>(x, y)
                    },
                    SelectedArch::Fallback => {
                        crate::danger::generic_xany_fallback_nofma_mul_vertical(x, y)
                    },
                }
            }

            #[inline]
            unsafe fn div_vertical(&self, x: &mut [f32], y: &[f32]) {
                match self.1 .0 {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2 => {
                        crate::danger::f32_xconst_avx2_nofma_div_vertical::<{$dim::DIMS}>(x, y)
                    },
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2Fma => {
                        crate::danger::f32_xconst_avx2_nofma_div_vertical::<{$dim::DIMS}>(x, y)
                    },
                    #[cfg(all(
                        any(target_arch = "x86", target_arch = "x86_64"),
                        feature = "nightly"
                    ))]
                    SelectedArch::Avx512 => {
                        crate::danger::f32_xconst_avx512_nofma_div_vertical::<{$dim::DIMS}>(x, y)
                    },
                    SelectedArch::Fallback => {
                        crate::danger::generic_xany_fallback_nofma_div_vertical(x, y)
                    },
                }
            }

            #[inline]
            unsafe fn sum(&self, x: &[f32]) -> f32 {
                match self.1 .0 {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2 => {
                        crate::danger::f32_xconst_avx2_nofma_sum_horizontal::<{$dim::DIMS}>(x)
                    },
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2Fma => {
                        crate::danger::f32_xconst_avx2_nofma_sum_horizontal::<{$dim::DIMS}>(x)
                    },
                    #[cfg(all(
                        any(target_arch = "x86", target_arch = "x86_64"),
                        feature = "nightly"
                    ))]
                    SelectedArch::Avx512 => {
                        crate::danger::f32_xconst_avx512_nofma_sum_horizontal::<{$dim::DIMS}>(x)
                    },
                    SelectedArch::Fallback => {
                        crate::danger::generic_xany_fallback_nofma_sum_horizontal(x)
                    },
                }
            }

            #[inline]
            unsafe fn min(&self, x: &[f32]) -> f32 {
                match self.1 .0 {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2 => {
                        crate::danger::f32_xconst_avx2_nofma_min_horizontal::<{$dim::DIMS}>(x)
                    },
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2Fma => {
                        crate::danger::f32_xconst_avx2_nofma_min_horizontal::<{$dim::DIMS}>(x)
                    },
                    #[cfg(all(
                        any(target_arch = "x86", target_arch = "x86_64"),
                        feature = "nightly"
                    ))]
                    SelectedArch::Avx512 => {
                        crate::danger::f32_xconst_avx512_nofma_min_horizontal::<{$dim::DIMS}>(x)
                    },
                    SelectedArch::Fallback => {
                        crate::danger::f32_xany_fallback_nofma_min_horizontal(x)
                    },
                }
            }

            #[inline]
            unsafe fn max(&self, x: &[f32]) -> f32 {
                match self.1 .0 {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2 => {
                        crate::danger::f32_xconst_avx2_nofma_max_horizontal::<{$dim::DIMS}>(x)
                    },
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    SelectedArch::Avx2Fma => {
                        crate::danger::f32_xconst_avx2_nofma_max_horizontal::<{$dim::DIMS}>(x)
                    },
                    #[cfg(all(
                        any(target_arch = "x86", target_arch = "x86_64"),
                        feature = "nightly"
                    ))]
                    SelectedArch::Avx512 => {
                        crate::danger::f32_xconst_avx512_nofma_max_horizontal::<{$dim::DIMS}>(x)
                    },
                    SelectedArch::Fallback => {
                        crate::danger::generic_xany_fallback_nofma_max_horizontal(x)
                    },
                }
            }
        }
    };
}

impl_dangerous_auto_ops!(X1024);
impl_dangerous_auto_ops!(X768);
impl_dangerous_auto_ops!(X512);

impl DangerousOps for (XAny, Auto) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len(), "Lengths of `x` and `y` must be equal");
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_dot(x, y),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2Fma => crate::danger::f32_xany_avx2_fma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_xany_avx512_fma_dot(x, y),
            SelectedArch::Fallback => {
                crate::danger::generic_xany_fallback_nofma_dot(x, y)
            },
        }
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len(), "Lengths of `x` and `y` must be equal");
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_cosine(x, y),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2Fma => crate::danger::f32_xany_avx2_fma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_xany_avx512_fma_cosine(x, y),
            SelectedArch::Fallback => {
                crate::danger::generic_xany_fallback_nofma_cosine(x, y)
            },
        }
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len(), "Lengths of `x` and `y` must be equal");
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_euclidean(x, y),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2Fma => crate::danger::f32_xany_avx2_fma_euclidean(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_xany_avx512_fma_euclidean(x, y),
            SelectedArch::Fallback => {
                crate::danger::generic_xany_fallback_nofma_euclidean(x, y)
            },
        }
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), y.len(), "Lengths of `x` and `y` must be equal");
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xany_avx2_nofma_angular_hyperplane(x, y)
            },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xany_avx2_fma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xany_avx512_fma_angular_hyperplane(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_angular_hyperplane(x, y)
            },
        }
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        assert_eq!(x.len(), y.len(), "Lengths of `x` and `y` must be equal");
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xany_avx2_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xany_avx2_fma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xany_avx512_fma_euclidean_hyperplane(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_euclidean_hyperplane(x, y)
            },
        }
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_norm(x),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2Fma => crate::danger::f32_xany_avx2_fma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_xany_avx512_fma_norm(x),
            SelectedArch::Fallback => {
                crate::danger::generic_xany_fallback_nofma_dot(x, x)
            },
        }
    }

    #[inline]
    unsafe fn add_value(&self, x: &mut [f32], val: f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_add_value(x, val),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xany_avx2_nofma_add_value(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xany_avx512_nofma_add_value(x, val)
            },
            SelectedArch::Fallback => {
                crate::danger::generic_xany_fallback_nofma_add_value(x, val)
            },
        }
    }

    #[inline]
    unsafe fn sub_value(&self, x: &mut [f32], val: f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_sub_value(x, val),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xany_avx2_nofma_sub_value(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xany_avx512_nofma_sub_value(x, val)
            },
            SelectedArch::Fallback => {
                crate::danger::generic_xany_fallback_nofma_sub_value(x, val)
            },
        }
    }

    #[inline]
    unsafe fn mul_value(&self, x: &mut [f32], val: f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_mul_value(x, val),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xany_avx2_nofma_mul_value(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xany_avx512_nofma_mul_value(x, val)
            },
            SelectedArch::Fallback => {
                crate::danger::generic_xany_fallback_nofma_mul_value(x, val)
            },
        }
    }

    #[inline]
    unsafe fn div_value(&self, x: &mut [f32], val: f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_div_value(x, val),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xany_avx2_nofma_div_value(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xany_avx512_nofma_div_value(x, val)
            },
            SelectedArch::Fallback => {
                crate::danger::generic_xany_fallback_nofma_div_value(x, val)
            },
        }
    }

    #[inline]
    unsafe fn add_vertical(&self, x: &mut [f32], y: &[f32]) {
        assert_eq!(x.len(), y.len(), "Lengths of `x` and `y` must be equal");
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_add_vertical(x, y),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xany_avx2_nofma_add_vertical(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xany_avx512_nofma_add_vertical(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::generic_xany_fallback_nofma_add_vertical(x, y)
            },
        }
    }

    #[inline]
    unsafe fn sub_vertical(&self, x: &mut [f32], y: &[f32]) {
        assert_eq!(x.len(), y.len(), "Lengths of `x` and `y` must be equal");
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_sub_vertical(x, y),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xany_avx2_nofma_sub_vertical(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xany_avx512_nofma_sub_vertical(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::generic_xany_fallback_nofma_sub_vertical(x, y)
            },
        }
    }

    #[inline]
    unsafe fn mul_vertical(&self, x: &mut [f32], y: &[f32]) {
        assert_eq!(x.len(), y.len(), "Lengths of `x` and `y` must be equal");
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_mul_vertical(x, y),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xany_avx2_nofma_mul_vertical(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xany_avx512_nofma_mul_vertical(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::generic_xany_fallback_nofma_mul_vertical(x, y)
            },
        }
    }

    #[inline]
    unsafe fn div_vertical(&self, x: &mut [f32], y: &[f32]) {
        assert_eq!(x.len(), y.len(), "Lengths of `x` and `y` must be equal");
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_div_vertical(x, y),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xany_avx2_nofma_div_vertical(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xany_avx512_nofma_div_vertical(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::generic_xany_fallback_nofma_div_vertical(x, y)
            },
        }
    }

    #[inline]
    unsafe fn sum(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_sum_horizontal(x),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xany_avx2_nofma_sum_horizontal(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xany_avx512_nofma_sum_horizontal(x)
            },
            SelectedArch::Fallback => {
                crate::danger::generic_xany_fallback_nofma_sum_horizontal(x)
            },
        }
    }

    #[inline]
    unsafe fn min(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_min_horizontal(x),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xany_avx2_nofma_min_horizontal(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xany_avx512_nofma_min_horizontal(x)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_min_horizontal(x)
            },
        }
    }

    #[inline]
    unsafe fn max(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_max_horizontal(x),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xany_avx2_nofma_max_horizontal(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xany_avx512_nofma_max_horizontal(x)
            },
            SelectedArch::Fallback => {
                crate::danger::generic_xany_fallback_nofma_max_horizontal(x)
            },
        }
    }
}
