use super::DangerousOps;
use crate::arch::*;
use crate::dims::*;

impl DangerousOps for (X1024, Auto) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x1024_avx2_nofma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x1024_avx2_fma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x1024_avx512_nofma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_x1024_avx512_fma_dot(x, y),
            SelectedArch::Fallback => crate::danger::f32_xany_fallback_nofma_dot(x, y),
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => crate::danger::f32_xany_fallback_fma_dot(x, y),
        }
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x1024_avx2_nofma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x1024_avx2_fma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x1024_avx512_nofma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_x1024_avx512_fma_cosine(x, y),
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_cosine(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_cosine(x, y)
            },
        }
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x1024_avx2_nofma_euclidean(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x1024_avx2_fma_euclidean(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_x1024_avx512_nofma_euclidean(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_x1024_avx512_fma_euclidean(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_euclidean(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_euclidean(x, y)
            },
        }
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_x1024_avx2_nofma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_x1024_avx2_fma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_x1024_avx512_nofma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_x1024_avx512_fma_angular_hyperplane(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_angular_hyperplane(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_angular_hyperplane(x, y)
            },
        }
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_x1024_avx2_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_x1024_avx2_fma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_x1024_avx512_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_x1024_avx512_fma_euclidean_hyperplane(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_euclidean_hyperplane(x, y)
            },
        }
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x1024_avx2_nofma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x1024_avx2_fma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x1024_avx512_nofma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_x1024_avx512_fma_norm(x),
            SelectedArch::Fallback => crate::danger::f32_xany_fallback_nofma_dot(x, x),
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => crate::danger::f32_xany_fallback_fma_dot(x, x),
        }
    }

    #[inline]
    unsafe fn add_value(&self, x: &mut [f32], val: f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_add_value::<1024>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_add_value::<1024>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_add_value::<1024>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_add_value::<1024>(x, val)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_add_value(x, val)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_add_value(x, val)
            },
        }
    }

    #[inline]
    unsafe fn sub_value(&self, x: &mut [f32], val: f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_sub_value::<1024>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_sub_value::<1024>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_sub_value::<1024>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_sub_value::<1024>(x, val)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_sub_value(x, val)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_sub_value(x, val)
            },
        }
    }

    #[inline]
    unsafe fn mul_value(&self, x: &mut [f32], val: f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_mul_value::<1024>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_mul_value::<1024>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_mul_value::<1024>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_mul_value::<1024>(x, val)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_mul_value(x, val)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_mul_value(x, val)
            },
        }
    }

    #[inline]
    unsafe fn div_value(&self, x: &mut [f32], val: f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_div_value::<1024>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_div_value::<1024>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_div_value::<1024>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_div_value::<1024>(x, val)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_div_value(x, val)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_div_value(x, val)
            },
        }
    }

    #[inline]
    unsafe fn add_vertical(&self, x: &mut [f32], y: &[f32]) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_add_vertical::<1024>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_add_vertical::<1024>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_add_vertical::<1024>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_add_vertical::<1024>(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_add_vertical(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_add_vertical(x, y)
            },
        }
    }

    #[inline]
    unsafe fn sub_vertical(&self, x: &mut [f32], y: &[f32]) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_sub_vertical::<1024>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_sub_vertical::<1024>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_sub_vertical::<1024>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_sub_vertical::<1024>(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_sub_vertical(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_sub_vertical(x, y)
            },
        }
    }

    #[inline]
    unsafe fn mul_vertical(&self, x: &mut [f32], y: &[f32]) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_mul_vertical::<1024>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_mul_vertical::<1024>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_mul_vertical::<1024>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_mul_vertical::<1024>(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_mul_vertical(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_mul_vertical(x, y)
            },
        }
    }

    #[inline]
    unsafe fn div_vertical(&self, x: &mut [f32], y: &[f32]) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_div_vertical::<1024>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_div_vertical::<1024>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_div_vertical::<1024>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_div_vertical::<1024>(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_div_vertical(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_div_vertical(x, y)
            },
        }
    }

    #[inline]
    unsafe fn sum(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_sum_horizontal::<1024>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_sum_horizontal::<1024>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_sum_horizontal::<1024>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_sum_horizontal::<1024>(x)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_sum_horizontal(x)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_sum_horizontal(x)
            },
        }
    }

    #[inline]
    unsafe fn min(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_min_horizontal::<1024>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_min_horizontal::<1024>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_min_horizontal::<1024>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_min_horizontal::<1024>(x)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_min_horizontal(x)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_nofma_min_horizontal(x)
            },
        }
    }

    #[inline]
    unsafe fn max(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_max_horizontal::<1024>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_max_horizontal::<1024>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_max_horizontal::<1024>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_max_horizontal::<1024>(x)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_max_horizontal(x)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_nofma_max_horizontal(x)
            },
        }
    }
}

impl DangerousOps for (X768, Auto) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x768_avx2_nofma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x768_avx2_fma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x768_avx512_nofma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_x768_avx512_fma_dot(x, y),
            SelectedArch::Fallback => crate::danger::f32_xany_fallback_nofma_dot(x, y),
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => crate::danger::f32_xany_fallback_fma_dot(x, y),
        }
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x768_avx2_nofma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x768_avx2_fma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x768_avx512_nofma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_x768_avx512_fma_cosine(x, y),
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_cosine(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_cosine(x, y)
            },
        }
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x768_avx2_nofma_euclidean(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x768_avx2_fma_euclidean(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x768_avx512_nofma_euclidean(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_x768_avx512_fma_euclidean(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_euclidean(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_euclidean(x, y)
            },
        }
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_x768_avx2_nofma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_x768_avx2_fma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_x768_avx512_nofma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_x768_avx512_fma_angular_hyperplane(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_angular_hyperplane(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_angular_hyperplane(x, y)
            },
        }
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_x768_avx2_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_x768_avx2_fma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_x768_avx512_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_x768_avx512_fma_euclidean_hyperplane(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_euclidean_hyperplane(x, y)
            },
        }
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x768_avx2_nofma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x768_avx2_fma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x768_avx512_nofma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_x768_avx512_fma_norm(x),
            SelectedArch::Fallback => crate::danger::f32_xany_fallback_nofma_dot(x, x),
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => crate::danger::f32_xany_fallback_fma_dot(x, x),
        }
    }

    #[inline]
    unsafe fn add_value(&self, x: &mut [f32], val: f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_add_value::<768>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_add_value::<768>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_add_value::<768>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_add_value::<768>(x, val)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_add_value(x, val)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_add_value(x, val)
            },
        }
    }

    #[inline]
    unsafe fn sub_value(&self, x: &mut [f32], val: f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_sub_value::<768>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_sub_value::<768>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_sub_value::<768>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_sub_value::<768>(x, val)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_sub_value(x, val)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_sub_value(x, val)
            },
        }
    }

    #[inline]
    unsafe fn mul_value(&self, x: &mut [f32], val: f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_mul_value::<768>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_mul_value::<768>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_mul_value::<768>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_mul_value::<768>(x, val)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_mul_value(x, val)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_mul_value(x, val)
            },
        }
    }

    #[inline]
    unsafe fn div_value(&self, x: &mut [f32], val: f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_div_value::<768>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_div_value::<768>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_div_value::<768>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_div_value::<768>(x, val)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_div_value(x, val)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_div_value(x, val)
            },
        }
    }

    #[inline]
    unsafe fn add_vertical(&self, x: &mut [f32], y: &[f32]) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_add_vertical::<768>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_add_vertical::<768>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_add_vertical::<768>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_add_vertical::<768>(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_add_vertical(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_add_vertical(x, y)
            },
        }
    }

    #[inline]
    unsafe fn sub_vertical(&self, x: &mut [f32], y: &[f32]) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_sub_vertical::<768>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_sub_vertical::<768>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_sub_vertical::<768>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_sub_vertical::<768>(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_sub_vertical(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_sub_vertical(x, y)
            },
        }
    }

    #[inline]
    unsafe fn mul_vertical(&self, x: &mut [f32], y: &[f32]) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_mul_vertical::<768>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_mul_vertical::<768>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_mul_vertical::<768>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_mul_vertical::<768>(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_mul_vertical(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_mul_vertical(x, y)
            },
        }
    }

    #[inline]
    unsafe fn div_vertical(&self, x: &mut [f32], y: &[f32]) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_div_vertical::<768>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_div_vertical::<768>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_div_vertical::<768>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_div_vertical::<768>(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_div_vertical(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_div_vertical(x, y)
            },
        }
    }

    #[inline]
    unsafe fn sum(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_sum_horizontal::<768>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_sum_horizontal::<768>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_sum_horizontal::<768>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_sum_horizontal::<768>(x)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_sum_horizontal(x)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_sum_horizontal(x)
            },
        }
    }

    #[inline]
    unsafe fn min(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_min_horizontal::<768>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_min_horizontal::<768>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_min_horizontal::<768>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_min_horizontal::<768>(x)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_min_horizontal(x)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_nofma_min_horizontal(x)
            },
        }
    }

    #[inline]
    unsafe fn max(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_max_horizontal::<768>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_max_horizontal::<768>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_max_horizontal::<768>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_max_horizontal::<768>(x)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_max_horizontal(x)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_nofma_max_horizontal(x)
            },
        }
    }
}

impl DangerousOps for (X512, Auto) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x512_avx2_nofma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x512_avx2_fma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x512_avx512_nofma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_x512_avx512_fma_dot(x, y),
            SelectedArch::Fallback => crate::danger::f32_xany_fallback_nofma_dot(x, y),
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => crate::danger::f32_xany_fallback_fma_dot(x, y),
        }
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x512_avx2_nofma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x512_avx2_fma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x512_avx512_nofma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_x512_avx512_fma_cosine(x, y),
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_cosine(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_cosine(x, y)
            },
        }
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x512_avx2_nofma_euclidean(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x512_avx2_fma_euclidean(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x512_avx512_nofma_euclidean(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_x512_avx512_fma_euclidean(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_euclidean(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_euclidean(x, y)
            },
        }
    }

    #[inline]
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_x512_avx2_nofma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_x512_avx2_fma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_x512_avx512_nofma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_x512_avx512_fma_angular_hyperplane(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_angular_hyperplane(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_angular_hyperplane(x, y)
            },
        }
    }

    #[inline]
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_x512_avx2_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_x512_avx2_fma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_x512_avx512_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_x512_avx512_fma_euclidean_hyperplane(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_euclidean_hyperplane(x, y)
            },
        }
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_x512_avx2_nofma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_x512_avx2_fma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_x512_avx512_nofma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_x512_avx512_fma_norm(x),
            SelectedArch::Fallback => crate::danger::f32_xany_fallback_nofma_dot(x, x),
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => crate::danger::f32_xany_fallback_fma_dot(x, x),
        }
    }

    #[inline]
    unsafe fn add_value(&self, x: &mut [f32], val: f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_add_value::<512>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_add_value::<512>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_add_value::<512>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_add_value::<512>(x, val)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_add_value(x, val)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_add_value(x, val)
            },
        }
    }

    #[inline]
    unsafe fn sub_value(&self, x: &mut [f32], val: f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_sub_value::<512>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_sub_value::<512>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_sub_value::<512>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_sub_value::<512>(x, val)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_sub_value(x, val)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_sub_value(x, val)
            },
        }
    }

    #[inline]
    unsafe fn mul_value(&self, x: &mut [f32], val: f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_mul_value::<512>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_mul_value::<512>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_mul_value::<512>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_mul_value::<512>(x, val)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_mul_value(x, val)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_mul_value(x, val)
            },
        }
    }

    #[inline]
    unsafe fn div_value(&self, x: &mut [f32], val: f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_div_value::<512>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_div_value::<512>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_div_value::<512>(x, val)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_div_value::<512>(x, val)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_div_value(x, val)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_div_value(x, val)
            },
        }
    }

    #[inline]
    unsafe fn add_vertical(&self, x: &mut [f32], y: &[f32]) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_add_vertical::<512>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_add_vertical::<512>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_add_vertical::<512>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_add_vertical::<512>(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_add_vertical(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_add_vertical(x, y)
            },
        }
    }

    #[inline]
    unsafe fn sub_vertical(&self, x: &mut [f32], y: &[f32]) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_sub_vertical::<512>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_sub_vertical::<512>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_sub_vertical::<512>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_sub_vertical::<512>(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_sub_vertical(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_sub_vertical(x, y)
            },
        }
    }

    #[inline]
    unsafe fn mul_vertical(&self, x: &mut [f32], y: &[f32]) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_mul_vertical::<512>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_mul_vertical::<512>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_mul_vertical::<512>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_mul_vertical::<512>(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_mul_vertical(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_mul_vertical(x, y)
            },
        }
    }

    #[inline]
    unsafe fn div_vertical(&self, x: &mut [f32], y: &[f32]) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_div_vertical::<512>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_div_vertical::<512>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_div_vertical::<512>(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_div_vertical::<512>(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_div_vertical(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_div_vertical(x, y)
            },
        }
    }

    #[inline]
    unsafe fn sum(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_sum_horizontal::<512>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_sum_horizontal::<512>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_sum_horizontal::<512>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_sum_horizontal::<512>(x)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_sum_horizontal(x)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_sum_horizontal(x)
            },
        }
    }

    #[inline]
    unsafe fn min(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_min_horizontal::<512>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_min_horizontal::<512>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_min_horizontal::<512>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_min_horizontal::<512>(x)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_min_horizontal(x)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_nofma_min_horizontal(x)
            },
        }
    }

    #[inline]
    unsafe fn max(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => {
                crate::danger::f32_xconst_avx2_nofma_max_horizontal::<512>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xconst_avx2_nofma_max_horizontal::<512>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xconst_avx512_nofma_max_horizontal::<512>(x)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xconst_avx512_nofma_max_horizontal::<512>(x)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_max_horizontal(x)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_nofma_max_horizontal(x)
            },
        }
    }
}

impl DangerousOps for (XAny, Auto) {
    #[inline]
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len(), "Lengths of `x` and `y` must be equal");
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_xany_avx2_fma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_xany_avx512_nofma_dot(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_xany_avx512_fma_dot(x, y),
            SelectedArch::Fallback => crate::danger::f32_xany_fallback_nofma_dot(x, y),
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => crate::danger::f32_xany_fallback_fma_dot(x, y),
        }
    }

    #[inline]
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len(), "Lengths of `x` and `y` must be equal");
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_xany_avx2_fma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_xany_avx512_nofma_cosine(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_xany_avx512_fma_cosine(x, y),
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_cosine(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_cosine(x, y)
            },
        }
    }

    #[inline]
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len(), "Lengths of `x` and `y` must be equal");
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_euclidean(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_xany_avx2_fma_euclidean(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_xany_avx512_nofma_euclidean(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xany_avx512_fma_euclidean(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_euclidean(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_euclidean(x, y)
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
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xany_avx2_fma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xany_avx512_nofma_angular_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xany_avx512_fma_angular_hyperplane(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_angular_hyperplane(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_angular_hyperplane(x, y)
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
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => {
                crate::danger::f32_xany_avx2_fma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => {
                crate::danger::f32_xany_avx512_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xany_avx512_fma_euclidean_hyperplane(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_euclidean_hyperplane(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_euclidean_hyperplane(x, y)
            },
        }
    }

    #[inline]
    unsafe fn squared_norm(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx2Fma => crate::danger::f32_xany_avx2_fma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512 => crate::danger::f32_xany_avx512_nofma_norm(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => crate::danger::f32_xany_avx512_fma_norm(x),
            SelectedArch::Fallback => crate::danger::f32_xany_fallback_nofma_dot(x, x),
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => crate::danger::f32_xany_fallback_fma_dot(x, x),
        }
    }

    #[inline]
    unsafe fn add_value(&self, x: &mut [f32], val: f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_add_value(x, val),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
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
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xany_avx512_nofma_add_value(x, val)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_add_value(x, val)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_nofma_add_value(x, val)
            },
        }
    }

    #[inline]
    unsafe fn sub_value(&self, x: &mut [f32], val: f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_sub_value(x, val),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
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
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xany_avx512_nofma_sub_value(x, val)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_sub_value(x, val)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_nofma_sub_value(x, val)
            },
        }
    }

    #[inline]
    unsafe fn mul_value(&self, x: &mut [f32], val: f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_mul_value(x, val),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
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
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xany_avx512_nofma_mul_value(x, val)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_mul_value(x, val)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_nofma_mul_value(x, val)
            },
        }
    }

    #[inline]
    unsafe fn div_value(&self, x: &mut [f32], val: f32) {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_div_value(x, val),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
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
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xany_avx512_nofma_div_value(x, val)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_div_value(x, val)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_div_value(x, val)
            },
        }
    }

    #[inline]
    unsafe fn add_vertical(&self, x: &mut [f32], y: &[f32]) {
        assert_eq!(x.len(), y.len(), "Lengths of `x` and `y` must be equal");
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_add_vertical(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
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
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xany_avx512_nofma_add_vertical(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_add_vertical(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_add_vertical(x, y)
            },
        }
    }

    #[inline]
    unsafe fn sub_vertical(&self, x: &mut [f32], y: &[f32]) {
        assert_eq!(x.len(), y.len(), "Lengths of `x` and `y` must be equal");
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_sub_vertical(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
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
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xany_avx512_nofma_sub_vertical(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_sub_vertical(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_sub_vertical(x, y)
            },
        }
    }

    #[inline]
    unsafe fn mul_vertical(&self, x: &mut [f32], y: &[f32]) {
        assert_eq!(x.len(), y.len(), "Lengths of `x` and `y` must be equal");
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_mul_vertical(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
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
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xany_avx512_nofma_mul_vertical(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_mul_vertical(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_mul_vertical(x, y)
            },
        }
    }

    #[inline]
    unsafe fn div_vertical(&self, x: &mut [f32], y: &[f32]) {
        assert_eq!(x.len(), y.len(), "Lengths of `x` and `y` must be equal");
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_div_vertical(x, y),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
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
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xany_avx512_nofma_div_vertical(x, y)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_div_vertical(x, y)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_div_vertical(x, y)
            },
        }
    }

    #[inline]
    unsafe fn sum(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_sum_horizontal(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
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
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xany_avx512_nofma_sum_horizontal(x)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_sum_horizontal(x)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_fma_sum_horizontal(x)
            },
        }
    }

    #[inline]
    unsafe fn min(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_min_horizontal(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
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
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xany_avx512_nofma_min_horizontal(x)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_min_horizontal(x)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_nofma_min_horizontal(x)
            },
        }
    }

    #[inline]
    unsafe fn max(&self, x: &[f32]) -> f32 {
        match self.1 .0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SelectedArch::Avx2 => crate::danger::f32_xany_avx2_nofma_max_horizontal(x),
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
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
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly"
            ))]
            SelectedArch::Avx512Fma => {
                crate::danger::f32_xany_avx512_nofma_max_horizontal(x)
            },
            SelectedArch::Fallback => {
                crate::danger::f32_xany_fallback_nofma_max_horizontal(x)
            },
            #[cfg(feature = "nightly")]
            SelectedArch::FallbackFma => {
                crate::danger::f32_xany_fallback_nofma_max_horizontal(x)
            },
        }
    }
}
