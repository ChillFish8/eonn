use igris_accel::danger::f32_x1024_avx2_fma_angular_hyperplane;

#[cfg(all(target_feature = "avx2", target_feature = "fma"))]
#[no_mangle]
#[inline(never)]
pub unsafe fn x1024_fma_angular_hyperplane(x: &[f32], y: &[f32]) -> Vec<f32> {
    f32_x1024_avx2_fma_angular_hyperplane(x, y)
}
