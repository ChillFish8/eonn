#[no_mangle]
#[inline(never)]
pub unsafe fn x1024_avx2_fma_angular_hyperplane(x: &[f32], y: &[f32]) -> Vec<f32> {
    eonn_accel::danger::f32_x1024_avx2_fma_angular_hyperplane(x, y)
}

#[no_mangle]
#[inline(never)]
pub unsafe fn x1024_avx512_fma_angular_hyperplane(x: &[f32], y: &[f32]) -> Vec<f32> {
    eonn_accel::danger::f32_x1024_avx512_fma_angular_hyperplane(x, y)
}

#[no_mangle]
#[inline(never)]
pub unsafe fn x1024_avx2_fma_cosine(x: &[f32], y: &[f32]) -> f32 {
    eonn_accel::danger::f32_x1024_avx2_fma_cosine(x, y)
}

#[no_mangle]
#[inline(never)]
pub unsafe fn x1024_avx512_fma_cosine(x: &[f32], y: &[f32]) -> f32 {
    eonn_accel::danger::f32_x1024_avx512_fma_cosine(x, y)
}

#[no_mangle]
#[inline(never)]
pub unsafe fn x1024_avx2_fma_dot(x: &[f32], y: &[f32]) -> f32 {
    eonn_accel::danger::f32_x1024_avx2_fma_dot(x, y)
}

#[no_mangle]
#[inline(never)]
pub unsafe fn x1024_avx512_fma_dot(x: &[f32], y: &[f32]) -> f32 {
    eonn_accel::danger::f32_x1024_avx512_fma_dot(x, y)
}

#[no_mangle]
#[inline(never)]
pub unsafe fn x1024_avx2_nofma_dot(x: &[f32], y: &[f32]) -> f32 {
    eonn_accel::danger::f32_x1024_avx2_nofma_dot(x, y)
}
