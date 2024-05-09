#![allow(clippy::missing_safety_doc)]

#[no_mangle]
#[inline(never)]
pub unsafe fn xany_avx2_nofma_dot(x: &[f32], y: &[f32]) -> f32 {
    eonn_accel::danger::f32_xany_avx2_nofma_dot(x, y)
}

#[no_mangle]
#[inline(never)]
pub unsafe fn xany_avx2_fma_dot(x: &[f32], y: &[f32]) -> f32 {
    eonn_accel::danger::f32_xany_avx2_fma_dot(x, y)
}

#[no_mangle]
#[inline(never)]
pub unsafe fn xconst_avx2_fma_dot(x: &[f32], y: &[f32]) -> f32 {
    eonn_accel::danger::f32_xconst_avx2_fma_dot::<1024>(x, y)
}

#[no_mangle]
#[inline(never)]
pub unsafe fn xconst_avx2_nofma_dot(x: &[f32], y: &[f32]) -> f32 {
    eonn_accel::danger::f32_xconst_avx2_nofma_dot::<1024>(x, y)
}

#[no_mangle]
#[inline(never)]
pub unsafe fn xany_avx2_nofma_sum(x: &[f32]) -> f32 {
    eonn_accel::danger::f32_xany_avx2_nofma_sum_horizontal(x)
}
