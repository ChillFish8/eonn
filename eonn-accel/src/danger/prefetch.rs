#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
/// Prefetch the next `64` elements in the block.
///
/// This routine assumes data is aligned to `64` bytes so it fits consistently
/// within the cache line.
pub(crate) unsafe fn f32_prefetch_x64(ptr: *const f32) {
    use std::arch::x86_64::*;

    _mm_prefetch::<_MM_HINT_T1>(ptr as _);
    _mm_prefetch::<_MM_HINT_T1>(ptr.add(16) as _); // 16 elements to a 64 byte boundary
    _mm_prefetch::<_MM_HINT_T1>(ptr.add(32) as _);
    _mm_prefetch::<_MM_HINT_T1>(ptr.add(48) as _);
}
