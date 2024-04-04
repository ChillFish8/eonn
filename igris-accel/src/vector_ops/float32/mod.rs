#[cfg(any(
    feature = "bypass-arch-flags",
    all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "avx2",
    )
))]
mod avx2;
pub(super) mod fallback;
