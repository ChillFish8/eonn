#[cfg(any(
    feature = "bypass-arch-flags",
    all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "avx2",
    )
))]
/// AVX2 enabled architectures.
pub struct Avx2;
#[cfg(any(
    feature = "bypass-arch-flags",
    all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "avx512",
    )
))]
/// AVX512 enabled architectures.
pub struct Avx512;
/// No specialised features detected, fallback impls.
pub struct Fallback;
#[cfg(any(
    feature = "bypass-arch-flags",
    all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "fma",
    )
))]
/// Enables FMA instructions
pub struct Fma;
/// Disables FMA instructions
pub struct NoFma;
