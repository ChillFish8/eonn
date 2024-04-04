/// AVX2 enabled architectures.
pub struct Avx2;
/// AVX512 enabled architectures.
pub struct Avx512;
/// No specialised features detected, fallback impls.
pub struct Fallback;
/// Enables FMA instructions
pub struct Fma;
/// Disables FMA instructions
pub struct NoFma;
