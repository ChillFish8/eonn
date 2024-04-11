# igris-accel

Various specialised vector operations, primarily designed for similarity search.

This system has several specializations for various CPU features and vector dimensions,
currently only `f32` vectors with dimensions of `512`, `768` or `1024` are supported.

Supported CPU features include `Avx512`, `Avx2` and `Fma`, fallback implementations can
be optimized relatively well by the compiler for other architectures e.g. ARM or SSE.

### Supported Operations & Distances

- Dot product
- Cosine 
  * _NOTE: The current cosine implementation is significantly slower than dot product or Euclidean
    and could still do with some more niche optimizations, that being said it is still faster than a BLAS
    configuration on my machine, but if you are doing cosine heavy work, 
    [simsimd](https://github.com/ashvardanian/SimSIMD) may be a better choice._
- Squared Euclidean
- Angular Hyperplanes
- Euclidean Hyperplanes
- Squared Norm

### Features

- `dangerous-access` Exposes access to the unsafe specialized functions, it is entirely on you to 
  ensure the data passed to these functions are correct and safe to call. USE AT YOUR OWN RISK.
- `nightly` Enables optimizations available only on nightly platforms, for best performance
  I recommend using this feature.
  * This is required for FMA due to their compiler intrinsics calls.
  * This is required for AVX512 support due to it currently being unstable.

