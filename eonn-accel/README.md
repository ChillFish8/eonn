# eonn-accel

Various specialised vector operations, primarily designed for similarity search.

This system has several specializations for various CPU features and vector dimensions,
currently only `f32` vectors with dimensions of  any size are supported, but with 
specialized `512`, `768` or `1024` dimension implementations.

Supported CPU features include `Avx512`, `Avx2` and `Fma`, fallback implementations can
be optimized relatively well by the compiler for other architectures e.g. ARM or SSE.

### Supported Operations & Distances

- `vector.dot(other)`
- `vector.dist_dot(other)`
- `vector.dist_cosine(other)`
- `vector.dist_squared_euclidean(other)`
- `vector.angular_hyperplane(other)`
- `vector.euclidean_hyperplane(other)`
- `vector.squared_norm()`
- `vector / value`
- `vector * value`
- `vector + value`
- `vector - value`
- `vector.sum()`
- `vector.max()`
- `vector.min()`
- `[vector].vertical_min()`
- `[vector].vertical_max()`

### Dangerous routine naming convention

If you've looked at the `danger` folder at all, you'll notice all functions implement a certain
naming scheme to indicate their specialization.

```
<dtype>_x<dims>_<arch>_<(no)fma>_<op_name>
```

#### Notes on what `nofma` and `fma` mean

`FMA` in this system indicated _both_ the `fma` CPU feature flag is available _and_ to use `fast-math`
intrinsics in the compiler.

Note that the fallback implementations will only use the intrinsics and will let the compiler
optimize for those intrinsics, however, it is likely that you do not want to use `_fma_op` 
type functions on any platform that does not support the `fma` CPU feature except maybe for
ARM, but this is untested.

### Features

- `dangerous-access` Exposes access to the unsafe specialized functions, it is entirely on you to 
  ensure the data passed to these functions are correct and safe to call. USE AT YOUR OWN RISK.
- `nightly` Enables optimizations available only on nightly platforms, for best performance
  I recommend using this feature.
  * This is required for FMA due to their compiler intrinsics calls.
  * This is required for AVX512 support due to it currently being unstable.

