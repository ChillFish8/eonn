# igris-accel

Various specialised vector operations

### Last benchmark
```
Timer precision: 10 ns
bench_dot_product             fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ bench_avx2_x1024_fma       39.78 ns      │ 40.09 ns      │ 40.09 ns      │ 40.05 ns      │ 100     │ 3200
├─ bench_avx2_x1024_nofma     41.03 ns      │ 50.09 ns      │ 41.34 ns      │ 41.66 ns      │ 100     │ 3200
├─ bench_llvmauto_x1024_auto  869.7 ns      │ 5.564 µs      │ 874.7 ns      │ 924.1 ns      │ 100     │ 200
╰─ bench_simsimd_x1024_auto   678.7 ns      │ 3.659 µs      │ 689.7 ns      │ 720.4 ns      │ 100     │ 100
```