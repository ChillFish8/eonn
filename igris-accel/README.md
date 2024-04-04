# igris-accel

Various specialised vector operations

### Last benchmark
```
dot autovec 1024 auto   time:   [837.56 ns 839.31 ns 841.29 ns]
                        change: [-1.9690% -0.8170% +0.5258%] (p = 0.23 > 0.05)
                        No change in performance detected.
Found 2 outliers among 100 measurements (2.00%)
  1 (1.00%) high mild
  1 (1.00%) high severe

dot simsimd 1024 auto   time:   [656.66 ns 658.71 ns 661.15 ns]
                        change: [+4.3836% +4.8913% +5.4190%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 10 outliers among 100 measurements (10.00%)
  7 (7.00%) high mild
  3 (3.00%) high severe

dot avx2 1024 nofma     time:   [39.073 ns 40.140 ns 41.170 ns]
                        change: [-3.5564% -1.4109% +0.6167%] (p = 0.22 > 0.05)
                        No change in performance detected.
Found 35 outliers among 100 measurements (35.00%)
  17 (17.00%) low severe
  18 (18.00%) high severe

dot avx2 1024 fma       time:   [37.024 ns 37.969 ns 38.924 ns]
                        change: [-4.3185% -1.8048% +0.6135%] (p = 0.15 > 0.05)
                        No change in performance detected.
Found 32 outliers among 100 measurements (32.00%)
  12 (12.00%) low mild
  18 (18.00%) high mild
  2 (2.00%) high severe
```