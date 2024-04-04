# igris-accel

Various specialised vector operations

### Last benchmark - Cosine
```
cosine autovec 1024 nofma
                        time:   [689.75 ns 702.80 ns 718.48 ns]
                        change: [+7.0021% +9.4335% +12.729%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 5 outliers among 100 measurements (5.00%)
  3 (3.00%) high mild
  2 (2.00%) high severe

cosine autovec 1024 fma time:   [864.81 ns 866.14 ns 867.65 ns]
                        change: [+1.5438% +2.0371% +2.4847%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 17 outliers among 100 measurements (17.00%)
  11 (11.00%) high mild
  6 (6.00%) high severe

cosine simsimd 1024 auto
                        time:   [661.44 ns 662.16 ns 662.86 ns]
                        change: [+5.2046% +5.5471% +5.8488%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 19 outliers among 100 measurements (19.00%)
  9 (9.00%) low severe
  6 (6.00%) low mild
  2 (2.00%) high mild
  2 (2.00%) high severe

cosine avx2 1024 nofma  time:   [99.945 ns 100.23 ns 100.58 ns]
                        change: [-8.9235% -8.5537% -8.1834%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 5 outliers among 100 measurements (5.00%)
  2 (2.00%) high mild
  3 (3.00%) high severe

cosine avx2 1024 fma    time:   [82.429 ns 82.634 ns 82.861 ns]
                        change: [-7.5036% -7.0584% -6.6422%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 19 outliers among 100 measurements (19.00%)
  8 (8.00%) high mild
  11 (11.00%) high severe
```

```
dot simsimd 1024 auto   time:   [627.81 ns 629.49 ns 631.31 ns]
                        change: [+0.5355% +0.8292% +1.1217%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 2 outliers among 100 measurements (2.00%)
  1 (1.00%) high mild
  1 (1.00%) high severe

dot fallback 1024 nofma time:   [635.51 ns 636.36 ns 637.32 ns]
                        change: [-0.5439% -0.2268% +0.0808%] (p = 0.16 > 0.05)
                        No change in performance detected.

dot fallback 1024 fma   time:   [842.32 ns 844.81 ns 847.63 ns]
                        change: [-1.4033% -0.9408% -0.4799%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 6 outliers among 100 measurements (6.00%)
  6 (6.00%) high mild

dot avx2 1024 nofma     time:   [42.192 ns 42.356 ns 42.611 ns]
                        change: [-0.8273% -0.4562% -0.0542%] (p = 0.02 < 0.05)
                        Change within noise threshold.
Found 2 outliers among 100 measurements (2.00%)
  1 (1.00%) high mild
  1 (1.00%) high severe

dot avx2 1024 fma       time:   [31.669 ns 31.703 ns 31.738 ns]
                        change: [-11.843% -11.681% -11.514%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 2 outliers among 100 measurements (2.00%)
  2 (2.00%) high mild

```