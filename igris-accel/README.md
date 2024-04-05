# igris-accel

Various specialised vector operations, primarily designed for similarity search.

This library requires nightly and is more unsafe than it is safe code, it is only intended to be used
within the other Igris tools, if you absolutely want to use this library directly, do so at your own risk.

### Benchmarks

#### Cosine
```
cosine autovec 1024 nofma
                        time:   [239.50 ns 240.24 ns 240.98 ns]
                        change: [-63.777% -63.552% -63.292%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 7 outliers among 100 measurements (7.00%)
  2 (2.00%) high mild
  5 (5.00%) high severe

cosine autovec 1024 fma time:   [389.57 ns 390.38 ns 391.26 ns]
                        change: [-54.889% -54.796% -54.701%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) high mild

cosine simsimd 1024 auto
                        time:   [659.42 ns 660.42 ns 661.84 ns]
                        change: [-1.0604% -0.9036% -0.7431%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 3 outliers among 100 measurements (3.00%)
  2 (2.00%) high mild
  1 (1.00%) high severe

cosine avx2 1024 nofma  time:   [86.290 ns 86.429 ns 86.596 ns]
                        change: [-14.741% -14.030% -13.379%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 2 outliers among 100 measurements (2.00%)
  2 (2.00%) high mild

cosine avx2 1024 fma    time:   [78.686 ns 78.802 ns 78.897 ns]
                        change: [-16.866% -16.458% -16.077%] (p = 0.00 < 0.05)
                        Performance has improved.
```

#### Dot Product
```
dot ndarray 1024 auto   time:   [45.937 ns 46.042 ns 46.149 ns]
                        change: [-1.0013% -0.7338% -0.4470%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 2 outliers among 100 measurements (2.00%)
  1 (1.00%) high mild
  1 (1.00%) high severe

dot simsimd 1024 auto   time:   [623.08 ns 624.04 ns 625.00 ns]
                        change: [-2.8595% -2.4275% -2.0105%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 3 outliers among 100 measurements (3.00%)
  3 (3.00%) high mild

dot fallback 1024 nofma time:   [224.78 ns 225.08 ns 225.38 ns]
                        change: [-1.0051% -0.8360% -0.6755%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) high mild

dot fallback 1024 fma   time:   [90.240 ns 90.461 ns 90.713 ns]
                        change: [-11.480% -10.868% -10.170%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 7 outliers among 100 measurements (7.00%)
  3 (3.00%) high mild
  4 (4.00%) high severe

dot avx2 1024 nofma     time:   [41.679 ns 41.739 ns 41.812 ns]
                        change: [-0.5080% +0.0217% +0.4952%] (p = 0.93 > 0.05)
                        No change in performance detected.
Found 8 outliers among 100 measurements (8.00%)
  7 (7.00%) high mild
  1 (1.00%) high severe

dot avx2 1024 fma       time:   [34.680 ns 34.715 ns 34.753 ns]
                        change: [-7.1005% -6.6157% -6.1079%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 16 outliers among 100 measurements (16.00%)
  10 (10.00%) low mild
  2 (2.00%) high mild
  4 (4.00%) high severe
```

#### Euclidean
```
euclidean autovec 1024 nofma
                        time:   [227.57 ns 227.90 ns 228.21 ns]
                        change: [-1.3105% -0.9597% -0.6517%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 2 outliers among 100 measurements (2.00%)
  1 (1.00%) high mild
  1 (1.00%) high severe

euclidean autovec 1024 fma
                        time:   [98.485 ns 98.667 ns 98.862 ns]
                        change: [+1.9323% +2.1488% +2.3498%] (p = 0.00 < 0.05)
                        Performance has regressed.

euclidean simsimd 1024 auto
                        time:   [625.81 ns 626.44 ns 627.10 ns]
                        change: [-3.4351% -3.1232% -2.8140%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 5 outliers among 100 measurements (5.00%)
  5 (5.00%) high mild

euclidean avx2 1024 nofma
                        time:   [42.271 ns 42.302 ns 42.334 ns]
                        change: [-0.1565% +0.0365% +0.2337%] (p = 0.72 > 0.05)
                        No change in performance detected.

euclidean avx2 1024 fma time:   [41.325 ns 41.439 ns 41.560 ns]
                        change: [-0.3190% -0.0238% +0.2737%] (p = 0.88 > 0.05)
                        No change in performance detected.
Found 11 outliers among 100 measurements (11.00%)
  7 (7.00%) high mild
  4 (4.00%) high severe
```