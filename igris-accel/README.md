# igris-accel

Various specialised vector operations

### Benchmarks

#### Cosine
```
cosine autovec 1024 nofma
                        time:   [664.06 ns 665.48 ns 667.22 ns]
                        change: [+1.6391% +2.2367% +3.1483%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 9 outliers among 100 measurements (9.00%)
  6 (6.00%) high mild
  3 (3.00%) high severe

cosine autovec 1024 fma time:   [858.76 ns 859.38 ns 860.08 ns]
                        change: [+0.3664% +0.5924% +0.7998%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 8 outliers among 100 measurements (8.00%)
  1 (1.00%) low mild
  3 (3.00%) high mild
  4 (4.00%) high severe

cosine simsimd 1024 auto
                        time:   [658.23 ns 658.90 ns 659.66 ns]
                        change: [-1.9375% -1.6116% -1.3082%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 11 outliers among 100 measurements (11.00%)
  7 (7.00%) high mild
  4 (4.00%) high severe

cosine avx2 1024 nofma  time:   [86.846 ns 86.907 ns 86.974 ns]
                        change: [-17.776% -17.418% -17.100%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 4 outliers among 100 measurements (4.00%)
  2 (2.00%) low mild
  1 (1.00%) high mild
  1 (1.00%) high severe

cosine avx2 1024 fma    time:   [78.376 ns 78.458 ns 78.551 ns]
                        change: [-11.037% -10.928% -10.816%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 4 outliers among 100 measurements (4.00%)
  1 (1.00%) high mild
  3 (3.00%) high severe
```

#### Dot Product
```
dot ndarray 1024 auto   time:   [45.813 ns 45.839 ns 45.868 ns]
                        change: [-0.9838% -0.7848% -0.6020%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 3 outliers among 100 measurements (3.00%)
  2 (2.00%) high mild
  1 (1.00%) high severe

dot simsimd 1024 auto   time:   [623.84 ns 624.95 ns 626.19 ns]
                        change: [-0.1856% +0.0046% +0.1989%] (p = 0.96 > 0.05)
                        No change in performance detected.
Found 5 outliers among 100 measurements (5.00%)
  5 (5.00%) high mild

dot fallback 1024 nofma time:   [634.03 ns 634.56 ns 635.20 ns]
                        change: [-1.9555% -1.7268% -1.4988%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 12 outliers among 100 measurements (12.00%)
  9 (9.00%) high mild
  3 (3.00%) high severe

dot fallback 1024 fma   time:   [838.73 ns 839.32 ns 840.01 ns]
                        change: [-0.9247% -0.6865% -0.4746%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 13 outliers among 100 measurements (13.00%)
  7 (7.00%) high mild
  6 (6.00%) high severe

dot avx2 1024 nofma     time:   [41.393 ns 41.415 ns 41.438 ns]
                        change: [+0.3461% +0.4237% +0.4978%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 4 outliers among 100 measurements (4.00%)
  3 (3.00%) high mild
  1 (1.00%) high severe

dot avx2 1024 fma       time:   [36.822 ns 36.855 ns 36.892 ns]
                        change: [+5.2839% +5.3934% +5.5023%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 4 outliers among 100 measurements (4.00%)
  4 (4.00%) high mild
```

#### Euclidean
```
euclidean autovec 1024 nofma
                        time:   [641.46 ns 642.78 ns 644.09 ns]
                        change: [+0.2768% +0.4417% +0.5957%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 9 outliers among 100 measurements (9.00%)
  6 (6.00%) high mild
  3 (3.00%) high severe

euclidean autovec 1024 fma
                        time:   [847.76 ns 849.99 ns 852.76 ns]
                        change: [+0.9907% +1.2895% +1.5777%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 3 outliers among 100 measurements (3.00%)
  3 (3.00%) high mild

euclidean simsimd 1024 auto
                        time:   [643.85 ns 645.70 ns 647.49 ns]
                        change: [+1.9857% +2.2779% +2.5745%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) high mild

euclidean avx2 1024 nofma
                        time:   [42.749 ns 43.011 ns 43.292 ns]
                        change: [+1.5659% +2.0110% +2.4906%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 10 outliers among 100 measurements (10.00%)
  6 (6.00%) high mild
  4 (4.00%) high severe

euclidean avx2 1024 fma time:   [41.554 ns 41.702 ns 41.862 ns]
                        change: [+1.6153% +1.8232% +2.0577%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 9 outliers among 100 measurements (9.00%)
  1 (1.00%) low mild
  6 (6.00%) high mild
  2 (2.00%) high severe
```