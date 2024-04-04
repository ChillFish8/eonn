# igris-accel

Various specialised vector operations

### Benchmarks

#### Cosine
```
cosine autovec 1024 nofma
                        time:   [654.31 ns 655.00 ns 655.81 ns]
                        change: [-1.1757% -0.9772% -0.7994%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 7 outliers among 100 measurements (7.00%)
  4 (4.00%) high mild
  3 (3.00%) high severe

cosine autovec 1024 fma time:   [859.07 ns 860.87 ns 862.83 ns]
                        change: [-5.4110% -3.1033% -1.2530%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 3 outliers among 100 measurements (3.00%)
  2 (2.00%) high mild
  1 (1.00%) high severe

cosine simsimd 1024 auto
                        time:   [666.04 ns 667.75 ns 669.64 ns]
                        change: [+0.4133% +0.6906% +0.9642%] (p = 0.00 < 0.05)
                        Change within noise threshold.

cosine avx2 1024 nofma  time:   [98.541 ns 98.839 ns 99.136 ns]
                        change: [+0.0274% +0.2303% +0.4521%] (p = 0.03 < 0.05)
                        Change within noise threshold.
Found 5 outliers among 100 measurements (5.00%)
  4 (4.00%) high mild
  1 (1.00%) high severe

cosine avx2 1024 fma    time:   [88.000 ns 88.179 ns 88.417 ns]
                        change: [-6.6581% -6.4573% -6.2083%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 8 outliers among 100 measurements (8.00%)
  5 (5.00%) high mild
  3 (3.00%) high severe
```

#### Dot Product
```
dot simsimd 1024 auto   time:   [622.74 ns 624.67 ns 626.63 ns]
                        change: [-0.2335% -0.0002% +0.2353%] (p = 1.00 > 0.05)
                        No change in performance detected.

dot fallback 1024 nofma time:   [640.86 ns 643.04 ns 645.18 ns]
                        change: [+0.7937% +1.0353% +1.2542%] (p = 0.00 < 0.05)
                        Change within noise threshold.

dot fallback 1024 fma   time:   [840.15 ns 841.00 ns 842.00 ns]
                        change: [-0.0655% +0.1866% +0.4248%] (p = 0.13 > 0.05)
                        No change in performance detected.

dot avx2 1024 nofma     time:   [41.678 ns 41.783 ns 41.896 ns]
                        change: [+0.8655% +0.9965% +1.1517%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 5 outliers among 100 measurements (5.00%)
  3 (3.00%) high mild
  2 (2.00%) high severe

dot avx2 1024 fma       time:   [35.280 ns 35.350 ns 35.443 ns]
                        change: [-4.2100% -4.0410% -3.8628%] (p = 0.00 < 0.05)
                        Performance has improved.
```

#### Euclidean
```
euclidean autovec 1024 nofma
                        time:   [640.13 ns 641.09 ns 642.21 ns]
                        change: [+0.6032% +0.8656% +1.1820%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 8 outliers among 100 measurements (8.00%)
  7 (7.00%) high mild
  1 (1.00%) high severe

euclidean autovec 1024 fma
                        time:   [848.70 ns 852.44 ns 856.34 ns]
                        change: [-2.0355% -1.6956% -1.3499%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 8 outliers among 100 measurements (8.00%)
  6 (6.00%) high mild
  2 (2.00%) high severe

euclidean simsimd 1024 auto
                        time:   [624.09 ns 625.18 ns 626.64 ns]
                        change: [-1.4581% -1.1684% -0.8933%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 8 outliers among 100 measurements (8.00%)
  4 (4.00%) high mild
  4 (4.00%) high severe

euclidean avx2 1024 nofma
                        time:   [41.972 ns 41.995 ns 42.018 ns]
                        change: [-20.838% -18.647% -16.927%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 3 outliers among 100 measurements (3.00%)
  2 (2.00%) low mild
  1 (1.00%) high mild

euclidean avx2 1024 fma time:   [41.229 ns 41.263 ns 41.307 ns]
                        change: [-18.592% -17.681% -16.916%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 8 outliers among 100 measurements (8.00%)
  2 (2.00%) low mild
  3 (3.00%) high mild
  3 (3.00%) high severe
```