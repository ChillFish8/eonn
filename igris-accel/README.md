# igris-accel

Various specialised vector operations, primarily designed for similarity search.

This library requires nightly and is more unsafe than it is safe code, it is only intended to be used
within the other Igris tools, if you absolutely want to use this library directly, do so at your own risk.

### Benchmarks

The benchmarks are located in the `bench/` directory and are ran in batches of 1000 iterations
to account for the clock margin of error, so if you want the time per vector, divide by 1000.

#### Cosine
```
cosine avx2 1024 nofma  time:   [113.57 µs 113.65 µs 113.72 µs]
                        change: [+9.8913% +10.379% +10.823%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 6 outliers among 500 measurements (1.20%)
  1 (0.20%) low mild
  4 (0.80%) high mild
  1 (0.20%) high severe

cosine avx2 1024 fma    time:   [82.346 µs 82.450 µs 82.562 µs]
                        change: [-26.756% -26.582% -26.391%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 27 outliers among 500 measurements (5.40%)
  20 (4.00%) high mild
  7 (1.40%) high severe

cosine autovec 1024 nofma
                        time:   [230.74 µs 230.98 µs 231.23 µs]
                        change: [-0.2116% -0.0955% +0.0171%] (p = 0.10 > 0.05)
                        No change in performance detected.
Found 5 outliers among 500 measurements (1.00%)
  5 (1.00%) high mild

cosine autovec 1024 fma time:   [386.07 µs 387.07 µs 388.15 µs]
                        change: [+0.3952% +0.6105% +0.8408%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 11 outliers among 500 measurements (2.20%)
  7 (1.40%) high mild
  4 (0.80%) high severe

Benchmarking cosine simsimd 1024 auto: Warming up for 3.0000 s
Warning: Unable to complete 500 samples in 60.0s. You may wish to increase target time to 82.6s, enable flat sampling, or reduce sample count to 300.
cosine simsimd 1024 auto
                        time:   [661.55 µs 662.07 µs 662.63 µs]
                        change: [-0.1862% -0.0206% +0.1995%] (p = 0.84 > 0.05)
                        No change in performance detected.
Found 60 outliers among 500 measurements (12.00%)
  21 (4.20%) high mild
  39 (7.80%) high severe
```

#### Dot Product
```
dot avx2 1024 nofma     time:   [45.344 µs 45.395 µs 45.448 µs]
                        change: [+0.0108% +0.1411% +0.2724%] (p = 0.04 < 0.05)
                        Change within noise threshold.
Found 24 outliers among 500 measurements (4.80%)
  13 (2.60%) high mild
  11 (2.20%) high severe

dot avx2 1024 fma       time:   [25.844 µs 25.878 µs 25.916 µs]
                        change: [-23.589% -23.462% -23.341%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 22 outliers among 500 measurements (4.40%)
  6 (1.20%) high mild
  16 (3.20%) high severe

dot ndarray 1024 auto   time:   [54.650 µs 54.751 µs 54.858 µs]
                        change: [+24.499% +24.651% +24.804%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 93 outliers among 500 measurements (18.60%)
  1 (0.20%) low severe
  36 (7.20%) low mild
  24 (4.80%) high mild
  32 (6.40%) high severe

Benchmarking dot simsimd 1024 auto: Warming up for 3.0000 s
Warning: Unable to complete 500 samples in 60.0s. You may wish to increase target time to 79.8s, enable flat sampling, or reduce sample count to 300.
dot simsimd 1024 auto   time:   [631.24 µs 632.19 µs 633.16 µs]
                        change: [-0.0290% +0.1701% +0.3684%] (p = 0.08 > 0.05)
                        No change in performance detected.
Found 20 outliers among 500 measurements (4.00%)
  20 (4.00%) high mild

dot fallback 1024 nofma time:   [228.00 µs 228.56 µs 229.19 µs]
                        change: [+1.3248% +1.5400% +1.8119%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 64 outliers among 500 measurements (12.80%)
  27 (5.40%) high mild
  37 (7.40%) high severe

dot fallback 1024 fma   time:   [95.686 µs 95.939 µs 96.218 µs]
                        change: [-0.1782% +0.0682% +0.3151%] (p = 0.59 > 0.05)
                        No change in performance detected.
Found 18 outliers among 500 measurements (3.60%)
  12 (2.40%) high mild
  6 (1.20%) high severe
```

#### Euclidean
```
euclidean avx2 1024 nofma
                        time:   [46.002 µs 46.052 µs 46.106 µs]
                        change: [+0.2598% +0.4551% +0.7081%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 28 outliers among 500 measurements (5.60%)
  15 (3.00%) high mild
  13 (2.60%) high severe

euclidean avx2 1024 fma time:   [37.702 µs 37.757 µs 37.811 µs]
                        change: [-0.3372% -0.2074% -0.0861%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 23 outliers among 500 measurements (4.60%)
  18 (3.60%) high mild
  5 (1.00%) high severe

euclidean autovec 1024 nofma
                        time:   [230.34 µs 230.76 µs 231.21 µs]
                        change: [+1.9804% +2.1995% +2.4300%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 24 outliers among 500 measurements (4.80%)
  14 (2.80%) high mild
  10 (2.00%) high severe

euclidean autovec 1024 fma
                        time:   [96.099 µs 96.330 µs 96.571 µs]
                        change: [-0.5210% +0.4549% +1.0191%] (p = 0.37 > 0.05)
                        No change in performance detected.
Found 51 outliers among 500 measurements (10.20%)
  1 (0.20%) low mild
  23 (4.60%) high mild
  27 (5.40%) high severe

Benchmarking euclidean simsimd 1024 auto: Warming up for 3.0000 s
Warning: Unable to complete 500 samples in 75.0s. You may wish to increase target time to 80.5s, enable flat sampling, or reduce sample count to 340.
euclidean simsimd 1024 auto
                        time:   [645.46 µs 647.67 µs 649.94 µs]
                        change: [+1.5618% +1.8118% +2.0929%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 33 outliers among 500 measurements (6.60%)
  28 (5.60%) high mild
  5 (1.00%) high severe
```