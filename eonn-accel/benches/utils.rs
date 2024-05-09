use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 2837564324875;

#[macro_export]
macro_rules! repeat {
    ($n:expr, $func:expr, $($x:expr $(,)?)*) => {{
        #[allow(unused_unsafe)]
        unsafe {
            for _ in 0..$n {
                black_box($func($(black_box($x),)*));
            }
        }
    }};
}

#[cfg(not(feature = "aligned-benchmarks"))]
pub fn get_sample_vectors(size: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);

    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..size {
        x.push(rng.gen());
        y.push(rng.gen());
    }

    (x, y)
}

#[cfg(feature = "aligned-benchmarks")]
pub fn get_sample_vectors(size: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);

    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..size {
        x.push(rng.gen());
        y.push(rng.gen());
    }

    (x, y)
}
