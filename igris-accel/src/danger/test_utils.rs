use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::danger::utils::cosine;
use crate::math::StdMath;

const SEED: u64 = 2837564324875;

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


pub fn simple_dot(x: &[f32], y: &[f32]) -> f32 {
    let mut dot_product = 0.0;
    
    for i in 0..x.len() {
        dot_product += x[i] * y[i];
    }
    
    dot_product
}

pub fn simple_cosine(x: &[f32], y: &[f32]) -> f32 {
    let mut dot_product = 0.0;
    let mut norm_x = 0.0;
    let mut norm_y = 0.0;

    for i in 0..x.len() {
        dot_product += x[i] * y[i];
        norm_x += x[i] * x[i];
        norm_y += y[i] * y[i];
    }

    cosine::<StdMath>(dot_product, norm_x, norm_y)
}

pub fn simple_euclidean(x: &[f32], y: &[f32]) -> f32 {
    let mut dist = 0.0;

    for i in 0..x.len() {
        let diff = x[i] - y[i];
        dist += diff * diff;
    }

    dist
}