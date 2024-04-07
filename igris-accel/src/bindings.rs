use crate::danger::f32_x1024_avx2_nofma_dot;


#[link(name = "igriskernels")]
extern {
    pub fn f32_x1024_dot(x: *const f32, y: *const f32) -> f32;
    
    pub fn f32_x1024_cosine(x: *const f32, y: *const f32) -> f32;
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use super::*;

    pub fn get_sample_vectors(size: usize) -> (Vec<f32>, Vec<f32>) {
        let mut rng = ChaCha8Rng::seed_from_u64(3546285762352);

        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..size {
            x.push(rng.gen());
            y.push(rng.gen());
        }

        (x, y)
    }
    
    #[test]
    fn test_fortran() {
        let (v1, v2) = get_sample_vectors(1024);
        let a1 = unsafe { f32_x1024_dot(v1.as_ptr(), v2.as_ptr()) };
        let a2 = unsafe { f32_x1024_avx2_nofma_dot(&v1, &v2) };
        println!("{a1} vs {a2}???");
        let v = unsafe { f32_x1024_cosine(v1.as_ptr(), v2.as_ptr()) };
        println!("{v}???");
    }
} 