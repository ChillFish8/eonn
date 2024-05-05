use rann_accel::{Arch, DangerousOps, Dim, Vector};

pub fn random_vector<D: Dim, A: Arch>() -> Vector<D, A>
where
    (D, A): DangerousOps,
{
    let mut vector = Vec::with_capacity(D::const_size());
    for _ in 0..D::const_size() {
        vector.push(fastrand::f32());
    }

    unsafe { Vector::from_vec_unchecked(vector) }
}
