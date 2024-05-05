pub trait VectorType {
    fn is_finite(&self) -> bool;
    fn is_nan(&self) -> bool;
}

impl VectorType for f32 {
    fn is_finite(&self) -> bool {
        f32::is_finite(*self)
    }

    fn is_nan(&self) -> bool {
        f32::is_nan(*self)
    }
}
