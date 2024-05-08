mod f32_auto_size;
mod f32_fixed_size;

#[allow(clippy::len_without_is_empty)]
/// Safe spacial type operations.
pub trait SpacialOps: Sized {
    /// Returns the length of the vector.
    fn len(&self) -> usize;
    /// Computes the dot product between self and another vector.
    fn dot(&self, other: &Self) -> f32;
    /// Computes the squared norm of the vector.
    fn squared_norm(&self) -> f32;
    /// Normalizes the vector.
    fn normalize(&mut self);
    /// Computes the dot product distance between self and another vector.
    fn dist_dot(&self, other: &Self) -> f32;
    /// Computes the cosine distance between self and another vector.
    fn dist_cosine(&self, other: &Self) -> f32;
    /// Computes the cosine distance between self and another vector.
    fn dist_squared_euclidean(&self, other: &Self) -> f32;
    /// Computes the angular hyperplane vector between self and another vector.
    fn angular_hyperplane(&self, other: &Self) -> Self;
    /// Computes the Euclidean hyperplane vector between self and another vector and
    /// returns the offset.
    fn euclidean_hyperplane(&self, other: &Self) -> (Self, f32);
}

/// A set of compute ops various archs and dims implement.
///
/// # Safety
/// All vectors must contain only finite values.
pub trait DangerousOps {
    /// Computes the dot product of the two provided vectors.
    ///
    /// # Safety
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn dot(&self, x: &[f32], y: &[f32]) -> f32;
    /// Computes the cosine distance of the two provided vectors.
    ///
    /// # Safety
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn cosine(&self, x: &[f32], y: &[f32]) -> f32;
    /// Computes the squared Euclidean distance of the two provided vectors.
    ///
    /// # Safety
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn squared_euclidean(&self, x: &[f32], y: &[f32]) -> f32;
    /// Computes the angular hyperplane to the two vector points.
    ///
    /// # Safety
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn angular_hyperplane(&self, x: &[f32], y: &[f32]) -> Vec<f32>;
    /// Computes the Euclidean hyperplane and hyperplane offset.
    ///
    /// # Safety
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn euclidean_hyperplane(&self, x: &[f32], y: &[f32]) -> (Vec<f32>, f32);
    /// Computes the squared norm of the given vector.
    ///
    /// # Safety
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn squared_norm(&self, x: &[f32]) -> f32;
    /// Adds the value `val` to each element in the vector `x`.
    ///
    /// # Safety
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn add_value(&self, x: &mut [f32], val: f32);
    /// Adds the value `val` to each element in the vector `x`.
    ///
    /// # Safety
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn sub_value(&self, x: &mut [f32], val: f32);
    /// Adds the value `val` to each element in the vector `x`.
    ///
    /// # Safety
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn mul_value(&self, x: &mut [f32], val: f32);
    /// Adds the value `val` to each element in the vector `x`.
    ///
    /// # Safety
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn div_value(&self, x: &mut [f32], y: f32);
    /// Performs a vertical addition of each element in `x` with each respective element in `y`.
    ///
    /// # Safety
    /// Vectors `x` and `y` must be equal in length.
    ///
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn add_vertical(&self, x: &mut [f32], y: &[f32]);
    /// Performs a vertical subtraction of each element in `x` with each respective
    /// element in `y`.
    ///
    /// # Safety
    /// Vectors `x` and `y` must be equal in length.
    ///
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn sub_vertical(&self, x: &mut [f32], y: &[f32]);
    /// Performs a vertical multiplication of each element in `x` by each respective
    /// element in `y`.
    ///
    /// # Safety
    /// Vectors `x` and `y` must be equal in length.
    ///
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn mul_vertical(&self, x: &mut [f32], y: &[f32]);
    /// Performs a vertical division of each element in `x` by each respective element in `y`.
    ///
    /// # Safety
    /// Vectors `x` and `y` must be equal in length.
    ///
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn div_vertical(&self, x: &mut [f32], y: &[f32]);
    /// Performs a horizontal sum of the given vector.
    ///
    /// # Safety
    ///
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn sum(&self, x: &[f32]) -> f32;
    /// Performs a horizontal min of the given vector.
    ///
    /// # Safety
    ///
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn min(&self, x: &[f32]) -> f32;
    /// Performs a horizontal max of the given vector.
    ///
    /// # Safety
    ///
    /// All vectors must contain only finite values and be not-nan. The dimensions
    /// of `x` and `y` must also be equal and align with the implementor's required
    /// dimension sizes.
    unsafe fn max(&self, x: &[f32]) -> f32;
}
