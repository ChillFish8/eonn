use std::fmt::{Debug, Display, Formatter};

use crate::arch::Arch;
use crate::ops::{DangerousOps, SpacialOps};
use crate::{Dim, VectorType};

/// A fixed-size SIMD accelerated vector of a given type and dimensions.
///
/// This type allows for various targeting of CPU features and dimensions
/// both at runtime or at compile time depending on application.
pub struct Vector<D: Dim, A: Arch, T: VectorType = f32>
where
    (D, A): DangerousOps,
{
    buffer: Vec<T>,
    ops: (D, A),
}

impl<D: Dim + Debug, A: Arch + Debug, T: VectorType + Debug> Debug for Vector<D, A, T>
where
    (D, A): DangerousOps,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if D::size() != 0 {
            let first = self.buffer.first().unwrap();
            let last = self.buffer.last().unwrap();

            write!(f, "Vector(ops={:?}, [ {first:?} ... {last:?} ])", self.ops)
        } else {
            write!(f, "Vector(ops={:?}, [])", self.ops)
        }
    }
}

impl<D: Dim, A: Arch, T: VectorType> Vector<D, A, T>
where
    (D, A): DangerousOps,
{
    #[inline]
    /// Attempt to create a new vector using the given input data.
    ///
    /// This method will verify that the data is not-nan and finite along
    /// with being valid dimensions.
    ///
    /// This method does not allocate.
    pub fn try_from_vec(data: Vec<T>) -> Result<Self, VectorCreateError> {
        if data.len() != D::size() {
            return Err(VectorCreateError::BadDimensions {
                expected: D::size(),
                got: data.len(),
            });
        }

        if data.iter().any(|v| !(v.is_finite() || v.is_nan())) {
            return Err(VectorCreateError::NonFinite);
        }

        Ok(unsafe { Self::from_vec_unchecked(data) })
    }

    #[inline]
    /// Creates a new vector from the given data without checking any values or dimensions.
    ///
    /// # Safety
    /// You **must** ensure all values within the provided data are both finite and not nan,
    /// along with ensuring the dimensions/length of the data matches the dimensions specified
    /// by `D`.
    ///
    /// If any of these checks are not performed or invalid, this creates immediate UB.
    pub unsafe fn from_vec_unchecked(data: Vec<T>) -> Self {
        debug_assert_eq!(data.len(), D::size());
        debug_assert!(!data.iter().any(|v| !(v.is_finite() || v.is_nan())));

        Self {
            buffer: data,
            ops: <(D, A) as Default>::default(),
        }
    }
}

impl<D: Dim, A: Arch> SpacialOps for Vector<D, A, f32>
where
    (D, A): DangerousOps,
{
    fn len() -> usize {
        D::size()
    }

    fn dot(&self, other: &Self) -> f32 {
        unsafe { self.ops.dot(&self.buffer, &other.buffer) }
    }

    fn squared_norm(&self) -> f32 {
        unsafe { self.ops.squared_norm(&self.buffer) }
    }

    fn normalize(&mut self) {
        let inverse_norm = 1.0 / self.squared_norm().sqrt();
        for v in self.buffer.iter_mut() {
            *v *= inverse_norm;
        }
    }

    fn dist_dot(&self, other: &Self) -> f32 {
        let product = unsafe { self.ops.dot(&self.buffer, &other.buffer) };

        if product <= 0.0 {
            1.0
        } else {
            1.0 - product
        }
    }

    fn dist_cosine(&self, other: &Self) -> f32 {
        unsafe { self.ops.cosine(&self.buffer, &other.buffer) }
    }

    fn dist_squared_euclidean(&self, other: &Self) -> f32 {
        unsafe { self.ops.squared_euclidean(&self.buffer, &other.buffer) }
    }

    fn angular_hyperplane(&self, other: &Self) -> Self {
        unsafe {
            let data = self.ops.angular_hyperplane(&self.buffer, &other.buffer);
            Self::from_vec_unchecked(data)
        }
    }

    fn euclidean_hyperplane(&self, other: &Self) -> (Self, f32) {
        unsafe {
            let (data, offset) =
                self.ops.euclidean_hyperplane(&self.buffer, &other.buffer);
            (Self::from_vec_unchecked(data), offset)
        }
    }
}

#[derive(Debug)]
/// An error that occurs while attempting to safely create a new `Vector` type
/// with a given set of dimensions.
pub enum VectorCreateError {
    /// The provided vector was not the correct dimensions.
    BadDimensions { expected: usize, got: usize },
    /// The provided vector contains some non-finite values or Nan.
    NonFinite,
}

impl Display for VectorCreateError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadDimensions { expected, got } => {
                write!(f, "Bad Dimensions: expected {expected} but got {got}")
            },
            Self::NonFinite => {
                write!(f, "Non-Finite Value: values in vector must be finite")
            },
        }
    }
}

impl std::error::Error for VectorCreateError {}
