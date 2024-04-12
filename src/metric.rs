use rann_accel::SpacialOps;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
/// The distance metrics available.
pub enum Metric {
    /// The dot product distance.
    Dot,
    /// The squared Euclidean distance.
    ///
    /// The relative distance of vectors is the same as 
    /// standard Euclidean. 
    SquaredEuclidean,
    /// The cosine distance.
    Cosine,
}

impl Metric {
    #[inline]
    /// Calculates the distance metric of two vectors.
    pub fn distance<V: SpacialOps>(&self, x: &V, y: &V) -> f32 {
        match self {
            Metric::Dot => x.dist_dot(y),
            Metric::SquaredEuclidean => x.dist_squared_euclidean(y),
            Metric::Cosine => x.dist_cosine(y),
        }
    }
    
    #[inline]
    /// Returns if the vectors should be normalized.
    pub fn requires_normalizing(&self) -> bool {
        matches!(self, Self::Dot)
    }

    #[inline]
    /// Returns if the vectors should be normalized.
    pub fn requires_angular_trees(&self) -> bool {
        matches!(self, Self::Dot | Self::Cosine)
    }
}