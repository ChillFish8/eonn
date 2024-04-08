use igris_accel::distance_ops::DistanceOps;

/// The type of random projection tree to build.
pub enum TreeType {
    /// Angular projection tree.
    Angular,
    /// Euclidean projection tree.
    Euclidean,
}

/// Builds a random projection forest containing `n_trees` with a given `leaf_size`.
pub fn build_forest<V: DistanceOps>(
    data: &[V],
    n_neighbors: usize,
    n_trees: usize,
    leaf_size: usize,
) {
}
