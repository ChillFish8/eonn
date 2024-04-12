use std::cmp;
use std::ops::Index;
use std::time::Instant;

use bitvec::bitvec;
use bitvec::prelude::Lsb0;
use bitvec::vec::BitVec;
use rann_accel::SpacialOps;
use tracing::{debug, info};

#[derive(Debug)]
pub struct Tree<V> {
    pub hyperplanes: Vec<Option<V>>,
    pub offsets: Vec<f32>,
    pub children: Vec<Option<(usize, usize)>>,
    pub point_indices: Vec<Option<Vec<usize>>>,
    pub leaf_size: usize,
    pub n_leaves: usize,
}

/// Builds a random project forest with`n_trees`.
pub fn make_forest<V: SpacialOps>(
    data: &[V],
    n_trees: usize,
    leaf_size: usize,
    angular: bool,
    max_depth: usize,
) -> Vec<Tree<V>> {
    let mut trees = Vec::with_capacity(n_trees);

    let total = Instant::now();
    for i in 0..n_trees {
        let start = Instant::now();
        let tree = make_dense_tree(data, leaf_size, angular, max_depth);
        trees.push(tree);
        debug!(elapsed = ?start.elapsed(), idx = i, "Built tree");
    }
    info!(elapsed = ?total.elapsed(), n_trees = n_trees, "Built forest");

    trees
}

#[cfg(feature = "rayon")]
/// Builds a random project forest with `n_trees` in parallel using the
/// given thread pool.
pub fn make_forest_parallel<V: SpacialOps + Send + Sync + 'static>(
    data: &[V],
    n_trees: usize,
    leaf_size: usize,
    angular: bool,
    max_depth: usize,
    pool: &rayon::ThreadPool,
) -> Vec<Tree<V>> {
    use rayon::prelude::*;

    // Safety:
    //   We know the threads that will have access to this will never outlive this function
    //   while still holding onto the data.
    let data = unsafe { std::mem::transmute::<&[V], &'static [V]>(data) };

    let total = Instant::now();

    let trees = pool.install(|| {
        (0..n_trees)
            .into_par_iter()
            .map(|idx| {
                let start = Instant::now();
                let tree = make_dense_tree(data, leaf_size, angular, max_depth);
                debug!(elapsed = ?start.elapsed(), idx = idx, "Built tree");
                tree
            })
            .collect()
    });

    info!(elapsed = ?total.elapsed(), n_trees = n_trees, "Built forest");

    trees
}

pub fn rp_tree_leaf_array<V: SpacialOps>(forest: &[Tree<V>]) -> Vec<Vec<usize>> {
    let mut forest_leaves = Vec::with_capacity(forest.len());

    for tree in forest {
        let l = get_leaves_from_tree(tree);
        forest_leaves.extend(l);
    }

    forest_leaves
}

fn get_leaves_from_tree<V: SpacialOps>(tree: &Tree<V>) -> Vec<Vec<usize>> {
    let leaves_iter = tree
        .children
        .iter()
        .enumerate()
        .filter(|(_, l)| l.is_none())
        .filter_map(|(i, _)| tree.point_indices[i].as_ref());

    let mut leaves = Vec::with_capacity(tree.n_leaves);
    for indices in leaves_iter {
        leaves.push(indices.clone());
    }

    leaves
}

fn make_dense_tree<V: SpacialOps>(
    data: &[V],
    leaf_size: usize,
    angular: bool,
    max_depth: usize,
) -> Tree<V> {
    let indices = (0..data.len()).collect::<Vec<usize>>();
    let mut tree = Tree {
        hyperplanes: vec![],
        offsets: vec![],
        children: vec![],
        point_indices: vec![],
        leaf_size,
        n_leaves: 0,
    };

    if angular {
        make_angular_tree(&mut tree, data, indices, leaf_size, max_depth)
    } else {
        make_euclidean_tree(&mut tree, data, indices, leaf_size, max_depth)
    }

    tree.leaf_size = cmp::max(
        leaf_size,
        tree.point_indices
            .iter()
            .filter_map(|v| v.as_ref())
            .map(|v| v.len())
            .max()
            .unwrap_or(leaf_size),
    );

    tree
}

fn make_angular_tree<V: SpacialOps>(
    tree: &mut Tree<V>,
    data: &[V],
    indices: Vec<usize>,
    leaf_size: usize,
    max_depth: usize,
) {
    if indices.len() > leaf_size && max_depth > 0 {
        let (left_indices, right_indices, hyperplane) =
            angular_random_project_split(data, indices);

        make_angular_tree(tree, data, left_indices, leaf_size, max_depth - 1);

        let left_node_num = tree.point_indices.len() - 1;

        make_angular_tree(tree, data, right_indices, leaf_size, max_depth - 1);

        let right_node_num = tree.point_indices.len() - 1;

        tree.hyperplanes.push(Some(hyperplane));
        tree.offsets.push(0.0);
        tree.children.push(Some((left_node_num, right_node_num)));
        tree.point_indices.push(None);
    } else {
        tree.hyperplanes.push(None);
        tree.offsets.push(f32::NEG_INFINITY);
        tree.children.push(None);
        tree.point_indices.push(Some(indices));
        tree.n_leaves += 1;
    }
}

fn make_euclidean_tree<V: SpacialOps>(
    tree: &mut Tree<V>,
    data: &[V],
    indices: Vec<usize>,
    leaf_size: usize,
    max_depth: usize,
) {
    if indices.len() > leaf_size && max_depth > 0 {
        let (left_indices, right_indices, hyperplane, offset) =
            euclidean_random_projection_split(data, indices);

        make_euclidean_tree(tree, data, left_indices, leaf_size, max_depth - 1);

        let left_node_num = tree.point_indices.len() - 1;

        make_euclidean_tree(tree, data, right_indices, leaf_size, max_depth - 1);

        let right_node_num = tree.point_indices.len() - 1;

        tree.hyperplanes.push(Some(hyperplane));
        tree.offsets.push(offset);
        tree.children.push(Some((left_node_num, right_node_num)));
        tree.point_indices.push(None);
    } else {
        tree.hyperplanes.push(None);
        tree.offsets.push(f32::NEG_INFINITY);
        tree.children.push(None);
        tree.point_indices.push(Some(indices));
        tree.n_leaves += 1;
    }
}

/// Given a set of `graph_indices` for graph_data points from `data`, create
/// a random hyperplane to split the graph_data, returning two arrays graph_indices
/// that fall on either side of the hyperplane. This is the basis for a
/// random projection tree, which simply uses this splitting recursively.
fn angular_random_project_split<V: SpacialOps>(
    data: &[V],
    indices: Vec<usize>,
) -> (Vec<usize>, Vec<usize>, V) {
    let (left, right) = select_left_right(data, &indices);

    let hyperplane = left.angular_hyperplane(right);

    let (left_indices, right_indices) = select_sides(data, indices, &hyperplane, 0.0);

    (left_indices, right_indices, hyperplane)
}

fn euclidean_random_projection_split<V: SpacialOps>(
    data: &[V],
    indices: Vec<usize>,
) -> (Vec<usize>, Vec<usize>, V, f32) {
    let (left, right) = select_left_right(data, &indices);

    let (hyperplane, offset) = left.euclidean_hyperplane(right);

    let (left_indices, right_indices) = select_sides(data, indices, &hyperplane, offset);

    (left_indices, right_indices, hyperplane, offset)
}

fn select_sides<V: SpacialOps>(
    data: &[V],
    indices: Vec<usize>,
    hyperplane: &V,
    offset: f32,
) -> (Vec<usize>, Vec<usize>) {
    let mut num_left = 0;
    let mut num_right = 0;
    let mut side: BitVec = bitvec![usize, Lsb0; 0; indices.len()];

    for i in 0..indices.len() {
        let margin = offset + hyperplane.dot(&data[indices[i]]);

        if margin.abs() < f32::EPSILON {
            let v = fastrand::bool();
            side.set(i, v);

            if v {
                num_left += 1;
            } else {
                num_right += 1;
            }
        } else if margin > 0.0 {
            side.set(i, true);
            num_left += 1;
        } else {
            side.set(i, false);
            num_right += 1;
        }
    }

    // If all points end up on one side, something went wrong numerically
    // In this case, assign points randomly; they are likely very close anyway
    if num_left == 0 || num_right == 0 {
        num_left = 0;
        num_right = 0;

        for i in 0..indices.len() {
            let v = fastrand::bool();
            side.set(i, v);

            if v {
                num_left += 1;
            } else {
                num_right += 1;
            }
        }
    }

    let mut indices_left = Vec::with_capacity(num_left);
    let mut indices_right = Vec::with_capacity(num_right);

    for i in 0..indices.len() {
        if *side.index(i) {
            indices_left.push(i);
        } else {
            indices_right.push(i);
        }
    }

    (indices_left, indices_right)
}

#[inline]
fn select_left_right<'a, V: SpacialOps>(
    data: &'a [V],
    indices: &[usize],
) -> (&'a V, &'a V) {
    let mut left_index = fastrand::usize(0..indices.len());
    let mut right_index = fastrand::usize(0..indices.len());
    right_index += (left_index == right_index) as usize;
    right_index %= indices.len();

    left_index = indices[left_index];
    right_index = indices[right_index];

    (&data[left_index], &data[right_index])
}

#[cfg(test)]
mod tests {
    use rann_accel::{Auto, Vector, X512};

    use super::*;

    fn test_data() -> Vec<Vector<X512, Auto>> {
        let mut data = Vec::with_capacity(10);

        for _ in 0..15 {
            let v =
                Vec::from_iter(std::iter::from_fn(|| Some(fastrand::f32())).take(512));
            let v = Vector::try_from_vec(v).expect("Load vec");
            data.push(v);
        }

        data
    }

    #[test]
    fn test_build_forest() {
        let data = test_data();
        let forest = make_forest(&data, 4, 3, true, 200);
        dbg!(forest);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_build_forest_parallel() {
        let data = test_data();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let forest = make_forest_parallel(&data, 4, 3, true, 200, &pool);
        dbg!(forest);
    }
}
