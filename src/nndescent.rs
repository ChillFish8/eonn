use std::cmp;
use std::time::Instant;
use tracing::info;
use rann_accel::{Auto, SpacialOps, Vector, X512};

use crate::graph::DynamicGraph;
use crate::metric::Metric;
use crate::rp_trees::Tree;


/// Approximate nearest neighbour graph construction and search using NNDescent.
///
/// This implementation is effectively a port of the Python package:
/// https://github.com/lmcinnes/pynndescent by Leland McInnes which combines
/// the NNDescent paper and Random Projection forests to quickly bootstrap graph
/// building and searching.
///
/// Unlike the Python version which supports sparse and dense vectors, this implementation
/// only supports _dense_ vectors and a limited set of distance measures, in particular,
/// `dot`, `cosine` and `squared_euclidean`. Which is enough for most cases.
pub struct NNDescent<V: SpacialOps> {
    data: Vec<V>,
    metric: Metric,
}


/// The builder for configuring the NNDescent process.
///
/// Approximate nearest neighbour graph construction and search using NNDescent.
///
/// This implementation is effectively a port of the Python package:
/// https://github.com/lmcinnes/pynndescent by Leland McInnes which combines
/// the NNDescent paper and Random Projection forests to quickly bootstrap graph
/// building and searching.
///
/// Unlike the Python version which supports sparse and dense vectors, this implementation
/// only supports _dense_ vectors and a limited set of distance measures, in particular,
/// `dot`, `cosine` and `squared_euclidean`. Which is enough for most cases.
pub struct NNDescentBuilder<V: SpacialOps = Vector<X512, Auto>> {
    data: Vec<V>,
    metric: Metric,
    n_neighbors: usize,
    n_trees: Option<usize>,
    leaf_size: Option<usize>,
    pruning_degree_multiplier: f32,
    diversify_prob: f32,
    low_memory: bool,
    max_rptree_depth: usize,
    n_iters: Option<usize>,
    delta: f32,
    skip_normalization: bool,
    #[cfg(feature = "rayon")]
    thread_pool: Option<rayon::ThreadPool>,
}


impl<V: SpacialOps> Default for NNDescentBuilder<V> {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            metric: Metric::SquaredEuclidean,
            n_neighbors: 30,
            n_trees: None,
            leaf_size: None,
            pruning_degree_multiplier: 1.5,
            diversify_prob: 1.0,
            low_memory: true,
            max_rptree_depth: 100,
            n_iters: None,
            delta: 0.001,
            skip_normalization: false,
            #[cfg(feature = "rayon")]
            thread_pool: None,
        }
    }
}

impl<V: SpacialOps + Send + Sync + 'static> NNDescentBuilder<V> {
    /// Sets the initial graph data points.
    pub fn with_data<V2: SpacialOps>(self, data: Vec<V2>) -> NNDescentBuilder<V2> {
        NNDescentBuilder {
            data,
            metric: self.metric,
            n_neighbors: self.n_neighbors,
            n_trees: self.n_trees,
            leaf_size: self.leaf_size,
            pruning_degree_multiplier: self.pruning_degree_multiplier,
            diversify_prob: self.diversify_prob,
            low_memory: self.low_memory,
            max_rptree_depth: self.max_rptree_depth,
            n_iters: self.n_iters,
            delta: self.delta,
            skip_normalization: true,
            #[cfg(feature = "rayon")]
            thread_pool: self.thread_pool,
        }
    }

    /// Set the distance metric used when constructing the graph.
    ///
    /// By default, this uses [Metric::SquaredEuclidean].
    pub fn with_metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the number of neighbors per node on the graph.
    ///
    /// Defaults to `30`.
    pub fn with_n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the number of trees to create when building the
    /// random projection forest.
    ///
    /// If not set, this will be calculated automatically.
    pub fn with_n_trees(mut self, n_trees: usize) -> Self {
        self.n_trees = Some(n_trees);
        self
    }

    /// Set the leaf size of RP trees.
    ///
    /// If not set, this will be calculated automatically.
    pub fn with_leaf_size(mut self, leaf_size: usize) -> Self {
        self.leaf_size = Some(leaf_size);
        self
    }

    /// Set how aggressively to prune the graph.
    ///
    /// Since the search graph is undirected (and thus includes nearest neighbors and
    /// reverse nearest neighbors) vertices can have very high degree
    /// -- the graph will be pruned such that no vertex has degree greater than
    /// `pruning_degree_multiplier * n_neighbors`.
    ///
    /// Defaults to `1.5`
    pub fn with_pruning_degree_multiplier(mut self, multiplier: f32) -> Self {
        self.pruning_degree_multiplier = multiplier;
        self
    }

    /// The search graph gets "diversified" by removing potentially unnecessary edges.
    /// This controls the volume of edges removed. A value of 0.0 ensures that no
    /// edges get removed, and larger values result in significantly more aggressive edge
    /// removal. A value of `1.0` will prune all edges that it can.
    ///
    /// Defaults to `1.0`
    pub fn with_diversify_prob(mut self, prob: f32) -> Self {
        self.diversify_prob = prob;
        self
    }

    /// Enables/disables low memory mode when constructing the graph.
    ///
    /// The low memory approach trades off build time for lower memory usage.
    ///
    /// Defaults to `true`.
    pub fn with_set_low_memory_mode(mut self, enabled: bool) -> Self {
        self.low_memory = enabled;
        self
    }

    /// Adjusts the maximum recursion depth when constructing RP trees.
    ///
    /// Increasing this may result in a richer, deeper random projection forest,
    /// but it may be composed of many degenerate branches.
    /// Increase leaf_size in order to keep shallower, wider nondegenerate trees.
    /// Such wide trees, however, may yield poor performance of the preparation of the NN descent.
    pub fn with_max_rptree_depth(mut self, depth: usize) -> Self {
        self.max_rptree_depth = depth;
        self
    }

    /// Set the maximum number of NN Descent iterations to perform.
    ///
    /// The NN-descent algorithm can abort early if limited progress is being
    /// made, so this only controls the worst case. Don't tweak this value unless you
    /// know what you're doing. The default of `None` means a value will be chosen based
    /// on the size of the graph_data.
    pub fn with_n_iters(mut self, n_iters: usize) -> Self {
        self.n_iters = Some(n_iters);
        self
    }

    /// Controls the early abort due to limited progress.
    ///
    /// Larger values will result in earlier aborts, providing less accurate indexes, and
    /// less accurate searching. Don't tweak this value unless you know what you're doing.
    pub fn with_delta(mut self, delta: f32) -> Self {
        self.delta = delta;
        self
    }

    /// Skips the normalization of the vectors if the metric
    /// would normally require normalizing the data.
    ///
    /// You should only do this if your data is already normalized
    /// and your metric actually requires it.
    pub fn with_skip_normalization(mut self, skip: bool) -> Self {
        self.skip_normalization = skip;
        self
    }

    #[cfg(feature = "rayon")]
    /// Sets the number of threads to use in parallel when constructing the graph.
    ///
    /// This will automatically construct a [rayon::ThreadPool].
    ///
    /// Requires the `rayon` feature to be enabled.
    ///
    /// Defaults to `None` (single threaded), values of `<=1` also remain single threaded
    pub fn with_n_threads(mut self, n_threads: usize) -> Self {
        if n_threads <= 1 {
            self.thread_pool = None;
        } else {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .thread_name(|idx| format!("rann-worker-{idx}"))
                .build()
                .expect("Create threadpool");
            self.thread_pool = Some(pool);
        }

        self
    }

    #[cfg(feature = "rayon")]
    /// Sets the thread pool to use when building the graph in parallel using an existing
    /// [rayon::ThreadPool].
    ///
    /// Requires the `rayon` feature to be enabled.
    pub fn with_thread_pool(mut self, pool: rayon::ThreadPool) -> Self {
        self.thread_pool = Some(pool);
        self
    }

    /// Constructs the approximate nearest neighbour using the current configuration.
    pub fn build(mut self) -> NNDescent<V> {
        if self.metric.requires_normalizing() && !self.skip_normalization {
            for vector in self.data.iter_mut() {
                vector.normalize();
            }
        }

        let start = Instant::now();
        let rp_forest = self.create_rp_forest();
        let leaf_array = crate::rp_trees::rp_tree_leaf_array(&rp_forest);
        info!(elapsed = ?start.elapsed(), "Finished creating RP forest");

        let graph = self.nn_descent(&leaf_array);

        todo!()
    }

    fn n_trees(&self) -> usize {
        if let Some(n_trees) = self.n_trees {
            n_trees
        } else {
            let mut n_trees = 5 + (self.data.len() as f32)
                .powf(0.25)
                .round() as usize;
            n_trees = cmp::max(32, n_trees);  // Only so many trees are useful
            n_trees
        }
    }

    fn n_iters(&self) -> usize {
        if let Some(n_iters) = self.n_iters {
            n_iters
        } else {
            5 + (self.data.len() as f32)
                .log2()
                .round() as usize
        }
    }

    #[cfg(not(feature = "rayon"))]
    fn create_rp_forest(&self) -> Vec<Tree<V>> {
        let leaf_size = self.leaf_size.unwrap_or_else(|| {
            cmp::max(10, self.n_neighbors)
        });

        crate::rp_trees::make_forest(
            &self.data,
            self.n_trees(),
            leaf_size,
            self.metric.requires_angular_trees(),
            self.max_rptree_depth,
        )
    }

    #[cfg(feature = "rayon")]
    fn create_rp_forest(&self) -> Vec<Tree<V>> {
        let n_trees = self.n_trees();
        let angular = self.metric.requires_angular_trees();
        let leaf_size = self.leaf_size.unwrap_or_else(|| {
            cmp::max(10, self.n_neighbors)
        });
        let parallel = self.thread_pool.is_some();

        info!(
            n_trees = n_trees,
            angular = angular,
            leaf_size = leaf_size,
            parallel = parallel,
            "Creating RP forest",
        );

        if let Some(pool) = self.thread_pool.as_ref() {
            crate::rp_trees::make_forest_parallel(
                &self.data,
                self.n_trees(),
                leaf_size,
                self.metric.requires_angular_trees(),
                self.max_rptree_depth,
                pool,
            )
        } else {
            crate::rp_trees::make_forest(
                &self.data,
                self.n_trees(),
                leaf_size,
                self.metric.requires_angular_trees(),
                self.max_rptree_depth,
            )
        }
    }

    fn nn_descent(
        &self,
        leaf_array: &[Vec<usize>],
    ) -> DynamicGraph {
        let mut graph = DynamicGraph::new(self.data.len(), self.n_neighbors);

        

        graph
    }
    
    fn init_graph(&self, graph: &mut DynamicGraph, leaf_array: &[Vec<usize>]) {
        
    }
}

