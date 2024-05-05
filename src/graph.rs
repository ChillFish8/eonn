use std::cmp::Ordering;
use std::fmt::{Debug, Formatter};
use std::mem;

use smallvec::SmallVec;

/// A nearest neighbor graph implementation.
///
/// This graph is dynamic which means points can be changed from it
/// in order to build and optimize the graph.
pub struct DynamicGraph {
    points: Vec<SortedNeighbors>,
    n_neighbors: usize,
}

impl DynamicGraph {
    /// Create a new [DynamicGraph].
    pub fn new(n_points: usize, n_neighbors: usize) -> Self {
        let mut points = Vec::with_capacity(n_points);

        for _ in 0..n_points {
            points.push(SortedNeighbors::new(n_neighbors));
        }

        Self {
            points,
            n_neighbors,
        }
    }

    #[inline]
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }

    #[inline]
    pub fn n_vertices(&self) -> usize {
        self.points.len()
    }

    #[inline]
    /// Get the point `p`.
    pub fn point(&self, p: usize) -> &SortedNeighbors {
        &self.points[p]
    }

    #[inline]
    /// Get the point `p`.
    pub fn point_mut(&mut self, p: usize) -> &mut SortedNeighbors {
        &mut self.points[p]
    }

    #[inline]
    /// The minimum threshold to be within the point's nearest neighbors.
    pub fn threshold(&self, p: usize) -> f32 {
        self.point(p).furthest().dist
    }
}

/// A BinaryHeap-like structure maintaining a fixed size.
///
/// Internally this is actually a set of sorted structures
/// going from smallest -> highest priority looking at layout.
pub struct SortedNeighbors {
    neighbors: SmallVec<[Point; 32]>,
}

impl SortedNeighbors {
    /// Creates a new sorted neighbor implementation with a given size.
    pub fn new(size: usize) -> Self {
        let mut neighbors = SmallVec::with_capacity(size);

        for _ in 0..size {
            neighbors.push(Point::default());
        }

        Self { neighbors }
    }

    /// Iterates over the neighbors of the point.
    pub fn iter_neighbors(&self) -> impl Iterator<Item = Point> + '_ {
        self.neighbors
            .iter()
            .take_while(|p| p.idx != u32::MAX)
            .copied()
    }

    #[inline]
    /// Returns the n'th neighbor at the given point.
    pub fn neighbor(&self, idx: usize) -> Point {
        self.neighbors[idx]
    }

    #[inline]
    /// Returns the neighbor at the given point.
    pub fn neighbor_mut(&mut self, idx: usize) -> &mut Point {
        &mut self.neighbors[idx]
    }

    #[inline]
    /// The furthest neighbor from the point.
    pub fn furthest(&self) -> Point {
        self.neighbors.last().copied().unwrap_or(Point {
            idx: u32::MAX,
            dist: f32::INFINITY,
            flag: false,
        })
    }

    #[inline]
    /// Returns the number of neighbors in the graph below this threshold
    pub fn num_lt(&self, threshold: f32) -> usize {
        let mut count = 0;
        for p in self.neighbors.iter() {
            if p.dist < threshold {
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    #[inline]
    /// Attempts to insert a new point into the heap.
    ///
    /// This checks if the point already exists.
    ///
    /// Returns if the value was inserted or not.
    ///
    /// NOTE:
    /// This implementation **can** have duplicate entries if the distance of the
    /// two duplicates is not the same, this is due to how the search point is found.
    pub fn checked_flagged_push(&mut self, dist: f32, idx: usize, flag: bool) -> bool {
        // Element does not meet the minimum
        if dist >= self.furthest().dist {
            return false;
        }

        let idx = idx as u32;

        let res = self.neighbors.binary_search_by(|neighbor| {
            if neighbor.idx == idx {
                Ordering::Equal
            } else if neighbor.dist < dist {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        });

        // Our starting index to begin swapping values to insert.
        let start = match res {
            Ok(_) => return false, // Element already exists
            Err(start) => start,
        };

        let p = Point { idx, dist, flag };
        self.sift_right(p, start);

        true
    }

    #[inline]
    fn sift_right(&mut self, mut p: Point, start: usize) {
        for i in start..self.neighbors.len() {
            p = mem::replace(&mut self.neighbors[i], p);
        }
    }

    #[inline]
    /// Attempts to insert a new point into the heap.
    ///
    /// This checks if the point already exists.
    ///
    /// Returns if the value was inserted or not.
    ///
    /// NOTE:
    /// This implementation **can** have duplicate entries if the distance of the
    /// two duplicates is not the same, this is due to how the search point is found.
    pub fn checked_push(&mut self, dist: f32, idx: usize) -> bool {
        // Element does not meet the minimum
        if dist >= self.furthest().dist {
            return false;
        }

        let idx = idx as u32;

        let res = self.neighbors.binary_search_by(|neighbor| {
            if neighbor.idx == idx {
                Ordering::Equal
            } else if neighbor.dist < dist {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        });

        // Our starting index to begin swapping values to insert.
        let start = match res {
            Ok(_) => return false, // Element already exists
            Err(start) => start,
        };

        let p = Point {
            idx,
            dist,
            flag: false,
        };
        self.sift_right(p, start);

        true
    }

    #[inline]
    /// Attempts to insert a new point into the heap.
    ///
    /// This does not check if the point already exists.
    ///
    /// Returns if the value was inserted or not.
    ///
    /// NOTE:
    /// This implementation **can** have duplicate entries if the distance of the
    /// two duplicates is not the same, this is due to how the search point is found.
    pub fn unchecked_push(&mut self, dist: f32, idx: usize) -> bool {
        // Element does not meet the minimum
        if dist >= self.furthest().dist {
            return false;
        }

        let idx = idx as u32;

        let res = self.neighbors.binary_search_by(|neighbor| {
            if neighbor.dist < dist {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        });

        // Our starting index to begin swapping values to insert.
        let start = match res {
            Ok(_) => return false, // Element already exists
            Err(start) => start,
        };

        let p = Point {
            idx,
            dist,
            flag: false,
        };
        self.sift_right(p, start);

        true
    }
}

impl Debug for SortedNeighbors {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "SortedNeighbors([",)?;

        for point in self.iter_neighbors() {
            write!(f, "{point:?}, ")?;
        }

        write!(f, "])")
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Point {
    pub(crate) idx: u32,
    pub(crate) dist: f32,
    pub(crate) flag: bool,
}

impl Default for Point {
    fn default() -> Self {
        Self {
            idx: u32::MAX,
            dist: f32::INFINITY,
            flag: false,
        }
    }
}

impl Point {
    #[inline]
    pub fn dist(&self) -> f32 {
        self.dist
    }

    #[inline]
    pub fn idx(&self) -> u32 {
        self.idx
    }

    #[inline]
    pub fn flag(&self) -> bool {
        self.flag
    }

    #[inline]
    pub fn set_flag(&mut self, flag: bool) {
        self.flag = flag;
    }
}

impl PartialOrd for Point {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Point {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.idx == other.idx {
            Ordering::Equal
        } else {
            self.dist.partial_cmp(&other.dist).unwrap()
        }
    }
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx
    }
}

impl Eq for Point {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checked_flagged_push() {
        let mut heap = SortedNeighbors::new(1);
        heap.checked_flagged_push(1.0, 0, true);
        assert_eq!(heap.neighbors[0].idx, 0);
        assert_eq!(heap.neighbors[0].dist, 1.0);
        assert!(heap.neighbors[0].flag);

        let mut heap = SortedNeighbors::new(10);
        for i in 0..9 {
            heap.checked_flagged_push(1.0, i, true);
        }
        assert_eq!(heap.furthest().idx, u32::MAX);
        assert_eq!(heap.furthest().dist, f32::INFINITY);
        assert!(!heap.furthest().flag);

        heap.checked_flagged_push(2.0, 10, false);
        assert_eq!(heap.furthest().idx, 10);
        assert_eq!(heap.furthest().dist, 2.0);
        assert!(!heap.furthest().flag);
    }

    #[test]
    fn test_checked_push() {
        let mut heap = SortedNeighbors::new(1);
        heap.checked_push(1.0, 0);
        assert_eq!(heap.neighbors[0].idx, 0);
        assert_eq!(heap.neighbors[0].dist, 1.0);

        let mut heap = SortedNeighbors::new(10);
        for i in 0..9 {
            heap.checked_push(1.0, i);
        }
        assert_eq!(heap.furthest().idx, u32::MAX);
        assert_eq!(heap.furthest().dist, f32::INFINITY);

        heap.checked_push(2.0, 10);
        assert_eq!(heap.furthest().idx, 10);
        assert_eq!(heap.furthest().dist, 2.0);
    }
}
