use std::collections::BinaryHeap;

use bitvec::vec::BitVec;
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
}


/// A BinaryHeap-like structure maintaining a fixed size.
pub struct SortedNeighbors {
    priorities: SmallVec<[f32; 32]>,
    indices: SmallVec<[u32; 32]>,
    flags: BitVec,
}

impl SortedNeighbors {
    /// Creates a new sorted neighbor implementation with a given size.
    pub fn new(size: usize) -> Self {
        let mut priorities = SmallVec::with_capacity(size);
        let mut indices = SmallVec::with_capacity(size);
        let mut flags = BitVec::with_capacity(size);
        
        for _ in 0..size {
            priorities.push(f32::INFINITY);
            indices.push(u32::MAX);
            flags.push(false);
        }
        
        Self {
            priorities,
            indices,
            flags
        }
    }
    
    #[inline]
    /// Attempts to insert a new point into the heap.
    ///
    /// This checks if the point already exists.
    /// 
    /// Returns if the value was inserted or not.
    pub fn checked_flagged_heap_push(&mut self, p: f32, n: usize, f: bool) -> bool {
        if p >= self.priorities[0] {
            return false; 
        }
        
        let n = n as u32;
        
        if self.indices.binary_search(&n).is_ok() {
            return false;
        }
        
        self.priorities[0] = p;
        self.indices[0] = n;
        self.flags.set(0, f);

        let i = self.sort_heap(p);
        
        self.priorities[i] = p;
        self.indices[i] = n;
        self.flags.set(i, f);
        
        true
    }

    #[inline]
    /// Attempts to insert a new point into the heap.
    ///
    /// This checks if the point already exists.
    /// 
    /// Returns if the value was inserted or not.
    pub fn checked_heap_push(&mut self, p: f32, n: usize) -> bool {
        if p >= self.priorities[0] {
            return false;
        }

        let n = n as u32;

        if self.indices.binary_search(&n).is_ok() {
            return false;
        }

        self.priorities[0] = p;
        self.indices[0] = n;

        let i = self.sort_heap(p);
        
        self.priorities[i] = p;
        self.indices[i] = n;

        true
    }

    #[inline]
    /// Attempts to insert a new point into the heap.
    /// 
    /// This does not check if the point already exists.
    ///
    /// Returns if the value was inserted or not.
    pub fn unchecked_heap_push(&mut self, p: f32, n: usize) -> bool {
        if p >= self.priorities[0] {
            return false;
        }

        let n = n as u32;

        self.priorities[0] = p;
        self.indices[0] = n;

        let i = self.sort_heap(p);

        self.priorities[i] = p;
        self.indices[i] = n;

        true
    }

    #[inline]
    fn sort_heap(&mut self, p: f32) -> usize {
        let size = self.priorities.len();
        
        let mut i = 0;
        loop {
            let ic1 = 2 * i + 1;
            let ic2 = ic1 + 1;
            let i_swap;

            if ic1 >= size {
                break
            } else if ic2 >= size {
                if self.priorities[ic1] <= p {
                    break;
                }
                i_swap = ic1;
            } else if self.priorities[ic1] >= self.priorities[ic2] {
                if p < self.priorities[ic1] {
                    i_swap = ic1;
                } else {
                    break
                }
            } else if p < self.priorities[ic2] {
                i_swap = ic2;
            } else {
                break
            }

            i = i_swap;
        }
        
        i
    }
}