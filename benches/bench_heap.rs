use std::hint::black_box;
use criterion::{Criterion, criterion_group, criterion_main};
use rann::graph::SortedNeighbors;


fn benchmark_heap(c: &mut Criterion) {
    fastrand::seed(2352356346463346);
    
    for n in [10, 32, 64] {
        c.bench_function(&format!("checked_flagged_heap_push {n} neighbors"), |b| {
            let mut inserts = Vec::new();
            for _ in 0..1000 {
                inserts.push((fastrand::f32(), fastrand::usize(0..n), fastrand::bool()));
            }
            
            b.iter(|| {
                let mut heap = SortedNeighbors::new(n);
                for (p, n, f) in inserts.iter() {
                    black_box(heap.checked_flagged_heap_push(
                        black_box(*p),
                        black_box(*n),
                        black_box(*f),
                    ));
                }
            })
        });
    }
    
    for n in [10, 32, 64] {
        c.bench_function(&format!("checked_heap_push {n} neighbors"), |b| {
            let mut inserts = Vec::new();
            for _ in 0..1000 {
                inserts.push((fastrand::f32(), fastrand::usize(0..n)));
            }

            b.iter(|| {
                let mut heap = SortedNeighbors::new(n);
                for (p, n) in inserts.iter() {
                    black_box(heap.checked_heap_push(
                        black_box(*p),
                        black_box(*n),
                    ));
                }
            })
        });
    }
    
    for n in [10, 32, 64] {
        c.bench_function(&format!("unchecked_heap_push {n} neighbors"), |b| {
            let mut inserts = Vec::new();
            for _ in 0..1000 {
                inserts.push((fastrand::f32(), fastrand::usize(0..n)));
            }

            b.iter(|| {
                let mut heap = SortedNeighbors::new(n);
                for (p, n) in inserts.iter() {
                    black_box(heap.unchecked_heap_push(
                        black_box(*p),
                        black_box(*n),
                    ));
                }
            })
        });
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = benchmark_heap,
);
criterion_main!(benches);