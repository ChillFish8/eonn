import numpy as np
import time
from usearch.index import Index

from fetch import get_ann_benchmark_data

hdf5_file = get_ann_benchmark_data("gist-960-euclidean")
test = np.array(hdf5_file["test"])
train = np.array(hdf5_file["train"])

start = time.perf_counter()
index = Index(ndim=960)
index.add(range(train.shape[0]), train, threads=1, log=True)
elapsed = time.perf_counter() - start
print(f"Took: {elapsed:.2f}s")