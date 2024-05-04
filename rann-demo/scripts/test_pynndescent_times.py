import time

import pynndescent
import numpy as np

from fetch import get_ann_benchmark_data

hdf5_file = get_ann_benchmark_data("gist-960-euclidean")
test = np.array(hdf5_file["test"])
train = np.array(hdf5_file["train"])


start = time.perf_counter()
index = pynndescent.NNDescent(train, verbose=True, n_jobs=2)
elapsed = time.perf_counter() - start
print(f"Took: {elapsed:.2f}s")
