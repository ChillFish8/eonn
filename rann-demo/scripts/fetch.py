import h5py
from urllib.request import urlretrieve
import os


def get_ann_benchmark_data(dataset_name):
    path = f"../datasets/{dataset_name}.hdf5"
    if not os.path.exists(path):
        print(f"Dataset {dataset_name} is not cached; downloading now ...")
        urlretrieve(f"http://ann-benchmarks.com/{dataset_name}.hdf5", path)
    return h5py.File(path, "r")
