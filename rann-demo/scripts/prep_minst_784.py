import numpy as np
from safetensors.numpy import save_file

from fetch import get_ann_benchmark_data

hdf5_file = get_ann_benchmark_data("mnist-784-euclidean")
test = np.array(hdf5_file["test"])
train = np.array(hdf5_file["train"])

test = np.pad(test, ((0, 0), (0, 1024 - test.shape[1])), "constant")
train = np.pad(train, ((0, 0), (0, 1024 - train.shape[1])), "constant")

print(test.shape, train.shape)
print(test)
print(train)

tensors = {
    "test": test,
    "train": train,
}

save_file(tensors, "../datasets/minst-784-euclidean.safetensors")

