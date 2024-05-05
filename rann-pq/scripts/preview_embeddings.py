import numpy as np
from safetensors.numpy import load_file, save_file


tensors = load_file("../datasets/train-embeddings.safetensors")

embeddings = tensors["embeddings"].reshape((-1, 1024))
max_ = embeddings.max(axis=0)
min_ = embeddings.min(axis=0)
delta = max_ - min_
step = delta / 255
print(f"Max: {max_}", f"Min: {min_}", f"Delta: {delta}", f"StepSize: {step}", sep="\n")

quantized = np.uint8(embeddings / step)
print(quantized)


