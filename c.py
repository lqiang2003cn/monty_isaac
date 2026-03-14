import numpy as np
d = np.load("docker_all/data/monty/scans/yellow_block/depth_float/000063.npy")
# d.shape == (720, 1280), dtype float32, values in metres
# d == 0 means no depth data for that pixel
print(d.min())
print(d.max())
print(d.mean())
print(d.std())
print(d.var())
print(d.min())
print(d.max())
print(d.mean())
print(d.std())
print(d.var())