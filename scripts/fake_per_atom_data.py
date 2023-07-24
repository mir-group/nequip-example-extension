import os

os.system("mkdir benchmark_data")
os.system(
    'cd benchmark_data; wget "http://quantum-machine.org/gdml/data/npz/aspirin_ccsd.zip"'
)
os.system("cd benchmark_data; unzip -o aspirin_ccsd.zip")

import numpy as np

d = dict(np.load("benchmark_data/aspirin_ccsd-train.npz"))
force_magnitudes = np.linalg.norm(d["F"], axis=-1)
force_magnitudes = (
    np.random.normal(loc=0.0, scale=0.05, size=force_magnitudes.shape)
    + 0.5678 * force_magnitudes
)
d["x"] = force_magnitudes.reshape((d["F"].shape[0], d["F"].shape[1], 1))
np.savez("benchmark_data/aspirin_ccsd-train-fakedata.npz", **d)
