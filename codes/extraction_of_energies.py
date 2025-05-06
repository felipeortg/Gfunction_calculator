import os

import jk
import os_lqft

import numpy as np

energies_path = "../recalc_Ecm_JackFiles"
paths = os.listdir(energies_path)

for path in paths:
    with open(os.path.join(energies_path, path), "r") as f:
        f.readline()
        data = np.array([float(line.split(" ")[2]) for line in f.readlines()])
        os_lqft.save("results.h5/energies/0")
