import numpy as np

import os
os.getcwd()

import sys
sys.path.append("codes")

import os_lqft
import lorentz_transformations as lt

Etemp = np.load("codes/Etemp.npy")
config = os_lqft.read_configs("codes/config_files/config_lorentz_test.txt")

# Will start working with the line format of data and will later generalize to
# the mesh format

Eicms = Etemp[0]
Efcms = Etemp[1]

Pivec = np.array([int(component) for component in config["Pi"]])
Pfvec = np.array([int(component) for component in config["Pf"]])
L = int(config["cube_num"])

Pivec = 2*np.pi*Pivec / L
Pfvec = 2*np.pi*Pfvec / L

Eis = np.sqrt(Eicms**2 + Pivec @ Pivec)
Efs = np.sqrt(Efcms**2 + Pfvec @ Pfvec)

for transition in range(len(Eis)):
    Pi = np.array([Eis[transition], *Pivec])
    Pf = np.array([Efs[transition], *Pfvec])
    L = lt.lorentz_transformation_2(Pi, Pf)

    Pinew = L @ Pi
    Pfnew = L @ Pf
