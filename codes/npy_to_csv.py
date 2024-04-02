#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-20 20:11:30
# @Author  : Felipe Ortega (felipeortegagama@gmail.com)
# @Version : 1.0


import numpy as np
import sys
import csv

fun_filename = str(sys.argv[1])
readblefilename = str(sys.argv[2])

# Energies file:
energ_file = str(sys.argv[3])

# Energies from file
with open(energ_file, 'rb') as f:

    eners = np.load(f)

Eistar = eners[0]
 
Efstar = eners[1]


# Get function value
with open(fun_filename, 'rb') as f:
    fun = np.load(f)

Rf = np.real(fun)
If = np.imag(fun)


with open(readblefilename, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    flatEi = np.ndarray.flatten(Eistar)
    flatR_fun = np.ndarray.flatten(Rf)
    flatI_fun = np.ndarray.flatten(If)

    writer.writerow(['Efst', 'Efst', 'Re_fun', 'Im_fun'])
    for n, Ef in enumerate(np.ndarray.flatten(Efstar)):
        writer.writerow([Ef, flatEi[n], 
            flatR_fun[n], flatI_fun[n]])


