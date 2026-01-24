#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-17 00:07:51
# @Author  : Felipe Ortega (felipeortegagama@gmail.com)
# @Version : 1.0

import sys
import numpy as np
# -----------------
# Energies file:
energ_file = str(sys.argv[1])

with open(energ_file, 'rb') as f:

    eners = np.load(f)

Eistar = eners[0]
 
Efstar = eners[1]

print('Eistar', 'Efstar')

for ei, ef in zip(Eistar, Efstar):
    print(ei, ef)

