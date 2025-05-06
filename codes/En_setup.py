#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-18 19:54:23
# @Author  : Felipe Ortega (felipeortegagama@gmail.com)
# @Version : 1.00


import numpy as np
import os
import sys


# -----------------
# Config file:
# -----------------
config_file = str(sys.argv[1])

# Ener file:
# -----------------
ener_file = str(sys.argv[2])


cval = dict()

with open(config_file, 'r') as f:
    for line in f.read().splitlines():

        if len(line) == 0: # skip empty lines
            continue

        if line[0] == '#': # skip comments
            continue

        spaceat = line.find(' ')
        key = line[0:spaceat]
        commentat = line.find('#')

        if commentat== -1: # no comment
            value = line[spaceat+1:]

        else:   #when there is a comment
            value = line[spaceat+1:commentat-1]

        cval[key] = value


if cval['shape'] == 'mesh':
    # Shift Efstar by hand so that IN does not go into error
    shift = int(cval['shift'])

    evals = int(cval['evals'])

    if shift:
        dE = (float(cval['Eimax']) - float(cval['Eimin']))/(evals - 1)
        cval['Efmin'] = float(cval['Eimin']) - 0.5 * dE
        cval['Efmax'] = float(cval['Eimax']) - 0.5 * dE


    Eistar = np.linspace(float(cval['Eimin']), float(cval['Eimax']), num=evals)
     
    Efstar = np.linspace(float(cval['Efmin']), float(cval['Efmax']), num=evals)

    msgg = np.meshgrid(Eistar, Efstar)

    with open(ener_file, 'wb') as f:

        np.save(f,np.array([msgg[0],msgg[1]]))

elif cval['shape'] == 'line':
    # Shift Efstar so that IN does not go into error
    def Efstar(Eistar, par):
        return par[0] * Eistar + par[1]

    evals = int(cval['evals'])

    
    Eistar = np.linspace(float(cval['Eimin']), float(cval['Eimax']), num=evals)
    pp = [float(cval['slop']), float(cval['intercpt'])]
    Efstar = Efstar(Eistar, pp)
    
    
    with open(ener_file, 'wb') as f:

        np.save(f,np.array([Eistar,Efstar]))

elif cval['shape'] == 'lineEf':
    # Shift Efstar so that IN does not go into error
    def Eistar(Efstar, par):
        return par[0] * Efstar + par[1]

    evals = int(cval['evals'])

    
    Efstar = np.linspace(float(cval['Efmin']), float(cval['Efmax']), num=evals)
    pp = [float(cval['slop']), float(cval['intercpt'])]
    Eistar = Eistar(Efstar, pp)
    
    
    with open(ener_file, 'wb') as f:

        np.save(f,np.array([Eistar,Efstar]))

else:
    print("Nothing was done")