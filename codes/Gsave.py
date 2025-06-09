#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-20 20:11:30
# @Author  : Felipe Ortega (felipeortegagama@gmail.com)
# @Version : 1.0


import numpy as np
import os
import sys
import csv

# -----------------
# Config file:
config_file = str(sys.argv[1])

# -----------------
# Energies file:
energ_file = str(sys.argv[2])

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


# Folders for input/output
sum_folder = cval['folder'] + 'Sum_vals/'

IA_folder = cval['folder'] + 'IA_vals/'

IN_folder = cval['folder'] + 'IN_vals/'

G_folder = cval['folder'] + 'G_vals/'

if not os.path.exists(G_folder):
    os.makedirs(G_folder)

# Define Lattice parameters
# -----------------
L = float(cval['L_inv_mass'])  # in terms of the mass

m1 = float(cval['m1'])
m2 = float(cval['m2'])

# Get the names
# G-indexing
indices = []
for ll in cval['indices'].split():
    nesttemplist = []
    if ll != 'n':
        for char in list(ll):
            nesttemplist.append(int(char))

    indices.append(nesttemplist)

# Kinematics
lab_moment_i_int = [int(ll) for ll in cval['Pi'].split()]
lab_moment_i = (2*np.pi/L)*np.array(lab_moment_i_int)

lab_moment_f_int = [int(ll) for ll in cval['Pf'].split()]
lab_moment_f = (2*np.pi/L)*np.array(lab_moment_f_int)

# Energies from file
with open(energ_file, 'rb') as f:

    eners = np.load(f)

Eistar = eners[0]
 
Efstar = eners[1]


#Get sum values
sumfilename = sum_folder + 'Sum_sig_[' + str(indices[0]) +';' +str(indices[1]) +';' +str(indices[2]) +(
']_vecPi_' + str(lab_moment_i_int[2]) + '_vecPf_' + str(lab_moment_f_int[2])  + '_L_' + str(int(L)) + '.npy')

with open(sumfilename, 'rb') as f:

    Sum = np.load(f)


#Get IN values
INfilename = IN_folder + 'IN_sig_[' + str(indices[0]) +';' +str(indices[1]) +';' +str(indices[2]) +(
']_vecPi_' + str(lab_moment_i_int[2]) + '_vecPf_' + str(lab_moment_f_int[2]) + '_L_' + str(int(L)) + '.npy')

with open(INfilename, 'rb') as f:

    INN = np.load(f)

#Get IA values
IAfilename = IA_folder + 'IA_sig_[' + str(indices[0]) +';' +str(indices[1]) +';' +str(indices[2]) +(
']_vecPi_' + str(lab_moment_i_int[2]) + '_vecPf_' + str(lab_moment_f_int[2]) + '_L_' + str(int(L)) + '.npy')

with open(IAfilename, 'rb') as f:

    IAA = np.load(f)


# Kinematic factors

[lf, mf] = indices[1]
[li, mi] = indices[2]

# Initial frame variables

q2star_i = 0.25 * (Eistar**2 - 2*(m1**2 + m2**2) + (m1**2 - m2**2)**2/Eistar**2) * complex(1.,0)

# Final frame variables

q2star_f = 0.25 * (Efstar**2 - 2*(m1**2 + m2**2) + (m1**2 - m2**2)**2/Efstar**2) * complex(1.,0)


# Calculate the G Function!!

RGfunction = np.real( 1./(q2star_i**(li/2.)) * 1./(q2star_f**(lf/2.)) * (Sum - IAA - INN))

IGfunction = np.imag( 1./(q2star_i**(li/2.)) * 1./(q2star_f**(lf/2.)) * (Sum - IAA - INN))


# Save numpy array binaries
binaryfilename =  G_folder + 'Bin_sig_[' + str(indices[0]) +';' +str(indices[1]) +';' +str(indices[2]) +(
']_vecPi_' + str(lab_moment_i_int[2]) + '_vecPf_' + str(lab_moment_f_int[2]) + '_L_' + str(int(L)) + '.npz')

with open(binaryfilename, 'wb') as f:
    np.savez_compressed(f,Eistar=Eistar, Efstar=Efstar,
        qstli= q2star_i**(li/2.), qstlf= q2star_f**(lf/2.),
        Sum = Sum, IA = IAA, IN = INN,
        ReG = RGfunction, ImG = IGfunction)


# Save readable data of the sums and integrals
readblefilename =  G_folder + 'Read_sig_[' + str(indices[0]) +';' +str(indices[1]) +';' +str(indices[2]) +(
']_vecPi_' + str(lab_moment_i_int[2]) + '_vecPf_' + str(lab_moment_f_int[2]) + '_L_' + str(int(L)) + '.csv')


with open(readblefilename, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    flatq2i = np.ndarray.flatten(q2star_i**(li/2.))
    flatq2f =  np.ndarray.flatten(q2star_f**(lf/2.))
    flatEf = np.ndarray.flatten(Efstar)
    flatReS = np.ndarray.flatten(np.real(Sum))
    flatImS = np.ndarray.flatten(np.imag(Sum))
    flatReIA = np.ndarray.flatten(np.real(IAA))
    flatImIA = np.ndarray.flatten(np.imag(IAA))
    flatReIN = np.ndarray.flatten(np.real(INN))
    flatImIN = np.ndarray.flatten(np.imag(INN))
    flatRG = np.ndarray.flatten(RGfunction)
    flatIG = np.ndarray.flatten(IGfunction)
    writer.writerow(['#Eist', 'q2i^(l/2)', 'Efst', 'q2f^(l/2)', 'ReG', 'ImG', 'ReSum', 'ImSum', 'ReIA', 'ImIA', 'ReIN', 'ImIN'])
    for n, Ei in enumerate(np.ndarray.flatten(Eistar)):
        writer.writerow([Ei, flatq2i[n], flatEf[n], flatq2f[n], 
            flatRG[n], flatIG[n],
            flatReS[n], flatImS[n], flatReIA[n], flatImIA[n], flatReIN[n], flatImIN[n]])



