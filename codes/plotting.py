#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-17 02:52:38
# @Author  : Felipe Ortega (felipeortegagama@gmail.com)
# @Version : 1.0


import numpy as np
import matplotlib.pyplot as plt
import sys
import csv


from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib import colors, ticker, cm


# -----------------
# Config file:
# config_file = str(sys.argv[1])

# with open(config_file, 'r') as f:
#     for line in f:
#         if line[:] == ' ':


sum_folder = '../felipe_results/Sum_vals/'

IA_folder = '../felipe_results/IA_vals/'

IN_folder = '../felipe_results/IN_vals/'

read_folder = '../felipe_results/Readable_vals/'


indices = [[0], [1,0], [1,0]]

if 0:
    indices = [[], [0,0], [0,0]]


# Kinematics
m1 = 1.0
m2 = 1.0

lab_moment_i = np.array([0,0,1])
lab_moment_f = np.array([0,0,1])

#Get sum values
sumfilename = sum_folder + 'Sum_sig_[' + str(indices[0]) +';' +str(indices[1]) +';' +str(indices[2]) +(
']_vecPi_' + str(int(lab_moment_i[2])) + '_vecPf_' + str(int(lab_moment_f[2])) + '.txt')

with open(sumfilename, 'r') as f:

    Eistars, Efstars, Sum = np.load(f)


#Get IN values
INfilename = IN_folder + 'IN_sig_[' + str(indices[0]) +';' +str(indices[1]) +';' +str(indices[2]) +(
']_vecPi_' + str(int(lab_moment_i[2])) + '_vecPf_' + str(int(lab_moment_f[2])) + '.txt')

with open(INfilename, 'r') as f:

    EistarN, EfstarN, INN = np.load(f)

#Get IA values
IAfilename = IA_folder + 'IA_sig_[' + str(indices[0]) +';' +str(indices[1]) +';' +str(indices[2]) +(
']_vecPi_' + str(int(lab_moment_i[2])) + '_vecPf_' + str(int(lab_moment_f[2])) + '.txt')

with open(IAfilename, 'r') as f:

    EistarA, EfstarA, IAA = np.load(f)


# Check that the energy values are all the same

shape = (np.shape(Eistars) == np.shape(EistarN)) * (np.shape(Eistars) == np.shape(EistarA))

minimi = (Eistars[0,0]==EistarA[0,0]) * (Eistars[0,0]==EistarN[0,0])

maximi = (Eistars[-1,-1]==EistarA[-1,-1]) * (Eistars[-1,-1]==EistarN[-1,-1])

minimf = (Efstars[0,0]==EfstarA[0,0]) * (Efstars[0,0]==EfstarN[0,0])

maximf = (Efstars[-1,-1]==EfstarA[-1,-1]) * (Efstars[-1,-1]==EfstarN[-1,-1])


if shape * minimi * maximi * minimf * maximf == 0:
    raise ValueError('They are not the same shape :(')



# Initial frame variables

q2star_i = 0.25 * (Eistars**2 - 2*(m1**2 + m2**2) + (m1**2 - m2**2)**2/Eistars**2) * np.complex(1.,0)

# Final frame variables

q2star_f = 0.25 * (Efstars**2 - 2*(m1**2 + m2**2) + (m1**2 - m2**2)**2/Efstars**2) * np.complex(1.,0)


[lf, mf] = indices[1]
[li, mi] = indices[2]



RGfunction = np.real( 1./(q2star_i**(li/2.)) * 1./(q2star_f**(lf/2.)) * (Sum - IAA - INN))


# Save the data to plot with Mathematica

if 0:
    with open("data.txt", 'w') as f:
        flatEf = np.ndarray.flatten(Efstars)
        flatG = np.ndarray.flatten(RGfunction)
        for n, Ei in enumerate(np.ndarray.flatten(Eistars)):
            f.write("{ " + str(Ei) +", "+ str(flatEf[n]) + ", " +str(flatG[n]) + " },\n")

# Save readable data of the sums and integrals
readblefilename =  read_folder + 'Read_sig_[' + str(indices[0]) +';' +str(indices[1]) +';' +str(indices[2]) +(
']_vecPi_' + str(int(lab_moment_i[2])) + '_vecPf_' + str(int(lab_moment_f[2])) + '.csv')

if 1:
    with open(readblefilename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        flatq2i = np.ndarray.flatten(q2star_i**(li/2.))
        flatq2f =  np.ndarray.flatten(q2star_f**(lf/2.))
        flatEf = np.ndarray.flatten(Efstars)
        flatS = np.ndarray.flatten(Sum)
        flatIA = np.ndarray.flatten(IAA)
        flatIN = np.ndarray.flatten(INN)
        flatG = np.ndarray.flatten(RGfunction)
        writer.writerow(['Eist', 'q2i^(l/2)', 'Efst', 'q2f^(l/2)', 'Sum', 'IA', 'IN', 'ReG'])
        for n, Ei in enumerate(np.ndarray.flatten(Eistars)):
            writer.writerow([Ei, flatq2i[n], flatEf[n], flatq2f[n], flatS[n], flatIA[n], flatIN[n], flatG[n]])


# print Sum[0,0] * 1./(q2star_i**(li/2.)) * 1./(q2star_f**(lf/2.))
# print INN[0,0] * 1./(q2star_i**(li/2.)) * 1./(q2star_f**(lf/2.))
# print IAA[0,0] * 1./(q2star_i**(li/2.)) * 1./(q2star_f**(lf/2.))
# print RGfunction[0,0]

# GG = []
# ee = []
# for ii in xrange(10):
#     ee.append( (Eistars[ii, ii + 1] + Eistars[ii, ii])/2. )
#     ee.append( (Eistars[ii + 1, ii + 1] + Eistars[ii, ii + 1])/2. )

#     GG.append( (RGfunction[ii, ii + 1] + RGfunction[ii, ii])/2. )
#     GG.append( (RGfunction[ii + 1, ii + 1] + RGfunction[ii, ii + 1])/2. )
# print ee
# print GG

# plt.plot(ee, GG)

# plt.plot(Efstars[:,2],RGfunction[:,2])
# print RGfunction

if 0:

    fig, ax = plt.subplots()

    levs = np.linspace(-.25,.25,num=31)
    cs = ax.contourf(Efstars, Eistars, RGfunction,levs,cmap=plt.cm.seismic)
    cbar = fig.colorbar(cs)

    plt.show()

