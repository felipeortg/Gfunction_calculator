#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-17 00:07:51
# @Author  : Felipe Ortega (felipeortegagama@gmail.com)
# @Version : 1.0


import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle

from scipy import integrate
from scipy import optimize
from scipy import special

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


cube_num = int(cval['cube_num'])

results_folder = cval['folder']

sum_folder = results_folder + 'Sum_vals/'

# -----------------
# Define constants
# -----------------
I = complex(0.0,1.0)

# -----------------
# Define Lattice parameters
# -----------------
L = float(cval['L_inv_mass'])  # in terms of the mass

m1 = float(cval['m1'])
m2 = float(cval['m2'])

# This is dimensionful, the dimensionless quantity should be less than 1
# UV regulator
alpha = float(cval['alpha']) # exponential suppression

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
with open(energ_file, 'r') as f:

    eners = np.load(f)

Eistar = eners[0]
 
Efstar = eners[1]

Ei = np.sqrt(Eistar**2 + np.dot(lab_moment_i, lab_moment_i))
Ef = np.sqrt(Efstar**2 + np.dot(lab_moment_f, lab_moment_f))


# -----------------
# Get triplets and form k_array
# -----------------

# Location to get triplets from:
trip_folder = './'

# -----------------
# Get the array of n triplets
filename = trip_folder + 'triplets/n_list_r<' + str(cube_num) + '.txt'
f = open(filename, 'r')
n_list = pickle.load(f)
n_arr = np.array(n_list)

# Get array of magnitude k vectors 
k_arr = (2*np.pi/L) * n_arr

k2_array = np.sum(k_arr*k_arr, axis = 1)
    
k_len = len(k2_array)


# -----------------
# Define some kinematic common used functions
# -----------------

# Define boost matrix
# Where boost(p4vector).p4vector = (E^*, vec(0))
def boost(p4vector):
    # Can receive a general 4 vector
    # Or also case when only two non-zero components
    if len(p4vector) == 2:
        # velocity = vec(p)/E
        beta = p4vector[1]/(1.0*p4vector[0])

        if beta==0:
            return np.identity(2)

        gamma = 1/np.sqrt(1-beta**2)

        return np.array([
        [gamma, -gamma*beta],
        [-gamma*beta, gamma]])

    else:

        # speed = |vec(p)|/E
        beta = np.sqrt(np.dot(p4vector[1:4],p4vector[1:4]))/p4vector[0]

        # no speed
        if beta==0:
            return np.identity(4)

        gamma = 1/np.sqrt(1-beta**2)
        norm = p4vector[1:4]/(beta*p4vector[0]) # normed velocity

        resul= np.array([
            [gamma, -gamma*norm[0]*beta, -gamma*norm[1]*beta, -gamma*norm[2]*beta],
            [-gamma*norm[0]*beta, 1+(gamma-1)*norm[0]**2, (gamma-1)*norm[0]*norm[1], (gamma-1)*norm[0]*norm[2]],
            [-gamma*norm[1]*beta, (gamma-1)*norm[0]*norm[1], 1+(gamma-1)*norm[1]**2, (gamma-1)*norm[1]*norm[2]],
            [-gamma*norm[2]*beta, (gamma-1)*norm[0]*norm[2], (gamma-1)*norm[1]*norm[2], 1+(gamma-1)*norm[2]**2]])

        return resul

# 4pi normalized solutions of the Laplacian , i.e. sqrt(4pi) * k^l * Y_{lm}(\hat{k})
def lap_sol(l, m, vector):
    # Can receive a vector of vectors
    # Only one value of (l,m) at the time
    # Receive azimutal m and polar l numbers
    # x,y,z components of the vector
    # r^2 squared magnitude of the vector
    
    vector = np.array(vector)
    
    if len(vector.shape) == 1:
        [x, y, z] = vector
    else:
        x = vector[:,0]
        y = vector[:,1]
        z = vector[:,2]
     
    #r2 = x**2 + y**2 + z**2

    if m>l:
        raise NameError('Azimutal number m bigger than l')
    
    #r2_sph = r2 + (r2==0) # Last part to avoid division by zero

    if l==0:
        sph_dict = 1.0
        
    elif l==1:
        if m==-1:
            sph_dict = np.sqrt(1.5) * (x - I*y)
            
        elif m==0:
            sph_dict = np.sqrt(3.) * z
            
        else:
            sph_dict = -np.sqrt(1.5) * (x + I*y)
    
    elif l==2:
        if m==-2:
            sph_dict = 0.5 * np.sqrt(7.5) * (x - I*y)**2
            
        elif m==-1:
            sph_dict = np.sqrt(7.5)* z * (x - I*y)
            
        elif m==0:
            sph_dict = 0.5 * np.sqrt(5.) * (2*z**2 - x**2 - y**2)
            
        elif m==1:
            sph_dict = -np.sqrt(7.5) * z * (x + I*y)
            
        else:
            sph_dict = 0.5 * np.sqrt(7.5) * (x + I*y)**2,
    
    else:
        raise NameError('Not implemented yet...')
    
    return sph_dict



#-----
# Do the sum



# -----------------
# General macro to do the Sum Pf \neq Pi
# -----------------

# index is a list, 
# first element a list of the 4-vector indices (only temporal or z-spatial)
# second element the value of lf, mf
# third element the value of li, mi

def Neq_sum(P_i, P_f, alpha, index):

    # Extract some variables

    # Initial frame variables
    Ei = P_i[0]

    Pivec = np.sqrt(sum([P_i[ii]**2 for ii in xrange(1,4)]))

    Eicm = np.sqrt(Ei**2 - Pivec**2)

    Lambdai = boost(P_i)
    # qstar doesn't inherit the m/Lambda dependence (only used in cutoff)
    q2star_i = 0.25 * (Eicm**2 - 2*(m1**2 + m2**2) + (m1**2 - m2**2)**2/Eicm**2) 



    # Final frame variables

    Ef = P_f[0]

    Pfvec = np.sqrt(sum([P_f[ii]**2 for ii in xrange(1,4)]))

    Lambdaf = boost(P_f)

    Efcm = np.sqrt(Ef**2 - Pfvec**2) 
    # qstar doesn't inherit the m/Lambda dependence (only used in cutoff)
    q2star_f = 0.25 * (Efcm**2 - 2*(m1**2 + m2**2) + (m1**2 - m2**2)**2/Efcm**2)


    # -----
    # Cut-off function 
    omega_k2 = np.array([np.sqrt( k2_array + m2**2 )])

    # Join energy and momentum, transpose omega row vector to column
    k4vectors = np.concatenate((omega_k2.T, k_arr), axis=1)

    # Transpose to do matrix multiplication, then transpose back
    kstar_i = (np.dot(Lambdai,k4vectors.T).T)[:,1:4]
    
    kstar_f = (np.dot(Lambdaf,k4vectors.T).T)[:,1:4]
    

    k2star_i = np.sum(kstar_i**2, axis=1)
    k2star_f = np.sum(kstar_f**2, axis=1)

    HH = np.exp(- alpha * (k2star_i - q2star_i) * (k2star_f - q2star_f))


    # Get the info of the index
    vector_coeff = 1

    #Lorentz vectors
    for ind in index[0]:

        vector_coeff *= k4vectors[:,ind]



    #Spherical harmonics
    [lf, mf] = index[1]
    [li, mi] = index[2]

    sphff = lap_sol(lf, mf, kstar_f)
    sphii = np.conj(lap_sol(li, mi, kstar_i))


    # Calculate the most used shorthands
    
    #P_{i}k
    Pik = np.repeat(np.transpose([P_i[1:4]]), k_len, axis=1).T - k4vectors[:,1:4]
    Pik2 = np.sum(Pik**2, axis=1)
    omega_Pik1 = np.sqrt(Pik2 + m1**2)

    #P_{f}k
    Pfk = np.repeat(np.transpose([P_f[1:4]]), k_len, axis=1).T - k4vectors[:,1:4]
    Pfk2 = np.sum(Pfk**2, axis=1)
    omega_Pfk1 = np.sqrt(Pfk2 + m1**2)

    # DD denominator
    DD = 1./(2 * omega_k2) * (
        1./((Ef - omega_k2)**2 - omega_Pfk1**2)) * (
        1./((Ei - omega_k2)**2 - omega_Pik1**2))

    
    return (1./L**3) * np.sum(HH * sphff * DD * vector_coeff * sphii)


#CALCULATE STUFF
print 'Sum alpha ',alpha
ener_shape = np.shape(Eistar)
Summ = np.ones(ener_shape) * np.complex(0.,0.)

if len(ener_shape) > 1: # for mesh inputs
    for mm, enirow in enumerate(Ei):
        for nn, enin in enumerate(enirow):
            enfin = Ef[mm,nn]
            PP_i = np.concatenate(([enin], lab_moment_i))
            PP_f = np.concatenate(([enfin], lab_moment_f))
           
            Summ[mm,nn] = Neq_sum(PP_i, PP_f, alpha, indices)
            print 'Sum: ', Eistar[mm,nn], Efstar[mm,nn], '---------', Summ[mm,nn]
else:
    for mm, enin in enumerate(Ei):
        enfin = Ef[mm]
        PP_i = np.concatenate(([enin], lab_moment_i))
        PP_f = np.concatenate(([enfin], lab_moment_f))

        Summ[mm] = Neq_sum(PP_i, PP_f, alpha, indices)
        print 'Sum: ', Eistar[mm], Efstar[mm], '---------', Summ[mm]



#Save the values
filename = sum_folder + 'Sum_sig_[' + str(indices[0]) +';' +str(indices[1]) +';' +str(indices[2]) +(
']_vecPi_' + str(lab_moment_i_int[2]) + '_vecPf_' + str(lab_moment_f_int[2])  + '_L_' + str(int(L)) + '.npy')

if not os.path.exists(sum_folder):
    os.makedirs(sum_folder)

msgg = np.meshgrid(Eistar, Efstar)

with open(filename, 'w') as f:

    np.save(f,Summ)



