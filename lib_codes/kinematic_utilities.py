#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2026-01-23 16:42:36
# @Author  : Felipe G. Ortega-Gama (felipeortegagama@gmail.com)
# @Version : 1.0
# General kinematic functions

import numpy as np

# -----------------
# Define constants
# -----------------
I = complex(0.0,1.0)

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



# Calculate the square of a Minkowski 4-momentum (+---) metric
def square_4vec(quadvec):
    # Can receive a vector of vectors
    # Must receive temporal, but can receive one, two or three spatial
    
    quadvec = np.array(quadvec)
    
    if len(quadvec.shape) == 1:
        return (quadvec[0])**2 - sum(quadvec[1:]**2)
    else:
        sqquadvec = quadvec**2
    
        return sqquadvec[0,:] - np.sum(sqquadvec[1:,:],axis=0)