#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-05 19:24:28
# @Author  : Felipe Ortega (felipeortegagama@gmail.com)
# @Version : 1.0


# Calculate I_N for several values of Ei^st/ Ef^st

import numpy as np
import sys
import os

from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt

# -----------------
# Config file:
# -----------------
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


results_folder = cval['folder']

IN_folder = results_folder + 'IN_vals/'


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

m1t = m1
m2t = m2


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

# UV regulators
ccslist = cval['ccs'].split()
ccs = [float(ll)/float(ccslist[-1]) for ll in ccslist[:-1]]

LAMBDA_param_1 = [int(ll) * m1 for ll in cval['LAMB'].split()]


if len(LAMBDA_param_1) > 0:
    #This is latter multiplied by 10
    cutoff = LAMBDA_param_1[-1]
else:
    #This is latter multiplied by 10
    cutoff = m1

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

# This was shown to be always the case
axial = 1
if axial:
    if len(indices[0])>0 and indices[0][0] == 3: #change from z=3 to z=1 since only two indices
        indices[0][0] = 1


# -----------------
# Useful functions
# -----------------

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

# This is zero at the k-value of the canceling pole of the Dr term
def canceling_pole_Dr_root(k_dum, k_dum_phi, k_dum_th, P_i, P_f, mass1):

    # Works with temporal plus only z-spatial
    if len(P_i)==2 and len(P_i)==2:
        #P_{i}k
        Pik2 = P_i[1]**2 - 2 * np.abs(P_i[1]) * k_dum * np.cos(k_dum_th) + k_dum**2
        omega_Pik1 = np.sqrt(Pik2 + m1**2)

        #P_{f}k
        Pfk2 = P_f[1]**2 - 2 * np.abs(P_f[1]) * k_dum * np.cos(k_dum_th) + k_dum**2
        omega_Pfk1 = np.sqrt(Pfk2 + m1**2)

    else:

        k_3vect = np.array([k_dum * np.sin(k_dum_th) * np.cos(k_dum_phi),
                            k_dum * np.sin(k_dum_th) * np.sin(k_dum_phi),
                            k_dum * np.cos(k_dum_th)])

        Pik = P_i[1:4] - k_3vect
        
        omega_Pik1 = np.sqrt(np.dot(Pik,Pik) + mass1**2)


        Pfk = P_f[1:4] - k_3vect
        
        omega_Pfk1 = np.sqrt(np.dot(Pfk,Pfk) + mass1**2)

    return (P_i[0] + omega_Pik1) - (P_f[0] + omega_Pfk1)

# This creates a linspace with some of its points removed
def discont_linspace(interval, remov_points, ls_lenght, dist = 2):
    
    # Check if the interval range makes sense
    if len(remov_points) * 2 * dist > (interval[1] - interval[0]):
        raise ValueError('Not enough space in here!')
    
    # Organize the points to create the intervals in order
    remov_points = sorted(remov_points, reverse = True)
    
    joint_flag = np.zeros(len(remov_points))
    
    # Check for jointed points
    for nn in xrange(len(remov_points)-1):
        # Remember of the reverse sort
        if remov_points[nn] - remov_points[nn + 1] < 2 * dist:
            joint_flag[nn] = 1 
            # this flag means point nn is joint with point nn + 1
    nn = 0
    
    while nn < len(remov_points):
        
        point = remov_points[nn]
        
        # Check if point is close to the upper boundaries
        if point > interval[-1] - dist:
            
            interval[-1] = point - dist
            
            # And all the joined points
            while joint_flag[nn]:
                interval[-1] = remov_points[nn + 1] - dist
                nn += 1
            
        elif point > interval[0] + dist:
            
            
            templist = [interval[0]]
            initialbound = point - dist
            finalbound = point + dist
            
            # And all the joined points
            while joint_flag[nn]:
                
                initialbound = remov_points[nn + 1] - dist
                nn += 1
            
            templist.extend([initialbound, finalbound])
            
            templist.extend(interval[1:])
            
            interval = templist
            
        else:
            
            interval[0] = point + dist
            
            break
            # Since the rest of the points will be now ignored
        
        nn += 1

    # Finally create the linspaces
    
    num_linspaces = len(interval)/2
    
    
    linspace = np.array([])
    
    for inters in xrange(num_linspaces):
        
        prop = float(interval[2*inters + 1] - interval[2*inters])/(interval[-1] - interval[0])
        
        linspace = np.concatenate(
            (linspace,
            np.linspace(interval[2*inters], interval[2*inters + 1], num = int(prop*ls_lenght))
            ))
            
    return linspace


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

# Calculate the corresponding velocity of a 4-momentum
def beta_4vector(p4vector):
    return np.sqrt(np.dot(p4vector[1:4],p4vector[1:4]))/p4vector[0]

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
        x = vector[0,:]
        y = vector[1,:]
        z = vector[2,:]
     
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

# 4pi normalized solutions of the Laplacian (only axial (up to l=1)) , i.e. sqrt(4pi) * k^l * Y_{lm}(\hat{k})
def lap_sol_a(l, m, vector):
    # Can receive a vector of vectors
    # Only one value of (l,m) at the time
    # Receive azimutal m and polar l numbers
    # z component only, for axial integration
    
    vector = np.array(vector)
    
    if len(vector) == 1:
        [z] = vector
    else:
        z = vector
     
    #r2 = x**2 + y**2 + z**2

    if m>l:
        raise NameError('Azimutal number m bigger than l')
    if l>1:
        raise NameError('This is the axial version, only up to l=1')
    
    if l==0:
        sph_dict = 1.0
        
    elif l==1:
        if m==0:
            sph_dict = np.sqrt(3.) * z
            
        else:
            NameError('Not implemented yet...')
    
    else:
        raise NameError('Not implemented yet...')
    
    return sph_dict
# -----------------
# Define Integrands
# -----------------

# -----------------
# Axial Integrands
# -----------------

# k-magnitude integrand
def Integrand_IN_a(k_dum, k_dum_th, P_i, P_f, alpha, index, region):
    # k_dum is the radial integration variable, let's make it a vector
    # k_dum_th is the azimutal angle, let's leave the vectorization for later
    # k_dum_phi is the polar, but I'll erase it since it's really a 2D integral

    islist = len(np.array(k_dum).shape)

    if islist:
        k_dum = np.array(k_dum)
        k2_dum = k_dum**2
        k_len = len(k_dum)
    else:
        k_dum = np.array([k_dum])
        k2_dum = k_dum**2
        k_len = 1

    # P_i and P_f are the kinematic variables, this code will only work for
    # them to be only in the z-direction
    if len(P_i) > 2 or len(P_f) > 2:
        raise ValueError('This is only functional for true axial integration')


    # alpha is the dimensionful UV regulator parameter

    # index is a list, 
    # first element a list of the 4-vector indices (only temporal or z-spatial)
    # second element the value of lf, mf
    # third element the value of li, mi

    # region small makes full evaluation
    # region large makes the evaluation faster, use when the exponentials are expected to be negligible


    k_dum = np.array(k_dum)
    k2_dum = k_dum**2

    integrand = 0

    # Use these globally to be able to adapt the Pauli Vilars subtractions
    global m1
    global m2


    # Extract kinematic variables

    # Initial frame variables
    Ei = P_i[0]

    Pivec = np.abs(P_i[1])

    Eicm = np.sqrt(square_4vec(P_i))

    Lambdai = boost(P_i)
    # qstar doesn't inherit the m/Lambda dependence (only used in cutoff)
    q2star_i = 0.25 * (Eicm**2 - 2*(m1**2 + m2**2) + (m1**2 - m2**2)**2/Eicm**2) 
    


    # Final frame variables

    Ef = P_f[0]

    Pfvec = np.abs(P_f[1])

    Efcm = np.sqrt(square_4vec(P_f)) 

    Lambdaf = boost(P_f)
    # qstar doesn't inherit the m/Lambda dependence (only used in cutoff)
    q2star_f = 0.25 * (Efcm**2 - 2*(m1**2 + m2**2) + (m1**2 - m2**2)**2/Efcm**2)

    
    # Cut-off function (independent of UV reg)
    omega_k1 = np.sqrt(k2_dum + m1**2)
    omega_k2 = np.sqrt(k2_dum + m2**2)

    # Only two interesting components
    k_dum_4vect = np.array([omega_k2,
                            k_dum * np.cos(k_dum_th)])

    kzstar_i = np.dot(Lambdai, k_dum_4vect)[1,:]
    kzstar_f = np.dot(Lambdaf, k_dum_4vect)[1,:]

    if region == 'small':
        # Other components are summarized in here
        k2perp = np.sin(k_dum_th)**2 * k2_dum

        k2star_i = k2perp + kzstar_i**2

        k2star_f = k2perp + kzstar_f**2

        HH = np.exp(- alpha * (k2star_i - q2star_i) * (k2star_f - q2star_f))

    # Evaluate each term in the sum
    # ccs is defined in the "preamble"
    for nn in xrange(len(ccs)):# UV convergence parts
        
        # Use Lambda (UV) or the mass
        if nn > 0:

            m1 = LAMBDA_param_1[nn - 1]
            m2 = LAMBDA_param_1[nn - 1]
            
            # Have to re calculate since UV dependence
            omega_k1 = np.sqrt(k2_dum + m1**2)
            omega_k2 = np.sqrt(k2_dum + m2**2)

            k_dum_4vect[0] = omega_k2

            kzstar_i = np.dot(Lambdai, k_dum_4vect)[1,:]
            kzstar_f = np.dot(Lambdaf, k_dum_4vect)[1,:]

        else:
            m1 = m1t
            m2 = m2t

        # Calculate the most used shorthands
        
        #P_{i}k
        Pik2 = Pivec**2 - 2 * Pivec * k_dum * np.cos(k_dum_th) + k2_dum
        omega_Pik1 = np.sqrt(Pik2 + m1**2)

        #P_{f}k
        Pfk2 = Pfvec**2 - 2 * Pfvec * k_dum * np.cos(k_dum_th) + k2_dum
        omega_Pfk1 = np.sqrt(Pfk2 + m1**2)

        # Drf denominator
        Drf = 1./(2 * omega_Pfk1) * (
                1./((Ef + omega_Pfk1)**2 - omega_k2**2) ) * (
                1./((Ei - Ef - omega_Pfk1)**2 - omega_Pik1**2) )
        
        #Drf numerator terms 
        Kf_4vect = np.concatenate(([Ef + omega_Pfk1], [k_dum_4vect[1,:]]))
        
        Kf_4vect_i = np.dot(Lambdai, Kf_4vect)[1,:]
        Kf_4vect_f = np.dot(Lambdaf, Kf_4vect)[1,:]

        # Dri denominator
        Dri = 1./(2 * omega_Pik1) * (
                1./((Ei + omega_Pik1)**2 - omega_k2**2) ) * (
                1./((Ef - Ei - omega_Pik1)**2 - omega_Pfk1**2) )      
        
        #Dri numerator terms 
        Ki_4vect = np.concatenate(([Ei + omega_Pik1], [k_dum_4vect[1,:]]))
        
        Ki_4vect_i = np.dot(Lambdai, Ki_4vect)[1,:]
        Ki_4vect_f = np.dot(Lambdaf, Ki_4vect)[1,:]
        
        # DD denominator
        DD = 1./(2 * omega_k2) * (
            1./((Ef - omega_k2)**2 - omega_Pfk1**2)) * (
            1./((Ei - omega_k2)**2 - omega_Pik1**2))
        
        # Get the info of the index
        #Spherical harmonics
        [lf, mf] = index[1]
        [li, mi] = index[2]
        
        if li==0 and lf==0:
            vector_coeff_D = 1.
            
            vector_coeff_f = 1.
            
            vector_coeff_i = 1.
        
        elif li>1 or lf>1:
            raise NameError('This works only for axial integrals, try to use index gymnastics ;)')

        else:
            vector_coeff_D = np.conj(lap_sol_a(li, mi, kzstar_i)) * lap_sol_a(lf, mf, kzstar_f)

            vector_coeff_f = np.conj(lap_sol_a(li, mi, Kf_4vect_i)) * lap_sol_a(lf, mf, Kf_4vect_f)

            vector_coeff_i = np.conj(lap_sol_a(li, mi, Ki_4vect_i)) * lap_sol_a(lf, mf, Ki_4vect_f)
        
        
        #Lorentz vectors
        for ind in index[0]:

            vector_coeff_D *= k_dum_4vect[ind]

            vector_coeff_f *= Kf_4vect[ind]

            vector_coeff_i *= Ki_4vect[ind]
        
        
        for term in [1, 2, 3]: # Individual smooth integrals
            
            # Term 2 doesn't have a nn = 0
            if term == 2 and nn ==0:
                continue
            
            # No 2, 3 term for I_N simplified (large k)
            if region == 'large' and term > 1:
                continue

            if term == 1:
                    
                if region == 'small':
                    
                    integrand += ccs[nn] * (vector_coeff_D * DD  +
                    (Drf * vector_coeff_f + Dri * vector_coeff_i)) * (HH - 1)

                # When in the large region simplify I_N
                elif region == 'large':
                    temp = (-1) * ccs[nn] * (vector_coeff_D * DD  +
                    (Drf * vector_coeff_f + Dri * vector_coeff_i))

                    integrand += temp
                          
                        
            elif term == 2:

                integrand += - ccs[nn] *  vector_coeff_D * (DD) * HH


            elif term == 3:

                integrand += - ccs[nn] * (Drf * vector_coeff_f  + Dri * vector_coeff_i) * HH

    
    # Make sure to return the original values to m1 and m2
    m1 = m1t
    m2 = m2t
    
    # Remember the k^2 from the integral measure
    return k2_dum * integrand

# Azimutal integrand
def Ang_Integrand_IN_a(k_dum_th, P_i, P_f, alpha, index):

    # print 'th', k_dum_th
    # quad integrate up to int_upperbound, then fit and analytics

    # integrate up up to Lambda^2/k^2 ~ 0.01
    # By this point also the exponential factors should be negligible
    int_upperbound = 10 * cutoff
    
    if np.exp(-alpha * int_upperbound**4) > 1e-10:
        int_upperbound = (10./alpha * np.log(10))**(.25)

    
    # Do a fit for the tail
    points_for_fit = 100
    size_of_fit = 100
    
   # Avoid the dr_poles
    def checkdr(mass1, kmin, kmax):
        # It is axial symmetric so
        k_dum_phi = 0
        return (canceling_pole_Dr_root(kmin, k_dum_phi, k_dum_th, P_i, P_f, mass1)*
                canceling_pole_Dr_root(kmax, k_dum_phi, k_dum_th, P_i, P_f, mass1))

    
    # pole of the m terms
    checkm = checkdr(m1, int_upperbound, int_upperbound + size_of_fit)

    #pole of the Lambda terms
    checkL = []
    for LAMBDAS in LAMBDA_param_1:
        checkL.append( checkdr(LAMBDAS, int_upperbound, int_upperbound + size_of_fit) )
    
    
    # NOW remove points if they are poles 
    removepoints = []
    if checkm < 0:
        # It is axial symmetric so
        k_dum_phi = 0
        zerom = optimize.brentq(canceling_pole_Dr_root, 
                   int_upperbound, int_upperbound + size_of_fit, 
                   args= (k_dum_phi, k_dum_th, P_i, P_f, m1))
        removepoints.append(zerom)    

    for ii, LAMBDAS in enumerate(LAMBDA_param_1):
        if checkL[ii] < 0:
            # It is axial symmetric so
            k_dum_phi = 0

            zeroL = optimize.brentq(canceling_pole_Dr_root, 
                       int_upperbound, int_upperbound + size_of_fit, 
                       args= (k_dum_phi, k_dum_th, P_i, P_f, LAMBDAS))
            removepoints.append(zeroL)
            
    if len(removepoints) == 0:
        # No poles: it is safe to do the fit
        # create k for smooth data
        kk = np.linspace(int_upperbound, int_upperbound + size_of_fit, num = points_for_fit)

    else:
        # create k for NON-smooth data
        kk = discont_linspace([int_upperbound, int_upperbound + size_of_fit], 
                         removepoints, points_for_fit, dist = 2)
    
       
    # Create the fit function of the tail
    def fit_func(k_mag, aa, bb):
        return aa/k_mag**bb       
    
    # It accepts vectors
    # Kinematic axial parameters
    gg = Integrand_IN_a(kk, k_dum_th, [P_i[0],P_i[3]], [P_f[0],P_f[3]], alpha, index, 'large')
    try:
        popt, pcov = optimize.curve_fit(fit_func, kk, gg)
    except:
        print 'ERROR HERE', k_dum_th, [P_i[0],P_i[3]], [P_f[0],P_f[3]]
        print '##########'
        print '##########'
        popt = [0.,2.]

    # Analytical value of the tail
    f_th = popt[0]/((popt[1]-1) * int_upperbound**(popt[1]-1))

    #FIT DONE, DO THE REST

    # Main contribution (hope I don't hit any dr pole)
    temp = integrate.quad(Integrand_IN_a, 0, int_upperbound, epsrel = 1e-4,
        args = (k_dum_th, [P_i[0],P_i[3]], [P_f[0],P_f[3]], alpha, index, 'small'))[0]

    f_th += temp

    # Remember that integration measure has a sine
    return np.sin(k_dum_th) * f_th


# -----------------
# Non-Axial Integrands
# -----------------

# k-magnitude integrand
def Integrand_IN(k_dum, k_dum_th, k_dum_phi, P_i, P_f, alpha, index, region):
    # k_dum is the radial integration variable, let's make it a vector
    # k_dum_th is the azimutal angle, let's leave the vectorization for later
    # k_dum_phi is the polar angle

    islist = len(np.array(k_dum).shape)

    if islist:
        k_dum = np.array(k_dum)
        k2_dum = k_dum**2
        k_len = len(k_dum)
    else:
        k_dum = np.array([k_dum])
        k2_dum = k_dum**2
        k_len = 1

    

    # P_i and P_f are the kinematic variables
    # alpha is the dimensionful UV regulator parameter

    # index is a list, 
    # first element a list of the 4-vector indices (only temporal or z-spatial)
    # second element the value of lf, mf
    # third element the value of li, mi

    # region small makes full evaluation
    # region large makes the evaluation faster, use when the exponentials are expected to be negligible

    integrand = 0

    # Use these globally to be able to adapt the IN code
    global m1
    global m2


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

    
    # Cut-off function (independent of UV reg)

    omega_k2 = np.sqrt(k2_dum + m2**2)

    k_dum_4vect = np.array([omega_k2, 
                            k_dum * np.sin(k_dum_th) * np.cos(k_dum_phi),
                            k_dum * np.sin(k_dum_th) * np.sin(k_dum_phi),
                            k_dum * np.cos(k_dum_th)])

    kstar_i = np.dot(Lambdai,k_dum_4vect)[1:4,:]
    

    kstar_f = np.dot(Lambdaf, k_dum_4vect)[1:4,:]
    

    if region == 'small':

        k2star_i = np.sum(kstar_i**2, axis=0)
        k2star_f = np.sum(kstar_f**2, axis=0)

        HH = np.exp(- alpha * (k2star_i - q2star_i) * (k2star_f - q2star_f))

    # Evaluate each term in the sum
    # ccs is defined in the "preamble"
    for nn in xrange(len(ccs)):# UV convergence parts
        
        # Use Lambda (UV) or the mass
        if nn > 0:

            m1 = LAMBDA_param_1[nn - 1]
            m2 = LAMBDA_param_1[nn - 1]
            
            # Have to re calculate since UV dependence
            omega_k2 = np.sqrt(k2_dum + m2**2)

            k_dum_4vect[0,:] = omega_k2

            kstar_i = np.dot(Lambdai, k_dum_4vect)[1:4,:]
            kstar_f = np.dot(Lambdaf, k_dum_4vect)[1:4,:]

        else:
            m1 = m1t
            m2 = m2t

        # Calculate the most used shorthands
        
        #P_{i}k
        Pik = np.repeat(np.transpose([P_i[1:4]]), k_len, axis=1) - k_dum_4vect[1:4,:]
        Pik2 = np.sum(Pik**2, axis=0)
        omega_Pik1 = np.sqrt(Pik2 + m1**2)

        #P_{f}k
        Pfk = np.repeat(np.transpose([P_f[1:4]]), k_len, axis=1) - k_dum_4vect[1:4,:]
        Pfk2 = np.sum(Pfk**2, axis=0)
        omega_Pfk1 = np.sqrt(Pfk2 + m1**2)

        # Drf denominator
        Drf = 1./(2 * omega_Pfk1) * (
                1./((Ef + omega_Pfk1)**2 - omega_k2**2) ) * (
                1./((Ei - Ef - omega_Pfk1)**2 - omega_Pik1**2) )
        
        #Drf numerator terms 
        Kf_4vect = np.concatenate(([Ef + omega_Pfk1], k_dum_4vect[1:4,:]))
        
        Kf_4vect_i = np.dot(Lambdai, Kf_4vect)[1:4,:]
        Kf_4vect_f = np.dot(Lambdaf, Kf_4vect)[1:4,:]

        # Dri denominator
        Dri = 1./(2 * omega_Pik1) * (
                1./((Ei + omega_Pik1)**2 - omega_k2**2) ) * (
                1./((Ef - Ei - omega_Pik1)**2 - omega_Pfk1**2) )      
        
        #Dri numerator terms 
        Ki_4vect = np.concatenate(([Ei + omega_Pik1], k_dum_4vect[1:4,:]))
        
        Ki_4vect_i = np.dot(Lambdai, Ki_4vect)[1:4,:]
        Ki_4vect_f = np.dot(Lambdaf, Ki_4vect)[1:4,:]
        
        # DD denominator
        PP_f = np.repeat(np.transpose([P_f]), k_len, axis=1)
        PP_i = np.repeat(np.transpose([P_i]), k_len, axis=1)
        DD = 1./(2 * omega_k2) * (
            1./(square_4vec(PP_f - k_dum_4vect) - m1**2)) * (
            1./(square_4vec(PP_i - k_dum_4vect) - m1**2))  
        
        # Get the info of the index
        #Spherical harmonics
        [lf, mf] = index[1]
        [li, mi] = index[2]
        
        if li==0 and lf==0:
            vector_coeff_D = 1
            
            vector_coeff_f = 1
            
            vector_coeff_i = 1
            
        else:
            vector_coeff_D = np.conj(lap_sol(li, mi, kstar_i)) * lap_sol(lf, mf, kstar_f)

            vector_coeff_f = np.conj(lap_sol(li, mi, Kf_4vect_i)) * lap_sol(lf, mf, Kf_4vect_f)

            vector_coeff_i = np.conj(lap_sol(li, mi, Ki_4vect_i)) * lap_sol(lf, mf, Ki_4vect_f)
        
            #print 'lap_sol', li, mi, kstar_i
        
        #Lorentz vectors
        for ind in index[0]:

            vector_coeff_D *= k_dum_4vect[ind,:]

            vector_coeff_f *= Kf_4vect[ind,:]

            vector_coeff_i *= Ki_4vect[ind,:]
        
        
        for term in [1, 2, 3]: # Individual smooth integrals
            
            # Term 2 doesn't have a nn = 0
            if term == 2 and nn ==0:
                continue
            
            # No 2, 3 term for I_N simplified (large k)
            if region == 'large' and term > 1:
                continue

            if term == 1:
                    
                if region == 'small':
                    
                    integrand += ccs[nn] * (vector_coeff_D * DD  +
                    (Drf * vector_coeff_f + Dri * vector_coeff_i)) * (HH - 1)

                # When in the large region simplify I_N
                elif region == 'large':
                    temp = (-1) * ccs[nn] * (vector_coeff_D * DD  +
                    (Drf * vector_coeff_f + Dri * vector_coeff_i))

                    integrand += temp                        
                        
            elif term == 2:

                integrand += - ccs[nn] *  vector_coeff_D * (DD) * HH


            elif term == 3:

                integrand += - ccs[nn] * (Drf * vector_coeff_f  + Dri * vector_coeff_i) * HH

    
    # Make sure to return the original values to m1 and m2
    m1 = m1t
    m2 = m2t
    
    # Remember the k^2 from the integral measure
    return k2_dum * integrand


def Ang_Integrand_IN(k_dum_phi, k_dum_th, P_i, P_f, alpha, index):

    # print 'th', k_dum_th
    # quad integrate up to int_upperbound, then fit and analytics

    # integrate up up to Lambda^2/k^2 ~ 0.01
    # By this point also the exponential factors should be negligible
    int_upperbound = 10 * cutoff
    
    if np.exp(-alpha * int_upperbound**4) > 1e-10:
        int_upperbound = (10./alpha * np.log(10))**(.25)

    
    # Do a fit for the tail
    points_for_fit = 100
    size_of_fit = 100
    
   # Avoid the dr_poles
    def checkdr(mass1, kmin, kmax):
        return (canceling_pole_Dr_root(kmin, k_dum_phi, k_dum_th, P_i, P_f, mass1)*
                canceling_pole_Dr_root(kmax, k_dum_phi, k_dum_th, P_i, P_f, mass1))

    
    # pole of the m terms
    checkm = checkdr(m1, int_upperbound, int_upperbound + size_of_fit)

    #pole of the Lambda terms
    checkL = []
    for LAMBDAS in LAMBDA_param_1:
        checkL.append( checkdr(LAMBDAS, int_upperbound, int_upperbound + size_of_fit) )
    
    
    # NOW remove points if they are poles 
    removepoints = []
    if checkm < 0:
        zerom = optimize.brentq(canceling_pole_Dr_root, 
                   int_upperbound, int_upperbound + size_of_fit, 
                   args= (k_dum_phi, k_dum_th, P_i, P_f, m1))
        removepoints.append(zerom)    

    for ii, LAMBDAS in enumerate(LAMBDA_param_1):
        if checkL[ii] < 0:
            zeroL = optimize.brentq(canceling_pole_Dr_root, 
                       int_upperbound, int_upperbound + size_of_fit, 
                       args= (k_dum_phi, k_dum_th, P_i, P_f, LAMBDAS))
            removepoints.append(zeroL)
            
    if len(removepoints) == 0:
        # No poles: it is safe to do the fit
        # create k for smooth data
        kk = np.linspace(int_upperbound, int_upperbound + size_of_fit, num = points_for_fit)

    else:
        # create k for NON-smooth data
        kk = discont_linspace([int_upperbound, int_upperbound + size_of_fit], 
                         removepoints, points_for_fit, dist = 2)
    
       
    # Create the fit function of the tail
    def fit_func(k_mag, aa, bb):
        return aa/k_mag**bb       
    
    # It accepts vectors     
    gg = Integrand_IN(kk, k_dum_th, k_dum_phi, P_i, P_f, alpha, index, 'large')
    popt, pcov = optimize.curve_fit(fit_func, kk, gg)

    # Analytical value of the tail
    f_th_phi = popt[0]/((popt[1]-1) * int_upperbound**(popt[1]-1))

    #FIT DONE, DO THE REST

    # Main contribution (hope I don't hit any dr pole)
    temp = integrate.quad(Integrand_IN, 0, int_upperbound, epsrel = 1e-4,
        args = (k_dum_th, k_dum_phi, P_i, P_f, alpha, index, 'small'))[0]

    f_th_phi += temp

    # Remember that integration measure has a sine
    return np.sin(k_dum_th) * f_th_phi

# -----------------
# General macro to do the I_N integral
# -----------------

# It has two methods,
# adap_quad (adpative gauss quadrature) 
# fix_qaud (fixed guass quadrature) 
# I think fix_quad is faster (and avoids hitting poles by mistake), both should work in general


# index is a list, 
# first element a list of the 4-vector indices (only temporal or z-spatial)
# second element the value of lf, mf
# third element the value of li, mi

def make_int(P_i, P_f, alpha, index, method):

    # Here we focus on the double angular integral
    # The radial integral is done for every evualtion needed for the ang integration

    integral = 0
    
    
    if method == 'adap_quad':
    
        #option where phi integral is not done (only for axial symmetric cases)
        if axial:
            # Remember the 2pi from the phi integral
            val_temp = (2 * np.pi) * integrate.quad(Ang_Integrand_IN_a, 0, np.pi, epsrel = 1e-4,
                args = (P_i, P_f, alpha, index), full_output=0)

            integral = 1./(2 * np.pi)**3 * val_temp[0]

        else:
            val_temp = 1./(2 * np.pi)**3 * integrate.dblquad(Ang_Integrand_IN, 0, np.pi, lambda th: 0, lambda th: 2*np.pi,
                epsrel = 1e-4, args = (P_i, P_f, alpha, index))[0]

            integral = val_temp
            
    if method == 'fix_quad':
        
        #option where phi integral is not done (only for axial symmetric cases)
        if axial:
            
            tolerance = 1e-4
            error = 1
            fix_ord = 10

            def Ang_Integrand_IN_nonphi(k_dum_th, P_i, P_f, alpha, index):

                reslts = [Ang_Integrand_IN_a( ths, P_i, P_f, alpha, index)
                         for ths in k_dum_th]
                
                # Remember the 2pi from the phi integral
                return 2 * np.pi * np.array(reslts)

            
            val_temp_old = integrate.fixed_quad(Ang_Integrand_IN_nonphi, 0, np.pi,
                args = (P_i, P_f, alpha, index), n = fix_ord)[0]
            
            print val_temp_old

            fix_ord += 1
            while error > tolerance:
                
                val_temp_new = integrate.fixed_quad(Ang_Integrand_IN_nonphi, 0, np.pi,
                    args = (P_i, P_f, alpha, index), n = fix_ord)[0]
                
                error = np.abs(val_temp_new - val_temp_old)/np.abs(val_temp_old)
                
                val_temp_old = val_temp_new
                
                print val_temp_old
                
                fix_ord += 1
                
                if fix_ord > 10:
                    print 'order bigger 10, E:', error
                    
                    if fix_ord > 15:
                        print 'order 15, E:', error
                        break
                        

            integral = 1./(2 * np.pi)**3 * val_temp_new
                

        else:
            raise ValueError('not implemented')


    return integral

#CALCULATE STUFF
print 'IN alpha ',alpha
ener_shape = np.shape(Eistar)
I_Nn = np.ones(ener_shape) * np.complex(0.,0.)

ang_integral_method = 'fix_quad'

if len(ener_shape) > 1: # for mesh inputs
    for mm, enirow in enumerate(Ei):
        for nn, enin in enumerate(enirow):
            enfin = Ef[mm,nn]
            PP_i = np.concatenate(([enin], lab_moment_i))
            PP_f = np.concatenate(([enfin], lab_moment_f))
            
            I_Nn[mm,nn] = make_int(PP_i, PP_f, alpha, indices, ang_integral_method)
            print 'INcalc: ', Eistar[mm,nn], Efstar[mm,nn], '---------', I_Nn[mm,nn]
else:
    for mm, enin in enumerate(Ei):
        enfin = Ef[mm]
        PP_i = np.concatenate(([enin], lab_moment_i))
        PP_f = np.concatenate(([enfin], lab_moment_f))
        
        I_Nn[mm] = make_int(PP_i, PP_f, alpha, indices, ang_integral_method)
        print 'INcalc: ', Eistar[mm], Efstar[mm], '---------', I_Nn[mm]


if axial:
    if len(indices[0])>0 and indices[0][0] == 1: #change from z=3 to z=1 since only two indices
        indices[0][0] = 3

#Save the values
filename = IN_folder + 'IN_sig_[' + str(indices[0]) +';' +str(indices[1]) +';' +str(indices[2]) +(
']_vecPi_' + str(lab_moment_i_int[2]) + '_vecPf_' + str(lab_moment_f_int[2]) + '_L_' + str(int(L)) + '.npy')

if not os.path.exists(IN_folder):
    os.makedirs(IN_folder)

msgg = np.meshgrid(Eistar, Efstar)

with open(filename, 'w') as f:

    np.save(f,I_Nn)


