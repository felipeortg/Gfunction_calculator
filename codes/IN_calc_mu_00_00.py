#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-05 19:24:28
# @Author  : Felipe Ortega (felipeortegagama@gmail.com)
# @Version : 1.0


# Calculate I_N for several values of Ei^st/ Ef^st
# Specialize in the vector

# We will also use provide point where Drf, Dri are singular
## quad: point for singularity
## fixed_quad: multiple integrals, singularities at boundaries


# The idea is to have two integrals:
# Have the angular integration last, the integrand performs the magnitude evaluation within

# boost and rotate to the initial frame, with the spatial part of Pf pointing in the z direction

test=False
if test:
    theta_vals = []
    to_plot = dict()
    integral_kradial = []
    plot_upperbound = 0


import numpy as np
import sys
import os

from scipy import integrate
from scipy import optimize
from scipy.spatial.transform import Rotation
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

# testing for different mass....
# alpha = float(cval['alpha']) / m1**4# exponential suppression


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
    #This is later multiplied by 10
    cutoff = LAMBDA_param_1[-1]
else:
    #This is later multiplied by 10
    cutoff = m1

# Kinematics
lab_moment_i_int = [int(ll) for ll in cval['Pi'].split()]
lab_moment_i = (2*np.pi/L)*np.array(lab_moment_i_int)

lab_moment_f_int = [int(ll) for ll in cval['Pf'].split()]
lab_moment_f = (2*np.pi/L)*np.array(lab_moment_f_int)

# -----------------
# Result file:
result_file = f'IN_sig_{indices[0]};{indices[1]};{indices[2]}_vecPf_{lab_moment_f_int}_vecPi_{lab_moment_i_int}.npy'
filename = IN_folder + result_file

# Energies from file
with open(energ_file, 'rb') as f:

    eners = np.load(f)

Eistar = eners[0]
 
Efstar = eners[1]

# print("Eistar", Eistar)
# print("Efstar", Efstar)

Ei = np.sqrt(Eistar**2 + np.dot(lab_moment_i, lab_moment_i))
Ef = np.sqrt(Efstar**2 + np.dot(lab_moment_f, lab_moment_f))

# # This was shown to be always the case
# axial = 1


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

# utility functions to employ identity IN(Pi, Pf) = L_{-bi} IN(Eist, Pfi)
def get_cm_gamma(PP_i, PP_f):
    Eicm = np.sqrt(PP_i[0]**2 - np.dot(PP_i[1:],PP_i[1:]))
    Efcm = np.sqrt(PP_f[0]**2 - np.dot(PP_f[1:],PP_f[1:]))

    gamma = (PP_i[0]*PP_f[0] - np.dot(PP_i[1:],PP_f[1:]))/(Eicm*Efcm)

    return Eicm, Efcm, gamma



# transformation to boost to Pi frame, and rotate spatial part of Pf to z-axis
def boost_rotation(PP_i, PP_f):
    Lambdai = boost(PP_i)

    PP_i_st = np.dot(Lambdai, PP_i)

    #check that the boost matrix works
    if np.abs(np.dot(PP_i_st[1:], PP_i_st[1:])) > 1e-10: #reasonable tolerance...
        print(Lambdai)
        print(PP_i, PP_i_st)
        raise ValueError("Something wrong with the boost to initial frame")

    # Get Pf in the initial frame
    PP_fi_st = np.dot(Lambdai, PP_f)

    Pfi_st = np.sqrt(np.dot(PP_fi_st[1:], PP_fi_st[1:]))

    # Co-moving Pf and Pi, no rotation 
    if Pfi_st == 0:
        return np.linalg.inv(Lambdai)

    # Extract spatial Pfi_st direction
    th = np.acos(PP_fi_st[3]/Pfi_st)
    phi = np.atan2(PP_fi_st[2], PP_fi_st[1])


    # Calculate rot matrix from euler angles: R(phi, -th, -phi)
    # Note: function from euler does rotations in the order of the inputs
    rot = np.zeros((4,4))
    rot[0,0]=1
    rot[1:,1:] = Rotation.from_euler('zyz', [-phi,-th,phi]).as_matrix()

    PP_fi_st_rot = rot @ PP_fi_st

    # Check that rotation placed us in z-axis
    if np.abs(PP_fi_st_rot[3] - Pfi_st) > 1e-10:
        print(rot)
        print(PP_fi_st, PP_fi_st_rot)
        raise ValueError("Something wrong with the rotation to Pf to z-axis")

    
    # print(np.linalg.inv(rot @ Lambdai))
    return np.linalg.inv(rot @ Lambdai)


def get_rotboost_indices(index, PP_i, PP_f):
    """
    Obtain the coefficients and indices of the linear combinations of IN needed
    receive the original index, PP_i, PP_f
    return two lists
    coeffs
    indices
    """

    [lf, mf] = index[1]
    [li, mi] = index[2] 

    if li*mi*lf*mf != 0:
        raise ValueError("Only implemented for S-wave")

    #scalar case, nothing to do
    if len(index[0]) == 0:
        return [1], [index]

    #vector case, one boost/rotation
    elif len(index[0]) == 1:
        br = boost_rotation(PP_i, PP_f)
        ix = index[0][0]

        coeffs = []
        list_indices = []
        
        coeffs.append(br[index[0][0], 0])
        list_indices.append([[0],index[1],index[2]])

        # in reality doing z (third) component, but write index as 1
        coeffs.append(br[index[0][0], 3])
        list_indices.append([[1],index[1],index[2]])

        return coeffs, list_indices

    else:
        raise ValueError("Only implemented up to vector")

# DELETE?
def check_valid_Dr_pole(k_dum, k_dum_z, Ei, Ef, Pfimag, mass1, tol=1e-10):

    if k_dum < 0:
        return 0

    omega_k1 = np.sqrt(k_dum**2 + mass1**2)
    omega_Pfk1 = np.sqrt(k_dum**2 - 2*k_dum*Pfimag*k_dum_z + Pfimag**2 + mass1**2)

    zeroloc = (Ei + omega_k1) - (Ef + omega_Pfk1)

    if np.abs(zeroloc) > tol:
        return 0

    return 1


# -----------------
# Define Integrands
# -----------------

# Azimutal integrand function with the radial integral within
def Integrand_IN(k_dum_z, Eicm, Efcm, gamma, alpha, index):
    """
    k_dum_z is the cosine of the azimutal angle
    Eicm, Efcm: cm energies
    gamma: PP_i cdot PP_f / sqrt(s_i * s_f)
    alpha: dimensionful UV regulator parameter
    index: list of the 4-vector indices, restricted to vector or none (S-wave for initial and final)
    """

    # Split the radial integral into separate regions, hoping to give an easier time to the adaptive method
    # quad integrate up to int_upperbound
    # quad integrate from int_upperbound to infty

    # int_upperbound k ~ 10 Lambda 
    # By this point also the exponential factors should be negligible
    int_upperbound = 10 * cutoff
    
    # otherwise make negligible by integrating up to exp(- a k^4)~1e-10
    if np.exp(-alpha * int_upperbound**4) > 1e-10:
        int_upperbound = (10./alpha * np.log(10))**(.25)

    if test:
        global plot_upperbound
        plot_upperbound = max([plot_upperbound, int_upperbound])


    #Some useful kinematics (defined in the initial cm frame)
    beta = np.sqrt(1-1/gamma**2)
    Ei = Eicm

    Ef = gamma*Efcm
    Pfimag = gamma*beta*Efcm

    q2star_i = 0.25 * (Eicm**2 - 2*(m1**2 + m2**2) + (m1**2 - m2**2)**2/Eicm**2) 
    q2star_f = 0.25 * (Efcm**2 - 2*(m1**2 + m2**2) + (m1**2 - m2**2)**2/Efcm**2)

    q2 = Efcm**2 + Eicm**2 - 2*gamma*Efcm*Eicm


    # define small region radial integral
    # will use adap: no need to vectorize k_dum
    def Integrand_IN_radial_small_region(k_dum):
        # calculate the radial integral in the small region

        # Use these globally to be able to adapt the Pauli Vilars subtractions
        global m1
        global m2

        k2_dum = k_dum**2

        k_dum_4vect = np.array([np.sqrt(k2_dum + m2**2),
                k_dum * k_dum_z])

        ###
        # Cut-off function (independent of Pauli-Villars reg)
        ###
        k2star_i = k2_dum

        # for k2star_f we need to boost k to the final frame
        kzstar_f = np.dot([-gamma*beta, gamma], k_dum_4vect)
        k2perp = (1 - k_dum_z**2) * k2_dum
        k2star_f = k2perp + kzstar_f**2

        HH = np.exp(- alpha * (k2star_i - q2star_i) * (k2star_f - q2star_f))


        ###
        # Evaluate each term in the sum of PV
        # ccs is defined in the "preamble"
        ###


        integrand = 0
        for nn in range(len(ccs)):# UV convergence parts
            
            # Use Lambda (UV) or the mass
            if nn > 0:
                m1 = LAMBDA_param_1[nn - 1]
                m2 = LAMBDA_param_1[nn - 1]

            else:
                m1 = m1t
                m2 = m2t


            # Have to re calculate due to PV mass dependence
            omega_Pfk1 = np.sqrt(Pfimag**2 - 2 * Pfimag * k_dum * k_dum_z + k2_dum + m1**2)

            omega_k1 = np.sqrt(k2_dum + m1**2)
            omega_k2 = np.sqrt(k2_dum + m2**2)


            ###
            # (H-1)DD term
            ###
            evaluate_DD = True

            # There are potential singularities to evaluate DD with nn=0
            if nn == 0:
                qi_singular = np.isclose(k2star_i, q2star_i, atol=1e-8, rtol=0)
                qf_singular = np.isclose(k2star_f, q2star_f, atol=1e-8, rtol=0)

                evaluate_DD = not (qi_singular or qf_singular)

                if qi_singular and qf_singular:
                    Hm1_DD = - alpha * np.sqrt(q2star_f + m2**2)/(2 * Ei * Efcm)

                elif qi_singular:
                    Hm1_DD = alpha * (k2star_f - q2star_f)/(2*Ei) * (
                        1./((Ef - omega_k2)**2 - omega_Pfk1**2))

                elif qf_singular:
                    Hm1_DD = 1./(2 * omega_k2) * (
                        alpha * (k2star_i - q2star_i) * np.sqrt(q2star_f + m2**2)/Efcm) *(
                        1./((Ei - omega_k2)**2 - omega_k1**2))

            # for the rest evaluate normally            
            if evaluate_DD:
                DD = 1./(2 * omega_k2) * (
                    1./((Ef - omega_k2)**2 - omega_Pfk1**2)) * (
                    1./((Ei - omega_k2)**2 - omega_k1**2))

                Hm1_DD = (HH - 1) * DD


            ###
            # Kr term
            ###
            # we will factor out the problematic pole term from Drf/Dri

            # Drf_t = (Ei - Ef - omega_Pfk1 + omega_k1) * Drf
            Drf_t = 1./(2 * omega_Pfk1) * (
                1./((Ef + omega_Pfk1)**2 - omega_k2**2) ) * (
                1./(Ei - Ef - omega_Pfk1 - omega_k1) )

            #Drf numerator terms 
            Kf_4vect = np.array([Ef + omega_Pfk1, k_dum_4vect[1]])

  
            # Dri_t = (Ei - Ef - omega_Pfk1 + omega_k1) * Dri
            Dri_t = 1./(2 * omega_k1) * (
                    1./((Ei + omega_k1)**2 - omega_k2**2) ) * (
                    -1./(Ef - Ei - omega_k1 - omega_Pfk1) )      
            
            #Dri numerator terms 
            Ki_4vect = np.array([Ei + omega_k1, k_dum_4vect[1]])


            ###
            # Calculate numerator terms
            ###
            vector_coeff_D = 1
            vector_coeff_f = 1
            vector_coeff_i = 1
            
            #Lorentz vectors
            for ind in index[0]:
                vector_coeff_D *= k_dum_4vect[ind]
                vector_coeff_f *= Kf_4vect[ind]
                vector_coeff_i *= Ki_4vect[ind]


            Kr_singular = np.isclose(Kf_4vect[0], Ki_4vect[0], atol=1e-8, rtol=0)

            if Kr_singular:

                # calculate scalar case by default
                Kr = -( omega_Pfk1 * (Ei**2 + 5*omega_k1**2 + 6*Ei*omega_k1 - omega_k2**2)
                    + omega_k1 * ((Ei + omega_k1)**2 - omega_k2**2) )

                Kr *= 2*Dri_t**2

                #Lorentz vectors
                for ind in index[0]:
                    #spatial case
                    if ind == 1:
                        Kr *= k_dum_4vect[ind]

                    #component zero case
                    # limit(mu=0) = (Ei+omegak1)*limit(none) + Dri_t
                    elif ind == 0:
                        Kr = Ki_4vect[ind] * Kr + Dri_t

            else:

                Kr = (Drf_t * vector_coeff_f + Dri_t * vector_coeff_i)/(Ei - Ef - omega_Pfk1 + omega_k1)


            ###
            # Individual smooth integrals
            ###
            # term 1 DD
            integrand += ccs[nn] * Hm1_DD * vector_coeff_D 

            # term 1 and 3 Kr
            integrand += - ccs[nn] * Kr
                        
           # term 2 does not have nn = 0
            if nn == 0:
                continue

            integrand += - ccs[nn] * vector_coeff_D * DD * HH

        ### end loop nn

        # Make sure to return the original values to m1 and m2
        m1 = m1t
        m2 = m2t

        # Performing test
        if test:
            if k_dum_z == 1:
                th = '0'
            else:
                th = str(int(np.pi/np.arccos(k_dum_z)))
            

            if th in to_plot:
                to_plot[th].append([k_dum, (Drf_t * vector_coeff_f )/(Ei - Ef - omega_Pfk1 + omega_k1),
                (Dri_t * vector_coeff_i)/(Ei - Ef - omega_Pfk1 + omega_k1),
                Kr, HH])
            else:
                to_plot[th]=[[k_dum, (Drf_t * vector_coeff_f )/(Ei - Ef - omega_Pfk1 + omega_k1),
                (Dri_t * vector_coeff_i)/(Ei - Ef - omega_Pfk1 + omega_k1),
                Kr, HH]]
        
        # Remember the k^2 from the integral measure
        return k2_dum * integrand


    # define large region radial integral
    # will use adap: no need to vectorize k_dum
    def Integrand_IN_radial_large_region(k_dum):
        # calculate the radial integral in the large region

        # Use these globally to be able to adapt the Pauli Vilars subtractions
        global m1
        global m2

        k2_dum = k_dum**2

        k_dum_4vect = np.array([np.sqrt(k2_dum + m2**2),
                k_dum * k_dum_z])

        ###
        # Evaluate each term in the sum of PV
        # ccs is defined in the "preamble"
        ###


        integrand = 0
        for nn in range(len(ccs)):# UV convergence parts
            
            # Use Lambda (UV) or the mass
            if nn > 0:
                m1 = LAMBDA_param_1[nn - 1]
                m2 = LAMBDA_param_1[nn - 1]

            else:
                m1 = m1t
                m2 = m2t


            # Have to re calculate due to PV mass dependence
            omega_Pfk1 = np.sqrt(Pfimag**2 - 2 * Pfimag * k_dum * k_dum_z + k2_dum + m1**2)

            omega_k1 = np.sqrt(k2_dum + m1**2)
            omega_k2 = np.sqrt(k2_dum + m2**2)


            ###
            # (H-1)DD term
            ###
            evaluate_DD = True

            # There should not be singularities to evaluate DD with nn=0 in large...
            if nn == 0:
                k2star_i = k2_dum

                # for k2star_f we need to boost k to the final frame
                kzstar_f = np.dot([-gamma*beta, gamma], k_dum_4vect)
                k2perp = (1 - k_dum_z**2) * k2_dum
                k2star_f = k2perp + kzstar_f**2

                qi_singular = np.isclose(k2star_i, q2star_i, atol=1e-8, rtol=0)
                qf_singular = np.isclose(k2star_f, q2star_f, atol=1e-8, rtol=0)

                if qi_singular or qf_singular:
                    raise Exception(f"Large region DD singularity.\nk2i q2i k2f q2f\n{k2star_i} {q2star_i} {k2star_f} {q2star_f}")

            # evaluate normally            
            DD = 1./(2 * omega_k2) * (
                1./((Ef - omega_k2)**2 - omega_Pfk1**2)) * (
                1./((Ei - omega_k2)**2 - omega_k1**2))


            ###
            # Kr term
            ###
            # we will factor out the problematic pole term from Drf/Dri

            # Drf_t = (Ei - Ef - omega_Pfk1 + omega_k1) * Drf
            Drf_t = 1./(2 * omega_Pfk1) * (
                1./((Ef + omega_Pfk1)**2 - omega_k2**2) ) * (
                1./(Ei - Ef - omega_Pfk1 - omega_k1) )

            #Drf numerator terms 
            Kf_4vect = np.array([Ef + omega_Pfk1, k_dum_4vect[1]])

  
            # Dri_t = (Ei - Ef - omega_Pfk1 + omega_k1) * Dri
            Dri_t = 1./(2 * omega_k1) * (
                    1./((Ei + omega_k1)**2 - omega_k2**2) ) * (
                    -1./(Ef - Ei - omega_k1 - omega_Pfk1) )      
            
            #Dri numerator terms 
            Ki_4vect = np.array([Ei + omega_k1, k_dum_4vect[1]])


            ###
            # Calculate numerator terms
            ###
            vector_coeff_D = 1
            vector_coeff_f = 1
            vector_coeff_i = 1
            
            #Lorentz vectors
            for ind in index[0]:
                vector_coeff_D *= k_dum_4vect[ind]
                vector_coeff_f *= Kf_4vect[ind]
                vector_coeff_i *= Ki_4vect[ind]


            Kr_singular = np.isclose(Kf_4vect[0], Ki_4vect[0], atol=1e-8, rtol=0)

            if Kr_singular:

                # calculate scalar case by default
                Kr = -( omega_Pfk1 * (Ei**2 + 5*omega_k1**2 + 6*Ei*omega_k1 - omega_k2**2)
                    + omega_k1 * ((Ei + omega_k1)**2 - omega_k2**2) )

                Kr *= 2*Dri_t**2

                #Lorentz vectors
                for ind in index[0]:
                    #spatial case
                    if ind == 1:
                        Kr *= k_dum_4vect[ind]

                    #component zero case
                    # limit(mu=0) = (Ei+omegak1)*limit(none) + Dri_t
                    elif ind == 0:
                        Kr = Ki_4vect[ind] * Kr + Dri_t

            else:

                Kr = (Drf_t * vector_coeff_f + Dri_t * vector_coeff_i)/(Ei - Ef - omega_Pfk1 + omega_k1)


            ###
            # Individual smooth integrals
            ###
            # term 1
            integrand += - ccs[nn] * (DD * vector_coeff_D + Kr)

        ### end loop nn

        # Make sure to return the original values to m1 and m2
        m1 = m1t
        m2 = m2t
        
        # Remember the k^2 from the integral measure
        return k2_dum * integrand

    # integrand in theta
    f_th = 0
    # do the integral in the small region
    f_th = integrate.quad(Integrand_IN_radial_small_region, 
        0, int_upperbound, 
        epsrel = 1e-6)[0]

    f_th += integrate.quad(Integrand_IN_radial_large_region, 
        int_upperbound, np.inf,
        epsrel = 1e-6)[0]

    return f_th

# -----------------
# Axial Integrands
# -----------------

# k-magnitude integrand
def Integrand_IN_a(k_dum, k_dum_z, Eicm, Efcm, gamma, alpha, index, region):
    """
    k_dum is the radial integration variable, let's make it a vector
    k_dum_z is the cosine of the azimutal angle
    Ei_st, Ef_st: cm energies
    gamma: PP_i cdot PP_f / sqrt(s_i * s_f)
    alpha: dimensionful UV regulator parameter
    index: list of the 4-vector indices, restricted to vector or none (S-wave for initial and final)
    region: to distinguish for region where exp factor is negligible
        region 'small' makes full evaluation
        region 'large' makes the evaluation faster, use when the exponentials are expected to be negligible
    """

    islist = len(np.array(k_dum).shape)

    if islist:
        k_dum = np.array(k_dum)
        k2_dum = k_dum**2
        k_len = len(k_dum)
    else:
        k_dum = np.array([k_dum])
        k2_dum = k_dum**2
        k_len = 1


    # Use these globally to be able to adapt the Pauli Vilars subtractions
    global m1
    global m2

    # DELETE?
    # k_dum = np.array(k_dum)
    # k2_dum = k_dum**2

    k_dum_4vect = np.array([np.sqrt(k2_dum + m2**2),
                    k_dum * k_dum_z])

    integrand = 0




    # Extract kinematic variables
    beta = np.sqrt(1-1/gamma**2)

    # Initial frame variables (we are evaluating in the initial frame: no need for Pivec)
    Ei = Eicm

    # DELETE?
    # Pivec = np.abs(P_i[1])
    # Eicm = np.sqrt(square_4vec(P_i))
    # Lambdai = boost(P_i)

    # qstar doesn't inherit the m/Lambda dependence (only used in cutoff)
    q2star_i = 0.25 * (Eicm**2 - 2*(m1**2 + m2**2) + (m1**2 - m2**2)**2/Eicm**2) 
    


    # Final frame variables

    Ef = gamma*Efcm
    Pfimag = gamma*beta*Efcm

    # DELETE?
    # Pfvec = np.abs(P_f[1])
    # Efcm = np.sqrt(square_4vec(P_f)) 
    # Lambdaf = boost(P_f)

    # qstar doesn't inherit the m/Lambda dependence (only used in cutoff)
    q2star_f = 0.25 * (Efcm**2 - 2*(m1**2 + m2**2) + (m1**2 - m2**2)**2/Efcm**2)

    

    # Cut-off function (independent of Pauli-Villars reg) only used for 'small' region
    if region == 'small':
        # need to calculate k2star_i and k2star_f

        # k2star_i is just k2 (evaluating in integrand in initial frame)

        k2star_i = k2_dum

        # for k2star_f we need to boost k to the final frame
        kzstar_f = np.dot([-gamma*beta, gamma], k_dum_4vect)

        k2perp = (1 - k_dum_z**2) * k2_dum

        k2star_f = k2perp + kzstar_f**2

        HH = np.exp(- alpha * (k2star_i - q2star_i) * (k2star_f - q2star_f))

        # DELETE?
        # if HH > 1e100:
        #     print(k_dum_z, k_dum, alpha, (k2star_i - q2star_i),  (k2star_f - q2star_f))

    elif region == 'large':
        HH = 0*k_dum


    # Evaluate each term in the sum
    # ccs is defined in the "preamble"
    for nn in range(len(ccs)):# UV convergence parts
        
        # Use Lambda (UV) or the mass
        if nn > 0:
            m1 = LAMBDA_param_1[nn - 1]
            m2 = LAMBDA_param_1[nn - 1]

        else:
            m1 = m1t
            m2 = m2t


        # Have to re calculate since UV dependence
        omega_Pfk1 = np.sqrt(Pfimag**2 - 2 * Pfimag * k_dum * k_dum_z + k2_dum + m1**2)

        omega_k1 = np.sqrt(k2_dum + m1**2)
        omega_k2 = np.sqrt(k2_dum + m2**2)

        # k_dum_4vect[0] = omega_k2

        # kzstar_i = np.dot(Lambdai, k_dum_4vect)[1,:]
        # kzstar_f = np.dot(Lambdaf, k_dum_4vect)[1,:]

        # Calculate the most used shorthands
        
        # #P_{i}k
        # Pik2 = Pivec**2 - 2 * Pivec * k_dum * np.cos(k_dum_th) + k2_dum
        # omega_Pik1 = np.sqrt(Pik2 + m1**2)

        # #P_{f}k
        # Pfk2 = Pfvec**2 - 2 * Pfvec * k_dum * np.cos(k_dum_th) + k2_dum
        # omega_Pfk1 = np.sqrt(Pfk2 + m1**2)


        # we will factor out the problematic pole term from Drf/Dri
        # Drf_t = (Ei - Ef - omega_Pfk1 + omega_k1) * Drf

        Drf_t = 1./(2 * omega_Pfk1) * (
                1./((Ef + omega_Pfk1)**2 - omega_k2**2) ) * (
                1./(Ei - Ef - omega_Pfk1 - omega_k1) )
        
        #Drf numerator terms 
        Kf_4vect = np.concatenate(([Ef + omega_Pfk1], [k_dum_4vect[1,:]]))

        # we will factor out the problematic pole term from Drf/Dri
        # Dri_t = (Ei - Ef - omega_Pfk1 + omega_k1) * Dri
        Dri_t = 1./(2 * omega_k1) * (
                1./((Ei + omega_k1)**2 - omega_k2**2) ) * (
                -1./(Ef - Ei - omega_k1 - omega_Pfk1) )      
        
        #Dri numerator terms 
        Ki_4vect = np.concatenate(([Ei + omega_k1], [k_dum_4vect[1,:]]))


        # DD denominator
        qi_singular = 0*k_dum + 1
        qf_singular = 0*k_dum + 1
        DD_Hm1 = 0*k_dum 
        
        # this is wrong...

        if q2star_i != q2star_f:

            qi_singular = np.isclose(k2_dum, q2star_i)
            qf_singular = np.isclose(k2_dum, q2star_f)

            any_sing = qi_singular | qf_singular

            DD_Hm1[~any_sing] = 1./(2 * omega_k2[~any_sing]) * (
                    1./((Ef - omega_k2[~any_sing])**2 - omega_Pfk1[~any_sing]**2)) * (
                    1./((Ei - omega_k2[~any_sing])**2 - omega_k1[~any_sing]**2)) * (
                    (HH[~any_sing] - 1))

            DD_Hm1[qi_singular] =  alpha * (k2star_f[qi_singular] - q2star_f)/(2*Ei) * (
                    1./((Ef - omega_k2[qi_singular])**2 - omega_Pfk1[qi_singular]**2)) 

            DD_Hm1[qf_singular] =  1./(2 * omega_k2[qf_singular]) * (
                    alpha * (k2star_i[qi_singular] - q2star_i) * np.sqrt(q2star_f + m2**2)/Efcm) *(
                    1./((Ei - omega_k2[qf_singular])**2 - omega_k1[qf_singular]**2))

        else:

            qboth_singular = np.isclose(k2_dum, q2star_f)

            DD_Hm1[~qboth_singular] = 1./(2 * omega_k2[~qboth_singular]) * (
                    1./((Ef - omega_k2[~qboth_singular])**2 - omega_Pfk1[~qboth_singular]**2)) * (
                    1./((Ei - omega_k2[~qboth_singular])**2 - omega_k1[~qboth_singular]**2)) * (
                    (HH[~qboth_singular] - 1))

            DD_Hm1[qboth_singular] = - alpha * np.sqrt(q2star_f + m2**2)/(2 * Ei * Efcm)


        vector_coeff_D = 1

        vector_coeff_f = 1

        vector_coeff_i = 1
        
        #Lorentz vectors
        for ind in index[0]:

            vector_coeff_D *= k_dum_4vect[ind]

            vector_coeff_f *= Kf_4vect[ind]

            vector_coeff_i *= Ki_4vect[ind]
        
        
        # Individual smooth integrals            
        
        # term 1           
        integrand += ccs[nn] * (vector_coeff_D * DD  +
            (Drf_t * vector_coeff_f + Dri_t * vector_coeff_i)/(Ei - Ef - omega_Pfk1 + omega_k1)
            ) * (HH - 1)

        # No 2, 3 term for I_N simplified (large k)
        if region == 'large':
            continue

        # term 3
        integrand += - ccs[nn] * HH * (
            (Drf_t * vector_coeff_f + Dri_t * vector_coeff_i)/(Ei - Ef - omega_Pfk1 + omega_k1))
                          

        # term 2 doesn't have a nn = 0
        if nn == 0:
            continue

        integrand += - ccs[nn] *  vector_coeff_D * (DD) * HH

    # loop over masses/PV counterterms



    
    # Make sure to return the original values to m1 and m2
    m1 = m1t
    m2 = m2t
    
    # Remember the k^2 from the integral measure
    if test:
        if k_dum_z == 1:
            th = '0'
        else:
            th = str(int(np.pi/np.arccos(k_dum_z)))
        

        if th in to_plot:
            to_plot[th].append([k_dum, 1*(Drf_t )/(Ei - Ef - omega_Pfk1 + omega_k1),
            (Dri_t  )/(Ei - Ef - omega_Pfk1 + omega_k1),
            (Drf_t  + Dri_t )/(Ei - Ef - omega_Pfk1 + omega_k1), k2_dum * integrand])
        else:
            to_plot[th]=[[k_dum, 1*(Drf_t  )/(Ei - Ef - omega_Pfk1 + omega_k1),
            (Dri_t  )/(Ei - Ef - omega_Pfk1 + omega_k1),
            (Drf_t  + Dri_t )/(Ei - Ef - omega_Pfk1 + omega_k1), k2_dum * integrand]]

    return k2_dum * integrand


# Azimutal integrand
def Ang_Integrand_IN_a(k_dum_z, Eicm, Efcm, gamma, alpha, index):
    """
    k_dum_z is the cosine of the azimutal angle
    Eicm, Efcm: cm energies
    gamma: PP_i cdot PP_f / sqrt(s_i * s_f)
    alpha: dimensionful UV regulator parameter
    index: list of the 4-vector indices, restricted to vector or none (S-wave for initial and final)
    """

    # Split the radial integral into separate regions, hoping to give an easier time to the adaptive method
    # quad integrate up to int_upperbound
    # quad integrate from int_upperbound to infty


    #Some useful kinematics (defined in the initial cm frame)
    beta = np.sqrt(1-1/gamma**2)
    Ei = Eicm

    Ef = gamma*Efcm
    Pfimag = gamma*beta*Efcm

    q2star_i = 0.25 * (Eicm**2 - 2*(m1**2 + m2**2) + (m1**2 - m2**2)**2/Eicm**2) 
    q2star_f = 0.25 * (Efcm**2 - 2*(m1**2 + m2**2) + (m1**2 - m2**2)**2/Efcm**2)

    q2 = Efcm**2 + Eicm**2 - 2*gamma*Efcm*Eicm

    # DELETE?
    # find the Drf/Dri potential pole locations
    # polesDr = []
    
    # for LAM_1 in [m1] + LAMBDA_param_1:
        
    #     pot_polesDr = []
    #     # their location are found as a solution to a second order polynomial equation
    #     # we need to check if the determinant is positive
    #     determinant = (Ef - Ei)**2 *( 
    #         q2**2 - 4 * LAM_1**2 * ((Ef - Ei)**2 - k_dum_z**2 * Pfimag**2)
    #         )

    #     if determinant > 0:
    #         twoa = 2 * ((Ef - Ei)**2 - k_dum_z**2 * Pfimag**2)

    #         # when twoa is zero only one solution is valid
    #         if twoa == 0:
    #             pot_polesDr.append(
    #                 np.sign(Ef - Ei)*(q2**2 - 4*(Ef - Ei)**2 * LAM_1**2)/(-4*(Ef - Ei)*q2)
    #             )
    #         # otherwise construct with the polynomial formula
    #         else:
    #             minusb = Pfimag * k_dum_z * q2 
    #             pot_polesDr.append( (minusb + np.sqrt(determinant))/twoa )
    #             pot_polesDr.append( (minusb - np.sqrt(determinant))/twoa )

    #     for k_pole in pot_polesDr:
    #         if check_valid_Dr_pole(k_pole, k_dum_z, Ei, Ef, Pfimag, LAM_1):
    #             polesDr.append(k_pole)

    
    # integrate up up to k ~ 10 Lambda 
    # By this point also the exponential factors should be negligible
    int_upperbound = 10 * cutoff
    
    # otherwise make negligible by integrating up to exp(- a k^4)~1e-10
    if np.exp(-alpha * int_upperbound**4) > 1e-10:
        int_upperbound = (10./alpha * np.log(10))**(.25)


    # DELETE?
    #split poles below/above upperbound
    # pointssmall = []
    # rest = []

    # #also add qst to integral
    # if q2star_i > 0:
    #     pointssmall.append(np.sqrt(q2star_i))
    # if q2star_f > 0:
    #     pointssmall.append(np.sqrt(q2star_f))

    # for k_pole in polesDr:
    #     if k_pole < int_upperbound:
    #         pointssmall.append(k_pole)
    #     else:
    #         rest.append(k_pole)

    if test:
        global plot_upperbound
        plot_upperbound = max([plot_upperbound, int_upperbound])

    # do the integral in the small region
    f_th = integrate.quad(Integrand_IN_a, 0, int_upperbound, epsrel = 1e-6,
        args = (k_dum_z,  Eicm, Efcm, gamma, alpha, index, 'small'),
        points=pointssmall)[0]

    # no Dr poles above upper bound
    if len(rest) == 0:

        temp = integrate.quad(Integrand_IN_a, int_upperbound, np.inf, epsrel = 1e-4,
        args = (k_dum_z, Eicm, Efcm, gamma, alpha, index, 'large'))[0]

        f_th += temp

    # one Dr pole above upper bound
    elif len(rest) == 1:
        temp = integrate.quad(Integrand_IN_a, int_upperbound, rest[0], epsrel = 1e-4,
        args = (k_dum_z, Eicm, Efcm, gamma, alpha, index, 'large'))[0]

        f_th += temp

        temp = integrate.quad(Integrand_IN_a, rest[0], np.inf, epsrel = 1e-4,
        args = (k_dum_z, Eicm, Efcm, gamma, alpha, index, 'large'))[0]

        f_th += temp

    # multiple Dr poles above upper bound
    else:
        temp = integrate.quad(Integrand_IN_a, int_upperbound, rest[-1], epsrel = 1e-4,
        args = (k_dum_z, Eicm, Efcm, gamma, alpha, index, 'large'),
        points=rest[:-1])[0]

        f_th += temp

        temp = integrate.quad(Integrand_IN_a, rest[-1], np.inf, epsrel = 1e-4,
        args = (k_dum_z, Eicm, Efcm, gamma, alpha, index, 'large'))[0]

        f_th += temp

    return f_th


# -----------------
# General macro to do the I_N integral
# -----------------

# It has two methods,
# adap_quad (adpative gauss quadrature) 
# fix_qaud (fixed gauss quadrature) 
# I think fix_quad is faster (and avoids hitting poles by mistake), both should work in general


# index is a list, 
# first element a list of the 4-vector indices
# second element the value of lf, mf
# third element the value of li, mi

def make_int(Eicm, Efcm, gamma, alpha, index, method):
    # we probably want to use np.isclose(x, 0.0)

    # This implementation performs a single angular integral
    # The radial integral is done for every evaluation needed for the ang integration

    integral = 0
    
    
    if method == 'adap_quad':
    
        # Remember the 2pi from the phi integral
        # integral done in variable z=costh, with limits (-1,1)

        if test:
            for th in [0, np.pi/4, np.pi/3, np.pi/2]:

                zz = np.cos(th)
                tmp = Integrand_IN(zz, Eicm, Efcm, gamma, alpha, index)
                print(len(to_plot))
                integral += tmp

                if zz == 1:
                    th = '0'
                else:
                    th = str(int(np.pi/np.arccos(zz)))
                
                theta_vals.append(th)
                integral_kradial.append(tmp)

            # fake the integration with factor of 1/4
            integral *= 1./(2 * np.pi)**2 * (1/4)

        else:

            val_temp = integrate.quad(Integrand_IN, -1, 1, epsrel = 1e-6,
                args = (Eicm, Efcm, gamma, alpha, index), full_output=0)[0]

            integral = 1./(2 * np.pi)**2 * val_temp

            
    elif method == 'fix_quad':
              
        tolerance = 1e-4
        error = 1
        fix_ord = 10

        def Ang_Integrand_IN_nonphi(k_dum_z, Eicm, Efcm, gamma, alpha, index):

            reslts = [Ang_Integrand_IN_a( zs, Eicm, Efcm, gamma, alpha, index)
                     for zs in k_dum_z]
            
            # Remember the 2pi from the phi integral
            return 2 * np.pi * np.array(reslts)

        
        val_temp_old = integrate.fixed_quad(Ang_Integrand_IN_nonphi, -1, 1,
            args = (Eicm, Efcm, gamma, alpha, index), n = fix_ord)[0]
            
        print(val_temp_old)

        fix_ord += 1
        while error > tolerance:
            
            val_temp_new = integrate.fixed_quad(Ang_Integrand_IN_nonphi, -1, 1,
                args = (Eicm, Efcm, gamma, alpha, index), n = fix_ord)[0]
            
            error = np.abs(val_temp_new - val_temp_old)/np.abs(val_temp_old)
            
            val_temp_old = val_temp_new
            
            print(val_temp_old)
            
            fix_ord += 1
            
            if fix_ord > 10:
                print('order bigger 10, E:', error)
                
            if fix_ord > 15:
                print('order 15, E:', error)
                break
                    

        integral = 1./(2 * np.pi)**3 * val_temp_new
            

    else:
        raise ValueError('not implemented')


    return integral

#CALCULATE STUFF
print('IN alpha ',alpha)
ener_shape = np.shape(Eistar)
I_Nn = np.ones(ener_shape) * complex(0.,0.)

ang_integral_method = 'adap_quad'


if len(ener_shape) > 1: # for mesh inputs
    for mm, enirow in enumerate(Ei):
        for nn, enin in enumerate(enirow):
            enfin = Ef[mm,nn]
            PP_i = np.concatenate(([enin], lab_moment_i))
            PP_f = np.concatenate(([enfin], lab_moment_f))

            Eicm, Efcm, gamma = get_cm_gamma(PP_i, PP_f)
            coeffs, list_indices = get_rotboost_indices(indices, PP_i, PP_f)

            I_Nn[mm,nn] = 0
            for coef, ix in zip(coeffs, list_indices):
                if coef==0:
                    continue
                I_Nn[mm,nn] += coef*make_int(Eicm, Efcm, gamma, alpha, ix, ang_integral_method)

                if test:
                    break

                # print(coef, ix, I_Nn)
            
            print('INcalc: ', Eistar[mm,nn], Efstar[mm,nn], '---------', I_Nn[mm,nn])
else:
    for mm, enin in enumerate(Ei):
        enfin = Ef[mm]
        PP_i = np.concatenate(([enin], lab_moment_i))
        PP_f = np.concatenate(([enfin], lab_moment_f))

        Eicm, Efcm, gamma = get_cm_gamma(PP_i, PP_f)

        # print(PP_i, PP_f, Eicm, Efcm, gamma)

        coeffs, list_indices = get_rotboost_indices(indices, PP_i, PP_f)

        # print(coeffs, list_indices, indices)
        
        I_Nn[mm] = 0
        for coef, ix in zip(coeffs, list_indices):
            if coef==0:
                continue
            I_Nn[mm] += coef*make_int(Eicm, Efcm, gamma, alpha, ix, ang_integral_method)

            if test:
                    break

            # print(coef, ix, I_Nn)

        print('INcalc: ', Eistar[mm], Efstar[mm], f"{gamma:.3f}", '---------', I_Nn[mm])



if test:
    f, axs = plt.subplots(4,1, figsize=(7,7), layout='constrained', sharex=True)
    ax = axs[0]
    ax2 = axs[-1]
    cols = [0,2,1,3]
    print("theta vals:", theta_vals)
    for nn, th in enumerate(theta_vals):
        # print(to_plot[th])
        xvals = np.array([v[0] for v in to_plot[th]])
        drfvals = [v[1] for v in to_plot[th]]
        drivals = [v[2] for v in to_plot[th]]
        drvals = [v[3] for v in to_plot[th]]
        hvals = [v[4] for v in to_plot[th]]

        ax.plot(xvals, drfvals, '.',
               color=f'C{cols[nn]}')

        axs[1].plot(xvals, drivals, '.',
               color=f'C{cols[nn]}')
        
        axs[2].plot(xvals, drvals, '.',
               color=f'C{cols[nn]}')

        ax2.plot(xvals, hvals, '.', label=rf'$\theta=\pi/{th}$, $\int\to$ {integral_kradial[nn]:.5f}',
               color=f'C{cols[nn]}')


    ax2.grid()

    # ax.set_title('Integrand')
    ax2.set_title('cut-off function')

    ax.set_xlim(0, plot_upperbound)
    # ax.set_ylim(0, .1)


    ax2.legend()

    ax2.set_xlabel(r'$|{k}|$')


    axs[0].set_ylabel('Drf')
    axs[1].set_ylabel('Dri')
    axs[2].set_ylabel('Drf+Dri')

    plt.show()



#Save the values
if not os.path.exists(IN_folder):
    os.makedirs(IN_folder)

with open(filename, 'wb') as f:

    np.save(f,I_Nn)







