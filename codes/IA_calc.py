#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-17 04:20:28
# @Author  : Felipe Ortega (felipeortegagama@gmail.com)
# @Version : 1.0


# Calculate I_A for several values of Ei^st/ Ef^st

import numpy as np
np.seterr(all = 'warn')
import sys
import os

from scipy import integrate
from scipy import optimize

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

results_folder = cval['folder']

IA_folder = results_folder + 'IA_vals/'

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
with open(energ_file, 'rb') as f:

    eners = np.load(f)

Eistar = eners[0]
 
Efstar = eners[1]

Ei = np.sqrt(Eistar**2 + np.dot(lab_moment_i, lab_moment_i))
Ef = np.sqrt(Efstar**2 + np.dot(lab_moment_f, lab_moment_f))

# Define useful functions
def y_noie(x, Lam1, Lam2, pm, q2, si, sf):
    
    x = complex(x, 0)
    
    if pm == 'p':
        pm = 1
    else:
        pm = -1 
              
    AA = 1 + (Lam2**2 - Lam1**2 + x * (q2 - sf - si)) / si
    
    BB = -4 * (Lam2**2 - x * (Lam2**2 - Lam1**2) - x * (1 - x) * sf) / si
    
    return 0.5 * (AA + pm * np.sqrt(AA**2 + BB))


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

# Singularities of the integrand
def avoid_points(m1, m2, si, sf, q2):

    # Divergences due to the division
    AA2pB_0 = (-4*m2**2)/si + (-m1**2 + m2**2 + si)**2/si**2 
    
    AA2pB_1 = ((-4*(m1**2 - m2**2 - sf))/si + (2*(q2 - sf - si)*(-m1**2 + m2**2 + si))/si**2)
    
    AA2pB_2 = ((q2 - sf - si)**2/si**2 - (4*sf)/si)
    
    xxs = np.roots([AA2pB_2,AA2pB_1,AA2pB_0])
    
    avoid = []

    if np.angle(xxs[0])==0 or np.angle(xxs[0])==np.pi:
        
        if np.real(xxs[0]) > 0 and np.real(xxs[0]) < 1:
            avoid.extend([xxs[0]])
        if np.real(xxs[1]) > 0 and np.real(xxs[1]) < 1:
            avoid.extend([xxs[1]])
    
    # Divergences due to the logarithm
    # Evaluate at the borders to find sign changes
    yp0 = y_noie(0, m1, m2, 'p', q2, si, sf)
    yp1 = y_noie(1, m1, m2, 'p', q2, si, sf)

    ym0 = y_noie(0, m1, m2, 'm', q2, si, sf)
    ym1 = y_noie(1, m1, m2, 'm', q2, si, sf)
    
    if yp0.real * yp1.real < 0:
        def realy(x):
            return y_noie(x, m1, m2, 'p', q2, si, sf).real

        avoid.extend([optimize.brentq(realy, 0, 1)])

    if ym0.real * ym1.real < 0:
        def realy(x):
            return y_noie(x, m1, m2, 'm', q2, si, sf).real

        avoid.extend([optimize.brentq(realy, 0, 1)])

    if (1 - yp0.real) * (- yp1.real) < 0:
        def realy(x):
            return 1 - x - y_noie(x, m1, m2, 'p', q2, si, sf).real

        avoid.extend([optimize.brentq(realy, 0, 1)])        

    if (1 - ym0.real) * (- ym1.real) < 0:
        def realy(x):
            return 1 - x - y_noie(x, m1, m2, 'm', q2, si, sf).real

        avoid.extend([optimize.brentq(realy, 0, 1)]) 

            
    if len(avoid) == 0:
        avoid = (0.5,)

    return avoid


# -----------
# F functions

# Log function with epsilon handling
# i.e. log(z +/- i eps)
# Careful that all terms are named with opposite pm of their front pm
def logeps(comp, pm):
    
    # plus case is the default Riemman sheet choice of python
    if np.imag(comp) == 0 and np.real(comp)<0:
        if pm == 'p':
            return np.log(np.abs(comp)) + I * np.pi
        elif pm == 'm':
            return np.log(np.abs(comp)) - I * np.pi

    else:
        return np.log(comp)
    
# Generalized antider of the 1 pole integral
def antiderlog(ymin, ymax, pole, pm):

    if pole == ymin or pole == ymax:
        return float('nan')

    if pm == 'p':
        pm = +1
    else:
        pm = -1

    # Case of the non-imag pole
    if np.angle(pole)==0 or np.abs(np.angle(pole)) == np.pi:
        rpole = np.real(pole)

        # Pole in the domain: PV +/- I pi 
        if (rpole - ymin) * (ymax - rpole) > 0:
            repart = np.log(np.abs((ymax - rpole) / (ymin - rpole))) #abs ensures the argument is real

            impart = pm * I * np.pi

            return repart + impart

        else: # no pole in the domain of integration 

            repart = np.log(np.abs((ymax - rpole) / (ymin - rpole)))


            return repart

    # Case of imaginary pole   
    else:
        rpole = np.real(pole)
        ipole = np.imag(pole)

        repart = 0.5 * np.log(((ymax - rpole)**2 + ipole**2)/((ymin - rpole)**2 + ipole**2))

        impart = I * (np.arctan((ymax - rpole) / ipole) - np.arctan((ymin - rpole) / ipole))

        return repart + impart
    


def F1_tt(x, Lams, kins):
    
    x = complex(x, 0)
    
    q2 = kins[0]
    si = kins[1]
    sf = kins[2]
    
    yp = y_noie(x, Lams[0], Lams[1], 'p', q2, si, sf)
    ym = y_noie(x, Lams[0], Lams[1], 'm', q2, si, sf)
    
    try:
        ffcoef = 1. / (si * (4 * np.pi)**2 * (yp - ym))

    except RuntimeWarning as e:
        print(e)
        ffcoef = 0

    
    return ffcoef * (antiderlog(0, 1 - x, yp, 'p') - antiderlog(0, 1 - x, ym, 'm'))

def F2_tt(x, Lams, kins):
    
    x = complex(x, 0)
    
    q2 = kins[0]
    si = kins[1]
    sf = kins[2]
    
    yp = y_noie(x, Lams[0], Lams[1], 'p', q2, si, sf)
    ym = y_noie(x, Lams[0], Lams[1], 'm', q2, si, sf)
    
    
    ffcoef = 1. / (16*np.pi**2*(-ym + yp)*si)

    
    return ffcoef * (-(ym * antiderlog(0, 1 - x, ym, 'm')) + yp * antiderlog(0, 1 - x, yp, 'p'))


def F3_tt(x, Lams, kins):
    
    x = complex(x, 0)
    
    q2 = kins[0]
    si = kins[1]
    sf = kins[2]
    
    yp = y_noie(x, Lams[0], Lams[1], 'p', q2, si, sf)
    ym = y_noie(x, Lams[0], Lams[1], 'm', q2, si, sf)
    
    
    ffcoef = 1. / (16*np.pi**2*(-ym + yp)*si)
    
    Lm = antiderlog(0, 1 - x, ym, 'm')
    
    Lp = antiderlog(0, 1 - x, yp, 'p')
    
    return ffcoef * (-(ym**2 * Lm) + yp**2 * Lp)


def F4_tt(x, Lams, kins):
    
    x = complex(x, 0)
    
    q2 = kins[0]
    si = kins[1]
    sf = kins[2]
    
    yp = y_noie(x, Lams[0], Lams[1], 'p', q2, si, sf)
    ym = y_noie(x, Lams[0], Lams[1], 'm', q2, si, sf)
    
    
    ffcoef = 1. / (16*np.pi**2*(-ym + yp)*si)
    
    Lm = antiderlog(0, 1 - x, ym, 'm')
    
    Lp = antiderlog(0, 1 - x, yp, 'p')
    
    return ffcoef * ((1 - x) * (-ym**2 + yp**2) + (-(ym**3 * Lm) + yp**3 * Lp))




def F5_tt(x, Lams, kins):
    
    x = complex(x, 0)
    
    q2 = kins[0]
    si = kins[1]
    sf = kins[2]
    
    yp = y_noie(x, Lams[0], Lams[1], 'p', q2, si, sf)
    ym = y_noie(x, Lams[0], Lams[1], 'm', q2, si, sf)
    
    
    ffcoef = -1. /(8 * np.pi**2)

    return ffcoef * ((1 - x - ym) * logeps(1 - x - ym, 'p')
                     + ym * logeps(-ym, 'p')
                     + (1 - x - yp) * logeps(1 - x - yp, 'm')
                     + yp * logeps(-yp, 'm'))

def F6_tt(x, Lams, kins):
    
    x = complex(x, 0)
    
    q2 = kins[0]
    si = kins[1]
    sf = kins[2]
    
    yp = y_noie(x, Lams[0], Lams[1], 'p', q2, si, sf)
    ym = y_noie(x, Lams[0], Lams[1], 'm', q2, si, sf)
    
    
    ffcoef = -1. /(16 * np.pi**2)
    
    
    return ffcoef * (-(1 - x) * (ym + yp)
                     + ((1 - x)**2 - ym**2) * logeps(1 - x - ym, 'p')
                     + ym**2 * logeps(-ym, 'p')
                     + ((1 - x)**2 - yp**2) * logeps(1 - x - yp, 'm')
                     + yp**2 * logeps(-yp, 'm'))

# I(a,b) Integrals

# Receive value of masses and kinematics (BEWARE: this uses the lower case q = Pf - Pi)
def I00_tt(ls, ks):
      
    q2 = ks[0]
    si = ks[1]
    sf = ks[2]
    
    m1 = ls[0]
    m2 = ls[1]
    
    def realF(x,ls,ks):
        return np.real(F1_tt(x,ls,ks))

    def imagF(x,ls,ks):
        return np.imag(F1_tt(x,ls,ks))
    
    avoid = avoid_points(m1, m2, si, sf, q2)
    
    
    ff = integrate.quad(realF, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
    ffi = integrate.quad(imagF, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
    
    return ff + I * ffi

def I11_tt(ls, ks):
      
    q2 = ks[0]
    si = ks[1]
    sf = ks[2]
    
    m1 = ls[0]
    m2 = ls[1]
    
    def realF(x,ls,ks):
        return x * np.real(F1_tt(x,ls,ks))

    def imagF(x,ls,ks):
        return x * np.imag(F1_tt(x,ls,ks))
    
    avoid = avoid_points(m1, m2, si, sf, q2)
    
    
    ff = integrate.quad(realF, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
    ffi = integrate.quad(imagF, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
    
    return ff + I * ffi

def I12_tt(ls, ks):
      
    q2 = ks[0]
    si = ks[1]
    sf = ks[2]
    
    m1 = ls[0]
    m2 = ls[1]
    
    def realF(x,ls,ks):
        return np.real(F2_tt(x,ls,ks))

    def imagF(x,ls,ks):
        return np.imag(F2_tt(x,ls,ks))
    
    avoid = avoid_points(m1, m2, si, sf, q2)
            
    ff = integrate.quad(realF, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
    ffi = integrate.quad(imagF, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
    
    return ff + I * ffi


def I31_tt(ls, ks):
      
    q2 = ks[0]
    si = ks[1]
    sf = ks[2]
    
    m1 = ls[0]
    m2 = ls[1]
    
    def realF(x,ls,ks):
        return x**3 * np.real(F1_tt(x,ls,ks))

    def imagF(x,ls,ks):
        return x**3 * np.imag(F1_tt(x,ls,ks))
    
    avoid = avoid_points(m1, m2, si, sf, q2)
    
    
    ff = integrate.quad(realF, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
    ffi = integrate.quad(imagF, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
    
    return ff + I * ffi

def I32_tt(ls, ks):
      
    q2 = ks[0]
    si = ks[1]
    sf = ks[2]
    
    m1 = ls[0]
    m2 = ls[1]
    
    def realF(x,ls,ks):
        return np.real(F4_tt(x,ls,ks))

    def imagF(x,ls,ks):
        return np.imag(F4_tt(x,ls,ks))
    
    avoid = avoid_points(m1, m2, si, sf, q2)
            
    ff = integrate.quad(realF, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
    ffi = integrate.quad(imagF, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
    
    return ff + I * ffi

def I33_tt(ls, ks):
      
    q2 = ks[0]
    si = ks[1]
    sf = ks[2]
    
    m1 = ls[0]
    m2 = ls[1]
    
    def realF(x,ls,ks):
        return x**2 * np.real(F2_tt(x,ls,ks))

    def imagF(x,ls,ks):
        return x**2 * np.imag(F2_tt(x,ls,ks))
    
    avoid = avoid_points(m1, m2, si, sf, q2)
            
    ff = integrate.quad(realF, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
    ffi = integrate.quad(imagF, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
    
    return ff + I * ffi

def I34_tt(ls, ks):
      
    q2 = ks[0]
    si = ks[1]
    sf = ks[2]
    
    m1 = ls[0]
    m2 = ls[1]
    
    def realF(x,ls,ks):
        return x * np.real(F3_tt(x,ls,ks))

    def imagF(x,ls,ks):
        return x * np.imag(F3_tt(x,ls,ks))
    
    AA2pB_0 = (-4*m2**2)/si + (-m1**2 + m2**2 + si)**2/si**2 
    
    AA2pB_1 = ((-4*(m1**2 - m2**2 - sf))/si + (2*(q2 - sf - si)*(-m1**2 + m2**2 + si))/si**2)
    
    AA2pB_2 = ((q2 - sf - si)**2/si**2 - (4*sf)/si)
    
    avoid = avoid_points(m1, m2, si, sf, q2)
            
    ff = integrate.quad(realF, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
    ffi = integrate.quad(imagF, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
    
    return ff + I * ffi

def I35_tt(ls, ks):
      
    q2 = ks[0]
    si = ks[1]
    sf = ks[2]
    
    m1 = ls[0]
    m2 = ls[1]
    
    def realF(x,ls,ks):
        return x * np.real(F5_tt(x,ls,ks))

    def imagF(x,ls,ks):
        return x * np.imag(F5_tt(x,ls,ks))

            
    ff = integrate.quad(realF, 0, 1, args=([m1,m2],[q2,si,sf]))[0]
    ffi = integrate.quad(imagF, 0, 1, args=([m1,m2],[q2,si,sf]))[0]
    
    return ff + I * ffi

def I36_tt(ls, ks):
      
    q2 = ks[0]
    si = ks[1]
    sf = ks[2]
    
    m1 = ls[0]
    m2 = ls[1]
    
    def realF(x,ls,ks):
        return  np.real(F6_tt(x,ls,ks))

    def imagF(x,ls,ks):
        return np.imag(F6_tt(x,ls,ks))

            
    ff = integrate.quad(realF, 0, 1, args=([m1,m2],[q2,si,sf]))[0]
    ffi = integrate.quad(imagF, 0, 1, args=([m1,m2],[q2,si,sf]))[0]
    
    return ff + I * ffi


def make_int(P_i, P_f, index):

    # Extract some variables

    Ei = P_i[0]

    Ef = P_f[0]

    Pfz = P_f[3]

    Piz = P_i[3]

    si = square_4vec(P_i)

    sf = square_4vec(P_f)

    q2 = square_4vec(P_f-P_i)

    kss = [q2, si, sf]


    betai = np.sqrt(np.dot(P_i[1:4],P_i[1:4]))/P_i[0]

    gammai = 1/np.sqrt(1-betai**2)


    betaf = np.sqrt(np.dot(P_f[1:4],P_f[1:4]))/P_f[0]

    gammaf = 1/np.sqrt(1-betaf**2)

    integral = 0

    for nn in range(len(ccs)):# UV convergence parts
        
        # Use Lambda (UV) or the mass
        if nn > 0:

            m1 = LAMBDA_param_1[nn - 1]
            m2 = LAMBDA_param_1[nn - 1]

            lss = [m1,m2]
            

        else:
            m1 = m1t
            m2 = m2t

            lss = [m1,m2]


        if index == [[],[0,0],[0,0]]:

            integral += ccs[nn] * I00_tt(lss, kss)

        # case IA_nu;00;00
        if index[1:] == [[0,0],[0,0]]:
            if len(index[0]) != 1:
                raise ValueError('Implementation only for scalar or vector current') 

            Inu = P_f[index[0]] * I11_tt(lss, kss) + P_i[index[0]] * I12_tt(lss, kss)

            # lower the indices or the spatial part
            if index[0] != 0:
                Inu *= -1 

            integral += ccs[nn] * Inu


        elif index == [[0], [1,0], [1,0]]:

            I000 = Ef**3 * I31_tt(lss, kss) + Ei**3 * I32_tt(lss, kss) + (
                    3 * Ef**2 * Ei * I33_tt(lss, kss) 
                    + 3 * Ef * Ei**2 * I34_tt(lss, kss) 
                    - (3 * Ef * I35_tt(lss, kss))/4. 
                    - (3 * Ei * I36_tt(lss, kss))/4.)

            I300 = -Ef**2 * I31_tt(lss, kss) *Pfz +  - Ei**2 * I32_tt(lss, kss) * Piz + (
                    I33_tt(lss, kss) * (-2 * Ef * Ei * Pfz - Ef**2 * Piz)
                    + I34_tt(lss, kss) * (-Ei**2 * Pfz - 2 * Ef * Ei * Piz)
                    + (I35_tt(lss, kss) * Pfz)/4. 
                    + (I36_tt(lss, kss) * Piz)/4.)
                    
                    

            I003 = I300

            I303 = Ef * I31_tt(lss, kss) * Pfz**2 + Ei * I32_tt(lss, kss) * Piz**2 + (
                    I33_tt(lss, kss) * (Ei * Pfz**2 + 2 * Ef * Pfz * Piz) 
                    + I34_tt(lss, kss) * (2* Ei* Pfz* Piz + Ef * Piz**2)
                    + (Ef * I35_tt(lss, kss))/4. + (Ei * I36_tt(lss, kss))/4.)




            temp = 3 * (gammaf * betaf * gammai * betai * I000+
                 + gammaf * gammai * betai * I300
                 + gammai * gammaf * betaf * I003
                 + gammaf * gammai * I303)

            integral += ccs[nn] * temp





    # Make sure to return the original values to m1 and m2
    m1 = m1t
    m2 = m2t

    return integral




#CALCULATE STUFF
ener_shape = np.shape(Eistar)
I_An = np.ones(ener_shape) * complex(0.,0.)

if len(ener_shape) > 1: # for mesh inputs
    for mm, enirow in enumerate(Ei):
        for nn, enin in enumerate(enirow):
            enfin = Ef[mm,nn]
            PP_i = np.concatenate(([enin], lab_moment_i))
            PP_f = np.concatenate(([enfin], lab_moment_f))
            
            I_An[mm,nn] = make_int(PP_i, PP_f, indices)
            print('IAcalc: ', Eistar[mm,nn], Efstar[mm,nn], '---------', I_An[mm,nn])
else:
    for mm, enin in enumerate(Ei):
        enfin = Ef[mm]
        PP_i = np.concatenate(([enin], lab_moment_i))
        PP_f = np.concatenate(([enfin], lab_moment_f))

        I_An[mm] = make_int(PP_i, PP_f, indices)
        print('IAcalc: ', Eistar[mm], Efstar[mm], '---------', I_An[mm])


#Save the values
filename = IA_folder + 'IA_sig_[' + str(indices[0]) +';' +str(indices[1]) +';' +str(indices[2]) +(
']_vecPi_' + str(lab_moment_i_int[2]) + '_vecPf_' + str(lab_moment_f_int[2]) + '_L_' + str(int(L)) + '.npy')

if not os.path.exists(IA_folder):
    os.makedirs(IA_folder)

msgg = np.meshgrid(Eistar, Efstar)

with open(filename, 'wb') as f:

    np.save(f, I_An)

