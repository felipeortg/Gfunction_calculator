#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2026-01-23 16:45:21
# @Author  : Felipe G. Ortega-Gama (felipeortegagama@gmail.com)
# @Version : 1.0
# IA calculation library


from kinematic_utilities import *
np.seterr(all = 'warn')
from scipy import integrate
from scipy import optimize

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


# Singularities of the integrand
def avoid_points(m1, m2, si, sf, q2):

    avoid = []

    # Divergences due to the division
    AA2pB_0 = (-4*m2**2)/si + (-m1**2 + m2**2 + si)**2/si**2 
    
    AA2pB_1 = ((-4*(m1**2 - m2**2 - sf))/si + (2*(q2 - sf - si)*(-m1**2 + m2**2 + si))/si**2)
    
    AA2pB_2 = ((q2 - sf - si)**2/si**2 - (4*sf)/si)

    if AA2pB_2 == 0 and AA2pB_1 == 0:
        pass

    elif AA2pB_2 == 0:
        xx = np.roots([AA2pB_1,AA2pB_0])
        
        if np.real(xx[0]) > 0 and np.real(xx[0]) < 1:
            avoid.extend([xx[0]])

    else:
    
        xxs = np.roots([AA2pB_2,AA2pB_1,AA2pB_0])

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


class IA_calculator:
    def __init__(self, config_file):

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

        # UV regulators
        ccslist = cval['ccs'].split()
        ccs = [float(ll)/float(ccslist[-1]) for ll in ccslist[:-1]]

        LAMBDA_param_1 = [int(ll) * m1 for ll in cval['LAMB'].split()]

        self.conf_params = [L, m1, m2, indices, alpha, ccs, LAMBDA_param_1]

        self.set_spat = False
        print("Configuration for IA loaded successfully.")



    def set_spatial_momentum(self, P_i, P_f):
        self.set_spat = True

        L = self.conf_params[0]

        self.Pi = (2*np.pi/L)*np.array(P_i)
        self.Pf = (2*np.pi/L)*np.array(P_f)

        print("Momenta set to 2pi/L ", P_i, P_f)

    # -----------------
    # General macro to do the Sum Pf \neq Pi
    # -----------------

    def make_int_ECM(self, ECM_initial, ECM_final):
        Pivec2 = sum(self.Pi**2)
        Pfvec2 = sum(self.Pf**2)

        E_initial = np.sqrt(ECM_initial**2 + Pivec2)
        E_final = np.sqrt(ECM_final**2 + Pfvec2)

        return self.make_int(E_initial, E_final)

    def make_int(self, E_initial, E_final):
        if not self.set_spat:
            raise ValueError("Momenta not set, do set_spatial_momentum.")


        L, m1, m2, index, alpha, ccs, LAMBDA_param_1 = self.conf_params

        P_i = np.array([E_initial, self.Pi[0], self.Pi[1], self.Pi[2]])

        P_f = np.array([E_final, self.Pf[0], self.Pf[1], self.Pf[2]])

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

                lss = [LAMBDA_param_1[nn - 1],LAMBDA_param_1[nn - 1]]
            else:
                lss = [m1,m2]

            if index[1:] == [[0,0],[0,0]]:
                if len(index[0]) > 1:
                    raise ValueError('Implementation only for scalar or vector current') 

                if index[0] == []:

                    integral += ccs[nn] * I00_tt(lss, kss)
                    continue

                Inu = P_f[index[0][0]] * I11_tt(lss, kss) + P_i[index[0][0]] * I12_tt(lss, kss)

                # # lower the indices for the spatial part
                # if index[0][0] != 0:
                #     print("lowered indices")
                #     Inu *= -1 

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


        return integral


