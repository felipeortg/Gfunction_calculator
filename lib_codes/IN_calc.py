#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2026-01-23 16:45:21
# @Author  : Felipe G. Ortega-Gama (felipeortegagama@gmail.com)
# @Version : 1.0
# Sum calculation library


from kinematic_utilities import *

from scipy import integrate
from scipy import optimize
from scipy.spatial.transform import Rotation

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


class IN_calculator:
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

        self.ang_integral_method = 'adap_quad'

        self.set_spat = False
        print("Configuration for IN loaded successfully.")

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


        L, m1, m2, indices, alpha, ccs, LAMBDA_param_1 = self.conf_params

        self.P_i = np.array([E_initial, self.Pi[0], self.Pi[1], self.Pi[2]])

        self.P_f = np.array([E_final, self.Pf[0], self.Pf[1], self.Pf[2]])

        coeffs, list_indices = get_rotboost_indices(indices, self.P_i, self.P_f)


        result = 0

        for coef, ix in zip(coeffs, list_indices):
            if coef==0:
                continue

            result += coef * self.axial_int(ix)

        return result



    def axial_int(self, index):
        """
        index is a list, 
        first element a list of the 4-vector indices
        second element the value of lf, mf
        third element the value of li, mi
        we probably want to use np.isclose(x, 0.0)

        This implementation performs a single angular integral
        The radial integral is done for every evaluation needed for the ang integration
        """

        L, m1, m2, indices, alpha, ccs, LAMBDA_param_1 = self.conf_params

        integral = 0

        Eicm, Efcm, gamma = get_cm_gamma(self.P_i, self.P_f)

        #Some useful kinematics (defined in the initial cm frame)
        beta = np.sqrt(1-1/gamma**2)
        Ei = Eicm

        Ef = gamma*Efcm
        Pfimag = gamma*beta*Efcm

        q2star_i = 0.25 * (Eicm**2 - 2*(m1**2 + m2**2) + (m1**2 - m2**2)**2/Eicm**2) 
        q2star_f = 0.25 * (Efcm**2 - 2*(m1**2 + m2**2) + (m1**2 - m2**2)**2/Efcm**2)

        q2 = Efcm**2 + Eicm**2 - 2*gamma*Efcm*Eicm

        self.initial_CM_kins = [Ei, Ef, Pfimag, gamma, beta, q2star_i, q2star_f, q2]

        
        
        if self.ang_integral_method == 'adap_quad':
        
            # Remember the 2pi from the phi integral
            # integral done in variable z=costh, with limits (-1,1)

            val_temp = integrate.quad(self.azimuthal_IN, -1, 1, epsrel = 1e-6,
                args = (index), full_output=0)[0]

            integral = 1./(2 * np.pi)**2 * val_temp


        # TODO ADAPT CODE OF FIX QUAD version                
        # elif self.ang_integral_method == 'fix_quad':
                  
        #     tolerance = 1e-4
        #     error = 1
        #     fix_ord = 10

        #     def Ang_Integrand_IN_nonphi(k_dum_z, Eicm, Efcm, gamma, alpha, index):

        #         reslts = [Ang_Integrand_IN_a( zs, Eicm, Efcm, gamma, alpha, index)
        #                  for zs in k_dum_z]
                
        #         # Remember the 2pi from the phi integral
        #         return 2 * np.pi * np.array(reslts)

            
        #     val_temp_old = integrate.fixed_quad(Ang_Integrand_IN_nonphi, -1, 1,
        #         args = (Eicm, Efcm, gamma, alpha, index), n = fix_ord)[0]
                
        #     print(val_temp_old)

        #     fix_ord += 1
        #     while error > tolerance:
                
        #         val_temp_new = integrate.fixed_quad(Ang_Integrand_IN_nonphi, -1, 1,
        #             args = (Eicm, Efcm, gamma, alpha, index), n = fix_ord)[0]
                
        #         error = np.abs(val_temp_new - val_temp_old)/np.abs(val_temp_old)
                
        #         val_temp_old = val_temp_new
                
        #         print(val_temp_old)
                
        #         fix_ord += 1
                
        #         if fix_ord > 10:
        #             print('order bigger 10, E:', error)
                    
        #         if fix_ord > 15:
        #             print('order 15, E:', error)
        #             break
                        

        #     integral = 1./(2 * np.pi)**3 * val_temp_new
                

        else:
            raise ValueError('not implemented')


        return integral


    def azimuthal_IN(self, k_dum_z, index):
        """
        k_dum_z is the cosine of the azimutal angle
        index: list of the 4-vector indices, restricted to vector or none (S-wave for initial and final)
        """

        # Split the radial integral into separate regions, hoping to give an easier time to the adaptive method
        # quad integrate up to int_upperbound
        # quad integrate from int_upperbound to infty

        L, m1, m2, indices, alpha, ccs, LAMBDA_param_1 = self.conf_params


        if len(LAMBDA_param_1) > 0:
            int_upperbound = 10*LAMBDA_param_1[-1]
        else:
            int_upperbound = 10*m1
        
        # int_upperbound k ~ 10 Lambda 
        # By this point also the exponential factors should be negligible        
        # otherwise make negligible by integrating up to exp(- a k^4)~1e-10
        if np.exp(-alpha * int_upperbound**4) > 1e-10:
            int_upperbound = (10./alpha * np.log(10))**(.25)


        # integrand in theta
        f_th = 0
        # do the integral in the small region
        f_th = integrate.quad(self.radial_kz_dep_IN_small_region, 
            0, int_upperbound, 
            epsrel = 1e-6,
            args = (k_dum_z, index))[0]

        f_th += integrate.quad(self.radial_kz_dep_IN_large_region, 
            int_upperbound, np.inf,
            epsrel = 1e-6,
            args = (k_dum_z, index))[0]

        return f_th


    def radial_kz_dep_IN_small_region(self, k_dum, k_dum_z, index):
        """
        define small region radial integral
        will use adap: no need to vectorize k_dum
        calculate the radial integral in the small region
        """
        
        L, m1, m2, indices, alpha, ccs, LAMBDA_param_1 = self.conf_params

        Ei, Ef, Pfimag, gamma, beta, q2star_i, q2star_f, q2 = self.initial_CM_kins

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
                m1t = LAMBDA_param_1[nn - 1]
                m2t = LAMBDA_param_1[nn - 1]

            else:
                m1t = m1
                m2t = m2


            # Have to re calculate due to PV mass dependence
            omega_Pfk1 = np.sqrt(Pfimag**2 - 2 * Pfimag * k_dum * k_dum_z + k2_dum + m1t**2)

            omega_k1 = np.sqrt(k2_dum + m1t**2)
            omega_k2 = np.sqrt(k2_dum + m2t**2)


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
                    Hm1_DD = - alpha * np.sqrt(q2star_f + m2t**2)/(2 * Ei * Efcm)

                elif qi_singular:
                    Hm1_DD = alpha * (k2star_f - q2star_f)/(2*Ei) * (
                        1./((Ef - omega_k2)**2 - omega_Pfk1**2))

                elif qf_singular:
                    Hm1_DD = 1./(2 * omega_k2) * (
                        alpha * (k2star_i - q2star_i) * np.sqrt(q2star_f + m2t**2)/Efcm) *(
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

        
        # Remember the k^2 from the integral measure
        return k2_dum * integrand


    def radial_kz_dep_IN_large_region(self, k_dum, k_dum_z, index):
        """
        define large region radial integral
        will use adap: no need to vectorize k_dum
        calculate the radial integral in the large region where cutoff HH factor can be neglected
        """
        
        L, m1, m2, indices, alpha, ccs, LAMBDA_param_1 = self.conf_params

        Ei, Ef, Pfimag, gamma, beta, q2star_i, q2star_f, q2 = self.initial_CM_kins

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
                m1t = LAMBDA_param_1[nn - 1]
                m2t = LAMBDA_param_1[nn - 1]

            else:
                m1t = m1
                m2t = m2


            # Have to re calculate due to PV mass dependence
            omega_Pfk1 = np.sqrt(Pfimag**2 - 2 * Pfimag * k_dum * k_dum_z + k2_dum + m1t**2)

            omega_k1 = np.sqrt(k2_dum + m1t**2)
            omega_k2 = np.sqrt(k2_dum + m2t**2)


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

        
        # Remember the k^2 from the integral measure
        return k2_dum * integrand
