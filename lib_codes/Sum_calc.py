#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2026-01-23 16:45:21
# @Author  : Felipe G. Ortega-Gama (felipeortegagama@gmail.com)
# @Version : 1.0
# Sum calculation library


from kinematic_utilities import *
import pickle


class Sum_calculator:
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


        cube_num = int(cval['cube_num'])

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
        # Location to get triplets from:
        trip_folder = './'

        # -----------------
        # Get the array of n triplets
        triplets_filename = trip_folder + 'triplets/n_list_r<' + str(cube_num) + '.txt'
        with open(triplets_filename, 'rb') as f:
            n_list = pickle.load(f)

        n_arr = np.array(n_list)

        # Get array of magnitude k vectors 
        k_arr = (2*np.pi/L) * n_arr

        k2_array = np.sum(k_arr*k_arr, axis = 1)
            
        k_len = len(k2_array)

        self.conf_params = [L, m1, m2, indices, alpha, k_arr, k2_array, k_len]

        self.set_spat = False
        print("Configuration for sum loaded successfully.")

    def set_spatial_momentum(self, P_i, P_f):
        self.set_spat = True

        L = self.conf_params[0]

        self.Pi = (2*np.pi/L)*np.array(P_i)
        self.Pf = (2*np.pi/L)*np.array(P_f)

        print("Momenta set to 2pi/L ", P_i, P_f)

    # -----------------
    # General macro to do the Sum Pf \neq Pi
    # -----------------

    def Neq_sum_ECM(self, ECM_initial, ECM_final):
        Pivec2 = sum(self.Pi**2)
        Pfvec2 = sum(self.Pf**2)

        E_initial = np.sqrt(ECM_initial**2 + Pivec2)
        E_final = np.sqrt(ECM_final**2 + Pfvec2)

        return self.Neq_sum(E_initial, E_final)

    def Neq_sum(self, E_initial, E_final):
        if not self.set_spat:
            raise ValueError("Momenta not set, do set_spatial_momentum.")


        L, m1, m2, index, alpha, k_arr, k2_array, k_len = self.conf_params

        P_i = [E_initial, self.Pi[0], self.Pi[1], self.Pi[2]]

        P_f = [E_final, self.Pf[0], self.Pf[1], self.Pf[2]]

        # Extract some variables

        # Initial frame variables
        Ei = P_i[0]

        Pivec = np.sqrt(sum([P_i[ii]**2 for ii in range(1,4)]))

        Eicm = np.sqrt(Ei**2 - Pivec**2)

        Lambdai = boost(P_i)
        # qstar doesn't inherit the m/Lambda dependence (only used in cutoff)
        q2star_i = 0.25 * (Eicm**2 - 2*(m1**2 + m2**2) + (m1**2 - m2**2)**2/Eicm**2) 



        # Final frame variables

        Ef = P_f[0]

        Pfvec = np.sqrt(sum([P_f[ii]**2 for ii in range(1,4)]))

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


