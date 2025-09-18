from numba import jit, njit

import numpy as np
import scipy
import math
from numpy.linalg import inv #routine to invert matrix
import matplotlib.pyplot as plt
from qutip import *
import time
import sys
from scipy.interpolate import interp1d
from scipy.special import factorial


def disp_flat_emitters(omega_0, M, dis):
    omega_ = np.zeros(shape=(M), dtype=np.complex_)
    omega_ =  omega_0  * np.ones(shape=(M)) + dis * np.random.normal(0., 1., M) + 1j*0
    return omega_

def ph_electric_field(E, M, R):

    omegaE = np.zeros(shape=(M), dtype=np.complex_)
    x_max = np.amax(R[:,0])
    x_min = np.amin(R[:,0])
    for n in range(M):
        omegaE[n] = E*(R[n][0] - x_min) + 1j*0
    return omegaE

def ph_confinement(E, M, alpha, R):

    omegaE = np.zeros(shape=(M), dtype=np.complex_)
    x_max = np.amax(R[:,0])
    x_min = np.amin(R[:,0])
    print(x_max, x_max)
    for n in range(M):
        omegaE[n] = E*np.abs(R[n][0] - (x_max+x_min)/2 )**alpha/np.abs(x_max-x_min)**alpha + 1j*0
    return omegaE

def omega_cavity(R, Nx, A):
    Xm = (np.amax(R[:,0])-np.amin(R[:,0]))/2.
    Ym = (np.amax(R[:,1])-np.amin(R[:,1]))/2.
    sigmaX = Nx/6.
    Ne = np.multiply(np.exp( - (R[:,0]-Xm)**2./(2.*sigmaX**2.) ), -(R[:,1]-Ym)**2. )
    Ne = Ne/np.amax(np.abs(Ne))
    return Ne*A

def ph_quadrupole_field(E, M, R, Ny, Nx):

    omegaE = np.zeros(shape=M,dtype=np.complex_)
    x_max = np.amax(R[:][0])
    x_min = np.amin(R[:][0])
    p_V = open("data/V.dat", "w")
    for n in range(M):

        #if R[n][1] < Ny/2.:
        #    omegaE[n] = E*(R[n][0] - x_min)
        #elif R[n][1] > Ny/2.:
        #    omegaE[n] = E*(x_max - R[n][0])



        #omegaE[n] = E*( np.square(R[n][0] - Nx/2.+.5) - np.square(R[n][1] - Ny/2.+.5) )
        omegaE[n] = E*( np.square(R[n][0] - Nx/2.+.5) - np.square(R[n][1] - Ny/2.+.5) + 2.*1j*(R[n][1] - Ny/2.+.5)*(R[n][0] - Nx/2.+.5) ) + 1j*0
        
        """p_V.write("%f " % (omegaE[n].imag) )
        p_V.flush()
        if (n+1) % Nx == 0:
            p_V.write("\n")
            p_V.flush()"""
        #omegaE[n] = E*R[n][0]*R[n][1]
    return omegaE


@njit
def hopping_matrix_HH( J, R, M, alpha ):    
    V = np.zeros(shape=(M,M), dtype=np.complex_)
    for i in range(M):
        for j in range(M):
            r = np.sum( np.square(R[i]-R[j]) )
            if r <= 1.01 and j > i :    #nearest neighbour hopping condition
                if np.abs(R[i][0]-R[j][0]) > 0 and int(np.abs(R[i][1]-R[j][1])) == 0: #hop on x
                    V[i][j] = J + 1j*0
                    V[j][i] = np.conj(V[i][j])
                elif np.abs(R[i][1]-R[j][1]) > 0 and int(np.abs(R[i][0]-R[j][0])) == 0: #hop on y (complex flux)
                    V[i][j] = J * np.exp( -1j*2.*np.pi*alpha*R[i][0] ) + 1j*0
                    V[j][i] = np.conj(V[i][j])
    return V


@njit
def inhomo_hopping_matrix_HH( J, R, M, alpha ):    #here alpha is an array so B can be non homogeneous
    V = np.zeros(shape=(M,M), dtype=np.complex_)
    for i in range(M):
        for j in range(M):
            r = np.sum( np.square(R[i]-R[j]) )
            if r <= 1.01 and j > i :    #nearest neighbour hopping condition
                if np.abs(R[i][0]-R[j][0]) > 0 and int(np.abs(R[i][1]-R[j][1])) == 0: #hop on x
                    V[i][j] = J + 1j*0
                    V[j][i] = np.conj(V[i][j])
                elif np.abs(R[i][1]-R[j][1]) > 0 and int(np.abs(R[i][0]-R[j][0])) == 0: #hop on y (complex flux)
                    V[i][j] = J * np.exp( -1j*2.*np.pi*alpha[i]*R[i][0] ) + 1j*0
                    V[j][i] = np.conj(V[i][j])
    return V

@njit
def symm_hopping_matrix_HH( J, R, M, alpha ):
    xmax = np.amax(R[:,0])
    xmin = np.amin(R[:,0])
    ymax = np.amax(R[:,1])
    ymin = np.amin(R[:,1])
    V = np.zeros(shape=(M,M), dtype=np.complex_)
    for i in range(M):
        for j in range(M):
            r = np.sqrt(np.sum( np.square(R[i]-R[j]) ) )
            if r <= 1.1 and j > i :    #nearest neighbour hopping condition
                V[i][j] = J * np.exp( - 1j*np.pi*alpha*(R[i][1]-0)*(R[i][0]-R[j][0]) + 1j*np.pi*alpha*(R[i][0]-0)*(R[i][1]-R[j][1]) )
                V[j][i] = np.conj(V[i][j])
            """if np.abs(R[i][0]-R[j][0]) == (xmax - xmin) and j > i :    #PBC on x
                 V[i][j] = J * np.exp( - 1j*np.pi*alpha*(R[i][1]-0)*(R[i][0]-R[j][0]) + 1j*np.pi*alpha*(R[i][0]-0)*(R[i][1]-R[j][1]) )
                 V[j][i] = np.conj(V[i][j])
            if np.abs(R[i][1]-R[j][1]) == (ymax - ymin) and j > i :    #PBC on y
                 V[i][j] = J * np.exp( - 1j*np.pi*alpha*(R[i][1]-0)*(R[i][0]-R[j][0]) + 1j*np.pi*alpha*(R[i][0]-0)*(R[i][1]-R[j][1]) )
                 V[j][i] = np.conj(V[i][j])"""
    return V

@njit
def PBC_hopping_matrix_HH( J, R, M, Nx, Ny, alpha, pbc ):
    V = np.zeros(shape=(M,M), dtype=np.complex_)
    for i in range(M):
        for j in range(M):
            r = np.sum( np.square(R[i]-R[j]) )
            if r <= 1.01 and j > i :    #nearest neighbour hopping condition
                if np.abs(R[i][0]-R[j][0]) > 0 and int(np.abs(R[i][1]-R[j][1])) == 0: #hop on x
                    V[i][j] = J+ 1j*0
                    V[j][i] = np.conj( V[i][j] )
                elif np.abs(R[i][1]-R[j][1]) > 0 and int(np.abs(R[i][0]-R[j][0])) == 0: #hop on y (complex flux)
                    V[i][j] = J * np.exp( - 1j*2.*np.pi*alpha*R[i][0] )
                    V[j][i] = np.conj( V[i][j] )
            if np.abs(R[i][0]-R[j][0]) >= Nx-1.1 and np.abs(R[i][1]-R[j][1]) < .1 and i > j and np.abs(R[i][0]-R[j][0]) > 0 and pbc == 1:    #PERIODIC BOUNDARY ON X
                V[i][j] = J
                V[j][i] = np.conj( V[i][j] )
            if np.abs(R[i][1]-R[j][1]) >= Ny-1.1 and np.abs(R[i][0]-R[j][0]) < .1 and i > j and np.abs(R[i][1]-R[j][1]) > 0:    #PERIODIC BOUNDARY ON Y
                V[i][j] = J * np.exp( - 1j*2.*np.pi*alpha*R[i][0] )
                V[j][i] = np.conj( V[i][j] )
    return V

@njit
def Y_PBC_hopping_matrix_HH( J, R, M, Nx, Ny, alpha, pbc ):
    V = np.zeros(shape=(M,M), dtype=np.complex_)
    for i in range(M):
        for j in range(M):
            r = np.sum( np.square(R[i]-R[j]) )
            if r <= 1. and j > i :    #nearest neighbour hopping condition
                if np.abs(R[i][0]-R[j][0]) > 0 and int(np.abs(R[i][1]-R[j][1])) == 0: #hop on x
                    V[i][j] = J * np.exp( - 1j*2.*np.pi*alpha*R[i][1] )
                    V[j][i] = np.conj( V[i][j] )
                elif np.abs(R[i][1]-R[j][1]) > 0 and int(np.abs(R[i][0]-R[j][0])) == 0: #hop on y (complex flux)
                    V[i][j] = J + 1j*0
                    V[j][i] = np.conj( V[i][j] )
            if np.abs(R[i][1]-R[j][1]) >= Ny-1.1 and np.abs(R[i][0]-R[j][0]) < .1 and i > j and np.abs(R[i][1]-R[j][1]) > 0 and pbc == 1:    #PERIODIC BOUNDARY ON Y
                V[i][j] = J
                V[j][i] = np.conj( V[i][j] )
            if np.abs(R[i][0]-R[j][0]) >= Nx-1.1 and np.abs(R[i][1]-R[j][1]) < .1 and i > j and np.abs(R[i][0]-R[j][0]) > 0:    #PERIODIC BOUNDARY ON X
                V[i][j] = J * np.exp( - 1j*2.*np.pi*alpha*R[i][1] )
                V[j][i] = np.conj( V[i][j] )
    return V


def V_perturb( R, M, Nx, Ny, g ):
    V_ = np.zeros(shape=(M), dtype=np.complex)

    #indx =  Nx*(int(Ny/2.)) + int((Nx-1)/2.) -1
    indx =  int(.5*Nx)  #pert centered in the middle of the bottom edge

    sigmaX = 1
    sigmaY = 1.

    #Gaussian pert
    V_ = g*np.exp( - np.square( R[:,0] - R[indx][0] )/(2*sigmaX*sigmaX) - np.square( R[:,1] - R[indx][1] )/(2*sigmaY*sigmaY) )/(2*np.pi*sigmaX*sigmaY)

    #Delta pert
    #V_[indx] = g

    p_V = open("data/V.dat", "w")
    count1 = 0
    for num_p in V_:
        p_V.write("%f " % ( num_p.real ) )
        p_V.flush()
        if (count1+1) % Nx == 0:
            p_V.write("\n")
            p_V.flush()
        count1 += 1

    #sys.stdout.write("\t\t\t -> /time_evo ---->> Delta position = [%.0f,%.0f]\n" % ( 1.*R[indx][0], 1.*R[indx][1] ) )
    #sys.stdout.flush()

    return V_


def print_hopping(arrJ):
    evalsJ, evecsJ = np.linalg.eig(arrJ)
    idx = evalsJ.argsort()
    evalsJ = evalsJ[idx]
    evecsJ = evecsJ[:,idx]

    p_evals = open("data/evalsJ.dat", "w")
    for e in evalsJ:
        p_evals.write("%f\n" % (e.real-evalsJ[0].real) )
        p_evals.flush()
    p_evals.close()

    return evalsJ[0], evecsJ[:,0]
