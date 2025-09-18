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


def triangular_lattice( Nx, Ny, n_print ):
    N = Nx*Ny
    R = np.zeros(shape=(N,2))
    n=0
    for y in range(Ny):
        for x in range(Nx):
            if y%2==0:
                R[n][0] = x
                R[n][1] = y*(np.sqrt(3.)/2.)
            else:
                R[n][0] = x + .5
                R[n][1] = y*(np.sqrt(3.)/2.)
            n += 1
    M = n

    if n_print == 0:
        p_lattice = open("data/lattice.dat", "w")
        for i in range(M):
            color = 0.0
            p_lattice.write("%f %f %f\n" % (1.*R[i][0], 1.*R[i][1], color))
            p_lattice.flush()
        p_lattice.close()

    return R, int(M)

def triang_triangular_lattice( Nx, Ny, n_print ):
    N = Nx*Ny
    R = np.zeros(shape=(N,2))
    n=0
    for y in range(Ny):
        if y < Nx:
            for x in range(Nx-y):
                if y%2==0:
                    R[n][0] = x + .5*(y)
                    R[n][1] = y*(np.sqrt(3.)/2.)
                else:
                    R[n][0] = x + .5 + .5*(y-1)
                    R[n][1] = y*(np.sqrt(3.)/2.)
                n += 1
    M = n

    if n_print == 0:
        p_lattice = open("data/lattice.dat", "w")
        for i in range(M):
            color = 0.0
            p_lattice.write("%f %f %f\n" % (1.*R[i][0], 1.*R[i][1], color))
            p_lattice.flush()
        p_lattice.close()

    return R, int(M)

def square_lattice( Nx, Ny, n_print ):
    N = Nx*Ny
    R = np.zeros(shape=(N,2))
    n=0
    for y in range(Ny):
        for x in range(Nx):
            R[n][0] = x
            R[n][1] = y
            n += 1
    M = n

    if n_print == 0:
        p_lattice = open("data/lattice.dat", "w")
        for i in range(M):
            color = 0.0
            p_lattice.write("%f %f %f\n" % (1.*R[i][0], 1.*R[i][1], color))
            p_lattice.flush()
        p_lattice.close()

    return R, int(M)
