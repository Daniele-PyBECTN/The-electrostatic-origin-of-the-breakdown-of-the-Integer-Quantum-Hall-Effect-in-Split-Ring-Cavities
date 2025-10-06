from numba import jit, njit
import kwant
from kwant.digest import uniform    # a (deterministic) pseudorandom number generator
import kwant.kpm


import scipy

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

import cmath

from qutip import *
from joblib import Parallel, delayed

from scipy.ndimage import gaussian_filter

import pandas as pd

def corr_dis(L, W, a, xic):
    Nx = int(L/a)
    Ny = int(W/a)
    Vxy = np.random.normal(0, 1., size=(Ny, Nx))/(a) + 1j*0.0
    
    kkVxy = np.fft.fft2(Vxy)
    kx = 2*np.pi * np.fft.fftfreq(kkVxy.shape[1],d=a)
    ky = 2*np.pi * np.fft.fftfreq(kkVxy.shape[0],d=a)
    kx, ky = np.meshgrid(kx,ky)
    
    kVxy = np.multiply(kkVxy, np.exp( - .125 * xic**2. * (kx**2.+ky**2.)  ) ) * xic/np.sqrt(2.)
    
    Vxy = np.fft.ifft2(kVxy)
    Vxy = Vxy - np.sum(Vxy)/(1.*Nx*Ny)

    return Vxy

def v_img_singleside(W, N, min_l):
    arr_x = np.linspace(min_l, W+min_l, N)
    arr_V = - 1./(np.abs( arr_x ))
    return arr_V


def v_img(W, N, min_l):
    arr_V = []
    arr_x = np.linspace(-W/2, W/2, N)
    d = 1000*W + 2*min_l
    for x in arr_x:
        V = 0
        nf = 50000
        n_Z = np.linspace( -nf, nf, 2*nf+1)
        for n in n_Z:
            if n != 0:
                V += ( 1./(2*np.abs(n)) - 1./(2.*np.abs(n+500*(x)/d )) )
        if x < 0:
            arr_V.append(V)
        else:
            arr_V.append(V)
    return np.asarray(arr_V)/d*1000


def v_expcutoff_img_singleside(W, N, min_l, xiV):
    arr_x = np.linspace(min_l, W+min_l, N)
    arr_V = - 1./(np.abs( arr_x ))*np.exp( -(arr_x-min_l)**2/(2*xiV**2) )
    #arr_V = - 1./(np.abs( arr_x )) /( np.exp( (arr_x-min_l - xiV)/(.01*xiV) ) + 1 )
    return arr_V

def v_bump_expcutoff_img_singleside(W, N, min_l, xiV, x_bump, Vbump):
    arr_x = np.linspace(min_l, W+min_l, N)
    arr_V = - 1./(np.abs( arr_x ))*np.exp( -(arr_x-min_l)**2/(2*xiV**2) )
    arr_V += Vbump*(arr_x-min_l-W+1)**2/W**2 * np.exp( - (np.abs(arr_x-min_l-W+1)/x_bump)**2 )
    return arr_V

def dos_fun(E_imgxx, E_dis, Vdis, Vimg, t, alpha, omB, Zeeman ):
    def dos_onsite(site):
            xi, yi = site.pos
            om0 = 4 * t * sigma_0  # Original onsite energy
            omRan = E_dis * Vdis[int(yi)][int(xi)] * sigma_0  # Disorder
            omImg = E_imgxx * Vimg[int(yi)] * sigma_0  # Image charge

            H_Zeeman = Zeeman / 2 * sigma_z  # Zeeman term (splits spin up/down)

            return om0 + omRan + omImg + H_Zeeman  # Return as a 2x2 matrix
    def dos_hopping(site_i, site_j):
        xi, yi = site_i.pos
        xj, yj = site_j.pos
        return - t*np.exp(-1j * 2*np.pi * alpha * (xi - xj) * (yi + yj)/2.)* sigma_0
    sys_nolead = kwant.Builder()
    sys_nolead[(lat(x, y) for x in range(L) for y in range(W))] = dos_onsite
    sys_nolead[lat.neighbors()] = dos_hopping
    sys_nolead = sys_nolead.finalized()

    #dos = kwant.kpm.SpectralDensity(sys_nolead, num_vectors=2000)#, bounds=[0, 1])
    dos = kwant.kpm.SpectralDensity(sys_nolead, energy_resolution = E_dis)# , bounds=[-1, 1] )
    
    return dos

@njit
def filling_fun_a(arr_mu, dos_energies, dos_densities, T, W, L, alpha):
    number_e = np.zeros( len(arr_mu) )
    N_dos = len(dos_energies)
    for ne, E in enumerate(arr_mu):
        number_e[ne]= np.sum( np.divide( dos_densities ,  (np.exp( ( dos_energies - E)/T )+1)  ) )/(1.*N_dos)
    
    return number_e/((W-1)*(L-1)*alpha)

@njit
def filling_fun(arr_mu, dos_energies, dos_densities, T, W, L, alpha):
    number_e = np.zeros( len(arr_mu) )
    N_dos = len(dos_energies)
    for ne, E in enumerate(arr_mu):
        for ndos in range(1,len(dos_densities)):
            dE = (dos_energies[ndos]-dos_energies[ndos-1])
            number_e[ne] += dE * dos_densities[ndos-1] / (np.exp( ( dos_energies[ndos-1] - E)/T )+1) 
    
    return number_e/((W-1)*(L-1)*alpha)

@njit
def filling_fun_0T(arr_mu, dos_energies, dos_densities, W, L, alpha):
    number_e = np.zeros( len(arr_mu) )
    N_dos = len(dos_energies)
    for ne, E in enumerate(arr_mu):
        indx_fermi = np.argmin(np.abs(E-dos_energies))
        dos_F = dos_densities[0:indx_fermi]
        for ndos in range(1,len(dos_F)):
            number_e[ne] += dos_densities[ndos-1]*(dos_energies[ndos]-dos_energies[ndos-1])
    
    return number_e/((W-1)*(L-1)*alpha)

@njit
def fermi_f(T, E, dE):
    if np.abs(E / T) < 100.0:
        if dE / T < 1.0:
            dF = - np.exp(E / T) / (np.exp(E / T) + 1.0) ** 2.0 / T * dE
        else:
            dF = 1.0
    else:
        dF = 0.0
    return dF

@njit
def finite_temp(T, G, energies):
    W = np.zeros( len(G) )
    num_points = len(G)
    
    for n in range(num_points):
        weighted_sum = 0.0
        norm_factor = 0.0
        for i in range(1,num_points):
            dE = np.abs(energies[i] - energies[i - 1])
            E = energies[i-1] - energies[n]
            dF = 0
            if np.abs(E / T) < 200.0:
                dF = - np.exp(E / T) / (np.exp(E / T) + 1.0) ** 2.0 / T * dE
            else:
                dF = 0.0
            norm_factor += dF
            weighted_sum += G[i-1] * dF
        W[n] = weighted_sum / norm_factor if norm_factor != 0 else 0.0
    
    return W

# Energy-Length-Magnetic scale (arbitrary) just to have a dimensional reference for the parameters
#a=0.6 # real lattice spacing in GaAs, in nm
#BandW0 = 40   #total bandwidth in meV
#l0 = np.sqrt(568/BandW0*8.) * 10**(-3.) # effective lattice spacing sqrt( \hbar^2/(2m*)/(\hbar J)) here I took \hbar^2/(2m*) = 568 nm^2 meV (GaAs) and \hbar J = BW/8 in micron!
#B0 = 4.14*10**3. / (l0 * 10**3.)**2. #in T, as 2\pi \hbar/e 1/a^2


lat = kwant.lattice.square(a=1, norbs=2)
Ncore = 5       #number of cores used in the calculation
iter_dis = 1
t = 1   # hopping rate for kwant
W = 200  # spatial width of the system ( Nx )
L = 200  # spatial length of the system ( Ny )

NET = 81    #number of states to compute T

# Define Pauli matrices
sigma_0 = np.array([[1, 0], [0, 1]])  # Identity (spin-independent)
sigma_x = np.array([[0, 1], [1, 0]])  # Pauli X
sigma_y = np.array([[0, -1j], [1j, 0]])  # Pauli Y
sigma_z = np.array([[1, 0], [0, -1]])  # Zeeman term

alpha0 = .003
iter_alpha = 300
dalpha = .01/iter_alpha
arr_alpha = np.linspace(alpha0, alpha0+dalpha*iter_alpha, iter_alpha)

Nelect_0 = 10*alpha0*(W-1)*(L-1)
Nelect_img = 2*alpha0*(W-1)*(L-1)


lB0 = 1./np.sqrt(2*np.pi*alpha0)
omB0 = 4*np.pi*alpha0

T = omB0/80

xic = lB0 * .5      # Correlation length and definition of disorder-potential array
Vdis = corr_dis(int(L), int(W), 1, xic)
plate_distance = 10*lB0      #edge-plate distance
Vimg0 = v_img_singleside(W, int(W), plate_distance)   # normalized image potential single plate
Vimg1 = v_expcutoff_img_singleside(W, int(W), plate_distance, 3*lB0)
Vimg2 = v_bump_expcutoff_img_singleside(W, int(W), plate_distance, 3*lB0, 5*lB0, omB0*3)

Vimg = Vimg0

V_img_min = np.amin(Vimg)
Vimg = Vimg - V_img_min - lB0/plate_distance**2 
E_img = omB0*lB0 * ( (plate_distance/lB0)**2 )/2. * 2         # Image charge amplitude in Zeeman
E_dis = omB0 * .03                   # disorder amplitude in cyclotrons

G0 = np.zeros(iter_alpha)
GIMG = np.zeros(iter_alpha)

def onsite(site, params):
    xi, yi = site.pos

    om0 = 4 * t * sigma_0  # Original onsite energy
    omRan = params.E_dis * params.V[int(yi)][int(xi)] * sigma_0  # Disorder
    omImg = params.E_img * Vimg[int(yi)] * sigma_0  # Image charge

    H_Zeeman = params.Zeeman / 2 * sigma_z  # Zeeman term (splits spin up/down)

    return om0 + omRan + omImg + H_Zeeman  # Return as a 2x2 matrix

def hopping(site_i, site_j, params):
    xi, yi = site_i.pos
    xj, yj = site_j.pos

    phase = -1j * 2 * np.pi * params.alpha * (xi - xj) * (yi + yj) / 2.
    
    return -t * np.exp(phase) * sigma_0  # Spin-preserving hopping

def onsite_lead(site, params):
    return 4 * t * sigma_0 + (params.Zeeman / 2) * sigma_z  # Zeeman in leads

def hopping_lead(site_i, site_j, params):
    return -t * sigma_0  # Spin-independent hopping

sys = kwant.Builder()
sys[(lat(x, y) for x in range(L) for y in range(W))] = onsite
sys[lat.neighbors()] = hopping

sym_left_lead = kwant.TranslationalSymmetry((-1, 0))
left_lead = kwant.Builder(sym_left_lead)
left_lead[(lat(0, y) for y in range(W))] = onsite_lead
left_lead[lat.neighbors()] = hopping_lead
sys.attach_lead(left_lead)
sys.attach_lead(left_lead.reversed())

sys = sys.finalized()

Zeeman0 = omB0/3
ei = 0 * omB0   #initial energy in cyclotrons
ef = 8. * omB0   #final energy (lowest)
Ne = 400        #number of points for the conductance sweep
energies = np.linspace( ei, ef, Ne)  # Define a range of fermi energies to scan
energies_fill = energies

dos0 = dos_fun(0, E_dis, Vdis, Vimg, 1, alpha0, omB0, Zeeman0 )    
dos0_energies, dos0_densities = dos0.energies, dos0.densities
filling0 = filling_fun(energies_fill, dos0_energies.real, (dos0_densities.real), T, W, L, alpha0)
#filling0 = filling_fun_0T(energies_fill, dos0_energies.real, (dos0_densities.real), W, L, alpha0)

idx_0 = np.argmin(np.abs(filling0 * alpha0*(W-1)*(L-1) - Nelect_0))
EF0 = energies[idx_0]

dosIMG = dos_fun(E_img, E_dis, Vdis, Vimg, 1, alpha0, omB0, Zeeman0 )
dosIMG_energies, dosIMG_densities = dosIMG.energies, dosIMG.densities
fillingIMG = filling_fun(energies_fill, dosIMG_energies.real, (dosIMG_densities.real), T, W, L, alpha0)
#fillingIMG = filling_fun_0T(energies_fill, dosIMG_energies.real, (dosIMG_densities.real), W, L, alpha0)


idx_img = np.argmin(np.abs(fillingIMG * alpha0*(W-1)*(L-1) - Nelect_img))
EFimg = energies[idx_img]

















def G_conductance(na, arr_alpha):
    alpha = arr_alpha[na]
    lB = 1./np.sqrt(2*np.pi*alpha)
    omB = 4*np.pi*alpha

    Zeeman = omB/3

    ei = 4.3 * omB0   #initial energy in cyclotrons
    ef = 6.1 * omB0   #final energy (lowest)
    Ne = 400        #number of points for the conductance sweep
    energies = np.linspace( ei, ef, Ne)  # Define a range of fermi energies to scan
    energies_fill = energies

    dos0 = dos_fun(0, E_dis, Vdis, Vimg, 1, alpha, omB0, Zeeman )    
    dos0_energies, dos0_densities = dos0.energies, dos0.densities
    filling0 = filling_fun(energies_fill, dos0_energies.real, (dos0_densities.real), T, W, L, alpha)

    idx_0 = np.argmin(np.abs(filling0 * alpha*(W-1)*(L-1) - Nelect_0))
    EF0 = energies[idx_0]

    dosIMG = dos_fun(E_img, E_dis, Vdis, Vimg, 1, alpha, omB0, Zeeman )
    dosIMG_energies, dosIMG_densities = dosIMG.energies, dosIMG.densities
    fillingIMG = filling_fun(energies_fill, dosIMG_energies.real, (dosIMG_densities.real), T, W, L, alpha)

    idx_img = np.argmin(np.abs(fillingIMG * alpha*(W-1)*(L-1) - Nelect_img))
    EFimg = energies[idx_img]

    params =SimpleNamespace( alpha=alpha, E_dis=E_dis, E_img=0, V=Vdis, Vimg=Vimg, Zeeman=Zeeman )
    params_img = SimpleNamespace( alpha=alpha, E_dis=E_dis, E_img=E_img, V=Vdis, Vimg=Vimg, Zeeman=Zeeman )
    
    
    if NET > 1:
        dT = 16*T/NET
        G0_EF_v = np.zeros(NET)
        GIMG_EF_v = np.zeros(NET)
        arr_en0 = np.zeros(NET)
        arr_enimg = np.zeros(NET)
        for nE in range(NET):
            
            ET0 = EF0 - 8*T + dT*nE
            arr_en0[nE] = ET0
            ETimg = EFimg - 8*T + dT*nE
            arr_enimg[nE] = ETimg
            
            smat = kwant.smatrix(sys, energy=ET0, args=[params])
            smat_img = kwant.smatrix(sys, energy=ETimg, args=[params_img])
            G0_EF = smat.transmission(0, 1)
            GIMG_EF = smat_img.transmission(0, 1)
            G0_EF_v[nE] = G0_EF
            GIMG_EF_v[nE] = GIMG_EF
            
        G0_EF = finite_temp(T, G0_EF_v, arr_en0)
        GIMG_EF = finite_temp(T, GIMG_EF_v, arr_enimg)
    else:
        smat = kwant.smatrix(sys, energy=EF0, args=[params])
        smat_img = kwant.smatrix(sys, energy=EFimg, args=[params_img])
        G0_EF = smat.transmission(0, 1)
        GIMG_EF = smat_img.transmission(0, 1)

    return G0_EF, GIMG_EF, filling0[idx_0], fillingIMG[idx_img], EF0, EFimg

# Define a function to compute conductance for a single iteration
def compute_conductance(na):
    return G_conductance( na, arr_alpha  )

# Use joblib to parallelize the loop
results = Parallel(n_jobs=Ncore)(
    delayed(compute_conductance)(na) for na in range(iter_alpha)
)

G0 = []
GIMG = []
arr_fil0 = []
arr_filimg = []
arr_E0 = []
arr_Eimg = []
# Aggregate the results
for sG0, sGIMG, fil0, filimg, EF0, EFimg in results:
    G0.append( sG0 )
    GIMG.append( sGIMG )
    arr_fil0.append( fil0 )
    arr_filimg.append( filimg)
    arr_E0.append( EF0 )
    arr_Eimg.append( EFimg )
G0 = np.asarray(G0)
GIMG = np.asarray(GIMG)
arr_fil0 = np.asarray(arr_fil0)
arr_filimg = np.asarray( arr_filimg)
arr_E0 = np.asarray(arr_E0)
arr_Eimg = np.asarray(arr_Eimg)

G0rr = G0
GIMGrr = GIMG

G0 = G0rr[:,int((NET-1)/2)]
GIMG = GIMGrr[:,int((NET-1)/2)]



nG = 0
# Parameters
params = {
    "t": t,
    "W": W,
    "L": L,
    "alpha0": alpha0,
    "xic": xic,
    "plate_distance": plate_distance,
    "E_dis": E_dis,
    "E_img": E_img,
    "omB0": omB0,
    "Zeeman0": Zeeman0,
    "T": T,
    "Nelect_0": Nelect_0,
    "Nelect_img": Nelect_img,
    "iter_dis": iter_dis,
    "Ncore": Ncore
}

# Create a DataFrame
data = pd.DataFrame({
    'alpha': arr_alpha,
    'G0': G0,
    'GIMG': GIMG
})

# Write parameters and data to a file
filename = 'G_%d.dat' % (nG)
with open(filename, 'w') as f:
    # Write parameters as comments
    for key, value in params.items():
        f.write(f"# {key} = {value}\n")
    # Write the data
    data.to_csv(f, sep=' ', index=False, header=True)
    
nG = 0
# Parameters
params = {
    "t": t,
    "W": W,
    "L": L,
    "alpha0": alpha0,
    "xic": xic,
    "plate_distance": plate_distance,
    "E_dis": E_dis,
    "E_img": E_img,
    "omB0": omB0,
    "Zeeman0": Zeeman0,
    "T": T,
    "Nelect_0": Nelect_0,
    "Nelect_img": Nelect_img,
    "iter_dis": iter_dis,
    "Ncore": Ncore
}

# Create a DataFrame
data = pd.DataFrame({
    'alpha': arr_alpha,
    'filling0': arr_fil0,
    'fillingIMG': arr_filimg
})

# Write parameters and data to a file
filename = 'filling_%d.dat' % (nG)
with open(filename, 'w') as f:
    # Write parameters as comments
    for key, value in params.items():
        f.write(f"# {key} = {value}\n")
    # Write the data
    data.to_csv(f, sep=' ', index=False, header=True)
    
nG = 0
# Parameters
params = {
    "t": t,
    "W": W,
    "L": L,
    "alpha0": alpha0,
    "xic": xic,
    "plate_distance": plate_distance,
    "E_dis": E_dis,
    "E_img": E_img,
    "omB0": omB0,
    "Zeeman0": Zeeman0,
    "T": T,
    "Nelect_0": Nelect_0,
    "Nelect_img": Nelect_img,
    "iter_dis": iter_dis,
    "Ncore": Ncore
}

# Create a DataFrame
data = pd.DataFrame({
    'alpha': arr_alpha,
    'EF0': arr_E0,
    'EFimg': arr_Eimg
})

# Write parameters and data to a file
filename = 'E_Fermi_%d.dat' % (nG)
with open(filename, 'w') as f:
    # Write parameters as comments
    for key, value in params.items():
        f.write(f"# {key} = {value}\n")
    # Write the data
    data.to_csv(f, sep=' ', index=False, header=True)