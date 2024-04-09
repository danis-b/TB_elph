import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from numba import jit, prange
import time

def kmesh_preparation(kmesh, cell_vec): 
    
    num_kpoints = kmesh[0] * kmesh[1] * kmesh[2]
    rec_vec = np.zeros((3,3))
    k_vec  = np.zeros((num_kpoints,3))

    rec_vec[0]  = (2 * np.pi / np.linalg.det(cell_vec)) * np.cross(cell_vec[1], cell_vec[2])
    rec_vec[1]  = (2 * np.pi / np.linalg.det(cell_vec)) * np.cross(cell_vec[2], cell_vec[0])
    rec_vec[2]  = (2 * np.pi / np.linalg.det(cell_vec)) * np.cross(cell_vec[0], cell_vec[1])

    q = 0
    for q1 in range(-int(0.5 * kmesh[0]),  int(0.5 * kmesh[0])):
        for q2 in range(-int(0.5 * kmesh[1]),  int(0.5 * kmesh[1])):
            # to avoid divergence
            if(q1 == 0 and q2 == 0):
                k_vec[q] = (rec_vec[0] * q1/ kmesh[0]) + (rec_vec[1] * q2 / kmesh[1]) + np.array([0.01, 0.0, 0.0])
            else:
                k_vec[q] = (rec_vec[0] * q1/ kmesh[0]) + (rec_vec[1] * q2 / kmesh[1])
            q +=1
                
                
    print('... kmesh is ready')
    return k_vec 


def energy_contour_preparation(ncol, nrow, e_fermi, e_low, smearing):
    # prepare the energy contour for integration
    num_freq = ncol + 2 * nrow
    de = complex((e_fermi - e_low) / ncol, smearing / nrow)

    freq = np.zeros(num_freq, dtype=np.complex128)
    d_freq = np.zeros(num_freq, dtype=np.complex128)
    e_const = complex(e_low, 0)

    idx_x = 0
    idx_y = 1

    idx = 0
    for i in range(num_freq):
        if (i == nrow):
            idx_x = 1
            idx_y = 0
            idx = 0
            e_const = complex(e_low, smearing)
        elif (i == nrow + ncol):
            idx_x = 0
            idx_y = -1
            idx = 0
            e_const = complex(e_fermi, smearing)
        
        freq[i] = e_const + complex(idx * idx_x * (de).real, idx * idx_y * (de).imag)
        d_freq[i] = complex(idx_x * (de).real, idx_y * (de).imag)
        idx = idx+1

    return freq, d_freq



@jit(nopython=True)
def calc_exchange(index_temp, num_freq, cell_vec, k_vec, E, dE, Ham_K, selfen):
    
    num_kpoints = k_vec.shape[0]
    weight = 1/num_kpoints
 
    corr_greenK = np.zeros(2, dtype=np.complex128)
    r = index_temp[0] * cell_vec[0] + index_temp[1] * cell_vec[1]
    
    exchange = 0.0

    phase = np.zeros((num_kpoints), dtype=np.complex128)
    for e in range(num_kpoints):
        phase[e] = np.exp( 1j * np.dot(k_vec[e],r) )

    for num in range(num_freq):
        delta_i = 0
        greenR_ij = 0
        greenR_ji = 0

        for  e in range(num_kpoints):
            for z in range(2):
                #G = 1/(E - H)
                
                corr_greenK[z] = 1/((E[num] - Ham_K[e, z]) - selfen[z, e, num])  
                
            delta_i += weight * (Ham_K[e, 0] - Ham_K[e, 1] + selfen[0, e, num] - selfen[1, e, num])
            greenR_ij += weight * phase[e] * corr_greenK[1] 
            greenR_ji += weight * np.conj(phase[e]) * corr_greenK[0]
            
        dot_product = delta_i * greenR_ij * delta_i * greenR_ji 
        exchange -= (2/np.pi) * (dot_product * dE[num]).imag
       
    return exchange


@jit(nopython=True)
def calc_green(freq, k_vec, Ham_K, selfen, smearing):
    
    num_kpoints = k_vec.shape[0]
    num_freq = freq.shape[0]
 
    greenR = np.zeros((2, num_kpoints, num_freq), dtype=np.complex128)
    greenR_corr = np.zeros((2, num_kpoints, num_freq), dtype=np.complex128)
    
    for num in range(num_freq):
        for e in range(num_kpoints):
            for z in range(2):
                #G = 1/(E - H)
                greenR[z, e, num] += 1/(freq[num] - Ham_K[e, z] + 1j*smearing)
                greenR_corr[z, e, num] += 1/((freq[num] - Ham_K[e, z]) - selfen[z, e, num])
        
    greenR *= -(1/np.pi) 
    greenR_corr *= -(1/np.pi)
    
    return greenR, greenR_corr

        
    
    
def calc_electron(t, delta, e_fermi, cell_vec, k_vec): 
    
    num_kpoints = k_vec.shape[0]
    el_k = np.zeros((num_kpoints, 2), dtype=np.complex128)
    Ham_r = np.zeros((3, 3, 2))
    
    # nn hopping t 
    Ham_r[0, 1, 0] = t
    Ham_r[1, 0, 0] = t
    Ham_r[1, 2, 0] = t
    Ham_r[2, 1, 0] = t
    
    Ham_r[0, 1, 1] = t
    Ham_r[1, 0, 1] = t
    Ham_r[1, 2, 1] = t
    Ham_r[2, 1, 1] = t
    
    Ham_r[1, 1, 0] = -0.5 * delta - e_fermi # spin_up
    Ham_r[1, 1, 1] =  0.5 * delta - e_fermi # spin_dn
    
    #vectorized version of Fourier transform
    for i in range(-1, 2):
        for j in range(-1, 2):
            r = i * cell_vec[0] + j * cell_vec[1]
            k_dot_r = (k_vec @ r).reshape(num_kpoints, 1)
            phase = np.exp(-1j * k_dot_r)
            el_k += phase * Ham_r[i + 1, j + 1, :]
            
    return el_k



#Model of phonons and elph coupling constant
def calc_phonon(model, d_const, v, hw0, k_vec, electron_k, verbose):
    num_kpoints = k_vec.shape[0]
    
    hbar = 6.582e-16 # (eV * s)
    m = 10 * 1.036e-28 # (amu -> eV * s^2/A^2) 
    a = 5 # unit cell parameter A 
    kB = 8.61733e-5 # eV/K
    
    if(model == 'D'):
        v = v * 1e10 #(m/s - > A/s) 
        if(verbose):
            Tc = np.pi * hbar * v / (kB * a) 
            print("Debye model of phonons is used")
            print("Speed of sound (m/s):", v / 1e10)
            print("Effective frequency (eV):", np.pi * hbar * v / a)
            print("Deformation value (eV):", d_const)
            print("Debye temperature (K):",Tc)
                       
        lambda_elph = calc_lambda(electron_k)
        lambda_elph *= d_const**2 / (2 * m * v**2) 
        ph_q = np.zeros((num_kpoints, 2))
        g = np.zeros(num_kpoints)
        for q in range(num_kpoints):
            q_abs = np.linalg.norm(k_vec[q])  # 1/A
            ph_q[q, 0] = hbar * v * q_abs # (eV * s) * (A/s) * (1/A) -> eV  
            ph_q[q, 1] = hbar * v * q_abs 
            
            g[q] = np.sqrt( hbar**2 /(m * ph_q[q, 0])) * q_abs * d_const  # eV 
                 
                   
    elif(model == 'E'):
        q0 = np.pi / a  # 1/A
        if(verbose):
            Tc = hw0 / kB
            print("Einstein model of phonons is used")
            print("Phonon frequency (eV):", hw0)
            print("Deformation value (eV):", d_const)
            print("Einstein temperature (K):",Tc)
    
        lambda_elph = calc_lambda(electron_k)
        lambda_elph *= (d_const * hbar * q0)**2 / (2 * m * hw0**2)  
        ph_q = hw0 * np.ones((num_kpoints, 2))
        g = np.zeros(num_kpoints)
        for q in range(num_kpoints):
            g[q] = np.sqrt( hbar**2 /(m * hw0)) * q0 * d_const # eV
          
    else:
        print("Set the correct model!!!")
        return None
            
    return ph_q, g, lambda_elph



#calculate el-ph coupling constant
@jit(nopython=True, parallel=True)
def calc_lambda(el_k):
    num_kpoints = el_k.shape[0]
    weight = 1 / num_kpoints
    el_k = el_k.reshape(kmesh[0], kmesh[1], 2).real
    
    def dirac_delta(kT,E):
        if(np.abs(E/kT) < 20):
            delta = (np.exp(E/kT)/kT)/(1 + np.exp(E/kT))**2
        else:
            delta = 0
        return delta
    
    lambda_elph = 0.0
    DOS_F = 0.0
    for k1 in prange(kmesh[0]):
        for k2 in prange(kmesh[1]):
            DOS_F -= dirac_delta(el_k[k1, k2, 0], 0.01)
            for q1 in range(kmesh[0]):
                for q2 in range(kmesh[1]):
                    kq1 = (k1 + q1) % kmesh[0]
                    kq2 = (k2 + q2) % kmesh[1]
                    lambda_elph += dirac_delta(el_k[k1, k2, 0], 0.01) * dirac_delta(el_k[kq1, kq2, 0], 0.01)

    lambda_elph *= weight / DOS_F
    return lambda_elph 
    


@jit(nopython=True, parallel=True)
def elph_selfen(g, kT, kmesh, freq, el_k, ph_q):
    
    def bose(kT, E):
        return 1.0 / (np.exp(E / kT) - 1)

    def fermi(kT,E):
        return 1 / (np.exp(E / kT) + 1)

    num_freq = freq.shape[0]
    num_kpoints = kmesh[0] * kmesh[1]

    selfen = np.zeros((2, num_kpoints, num_freq), dtype=np.complex128)
    weight = 1 / num_kpoints
    
    delta = 0.005 # 50K

    el_k = el_k.reshape(kmesh[0], kmesh[1], 2)
    ph_q = ph_q.reshape(kmesh[0], kmesh[1], 2)
    g = g.reshape(kmesh[0], kmesh[1])

    for k1 in prange(kmesh[0]):
        for k2 in prange(kmesh[1]):
            k = k2 + (k1 * kmesh[1])
            for q1 in range(kmesh[0]):
                for q2 in range(kmesh[1]):
                    kq1 = (k1 + q1) % kmesh[0]
                    kq2 = (k2 + q2) % kmesh[1]

                    for z in range(2):
                        bose_term = bose(kT, ph_q[q1, q2, z])
                        fermi_term = fermi(kT, el_k[kq1, kq2, z])
                        el_energy = el_k[kq1, kq2, z]
                        ph_energy = ph_q[q1, q2, z]
                        g2_const = g[q1, q2]**2 
                    
                        for w in range(num_freq):
                            selfen[z, k, w] += g2_const * ((bose_term + fermi_term) / (freq[w] - el_energy + ph_energy + 1j*delta) + \
                            (bose_term + 1 - fermi_term) / (freq[w] - el_energy - ph_energy + 1j*delta)) 

    selfen *= weight
    
    if(np.any(np.isnan(selfen))):
        print("There is a divergence!!!")
    
    return selfen  


@jit(nopython=True)    
def plot_dos(freq, energy_k, kT):
    
    def dirac_delta(kT,E_tot):
        E = E_tot.real
        if(np.abs(E/kT) < 20):
            delta = (np.exp(E/kT)/kT)/(1 + np.exp(E/kT))**2
        else:
            delta = 0
        return delta
    
    num_kpoints = energy_k.shape[0]
    num_enpoints = freq.shape[0]
    dos_F = np.zeros((1000, 2))
    
    for spin in range(2):
        for en in range(1000):
            for k in range(num_kpoints):        
                dos_F[en, spin] += dirac_delta(kT, freq[en] - energy_k[k, spin])
                
    return dos_F/num_kpoints

#############################################################################

t = -0.1 # hopping (in eV)
delta = 0.4 # spin splitting (in eV)
e_fermi = 0 # Fermi energy
smearing = 1e-3 # smearing for integration

ncol = 500
nrow = 10
num_freq = ncol + 2 * nrow 

kmesh = np.array([60, 60, 1])
num_kpoints = np.prod(kmesh)
print('kmesh: ', kmesh)

cell_vec = np.array([[5.0,  0.0, 0.0],
                     [0.0,  5.0, 0.0],
                     [0.0,  0.0, 20.0]])



k_vec = kmesh_preparation(kmesh, cell_vec)
el_k = calc_electron(t, delta, e_fermi, cell_vec, k_vec)

e_low = np.min(el_k.real) - 0.2*delta # min energy for integration

#plot 2d map
freq = np.linspace(e_low, e_fermi, ncol)
index_temp = np.array([1, 0])
E, dE = energy_contour_preparation(ncol, nrow, e_fermi, e_low, smearing)

v = np.linspace(1000, 3000, 50) # sound velocity (in m/c)
T = np.linspace(10, 300, 50)
data = np.zeros((50,50), dtype=np.float64)

for i in range(50):
    for j in range(50):
        d  = 0.05 * v[i] /410 
        ph_q, g_q, lambda_elph = calc_phonon('D', d, v[i], 0.01, k_vec, el_k, False)
        selfen = elph_selfen(g_q, T[j]*0.00008617, kmesh, freq, el_k, ph_q)
        selfen = np.pad(selfen, [(0, 0), (0, 0),(nrow, nrow)])
        J = calc_exchange(index_temp, num_freq, cell_vec, k_vec, E, dE, el_k, selfen)
        data[i, j] = J
        print(v[i], j, J, lambda_elph)
    


results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

xmesh, ymesh = np.meshgrid(T, v)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.pcolor(xmesh, ymesh, data, shading='nearest', cmap='Blues')
ax.set_xlabel('T(K)')
ax.set_ylabel('v(m/c)')
fig.savefig(os.path.join(results_dir, '1.png'), dpi=400, facecolor='w', transparent=False, bbox_inches='tight')


with open(os.path.join(results_dir, 'J_plot.dat'), "w") as fp:
    for i in range(50):
        print('{0.real:.4f}'.format(T[i]),'  '.join('{0.real:.4f}'.format(item) for item in data[:,i]), file=fp)
