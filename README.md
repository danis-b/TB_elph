**“Phonon-induced renormalization of exchange interactions in metallic two-dimensional magnets”**  
by Danis Badrtdinov, Alexander N. Rudenko and Mikhail I. Katsnelson  

[![arXiv](https://img.shields.io/badge/arXiv-2406.05229-b31b1b)](https://arxiv.org/abs/2406.05229)
[![DOI](https://img.shields.io/badge/DOI-10.1103/PhysRevB.110.L060409-blue)](https://doi.org/10.1103/PhysRevB.110.L060409)



This script calculates electron-phonon coupling contribution to exchange interaction $J_{ij}$ for a square lattice at half-filling interacting with acoustic phonons in the equilibrium:
```math
    H = t\sum_{\langle ij \rangle \sigma} c^\dagger_{i \sigma} c_{j \sigma} + \frac{\Delta}{2} \sum_{i} ( n^\uparrow_i - n^\downarrow_i) + 
   + \sum_{\bf q} \omega_{\bf q} b^{\dagger}_{\bf q} b_{\bf q} + \sum_{{\bf q},<ij>\sigma} g_{\bf q} (b_{\bf q}^{\dagger} + b_{-{\bf q}})c_{i\sigma}^{\dagger} c_{j\sigma},
```
where $c_{i \sigma}^\dagger (c_{i\sigma})$ and $b_{\bf q}^\dagger (b_{\bf q})$ are the creation (annihilation) operator of electrons at site $i$ and spin $\sigma$, and phonons with wave vector ${\bf q}$ and frequency $\omega_{\bf q}$, $n_i^\sigma = c_{i \sigma}^\dagger c_{i\sigma}$, $t$ is the nearest-neighbor hopping, and $\langle ij \rangle$ labels the nearest-neighbour pairs. $\Delta$ is the on-site interaction giving rise to the exchange splitting of electronic bands, and $g_{\bf q}$ is the electron-phonon coupling, which can be rewritten via dimensionless electron-phonon coupling constant $\lambda$.  Exchange coupling is calculated using Green's function technique (see [1](https://www.sciencedirect.com/science/article/abs/pii/0304885387907219), [2](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.71.184434)):
```math
J_{ij}(T) = -\frac{1}{2 \pi S^2} \int \limits_{-\infty}^{E_F} d \omega  \,{\rm Im} \left[  \widetilde{\Delta}_i (\omega, T) \widetilde{G}_{ij}^{\downarrow} (\omega, T) \widetilde{\Delta}_j (\omega, T) \widetilde{G}_{ji} ^{\uparrow} (\omega, T) \right],
```
Here $\widetilde{\Delta} (\omega, T)$ and $\widetilde{G}(\omega, T)$  stands for Fourier transformed component of correlated intra-orbital spin-splitting energy and Green's function due to electron-phonon coupling:

```math
\widetilde{\Delta}_i (\omega, \mathbf{k}, T) = H_{i}^{\uparrow}(\mathbf{k}) - H_{i}^{\downarrow}(\mathbf{k}) + \Sigma^\uparrow(\omega, \mathbf{k}, T)  - \Sigma_{\textrm{elph}}^\downarrow (\omega, \mathbf{k}, T),  
```

```math
\widetilde{G}_{ij}^{-1} (\omega,  \mathbf{k}, T) = G_{ij}^{-1} (\omega,  \mathbf{k}) - \Sigma_{\textrm{elph}}(\omega, \mathbf{k}, T).
```

The electron self-energy of electron-phonon interaction in Migdal approximation has the form
``` math
\Sigma^\sigma_{\textrm{elph}}(\omega, \mathbf{k}, T)  =  \sum_{\mathbf{q}} g^2_{\bf q} \left[ \frac{b_{\mathbf{q}} + f^\sigma_{\mathbf{k + q}}}{\omega - \varepsilon^\sigma_{\mathbf{k + q}} + \hbar  \omega_{\mathbf{q}} + i\eta}  + \frac{b_{\mathbf{q}} +1 - f^\sigma_{\mathbf{k + q}}}{\omega - \varepsilon^\sigma_{\mathbf{k + q}} - \hbar  \omega_{\mathbf{q}} + i\eta} \right].
```

At long wavelengths and high temperature limit it can be recast in a simple form as:
``` math
\Sigma^\sigma_{\textrm{elph}}(\omega, \mathbf{k}, T)  = 2 \lambda \frac{T}{N^\sigma_F}    \sum_{\mathbf{q}} (\omega - \varepsilon^\sigma_{\mathbf{k + q}} + i\eta)^{-1}.
```

The presence of electron-phonon coupling with equilibrium phonon distribution leads to a notable suppression of exchange interactions with temperature:


![alt text](https://github.com/danis-b/TB_elph/blob/main/example/square_lattice.jpg)
