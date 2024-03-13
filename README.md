This script calculates electron-phonon coupling contribution to exchange interaction $J_{ij}$ for one band model given by nn hopping parameter $t$ and on-site splitting $\Delta$. Exchange coupling is calculated using Green's function technique (see [1](https://www.sciencedirect.com/science/article/abs/pii/0304885387907219), [2](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.71.184434)):
```math
J_{ij}(T) = -\frac{1}{2 \pi S^2} \int \limits_{-\infty}^{E_F} d \omega  \,{\rm Im} \left[  \widetilde{\Delta}_i (\omega, T) \widetilde{G}_{ij}^{\downarrow} (\omega, T) \widetilde{\Delta}_j (\omega, T) \widetilde{G}_{ji} ^{\uparrow} (\omega, T) \right],
```
Here $\widetilde{\Delta} (\omega, T)$ and $\widetilde{G}(\omega, T)$  stands for Fourier transformed component of correlated intra-orbital spin-splitting energy and Green's function due to electron-phonon coupling:

```math
\widetilde{\Delta}_i (\omega, \mathbf{k}, T) = H_{i}^{\uparrow}(\mathbf{k}) - H_{i}^{\downarrow}(\mathbf{k}) + \Sigma^\uparrow(\omega, \mathbf{k}, T)  - \Sigma^\downarrow (\omega, \mathbf{k}, T),  \\

\widetilde{G}_{ij}^{-1} (\omega,  \mathbf{k}, T) = G_{ij}^{-1} (\omega,  \mathbf{k}) - \Sigma(\omega, \mathbf{k}, T).
```

![alt text](https://github.com/danis-b/TB_elph/blob/main/example/Square_lattice.png)
