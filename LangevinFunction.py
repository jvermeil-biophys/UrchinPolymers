# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 14:41:38 2025

@author: Utilisateur
"""

# %% Imports & Constants

import numpy as np
import matplotlib.pyplot as plt

from mpmath import coth

mu0 = 4*np.pi*1e-7 # Permeabilite magnetique du vide [µ0] - H/m (Henry/metre)
kB = 1.380649e-23 # Constante de Boltzman - J/K (Joule/Kelvin)
T = 20 + 273.15 # Temperature - K (Kelvin)


# %% Functions

def Langevin(x):
    # if x < 0.1:
    return(np.tanh(x/3))
    # else:
    #     return((1/np.tanh(x)) - (1/x))
    
def Langevin_A(X):
    mask = (np.abs(X) < 0.1)
    res = np.zeros_like(X)
    res[~mask]= np.tanh(X[~mask]/3)
    res[mask] = (1/np.tanh(X[mask])) - (1/X[mask])
    return(res)
        

def Magnetization(mu, n, B):
    # try:
    A = mu*n
    X = mu*B / (kB*T)
    return(A * Langevin(X))
    # except:
    #     A = mu*n
    #     X = mu*B / (kB*T)
    #     return(A * Langevin(X))

def ChampMag(m, r):
    return((mu0/(4*np.pi))*(m/(r**3)))

def GradMag(m, r):
    return(-(3*mu0/(4*np.pi))*(m/r**4))


def ForceMag(m_magnet, V_b, MagFun_b, r):
    B = ChampMag(m_magnet, r)
    m_bead = V_b * MagFun_b(B)
    F = - ((3*mu0)/(4*np.pi*r**4)) * m_magnet * m_bead
    return(F)
    


# %% Proprietes de l'aimant

# Disons que le champ magnetique dans l'axe de l'aimant est de 5 mT à 200 µm

B1 = 10*1e-3 # T
R1 = 200*1e-6 # m
m1 = 4*np.pi*B1*(R1**3)/mu0 # A.m² (Ampere.metre^2)


# %% Propriétés des billes

mu_b = 8e-4 # Permeabilite magnetique du materiau des billes [µ0.µr] - H/m (Henry/metre)
n_b = 0.2 # Densite en elements magnetique - % (ratio)
R_b = 5e-6 # Rayon des billes - m (metre)
V_b = (4/3)*np.pi*R_b**3

Btest = 10e-3
A = mu_b*n_b
X = mu_b*Btest / (kB*T)
Mtest = A * np.tanh(X)

MagFun_b = lambda B : Magnetization(mu_b, n_b, B)

BB = np.arange(1, 100, 1)*1e-12 # Magnetic Field from -0.1 to +0.1 T
MM = MagFun_b(BB)
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax = ax
ax.plot(BB*1e3, np.tanh(BB/3), 'g')
ax.grid()
ax.set_xlabel('Mag Field (mT)', color = 'b')
ax.set_ylabel('Bead Magnetization (A/m)', color = 'g')

plt.tight_layout()
plt.show()



# %% Plots

RR = np.arange(100, 500, 1)*1e-6 # From Ri to Rf distance (metre)

B1 = ChampMag(m1, RR) # T or N/(A.m)
GB1 = GradMag(m1, RR) # T/m or N/(A.m²)
M1 = Magnetization(mu_b, n_b, B1) # A/m -> magnetic moment is A.m² (product by volume)
F1 = ForceMag(m1, V_b, MagFun_b, RR) # N

# fig, ax = plt.subplots(1, 1)
# ax.plot(RR*1e6, B1*1000)
# ax.grid()
# ax.set_xlim([0, ax.get_xlim()[1]])

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
ax = axes[0,0]
ax.plot(RR*1e6, B1*1000, 'b')
axbis = ax.twinx()
axbis.plot(RR*1e6, np.abs(GB1/1000), 'r')
ax.grid()
ax.set_xlabel('Distance from magnet (µm)')
ax.set_ylabel('Mag Field (mT)', color = 'b')
axbis.set_ylabel('Mag Field Grad (mT/µm)', color = 'r')

ax = axes[0,1]
ax.plot(RR*1e6, B1*1000, 'b')
axbis = ax.twinx()
axbis.plot(RR*1e6, M1, 'g')
ax.grid()
ax.set_xlabel('Distance from magnet (µm)')
ax.set_ylabel('Mag Field (mT)', color = 'b')
axbis.set_ylabel('Bead Magnetization (A/m)', color = 'g')

ax = axes[1,0]
ax.plot(RR*1e6, B1*1000, 'b')
axbis = ax.twinx()
axbis.plot(RR*1e6, np.abs(F1)*1e12, 'purple')
ax.grid()
ax.set_xlabel('Distance from magnet (µm)')
ax.set_ylabel('Mag Field (mT)', color = 'b')
axbis.set_ylabel('Attractive Force (pN)', color = 'purple')

plt.tight_layout()
plt.show()


# %%