# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 14:41:38 2025

@author: Utilisateur
"""

# %% Imports & Constants

import numpy as np
import matplotlib.pyplot as plt

# from mpmath import coth

mu0 = 4*np.pi*1e-7 # Permeabilite magnetique du vide [µ0] - H/m (Henry/metre)
kB = 1.380649e-23 # Constante de Boltzman - J/K (Joule/Kelvin)
T = 20 + 273.15 # Temperature - K (Kelvin)

# kB* = 4.0473725435e-21

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
    return(1 * Langevin(X))
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

# MyOne Beads
MS_part = 336e3 # A/m
R_part = 7.6/2 * 1e-9 # m
V_part = (4/3)*np.pi*R_part**3 # m3

n_b = 0.255 # Densite M/M en elements magnetique - ratio
phi_b = 0.119 # Densite V/V en elements magnetique - ratio
R_b = 1.05 * 1e-6 / 2 # m
V_b = (4/3)*np.pi*R_b**3 # m3
rho_b = 1.7 # g/mL or g/cm3 or kg/L
mass_b = rho_b * V_b * 1000 # kg

N = phi_b*V_b/V_part

mu_b = V_b*phi_b*MS_part
M0_b = V_b*phi_b*MS_part/mass_b
M0_b_paper = 23.5 # Am2/kg

Chi_b = N*(V_part*MS_part)**2/(3*kB*T*mass_b)
Chi_b_paper = 81e-5 # m3/kg


mu_b = 2e-19 # Permeabilite magnetique du materiau des billes [µ0.µr] - H/m (Henry/metre)
n_b = 0.2 # Densite en elements magnetique - % (ratio)
R_b = 5e-6 # Rayon des billes - m (metre)
V_b = (4/3)*np.pi*R_b**3
Chi = mu_b/(kB*T)

Btest = 10e-3
A = mu_b*n_b
X = mu_b*Btest / (kB*T)
Mtest = A * np.tanh(X)

MagFun_b = lambda B : Magnetization(mu_b, n_b, B)

BB = np.arange(-100, 100, 1)*1e-3 # Magnetic Field from -0.1 to +0.1 T
MM = MagFun_b(BB)
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax = ax
ax.plot(BB*1000, MM, 'g')
ax.plot(BB*1000, BB*Chi/3, 'r--')
ax.grid()
ax.set_xlabel('Mag Field (mT)', color = 'b')
ax.set_ylabel('Bead Magnetization (A/m)', color = 'g')

plt.tight_layout()
plt.show()

# %% Tests with Langevin Function

def L1_A(a1, a2, X):
    mask = (np.abs(X) < 0.01)
    res = np.zeros_like(X)
    res[mask]= a1*np.tanh(a2*X[mask]/3)
    res[~mask] = a1*((1/np.tanh(a2*X[~mask])) - (1/(a2*X[~mask])))
    return(res)


XX = np.linspace(-5, 5, 1000)
YYref = L1_A(1, 1, XX)

fig, ax = plt.subplots(1, 1)

a1 = 1
for a2 in [0.8, 0.9, 1, 1.1, 1.2]:
    ax.plot(XX, L1_A(a1, a2, XX)/YYref)
    
ax.grid()
fig.tight_layout()
plt.show()


# fig, ax = plt.subplots(1, 1)

# a = 1
# for b in [0.25, 0.5, 1, 2, 4]:
#     ax.plot(XX, L1_A(a, b, XX)/YYref)

# ax.grid()

# fig.tight_layout()
# plt.show()

# %% Computation with the paper values

ms_p = 336e3
D_p = 7.0e-9
R_p = D_p/2
V_p = (4/3)*np.pi*R_p**3

rho_b = 1.7
phi_b = 0.119

M0 = phi_b * ms_p / (rho_b*1000)

# Chi = phi_b*ms_p * (V_p*ms_p) * (1/(3*kB*T)) * (1/rho_b*1000)
Chi = M0 * (V_p*ms_p) * (1/(3*kB*T)) * (mu0/2)

A1 = 10.8
A2 = 3*54e-5/(A1*mu0)
B1 = 19.6
B2 = 3*102e-5/(B1*mu0)
C1 = 23.5
C2 = 3*81e-5/(C1*mu0)

def L1_A(a1, a2, X):
    mask = (np.abs(X) < 0.001)
    res = np.zeros_like(X)
    res[mask]= a1*np.tanh(a2*X[mask]/3)
    res[~mask] = a1*((1/np.tanh(a2*X[~mask])) - (1/(a2*X[~mask])))
    return(res)

XX = np.linspace(-1, 1, 1000)
# XX = np.linspace(-0.015, 0.015, 1000)

fig, ax = plt.subplots(1, 1)
ax.plot(XX, L1_A(A1, A2, XX))
    
ax.grid()
fig.tight_layout()
plt.show()


# %% Estimate B

# Viscosity of glycerol 80% v/v glycerol/water at 21°C [Pa.s]
viscosity_glycerol = 0.0857  
# Magnet function distance (µm) to velocity (µm/s) [expected velocity in glycerol]
mag_d2v = lambda x: 80.23*np.exp(-x/47.49) + 1.03*np.exp(-x/22740.0)

DragC = 6*np.pi*viscosity_glycerol*0.5e-6
Vb = (4/3)*np.pi*(0.5e-6)**3

C1b = 23.5*1700
C2b = 3*81e-5/(C1*mu0)

XX = np.linspace(61, 800, 10000)*1e-6
VV = mag_d2v(XX*1e6)
VV2 = mag_d2v(XX*1e6 - 60)
FF = DragC * (VV2*1e-6)

m_mag = 1.8e-7
BB = ChampMag(m_mag, XX)
GBGB = GradMag(m_mag, XX)
MM = L1_A(C1b, C2b, BB)
mm = MM*Vb
FFmag = -mm*GBGB

def Fmag(m_mag, XX): 
    Vb = (4/3)*np.pi*(0.5e-6)**3
    C1b = 23.5
    C2b = 3*81e-5/(C1*mu0)
    
    BB = ChampMag(m_mag, XX)
    GBGB = GradMag(m_mag, XX)
    MM = L1_A(C1b, C2b, BB)
    mm = MM*Vb/1700
    FFmag = -mm*GBGB
    




fig, axes = plt.subplots(2, 2, figsize=(16,12))
ax = axes[0, 0]
ax.plot(XX*1e6, VV)
ax.plot(XX*1e6, VV2)
ax.grid()

ax = axes[0, 1]
ax.plot(XX*1e6, FF*1e12, 'darkred')
ax.grid()

ax = axes[1, 0]
ax.plot(XX*1e6, BB*1000, 'purple')
ax.grid()
axbis = ax.twinx()
axbis.plot(XX*1e6, GBGB*1000, 'r')

ax = axes[1, 1]
ax.plot(XX*1e6, FFmag*1e12, 'darkred')
ax.grid()
axbis = ax.twinx()
axbis.plot(XX*1e6, MM, 'g')

fig.tight_layout()
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