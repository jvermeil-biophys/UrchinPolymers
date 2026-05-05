# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 11:58:43 2026

@author: Joseph
"""


#Required packages ----------------


import math
import numpy as np
import pandas as pd

# %%

def truncArray(A, N):
    f = 10**N
    out = np.trunc(A * f) / f
    if N == 0:
        out = out.astype(int)
    return(out)

def getGlycerolViscosity(ratio, T):
    T = T 			       	#temperature (degrees Celcius)
    waterVol = 1 - ratio	#volume of water required (ml)
    glycerolVol = ratio		#volume of Glycerol used (ml)


    #Densities ----------------

    glycerolDen = (1273.3-0.6121*T)/1000 			#Density of Glycerol (g/cm3)
    waterDen = (1-math.pow(((abs(T-4))/622),1.7)) 	#Density of water (g/cm3)


    #Fraction cacluator ----------------

    glycerolMass=glycerolDen*glycerolVol
    waterMass=waterDen*waterVol
    totalMass=glycerolMass+waterMass
    mass_fraction=glycerolMass/totalMass
    vol_fraction= glycerolVol/(glycerolVol+waterVol)
     
    # print ("Mass fraction of mixture =", round(mass_fraction,5))
    # print ("Volume fraction of mixture =", round(vol_fraction,5))


    #Density calculator ----------------

    ##Andreas Volk polynomial method
    contraction_av = 1-math.pow(3.520E-8*((mass_fraction*100)),3)+math.pow(1.027E-6*((mass_fraction*100)),2)+2.5E-4*(mass_fraction*100)-1.691E-4
    contraction = 1+contraction_av/100

    ## Distorted sine approximation method
    #contraction_pc = 1.1*math.pow(math.sin(numpy.radians(math.pow(mass_fraction,1.3)*180)),0.85)
    #contraction = 1 + contraction_pc/100

    density_mix=(glycerolDen*vol_fraction+waterDen*(1-vol_fraction))*contraction

    # print ("Density of mixture =",round(density_mix,5),"g/cm3")


    #Viscosity calcualtor ----------------

    glycerolVisc=0.001*12100*np.exp((-1233+T)*T/(9900+70*T))
    waterVisc=0.001*1.790*np.exp((-1230-T)*T/(36100+360*T))

    a=0.705-0.0017*T
    b=(4.9+0.036*T)*np.power(a,2.5)
    alpha=1-mass_fraction+(a*b*mass_fraction*(1-mass_fraction))/(a*mass_fraction+b*(1-mass_fraction))
    A=np.log(waterVisc/glycerolVisc)

    viscosity_mix=glycerolVisc*np.exp(A*alpha)

    # print ("Viscosity of mixture =",round(viscosity_mix,5), "Ns/m2")
    
    return(viscosity_mix)

# %% Make a table !

RR = np.arange(0, 1.05, 0.05)
TT = np.arange(15, 25.5, 0.5)
nR = len(RR)
nT = len(TT)
M = np.zeros((nR, nT))
DictVisco = {}
for j in range(nT):
    DictVisco[TT[j]] = []
    for i in range(nR):
        V = getGlycerolViscosity(RR[i], TT[j]) * 1000
        V = truncArray(V, 2)
        DictVisco[TT[j]].append(V)

df_Visco = pd.DataFrame(DictVisco)
df_Visco.index = truncArray(RR*100, 0)
df_Visco2 = df_Visco.transpose()
