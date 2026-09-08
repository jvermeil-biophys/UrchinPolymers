# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 09:54:27 2026

@author: Utilisateur
"""

# %% 1. Imports

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from Libs.GlycerolCalc import getGlycerolViscosity

import Libs.PlotMaker as pm
import Libs.UtilityFunctions as ufun

pm.setGraphicOptions(mode = 'screen', palette = 'Set2', colorList = pm.cL_Set21)

# %% 2. Utility scripts

# %%% a) Split single big file

# mainDir = 'C:/Users/Joseph/Desktop/RheoMacro/2025-10-08+09+10_Rheology/'
# fileName = '2025-10-08+09+10_AllMeasures.csv'
mainDir = 'C:/Users/Joseph/Desktop/RheoMacro/25-10-30_Rheology/'
fileName = '2025-10-30_AllMeasures.csv'
filePath = os.path.join(mainDir, fileName)

Names = []
i_init = []
i_fin  = []

with open(filePath, mode='r', encoding='utf_16_le') as f:
    Lines = f.readlines()
    for i, L in enumerate(Lines):
        if L.startswith('Test:'):
            W = L[:-1].split('\t')
            Names.append(W[1])
            if len(i_init) > 0:
                i_fin.append(i)
            i_init.append(i)
    i_fin.append(len(Lines))
    
    for k in range(len(Names)):
        newFileName = Names[k] + '.csv'
        i_i, i_f = i_init[k], i_fin[k]
        fileLines = Lines[i_i:i_f]
        newFilePath = os.path.join(mainDir, newFileName)
        with open(newFilePath, mode = 'x', encoding='utf_8') as nf:
            for fL in fileLines:
                nf.write(fL)
                
# %%% b) Load files and make a synthetic table

# %%%% i. UV version

mainDir = 'C:/Users/Joseph/Desktop/RheoMacro'

listFiles = []
listPaths = []
for d in os.listdir(mainDir):
    dP = os.path.join(mainDir, d)
    if os.path.isdir(dP) and '25-10-30' in dP:
        lF = [f for f in os.listdir(dP) if f.endswith('.csv')]
        lP = [os.path.join(dP, f) for f in lF]
        listFiles += lF
        listPaths += lP

dictResults = {'date':[],
               'solvent':[],
               'polymer':[],
               'PI':[],
               'UV':[],
               'viscosity':[],
               'temperature':[],
               'fileName':[],}

for f, fp in zip(listFiles, listPaths):
    if f.endswith('.csv') and not f.startswith('Results'):
        blocks = f[:-4].split('_')
        dictResults['date'].append(blocks[0])
        dictResults['solvent'].append(blocks[1])
        dictResults['polymer'].append(blocks[2])
        dictResults['PI'].append(blocks[3])
        dictResults['UV'].append(blocks[4])
        dictResults['fileName'].append(f)
        
        path = fp
        df = pd.read_csv(path, header = 3, sep='\t', #skiprows=2,
                         on_bad_lines='skip', encoding='utf_8') # 'utf_16_le'
        df = df.drop(df.columns[:2], axis = 1).drop(df.index[:2], axis = 0).reset_index(drop=True)
        viscosity = np.median(df['Viscosity'].astype(float).values)
        try:
            temperature = np.median(df['Temperature'].astype(float).values)
        except:
            temperature = np.nan
        
        dictResults['viscosity'].append(viscosity)
        dictResults['temperature'].append(temperature)
        
df_summary = pd.DataFrame(dictResults)
df_summary.to_csv(os.path.join(mainDir, 'ResultsMacroRheo_Round04.csv'), index=False)


# %%%% ii. Temperature version

# mainDir = 'C:/Users/Joseph/Desktop/RheoMacro/26-09-07_TestGlycerol_forCalibs/S-01-Ref'
# mainDir = 'C:/Users/Utilisateur/Desktop/RheoMacro/26-09-07_TestGlycerol_forCalibs/LastReps'
mainDir = 'C:/Users/Utilisateur/Desktop/RheoMacro/26-09-07_TestGlycerol_forCalibs'


listFiles = [f for f in os.listdir(mainDir) if \
      (f.startswith('2026-09-07') and f.endswith('.csv') and (not 'all' in f))]
listPaths = [os.path.join(mainDir, f) for f in listFiles]

df_temp = pd.read_csv(os.path.join(mainDir, 'TableTemperatures.csv'))

dictResults = {'date':[],
               'liquid':[],
               'sample':[],
               'meas no':[],
               'rep':[],
               'viscosity':[],
               'T_thc':[],
               'T_pt100':[],
               'fileName':[],}


for f, fp in zip(listFiles, listPaths):
    if f.endswith('.csv') and not f.startswith('Results'):
        blocks = f[:-4].split('_')
        dictResults['date'].append(blocks[0])
        dictResults['liquid'].append(blocks[2])
        dictResults['sample'].append(blocks[4])
        dictResults['meas no'].append(blocks[5])
        dictResults['rep'].append(blocks[6])
        dictResults['fileName'].append(f)
        
        T_thc = df_temp[df_temp['fileName'] == f]['tempThermocouple'].values[0]
        T_pt100 = df_temp[df_temp['fileName'] == f]['tempPt100'].values[0]
        dictResults['T_thc'].append(T_thc)
        dictResults['T_pt100'].append(T_pt100)
        
        path = fp
        df = pd.read_csv(path, header = 3, sep='\t', #skiprows=2,
                         on_bad_lines='skip', encoding='utf_8') # 'utf_16_le'
        df = df.drop(df.columns[:2], axis = 1).drop(df.index[:2], axis = 0).reset_index(drop=True)
        viscosity = np.median(df['Viscosity'].astype(float).values)
        dictResults['viscosity'].append(viscosity)
        
        
df_summary = pd.DataFrame(dictResults)
df_summary.to_csv(os.path.join(mainDir, 'ResultsMacroRheo_All.csv'), index=False)


# %% 2. Plot Rheometer

# %%% 26-09-07 - Figure out Glycerol concentration

# %%%% 1. With only the reference

srcDir = 'C://Users//Utilisateur//Desktop//RheoMacro//26-09-07_TestGlycerol_forCalibs//S-01-Ref'
fileName = 'ResultsMacroRheo_S-01-Ref.csv'
filePath = os.path.join(srcDir, fileName)

df = pd.read_csv(filePath)

#### Set mix=80% and compare expected temperatures

ratio = 0.8
TT = np.linspace(22, 26, 4000)
best_Ts = []

for visco in df['viscosity']:
    A = getGlycerolViscosity(ratio, TT)
    B = np.abs(A - visco)
    i = np.argmin(B)
    best_t = float(TT[i])
    best_Ts.append(best_t)
    

best_Ts = np.array(best_Ts)

fig, ax = plt.subplots(1, 1)
xx = np.arange(1, 1+len(best_Ts))
ax.plot(xx, df.T_thc, label='T thc')
ax.plot(xx, df.T_pt100, label='T pt100')
ax.plot(xx, best_Ts, label='T best')

ax.set_xlabel('Meas no.')
ax.set_ylabel('Temp °C')
ax.legend()
ax.grid()
plt.show()


# Conclusion: it seems T thc is the best temperature reference !

# %%%% 2. With the latest reps

srcDir = 'C://Users//Utilisateur//Desktop//RheoMacro//26-09-07_TestGlycerol_forCalibs//LastReps'
fileName = 'ResultsMacroRheo_LastReps.csv'
filePath = os.path.join(srcDir, fileName)

df = pd.read_csv(filePath)

#### Set mix=80% and compare expected temperatures

RR = np.linspace(0.75, 0.85, 2000)
TT = df['T_thc'].values
VV = df['viscosity'].values

best_Rs = []

for v, t in zip(VV, TT):
    A = getGlycerolViscosity(RR, t)
    B = np.abs(A - v)
    i = np.argmin(B)
    best_r = float(RR[i])
    best_Rs.append(best_r)
    

best_Rs = np.array(best_Rs)
df['Ratio_with_T_thc'] = best_Rs
df['rep_int'] = df['rep'].apply(lambda x: int(x[-1]))

fig, ax = plt.subplots(1, 1)

sns.scatterplot(data=df, ax=ax, x='rep_int', y='Ratio_with_T_thc',
                hue='sample', style='meas no')

ax.set_xlabel('Replicates')
ax.set_ylabel('Ratio (%)')
ax.legend()
ax.grid()
plt.show()



# %%%% 3. With all reps

srcDir = 'C://Users//Utilisateur//Desktop//RheoMacro//26-09-07_TestGlycerol_forCalibs//'
fileName = 'ResultsMacroRheo_All_DataSelection.csv'
filePath = os.path.join(srcDir, fileName)

df = pd.read_csv(filePath)

#### Set mix=80% and compare expected temperatures

RR = np.linspace(0.75, 0.85, 2000)
TT = df['T_thc'].values
VV = df['viscosity'].values

best_Rs = []

for v, t in zip(VV, TT):
    A = getGlycerolViscosity(RR, t)
    B = np.abs(A - v)
    i = np.argmin(B)
    best_r = float(RR[i])
    best_Rs.append(best_r)
    

best_Rs = np.array(best_Rs)
df['Ratio_with_T_thc'] = best_Rs*100

samples = df['sample'].unique()
labels = ['Ref', 'MA', 'AR']
samples_R = []
for s in samples:
    df_s = df[df['sample'] == s]
    mean_R = np.mean(df_s['Ratio_with_T_thc'])
    samples_R.append(mean_R)
    
best_Rs = np.array(best_Rs)
df['Ratio_calc_with_T_thc'] = best_Rs*100
df['rep_int'] = df['rep'].apply(lambda x: int(x[-1]))

fig, ax = plt.subplots(1, 1, layout='compressed')

sns.scatterplot(data=df, ax=ax, x='rep_int', y='Ratio_with_T_thc',
                hue='sample')
for i, R in enumerate(samples_R):
    ax.axhline(R, color=pm.cL_Set21[i], label=labels[i] + f', r = {R:.2f}%',
               lw=0.75)

ax.set_xlabel('Replicates')
ax.set_ylabel('Ratio (%)')
ax.set_xticks([1, 2, 3, 4])
ax.legend().set_visible(False)
fig.legend(loc='outside right center')
fig.suptitle('Glycerol Ratio in samples')
# ax.grid()
plt.show()

# df.to_csv(filePath, index=False)

# %%%% 4. As a reference

# RR = np.arange(79, 81.25, 0.25)
# TT = np.linspace(20, 28, 1000)
RR = np.arange(79.75, 81, 0.25)
TT = np.linspace(20, 24, 1000)


VVV = [getGlycerolViscosity(R/100, TT) for R in RR]

fig, ax = plt.subplots(1, 1)
for R, VV in zip(RR, VVV):
    ax.plot(TT, VV, label=f'{R:.2f} %')
    
ax.set_xlabel('Temp (°C)')
ax.set_ylabel('Visco (mPa.s)')
ax.legend(title='Glycerol ratio')
ax.grid()
plt.show()

