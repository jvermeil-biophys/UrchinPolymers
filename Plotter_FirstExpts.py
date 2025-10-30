# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 13:46:39 2025

@author: Joseph Vermeil
"""

# %% 1. Imports

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

def set_default_options_jv(palette = 'Set2'):
    SMALLER_SIZE = 9
    SMALL_SIZE = 13
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 20
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALLER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    sns.set_palette(sns.color_palette(palette))

set_default_options_jv(palette = 'Set2')

# %% 2. Plot Rheometer

# %%% Load files

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

# %%% Split single big file

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
                
            
# %% 3. Plot Droplet Pulling

mainDir = 'C:/Users/Utilisateur/Desktop/MicroscopeData/Analysis_Pulls/Results'
date = '25-09-19'
# date = '25-09-19'

df = pd.read_csv(os.path.join(mainDir, date + '_NaSS_results.csv'))
df['UV'] = pd.Series(['']*len(df))
df.loc[df['treatment']=='none', 'UV'] = 'No'
df.loc[df['treatment']!='none', 'UV'] = '20min 100%'

fig1, ax1 = plt.subplots(1, 1, figsize = (3, 4))
fig, ax = fig1, ax1
sns.boxplot(ax=ax, data=df, x='UV', y='viscosity')
sns.swarmplot(ax=ax, data=df, x='UV', y='viscosity', size=8, edgecolor='k', linewidth=0.5)
# ax.set_ylim([0, 15])
ax.grid(axis='y')
ax.set_xlabel('UV')
ax.set_ylabel('Viscosity (mPa.s)')
fig.tight_layout()
plt.show()

fig.savefig(os.path.join(mainDir, date + '_NaSS_results.png'))


fig2, ax2 = plt.subplots(1, 1, figsize = (6, 4))
fig, ax = fig2, ax2
sns.scatterplot(ax=ax, data=df, x='bead radius', y='viscosity', hue='treatment', s=40)
ax.set_ylim([0, ax2.get_ylim()[1]])
ax.grid(axis='y')
ax.set_xlabel('Bead radius (µm)')
ax.set_ylabel('Viscosity (mPa.s)')
fig.tight_layout()
plt.show()


fig3, ax3 = plt.subplots(1, 1, figsize = (6, 4))
fig, ax = fig3, ax3
sns.scatterplot(ax=ax, data=df, x='R min', y='viscosity', hue='treatment', s=40)
ax.set_ylim([0, ax2.get_ylim()[1]])
ax.grid(axis='y')
ax.set_xlabel('Min dist to magnet tip (µm)')
ax.set_ylabel('Viscosity (mPa.s)')
fig.tight_layout()
plt.show()

fig.savefig(os.path.join(mainDir, date + '_NaSS_biasWithR.png'))


fig4, ax4 = plt.subplots(1, 1, figsize = (6, 4))
fig, ax = fig4, ax4
sns.scatterplot(ax=ax, data=df, x='theta', y='viscosity', hue='treatment', s=40)
ax.set_ylim([0, ax2.get_ylim()[1]])
ax.grid(axis='y')
ax.set_xlabel('Angle with the horiz (rad)')
ax.set_ylabel('Viscosity (mPa.s)')
fig.tight_layout()
plt.show()


# %% 4. Plot Maribel's calibration

# %%%

mainDir = 'C:/Users/Utilisateur/Desktop/Data From Maribel/Magnet_Calibration/dynabeads'
fileName = 'dist-speed_ALL.txt'
filePath = os.path.join(mainDir, fileName)

df = pd.read_csv(filePath, sep = ' ', engine='python', names = ['id', 'r', 'v'])

fig, ax = plt.subplots(1, 1)
sns.scatterplot(data=df, ax=ax, x='r', y='v', s = 10)

ax.grid()
fig.tight_layout()

plt.show()

# %%%

d2v = lambda x: 80.23*np.exp((-x)/47.49) + 1.03*np.exp((-x)/22740.0)

def Classic_D2V(x, A1, k1, A2, k2):
    return(A1*np.exp(-x/k1) + A2*np.exp(-x/k2))


def Langevin(x):
    # if x < 0.05:
    #     return(np.tanh(x/3))
    # else:
    return((1/np.tanh(x)) - (1/x))

# def New_D2V(x, A, B, x0):
#     return(A * Langevin(B/(x-x0)**3) * 1/(x-x0)**4)

def New_D2V(x, A, B):
    X0 = -140
    return(A * Langevin(B/(x-X0)**3) * 1/(x-X0)**4)

Filter = df['r'] > 10

XX = df[Filter]['r'].values
YY = df[Filter]['v'].values

popt1, pcov1 = curve_fit(Classic_D2V, XX, YY, p0=[80, 40, 0.5, 2000]) # p0=[1e10, 1e21, -100]
popt2, pcov2 = curve_fit(New_D2V, XX, YY, p0=[1e10, 1e18]) # p0=[1e10, 1e21, -100]

Xfit = np.linspace(min(XX), max(XX), 1000)
Yfit1 = Classic_D2V(Xfit, *popt1)
Yfit2 = New_D2V(Xfit, *popt2)

fig, ax = plt.subplots(1, 1)
sns.scatterplot(data=df, ax=ax, x='r', y='v', s = 10)
ax.plot(Xfit, Yfit1, c='darkred', ls='-', lw=1)
ax.plot(Xfit, Yfit2, c='darkorange', ls='-', lw=1)
ax.grid()
fig.tight_layout()


# new_d2v = lambda x: New_D2V(x, 2.75379e+10, 6.87603e+21, -124.896)
new_d2v = lambda x: New_D2V(x, 1.7435e+10, 4.65859e+19)

# With X0 = 0
# popt = [6.46079e+10, 1.09389e+21]

# With X0 free
# popt = [5.18837e+11, 9.3768e+20, -183.281]

# With X0 = -100
# popt = [5.18837e+11, 9.3768e+20]

