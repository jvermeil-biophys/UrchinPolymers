# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 15:05:01 2025

@author: Utilisateur
"""

# %% Imports

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import GraphicStyles as gs


# %% Open table

gs.set_default_options_jv(palette = 'Set2')

dirPath = 'C:/Users/Utilisateur/Desktop/WorkingData/SmallLeicaData/UV survival tests'
fileName = '25-10-16_FilmSurvieQuantif.csv'
filePath = os.path.join(dirPath, fileName)

df = pd.read_csv(filePath, na_values=[np.nan])
df = df.rename(columns={'T 2-cells':'2C', 'T 4-cells':'4C', 'T 8-cells':'8C', 'T dead':'D'})

for col in ['2C', '4C', '8C']:
    df[col] = df[col].replace({np.nan: np.inf})
        
df_desc = df.describe()

TT = np.arange(1, 170, 1)
n1C = np.array([np.sum(df['2C'] > t) - np.sum(df['D'] <= t) for t in TT])
n2C = np.array([np.sum(df['2C'] <= t) - np.sum(df['4C'] <= t) for t in TT])
n4C = np.array([np.sum(df['4C'] <= t) - np.sum(df['8C'] <= t) for t in TT])
n8C = np.array([np.sum(df['8C'] <= t) for t in TT])
nD = np.array([np.sum(df['D'] <= t) for t in TT])

Ncycling = np.sum(df['2C'] < np.inf)
Narrested = np.sum(df['2C'] == np.inf) - np.sum(df['D'] > 0)
Ndead = np.sum(df['D'] > 0)
Ntot = Ncycling+Narrested+Ndead
df_status = pd.DataFrame({'time':['end', 'end', 'end'],
                          'status':['cycling', 'arrested', 'dead'],
                          'N':[Ncycling, Narrested, Ndead]})
df_status['N%'] = 100*df_status['N']/Ntot


fig, axes = plt.subplots(1, 2, figsize = (8, 4.5))
# fig, axes = plt.subplots(1, 3, figsize = (12, 4))

ax = axes[0]
ax.plot(TT, n1C, lw=2, label='1 cell stage', alpha=0.75)
ax.plot(TT, n2C, lw=2, label='2 cell stage', alpha=0.75)
ax.plot(TT, n4C, lw=2, label='4 cell stage', alpha=0.75)
ax.plot(TT, n8C, lw=2, label='8 cell stage', alpha=0.75)
ax.plot(TT, nD, lw=2, label='dead', c='dimgray', alpha=0.75)
ax.set_xlabel('Time (min)')
ax.set_ylabel('N cells')
ax.set_ylim([0, 25])
ax.grid()
ax.legend(ncols = 3, handlelength=1, fontsize=7)

ax = axes[1]
#df = df.rename(columns={'T 2-cells':'2C', 'T 4-cells':'4C', 'T 8-cells':'8C'})
df_f = df.dropna(subset=['2C'])
df_m = df_f.melt(id_vars=['Id Cell'], 
                 value_vars=['2C', '4C', '8C'], 
                 var_name = 'stage', value_name='time')
sns.boxplot(ax = ax, data = df_m, x = 'stage', y = 'time', hue = 'stage', palette=gs.cL_Set2[1:1+3])
sns.swarmplot(ax = ax, data = df_m, x = 'stage', y = 'time', color='k')
ax.set_xlabel('Stage')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_ylabel('Time (min)')
ax.grid()

# ax = axes[2]
# sns.barplot(data=df_status, ax=ax,
#             x='time', y='N%', hue='status', # forward order
#             dodge=False)

fig.suptitle('UV 10min 100%, N=21 cells')

fig.tight_layout()

plt.show()
