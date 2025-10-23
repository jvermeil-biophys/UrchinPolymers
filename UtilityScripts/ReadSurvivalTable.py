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

df = pd.read_csv(filePath)
df_desc = df.describe()

TT = np.arange(1, 170, 1)
n1C = np.array([np.sum(df['T 2-cells'] > t) for t in TT])
n2C = np.array([np.sum(df['T 2-cells'] <= t) - np.sum(df['T 4-cells'] <= t)  for t in TT])
n4C = np.array([np.sum(df['T 4-cells'] <= t) - np.sum(df['T 8-cells'] <= t)  for t in TT])
n8C = np.array([np.sum(df['T 8-cells'] <= t) for t in TT])

fig, axes = plt.subplots(1, 2, figsize = (8, 4))

ax = axes[0]
ax.plot(TT, n1C, lw=2, label='1 cell stage')
ax.plot(TT, n2C, lw=2, label='2 cell stage')
ax.plot(TT, n4C, lw=2, label='4 cell stage')
ax.plot(TT, n8C, lw=2, label='8 cell stage')
ax.set_xlabel('Time (min)')
ax.set_ylabel('N cells')
ax.set_ylim([0, 20])
ax.grid()
ax.legend(ncols = 2)

ax = axes[1]
df_f = df.dropna(subset=['T 2-cells']).rename(columns={'T 2-cells':'2C', 'T 4-cells':'4C', 'T 8-cells':'8C'})
df_m = df_f.melt(id_vars=['Id Cell'], 
                 value_vars=['2C', '4C', '8C'], 
                 var_name = 'stage', value_name='time')
sns.boxplot(ax = ax, data = df_m, x = 'stage', y = 'time')
ax.set_xlabel('Stage')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_ylabel('Time (min)')
ax.grid()

fig.tight_layout()

plt.show()
