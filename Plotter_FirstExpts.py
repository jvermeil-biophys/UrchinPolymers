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

mainDir = 'C:/Users/josep/Desktop/Pulls/RheoMacro'
listFiles = os.listdir(mainDir)
listPaths = [os.path.join(mainDir, f) for f in listFiles if f.endswith('.csv')]

dictResults = {'date':[],
               'polymer':[],
               'PI':[],
               'UV':[],
               'viscosity':[],
               'fileName':[],}

for f in listFiles:
    if f.endswith('.csv') and not f.startswith('Results'):
        blocks = f[:-4].split('_')
        dictResults['date'].append(blocks[0])
        dictResults['polymer'].append(blocks[1])
        dictResults['PI'].append(blocks[2])
        dictResults['UV'].append(blocks[3])
        dictResults['fileName'].append(f)
        
        path = os.path.join(mainDir, f)
        df = pd.read_csv(path, header = 4, sep='\t', #skiprows=2,
                         on_bad_lines='skip', encoding='utf_16_le')
        df = df.drop(df.columns[:2], axis = 1).drop(df.index[:2], axis = 0).reset_index(drop=True)
        viscosity = np.median(df['Viscosity'].astype(float).values)
        
        dictResults['viscosity'].append(viscosity)
        
df_summary = pd.DataFrame(dictResults)
df_summary.to_csv(os.path.join(mainDir, 'ResultsMacroRheo.csv'), index=False)
    

# %% 3. Plot Droplet Pulling

mainDir = 'C:/Users/josep/Desktop/Pulls/'

df = pd.read_csv(os.path.join(mainDir, 'NaSS_results.csv'))
df['UV'] = ['']*len(df)
df['UV'].loc[df['treatment']=='none'] = 'No'
df['UV'].loc[df['treatment']!='none'] = '20min 100%'

fig, ax = plt.subplots(1, 1, figsize = (3, 4))
sns.swarmplot(ax=ax, data=df, x='UV', y='viscosity', size=10)
ax.set_ylim([0, 15])
ax.grid(axis='y')
ax.set_xlabel('UV')
ax.set_ylabel('Viscosity (mPa.s)')
plt.tight_layout()
plt.show()

fig.savefig(os.path.join(mainDir, 'NaSS_results.png'))