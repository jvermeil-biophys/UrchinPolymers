# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 11:41:22 2026

@author: Utilisateur
"""

# %% 1. Imports

import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from copy import deepcopy
from scipy.io import savemat
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import make_splrep

# Local imports
import PlotMaker as pm
import UrchinPaths as up
import UtilityFunctions as ufun
pm.setGraphicOptions(mode = 'screen', palette = 'Set2', colorList = pm.colorList10)


# %% 2. Functions

def PlotPullMetrics_swarm(df, group_by_cell = 'False', agg_func = 'mean'):
    """
    
    """
    metrics = ['J_k', 'J_gamma1', 'J_gamma2', 'N_viscosity', ]
    metric_names = ['$k_1$ (pN/µm)', '$\gamma_1$ (pN.s/µm)', '$\gamma_2$ (pN.s/µm)', '$\eta_N$ (Pa.s)']
    metric_dict = {m:mn for (m,mn) in zip(metrics, metric_names)}
    
    if group_by_cell:
        group = df.groupby(['Cell_textId', 'Pa_id'])
        agg_dict = {m:agg_func for m in metrics}
        dfg = group.agg(agg_dict)
        df = dfg.reset_index()
        
    
    fig, axes = plt.subplots(2, 2, figsize=(8,5))
    axes_f = axes.flatten()
    
    for k in range(4):
        ax = axes_f[k]
        metric = metrics[k]
        
        medians = [np.median(df.loc[df['Pa_id']==i, metric]) for i in [0, 1]]
        
        for i in [0, 1]:
            HW = 0.35
            ax.plot([i-HW, i+HW], [medians[i], medians[i]], ls='--', lw=2, c='dimgrey')
        
        sns.swarmplot(data = df, ax=ax, 
                      x = 'Pa_id', y = metric, hue = 'Cell_textId',
                      size = 6)
    
        ax.set_xlim([-0.5, 1.5])
        ax.set_xticks([0, 1], ['Ctrl', '+UV'])
        ax.set_xlabel('')
        ax.set_ylabel(metric_dict[metric])
        yM = ax.get_ylim()[1]
        ax.set_ylim([0, 1.25*yM])
        ax.grid(axis='y')
        
        for i in [0, 1]:
            ax.text(i, 1.1*yM, f'{medians[i]:.2f}', ha='center', size = 10, style='italic', c='dimgrey')
        
        if k == 1:
            ax.legend(title='Cell IDs',
                      loc="upper left",
                      bbox_to_anchor=(1, 0, 0.5, 1))
        else:
            ax.legend().set_visible(False)
    
    fig.suptitle('All pulls & beads' + specif, fontsize=16)
    fig.tight_layout()
    plt.show()
    
    return(fig, axes)
    
    # fig.savefig(savePath + '/res_allPulls' + specif + '.png', dpi=500)


def PlotPullMetrics_bipart(df, agg_func = 'mean', normalized = False):
    """
    
    """
    metrics = ['J_k', 'J_gamma1', 'J_gamma2', 'N_viscosity', ]
    metric_names = ['$k_1$ (pN/µm)', '$\gamma_1$ (pN.s/µm)', '$\gamma_2$ (pN.s/µm)', '$\eta_N$ (Pa.s)']
    metric_names_2 = ['$k_1$ (ratio)', '$\gamma_1$ (ratio)', '$\gamma_2$ (ratio)', '$\eta_N$ (ratio)']
    
    if not normalized:
        metric_dict = {m:mn for (m,mn) in zip(metrics, metric_names)}
    if normalized:
        metric_dict = {m+'_norm':mn for (m,mn) in zip(metrics, metric_names_2)}
    
    group = df.groupby(['Cell_textId', 'Pa_id'])
    agg_dict = {m:agg_func for m in metrics}
    dfg = group.agg(agg_dict)
    dfg = dfg.reset_index()
    
    list_cell_id = df['Cell_textId'].values
    for metric in agg_dict.keys():
        A_norm = np.array([dfg.loc[((dfg['Pa_id']==0) & (dfg['Cell_textId']==cid)), metric] for cid in list_cell_id]).T
        A_norm = A_norm[0]
        dfg[metric + '_norm'] = dfg[metric] / A_norm   
        
    fig, axes = plt.subplots(2, 2)
    axes_f = axes.flatten()

    for k in range(4):
        ax = axes_f[k]
        if not normalized:
            metric = metrics[k]
        if normalized:
            metric = metrics[k] + '_norm'
        
        for cid, c in zip(dfg['Cell_textId'].unique(), pm.colorList10):
            val0 = dfg.loc[((dfg['Pa_id']==0) & (dfg['Cell_textId']==cid)), metric].values[0]
            val1 = dfg.loc[((dfg['Pa_id']==1) & (dfg['Cell_textId']==cid)), metric].values[0]
            ax.plot(0, val0, 'o', c=c, zorder = 5)
            ax.plot(1, val1, 'o', c=c, zorder = 5)
            ax.plot([0, 1], [val0, val1], ls='-', c='dimgray', zorder = 4)
            
        ax.set_xlim([-0.5, 1.5])
        ax.set_xticks([0, 1], ['Ctrl', '+UV'])
        ax.set_xlabel('')
        ax.set_ylabel(metric_dict[metric])
        yM = ax.get_ylim()[1]
        ax.set_ylim([0, 1.1*yM])
        
        ax.grid(axis='y')
            
    fig.suptitle('Average values by cell' + specif, fontsize=16)
    fig.tight_layout()
    plt.show()
    
    return(fig, axes)
    
    # fig.savefig(savePath + '/res_allPulls' + specif + '.png', dpi=500)


def PlotPullMetrics_oneBead(df, Cell_textId, BeadNo=1, iUV=2):
    Filters = [
               (df['Cell_textId'] == Cell_textId),
               (df['bead'] == BeadNo),
               ]
    
    iUV = iUV # nPulls_beforeUV
    
    GlobalFilter = np.ones_like(Filters[0]).astype(bool)
    for F in Filters:
        GlobalFilter = GlobalFilter & F
    
    df_1c1b = df[GlobalFilter]
    df_1c1b['Idx Pull'] = np.arange(1, 1+len(df_1c1b))
    
    fig, axes = plt.subplots(1, 2, figsize = (10, 4))
    
    ax = axes[0]
    ax.plot(df_1c1b['Idx Pull'], df_1c1b['J_visco2'], 'o', markerfacecolor = 'None', label='$\eta_2$ (Jeffrey)')
    ax.plot(df_1c1b['Idx Pull'], df_1c1b['N_viscosity'], 'o', markerfacecolor = 'None', label='$\eta_N$  (Newton)')
    ax.axvspan(iUV + 0.25, iUV + 0.75, color='indigo', zorder=5)
    ymid = np.mean(ax.get_ylim())
    ax.text(iUV + 0.5, ymid, 'UV', c='w', style='italic', size=10, ha='center', va='center', zorder=6)
    ax.set_xlabel('Pull No.')
    ax.set_ylabel('$\eta$ (Pa.s)')
    ax.grid(axis='y')
    ax.legend()
    
    ax = axes[1]
    axbis = ax.twinx()
    ax.plot(df_1c1b['Idx Pull'], df_1c1b['J_visco1'], 'o', color=pm.colorList10[0], markerfacecolor = 'None')
    ax.plot([], [], 'o', color=pm.colorList10[1], markerfacecolor = 'None', label='k')
    axbis.plot(df_1c1b['Idx Pull'], df_1c1b['J_k'], 'o', color=pm.colorList10[1], markerfacecolor = 'None')
    ax.axvspan(iUV + 0.25, iUV + 0.75, color='indigo', zorder=5)
    ymid = np.mean(ax.get_ylim())
    ax.text(iUV + 0.5, ymid, 'UV', c='w', style='italic', size=10, ha='center', va='center', zorder=6)
    ax.set_xlabel('Pull No.')
    ax.set_ylabel('$\eta_1$ (Pa.s)', color=pm.colorList10[0])
    axbis.set_ylabel('$k_1$ (pN/µm)', color=pm.colorList10[1])
    ax.grid(axis='y')
    
    fig.suptitle(f'Repeated pulls on {Cell_textId} Bead n°{BeadNo:.0f}', fontsize = 12)
    fig.tight_layout()
    plt.show()
    
    return(fig, axes)

# %% 3. Run plots

# %%% 26-01-27

mainDir = up.Path_AnalysisPulls
fileName = 'AllResults_BeadsPulling.csv'
filePath = os.path.join(mainDir, fileName)

specif = '_26-01-27_NoI2959'

df = pd.read_csv(filePath)

df['date'] = df['track_id'].apply(lambda x : x.split('_')[0])
df['Lc'] = 6*np.pi*df['bead radius']
df['J_modulus'] = df['J_k'] / df['Lc']
df['J_visco1'] = df['J_gamma1'] / df['Lc']
df['J_visco2'] = df['J_gamma2'] / df['Lc']

Filters = [(df['J_gamma2'] < 500),
           (df['J_fit_error'] == False),
           (df['J_R2'] >= 0.7),
           (df['date'] == '26-01-27'),
           (df['Cell_textId'] != '26-01-14_M1_C1'),
           ]
GlobalFilter = np.ones_like(Filters[0]).astype(bool)
for F in Filters:
    GlobalFilter = GlobalFilter & F

df = df[GlobalFilter]
savePath = mainDir

OneBead_Cell_textId = '26-01-27_M1_C1'

fig, axes = PlotPullMetrics_swarm(df, group_by_cell = 'False', agg_func = 'mean')
fig, axes = PlotPullMetrics_bipart(df, agg_func = 'mean', normalized = False)
fig, axes = PlotPullMetrics_oneBead(df, OneBead_Cell_textId, BeadNo=1, iUV=3)

# %%% 26-01-14

mainDir = up.Path_AnalysisPulls
fileName = 'AllResults_BeadsPulling.csv'
filePath = os.path.join(mainDir, fileName)

specif = '_26-01-14_FullMix'

df = pd.read_csv(filePath)

df['date'] = df['track_id'].apply(lambda x : x.split('_')[0])
df['Lc'] = 6*np.pi*df['bead radius']
df['J_modulus'] = df['J_k'] / df['Lc']
df['J_visco1'] = df['J_gamma1'] / df['Lc']
df['J_visco2'] = df['J_gamma2'] / df['Lc']

Filters = [(df['J_gamma2'] < 500),
           (df['J_fit_error'] == False),
           (df['J_R2'] >= 0.7),
           (df['date'] == '26-01-14'),
           (df['Cell_textId'] != '26-01-14_M1_C1'),
           ]
GlobalFilter = np.ones_like(Filters[0]).astype(bool)
for F in Filters:
    GlobalFilter = GlobalFilter & F

df = df[GlobalFilter]
savePath = mainDir



# %%% Plots by cell

metrics = ['J_k', 'J_gamma1', 'J_gamma2', 'N_viscosity', ]
group = df.groupby(['cell_id', 'Pa'])
agg_dict = {m:'mean' for m in metrics}
dfg = group.agg(agg_dict)
dfg = dfg.reset_index()
dfg = dfg[dfg['cell_id'] != '26-01-14_M1_C1']

list_cell_id = dfg['cell_id'].values
for metric in agg_dict.keys():
    A_norm = np.array([dfg.loc[((dfg['Pa']==0) & (dfg['cell_id']==cid)), metric] for cid in list_cell_id]).T
    A_norm = A_norm[0]
    dfg[metric + '_norm'] = dfg[metric] / A_norm

metric_names = ['$k_1$ (pN/µm)', '$\gamma_1$ (pN.s/µm)', '$\gamma_2$ (pN.s/µm)', '$\eta_N$ (Pa.s)']
metric_dict = {m:mn for (m,mn) in zip(metrics, metric_names)}

fig, axes = plt.subplots(2, 2)
axes_f = axes.flatten()

for k in range(4):
    ax = axes_f[k]
    metric = metrics[k]
    
    for cid, c in zip(dfg['cell_id'].unique(), pm.colorList10):
        val0 = dfg.loc[((dfg['Pa']==0) & (dfg['cell_id']==cid)), metric].values[0]
        val1 = dfg.loc[((dfg['Pa']==1) & (dfg['cell_id']==cid)), metric].values[0]
        ax.plot(0, val0, 'o', c=c, zorder = 5)
        ax.plot(1, val1, 'o', c=c, zorder = 5)
        ax.plot([0, 1], [val0, val1], ls='-', c='dimgray', zorder = 4)
        
    
    
    ax.set_xlim([-0.5, 1.5])
    ax.set_xticks([0, 1], ['Ctrl', '+UV'])
    ax.set_xlabel('')
    ax.set_ylabel(metric_dict[metric])
    yM = ax.get_ylim()[1]
    ax.set_ylim([0, 1.1*yM])
    
    ax.grid(axis='y')
        
fig.suptitle('Average values by cell' + specif, fontsize=16)
fig.tight_layout()
plt.show()
    
fig.savefig(savePath + '/res_avgByCell' + specif + '.png', dpi=500)
    
    
metric_names_2 = ['$k_1$ (ratio)', '$\gamma_1$ (ratio)', '$\gamma_2$ (ratio)', '$\eta_N$ (ratio)']
metric_dict_2 = {m+'_norm':mn for (m,mn) in zip(metrics, metric_names_2)}

fig, axes = plt.subplots(2, 2)
axes_f = axes.flatten()

for k in range(4):
    ax = axes_f[k]
    metric = metrics[k] + '_norm'
    
    for cid, c in zip(dfg['cell_id'].unique(), pm.colorList10):
        val0 = dfg.loc[((dfg['Pa']==0) & (dfg['cell_id']==cid)), metric].values[0]
        val1 = dfg.loc[((dfg['Pa']==1) & (dfg['cell_id']==cid)), metric].values[0]
        ax.plot(0, val0, 'o', c=c, zorder = 5)
        ax.plot(1, val1, 'o', c=c, zorder = 5)
        ax.plot([0, 1], [val0, val1], ls='-', c='dimgray', zorder = 4)
    
    ax.set_xlim([-0.5, 1.5])
    ax.set_xticks([0, 1], ['Ctrl', '+UV'])
    ax.set_xlabel('')
    ax.set_ylabel(metric_dict_2[metric])
    yM = ax.get_ylim()[1]
    ax.set_ylim([0, 1.1*yM])
    
    ax.grid(axis='y')

fig.suptitle('Average values by cell, normalized' + specif, fontsize=16)    
fig.tight_layout()
plt.show()
    
fig.savefig(savePath + '/res_avgByCell_norm' + specif + '.png', dpi=500)
    
    
    

# %%% Data for one bead across pullings





# %%%

# fig.savefig(savePath + '/RepPull_v2vN_26-01-27_M1_C1_B1.png', dpi=500)

# # ------

# fig, ax = plt.subplots(1, 1, figsize = (5.5,4))

# # ax.legend() 

# fig.suptitle('Repeated pulls on 26-01-27_M1_C1 Bead n°1', fontsize = 12)
# fig.tight_layout()
# plt.show()


# fig.savefig(savePath + '/RepPull_k1v1_26-01-27_M1_C1_B1.png', dpi=500)


# %%% Compare before / after UV on an example

#### Parms

mainDir = os.path.join("C:/Users/Utilisateur/Desktop/")
date = '26-01-14'
subfolder = date + '_BeadTracking'

analysisDir = os.path.join(mainDir, 'AnalysisPulls') # where the csv tables are
tracksDir = os.path.join(analysisDir, subfolder, 'Tracks') # where the tracks are
resultsDir = os.path.join(analysisDir, subfolder, 'Results')
plotsDir = os.path.join(analysisDir, subfolder, 'Plots')

calibFuncType = 'PowerLaw'
PLOT, SHOW = True, True

listTrackFiles = [
    '26-01-14_M1_C3_Pa0_P2_B1_Tracks.xml',
    '26-01-14_M1_C3_Pa1_P1_B1_Tracks.xml',
    ]


#### Data
if not os.path.exists(resultsDir):
    os.makedirs(resultsDir)
if not os.path.exists(plotsDir):
    os.makedirs(plotsDir)
    
df_manips = pd.read_csv(os.path.join(analysisDir, 'MainExperimentalConditions.csv'))
df_pulls = pd.read_csv(os.path.join(analysisDir, date + '_ExperimentalConditions.csv'))

listTrackIds = ['_'.join(string.split('_')[:6]) for string in listTrackFiles]
listPullIds = ['_'.join(string.split('_')[:5]) for string in listTrackFiles]
listManipIds = ['_'.join(string.split('_')[:2]) for string in listTrackFiles]
list_co_dict = [df_manips[df_manips['id'] == mid].to_dict('list') for mid in listManipIds]

dict_TrackIds2File = {tid : tf for (tid, tf) in zip(listTrackIds, listTrackFiles)}
dict_TrackIds2PullId = {tid : pid for (tid, pid) in zip(listTrackIds, listPullIds)}

list_tracks, list_track_ids, list_dict_pull, list_mag_d2f = [], [], [], []

for it, T_id in enumerate(listTrackIds):
    trackFileName = dict_TrackIds2File[T_id]
    tracks = importTrackMateTracks(os.path.join(tracksDir, trackFileName))
    track = trackSelection(tracks, mode = 'longest')
    
    P_id = dict_TrackIds2PullId[T_id]
    dict_pull = Df2Dict(df_pulls[df_pulls['id'] == P_id])
    co_dict = list_co_dict[it]
    
    magnet_id = co_dict['magnet'][0]
    bead_type = co_dict['bead type'][0]
    pixel_size = df_pulls.loc[df_pulls['id']==P_id, 'film_pixel_size'].values[0]
    bead_radius = df_pulls.loc[df_pulls['id']==P_id, 'b_r'].values[0]
    
    case_mag = magnet_id + '_' + bead_type
    if calibFuncType == 'PowerLaw':
        parms_type = 'F_popt_pL'
        parms = MagnetDict[case_mag][parms_type]
        mag_d2f = (lambda x : powerLaw(x, *parms))
    elif calibFuncType == 'DoubleExpo':
        parms_type = 'F_popt_2exp'
        parms = MagnetDict[case_mag][parms_type]
        mag_d2f = (lambda x : doubleExpo(x, *parms))
    
    list_tracks.append(track)
    list_track_ids.append(T_id)
    list_dict_pull.append(dict_pull)
    list_mag_d2f.append(mag_d2f)
        
#### Run the function

pfTitle = '26-01-14_M1_C3_ctrl-vs-UV'

N_Rd, N_error = pullAnalyzer_compareTracks(list_tracks, list_track_ids, list_dict_pull, list_mag_d2f,
                             mode = 'newton', 
                             PLOT = PLOT, SHOW = SHOW, plotsDir = resultsDir, plotFileTitle = pfTitle)

J_Rd, J_error = pullAnalyzer_compareTracks(list_tracks, list_track_ids, list_dict_pull, list_mag_d2f,
                             mode = 'jeffrey', 
                             PLOT = PLOT, SHOW = SHOW, plotsDir = resultsDir, plotFileTitle = pfTitle)



