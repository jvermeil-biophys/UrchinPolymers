# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 11:41:22 2026

@author: Utilisateur
"""

# %% 1. Imports

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Local imports
import PlotMaker as pm
import UrchinPaths as up
import UtilityFunctions as ufun
pm.setGraphicOptions(mode = 'screen', palette = 'Set2', colorList = pm.colorList10)


#### Settings

SCALE_20X = 0.461
SCALE_40X = 0.229
FPS = 1


# %% 2. Functions

# %%% Update dataset

mainDir = up.Path_AnalysisPulls
tablePath = os.path.join(mainDir, 'AllResults_BeadsPulling.csv')


# def get_bead_number(S):
#     try:
#         return(ufun.get_numbers_following_text(S, '_B'))
#     except:
#         return(1)


# df_Pulls = pd.read_csv(tablePath)
# df_Pulls['Manip_textid'] = df_Pulls['track_id'].apply(lambda x : '_'.join(x.split('_')[:2]))
# df_Pulls['Cell_textid'] = df_Pulls['track_id'].apply(lambda x : '_'.join(x.split('_')[:3]))
# df_Pulls['Manip_id'] = df_Pulls['track_id'].apply(lambda x : ufun.get_numbers_following_text(x, '_M'))
# df_Pulls['Cell_id'] = df_Pulls['track_id'].apply(lambda x : ufun.get_numbers_following_text(x, '_C'))
# df_Pulls['Pa_id'] = df_Pulls['track_id'].apply(lambda x : ufun.get_numbers_following_text(x, '_Pa'))
# df_Pulls['Pull_id'] = df_Pulls['track_id'].apply(lambda x : ufun.get_numbers_following_text(x, '_P'))
# df_Pulls['Bead_id'] = df_Pulls['track_id'].apply(lambda x : get_bead_number(x))




# df_Pulls.to_csv(tablePath, index=False)



# %%% Functions pull

def PlotPullMetrics_swarm(df, group_by_cell = 'False', agg_func = 'mean'):
    """
    
    """
    metrics = ['J_k', 'J_gamma1', 'J_gamma2', 'N_viscosity', ]
    metric_names = ['$k_1$ (pN/µm)', '$\gamma_1$ (pN.s/µm)', '$\gamma_2$ (pN.s/µm)', '$\eta_N$ (Pa.s)']
    metric_dict = {m:mn for (m,mn) in zip(metrics, metric_names)}
    
    Pa_ids = df['Pa_id'].unique()
    
    # print(df)
    
    # if group_by_cell:
    #     group = df.groupby(['Cell_textId', 'Pa_id'])
    #     agg_dict = {m:agg_func for m in metrics}
    #     dfg = group.agg(agg_dict)
    #     df = dfg.reset_index()
        
    # print(df)
    
    fig, axes = plt.subplots(2, 2, figsize=(8,5))
    axes_f = axes.flatten()
    
    for k in range(4):
        ax = axes_f[k]
        metric = metrics[k]
        
        medians = [np.median(df.loc[df['Pa_id']==i, metric]) for i in Pa_ids]
        
        for i in range(len(Pa_ids)):
            HW = 0.35
            ax.plot([i-HW, i+HW], [medians[i], medians[i]], ls='--', lw=2, c='dimgrey')
        
        sns.swarmplot(data = df, ax=ax, 
                      x = 'Pa_id', y = metric, hue = 'Cell_textId',
                      size = 6)
    
        ax.set_xlim([-0.5, len(Pa_ids)-0.5])
        xticks_labels = ['Ctrl'] + [f'+UV (Pa{str(k)})' for k in Pa_ids[1:]]
        ax.set_xticks([k for k in range(len(Pa_ids))], xticks_labels, rotation=15)
        ax.set_xlabel('')
        ax.set_ylabel(metric_dict[metric])
        yM = ax.get_ylim()[1]
        ax.set_ylim([0, 1.25*yM])
        ax.grid(axis='y')
        
        for i in range(len(Pa_ids)):
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
    
    Pa_ids = df['Pa_id'].unique()
    Pa_ids_X = {Pa_ids[k]:k for k in range(len(Pa_ids))}
    
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
    # return(dfg)
    
    list_cell_id = df['Cell_textId'].values
    for metric in agg_dict.keys():
        A_norm = np.array([dfg.loc[((dfg['Pa_id']==0) & (dfg['Cell_textId']==cid)), metric] for cid in list_cell_id]).T
        A_norm = A_norm[0]
        print(dfg[metric], A_norm)
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
            dfg_cell = dfg[dfg['Cell_textId']==cid]
            Pa_ids_cell = dfg_cell['Pa_id'].unique()
            vals_cell = []
            X_cell = []
            for Pa_val in Pa_ids_cell:
                val = dfg_cell.loc[dfg['Pa_id']==Pa_val, metric].values[0]
                X = Pa_ids_X[Pa_val]
                ax.plot(X, val, 'o', c=c, zorder = 5)
                vals_cell.append(val)
                X_cell.append(X)
                
            ax.plot(Pa_ids_cell, vals_cell, ls='-', c='dimgray', zorder = 4)
                
            # val0 = dfg_cell.loc[((dfg['Pa_id']==0) & (dfg['Cell_textId']==cid)), metric].values[0]
            # val1 = dfg_cell.loc[((dfg['Pa_id']==1) & (dfg['Cell_textId']==cid)), metric].values[0]
            # ax.plot(0, val0, 'o', c=c, zorder = 5)
            # ax.plot(1, val1, 'o', c=c, zorder = 5)
            # ax.plot([0, 1], [val0, val1], ls='-', c='dimgray', zorder = 4)
            
        ax.set_xlim([-0.5, 1.5])
        # ax.set_xticks([0, 1], ['Ctrl', '+UV'])
        xticks_labels = ['Ctrl'] + [f'+UV (Pa{str(k)})' for k in Pa_ids[1:]]
        ax.set_xticks([k for k in range(len(Pa_ids))], xticks_labels)
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

# %% 3.1 Autocorrelation plots

# %%% Dataset

SCALE = SCALE_40X
FPS = 1

dirPath = up.Path_AnalysisPulls

df_ACF = pd.read_csv(os.path.join(dirPath, 'AllResults_ACF.csv'))
df_ExpCond = pd.read_csv(os.path.join(dirPath, 'MainExperimentalConditions.csv'))
df_ACF['N_Pa'] = df_ACF['Pa_dt'].apply(lambda x : len(str(x).split('_')))
df_ACF['manip_id'] = df_ACF['id'].apply(lambda x : '_'.join(str(x).split('_')[:2]))

# if 'long_cell_id' not in df_ACF.columns:
#     get_long_cell_id = lambda x : '_'.join(x.split('_')[:4])# + [x.split('_')[-1]])
#     df_ACF['long_cell_id'] = df_ACF['id'].apply(get_long_cell_id)
#     # df_ACF.to_csv(os.path.join(dirPath, 'Results', 'results_ACF.csv'), index=False)

metrics_cols = [col for col in df_ACF.columns if col.startswith('t_')]
for mc in metrics_cols:
    mcn = mc + '_norm'
    df_ACF[mcn] = df_ACF[mc]
for k, cid in enumerate(df_ACF['long_cell_id'].unique()):
    index_cell = df_ACF[df_ACF['long_cell_id'] == cid].index
    index_cell_control = df_ACF[(df_ACF['long_cell_id'] == cid) & (df_ACF['Pa'] == 0)].index
    for mc in metrics_cols:
        mcn = mc + '_norm'
        val_ctrl = df_ACF.loc[index_cell_control, mc].values[0]
        df_ACF.loc[index_cell, mcn] /= val_ctrl
        
df_ACF = pd.merge(left = df_ACF, right = df_ExpCond, 
                  left_on = 'manip_id', right_on = 'id', how = 'inner',
                  suffixes=(None, '_copy'))

for col in df_ACF.columns:
    if ('Unnamed' in col) or ('_copy' in col):
        df_ACF = df_ACF.drop(labels = col, axis = 1)
        
HUE_ORDER = []
for k in range(0, 3100, 100):
    HUE_ORDER += [
                  f'{k:.0f}', f'{k:.0f}_{k:.0f}', 
                  f'{k:.0f}_{k:.0f}_{k:.0f}', 
                  f'{k:.0f}_{k:.0f}_{k:.0f}_{k:.0f}',
                  ]
    
#### Filters
# df = df_ACF
# Filters = [
#            (df['id'].apply(lambda x : x.startswith('26-02-11_M2'))),
#            ]
# GlobalFilter = np.ones_like(Filters[0]).astype(bool)
# for F in Filters:
#     GlobalFilter = GlobalFilter & F
# df = df[GlobalFilter]
# df_ACF = df

# df_MSD = pd.read_csv(os.path.join(dirPath, 'Results', 'results_MSD.csv'))

# df_merged = pd.merge(left=df_MSD, right=df_ACF, on='id', how='inner', suffixes=(None, '_2'))
# df_merged['Pa'] = df_merged['Pa'].apply(lambda x : str(x))


# %%% ACF - Only one activation - Non-injected

df = df_ACF

Id_cols = ['pos_id']
Group_cols = ('Pa')
Xplot = 'Pa_total_power'
Yplot = 't_50p'
Hplot = 'Pa_irradiance'
dates = ['']

Filters = [
           (df['N_Pa'] == 1),
           (df['injection solution'] == 'none'),
           # (df['date'].apply(lambda x : x in dates)),
           ]

GlobalFilter = np.ones_like(Filters[0]).astype(bool)
for F in Filters:
    GlobalFilter = GlobalFilter & F
df = df[GlobalFilter]

hue_order = [h for h in HUE_ORDER if h in df[Hplot].unique()]

group = df.groupby(Xplot)
agg_dict = {k:'first' for k in Id_cols}
agg_dict.update({Yplot:'mean', Yplot + '_norm':'mean'})
df_g = group.agg(agg_dict).reset_index()


fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
ax.grid(zorder=0)
ax.axhline(df_g.loc[df_g['Pa_total_power']==0, Yplot].values[0], 
           color='k', linestyle='--', linewidth=1)
sns.scatterplot(data=df, ax=ax, x=Xplot, y=Yplot, 
                hue=Hplot, hue_order = hue_order,
                alpha = 0.75, zorder=6)
sns.scatterplot(data=df_g, ax=ax, x=Xplot, y=Yplot, marker = 'o',
                color='None', edgecolor='k', s=75, zorder=6)
ax.set_ylim([0, ax.get_ylim()[1]*1.05])
ax.set_xlabel(r'Total energy (J/cm2)')
ax.set_ylabel(r'$T_{50\%}$ (s)')
ax.legend().set_visible(False)

ax = axes[1]
ax.grid(zorder=0)
ax.axhline(1, color='k', linestyle='--', linewidth=1)
sns.scatterplot(data=df, ax=ax, x=Xplot, y=Yplot + '_norm', 
                hue=Hplot, hue_order = hue_order,
                alpha = 0.75, zorder=6)
sns.scatterplot(data=df_g, ax=ax, x=Xplot, y=Yplot + '_norm', marker = 'o',
                color='None', edgecolor='k', s=75, zorder=6, label='Mean\nvalues')
ax.set_ylim([0, ax.get_ylim()[1]*1.05])
ax.set_xlabel(r'Total energy (J/cm2)')
ax.set_ylabel(r'$T_{50\%}$ - normalized')
ax.legend(title='Photo-activation\nsequence [mW/cm2]', title_fontsize=10,
          loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1), ncols = 2, fontsize=9)


plt.tight_layout()
plt.show()


# %%% ACF - Several activation - Non-injected

df = df_ACF

Id_cols = ['pos_id']
Group_cols = ('Pa')
Xplot = 'Pa_total_power'
Yplot = 't_50p'
Hplot = 'Pa_irradiance'
hue_order = [h for h in HUE_ORDER if h in df[Hplot].unique()]

Filters = [
           (df['injection solution'] == 'none'),
           ]
GlobalFilter = np.ones_like(Filters[0]).astype(bool)
for F in Filters:
    GlobalFilter = GlobalFilter & F
df = df[GlobalFilter]

Filters = [
           (df['N_Pa'] == 1),
           ]
GlobalFilter = np.ones_like(Filters[0]).astype(bool)
for F in Filters:
    GlobalFilter = GlobalFilter & F
df1 = df[GlobalFilter]

Filters = [
           (df['N_Pa'] == 2),
           ]
GlobalFilter = np.ones_like(Filters[0]).astype(bool)
for F in Filters:
    GlobalFilter = GlobalFilter & F
df2 = df[GlobalFilter]

Filters = [
           (df['N_Pa'] > 2),
           ]
GlobalFilter = np.ones_like(Filters[0]).astype(bool)
for F in Filters:
    GlobalFilter = GlobalFilter & F
df3 = df[GlobalFilter]




agg_dict = {k:'first' for k in Id_cols}
agg_dict.update({Yplot:'mean', Yplot + '_norm':'mean'})

group = df1.groupby(Xplot)
df_g1 = group.agg(agg_dict).reset_index()
group = df2.groupby(Xplot)
df_g2 = group.agg(agg_dict).reset_index()
group = df3.groupby(Xplot)
df_g3 = group.agg(agg_dict).reset_index()



fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.grid(zorder=0)
ax.axhline(df_g1.loc[df_g['Pa_total_power']==0, Yplot].values[0], 
           color='k', linestyle='--', linewidth=1)
sns.scatterplot(data=df, ax=ax, x=Xplot, y=Yplot, 
                hue=Hplot, hue_order = hue_order,
                alpha = 0.75, zorder=6)
sns.scatterplot(data=df_g1, ax=ax, x=Xplot, y=Yplot, marker = 'o',
                color='None', edgecolor='b', s=75, zorder=6)
sns.scatterplot(data=df_g2, ax=ax, x=Xplot, y=Yplot, marker = 'o',
                color='None', edgecolor='g', s=75, zorder=6)
sns.scatterplot(data=df_g3, ax=ax, x=Xplot, y=Yplot, marker = 'o',
                color='None', edgecolor='r', s=75, zorder=6)
ax.set_ylim([0, ax.get_ylim()[1]*1.05])
ax.set_xlabel(r'Total energy (J/cm2)')
ax.set_ylabel(r'$T_{50\%}$ (s)')
ax.legend().set_visible(False)


ax = axes[1]
ax.grid(zorder=0)
ax.axhline(1, color='k', linestyle='--', linewidth=1)
sns.scatterplot(data=df, ax=ax, x=Xplot, y=Yplot + '_norm', 
                hue=Hplot, hue_order = hue_order,
                alpha = 0.75, zorder=6)
sns.scatterplot(data=df_g1, ax=ax, x=Xplot, y=Yplot + '_norm', marker = 'o',
                color='None', edgecolor='b', s=75, zorder=6, label='Mean values\nNPa=1')
sns.scatterplot(data=df_g2, ax=ax, x=Xplot, y=Yplot + '_norm', marker = 'o',
                color='None', edgecolor='g', s=75, zorder=6, label='Mean values\nNPa=2')
sns.scatterplot(data=df_g3, ax=ax, x=Xplot, y=Yplot + '_norm', marker = 'o',
                color='None', edgecolor='r', s=75, zorder=6, label='Mean values\nNPa>2')
ax.set_ylim([0, ax.get_ylim()[1]*1.05])
ax.set_xlabel(r'Total energy (J/cm²)')
ax.set_ylabel(r'$T_{50\%}$ - normalized')
ax.legend(title='Photo-activation\nsequence [mW/cm2]', title_fontsize=10, 
          loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1), ncols = 2, fontsize=8,)


plt.tight_layout()
plt.show()


# %%% ACF - Date by date

df = df_ACF

Id_cols = ['pos_id']
Group_cols = ('Pa')
Xplot = 'Pa_total_power'
Yplot = 't_50p'
Hplot = 'Pa_irradiance'

dates = df_ACF['date'].unique()
N_r = len(dates)

Filters = [
           (df['N_Pa'] == 1),
           ]
GlobalFilter = np.ones_like(Filters[0]).astype(bool)
for F in Filters:
    GlobalFilter = GlobalFilter & F
df = df[GlobalFilter]

fig, axes = plt.subplots(N_r, 2, figsize=(9, 2.5*N_r), sharex=True, sharey='col')

for k, date in enumerate(dates):
    df_d = df[df['date']==date]
    hue_order_date = [h for h in HUE_ORDER if h in df_d[Hplot].unique()]
    
    group = df_d.groupby(Xplot)
    agg_dict = {k:'first' for k in Id_cols}
    agg_dict.update({Yplot:'mean', Yplot + '_norm':'mean'})
    df_g = group.agg(agg_dict).reset_index()
    
    ax = axes[k, 0]
    ax.grid(zorder=0)
    ax.axhline(df_g.loc[df_g['Pa_total_power']==0, Yplot].values[0], 
               color='k', linestyle='--', linewidth=1)
    sns.scatterplot(data=df_d, ax=ax, x=Xplot, y=Yplot, 
                    hue=Hplot, hue_order = hue_order_date,
                    alpha = 0.75, zorder=6)
    sns.scatterplot(data=df_g, ax=ax, x=Xplot, y=Yplot, marker = 'o',
                    color='None', edgecolor='k', s=75, zorder=6)
    # ax.set_ylim([0, ax.get_ylim()[1]*1.05])
    ax.set_xlabel(r'Total energy (J/cm2)')
    ax.set_ylabel(f'{date}\n' + r'$T_{50\%}$ (s)')
    ax.legend().set_visible(False)
    
    ax = axes[k, 1]
    ax.grid(zorder=0)
    ax.axhline(1, color='k', linestyle='--', linewidth=1)
    sns.scatterplot(data=df_d, ax=ax, x=Xplot, y=Yplot + '_norm', 
                    hue=Hplot, hue_order = hue_order_date,
                    alpha = 0.75, zorder=6)
    sns.scatterplot(data=df_g, ax=ax, x=Xplot, y=Yplot + '_norm', marker = 'o',
                    color='None', edgecolor='k', s=75, zorder=6, label='Mean\nvalues')
    # ax.set_ylim([0, ax.get_ylim()[1]*1.05])
    ax.set_xlabel(r'Total energy (J/cm2)')
    ax.set_ylabel(r'$T_{50\%}$ - normalized')
    ax.legend(title='Photo-activation\nsequence [mW/cm2]', title_fontsize=10, 
              loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1), ncols = 2, fontsize=8,)


plt.tight_layout()
plt.show()



# %%% ACF - Only one activation - '26-03-04'

df = df_ACF

Filters = [
           (df['N_Pa'] == 1),
           (df['injection solution'].apply(lambda x : ('I2959' in x))),
           (df['date'].apply(lambda x : ('26-02-11' not in x)))
           ]
GlobalFilter = np.ones_like(Filters[0]).astype(bool)
for F in Filters:
    GlobalFilter = GlobalFilter & F
df = df[GlobalFilter]


Id_cols = ['pos_id']
Group_cols = ('Pa')
Xplot = 'Pa_total_power'
Yplot = 't_50p'
Hplot = 'Pa_irradiance'
# hue_order=['0', '200', '200_200', '400', '400_400', '800', '800_800', 
#            '1600', '1600_1600', '2400', '2400_2400', '2400_2400_2400']
# hue_order = []
# for k in range(0, 3100, 100):
#     hue_order += [f'{k:.0f}', f'{k:.0f}_{k:.0f}', f'{k:.0f}_{k:.0f}_{k:.0f}', f'{k:.0f}_{k:.0f}_{k:.0f}_{k:.0f}']
    
hue_order = [h for h in HUE_ORDER if h in df[Hplot].unique()]

group = df.groupby(Xplot)
agg_dict = {k:'first' for k in Id_cols}
agg_dict.update({Yplot:'mean', Yplot + '_norm':'mean'})
df_g = group.agg(agg_dict).reset_index()


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax = axes[0]
ax.grid(zorder=0)
ax.axhline(df_g.loc[df_g['Pa_total_power']==0, Yplot].values[0], 
           color='k', linestyle='--', linewidth=1)
sns.scatterplot(data=df, ax=ax, x=Xplot, y=Yplot, 
                hue=Hplot, hue_order = hue_order, s=70,
                alpha = 0.75, zorder=6)
sns.scatterplot(data=df_g, ax=ax, x=Xplot, y=Yplot, marker = 'o',
                color='None', edgecolor='k', s=100, zorder=6)
ax.set_ylim([0, ax.get_ylim()[1]*1.05])
ax.set_xlabel(r'Total energy (J/cm2)')
ax.set_ylabel(r'$T_{50\%}$ (s)')
ax.legend().set_visible(False)

ax = axes[1]
ax.grid(zorder=0)
ax.axhline(1, color='k', linestyle='--', linewidth=1)
sns.scatterplot(data=df, ax=ax, x=Xplot, y=Yplot + '_norm', 
                hue=Hplot, hue_order = hue_order, s=70,
                alpha = 0.75, zorder=6)
sns.scatterplot(data=df_g, ax=ax, x=Xplot, y=Yplot + '_norm', marker = 'o',
                color='None', edgecolor='k', s=100, zorder=6, label='Mean\nvalues')
ax.set_ylim([0.75, ax.get_ylim()[1]*1.05])
ax.set_xlabel(r'Total energy (J/cm2)')
ax.set_ylabel(r'$T_{50\%}$ - normalized')
ax.legend(title='Photo-activation\nsequence [mW/cm2]', 
          loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1), ncols = 2)


plt.tight_layout()
plt.show()





# %% 3.2 Magnetic Pulling

# %%% Main Dataset

mainDir = up.Path_AnalysisPulls

resTableName = 'AllResults_BeadsPulling.csv'
resTablePath = os.path.join(mainDir, resTableName)
df_Pulls = pd.read_csv(resTablePath)
df_Pulls['date'] = df_Pulls['track_id'].apply(lambda x : x.split('_')[0])
df_Pulls['Merge_id'] = df_Pulls['Manip_textid'] + '_Pa' +  df_Pulls['Pa_id'].astype(str)

PaTableName = 'MainIrradianceConditions.csv'
PaTablePath = os.path.join(mainDir, PaTableName)
df_Pa = pd.read_csv(PaTablePath)
df_Pa['Merge_id'] = df_Pa['date'] + '_' + df_Pa['manip'] + '_Pa' + df_Pa['Pa'].astype(str)
df_Pa['total energy'] = df_Pa['irradiance'] * df_Pa['duration'] / 1000
df_Pa.to_csv(PaTablePath, index=False)
df_Pa = df_Pa.drop(columns = ['manip'])

ExpCondTableName = 'MainExperimentalConditions.csv'
ExpCondTablePath = os.path.join(mainDir, ExpCondTableName)
df_ExpCond = pd.read_csv(ExpCondTablePath)

df_Pulls = pd.merge(left = df_Pulls, right = df_Pa, 
                    on = 'Merge_id', 
                    how = 'inner', suffixes=(None, '_mergeCopy'))
# df_Pulls = pd.merge(left = df_Pulls, right = df_ExpCond, 
#                     left_on = 'Manip_textid', right_on = 'id', 
#                     how = 'inner', suffixes=(None, '_mergeCopy'))
df_Pulls = df_Pulls.drop(columns=[col for col in df_Pulls.columns if ('_mergeCopy' in col)])

df_Pulls['total energy'] = df_Pulls['irradiance'] * df_Pulls['duration'] / 1000

#### Extra settings

EXCLUDED_CELLS = ['26-01-14_M1_C1']


# %%% 26-03-04 compared to previous


# %%%% Filter pulling dataset

df = df_Pulls
dates = ['26-01-14', '26-01-27', '26-02-11', '26-03-04']

df['Lc'] = 6*np.pi*df['bead radius']  
df['J_modulus'] = df['J_k'] / df['Lc']
df['J_visco1'] = df['J_gamma1'] / df['Lc']
df['J_visco2'] = df['J_gamma2'] / df['Lc']

Filters = [
           (df['N_fit_error'] == False),
           # (df['N_viscosity'] <= 40),
           (df['date'].apply(lambda x: x in dates)),
           (df['Cell_textid'].apply(lambda x : x not in EXCLUDED_CELLS)),
           ]
GlobalFilter = np.ones_like(Filters[0]).astype(bool)
for F in Filters:
    GlobalFilter = GlobalFilter & F

df_Pullsf = df[GlobalFilter]



# %%%% Show values of metrics - boxplot

df = df_Pullsf

#### Absolute values

metrics = ['N_viscosity']
metric_names = ['$\eta_N$ (Pa.s)']
metric_dict = {m:mn for (m,mn) in zip(metrics, metric_names)}

cond_col = 'total energy'
conditions = df['total energy'].unique()
conditions = np.sort(conditions)

hue_col = 'Cell_textid'

# style_col = 'date'
# style_list = ['o', 's', '^', 'X', 'P', '>']
# style_dict = {df[style_col].unique()[i]:style_list[i] for i in range(len(df[style_col].unique()))}
# df.loc[:,'marker_style'] = df[style_col].apply(lambda x : style_dict[x])

fig, axes = plt.subplots(len(metrics), 1, figsize=(7,5*len(metrics)))
axes_f = [axes]

for k in range(len(axes_f)):
    ax = axes_f[k]
    ax.set_yscale('log')
    metric = metrics[k]
    medians = [np.median(df.loc[df[cond_col]==co, metric]) for co in conditions]
    
    for i in range(len(conditions)):
        HW = 0.4
        ax.plot([i-HW, i+HW], [medians[i], medians[i]], ls='--', lw=2, c='dimgrey', zorder=8)
    
    sns.swarmplot(data = df, ax=ax, 
                  x = cond_col, y = metric, hue = hue_col, 
                  size = 7, zorder=9)

    ax.set_xlim([-0.5, len(conditions)-0.5])
    # xticks_labels = ['Ctrl'] + [f'+UV (Pa{str(k)})' for k in Energies[1:]]
    # ax.set_xticks([k for k in range(len(Energies))], xticks_labels, rotation=15)
    ax.set_xlabel('Total energy [J/cm²]')
    ax.set_ylabel(metric_dict[metric])
    yM = ax.get_ylim()[1]
    # ax.set_ylim([0, 1.25*yM])
    ax.set_ylim([0, 2*yM])
    ax.grid(axis='y')
    
    for i in range(len(conditions)):
        # ax.text(i, 1.1*yM, f'{medians[i]:.2f}', ha='center', zorder=10,
        #         size = 10, style='italic', c='dimgrey')
        ax.text(i, 1.5*yM, f'{medians[i]:.2f}', ha='center', zorder=10,
                size = 10, style='italic', c='dimgrey')
    
    if k == 1:
        ax.legend(title='Cell IDs',
                  loc="upper left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
    else:
        ax.legend().set_visible(False)

fig.suptitle('All pulls', fontsize=16)
fig.tight_layout()

#### Normalized

id_cols = ['track_id', 'Manip_textid', 'date', 'irradiance', 'total energy']
group = df.groupby(['Cell_textid', 'Pa_id'])
agg_dict = {k:'first' for k in id_cols}
agg_dict.update({m:'median' for m in metrics})
df_g = group.agg(agg_dict).reset_index()
for m in metrics:
    m_N = m + '_norm'
    df.loc[:,m_N] = df[m]
    df_g.loc[:,m_N] = df_g.loc[:,m]
    for k, cid in enumerate(df['Cell_textid'].unique()):
        index_cell_control = df_g[(df_g['Cell_textid'] == cid) & (df_g['Pa_id'] == 0)].index
        val_ctrl = df_g.loc[index_cell_control, Yplot].values[0]
        
        index_cell = df[df['Cell_textid'] == cid].index
        df.loc[index_cell, m_N] /= val_ctrl
        index_cell = df_g[df_g['Cell_textid'] == cid].index
        df_g.loc[index_cell, m_N] /= val_ctrl

metrics = [m + '_norm' for m in metrics]
metric_names = [mn.split(' ')[0] + ' - normalized' for mn in metric_names]
metric_dict = {m:mn for (m,mn) in zip(metrics, metric_names)}

fig, axes = plt.subplots(len(metrics), 1, figsize=(7,5*len(metrics)))
axes_f = [axes]

for k in range(len(axes_f)):
    ax = axes_f[k]
    ax.set_yscale('log')
    metric = metrics[k]
    medians = [np.median(df.loc[df[cond_col]==co, metric]) for co in conditions]
    
    for i in range(len(conditions)):
        HW = 0.4
        ax.plot([i-HW, i+HW], [medians[i], medians[i]], ls='--', lw=2, c='dimgrey', zorder=8)
    
    sns.swarmplot(data = df, ax=ax, 
                  x = cond_col, y = metric, hue = hue_col,
                  size = 7, zorder=9)

    ax.set_xlim([-0.5, len(conditions)-0.5])
    # xticks_labels = ['Ctrl'] + [f'+UV (Pa{str(k)})' for k in Energies[1:]]
    # ax.set_xticks([k for k in range(len(Energies))], xticks_labels, rotation=15)
    ax.set_xlabel('Total energy [J/cm²]')
    ax.set_ylabel(metric_dict[metric])
    yM = ax.get_ylim()[1]
    # ax.set_ylim([0, 1.25*yM])
    ax.set_ylim([0, 2*yM])
    ax.grid(axis='y')
    
    for i in range(len(conditions)):
        # ax.text(i, 1.1*yM, f'{medians[i]:.2f}', ha='center', zorder=10,
        #         size = 10, style='italic', c='dimgrey')
        ax.text(i, 1.5*yM, f'{medians[i]:.2f}', ha='center', zorder=10,
                size = 10, style='italic', c='dimgrey')
    
    if k == 1:
        ax.legend(title='Cell IDs',
                  loc="upper left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
    else:
        ax.legend().set_visible(False)

fig.suptitle('All pulls - normalized', fontsize=16)
fig.tight_layout()


plt.show()

# %%%% Show values of one metric - Scatterplot version

df = df_Pullsf

Xplot = 'total energy'
Yplot = 'N_viscosity'
Hplot = 'Cell_textid'

# hue_order = [h for h in HUE_ORDER if h in df[Hplot].unique()]

style_col = 'date'
style_list = ['o', 's', '^', 'X', 'P', '>']
style_dict = {df[style_col].unique()[i]:style_list[i] for i in range(len(df[style_col].unique()))}
df.loc[:,'marker_style'] = df[style_col].apply(lambda x : style_dict[x])

id_cols = ['track_id', 'Manip_textid', 'date', 'irradiance', 'total energy']
group = df.groupby(['Cell_textid', 'Pa_id'])
agg_dict = {k:'first' for k in id_cols}
agg_dict.update({Yplot:'median'})
df_g = group.agg(agg_dict).reset_index()

Yplot_N = Yplot + '_norm'
df[Yplot_N] = df[Yplot]
df_g[Yplot_N] = df_g[Yplot]
for k, cid in enumerate(df['Cell_textid'].unique()):
    index_cell_control = df_g[(df_g['Cell_textid'] == cid) & (df_g['Pa_id'] == 0)].index
    val_ctrl = df_g.loc[index_cell_control, Yplot].values[0]
    
    index_cell = df[df['Cell_textid'] == cid].index
    df.loc[index_cell, Yplot_N] /= val_ctrl
    index_cell = df_g[df_g['Cell_textid'] == cid].index
    df_g.loc[index_cell, Yplot_N] /= val_ctrl


fig, axes = plt.subplots(1, 2, figsize=(10, 6))
ax = axes[0]
ax.set_yscale('log')
ax.grid(zorder=0)
# ax.axhline(df_g.loc[df_g['irradiance']==0, Yplot].values[0], 
#            color='k', linestyle='--', linewidth=1)
sns.scatterplot(data=df, ax=ax, x=Xplot, y=Yplot, s=50,
                hue=Hplot, # markers = 'marker_style', #hue_order = hue_order,
                alpha = 0.75, zorder=6)
# sns.scatterplot(data=df_g, ax=ax, x=Xplot, y=Yplot, marker = 'o',
#                 color='None', edgecolor='k', s=75, zorder=6)
# ax.set_ylim([0, ax.get_ylim()[1]*1.05])
ax.set_xlabel(r'Total energy (J/cm2)')
ax.set_ylabel(r'$\eta_N$ (Pa.s)')
ax.legend(fontsize=7, ncols = 2) #.set_visible(False)

ax = axes[1]
ax.set_yscale('log')
ax.grid(zorder=0)
# ax.axhline(1, color='k', linestyle='--', linewidth=1)
# sns.scatterplot(data=df, ax=ax, x=Xplot, y=Yplot + '_norm', 
#                 hue=Hplot, hue_order = hue_order,
#                 alpha = 0.75, zorder=6)
sns.scatterplot(data=df_g, ax=ax, x=Xplot, y=Yplot + '_norm', marker = 'o', hue='Manip_textid', 
                alpha = 0.5, edgecolor='k', s=75, zorder=6)
ax.set_ylim([0, ax.get_ylim()[1]*1.05])
ax.set_xlabel(r'Total energy (J/cm2)')
ax.set_ylabel(r'Median $\eta_N$ - normalized')
ax.legend(fontsize=9)
# ax.legend(title='Photo-activation\nsequence [mW/cm2]', title_fontsize=10,
          # loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1), ncols = 2, fontsize=9)
          
plt.tight_layout()
plt.show()




# %%% 26-03-04

# %%%% Filter pulling dataset


df = df_Pulls

df['Lc'] = 6*np.pi*df['bead radius']  
df['J_modulus'] = df['J_k'] / df['Lc']
df['J_visco1'] = df['J_gamma1'] / df['Lc']
df['J_visco2'] = df['J_gamma2'] / df['Lc']

Filters = [
           (df['N_fit_error'] == False),
           (df['date'] == '26-03-04'),
           (df['Cell_textid'].apply(lambda x : x not in EXCLUDED_CELLS)),
           ]
GlobalFilter = np.ones_like(Filters[0]).astype(bool)
for F in Filters:
    GlobalFilter = GlobalFilter & F

df_Pullsf = df[GlobalFilter]




# %%%% Show values of metrics - boxplot

df = df_Pullsf

#### Absolute values

metrics = ['N_viscosity']
metric_names = ['$\eta_N$ (Pa.s)']
metric_dict = {m:mn for (m,mn) in zip(metrics, metric_names)}

cond_col = 'total energy'
conditions = df['total energy'].unique()
conditions = np.sort(conditions)

hue_col = 'Cell_textid'

fig, axes = plt.subplots(len(metrics), 1, figsize=(7,5*len(metrics)))
axes_f = [axes]

for k in range(len(axes_f)):
    ax = axes_f[k]
    metric = metrics[k]
    medians = [np.median(df.loc[df[cond_col]==co, metric]) for co in conditions]
    
    for i in range(len(conditions)):
        HW = 0.4
        ax.plot([i-HW, i+HW], [medians[i], medians[i]], ls='--', lw=2, c='dimgrey', zorder=8)
    
    sns.swarmplot(data = df, ax=ax, 
                  x = cond_col, y = metric, hue = hue_col,
                  size = 9, zorder=9)

    ax.set_xlim([-0.5, len(conditions)-0.5])
    # xticks_labels = ['Ctrl'] + [f'+UV (Pa{str(k)})' for k in Energies[1:]]
    # ax.set_xticks([k for k in range(len(Energies))], xticks_labels, rotation=15)
    ax.set_xlabel('Total energy [J/cm²]')
    ax.set_ylabel(metric_dict[metric])
    yM = ax.get_ylim()[1]
    ax.set_ylim([0, 1.25*yM])
    ax.grid(axis='y')
    
    for i in range(len(conditions)):
        ax.text(i, 1.1*yM, f'{medians[i]:.2f}', ha='center', zorder=10,
                size = 10, style='italic', c='dimgrey')
    
    if k == 1:
        ax.legend(title='Cell IDs',
                  loc="upper left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
    else:
        ax.legend().set_visible(False)

fig.suptitle('26-03-04 - all pulls', fontsize=16)
fig.tight_layout()

#### Normalized

id_cols = ['track_id', 'Manip_textid', 'date', 'irradiance', 'total energy']
group = df.groupby(['Cell_textid', 'Pa_id'])
agg_dict = {k:'first' for k in id_cols}
agg_dict.update({m:'median' for m in metrics})
df_g = group.agg(agg_dict).reset_index()
for m in metrics:
    m_N = m + '_norm'
    df.loc[:,m_N] = df.loc[:,m]
    df_g.loc[:,m_N] = df_g.loc[:,m]
    for k, cid in enumerate(df['Cell_textid'].unique()):
        index_cell_control = df_g[(df_g['Cell_textid'] == cid) & (df_g['Pa_id'] == 0)].index
        val_ctrl = df_g.loc[index_cell_control, Yplot].values[0]
        
        index_cell = df[df['Cell_textid'] == cid].index
        df.loc[index_cell, m_N] /= val_ctrl
        index_cell = df_g[df_g['Cell_textid'] == cid].index
        df_g.loc[index_cell, m_N] /= val_ctrl

metrics = [m + '_norm' for m in metrics]
metric_names = [mn.split(' ')[0] + ' - normalized' for mn in metric_names]
metric_dict = {m:mn for (m,mn) in zip(metrics, metric_names)}

fig, axes = plt.subplots(len(metrics), 1, figsize=(7,5*len(metrics)))
axes_f = [axes]

for k in range(len(axes_f)):
    ax = axes_f[k]
    metric = metrics[k]
    medians = [np.median(df.loc[df[cond_col]==co, metric]) for co in conditions]
    
    for i in range(len(conditions)):
        HW = 0.4
        ax.plot([i-HW, i+HW], [medians[i], medians[i]], ls='--', lw=2, c='dimgrey', zorder=8)
    
    sns.swarmplot(data = df, ax=ax, 
                  x = cond_col, y = metric, hue = hue_col,
                  size = 9, zorder=9)

    ax.set_xlim([-0.5, len(conditions)-0.5])
    # xticks_labels = ['Ctrl'] + [f'+UV (Pa{str(k)})' for k in Energies[1:]]
    # ax.set_xticks([k for k in range(len(Energies))], xticks_labels, rotation=15)
    ax.set_xlabel('Total energy [J/cm²]')
    ax.set_ylabel(metric_dict[metric])
    yM = ax.get_ylim()[1]
    ax.set_ylim([0, 1.25*yM])
    ax.grid(axis='y')
    
    for i in range(len(conditions)):
        ax.text(i, 1.1*yM, f'{medians[i]:.2f}', ha='center', zorder=10,
                size = 10, style='italic', c='dimgrey')
    
    if k == 1:
        ax.legend(title='Cell IDs',
                  loc="upper left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
    else:
        ax.legend().set_visible(False)

fig.suptitle('26-03-04 - all pulls - normalized', fontsize=16)
fig.tight_layout()


plt.show()

# %%%% Show values of one metric - Scatterplot version

df = df_Pullsf

Xplot = 'total energy'
Yplot = 'N_viscosity'
Hplot = 'Cell_textid'

# hue_order = [h for h in HUE_ORDER if h in df[Hplot].unique()]

id_cols = ['track_id', 'Manip_textid', 'date', 'irradiance', 'total energy']
group = df.groupby(['Cell_textid', 'Pa_id'])
agg_dict = {k:'first' for k in id_cols}
agg_dict.update({Yplot:'median'})
df_g = group.agg(agg_dict).reset_index()

Yplot_N = Yplot + '_norm'
df[Yplot_N] = df[Yplot]
df_g[Yplot_N] = df_g[Yplot]
for k, cid in enumerate(df['Cell_textid'].unique()):
    index_cell_control = df_g[(df_g['Cell_textid'] == cid) & (df_g['Pa_id'] == 0)].index
    val_ctrl = df_g.loc[index_cell_control, Yplot].values[0]
    
    index_cell = df[df['Cell_textid'] == cid].index
    df.loc[index_cell, Yplot_N] /= val_ctrl
    index_cell = df_g[df_g['Cell_textid'] == cid].index
    df_g.loc[index_cell, Yplot_N] /= val_ctrl


fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
ax.grid(zorder=0)
# ax.axhline(df_g.loc[df_g['irradiance']==0, Yplot].values[0], 
#            color='k', linestyle='--', linewidth=1)
sns.scatterplot(data=df, ax=ax, x=Xplot, y=Yplot, 
                hue=Hplot, s=70, #hue_order = hue_order,
                alpha = 0.75, zorder=8)
# sns.scatterplot(data=df_g, ax=ax, x=Xplot, y=Yplot, marker = 'P',
#                 hue=Hplot, edgecolor='k', s=100, zorder=6)
# ax.set_ylim([0, ax.get_ylim()[1]*1.05])
ax.set_xlabel(r'Total energy (J/cm2)')
ax.set_ylabel(r'$\eta_N$ (Pa.s)')
ax.legend(fontsize=6) #.set_visible(False)

ax = axes[1]
ax.grid(zorder=0)
# ax.axhline(1, color='k', linestyle='--', linewidth=1)
# sns.scatterplot(data=df, ax=ax, x=Xplot, y=Yplot + '_norm', 
#                 hue=Hplot, hue_order = hue_order,
#                 alpha = 0.75, zorder=6)
sns.scatterplot(data=df_g, ax=ax, x=Xplot, y=Yplot + '_norm', marker = 'o', hue='Manip_textid', 
                alpha = 0.5, edgecolor='k', s=100, zorder=6)
ax.set_ylim([0, ax.get_ylim()[1]*1.05])
ax.set_xlabel(r'Total energy (J/cm2)')
ax.set_ylabel(r'Median $\eta_N$ - normalized')
ax.legend(fontsize=9)
# ax.legend(title='Photo-activation\nsequence [mW/cm2]', title_fontsize=10,
          # loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1), ncols = 2, fontsize=9)
          
plt.tight_layout()
plt.show()

# %%%% Add ACF dataset

SCALE = SCALE_40X
FPS = 1

dirPath = up.Path_AnalysisPulls

df_ACF = pd.read_csv(os.path.join(dirPath, 'AllResults_ACF.csv'))
df_ExpCond = pd.read_csv(os.path.join(dirPath, 'MainExperimentalConditions.csv'))
df_ACF['N_Pa'] = df_ACF['Pa_dt'].apply(lambda x : len(str(x).split('_')))
df_ACF['manip_id'] = df_ACF['id'].apply(lambda x : '_'.join(str(x).split('_')[:2]))

# if 'long_cell_id' not in df_ACF.columns:
#     get_long_cell_id = lambda x : '_'.join(x.split('_')[:4])# + [x.split('_')[-1]])
#     df_ACF['long_cell_id'] = df_ACF['id'].apply(get_long_cell_id)
#     # df_ACF.to_csv(os.path.join(dirPath, 'Results', 'results_ACF.csv'), index=False)

metrics_cols = [col for col in df_ACF.columns if col.startswith('t_')]
for mc in metrics_cols:
    mcn = mc + '_norm'
    df_ACF[mcn] = df_ACF[mc]
for k, cid in enumerate(df_ACF['long_cell_id'].unique()):
    index_cell = df_ACF[df_ACF['long_cell_id'] == cid].index
    index_cell_control = df_ACF[(df_ACF['long_cell_id'] == cid) & (df_ACF['Pa'] == 0)].index
    for mc in metrics_cols:
        mcn = mc + '_norm'
        val_ctrl = df_ACF.loc[index_cell_control, mc].values[0]
        df_ACF.loc[index_cell, mcn] /= val_ctrl
        
df_ACF = pd.merge(left = df_ACF, right = df_ExpCond, 
                  left_on = 'manip_id', right_on = 'id', how = 'inner',
                  suffixes=(None, '_copy'))

for col in df_ACF.columns:
    if ('Unnamed' in col) or ('_copy' in col):
        df_ACF = df_ACF.drop(labels = col, axis = 1)
        
HUE_ORDER = []
for k in range(0, 3100, 100):
    HUE_ORDER += [
                  f'{k:.0f}', f'{k:.0f}_{k:.0f}', 
                  f'{k:.0f}_{k:.0f}_{k:.0f}', 
                  f'{k:.0f}_{k:.0f}_{k:.0f}_{k:.0f}',
                  ]

Filters = [
           (df_ACF['date'] == '26-03-04'),
           ]
GlobalFilter = np.ones_like(Filters[0]).astype(bool)
for F in Filters:
    GlobalFilter = GlobalFilter & F

df_ACFf = df_ACF[GlobalFilter]

# %%%% ACF - Only one activation - '26-03-04'

df = df_ACFf

Filters = [
           (df['N_Pa'] == 1),
           (df['injection solution'].apply(lambda x : ('I2959' in x))),
           (df['date'].apply(lambda x : ('26-02-11' not in x)))
           ]
GlobalFilter = np.ones_like(Filters[0]).astype(bool)
for F in Filters:
    GlobalFilter = GlobalFilter & F
df = df[GlobalFilter]


Id_cols = ['pos_id']
Group_cols = ('Pa')
Xplot = 'Pa_total_power'
Yplot = 't_50p'
Hplot = 'Pa_irradiance'
# hue_order=['0', '200', '200_200', '400', '400_400', '800', '800_800', 
#            '1600', '1600_1600', '2400', '2400_2400', '2400_2400_2400']
# hue_order = []
# for k in range(0, 3100, 100):
#     hue_order += [f'{k:.0f}', f'{k:.0f}_{k:.0f}', f'{k:.0f}_{k:.0f}_{k:.0f}', f'{k:.0f}_{k:.0f}_{k:.0f}_{k:.0f}']
    
hue_order = [h for h in HUE_ORDER if h in df[Hplot].unique()]

group = df.groupby(Xplot)
agg_dict = {k:'first' for k in Id_cols}
agg_dict.update({Yplot:'median', Yplot + '_norm':'median'})
df_g = group.agg(agg_dict).reset_index()


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax = axes[0]
ax.grid(zorder=0)
ax.axhline(df_g.loc[df_g['Pa_total_power']==0, Yplot].values[0], 
           color='k', linestyle='--', linewidth=1)
sns.scatterplot(data=df, ax=ax, x=Xplot, y=Yplot, 
                hue=Hplot, hue_order = hue_order, s=70,
                alpha = 0.65, zorder=6)
sns.scatterplot(data=df_g, ax=ax, x=Xplot, y=Yplot, marker = 'o',
                color='None', edgecolor='k', s=100, zorder=6)
ax.set_ylim([0, ax.get_ylim()[1]*1.05])
ax.set_xlabel(r'Total energy (J/cm2)')
ax.set_ylabel(r'$T_{50\%}$ (s)')
ax.legend().set_visible(False)

ax = axes[1]
ax.grid(zorder=0)
ax.axhline(1, color='k', linestyle='--', linewidth=1)
sns.scatterplot(data=df, ax=ax, x=Xplot, y=Yplot + '_norm', 
                hue=Hplot, hue_order = hue_order, s=70,
                alpha = 0.65, zorder=6)
sns.scatterplot(data=df_g, ax=ax, x=Xplot, y=Yplot + '_norm', marker = 'o',
                color='None', edgecolor='k', s=100, zorder=6, label='Mean\nvalues')
ax.set_ylim([0.75, ax.get_ylim()[1]*1.05])
ax.set_xlabel(r'Total energy (J/cm2)')
ax.set_ylabel(r'$T_{50\%}$ - normalized')
ax.legend(title='Photo-activation\nsequence [mW/cm2]', 
          loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1), ncols = 2)


plt.tight_layout()
plt.show()



# %%% 26-02-11

mainDir = up.Path_AnalysisPulls
fileName = 'AllResults_BeadsPulling.csv'
filePath = os.path.join(mainDir, fileName)

specif = '_26-02-11'
EXCLUDED_CELLS = ['26-01-14_M1_C1']

df = pd.read_csv(filePath)

df['date'] = df['track_id'].apply(lambda x : x.split('_')[0])
df['Lc'] = 6*np.pi*df['bead radius']
df['J_modulus'] = df['J_k'] / df['Lc']
df['J_visco1'] = df['J_gamma1'] / df['Lc']
df['J_visco2'] = df['J_gamma2'] / df['Lc']

Filters = [
           (df['J_gamma2'] < 500),
           (df['J_fit_error'] == False),
           (df['J_R2'] >= 0.7),
           (df['date'] == '26-02-11'),
           (df['Cell_textid'].apply(lambda x : x not in EXCLUDED_CELLS)),
           ]
GlobalFilter = np.ones_like(Filters[0]).astype(bool)
for F in Filters:
    GlobalFilter = GlobalFilter & F

df = df[GlobalFilter]
savePath = mainDir

OneBead_Cell_textId = '26-02-11_M1_C1'

fig, axes = PlotPullMetrics_swarm(df, group_by_cell = 'False', agg_func = 'mean')
# dfg = PlotPullMetrics_bipart(df, agg_func = 'mean', normalized = False)
# fig, axes = PlotPullMetrics_bipart(df, agg_func = 'mean', normalized = False)
# fig, axes = PlotPullMetrics_oneBead(df, OneBead_Cell_textId, BeadNo=1, iUV=3)

# %%% 26-01-27

mainDir = up.Path_AnalysisPulls
fileName = 'AllResults_BeadsPulling.csv'
filePath = os.path.join(mainDir, fileName)

specif = '_26-01-27_NoI2959'
EXCLUDED_CELLS = ['26-01-14_M1_C1']

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
           (df['Cell_textId'].apply(lambda x : x not in EXCLUDED_CELLS)),
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
EXCLUDED_CELLS = ['26-01-14_M1_C1']

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
           (df['Cell_textId'].apply(lambda x : x not in EXCLUDED_CELLS)),
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
    
