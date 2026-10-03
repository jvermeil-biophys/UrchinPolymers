# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:33:46 2026

@author: Joseph Vermeil

UtilityFunctions.py - contains all kind of small functions used by CortExplore programs, 
to be imported with "import UtilityFunctions as ufun" and call with "ufun.my_function".
Joseph Vermeil, 2026

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# %% Imports

import os
import cv2
import time
import alphashape

import numpy as np
import pandas as pd
import trackpy as tp
import skimage as skm
import seaborn as sns
import matplotlib as mpl
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import make_splrep
from scipy.ndimage import gaussian_filter1d

from shapely.geometry import MultiPoint, MultiPolygon, Polygon

import Libs.PlotMaker as pm
import Libs.UrchinPaths as up
import Libs.CalibrationData as cd
import Libs.UtilityFunctions as ufun
import Libs.ToolboxCytoplasmAnalysis as tbca
import Libs.ToolboxStructureAnalysis as tbsa



# %% Utility functions

def Tpf_str2num(tpf_str):
    L = tpf_str.split('min')
    tpf_num = int(L[0])*60
    if len(L) > 1 and len(L[1]) > 0:
        tpf_num += int(L[1])
    return(tpf_num)


def tri_to_short_edges(tri, points, thresh_d):
    # Extract all edges from each triangle
    edges = np.vstack([
        tri.simplices[:, [0, 1]],
        tri.simplices[:, [1, 2]],
        tri.simplices[:, [2, 0]]
    ])

    # Sort indices within each edge to make (i, j) and (j, i) identical
    edges = np.sort(edges, axis=1)

    # Remove duplicate edges
    edges = np.unique(edges, axis=0)

    # Convert to a Python list of pairs
    edges = np.array(edges)
    
    pairs = points[edges]
    dists = np.power(np.sum((pairs[:,1,:]-pairs[:,0,:])**2, axis=1), 0.5)
    idx_close_neighbours = (dists < thresh_d)
    edges_close_neighbours = edges[idx_close_neighbours]
    return(edges_close_neighbours, dists)


def distribute_in_boxes(df, L, M, 
                        str_Id = 'Id', str_X = 'X', str_Y = 'Y'):
    box_size = L / M
    
    # Compute box indices
    cols = np.floor(df[str_X] / box_size).astype(int)
    rows = np.floor(df[str_Y] / box_size).astype(int)
    
    # Handle points exactly at X=N or Y=N
    cols = np.minimum(cols, M - 1)
    rows = np.minimum(rows, M - 1)
    
    # Group IDs by box
    result = (
        df.assign(row=rows, col=cols)
          .groupby(["row", "col"])[str_Id]
          .apply(list)
          .to_dict()
    )    
    return(result)


def df_grid_2_matrix(df_grid, Nx, Ny, xy_col = 'Bxy', parm_col = 'k_full'):
    V = df_grid[xy_col].values
    L = [[*tup] for tup in V]
    XY_g = np.array(L)
    df_grid['X_g'] = XY_g[:, 0]
    df_grid['Y_g'] = XY_g[:, 1]
    if max(XY_g[:, 0]) > (Nx-1):
        Nx = max(XY_g[:, 0])+1
    if max(XY_g[:, 1]) > (Ny-1):
        Ny = max(XY_g[:, 1])+1
        
    Hm = np.zeros((Nx, Ny))
    Hm.fill(np.nan)
    for xy in XY_g:
        [x, y] = xy
        Hm[y, x] = df_grid.loc[(df_grid['X_g'] == x) & (df_grid['Y_g'] == y), parm_col].values[0]
        
    return(Hm)


def MSD_HeatMap(df_grid, M_boxes, xy_col, parm_col,
                axtitle = '', cbarlabel = '',
                cmap='viridis', norm_type = 'lin',
                c_vmin=None, c_vmax=None,
                fig=None, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        
    parm = parm_col
    if axtitle=='':
        axtitle=parm
    ax.set_title(axtitle)
    
    HeatMap = df_grid_2_matrix(df_grid_MSD, M_boxes, M_boxes, 
                               xy_col = xy_col, 
                               parm_col = parm_col)

    # mpl.colors.Normalize() # mpl.colors.LogNorm()
    # norm = mpl.colors.Normalize()
    if norm_type == 'lin':
        norm = mpl.colors.Normalize(vmin=c_vmin, vmax=c_vmax)
    elif norm_type == 'log':
        norm = mpl.colors.LogNorm(vmin=c_vmin, vmax=c_vmax)
    else:
        norm = None
        
    axim = ax.imshow(HeatMap, aspect='equal',
                     cmap=cmap, norm=norm, )
    ax.invert_yaxis()

    # Create colorbar
    cbar = fig.colorbar(axim, ax=ax, label=cbarlabel)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")#, va="bottom")

    xt = np.linspace(0, M_boxes, 3, endpoint=True)
    yt = np.linspace(0, M_boxes, 3, endpoint=True)
    xticks = xt - 0.5
    yticks = yt - 0.5
    xlabels = [f'{x*(N_pix)/M_boxes:.0f}' for x in xt]
    ylabels = [f'{y*(N_pix)/M_boxes:.0f}' for y in yt]
    xlabels = ['' for x in xt]
    ylabels = ['' for y in yt]
    ax.tick_params(axis='both', length=0,)

    ax.set_xticks(xticks, labels=xlabels,) # rotation=-30, rotation_mode="xtick")
    ax.set_yticks(yticks, labels=ylabels)
    # ax.spines[:].set_visible(False)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
        
    return(fig, ax)



def df_2_heatmap(df, ax, parmCol='k', boxCol='Bxy', y_ascending=False,
                 cmap="viridis", annotate=False, colorScale='linear'):
    """
    Plot k values on an M x M grid.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns parmCol and BoxCol.
        BoxCol contains tuples (x, y).
    M : int
        Grid size.
    cmap : str
        Matplotlib colormap name.
    annotate : bool
        Whether to display k values in cells.
    """
    
    M = np.max(np.array(df[boxCol].tolist())) + 1
    
    # Extract coordinates
    coords = pd.DataFrame(
        df[boxCol].tolist(),
        columns=["x", "y"],
        index=df.index
    )

    data = pd.concat(
        [coords, df[parmCol]],
        axis=1
    )

    # Convert to matrix
    matrix = data.pivot(
        index="y",
        columns="x",
        values=parmCol
    )

    # Ensure all M x M boxes are represented
    matrix = matrix.reindex(
        index=range(M),
        columns=range(M)
    )

    # Plot
    if colorScale == 'linear':
        sns.heatmap(
            matrix,
            cmap=cmap,
            square=True,
            annot=annotate,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": parmCol,
                      "shrink": .5},
            ax=ax
        )
    elif colorScale == 'log':
        sns.heatmap(
            matrix,
            cmap=cmap,
            square=True,
            annot=annotate,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": parmCol,
                      "shrink": .5},
            ax=ax,
            norm=mpl.colors.LogNorm(
                vmin=matrix.min().min(),
                vmax=matrix.max().max()
                )        
        )
    
    if y_ascending:
        ax.invert_yaxis()

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Heatmap of {parmCol}")

    return(ax)



def autocorr_fft(x):
    N = len(x)
    F = np.fft.fft(x, n = 2*N)  # 2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real  # now we have the autocorrelation in convention B
    n = N * np.ones(N) - np.arange(0, N) # divide res(m) by (N-m)
    return(res / n)  # this is the autocorrelation in convention A



def msd_fft_trackpyStyle(traj, mpp, fps, max_lagtime=100, pos_columns=['x', 'y']):
    """
    https://github.com/hadim/Public-Notebooks/blob/master/Code/Quick_MSD/notebook.ipynb
    """
    
    r = traj[pos_columns].values
    r *= mpp

    t = traj['frame']

    max_lagtime = min(max_lagtime, len(t))  # checking to be safe
    lagtimes = 1 + np.arange(max_lagtime - 1)    

    N = len(r)

    D = np.square(r).sum(axis=1) 
    D = np.append(D, 0)
    S2 = sum([autocorr_fft(r[:, i]) for i in range(len(pos_columns))])

    Q = 2 * D.sum()
    S1 = np.zeros(max_lagtime)

    for m in range(max_lagtime):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)

    msd = S1 - 2 * S2[:max_lagtime]
    msd = msd[1:]

    lagt = lagtimes / fps

    results = pd.DataFrame(np.array([msd, lagt]).T, columns=['msd', 'lagt'])
    results.index = 1 + np.arange(max_lagtime - 1)
    results.index.name = 'lagt'
    
    return(results)



def msd_fft_1D(pos, mpp, fps, max_lagtime=100):
    """
    https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft/34222273#34222273
    """
    
    r = pos
    r *= mpp

    t = np.arange(len(pos))

    max_lagtime = min(max_lagtime, len(t))  # checking to be safe
    lagtimes = 1 + np.arange(max_lagtime)    

    N = len(r)

    D = np.square(r)
    D = np.append(D, 0)
    S2 = sum([autocorr_fft(r[:])])

    Q = 2 * D.sum()
    S1 = np.zeros(max_lagtime+1)

    for m in range(max_lagtime+1):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)

    msd = S1 - 2 * S2[:max_lagtime+1]
    msd = msd[1:]

    lagt = lagtimes / fps

    results = pd.DataFrame(np.array([msd, lagt]).T, columns=['msd', 'lagt'])
    results.index = 1 + np.arange(max_lagtime)
    results.index.name = 'lagt'
    
    return(results)




# %% 1. Tracking and MSD

# %%%% Settings

#### Paths

mainDir = os.path.join(up.Path_IntraCellTracking, '26-07-29_FastAcq_Fec_NB-Yolk')
srcDir = os.path.join(mainDir, 'Crops')
dstDir = os.path.join(mainDir, 'SPT_results')

tifNames = ['26-07-29_PostF_2min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_6min30_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_12min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_20min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_30min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_45min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_60min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_70min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            ]

tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
fileNames = [fN.split('.')[0] for fN in tifNames]

rawTracksDir = 'TrackMate_raw_tracks'
rawTracksNames = [fN + '_TmTracks.xml' for fN in tifPaths]
rawTracksPaths = [os.path.join(dstDir, rawTracksDir, rtN)  for rtN in rawTracksNames]

cleanTracksDir = 'Clean_tracks'
cleanTracksNames = [fN + '_PyTracks.csv' for fN in tifPaths]
cleanTracksPaths = [os.path.join(dstDir, cleanTracksDir, ctN)  for ctN in cleanTracksNames]


#### Settings

UmPerPix = cd.UmPerPix_60X_W1
SCALE = 1/UmPerPix
nbimages = 2000
FPS = 10

N_pix = 512
C_pix = np.median(np.arange(N_pix)) # Center (pixels)
L_um = N_pix*UmPerPix

max_lagtime = 50
lowDt_upper = 0.5
highDt_lower = 1.0





# %%%% Run Trackmate

for tifPath, rawTrackName in zip(tifPaths, rawTracksNames):
    tbca.pretreatAndTrack_CropedYolk(tifPath, rawTrackName, dstDir,
                                     PLOT = True, SAVEPLOT = True)


# %%%% Import & format tracks

for ii in range(len(rawTracksPaths)):
    xmlPath = rawTracksPaths[ii]
    dfName = cleanTracksNames[ii]
    
    Tracks = tbca.importTrackMateTracks(xmlPath)
    
    Np = len(Tracks)
        
    column_names = ['frame', 'x', 'y', 'particle']
    all_tracks = []
    for i, track in enumerate(Tracks):
        nT = len(track)
        # test_x_sat = ((np.max(track[:, 1]) - np.min(track[:, 1])) < 1)
        # test_y_sat = ((np.max(track[:, 2]) - np.min(track[:, 2])) < 1)
        test_x_sat = ((np.max(track[:, 1]) == (N_pix-1)) or (np.min(track[:, 1]) == 0))
        test_y_sat = ((np.max(track[:, 2]) == (N_pix-1)) or (np.min(track[:, 2]) == 0))
        if (not test_x_sat) and (not test_y_sat) and (nT >= 30):
            track = np.concat((track, np.ones((len(track[:,0]), 1), dtype=int) * (i+1)), axis = 1)
            track[:, 0] = track[:, 0].astype(int) + 1
            all_tracks.append(track)
    
    concat_tracks = np.concat(all_tracks, axis = 0)
    df = pd.DataFrame({column_names[k] : concat_tracks[:,k] for k in range(len(column_names))})
    df.to_csv(os.path.join(dstDir, dfName), index=False, sep = '\t')
    

# %%%% Import tracks & run msd



for ii in range(len(dfNames)):
    dfName = dfNames[ii]
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    jsonName = jsonNames[ii]
    msdName = msdNames[ii]
    
    res_emsd = tp.motion.emsd(df, UmPerPix, FPS, max_lagtime=max_lagtime).reset_index()
    res_emsd.to_csv(os.path.join(dstDir, msdName), index=False, sep='\t')
    
    T, MSD = res_emsd['lagt'].values, res_emsd['msd'].values
    iLow = ufun.findFirst(lowDt_upper, T) + 1
    iHigh = ufun.findFirst(highDt_lower, T)
    
    parms, results = ufun.fitLineHuber(T, MSD, with_intercept = False)
    D_linear = parms[0]/4
    
    parms, results = ufun.fitLineHuber(np.log(T), np.log(MSD), with_intercept = True)
    b, a = parms
    k_full = a
    D_full = np.exp(b)/4

    parms, results = ufun.fitLineHuber(np.log(T[:iLow]), np.log(MSD[:iLow]), with_intercept = True)
    b, a = parms
    k_lowDt = a
    D_lowDt = np.exp(b)/4

    parms, results = ufun.fitLineHuber(np.log(T[iHigh:]), np.log(MSD[iHigh:]), with_intercept = True)
    b, a = parms
    k_highDt = a
    D_highDt = np.exp(b)/4
    
    dict_MSDfits = {'D_linear': D_linear,
                    'k_full': k_full,
                    'D_full': D_full,
                    'k_lowDt': k_lowDt,
                    'D_lowDt': D_lowDt,
                    'k_highDt': k_highDt,
                    'D_highDt': D_highDt,}
    
    ufun.dict2json(dict_MSDfits, dstDir, jsonName)
    
# %%%% Import MSD & plot

# for ii in range(len(msdNames)):
#     df = pd.read_csv(os.path.join(dstDir, dfNames[ii]), sep='\t')
#     res_emsd = pd.read_csv(os.path.join(dstDir, msdNames[ii]), sep='\t')
#     T, MSD = res_emsd['lagt'], res_emsd['msd']
    
#     dict_MSDfits = ufun.json2dict(dstDir, jsonNames[ii])
#     D_linear = dict_MSDfits['D_linear']
#     k_full = dict_MSDfits['k_full']
#     D_full = dict_MSDfits['D_full']
#     k_lowDt = dict_MSDfits['k_lowDt']
#     D_lowDt = dict_MSDfits['D_lowDt']
#     k_highDt = dict_MSDfits['k_highDt']
#     D_highDt = dict_MSDfits['D_highDt']
    
#     Tc = (D_lowDt/D_highDt)**(1/(k_highDt-k_lowDt))
    
#     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#     ax.set_xscale('log')
#     ax.set_yscale('log')

#     Xp = np.array([1e-2, 1e2])

#     ax.plot(T, MSD, 'wo', mec='k')
#     # ax.plot(Xp, 4*D_full*(Xp**k_full), ls='-', color=pm.cL_Set21[0], mec='k', label='Full curve')
#     ax.plot(Xp, 4*D_lowDt*(Xp**k_lowDt), ls='-', color=pm.cL_Set21[1], mec='k', 
#             label=f'First 4 pts\n$\\alpha$ = {k_lowDt:.2f}')
#     ax.plot(Xp, 4*D_highDt*(Xp**k_highDt), ls='-', color=pm.cL_Set21[2], mec='k', 
#             label=f'Last 15 pts\n$\\alpha$ = {k_highDt:.2f}')
#     ax.axvline(Tc, color='gray', lw=1.5, label=f'$T_c$ = {Tc:.2f}')
#     ax.legend()
#     ax.grid()
#     ax.set_xlim([0.5e-1, 2e1])
#     ax.set_ylim([0.5e-3, 2e0])
#     ax.set_ylabel('MSD (um²)')
#     ax.set_xlabel('T (s)')
#     plt.show()
    
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_xscale('log')
ax.set_yscale('log')

dict_res = {'label':[],
            'D_full':[],
            'k_full':[],
            'D_lowDt':[],
            'k_lowDt':[],
            'D_highDt':[],
            'k_highDt':[],
            }

for ii in range(len(msdNames)):
    df = pd.read_csv(os.path.join(dstDir, dfNames[ii]), sep='\t')
    res_emsd = pd.read_csv(os.path.join(dstDir, msdNames[ii]), sep='\t')
    T, MSD = res_emsd['lagt'], res_emsd['msd']
    
    dict_MSDfits = ufun.json2dict(dstDir, jsonNames[ii])
    D_linear = dict_MSDfits['D_linear']
    k_full = dict_MSDfits['k_full']
    D_full = dict_MSDfits['D_full']
    k_lowDt = dict_MSDfits['k_lowDt']
    D_lowDt = dict_MSDfits['D_lowDt']
    k_highDt = dict_MSDfits['k_highDt']
    D_highDt = dict_MSDfits['D_highDt']
    
    Tc = (D_lowDt/D_highDt)**(1/(k_highDt-k_lowDt))  

    label = msdNames[ii].split('_')[2]

    ax.plot(T, MSD, label=label, color=pm.cL_Set21[ii], 
            # ls='-', lw = 1, alpha=0.85,)
            marker='o', markersize=3, alpha=0.85)
    
    dict_res['label'].append(label)
    dict_res['D_full'].append(D_full)
    dict_res['k_full'].append(k_full)
    dict_res['D_lowDt'].append(D_lowDt)
    dict_res['k_lowDt'].append(k_lowDt)
    dict_res['D_highDt'].append(D_highDt)
    dict_res['k_highDt'].append(k_highDt)
    

Xp1 = np.array([1e-1, 5e-1])
Xp2 = np.array([1, 2])
ax.plot(Xp1, 12e-3*Xp1**0.5, color = 'gray', ls=':', label=r'$y \propto x^{1/2}$')
ax.plot(Xp2, 0.15e-1*Xp2**1, color = 'gray', ls='--', label=r'$y \propto x^{1}$')
ax.legend(edgecolor='None')#, title='Tpf')
ax.grid()
ax.set_xlim([0.4e-1, 0.6e1])
ax.set_ylim([2e-3, 0.5e0])
ax.set_ylabel('MSD (µm²)')
ax.set_xlabel(r'$\Delta t$ (s)')

plt.show()


df_Diffusion = pd.DataFrame(dict_res)
df_Diffusion['Tpf_s'] = df_Diffusion['label'].apply(lambda x : Tpf_str2num(x))
df_Diffusion['Tpf_min'] = df_Diffusion['Tpf_s']/60

fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex = True, layout='compressed')
ax = axes[0]
# ax.plot(np.arange(len(df_Diffusion)), df_Diffusion.D_full, ls='-', marker='o', label=r'$D_{full}$')
# ax.plot(np.arange(len(df_Diffusion)), df_Diffusion.D_lowDt, ls='-', marker='o', label=r'$D_{\Delta t \leq 0.5s}$')
# ax.plot(np.arange(len(df_Diffusion)), df_Diffusion.D_highDt, ls='-', marker='o', label=r'$D_{\Delta t \geq 1s}$')
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.D_full, ls='-', marker='o', label=r'All $\Delta t$')
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.D_lowDt, ls='-', marker='o', label=r'$\Delta t \leq 0.5s$')
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.D_highDt, ls='-', marker='o', label=r'$\Delta t \geq 1s$')
ax.set_ylabel(r'$D_{eff}\ (\mu m^2/s^\alpha)$')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1]
# ax.plot(np.arange(len(df_Diffusion)), df_Diffusion.k_full, ls='-', marker='o', label=r'$\alpha_{full}$')
# ax.plot(np.arange(len(df_Diffusion)), df_Diffusion.k_lowDt, ls='-', marker='o', label=r'$\alpha_{\Delta t \leq 0.5s}$')
# ax.plot(np.arange(len(df_Diffusion)), df_Diffusion.k_highDt, ls='-', marker='o', label=r'$\alpha_{\Delta t \geq 1s}$')
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.k_full, ls='-', marker='o', label=r'All $\Delta t$')
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.k_lowDt, ls='-', marker='o', label=r'$\Delta t \leq 0.5s$')
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.k_highDt, ls='-', marker='o', label=r'$\Delta t \geq 1s$')
ax.set_ylabel(r'$\alpha$')
ax.set_xticks(df_Diffusion['Tpf_min'].values)
ax.set_xticklabels(df_Diffusion['Tpf_min'].values, rotation = 20)
ax.set_xlabel('Tpf (min)')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

plt.show()

# %%%% Import tracks -> run imsd

IMSD = []

for ii in range(len(dfNames)): # len(dfNames)
    dfName = dfNames[ii]
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    df.particle = df.particle.astype(int)
    
    res_imsd = tp.motion.imsd(df, UmPerPix, FPS, max_lagtime=200).reset_index()
    IMSD.append(res_imsd)

# %%%% Plot a few imsd

res_imsd = IMSD[0]
df = res_imsd.dropna(axis=1)

dt = df['lag time [s]']
Np = len(df.columns)
plot_cols = df.columns.values[np.arange(1, Np, Np//33)] #.astype(int)

fig, ax = plt.subplots(1, 1)
ax.set_xscale('log')
ax.set_yscale('log')
for col in plot_cols:
    msd = df.loc[:, col].values
    ax.plot(dt, msd, ls='-', color='gray', lw=1,
            marker='', alpha = 0.75)
    
ax.plot([], [], ls='-', color='gray', lw=1,
        marker='', alpha = 0.75, label='Single particles MSD')
    

#### Long term MSD
ii = 0
xmlPath = xmlPaths[ii]
dfName = dfNames[ii]
print(dfName)
Tracks = tbca.importTrackMateTracks(xmlPath)

Np = len(Tracks)
    
column_names = ['frame', 'x', 'y', 'particle']
all_tracks = []
for i, track in enumerate(Tracks):
    nT = len(track)
    # test_x_sat = ((np.max(track[:, 1]) - np.min(track[:, 1])) < 1)
    # test_y_sat = ((np.max(track[:, 2]) - np.min(track[:, 2])) < 1)
    test_x_sat = ((np.max(track[:, 1]) == (N_pix-1)) or (np.min(track[:, 1]) == 0))
    test_y_sat = ((np.max(track[:, 2]) == (N_pix-1)) or (np.min(track[:, 2]) == 0))
    if (not test_x_sat) and (not test_y_sat) and (nT >= 250):
        track = np.concat((track, np.ones((len(track[:,0]), 1), dtype=int) * (i+1)), axis = 1)
        track[:, 0] = track[:, 0].astype(int) + 1
        all_tracks.append(track)

concat_tracks = np.concat(all_tracks, axis = 0)
df = pd.DataFrame({column_names[k] : concat_tracks[:,k] for k in range(len(column_names))})
res_emsd = tp.motion.emsd(df, UmPerPix, FPS, max_lagtime=200).reset_index()

ax.plot(res_emsd.lagt, res_emsd.msd, ls='-', color='k', lw=2,
        marker='', alpha = 1, label='Global MSD')

ax.legend(edgecolor='None')#, title='Tpf')
ax.grid()
ax.set_xlim([0.8e-1, 25e0])
ax.set_ylim([2e-3, 5e0])
ax.set_ylabel('MSD (µm²)')
ax.set_xlabel(r'$\Delta t$ (s)')

plt.show()

# %%%% From imsd -> Diffusion Map

DF_PART_MSD = []

for ii in range(len(dfNames)): # range(len(dfNames)) [2]
    im = ufun.load_stack_region(tifPaths[ii], time_indices=[0])[0]
    dfName = dfNames[ii]
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    df.particle = df.particle.astype(int)
    
    res_imsd = IMSD[ii]
    dt = res_imsd.loc[:, 'lag time [s]']
    
    dict_particle_MSD = {
                        'Pid':[],
                        'Xc':[],
                        'Yc':[],
                        'theta':[],
                        'fmin':[],
                        'fmax':[],
                        'D_lin':[],
                        'D_full':[],
                        'k_full':[],
                        }
    
    # fig, ax = plt.subplots(1, 1)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    
    Pid_list = df.particle.unique()
    for p in Pid_list[:]:
        dfp = df[df['particle']==p]
        msd = res_imsd.loc[:, p].dropna()
        # ax.plot(dt, msd, 'o', markersize=4, alpha=0.02, 
        #         color='navy', mec='None')
                
        parms, results = ufun.fitLineHuber(dt[:len(msd)], msd, 
                                           with_intercept = False)
        D_linear = parms.values[0]/4
        
        parms, results = ufun.fitLineHuber(np.log(dt[:len(msd)]), np.log(msd), 
                                           with_intercept = True)
        b, a = parms
        k_full = a
        D_full = np.exp(b)/4
        
        fmin, fmax = np.min(dfp.frame), np.max(dfp.frame)
        X, Y = dfp.x, dfp.y
        XY = np.array([X, Y]).T
        CvxHull = MultiPoint(XY).convex_hull
        Xc, Yc = CvxHull.centroid.coords[0]
        theta = np.atan2(Yc - C_pix, Xc - C_pix)
        
        dict_particle_MSD['Pid'].append(p)
        dict_particle_MSD['Xc'].append(Xc)
        dict_particle_MSD['Yc'].append(Yc)
        dict_particle_MSD['theta'].append(theta) # *180/np.pi
        dict_particle_MSD['fmin'].append(fmin)
        dict_particle_MSD['fmax'].append(fmax)
        dict_particle_MSD['D_lin'].append(D_linear)
        dict_particle_MSD['D_full'].append(D_full)
        dict_particle_MSD['k_full'].append(k_full)
        
    df_particle_MSD = pd.DataFrame(dict_particle_MSD)
    DF_PART_MSD.append(df_particle_MSD)


#### SAVE

tableNames = [tifName.split('.')[0] + '_partTrajData.csv' for tifName in tifNames]
for ii in range(len(dfNames)):
    df_particle_MSD = DF_PART_MSD[ii]
    df_particle_MSD.to_csv(os.path.join(dstDir, tableNames[ii]), sep=';', index=False)


# %%%% Plot the Map

pm.setGraphicOptions(mode='print')
df_centers = pd.read_csv(os.path.join(srcDir, 'OrganizingCenters.csv'), sep=';')
tableNames = [tifName.split('.')[0] + '_partTrajData.csv' for tifName in tifNames]


for ii in [0, 2, 3, 5, 7]: # len(dfNames)
    try:
        t = nbimages//2
        im = ufun.load_stack_region(tifPaths[ii], time_indices=[t])[0]
        NoImg=False
    except:
        NoImg=True
    df_particle_MSD = pd.read_csv(os.path.join(dstDir, tableNames[ii]), sep=';')
    label = msdNames[ii].split('_')[2]
    
    M_boxes = 15
    L_box = N_pix/M_boxes
    
    Xc, Yc = df_centers.loc[ii, 'xc'], df_centers.loc[ii, 'yc']

    df_particle_MSD['Xb'] = (df_particle_MSD['Xc'].values//L_box).astype(int)
    df_particle_MSD['Yb'] = (df_particle_MSD['Yc'].values//L_box).astype(int)

    df_particle_MSD['Bxy'] = [(x,y) for x, y in zip(df_particle_MSD['Xb'], df_particle_MSD['Yb'])]

    grouped = df_particle_MSD.groupby('Bxy')
    df_grid_MSD = grouped.agg({'Pid':'count',
                               'D_lin':'median',
                               'D_full':'median',
                               'k_full':'median',
                               }).rename(columns={'Pid':'count'}).reset_index()

    df_grid_MSD = df_grid_MSD[df_grid_MSD['count'] >= 5]

    lims = np.linspace(0, N_pix-1, (M_boxes+1))
    fig, axes = plt.subplots(2, 3, figsize = (10, 6), layout='compressed')
    axes_f = axes.flatten()
    
    #### 2.1 - Image
    ax = axes_f[0]
    if not NoImg:
        vmin, vmax = np.percentile(im, 0.5), np.percentile(im, 99.5)
        ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
        ax.hlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
        ax.vlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
        ax.plot(Xc, Yc, 'ro', markersize=3)
    ax.set_xlim([0, 511])
    ax.set_ylim([0, 511])
    ax.set_title(f'Tpf {label} - tiled image')
    
    #### 2.2 - Count Heatmap
    ax = axes_f[4]
    parm = 'count'
    axtitle = r'Trajectories / tile'
    cbarlabel = r'$N$'
    
    fig, ax = MSD_HeatMap(df_grid_MSD, M_boxes, 'Bxy', parm,
                            axtitle = axtitle, cbarlabel = cbarlabel,
                            cmap = 'GnBu', norm_type = 'lin',
                            fig=fig, ax=ax)
    
    
    #### 2.3 - D_full Heatmap
    ax = axes_f[2]
    parm = 'D_full'
    axtitle = r'$D$ - from $MSD = 4D\cdot\Delta t^\alpha$'
    cbarlabel = r'$D$ (µm²/$s^\alpha$)'
    
    fig, ax = MSD_HeatMap(df_grid_MSD, M_boxes, 'Bxy', parm,
                            axtitle = axtitle, cbarlabel = cbarlabel,
                            cmap='RdYlBu_r', norm_type = 'lin',
                            c_vmin = 5e-3, c_vmax = 12.5e-3,
                            fig=fig, ax=ax)
    
    
    #### 2.4 k_full Heatmap
    ax = axes_f[5]
    parm = 'k_full'
    axtitle = r'$\alpha$ - from $MSD = 4D\cdot\Delta t^\alpha$'
    cbarlabel = r'$\alpha$'
    
    fig, ax = MSD_HeatMap(df_grid_MSD, M_boxes, 'Bxy', parm,
                            axtitle = axtitle, cbarlabel = cbarlabel,
                            cmap='PuOr_r', norm_type = 'lin',
                            c_vmin = 0.6, c_vmax = 1.15,
                            fig=fig, ax=ax)

    
    
    #### 2.5 D_lin Heatmap
    ax = axes_f[1]
    parm = 'D_lin'
    axtitle = r'$D$ - from $MSD = 4D\cdot\Delta t$'
    cbarlabel = r'$D$ (µm²/s)'
    
    fig, ax = MSD_HeatMap(df_grid_MSD, M_boxes, 'Bxy', parm,
                            axtitle = axtitle, cbarlabel = cbarlabel,
                            cmap='RdYlBu_r', norm_type = 'lin',
                            c_vmin = 5e-3, c_vmax = 12.5e-3,
                            fig=fig, ax=ax)

    plt.show()
    
    


# %%%% Plot the points

df_centers = pd.read_csv(os.path.join(srcDir, 'OrganizingCenters.csv'), sep=';')
tableNames = [tifName.split('.')[0] + '_partTrajData.csv' for tifName in tifNames]

# norm=mpl.colors.LogNorm() # norm=mpl.colors.Normalize()

for ii in [0, 2, 3, 5, 7]: #range(len(dfNames)): #
    t = nbimages//2
    im = ufun.load_stack_region(tifPaths[ii], time_indices=[t])[0]
    df_particle_MSD = pd.read_csv(os.path.join(dstDir, tableNames[ii]), sep=';')
    label = msdNames[ii].split('_')[2]
    
    M_boxes = 15
    L_box = N_pix/M_boxes

    lims = np.linspace(0, N_pix-1, (M_boxes+1))
    fig, axes = plt.subplots(2, 2, figsize = (8, 6), layout='compressed')
    axes_f = axes.flatten()
    ax = axes_f[0]
    vmin, vmax = np.percentile(im, 0.5), np.percentile(im, 99.5)
    ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
    ax.hlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
    ax.vlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
    ax.set_xlim([0, 511])
    ax.set_ylim([0, 511])
    ax.set_title(f'Tpf {label}')
    
    ax = axes_f[1]
    parm = 'k_full' # 'D_full', 'k_full'
    axtitle = r'$\alpha$ - from $MSD = 4D\cdot\Delta t^\alpha$'
    cbarlabel = r'$\alpha$'
    ax.set_title(axtitle)
    ax.set_aspect('equal', adjustable='box')
    v_high = np.percentile(df_particle_MSD[parm], 98)
    df_f = df_particle_MSD[df_particle_MSD[parm] < v_high]
    
    g = ax.scatter(df_f['Xc'], df_f['Yc'], 
                   c=df_f[parm], cmap='PuOr_r',
                   s = 6, alpha = 1, edgecolor='None',
                   norm=mpl.colors.Normalize(vmin = 0.6, vmax = 1.15),
                   )
    cbar = fig.colorbar(g, label=cbarlabel)
    ax.set_xlim([0, 511])
    ax.set_ylim([0, 511])
    
    
    
    ax = axes_f[2]
    parm = 'D_lin' # 'D_full', 'k_full'
    axtitle = r'$D$ - from $MSD = 4D\cdot\Delta t$'
    cbarlabel = r'$D$ (µm²/s)'
    ax.set_title(axtitle)
    ax.set_aspect('equal', adjustable='box')
    v_high = np.percentile(df_particle_MSD[parm], 98)
    df_f = df_particle_MSD[df_particle_MSD[parm] < v_high]
    
    g = ax.scatter(df_f['Xc'], df_f['Yc'], 
                   c=df_f[parm], cmap='RdYlBu_r',
                   s = 6, alpha = 1, edgecolor='None',
                   norm=mpl.colors.Normalize(vmin = 5e-3, vmax = 12.5e-3),
                   )
    cbar = fig.colorbar(g, label=cbarlabel)
    ax.set_xlim([0, 511])
    ax.set_ylim([0, 511])

    
    
    ax = axes_f[3]
    parm = 'D_full' # 'D_full', 'k_full'
    axtitle = r'$D$ - from $MSD = 4D\cdot\Delta t^\alpha$'
    cbarlabel = r'$D$ (µm²/$s^\alpha$)'
    ax.set_title(axtitle)
    ax.set_aspect('equal', adjustable='box')
    v_high = np.percentile(df_particle_MSD[parm], 98)
    df_f = df_particle_MSD[df_particle_MSD[parm] < v_high]
    
    g = ax.scatter(df_f['Xc'], df_f['Yc'], 
                   c=df_f[parm], cmap='RdYlBu_r',
                   s = 6, alpha = 1, edgecolor='None',
                   norm=mpl.colors.Normalize(vmin = 5e-3, vmax = 12.5e-3),
                   )
    cbar = fig.colorbar(g, label=cbarlabel)
    ax.set_xlim([0, 511])
    ax.set_ylim([0, 511])
    
    
    plt.show()
        


    
# %%%% Radial / OrthoRadial


max_lagtime = 30
lagT = np.arange(1, 31) / FPS

iLow = ufun.findFirst(lowDt_upper, lagT) + 1
iHigh = ufun.findFirst(highDt_lower, lagT)

df_centers = pd.read_csv(os.path.join(srcDir, 'OrganizingCenters.csv'), sep=';')

tableNames = [tifName.split('.')[0] + '_partTrajData_RandOR.csv' for tifName in tifNames]


for ii in range(len(dfNames)): # len(dfNames)
    print(ii)
    X_MTcenter, Y_MTcenter = df_centers.loc[ii, 'xc'], df_centers.loc[ii, 'yc']
    
    dfName = dfNames[ii]
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    df.particle = df.particle.astype(int)
    
    dict_particle_MSD = {
                        'Pid':[],
                        'Xc':[],
                        'Yc':[],
                        'theta':[],
                        'fmin':[],
                        'fmax':[],
                        'D_r_lin':[],
                        'D_r_linHighDt':[],
                        'D_r_full':[],
                        'k_r_full':[],
                        'D_r_highDt':[],
                        'k_r_highDt':[],
                        'D_r_lowDt':[],
                        'k_r_lowDt':[],
                        'D_or_lin':[],
                        'D_or_linHighDt':[],
                        'D_or_full':[],
                        'k_or_full':[],
                        'D_or_highDt':[],
                        'k_or_highDt':[],
                        'D_or_lowDt':[],
                        'k_or_lowDt':[],
                        }

    Pid_list = df.particle.unique()
    for p in Pid_list[:]:
        dfp = df[df['particle']==p]
        
        # msd = res_imsd.loc[:, p]
        # # ax.plot(dt, msd, 'o', markersize=4, alpha=0.02, 
        # #         color='navy', mec='None')
                
        # parms, results = ufun.fitLineHuber(dt, msd, with_intercept = False)
        # D_linear = parms.values[0]/4
        
        # parms, results = ufun.fitLineHuber(np.log(dt), np.log(msd), 
        #                                    with_intercept = True)
        # b, a = parms
        # k_full = a
        # D_full = np.exp(b)/4
        
        fmin, fmax = np.min(dfp.frame), np.max(dfp.frame)
        X, Y = dfp.x, dfp.y
        XY = np.array([X, Y]).T
        CvxHull = MultiPoint(XY).convex_hull
        Xc, Yc = CvxHull.centroid.coords[0]
        theta = np.atan2(Yc - Y_MTcenter, Xc - X_MTcenter)
        
        RM = np.array([[np.cos(theta), - np.sin(theta)], 
                       [np.sin(theta),   np.cos(theta)]])
        XY_center = np.array([X_MTcenter, Y_MTcenter])
        XY_r = ((XY - XY_center) @ RM) + XY_center
        
        # PLOT A ROTATED TRAJECTORY
        # fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))
        # ax = axes[0]
        # ax.set_aspect('equal', adjustable='box')
        # ax.plot(XY[:,0]-C_pix, XY[:,1]-C_pix, lw=1)
        # ax.plot(XY_r[:,0]-C_pix, XY_r[:,1]-C_pix, lw=1)
        # ax.set_xlim([-C_pix, C_pix])
        # ax.set_ylim([-C_pix, C_pix])
        # ax.grid()
        
        # ax = axes[1]
        # ax.set_aspect('equal', adjustable='box')
        # ax.plot(XY[:,0]-C_pix, XY[:,1]-C_pix, lw=1)
        # ax.grid()
        
        # ax = axes[2]
        # ax.set_aspect('equal', adjustable='box')
        # ax.plot(XY_r[:,0]-C_pix, XY_r[:,1]-C_pix, lw=1)
        # ax.grid()
        
        # plt.show()
        
        # Radial / Orthoradial coordinates
        Rd, ORd = XY_r[:, 0], XY_r[:, 1]
        
        # MSD 1D for each
        # lagT
        MSD_r = (msd_fft_1D(Rd, UmPerPix, FPS, max_lagtime=max_lagtime))['msd']
        MSD_or = (msd_fft_1D(ORd, UmPerPix, FPS, max_lagtime=max_lagtime))['msd']
        
        
        # Fits for D and k
        
        # Radial
        parms, results = ufun.fitLineHuber(lagT, MSD_r, with_intercept = False)
        D_r_linear = parms.values[0]/2
        
        parms, results = ufun.fitLineHuber(lagT[iHigh:], MSD_r[iHigh:], 
                                           with_intercept = False)
        D_r_lin_highDt = parms.values[0]/2
        
        parms, results = ufun.fitLineHuber(np.log(lagT), np.log(MSD_r), 
                                           with_intercept = True)
        b, a = parms
        k_r_full = a
        D_r_full = np.exp(b)/2
        
        parms, results = ufun.fitLineHuber(np.log(lagT[iHigh:]), np.log(MSD_r[iHigh:]), 
                                           with_intercept = True)
        b, a = parms
        k_r_highDt = a
        D_r_highDt = np.exp(b)/2
        
        parms, results = ufun.fitLineHuber(np.log(lagT[:iLow]), np.log(MSD_r[:iLow]), 
                                           with_intercept = True)
        b, a = parms
        k_r_lowDt = a
        D_r_lowDt = np.exp(b)/2
        
        
        
        # OrthoRadial
        parms, results = ufun.fitLineHuber(lagT, MSD_or, with_intercept = False)
        D_or_linear = parms.values[0]/2
        
        parms, results = ufun.fitLineHuber(lagT[iHigh:], MSD_or[iHigh:], 
                                           with_intercept = False)
        D_or_lin_highDt = parms.values[0]/2
        
        parms, results = ufun.fitLineHuber(np.log(lagT), np.log(MSD_or), 
                                           with_intercept = True)
        b, a = parms
        k_or_full = a
        D_or_full = np.exp(b)/2
        
        parms, results = ufun.fitLineHuber(np.log(lagT[iHigh:]), np.log(MSD_or[iHigh:]), 
                                           with_intercept = True)
        b, a = parms
        k_or_highDt = a
        D_or_highDt = np.exp(b)/2
        
        parms, results = ufun.fitLineHuber(np.log(lagT[:iLow]), np.log(MSD_or[:iLow]), 
                                           with_intercept = True)
        b, a = parms
        k_or_lowDt = a
        D_or_lowDt = np.exp(b)/2
        
        
        # Save
        dict_particle_MSD['Pid'].append(p)
        dict_particle_MSD['Xc'].append(Xc)
        dict_particle_MSD['Yc'].append(Yc)
        dict_particle_MSD['theta'].append(theta) # *180/np.pi
        dict_particle_MSD['fmin'].append(fmin)
        dict_particle_MSD['fmax'].append(fmax)
        dict_particle_MSD['D_r_lin'].append(D_r_linear)
        dict_particle_MSD['D_r_linHighDt'].append(D_r_lin_highDt)
        dict_particle_MSD['D_r_full'].append(D_r_full)
        dict_particle_MSD['k_r_full'].append(k_r_full)
        dict_particle_MSD['D_r_highDt'].append(D_r_highDt)
        dict_particle_MSD['k_r_highDt'].append(k_r_highDt)
        dict_particle_MSD['D_r_lowDt'].append(D_r_lowDt)
        dict_particle_MSD['k_r_lowDt'].append(k_r_lowDt)
        dict_particle_MSD['D_or_lin'].append(D_or_linear)
        dict_particle_MSD['D_or_linHighDt'].append(D_or_lin_highDt)
        dict_particle_MSD['D_or_full'].append(D_or_full)
        dict_particle_MSD['k_or_full'].append(k_or_full)
        dict_particle_MSD['D_or_highDt'].append(D_or_highDt)
        dict_particle_MSD['k_or_highDt'].append(k_or_highDt)
        dict_particle_MSD['D_or_lowDt'].append(D_or_lowDt)
        dict_particle_MSD['k_or_lowDt'].append(k_or_lowDt)
        
        
        
    df_particle_MSD_CylCoo = pd.DataFrame(dict_particle_MSD)
    df_particle_MSD_CylCoo.to_csv(os.path.join(dstDir, tableNames[ii]), sep=';', index=False)

    
    # dict_boxes = distribute_in_boxes(dict_particle_MSD, N_pix, M_boxes, 
    #                     str_Id = 'Pid', str_X = 'Xc', str_Y = 'Yc')
    # L = [len(dict_boxes[k]) for k in dict_boxes.keys()]

    

    # plt.show()
    
    
# %%%% Plot the Maps

df_centers = pd.read_csv(os.path.join(srcDir, 'OrganizingCenters.csv'), sep=';')
tableNames = [tifName.split('.')[0] + '_partTrajData_RandOR.csv' for tifName in tifNames]

# norm=mpl.colors.LogNorm() # norm=mpl.colors.Normalize()

for ii in [0, 2, 3, 5, 7]: #range(len(dfNames)): #
    t = nbimages//2
    im = ufun.load_stack_region(tifPaths[ii], time_indices=[t])[0]
    df_particle_MSD_CylCoo = pd.read_csv(os.path.join(dstDir, tableNames[ii]), sep=';')
    label = msdNames[ii].split('_')[2]
    
    X_MTcenter, Y_MTcenter = df_centers.loc[ii, 'xc'], df_centers.loc[ii, 'yc']

    df = df_particle_MSD_CylCoo
    
    M_boxes = 15
    L_box = N_pix/M_boxes
    
    df['Xb'] = (df['Xc'].values//L_box).astype(int)
    df['Yb'] = (df['Yc'].values//L_box).astype(int)
    
    df['Bxy'] = [(x,y) for x, y in zip(df['Xb'], df['Yb'])]
    
    grouped = df.groupby('Bxy')
    df_grid_MSD = grouped.agg({'Pid':'count',
                               'D_r_lin':'median',
                               'D_r_highDt':'median',
                               'k_r_highDt':'median',
                               'D_or_lin':'median',
                               'D_or_highDt':'median',
                               'k_or_highDt':'median',
                               }).rename(columns={'Pid':'count'}).reset_index()
    
    df_grid_MSD = df_grid_MSD[df_grid_MSD['count'] >= 5]
    
    lims = np.linspace(0, N_pix-1, (M_boxes+1))
    fig, axes = plt.subplots(2, 3, figsize = (12, 8), layout='compressed')
    axes_f = axes.flatten()
    
    ax = axes_f[0]
    vmin, vmax = np.percentile(im, 0.5), np.percentile(im, 99.5)
    ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
    ax.hlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
    ax.vlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
    ax.plot(X_MTcenter, Y_MTcenter, 'ro', markersize=3)
    ax.set_xlim([0, 511])
    ax.set_ylim([0, 511])
    ax.set_title(f'Tpf {label} - tiled image')
    
    # df_2_heatmap(df_grid_MSD, axes_f[1], parmCol='k_r_full', boxCol='Bxy', 
    #              y_ascending=True, cmap="viridis", annotate=False, colorScale='linear')
    
    # df_2_heatmap(df_grid_MSD, axes_f[2], parmCol='k_or_full', boxCol='Bxy', 
    #              y_ascending=True, cmap="viridis", annotate=False, colorScale='linear')
    
    # df_2_heatmap(df_grid_MSD, axes_f[4], parmCol='D_r_full', boxCol='Bxy', 
    #              y_ascending=True, cmap="viridis", annotate=False, colorScale='log')
    
    # df_2_heatmap(df_grid_MSD, axes_f[5], parmCol='D_or_full', boxCol='Bxy', 
    #              y_ascending=True, cmap="viridis", annotate=False, colorScale='log')
    
    ax = axes_f[3]
    parm = 'count'
    axtitle = r'Trajectories / tile'
    cbarlabel = r'$N$'
    
    fig, ax = MSD_HeatMap(df_grid_MSD, M_boxes, 'Bxy', parm,
                            axtitle = axtitle, cbarlabel = cbarlabel,
                            cmap='GnBu', norm_type = 'lin',
                            fig=fig, ax=ax)
    
    
    ax = axes_f[1]
    parm = 'k_r_highDt'
    axtitle = r'$\alpha$ radial ($\Delta t \geq$ 1s)'
    cbarlabel = r'$\alpha$'
    
    fig, ax = MSD_HeatMap(df_grid_MSD, M_boxes, 'Bxy', parm,
                            axtitle = axtitle, cbarlabel = cbarlabel,
                            cmap='PuOr_r', norm_type = 'lin',
                            c_vmin=0.6, c_vmax=1.15,
                            fig=fig, ax=ax)
    
    ax = axes_f[2]
    parm = 'k_or_highDt'
    axtitle = r'$\alpha$ ortho-radial ($\Delta t \geq$ 1s)'
    cbarlabel = r'$\alpha$'
    
    fig, ax = MSD_HeatMap(df_grid_MSD, M_boxes, 'Bxy', parm,
                            axtitle = axtitle, cbarlabel = cbarlabel,
                            cmap='PuOr_r', norm_type = 'lin',
                            c_vmin=0.6, c_vmax=1.15,
                            fig=fig, ax=ax)
    
    ax = axes_f[4]
    parm = 'D_r_lin'
    axtitle = r'$D$ radial (linear fit, $\Delta t \geq$ 1s)'
    cbarlabel = r'$D$ (µm²/s)'
    
    fig, ax = MSD_HeatMap(df_grid_MSD, M_boxes, 'Bxy', parm,
                            axtitle = axtitle, cbarlabel = cbarlabel,
                            cmap='RdYlBu_r', norm_type = 'lin',
                            c_vmin = 3e-3, c_vmax = 11.5e-3,
                            fig=fig, ax=ax)
    
    ax = axes_f[5]
    parm = 'D_or_lin'
    axtitle = r'$D$ ortho-radial (linear fit, $\Delta t \geq$ 1s)'
    cbarlabel = r'$D$ (µm²/s)'
    
    fig, ax = MSD_HeatMap(df_grid_MSD, M_boxes, 'Bxy', parm,
                            axtitle = axtitle, cbarlabel = cbarlabel,
                            cmap='RdYlBu_r', norm_type = 'lin',
                            c_vmin = 3e-3, c_vmax = 11.5e-3,
                            fig=fig, ax=ax)
    
    plt.show()






# lims = np.linspace(0, N_pix-1, (M_boxes+1))
# fig, axes = plt.subplots(2, 3, figsize = (12, 8), layout='compressed')
# axes_f = axes.flatten()

# ax = axes_f[0]
# vmin, vmax = np.percentile(im, 0.5), np.percentile(im, 99.5)
# ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
# ax.hlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
# ax.vlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
# ax.set_xlim([0, 511])
# ax.set_ylim([0, 511])
# ax.set_title('Tiled image')

# df_2_heatmap(df_grid_MSD, axes_f[1], parmCol='k_r_full', boxCol='Bxy', 
#              y_ascending=True, cmap="viridis", annotate=False, colorScale='linear')

# df_2_heatmap(df_grid_MSD, axes_f[2], parmCol='k_or_full', boxCol='Bxy', 
#              y_ascending=True, cmap="viridis", annotate=False, colorScale='linear')

# df_2_heatmap(df_grid_MSD, axes_f[4], parmCol='D_r_full', boxCol='Bxy', 
#              y_ascending=True, cmap="viridis", annotate=False, colorScale='log')

# df_2_heatmap(df_grid_MSD, axes_f[5], parmCol='D_or_full', boxCol='Bxy', 
#              y_ascending=True, cmap="viridis", annotate=False, colorScale='log')


# plt.show()

# %%%% Plot the points

pm.setGraphicOptions(mode='print')

df_centers = pd.read_csv(os.path.join(srcDir, 'OrganizingCenters.csv'), sep=';')

tableNames = [tifName.split('.')[0] + '_partTrajData_RandOR.csv' for tifName in tifNames]

M_boxes=15

ii = 7

for ii in [0, 2, 3, 5, 7]: # len(dfNames)
    print(ii)
    X_MTcenter, Y_MTcenter = df_centers.loc[ii, 'xc'], df_centers.loc[ii, 'yc']

    df_particle_MSD_CylCoo = pd.read_csv(os.path.join(dstDir, tableNames[ii]), sep=';')
    # dfName = dfNames[ii]
    # df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    # df.particle = df.particle.astype(int)
    
    t = nbimages//2
    im = ufun.load_stack_region(tifPaths[ii], time_indices=[t])[0]
    df_particle_MSD_CylCoo = pd.read_csv(os.path.join(dstDir, tableNames[ii]), sep=';')
    label = tableNames[ii].split('_')[2]


    df = df_particle_MSD_CylCoo
    
    lims = np.linspace(0, N_pix-1, (M_boxes+1))
    fig, axes = plt.subplots(1, 4, figsize = (12, 4), layout='compressed')
    axes_f = axes.flatten()
    
    ax = axes_f[0]
    vmin, vmax = np.percentile(im, 0.5), np.percentile(im, 99.5)
    ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
    # ax.hlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
    # ax.vlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
    ax.plot(X_MTcenter, Y_MTcenter, 'ro', markersize=3)
    ax.set_xlim([0, 511])
    ax.set_ylim([0, 511])
    ax.set_title('Original image')
    
    ax = axes_f[1]
    parm1 = 'D_r_full' # 'D_r_full', 'k_r_full'
    v_high1 = np.percentile(df[parm1], 98)
    df_f = df[df[parm1] < v_high1]
    
    g = ax.scatter(df_f['Xc'], df_f['Yc'], 
                   c=df_f[parm1], cmap='viridis',
                   s = 5, alpha = 1, edgecolor='None',
                   norm=mpl.colors.Normalize(), # LogNorm
                   )
    cbar = fig.colorbar(g)
    ax.set_xlim([0, 511])
    ax.set_ylim([0, 511])
    ax.set_title(parm1)
    
    
    ax = axes_f[2]
    parm2 = 'D_or_full' # 'D_or_full', 'k_or_full'
    v_high2 = np.percentile(df[parm2], 98)
    df_f = df[df[parm2] < v_high2]
    
    g = ax.scatter(df_f['Xc'], df_f['Yc'], 
                   c=df_f[parm2], cmap='PuRd',
                   s = 5, alpha = 1, edgecolor='None',
                   norm=mpl.colors.Normalize(),
                   )
    cbar = fig.colorbar(g)
    ax.set_xlim([0, 511])
    ax.set_ylim([0, 511])
    ax.set_title(parm2)
    
    
    ax = axes_f[3]
    df_f = df[(df[parm1] < v_high1) & (df[parm2] < v_high2)]
    df_f['delta'] = df_f[parm1] - df_f[parm2] 
    g = ax.scatter(df_f['Xc'], df_f['Yc'], 
                   c=df_f['delta'], cmap='BuPu',
                   s = 5, alpha = 1, edgecolor='None',
                   norm=mpl.colors.Normalize(),
                   )
    cbar = fig.colorbar(g)
    ax.set_xlim([0, 511])
    ax.set_ylim([0, 511])
    ax.set_title(f'{parm1} - {parm2}')
    
    # Remove the legend and add a colorbar
    
    plt.show()

# %%%% Plot the distributions

pm.setGraphicOptions(mode='print')
ii = 2

df_centers = pd.read_csv(os.path.join(srcDir, 'OrganizingCenters.csv'), sep=';')

tableNames = [tifName.split('.')[0] + '_partTrajData_RandOR.csv' for tifName in tifNames]
df_particle_MSD_CylCoo = pd.read_csv(os.path.join(dstDir, tableNames[ii]), sep=';')
df = df_particle_MSD_CylCoo



fig, axes = plt.subplots(3, 1, figsize = (9, 7), layout='compressed', sharex='col')
axes_f = axes.flatten()

ax = axes_f[0]
# vmin, vmax = np.percentile(im, 0.5), np.percentile(im, 99.5)
# ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
# ax.hlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
# ax.vlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
ax.set_xlim([0, 511])
ax.set_ylim([0, 511])
ax.set_title('Original image')

ax = axes_f[1]
parm1 = 'D_r_lin' # 'D_r_full', 'k_r_full'
v_high1 = np.percentile(df[parm1], 98)
df_f = df[df[parm1] < v_high1]
ax.hist(df_f[parm1].values, bins=60, alpha=0.4, label=parm1)

parm2 = 'D_or_lin' # 'D_or_full', 'k_or_full'
v_high2 = np.percentile(df[parm2], 98)
df_f = df[df[parm2] < v_high2]
ax.hist(df_f[parm2].values, bins=60, alpha=0.4, label=parm2)

ax.legend()
ax.set_title(f'{parm1} & {parm2}')

ax = axes_f[2]
df_f = df[(df[parm1] < v_high1) & (df[parm2] < v_high2)]
df_f['delta'] = df_f[parm1] - df_f[parm2] 
ax.hist(df_f['delta'].values, bins=60)
ax.axvline(0, color='gray', ls='-', lw=0.75, alpha=0.8)
ax.set_title(f'{parm1} - {parm2}')


ax = axes_f[3+1]
parm1 = 'D_r_lowDt' # 'D_r_full', 'k_r_full'
v_high1 = np.percentile(df[parm1], 98)
df_f = df[df[parm1] < v_high1]
ax.hist(df_f[parm1].values, bins=60, alpha=0.4, label=parm1)

parm2 = 'D_or_lowDt' # 'D_or_full', 'k_or_full'
v_high2 = np.percentile(df[parm2], 98)
df_f = df[df[parm2] < v_high2]
ax.hist(df_f[parm2].values, bins=60, alpha=0.4, label=parm2)
ax.legend()
ax.set_title(f'{parm1} & {parm2}')

ax = axes_f[3+2]
df_f = df[(df[parm1] < v_high1) & (df[parm2] < v_high2)]
df_f['delta'] = df_f[parm1] - df_f[parm2] 
ax.hist(df_f['delta'].values, bins=60)
ax.axvline(0, color='gray', ls='-', lw=0.75, alpha=0.8)
ax.set_title(f'{parm1} - {parm2}')


ax = axes_f[6+1]
parm1 = 'D_r_highDt' # 'D_r_full', 'k_r_full'
v_high1 = np.percentile(df[parm1], 98)
df_f = df[df[parm1] < v_high1]
ax.hist(df_f[parm1].values, bins=60, alpha=0.4, label=parm1)

parm2 = 'D_or_highDt' # 'D_or_full', 'k_or_full'
v_high2 = np.percentile(df[parm2], 98)
df_f = df[df[parm2] < v_high2]
ax.hist(df_f[parm2].values, bins=60, alpha=0.4, label=parm2)
ax.legend()
ax.set_title(f'{parm1} & {parm2}')

ax = axes_f[6+2]
df_f = df[(df[parm1] < v_high1) & (df[parm2] < v_high2)]
df_f['delta'] = df_f[parm1] - df_f[parm2] 
ax.hist(df_f['delta'].values, bins=60)
ax.axvline(0, color='gray', ls='-', lw=0.75, alpha=0.8)
ax.set_title(f'{parm1} - {parm2}')

plt.show()


fig, axes = plt.subplots(3, 3, figsize = (9, 7), layout='compressed', sharex='col')
axes_f = axes.flatten()

ax = axes_f[0]
# vmin, vmax = np.percentile(im, 0.5), np.percentile(im, 99.5)
# ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
# ax.hlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
# ax.vlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
ax.set_xlim([0, 511])
ax.set_ylim([0, 511])
ax.set_title('Original image')

ax = axes_f[1]
parm1 = 'k_r_full' # 'D_r_full', 'k_r_full'
v_high1 = np.percentile(df[parm1], 98)
df_f = df[df[parm1] < v_high1]
ax.hist(df_f[parm1].values, bins=60, alpha=0.4, label=parm1)

parm2 = 'k_or_full' # 'D_or_full', 'k_or_full'
v_high2 = np.percentile(df[parm2], 98)
df_f = df[df[parm2] < v_high2]
ax.hist(df_f[parm2].values, bins=60, alpha=0.4, label=parm2)
ax.legend()
ax.set_title(f'{parm1} & {parm2}')

ax = axes_f[2]
df_f = df[(df[parm1] < v_high1) & (df[parm2] < v_high2)]
df_f['delta'] = df_f[parm1] - df_f[parm2] 
ax.hist(df_f['delta'].values, bins=60)
ax.axvline(0, color='gray', ls='-', lw=0.75, alpha=0.8)
ax.set_title(f'{parm1} - {parm2}')


ax = axes_f[3+1]
parm1 = 'k_r_lowDt' # 'D_r_full', 'k_r_full'
v_high1 = np.percentile(df[parm1], 98)
df_f = df[df[parm1] < v_high1]
ax.hist(df_f[parm1].values, bins=60, alpha=0.4, label=parm1)

parm2 = 'k_or_lowDt' # 'D_or_full', 'k_or_full'
v_high2 = np.percentile(df[parm2], 98)
df_f = df[df[parm2] < v_high2]
ax.hist(df_f[parm2].values, bins=60, alpha=0.4, label=parm2)
ax.legend()
ax.set_title(f'{parm1} & {parm2}')

ax = axes_f[3+2]
df_f = df[(df[parm1] < v_high1) & (df[parm2] < v_high2)]
df_f['delta'] = df_f[parm1] - df_f[parm2] 
ax.hist(df_f['delta'].values, bins=60)
ax.axvline(0, color='gray', ls='-', lw=0.75, alpha=0.8)
ax.set_title(f'{parm1} - {parm2}')


ax = axes_f[6+1]
parm1 = 'k_r_highDt' # 'D_r_full', 'k_r_full'
v_high1 = np.percentile(df[parm1], 98)
df_f = df[df[parm1] < v_high1]
ax.hist(df_f[parm1].values, bins=60, alpha=0.4, label=parm1)

parm2 = 'k_or_highDt' # 'D_or_full', 'k_or_full'
v_high2 = np.percentile(df[parm2], 98)
df_f = df[df[parm2] < v_high2]
ax.hist(df_f[parm2].values, bins=60, alpha=0.4, label=parm2)
ax.legend()
ax.set_title(f'{parm1} & {parm2}')

ax = axes_f[6+2]
df_f = df[(df[parm1] < v_high1) & (df[parm2] < v_high2)]
df_f['delta'] = df_f[parm1] - df_f[parm2] 
ax.hist(df_f['delta'].values, bins=60)
ax.axvline(0, color='gray', ls='-', lw=0.75, alpha=0.8)
ax.set_title(f'{parm1} - {parm2}')


plt.show()

# %%%% Plot the distributions - GOOD FOR PRES

pm.setGraphicOptions(mode='print')

idx_to_plot = [0, 2, 4, 5, 7]

fig, axes = plt.subplots(2, len(idx_to_plot), figsize = (10, 4), layout='compressed', sharex='row')
axes_f = axes.flatten(order='F')
axes_f[0].set_xlim([0, 0.018])

for k, ii in enumerate(idx_to_plot):


    df_centers = pd.read_csv(os.path.join(srcDir, 'OrganizingCenters.csv'), sep=';')
    
    tableNames = [tifName.split('.')[0] + '_partTrajData_RandOR.csv' for tifName in tifNames]
    
    df_particle_MSD_CylCoo = pd.read_csv(os.path.join(dstDir, tableNames[ii]), sep=';')
    df = df_particle_MSD_CylCoo
    title = tableNames[ii].split('_')[2]
    
    
    
    
    # ax = axes_f[2*k]
    # vmin, vmax = np.percentile(im, 0.5), np.percentile(im, 99.5)
    # ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
    # ax.hlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
    # ax.vlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
    # ax.set_xlim([0, 511])
    # ax.set_ylim([0, 511])
    # ax.set_title('Original image')
    
    ax = axes_f[2*k]
    [parm1, parm2] = [f'D_{c}_lin' for c in ['r', 'or']] # D_{c}_lin # D_{c}_linHighDt
    
    v_high1 = np.percentile(df[parm1], 98)
    df_f = df[df[parm1] < v_high1]
    ax.hist(df_f[parm1].values, bins=30, alpha=0.4, label='Radial')
    
    v_high2 = np.percentile(df[parm2], 98)
    df_f = df[df[parm2] < v_high2]
    ax.hist(df_f[parm2].values, bins=30, alpha=0.4, label='Orthoradial')
    
    ax.set_xlabel(r'$D$ (µm²/s)')
    
    if k==0:
        ax.set_ylabel('N particles')
    
    if k==len(idx_to_plot)-1:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
    ax.set_title('Tpf ' + title)
    
    
    ax = axes_f[2*k + 1]
    [parm1, parm2] = [f'k_{c}_full' for c in ['r', 'or']] # k_{c}_full # k_{c}_highDt
    
    v_high1 = np.percentile(df[parm1], 98)
    df_f = df[df[parm1] < v_high1]
    ax.hist(df_f[parm1].values, bins=30, alpha=0.4, label='Radial')
    
    v_high2 = np.percentile(df[parm2], 98)
    df_f = df[df[parm2] < v_high2]
    ax.hist(df_f[parm2].values, bins=30, alpha=0.4, label='Orthoradial')
    
    ax.set_xlabel(r'$\alpha$')
    
    if k==0:
        ax.set_ylabel('N particles')
    
    if k==len(idx_to_plot)-1:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
    # ax.set_title('\n')
    
    
    
plt.show()

# %%%% Check the nature of the distribution

import statsmodels.api as sm

pm.setGraphicOptions(mode='print')

tableNames = [tifName.split('.')[0] + '_partTrajData_RandOR.csv' for tifName in tifNames]

M_boxes=15

ii = 2

# for ii in range(len(dfNames)): # len(dfNames)
#     print(ii)

df_particle_MSD_CylCoo = pd.read_csv(os.path.join(dstDir, tableNames[ii]), sep=';')
# dfName = dfNames[ii]
# df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
# df.particle = df.particle.astype(int)
    


# -----
df = df_particle_MSD_CylCoo
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes_f = axes.flatten()

# ---
ax = axes_f[0]
parm = 'D_r_full' # 'D_r_full', 'k_r_full'
axtitle = r'QQplot for $D$ radial'
df_f = df[df[parm] > 0] 
data = df_f[parm].values

ax.set_title(axtitle)
ax.axline((0, 0), slope=1, color="k", linestyle='-.', linewidth=1, zorder=6)
sm.qqplot(data, fit=True, line=None, ax=ax, markerfacecolor = pm.cL_Set21[0], 
          markeredgecolor = 'None', markersize=4, alpha=0.75)
ax.plot([], [], label='QQplot for\nnormal distribution', ls='', marker='o', 
        markerfacecolor = pm.cL_Set21[0], markeredgecolor = 'None', markersize=5)
sm.qqplot(np.log(data), fit=True, line=None, ax=ax, markerfacecolor = pm.cL_Set21[1], 
          markeredgecolor = 'None', markersize=4, alpha=0.75)
ax.plot([], [], label='QQplot for\nlog-normal distribution', ls='', marker='o', 
        markerfacecolor = pm.cL_Set21[1], markeredgecolor = 'None', markersize=5)
ax.grid()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
ax.legend(fontsize=8)
plt.show()


# ---
ax = axes_f[1]
parm = 'k_r_full' # 'D_r_full', 'k_r_full'
axtitle = r'QQplot for $\alpha$ radial'
df_f = df[df[parm] > 0] 
data = df_f[parm].values

ax.set_title(axtitle)
ax.axline((0, 0), slope=1, color="k", linestyle='-.', linewidth=1, zorder=6)
sm.qqplot(data, fit=True, line=None, ax=ax, markerfacecolor = pm.cL_Set21[0], 
          markeredgecolor = 'None', markersize=4, alpha=0.75)
ax.plot([], [], label='QQplot for\nnormal distribution', ls='', marker='o', 
        markerfacecolor = pm.cL_Set21[0], markeredgecolor = 'None', markersize=5)
sm.qqplot(np.log(data), fit=True, line=None, ax=ax, markerfacecolor = pm.cL_Set21[1], 
          markeredgecolor = 'None', markersize=4, alpha=0.75)
ax.plot([], [], label='QQplot for\nlog-normal distribution', ls='', marker='o', 
        markerfacecolor = pm.cL_Set21[1], markeredgecolor = 'None', markersize=5)
ax.grid()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
ax.legend(fontsize=8)
plt.show()

# %%%% Plot the distributions


list_parm_cols = [
                'D_r_lin',
                'D_r_linHighDt',
                'D_r_full',
                'k_r_full',
                'D_r_highDt',
                'k_r_highDt',
                'D_r_lowDt',
                'k_r_lowDt',
                'D_or_lin',
                'D_or_linHighDt',
                'D_or_full',
                'k_or_full',
                'D_or_highDt',
                'k_or_highDt',
                'D_or_lowDt',
                'k_or_lowDt',
                ]

res_dict = {'Tpf':[], 'N':[]}
res_dict.update({k + '_mean' : [] for k in list_parm_cols})
res_dict.update({k + '_std' : [] for k in list_parm_cols})

tableNames = [tifName.split('.')[0] + '_partTrajData_RandOR.csv' for tifName in tifNames]


for ii in range(len(tableNames)): #
    df_particle_MSD_CylCoo = pd.read_csv(os.path.join(dstDir, tableNames[ii]), sep=';')
    df = df_particle_MSD_CylCoo
    
    label = tableNames[ii].split('_')[2]
    print(label)
    
    res_dict['Tpf'].append(label)
    res_dict['N'].append(len(df))
    
    for pcol in list_parm_cols:
        df_f = df[df[pcol] > 0] 
        data = df_f[pcol].values

        if 'k_' in pcol:
            m, std = np.mean(data), np.std(data)
        elif 'D_' in pcol:
            data = np.log(data)
            m, std = np.mean(data), np.std(data)
            
        res_dict[pcol + '_mean'].append(m)
        res_dict[pcol + '_std'].append(std)
    
res_df = pd.DataFrame(res_dict)
res_df['Tpf_s'] = res_df['Tpf'].apply(lambda x : Tpf_str2num(x))
res_df['Tpf_min'] = res_df['Tpf_s']/60



fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex = True, layout='compressed')
ax = axes[0]
# ax.plot(res_df.Tpf_min, df_Diffusion.D_full, ls='-', marker='o', label=r'All $\Delta t$')
# ax.plot(res_df.Tpf_min, df_Diffusion.D_lowDt, ls='-', marker='o', label=r'$\Delta t \leq 0.5s$')
# ax.plot(res_df.Tpf_min, df_Diffusion.D_highDt, ls='-', marker='o', label=r'$\Delta t \geq 1s$')
# ax.errorbar(res_df.Tpf_min, np.exp(res_df.D_r_linHighDt_mean), 
#             ls='-', marker='o', label=r'Radial',
#             yerr=[np.exp(res_df.D_r_linHighDt_mean - res_df.D_r_linHighDt_std)/(res_df.N**0.5), 
#                   np.exp(res_df.D_r_linHighDt_mean + res_df.D_r_linHighDt_std)/(res_df.N**0.5)],
#             ecolor='k', capsize=2)
# ax.errorbar(res_df.Tpf_min, np.exp(res_df.D_or_full_mean), 
#         ls='-', marker='o', label=r'Ortho-radial',
#         yerr=[np.exp(res_df.D_or_linHighDt_mean - res_df.D_or_linHighDt_std)/(res_df.N**0.5), 
#               np.exp(res_df.D_or_linHighDt_mean + res_df.D_or_linHighDt_std)/(res_df.N**0.5)],
#         ecolor='k', capsize=2)
ax.errorbar(res_df.Tpf_min, np.exp(res_df.D_r_lin_mean), 
            ls='-', marker='o', label=r'Radial',
            yerr=[np.exp(res_df.D_r_lin_mean - res_df.D_r_lin_std)/(res_df.N**0.5), 
                  np.exp(res_df.D_r_lin_mean + res_df.D_r_lin_std)/(res_df.N**0.5)],
            ecolor='k', capsize=2)
ax.errorbar(res_df.Tpf_min, np.exp(res_df.D_or_lin_mean), 
        ls='-', marker='o', label=r'Ortho-radial',
        yerr=[np.exp(res_df.D_or_lin_mean - res_df.D_or_lin_std)/(res_df.N**0.5), 
              np.exp(res_df.D_or_lin_mean + res_df.D_or_lin_std)/(res_df.N**0.5)],
        ecolor='k', capsize=2)
ax.set_ylabel(r'$D\ (\mu m^2/s)$')
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.legend(loc='upper right')
ax.grid()

ax = axes[1]
# ax.plot(res_df.Tpf_min, df_Diffusion.k_full, ls='-', marker='o', label=r'All $\Delta t$')
# ax.plot(res_df.Tpf_min, df_Diffusion.k_lowDt, ls='-', marker='o', label=r'$\Delta t \leq 0.5s$')
# ax.plot(res_df.Tpf_min, df_Diffusion.k_highDt, ls='-', marker='o', label=r'$\Delta t \geq 1s$')
ax.errorbar(res_df.Tpf_min, res_df.k_r_full_mean, 
        ls='-', marker='o', label=r'Radial',
        yerr=[(res_df.k_r_highDt_mean - res_df.k_r_highDt_std)/(res_df.N**0.5), 
              (res_df.k_r_highDt_mean + res_df.k_r_highDt_std)/(res_df.N**0.5)],
        ecolor='k', capsize=2)
ax.errorbar(res_df.Tpf_min, res_df.k_or_full_mean, 
        ls='-', marker='o', label=r'Ortho-radial',
        yerr=[(res_df.k_or_highDt_mean - res_df.k_or_highDt_std)/(res_df.N**0.5), 
              (res_df.k_or_highDt_mean + res_df.k_or_highDt_std)/(res_df.N**0.5)],
        ecolor='k', capsize=2)
ax.set_ylabel(r'$\alpha$')
ax.set_xticks(res_df['Tpf_min'].values)
ax.set_xticklabels(res_df['Tpf_min'].values, rotation = 20)
ax.set_xlabel('Tpf (min)')
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.legend(loc='upper right')
ax.grid()

plt.show()


# %%%% Dev MRSD (Pairwise MSD)

pm.setGraphicOptions('print')

Nframe_per_chunk = 100
Nc = nbimages//Nframe_per_chunk
Fi = np.arange(0, 2000, step = Nframe_per_chunk)
Ff = Fi + Nframe_per_chunk
Dict_TRanges = {}

for ii in [2]: # len(dfNames)
    dfName = dfNames[ii]
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    df.particle = df.particle.astype(int)
    df.frame = df.frame.astype(int)
    
    PIDs = df.particle.unique()
    dict_fi_ff = {}
    
    for pid in PIDs:
        pfi = np.min(df[df['particle'] == pid]['frame'].values) - 1
        pff = np.max(df[df['particle'] == pid]['frame'].values) - 1
        dict_fi_ff[pid] = [pfi, pff]
    
    
for ii in [2]: # len(dfNames)
    dfName = dfNames[ii]
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    df.particle = df.particle.astype(int)
    df.frame = df.frame.astype(int)
    
    PIDs = df.particle.unique()
    for jj in range(Nc):
        fi, ff = Fi[jj], Ff[jj]
        TRange = []
        # print('bounds', fi, ff)
        
        for pid in PIDs[:]:
            pfi, pff = dict_fi_ff[pid]
            if (pfi <= fi) and (ff-1 <= pff):
                # print(pfi, pff)
                TRange.append(pid)
                
        Dict_TRanges[fi] = TRange
        
ncols = 5
nrows = ((Nc - 1)//ncols) + 1

fig, axes = plt.subplots(nrows, ncols, figsize=(2.0*ncols, 2.0*nrows))
axes_f = axes.flatten()

for k in range(Nc):
    print(k)
    ax = axes_f[k]
    fi, ff = Fi[k], Ff[k]
    TRange = Dict_TRanges[fi]
    df_TRange = df[df['particle'].apply(lambda x : x in TRange)]
    sns.lineplot(ax=ax,
                 data=df_TRange, x=df_TRange.x, y=df_TRange.y,
                 hue=df_TRange.particle, palette=pm.cL_Set21,
                 ls='-', lw=1, 
                 legend=False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # for p in df_chunk.particle.unique():
    #     df_chunk_p = df_chunk[df_chunk['particle'] == p]
    #     ax.plot(df_chunk_p.x, df_chunk_p.y, ls='-', lw=1)



for k in range(Nc):
    print(k)
    ax = axes_f[k]
    fi, ff = Fi[k], Ff[k]
    TRange = Dict_TRanges[fi]
    df_TRange = df[df['particle'].apply(lambda x : x in TRange)]


# %%%% Test Delaunay Dist

import numpy as np
import matplotlib.pyplot as plt


points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1], 
                   [0.25, 0.25], [0.25, -0.25], [0.2, 0.4], [0.8, 0.95],
                   [0.1, -0.15], [0.1, 0.98], [0.95, 0.05], [1.2, 0.87],])
tri = Delaunay(points)
indptr, indices = tri.vertex_neighbor_vertices
k = 4
neigh_k = indices[indptr[k]:indptr[k+1]]

XY = points

for k in range(len(points)):
    neigh_k = indices[indptr[k]:indptr[k+1]]
    X, Y = XY[k]
    X_neigh, Y_neigh = XY[neigh_k, 0], XY[neigh_k, 1]
    XY_p = np.array([[X, Y]]).T
    XY_neigh = np.array([X_neigh, Y_neigh])
    Dist_neigh = np.power(np.sum((XY_neigh - XY_p)**2, axis=1), 0.5)
    
fig, axes = plt.subplots(1, 2, figsize = (7, 3.5))

ax = axes[0]
ax.triplot(points[:,0], points[:,1], tri.simplices)
ax.plot(points[:,0], points[:,1], 'o')
ax.plot(points[k,0], points[k,1], 'ro')
ax.plot(points[neigh_k,0], points[neigh_k,1], 'ko')

ax = axes[1]
# ax.triplot(points[:,0], points[:,1], tri.simplices)
ax.plot(points[:,0], points[:,1], 'o')

plt.show()

# Extract all edges from each triangle
edges = np.vstack([
    tri.simplices[:, [0, 1]],
    tri.simplices[:, [1, 2]],
    tri.simplices[:, [2, 0]]
])

# Sort indices within each edge to make (i, j) and (j, i) identical
edges = np.sort(edges, axis=1)

# Remove duplicate edges
edges = np.unique(edges, axis=0)

# Convert to a Python list of pairs
edges = np.array(edges)

print(edges)

dists = []
for e in edges:
    i, j = e
    d = np.power(np.sum((points[i]-points[j])**2), 0.5)
    dists.append(d)

dists = np.array(dists)
idx_close_neighbours = (dists < 0.3)
edges_close_neighbours = edges[idx_close_neighbours]

for e in edges_close_neighbours:
    i, j = e
    ax.plot([points[i,0], points[j,0]], [points[i,1], points[j,1]], 'g-')

print(dists)




pm.setGraphicOptions(mode='screen')

for ii in [2]:   
    dfName = dfNames[ii]
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    title = '_'.join(dfName.split('_')[:5])
    
    for jj in [1500]:
        df_j = df[df['frame'] == jj+1]
        title_j = title + f' - Frame no {jj+1:.0f}'
        fName_j = title + f'_Fn{jj+1:.0f}_PCF.png'
        XX, YY = df_j.x*UmPerPix, df_j.y*UmPerPix
        XXYY = np.array([XX, YY]).T
        
        tri = Delaunay(XXYY)
        indptr, indices = tri.vertex_neighbor_vertices
        
        short_edges = tri_to_short_edges(tri, XXYY, 3)
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout='compressed')
        axes_f = axes.flatten()
        
        ax = axes_f[0]
        ax.set_aspect('equal', adjustable='box')
        ax.plot(XX, YY, ls='', marker='.')
        ax.set_xlabel(r'$x\ (\mu m)$')
        ax.set_ylabel(r'$y\ (\mu m)$')
        ax.set_xlim([0, 511*UmPerPix])
        ax.set_ylim([0, 511*UmPerPix])
        
        
        ax = axes_f[1]
        # ax.triplot(XXYY[:, 0], XXYY[:, 1], tri.simplices)
        ax.plot(XXYY[:, 0], XXYY[:, 1], 'o')
        
        for e in short_edges:
            i, j = e
            ax.plot([XXYY[i,0], XXYY[j,0]], [XXYY[i,1], XXYY[j,1]], 'g-')
        
        plt.show()
        

# %%%% MSRD functions

def get_pairs_for_TRanges_Delaunay(df, SCALE, FPS, Nframes,
                                  len_TRanges = 200, delta_TRanges = -1,
                                  dist_th_um = 5):
    df.frame = df.frame.astype(int)
    df.particle = df.particle.astype(int)
    dist_th = dist_th_um * SCALE
    
    if delta_TRanges < 0:
        delta_TRanges = len_TRanges
    FI = np.arange(0, Nframes, step=delta_TRanges)
    FF = FI + len_TRanges
    valid = (FF <= Nframes)
    if valid[-1]:
        pass
    else:
        i_stop = ufun.findFirst(True, (FF>Nframes))
        FI = FI[:i_stop]
        FF = FF[:i_stop]
    
    dict_TRanges2particles = {f'{fi}_{ff}':{'pid':[], 'xm':[], 'ym':[]} \
                              for fi, ff in zip(FI, FF)}
    dict_TRanges2pairs = {f'{fi}_{ff}':[] for fi, ff in zip(FI, FF)}
    
    PIDs = df.particle.unique()
    for pid in PIDs:
        pfi = np.min(df[df['particle'] == pid]['frame'].values) - 1
        pff = np.max(df[df['particle'] == pid]['frame'].values) - 1
        
        for fi, ff in zip(FI, FF):
            if (pfi <= fi) and (ff <= pff):
                df_p = df[df['particle'] == pid]
                df_p_TR = df_p[df_p['frame'].apply(lambda x : fi <= (x-1) < ff)]
                
                xm = np.median(df_p_TR['x'].values)
                ym = np.median(df_p_TR['y'].values)
                dict_TRanges2particles[f'{fi}_{ff}']['pid'].append(pid)
                dict_TRanges2particles[f'{fi}_{ff}']['xm'].append(xm)
                dict_TRanges2particles[f'{fi}_{ff}']['ym'].append(ym)
    
    for TRange in dict_TRanges2particles.keys():
        df_parts = pd.DataFrame(dict_TRanges2particles[TRange])
        XY = np.array([df_parts['xm'].values[:],
                       df_parts['ym'].values[:]]).T
        
        tri = Delaunay(XY)
        edges_short, _ = tri_to_short_edges(tri, XY, dist_th)
        close_pairs = df_parts['pid'].values[edges_short]
        
        dict_TRanges2pairs[TRange] = np.array(close_pairs)        
            
    return(dict_TRanges2pairs)




def get_pairs_for_TRanges(df, SCALE, FPS, Nframes,
                          len_TRanges = 200, delta_TRanges = -1,
                          dist_th_um = 5):
    df.frame = df.frame.astype(int)
    df.particle = df.particle.astype(int)
    dist_th = dist_th_um * SCALE
    
    if delta_TRanges < 0:
        delta_TRanges = len_TRanges
    FI = np.arange(0, Nframes, step=delta_TRanges)
    FF = FI + len_TRanges
    valid = (FF <= Nframes)
    if valid[-1]:
        pass
    else:
        i_stop = ufun.findFirst(True, (FF>Nframes))
        FI = FI[:i_stop]
        FF = FF[:i_stop]
    
    dict_TRanges2particles = {f'{fi}_{ff}':{'pid':[],'xm':[],'ym':[]} \
                              for fi, ff in zip(FI, FF)}
    dict_TRanges2pairs = {f'{fi}_{ff}':[] for fi, ff in zip(FI, FF)}
    
    PIDs = df.particle.unique()
    for pid in PIDs:
        pfi = np.min(df[df['particle'] == pid]['frame'].values) - 1
        pff = np.max(df[df['particle'] == pid]['frame'].values) - 1
        
        for fi, ff in zip(FI, FF):
            if (pfi <= fi) and (ff-1 <= pff):
                xm = np.median(df[df['particle'] == pid]['x'].values)
                ym = np.median(df[df['particle'] == pid]['y'].values)
                dict_TRanges2particles[f'{fi}_{ff}']['pid'].append(pid)
                dict_TRanges2particles[f'{fi}_{ff}']['xm'].append(xm)
                dict_TRanges2particles[f'{fi}_{ff}']['ym'].append(ym)
    
    for TRange in dict_TRanges2particles.keys():
        df_parts = pd.DataFrame(dict_TRanges2particles[TRange])
        listPairs = []
        while len(df_parts)>1:
            p1 = df_parts['pid'].values[0]
            XY1 = np.array([df_parts['xm'].values[0],
                            df_parts['ym'].values[0]])
            XYothers = np.array([df_parts['xm'].values[1:],
                                 df_parts['ym'].values[1:]]).T
            dists = np.power((np.sum((XYothers - XY1)**2, axis=1)), 0.5)
            min_d = np.min(dists)
            if min_d > dist_th:
                idx_to_drop = df_parts[(df_parts["pid"] == p1)].index
                df_parts.drop(axis=0, index=idx_to_drop, inplace=True)
            else:
                idx_min = np.argmin(dists) + 1
                p2 = df_parts['pid'].values[idx_min]
                listPairs.append((p1, p2))
                idx_to_drop = df_parts[(df_parts["pid"] == p1) | (df_parts["pid"] == p2)].index
                df_parts.drop(axis=0, index=idx_to_drop, inplace=True)
                # except:
                #     print(df_parts)
                
        dict_TRanges2pairs[TRange] = np.array(listPairs)
            
    return(dict_TRanges2pairs)

# Test run
dfName = dfNames[2]
df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')

UmPerPix = cd.UmPerPix_60X_W1
SCALE = 1/UmPerPix
Nframes = 2000
FPS = 10

top = time.time()
dict_TRanges2pairs_N = get_pairs_for_TRanges(df, SCALE, FPS, Nframes,
                          len_TRanges = 100, delta_TRanges = -1,
                          dist_th_um = 5)
print(f'Dt = {time.time()-top:.3f} s')

top = time.time()
dict_TRanges2pairs_D = get_pairs_for_TRanges_Delaunay(df, SCALE, FPS, Nframes,
                          len_TRanges = 100, delta_TRanges = -1,
                          dist_th_um = 5)
print(f'Dt = {time.time()-top:.3f} s')

# %%%% MSRD functions II


def get_pairsXY_byTRange(df, pairs, TRange):
    df.frame = df.frame.astype(int)
    df.particle = df.particle.astype(int)
    
    FI, FF = np.array(TRange.split('_')).astype(int)
    # T_array = np.arange(FI, FF)-1 
    T_array_shifted = np.arange(0, FF-FI)
    
    ids_in_pairs = np.unique(pairs.flatten())

    df_f = df
    df_f = df_f[df_f['frame'].apply(lambda x : FI <= (x-1) < FF)]
    df_f = df_f[df_f['particle'].apply(lambda x : x in ids_in_pairs)]
    
    # pair_2_pairId = {pairs[i] : i for i in range(len(pairs))}
    # pairs[pairId] = pair
    # pair_2_pairId[pair] = pairId
    
    # df_pairs = pd.DataFrame({'pair_id':[],'x':[],'y':[],'frame':[],})
    list_df_pairs = []
    
    for i in range(len(pairs)):
        pair = pairs[i]
        # i = pairId
        id1, id2 = pair
        idx1, idx2 = (df_f['particle']==id1), (df_f['particle']==id2)
        Xpair = df_f[idx2]['x'].values-df_f[idx1]['x'].values
        Ypair = df_f[idx2]['y'].values-df_f[idx1]['y'].values
        
        N = len(T_array_shifted)
        
        # df_pairs = pd.concat([df_pairs, pd.DataFrame(
        #                                              {'pair_id':np.ones(N, dtype=int)*i,
        #                                               'x':Xpair, 'y':Ypair,
        #                                               'frame':T_array_shifted,}
        #                                              )],
        #                      axis=0)
        list_df_pairs.append(pd.DataFrame({'pair_id':np.ones(N, dtype=int)*i,
                                           'x':Xpair, 'y':Ypair,
                                           'frame':T_array_shifted + 1,}
                                          ))
        
    df_pairs = pd.concat(list_df_pairs, axis=0)
    return(df_pairs)

pm.setGraphicOptions(mode='screen')

TRanges = np.array(list(dict_TRanges2pairs_D.keys()))
dict_TRanges2pairMSD = {}



for TRange in TRanges:

    pairs = dict_TRanges2pairs_D[TRange]

    # top = time.time()
    df_pairs = get_pairsXY_byTRange(df, pairs, TRange)
    # print(f'Dt = {time.time()-top:.3f} s')
    
    # top = time.time()
    res_pair_emsd = tp.motion.emsd(df_pairs.rename(columns={'pair_id':'particle'}), 
                                   UmPerPix, FPS, max_lagtime=50).reset_index()
    # print(f'Dt = {time.time()-top:.3f} s')

    dict_TRanges2pairMSD[TRange] = res_pair_emsd

    ax.plot(res_pair_emsd.lagt, res_pair_emsd.msd, ls='', marker='.', label=TRange)


fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_xscale('log')
ax.set_yscale('log')

for TRange in TRanges[4:]:
    res_pair_emsd = dict_TRanges2pairMSD[TRange]
    ax.plot(res_pair_emsd.lagt, res_pair_emsd.msd, ls='', marker='.', label=TRange)
    
ax.grid()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()

# %%%% Functions to define ROI

testDir = "C://Users//Joseph//Desktop//IntraCellTracking//26-09-30_FastAcq-Channel_Fec_NB-Yolk//D2"
# imgFile = "26-09-30_D1_PostF_45min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2"
imgFile = "26-09-30_D2-R_PostF_53min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2"
imgPath = os.path.join(testDir, imgFile)

pm.setGraphicOptions(mode='print')

ii = 2
tt = range(0, nbimages, 100)
img = ufun.load_stack_region(imgPath, time_indices=tt)

nT, nY, nX = img.shape
img_max = np.max(img, axis = 0)

# binarize = True
k_th = 1.0
zero_padding = 10

# 1. Binarize
th1 = skm.filters.threshold_li(img_max) * k_th

# img_min = ndi.binary_closing(img_min, iterations=5)
img_bin = (img_max > th1)
img_bin = ndi.binary_opening(img_bin, iterations = 5)
img_bin = ndi.binary_fill_holes(img_bin)

FoundContours = skm.measure.find_contours(img_bin, 0.5)
MoreContours = FoundContours[:]
Nr = 2
for k in range(Nr):
    img_bin_bis = ndi.binary_erosion(img_bin, iterations=k+1)
    MoreContours += skm.measure.find_contours(img_bin_bis, 0.5)

longestContour = FoundContours[np.argmax([len(c) for c in FoundContours])]
x = longestContour[:, 0]
y = longestContour[:, 1]   
SIGMA = 5
x_smooth = gaussian_filter1d(x, sigma=SIGMA, mode="wrap")
y_smooth = gaussian_filter1d(y, sigma=SIGMA, mode="wrap")
smooth_contour = np.column_stack((x_smooth, y_smooth))

concat_contours = np.concatenate(MoreContours)
points = MultiPoint(concat_contours[:,::-1])
x_ch, y_ch = points.convex_hull.exterior.xy

ALPHA = 0.1
alpha_shape = alphashape.alphashape(concat_contours[:,::-1], ALPHA)
x_as, y_as = alpha_shape.exterior.xy

inner_shape = alpha_shape.buffer(-2.5/UmPerPix)
x_is, y_is = inner_shape.exterior.xy

# list_geoms = list(alpha_shape.geoms)
# for poly in list_geoms:
#     x_as, y_as = alpha_shape.exterior.xy


fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
axes_f = axes.flatten()

ax = axes_f[0]
ax.set_aspect('equal', adjustable='box')
ax.imshow(img_max, cmap='gray')

ax = axes_f[1]
ax.set_aspect('equal', adjustable='box')
ax.imshow(img_bin, cmap='gray')

ax = axes_f[2]
ax.set_aspect('equal', adjustable='box')
for c in FoundContours:
    ax.plot(c[:, 1], c[:, 0], lw=1)
ax.plot(x_ch, y_ch, lw=1)
    
ax = axes_f[3]
ax.set_aspect('equal', adjustable='box')
for c in MoreContours:
    ax.plot(c[:, 1], c[:, 0], lw=1)



for i in [0, 1, 4]:
    ax = axes_f[i]
    ax.set_aspect('equal', adjustable='box')
    ax.plot(x_as, y_as, lw='0.75')
    ax.plot(x_is, y_is, lw='0.75')

# list_geoms = list(alpha_shape.geoms)
# for poly in list_geoms:
#     x_as, y_as = poly.exterior.xy
#     ax.plot(x_as, y_as, lw='0.75')

plt.show()

# %%% 3. Tracking and structure analysis

# %%%% Import tracks & analyse shape of explored zone

idx_films = [0, 2, 3, 5, 7]

pm.setGraphicOptions(mode='screen')
fig, axes = plt.subplots(2, len(idx_films), figsize=(len(idx_films)*3, 6),
                         layout='compressed')
colors = pm.cL_Set21

dict_res = {'label':[],
            'AngleDiffs':[]}

GEOM_DATA = []

df_centers = pd.read_csv(os.path.join(srcDir, 'OrganizingCenters.csv'), sep=';')


for k, ii in enumerate(idx_films):
    X_MTcenter, Y_MTcenter = df_centers.loc[ii, 'xc'], df_centers.loc[ii, 'yc']
    print(X_MTcenter, Y_MTcenter)

    dfName = dfNames[ii]
    title = '_'.join(dfName.split('_')[1:3])
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    Lp = df['particle'].unique().astype(int)
    dict_geom = {'particle':[],
                 'np':[],
                 'xc':[],
                 'yc':[],
                 'theta':[],
                 'L':[],
                 'l':[],
                 'AR':[],
                 'phi':[],}
    
    ax = axes[0, k]
    ax.set_xlim([0, 511])
    ax.set_ylim([0, 511])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    
    for j in Lp[:]:
        c = colors[j%len(colors)]
        df_j = df[df['particle']==j]
        # x, y = 
        points = np.array([[x, y] for (x, y) in zip(df_j.x, df_j.y)])
        
        MPoints = MultiPoint(points)
        Hull = MPoints.convex_hull
        xy_ch = np.array([xy for xy in Hull.exterior.coords[1:]])
        xc, yc = Hull.centroid.coords[0]
        
        out = ufun.fit_ellipse_fixedCenter(xy_ch[:, 0], xy_ch[:, 1], 
                                           xc, yc, mode='cartesian')
        _, _, a, b, phi = out
        
        if j%10 == 0:
            aa = np.linspace(0, 2*np.pi, 360)
            xp, yp = ufun.get_ellipse_xy(xc, yc, a, b, phi, aa=aa)
            # ax.plot(xy_ch[:, 0], xy_ch[:, 1],
            #         c=c, marker='.', ls='', ) #ls='-', lw=1)
            ax.plot(xp, yp, c=c, ls='-', lw=0.5)
        
        
        
        theta = np.atan2(yc - Y_MTcenter, xc - X_MTcenter)
        theta_deg = theta * 180/np.pi
        
        dotprod = np.cos(theta)*np.cos(phi) + np.sin(theta)*np.sin(phi)
        if dotprod < 0:
            phi = phi - np.sign(phi) * np.pi
            
        phi_deg = phi * 180/np.pi
        # print(phi_deg, theta_deg)
        
        L, l = max(a, b), min(a, b)
        AR = L/l
        
        dict_geom['particle'].append(j)
        dict_geom['np'].append(len(df_j))
        dict_geom['xc'].append(xc)
        dict_geom['yc'].append(yc)
        dict_geom['theta'].append(theta)
        dict_geom['L'].append(L)
        dict_geom['l'].append(l)
        dict_geom['AR'].append(AR)
        dict_geom['phi'].append(phi)
        
        # # create a minimum rotated rectangle containing all the points
        # points = np.array(hull_xy).T 
        # multipoints = MultiPoint(points)
        # polygon = multipoints.minimum_rotated_rectangle
        # Xr, Yr = polygon.exterior.coords.xy
        # ax.plot(Xr, Yr, 'k-', lw=0.5)
        
        # d1 = ((Xr[0]-Xr[1])**2 + (Yr[0]-Yr[1])**2)**0.5
        # d2 = ((Xr[2]-Xr[1])**2 + (Yr[2]-Yr[1])**2)**0.5
        # # print(d1, d2)
        # L, l = max(d1, d2), min(d1, d2)
        # AR = L/l
        
        # V1 = (Xr[1]-Xr[0], Yr[1]-Yr[0])
        # theta = np.atan2(V1[1], V1[0])
        # print(theta*180/np.pi
    
    plt.show()
    
    
    df_geom_raw = pd.DataFrame(dict_geom)
    GEOM_DATA.append(df_geom_raw)
    
    
    df_geom = df_geom_raw[df_geom_raw['AR'] > 1.5]
    # N = len(df_geom)
    
    GEOM_DATA.append(df_geom_raw)
    
    df_geom['theta_bin'] = (df_geom['theta'].values * 18/np.pi).astype(int) * 10 + 5
    
    C = np.cos(df_geom['theta'].values) * np.cos(df_geom['phi'].values) + \
        np.sin(df_geom['theta'].values) * np.sin(df_geom['phi'].values)
    DA = np.acos(C)
    
    ax = axes[1, k]
    ax.hist(DA, bins=20)
    ax.set_ylabel('N trajectories')
    ax.set_xlabel(r'$|\theta - \phi|$ (rad)')
    ax.set_xticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
    ax.set_xticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
    ax.grid()
    
    
    dict_res['label'].append(title)
    dict_res['AngleDiffs'].append(DA)

plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(6, 6),
#                          layout='compressed')
# for i, L in enumerate(dict_res['label']):
#     ax.hist(dict_res['AngleDiffs'][i], bins=30, 
#             density=True, alpha=1, label=L, histtype='step')
# ax.legend()
# ax.set_ylabel('Frequency')
# ax.set_xlabel(r'$|\theta - \phi|$ (rad)')
# plt.show()


# %%%% Plot the geometry

df_geom = pd.DataFrame(dict_geom)
df_geom['theta_bin'] = (df_geom['theta'].values * 18/np.pi).astype(int) * 10 + 5

C = np.cos(df_geom['theta'].values) * np.cos(df_geom['phi'].values) + np.sin(df_geom['theta'].values) * np.sin(df_geom['phi'].values)
DA = np.acos(C)

fig, ax = plt.subplots(1, 1, layout='compressed')
ax.hist(DA, bins=40)
ax.set_ylabel('N trajectories')
ax.set_xlabel(r'$|\theta - \phi|$ (rad)')
ax.set_xticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
ax.set_xticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
ax.grid()
plt.show()

# %%%% Plot the inferred center ?

df_geom = GEOM_DATA[0]

im = ufun.load_stack_region(tifPaths[2], time_indices=[1000])[0]

Filters = [(df_geom['AR'] > 4)]

df_f = pm.filterDf(df_geom, Filters).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes[0]
ax.set_aspect('equal', adjustable='box')
ax.set_xlim([0, 511])
ax.set_ylim([0, 511])
ax.imshow(im, cmap='gray')

ax = axes[1]
ax.set_aspect('equal', adjustable='box')
ax.set_xlim([0, 511])
ax.set_ylim([0, 511])
ax.scatter(df_f['xc'], df_f['yc'], s=6, marker='.', color='c', alpha = 0.05)

for i in range(len(df_f)):
    xc, yc, phi = df_f.loc[i, 'xc'], df_f.loc[i, 'yc'], df_f.loc[i, 'phi']
    S = np.tan(phi)
    ax.axline((xc, yc), slope=S, linestyle='-', color='c', alpha = 0.04, lw=10)


plt.show()



# %%%% Import tracks & run pcf2d

pm.setGraphicOptions(mode='screen')

for ii in [2]:   
    dfName = dfNames[ii]
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    title = '_'.join(dfName.split('_')[:5])
    
    for jj in [1750]:
        df_j = df[df['frame'] == jj+1]
        im = ufun.load_stack_region(tifPaths[ii], time_indices=[jj])[0]
        title_j = title + f' - Frame no {jj+1:.0f}'
        fName_j = title + f'_Fn{jj+1:.0f}_PCF.png'
        XX, YY = df_j.x*UmPerPix, df_j.y*UmPerPix
        XXYY = np.array([XX, YY]).T
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), layout='compressed')
        axes_f = axes.flatten()
        
        #### EQUALIZE
        p1, p99 = np.percentile(im, (1, 99))
        im_pt = skm.exposure.rescale_intensity(im, in_range=(p1, p99))
        
        ax = axes_f[0]
        ax.set_aspect('equal', adjustable='box')
        ax.imshow(im_pt, cmap='gray')
        ax.plot(XX/UmPerPix, YY/UmPerPix, ls='', marker='.',  markersize=2, color='red')
        ax.set_xlabel(r'$x\ (px)$')
        ax.set_ylabel(r'$y\ (px)$')
        ax.set_xlim([0, 511])
        ax.set_ylim([0, 511])
        
        ax = axes_f[1]
        ax.set_aspect('equal', adjustable='box')
        ax.plot(XX, YY, ls='', marker='.')
        ax.set_xlabel(r'$x\ (\mu m)$')
        ax.set_ylabel(r'$y\ (\mu m)$')
        ax.set_xlim([0, 511*UmPerPix])
        ax.set_ylim([0, 511*UmPerPix])
        
        array_positions = XXYY
        bins_distances = np.arange(0, 15, 0.2)
        
        out = tbsa.pcf2d(array_positions, bins_distances, 
                  coord_border=None, coord_holes=None, fast_method=False,
                  show_timing=False, plot=False, full_output=False)
        
        (g_of_r_normalized, radii) = out
        N_of_r_normalized = 2*np.pi * np.array([np.sum(radii[:k]*g_of_r_normalized[:k]) for k in range(len(g_of_r_normalized))])
        
        ax = axes_f[2]
        ax.plot(radii, g_of_r_normalized, color=pm.cL_Set2[0])
        ax.set_xlabel(r'$r\ (\mu m)$')
        ax.set_ylabel(r'$G(r)$')
        # ax.grid()
        ax.axhline(1, linestyle=':', color='gray')

        
        ax = axes_f[3]
        ax.set_xscale('log')
        ax.set_yscale('log')
        idx_start_fit = 30
        
        ax.plot(radii[:], N_of_r_normalized[:],
                'k.', label='Data')
        
        xfit = np.log(radii[idx_start_fit:])
        yfit = np.log(N_of_r_normalized[idx_start_fit:])
        parms, res = ufun.fitLineHuber(xfit, yfit)
        b, a = parms
        k, A = a, np.exp(b)
        xx = radii[idx_start_fit:]
        ax.plot(xx, A*xx**k, lw=1.5,
                label=r'Fit $y=Ax^k$' + f'\nk={k:.2f}')
        ax.legend()
        ax.grid()
        ax.set_xlabel(r'$r\ (\mu m)$')
        ax.set_ylabel(r'$N(r)$')
        
        fig.suptitle(title_j + ' - points spatial stats')
        
        # fig.suptitle(title_j)
        # figpath = os.path.join(dstDir, fName_j)
        # fig.savefig(figpath, dpi=500, )
        

# %%%% Import image, treat and & run pcf2d

pm.setGraphicOptions(mode='screen')

for ii in [2]:   
    dfName = dfNames[ii]
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    title = '_'.join(dfName.split('_')[:5])
    
    for jj in [1750]:
        # df_j = df[df['frame'] == jj+1]
        im = ufun.load_stack_region(tifPaths[ii], time_indices=[jj])[0]
        title_j = title + f' - F {jj+1:.0f}'
        fName_j = title + f'_Fn{jj+1:.0f}_PCF.png'
        # XX, YY = df_j.x*UmPerPix, df_j.y*UmPerPix
        # XXYY = np.array([XX, YY]).T
        
        #### EQUALIZE
        p1, p99 = np.percentile(im, (1, 99))
        im_pt = skm.exposure.rescale_intensity(im, in_range=(p1, p99))   
            
        #### FILTER
        k = 3
        im_pt = cv2.medianBlur(im_pt, k)
        
        #### TOP_HAT
        filterSize = (12, 12)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
        im_pt = cv2.morphologyEx(im_pt, cv2.MORPH_TOPHAT, kernel)
        p1, p99 = np.percentile(im_pt, (1, 99))
        im_pt = skm.exposure.rescale_intensity(im_pt, in_range=(p1, p99))

        #### BINARIZE
        th = skm.filters.threshold_li(im_pt)
        im_bin = (im_pt >= th)
        k = 3
        im_bin = ndi.binary_opening(im_bin)
        
        # PLOT PRETREATMENT
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='compressed')
        axes_f = axes.flatten()
        ax = axes_f[0]
        ax.imshow(im, cmap='gray')
        ax.set_xlabel(r'$x\ (px)$')
        ax.set_ylabel(r'$y\ (px)$')
        
        ax = axes_f[1]
        ax.imshow(im_pt, cmap='gray')
        ax.set_xlabel(r'$x\ (px)$')
        ax.set_ylabel(r'$y\ (px)$')
        
        ax = axes_f[2]
        ax.imshow(im_bin, cmap='gray')
        ax.set_xlabel(r'$x\ (px)$')
        ax.set_ylabel(r'$y\ (px)$')
        
        fig.suptitle(title_j + ' - pretreatment')
        plt.show()
        
        
        # COMPUTE SPATIAL STATS
        M = 150
        r_values = np.linspace(1, 300, 300)
        selected_points, all_points = tbsa.sample_white_pixels(im_bin, M=M, seed=40)
        
        K_local, K_global = tbsa.ripley_K_sampled_references(selected_points,
                                                             all_points,
                                                             im_bin.shape,
                                                             r_values)
        L_global = (K_global/np.pi)**0.5
        
        spline_Kg = make_splrep(r_values, K_global, s=6)
        Kg_splinified = spline_Kg(r_values)
        spline_Kg_der = spline_Kg.derivative(nu=1)
        G = spline_Kg_der(r_values) * 1/(2*np.pi*r_values)
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), layout='compressed')
        axes_f = axes.flatten()
        ax = axes_f[0]
        ax.imshow(im_bin, cmap='gray')
        ax.plot(
            selected_points[:,0], selected_points[:,1],
            ls='', marker='.',  markersize=3, color='red',
        )
        ax.set_xlabel(r'$x\ (px)$')
        ax.set_ylabel(r'$y\ (px)$')
        
        ax = axes_f[1]
        for i in range(M):
            ax.plot(r_values, K_local[i, :], alpha=0.05)
        ax.grid()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Radius r")
        ax.set_ylabel("Local Ripley K(r)")
        
        ax = axes_f[2]
        ax.plot(r_values, K_global, ls='', marker='.', label='K(r)')
        ax.plot(r_values, Kg_splinified, linewidth=1.5, ls='-', label='spline rep', color='k')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Radius r")
        ax.set_ylabel("Ripley K(r)")
        idx_start_fit = 30
        xfit = np.log(r_values[idx_start_fit:])
        yfit = np.log(K_global[idx_start_fit:])
        parms, res = ufun.fitLineHuber(xfit, yfit)
        b, a = parms
        k, A = a, np.exp(b)
        xx = r_values[idx_start_fit:]
        ax.plot(xx, A*xx**k, lw=1.5, ls='--', color='red',
                label=r'Fit $y=Ax^k$' + f'\nk={k:.2f}')        
        ax.legend()
        ax.grid()
        

        ax = axes_f[3]
        ax.axhline(1, linestyle='--', color='gray', alpha=1)
        ax.plot(r_values, G, linewidth=2, ls='-', label='g(r)')
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_xlabel("Radius r")
        ax.set_ylabel(r"$g(r)$")
        
        # ax = axes_f[4]
        # ax.plot(r_values, L_global - r_values, linewidth=2, label='K(r)')
        # # ax.set_xscale('log')
        # # ax.set_yscale('log')
        # ax.set_xlabel("Radius r")
        # ax.set_ylabel(r"$\sqrt{K(r)/\pi} - r$")

        fig.suptitle(title_j + ' - spatial stats')
        plt.show()

        
        # array_positions = XXYY
        # bins_distances = np.arange(0, 15, 0.2)
        
        # out = tbsa.pcf2d(array_positions, bins_distances, 
        #           coord_border=None, coord_holes=None, fast_method=False,
        #           show_timing=False, plot=False, full_output=False)
        
        # (g_of_r_normalized, radii) = out
        # N_of_r_normalized = 2*np.pi * np.array([np.sum(radii[:k]*g_of_r_normalized[:k]) for k in range(len(g_of_r_normalized))])
        
        # ax = axes_f[2]
        # ax.plot(radii, g_of_r_normalized, color=pm.cL_Set2[0])
        # ax.set_xlabel(r'$r\ (\mu m)$')
        # ax.set_ylabel(r'$G(r)$')
        # # ax.grid()
        # ax.axhline(1, linestyle=':', color='gray')

        
        # ax = axes_f[3]
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        # idx_start_fit = 30
        

        # ax.plot(radii[:], N_of_r_normalized[:],
        #         'k.', label='Data')
        
        # xfit = np.log(radii[idx_start_fit:])
        # yfit = np.log(N_of_r_normalized[idx_start_fit:])
        # parms, res = ufun.fitLineHuber(xfit, yfit)
        # b, a = parms
        # k, A = a, np.exp(b)
        # xx = radii[idx_start_fit:]
        # ax.plot(xx, A*xx**k, lw=1.5,
        #         label=r'Fit $y=Ax^k$' + f'\nk={k:.2f}')
        # ax.legend()
        # ax.grid()
        # ax.set_xlabel(r'$r\ (\mu m)$')
        # ax.set_ylabel(r'$N(r)$')
        
        # fig.suptitle(title_j)
        # figpath = os.path.join(dstDir, fName_j)
        # fig.savefig(figpath, dpi=500, )
        
        
# %%%% Import FAKE image, treat and & run pcf2d

pm.setGraphicOptions(mode='screen')

srcDir = os.path.join(up.Path_IntraCellTracking, '26-09-11_FakeImages')
fileNames = ['dots.tiff', 'branches.tiff', 'clustered_dots.tiff']
filePaths = [os.path.join(srcDir, fN) for fN in fileNames]

for ii in range(len(fileNames)):   
    fN = fileNames[ii]
    fP = filePaths[ii]
    
    # df_j = df[df['frame'] == jj+1]
    im = skm.io.imread(fP, as_gray=True)
    im = skm.util.img_as_ubyte(im)
    title_j = f'{fN}'
    fName_j = f'{fN}_KRF.png'
    # XX, YY = df_j.x*UmPerPix, df_j.y*UmPerPix
    # XXYY = np.array([XX, YY]).T
    
    #### EQUALIZE
    # p1, p99 = np.percentile(im, (1, 99))
    # im_pt = skm.exposure.rescale_intensity(im, in_range=(p1, p99))   
        
    #### FILTER
    # k = 3
    # im_pt = cv2.medianBlur(im_pt, k)
    
    #### TOP_HAT
    # filterSize = (12, 12)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    # im_pt = cv2.morphologyEx(im_pt, cv2.MORPH_TOPHAT, kernel)
    # p1, p99 = np.percentile(im_pt, (1, 99))
    # im_pt = skm.exposure.rescale_intensity(im_pt, in_range=(p1, p99))

    #### BINARIZE
    # th = skm.filters.threshold_li(im_pt)
    # im_bin = (im_pt >= th)
    im_bin = ndi.binary_opening(im)
    
    
    M = 150

    selected_points, all_points = tbsa.sample_white_pixels(im_bin, M=M, seed=40)
    
    r_values = np.linspace(1, 300, 300)
    
    K_local, K_global = tbsa.ripley_K_sampled_references(selected_points,
                                                         all_points,
                                                         im_bin.shape,
                                                         r_values)
    L_global = (K_global/np.pi)**0.5
    
    spline_Kg = make_splrep(r_values, K_global, s=6)
    Kg_splinified = spline_Kg(r_values)
    spline_Kg_der = spline_Kg.derivative(nu=1)
    G = spline_Kg_der(r_values) * 1/(2*np.pi*r_values)
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), layout='compressed')
    axes_f = axes.flatten()
    ax = axes_f[0]
    ax.imshow(im_bin, cmap='gray')
    ax.plot(
        selected_points[:,0], selected_points[:,1],
        ls='', marker='.',  markersize=3, color='red',
    )
    ax.set_xlabel(r'$x\ (\mu m)$')
    ax.set_ylabel(r'$y\ (\mu m)$')
    
    ax = axes_f[1]
    for i in range(M):
        ax.plot(r_values, K_local[i, :], alpha=0.05)
    ax.grid()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Radius r")
    ax.set_ylabel("Local Ripley K(r)")
    
    ax = axes_f[2]
    ax.plot(r_values, K_global, linewidth=2, label='K(r)')
    ax.plot(r_values, Kg_splinified, linewidth=1, ls='--', label='spline rep')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Radius r")
    ax.set_ylabel("Ripley K(r)")
    
    idx_start_fit = 30
    xfit = np.log(r_values[idx_start_fit:])
    yfit = np.log(K_global[idx_start_fit:])
    parms, res = ufun.fitLineHuber(xfit, yfit)
    b, a = parms
    k, A = a, np.exp(b)
    xx = r_values[idx_start_fit:]
    ax.plot(xx, A*xx**k, lw=1.5,
            label=r'Fit $y=Ax^k$' + f'\nk={k:.2f}')
    
    ax.legend()
    ax.grid()
    
    
    
    # ax = axes_f[3]
    # ax.plot(r_values, L_global - r_values, linewidth=2, label='K(r)')
    # # ax.set_xscale('log')
    # # ax.set_yscale('log')
    # ax.set_xlabel("Radius r")
    # ax.set_ylabel(r"$\sqrt{K(r)/\pi} - r$")
    
    ax = axes_f[3]
    ax.axhline(1, linestyle=':', color='gray', alpha=0.7)
    ax.plot(r_values, G, linewidth=2, ls='-', label='g(r)')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel("Radius r")
    ax.set_ylabel(r"$g(r)$")
    
    
    
    
    

    plt.show()
        
        



# %% -----------------------



