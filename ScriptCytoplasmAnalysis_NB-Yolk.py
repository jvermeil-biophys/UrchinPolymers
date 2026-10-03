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
    #### HAVE TO BE REDONE FOR NON SQUARE ROIs !!
    # xlabels = [f'{x*(N_pix)/M_boxes:.0f}' for x in xt]
    # ylabels = [f'{y*(N_pix)/M_boxes:.0f}' for y in yt]
    # xlabels = ['' for x in xt]
    # ylabels = ['' for y in yt]
    # ax.tick_params(axis='both', length=0,)

    # ax.set_xticks(xticks, labels=xlabels,) # rotation=-30, rotation_mode="xtick")
    # ax.set_yticks(yticks, labels=ylabels)
    # ax.spines[:].set_visible(False)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
        
    return(fig, ax)


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
# mainDir = os.path.join(up.Path_IntraCellTracking, '26-07-29_FastAcq_Fec_NB-Yolk')
# srcDir = os.path.join(mainDir, 'Crops')
# dstDir = os.path.join(mainDir, 'SPT_results')

# tifNames = [
#             '26-07-29_PostF_2min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
#             '26-07-29_PostF_6min30_Pos11_10fps_Texp100ms_CSU642_crop.tif',
#             '26-07-29_PostF_12min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
#             '26-07-29_PostF_20min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
#             '26-07-29_PostF_30min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
#             '26-07-29_PostF_45min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
#             '26-07-29_PostF_60min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
#             '26-07-29_PostF_70min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
#             ]

mainDir = os.path.join(up.Path_IntraCellTracking, '26-09-30_FastAcq-Channel_Fec_NB-Yolk')
srcDir = os.path.join(mainDir, 'D1')
dstDir = os.path.join(mainDir, 'SPT_results')

tifNames = ['26-09-30_D1_PreF_C1_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
            '26-09-30_D1_PreF_C2_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
            '26-09-30_D1_PreF_C3_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
            '26-09-30_D1_PostF_10min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
             '26-09-30_D1_PostF_13min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
             '26-09-30_D1_PostF_18min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
             '26-09-30_D1_PostF_25min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
             '26-09-30_D1_PostF_30min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
             '26-09-30_D1_PostF_35min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
             '26-09-30_D1_PostF_40min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
             '26-09-30_D1_PostF_40min_C4_20fps_Texp50ms_L20p2_CSU642.ome.tf2',
             '26-09-30_D1_PostF_45min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
             '26-09-30_D1_PostF_4min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
             '26-09-30_D1_PostF_52min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
             '26-09-30_D1_PostF_60min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
             '26-09-30_D1_PostF_65min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
             '26-09-30_D1_PostF_6min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
             '26-09-30_D1_PostF_70min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
             ]
             


tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
fileNames = [fN.split('.')[0] for fN in tifNames]

# rawTracksDir = 'TrackMate_raw_tracks'
# rawTracksNames = [fN + '_TmTracks.xml' for fN in tifPaths]
# rawTracksPaths = [os.path.join(dstDir, rawTracksDir, rtN)  for rtN in rawTracksNames]

# cleanTracksDir = 'Clean_tracks'
# cleanTracksNames = [fN + '_PyTracks.csv' for fN in tifPaths]
# cleanTracksPaths = [os.path.join(dstDir, cleanTracksDir, ctN)  for ctN in cleanTracksNames]

suffix_contour = '_cellContour'
suffix_mask = '_cellMask'
suffix_rawTracks = '_TmTracks'
suffix_cleanTracks = '_PyTracks'
suffix_globalMsd = '_GlobalMsd'
suffix_globalMsdFits = '_GlobalMsdFits'

#### Settings

UmPerPix = cd.UmPerPix_60X_W1
PixPerUm = 1/UmPerPix
nbimages = 2000
FPS = 20

# N_pix = 512
# C_pix = np.median(np.arange(N_pix)) # Center (pixels)
# L_um = N_pix*UmPerPix

max_lagtime = 50
lowDt_upper = 0.5
highDt_lower = 1.0


# %%%% First Analysis Block (CHANGE NAME)

#### Make and save cell contours and masks
# print('\n\n1. Contours step')
# for i in range(len(fileNames)):
#     tN, tP, fN = tifNames[i], tifPaths[i], fileNames[i]
#     print(i+1, len(fileNames), fN)
    
#     shape, dtype = ufun.tiff_inspect(tP)
#     nT = shape[0]
#     TT = range(0, nT, nT//100)
#     img = ufun.load_stack_region(tP, time_indices=TT)
    
#     Contour_cell, Mask_cell = tbca.make_NbYolkCell_contour_and_mask(img, PixPerUm,
#                                                                     mode = 'dark_background', 
#                                                                     PLOT = False)
    
#     contourFile = fN + suffix_contour + '.npy'
#     maskFile = fN + suffix_mask + '.npy'
#     np.save(os.path.join(srcDir, contourFile), Contour_cell)
#     np.save(os.path.join(srcDir, maskFile), Mask_cell)


#### Run Trackmate
print('\n\n2. Tracking step')
for i in range(len(fileNames)):
    tifPath, fN = tifPaths[i], fileNames[i]
    print(i+1, len(fileNames), fN)
    
    rawTrackName = fN + suffix_rawTracks + '.xml'
    maskFile = fN + suffix_mask + '.npy'
    Mask_cell = np.load(os.path.join(srcDir, maskFile))
    
    tbca.pretreat_and_track_NbYolk(tifPath, rawTrackName, dstDir, 
                                   Mask_cell = Mask_cell,
                                   PLOT = True, SAVEPLOT = True)


#### Import & format tracks
print('\n\n3. Tracks formatting step')
for i in range(len(fileNames)):
    tifPath, fN = tifPaths[i], fileNames[i]
    print(i+1, len(fileNames), fN)
    
    rawTrackName = fN + suffix_rawTracks + '.xml'
    cleanTrackName = fN + suffix_cleanTracks + '.csv'
    contourPath = os.path.join(srcDir, fN + suffix_contour + '.npy')
    
    rawTracks = tbca.import_TrackMate_tracks(os.path.join(dstDir, rawTrackName))
    Contour_cell = np.load(contourPath)
    
    tbca.rawTracks_2_cleanTracks(rawTracks, dstDir, cleanTrackName,
                                 Contour_cell, PixPerUm,
                                 edgeBuffer_cutoff = 2.5, nPoints_cuttoff = 30,
                                )

#### Import tracks, run trackpy.emsd, fit MSD
print('\n\n3. MSD conpute step')
for i in range(len(fileNames)):
    tifPath, fN = tifPaths[i], fileNames[i]
    print(i+1, len(fileNames), fN)
    
    rawTrackName = fN + suffix_rawTracks + '.xml'
    cleanTrackName = fN + suffix_cleanTracks + '.csv'
    msdName = fN + suffix_globalMsd + '.csv'
    msdFitsName = fN + suffix_globalMsdFits
    
    df = pd.read_csv(os.path.join(dstDir, cleanTrackName), sep='\t')
    
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
    
    ufun.dict2json(dict_MSDfits, dstDir, msdFitsName)
    
    
# %%%% Import MSD & plot
    
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

for i in range(len(fileNames)):
    tifPath, fN = tifPaths[i], fileNames[i]
    cleanTrackName = fN + suffix_cleanTracks + '.csv'
    msdName = fN + suffix_globalMsd + '.csv'
    msdFitsName = fN + suffix_globalMsdFits
    
    df = pd.read_csv(os.path.join(dstDir, cleanTrackName), sep='\t')
    res_emsd = pd.read_csv(os.path.join(dstDir, msdName), sep='\t')
    T, MSD = res_emsd['lagt'], res_emsd['msd']
    
    dict_MSDfits = ufun.json2dict(dstDir, msdFitsName)
    D_linear = dict_MSDfits['D_linear']
    k_full = dict_MSDfits['k_full']
    D_full = dict_MSDfits['D_full']
    k_lowDt = dict_MSDfits['k_lowDt']
    D_lowDt = dict_MSDfits['D_lowDt']
    k_highDt = dict_MSDfits['k_highDt']
    D_highDt = dict_MSDfits['D_highDt']
    
    Tc = (D_lowDt/D_highDt)**(1/(k_highDt-k_lowDt))  

    label = fN.split('_')[3]

    ax.plot(T, MSD, label=label, color=pm.cL_Set21[i], 
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
        
    
# %%%% MSRD functions


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
img_proj = np.max(img, axis = 0)

# binarize = True
k_th = 1.0
zero_padding = 10

# 1. Binarize
th1 = skm.filters.threshold_li(img_proj) * k_th

# img_min = ndi.binary_closing(img_min, iterations=5)
img_bin = (img_proj > th1)
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
ax.imshow(img_proj, cmap='gray')

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




# %% -----------------------



