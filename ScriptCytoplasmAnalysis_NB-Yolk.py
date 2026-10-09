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
import time
import alphashape
# import cv2

import numpy as np
import pandas as pd
import trackpy as tp
import skimage as skm
import seaborn as sns
import matplotlib as mpl
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

from shapely.geometry import MultiPoint
from scipy.spatial import ConvexHull, Delaunay

import Libs.PlotMaker as pm
import Libs.UrchinPaths as up
import Libs.CalibrationData as cd
import Libs.UtilityFunctions as ufun
import Libs.ToolboxCytoplasmAnalysis as tbca
import Libs.ToolboxStructureAnalysis as tbsa



# %% Helper functions

#### Analysis functions

def NByolk_analysis_sequence(mainDir, srcDir, dstDir, tifNames, Dict_Files_Suffix, 
                             Dict_Image_Settings, 
                             Dict_TrackMate_Settings, 
                             Dict_Analysis_Settings,
                             Do_Contours = True, 
                             Do_Tracking = True, 
                             Do_TrackCleaning = True, 
                             Do_MSD = True):
    
    tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
    fileNames = [fN.split('.')[0] for fN in tifNames]
    
    suffix_contour = Dict_Files_Suffix['suffix_contour']
    suffix_mask = Dict_Files_Suffix['suffix_mask']
    suffix_rawTracks = Dict_Files_Suffix['suffix_rawTracks']
    suffix_cleanTracks = Dict_Files_Suffix['suffix_cleanTracks']
    suffix_globalMsd = Dict_Files_Suffix['suffix_globalMsd']
    suffix_globalMsdFits = Dict_Files_Suffix['suffix_globalMsdFits']
    
    UmPerPix = Dict_Image_Settings['UmPerPix']
    PixPerUm = Dict_Image_Settings['PixPerUm']
    FPS = Dict_Image_Settings['FPS']
    
    mask_buffer_um = Dict_Analysis_Settings['mask_buffer_um']
    edge_buffer_cutoff_um = Dict_Analysis_Settings['edge_buffer_cutoff_um']
    max_Dt_s = Dict_Analysis_Settings['max_Dt_s']
    lowDt_upper = Dict_Analysis_Settings['lowDt_upper']
    highDt_lower = Dict_Analysis_Settings['highDt_lower']
    
    #### Make and save cell contours and masks
    if Do_Contours:
        print(pm.BRIGHTORANGE + '\n\n1. Contours step\n' + pm.NORMAL)
        for i in range(len(fileNames)): # len(fileNames)
            tP, fN = tifPaths[i], fileNames[i]
            print(pm.CYAN + f'File {i+1:.0f}/{len(fileNames):.0f} - {fN}\n' + pm.NORMAL)
            
            shape, dtype = ufun.tiff_inspect(tP)
            nT = shape[0]
            TT = range(0, nT, nT//100)
            img = ufun.load_stack_region(tP, time_indices=TT)
            
            Contour_cell, Mask_cell = tbca.make_NbYolkCell_contour_and_mask(img, PixPerUm,
                                                                            buffer_um = 0.0,
                                                                            mode = 'dark_background', 
                                                                            PLOT = False)
            
            contourFile = fN + suffix_contour + '.npy'
            maskFile = fN + suffix_mask + '.npy'
            np.save(os.path.join(srcDir, contourFile), Contour_cell)
            np.save(os.path.join(srcDir, maskFile), Mask_cell)

    #### Run Trackmate
    if Do_Tracking:
        print(pm.BRIGHTORANGE + '\n\n2. Tracking step\n' + pm.NORMAL)
        for i in range(len(fileNames)): #len(fileNames)
            tifPath, fN = tifPaths[i], fileNames[i]
            print(pm.CYAN + f'\nFile {i+1:.0f}/{len(fileNames):.0f} - {fN}' + pm.NORMAL)
            
            rawTrackName = fN + suffix_rawTracks + '.xml'
            maskFile = fN + suffix_mask + '.npy'
            Mask_cell = np.load(os.path.join(srcDir, maskFile))
            
            tbca.pretreat_and_track_NbYolk(tifPath, rawTrackName, dstDir, PixPerUm,
                                           Dict_TrackMate_Settings = Dict_TrackMate_Settings,
                                           Mask_cell = Mask_cell, mask_buffer_um = mask_buffer_um,
                                           PLOT = True, SHOWPLOT = False, SAVEPLOT = True)

    #### Import & format tracks
    if Do_TrackCleaning:
        print(pm.BRIGHTORANGE + '\n\n3. Tracks formatting step\n' + pm.NORMAL)
        for i in range(len(fileNames)): #len(fileNames)
            tP, fN = tifPaths[i], fileNames[i]
            print(pm.CYAN + f'File {i+1:.0f}/{len(fileNames):.0f} - {fN}' + pm.NORMAL)
            img_0 = ufun.load_stack_region(tP, time_indices=[0])[0]
            
            rawTrackName = fN + suffix_rawTracks + '.xml'
            cleanTrackName = fN + suffix_cleanTracks + '.csv'
            contourPath = os.path.join(srcDir, fN + suffix_contour + '.npy')
            
            rawTracks = tbca.import_TrackMate_tracks(os.path.join(dstDir, rawTrackName))
            Contour_cell = np.load(contourPath)

            tbca.rawTracks_2_cleanTracks(rawTracks, dstDir, cleanTrackName,
                                         Contour_cell, PixPerUm,
                                         edge_buffer_cutoff_um = edge_buffer_cutoff_um, nPoints_cuttoff = 30,
                                         RefImg = img_0, PLOT = True, SHOWPLOT = False, SAVEPLOT = True,
                                        )

    #### Import tracks, run trackpy.emsd, fit MSD
    if Do_MSD:
        print(pm.BRIGHTORANGE + '\n\n4. MSD compute & fit step\n' + pm.NORMAL)
        for i in range(len(fileNames)): #len(fileNames)
            tifPath, fN = tifPaths[i], fileNames[i]
            print(pm.CYAN + f'File {i+1:.0f}/{len(fileNames):.0f} - {fN}' + pm.NORMAL)
            
            rawTrackName = fN + suffix_rawTracks + '.xml'
            cleanTrackName = fN + suffix_cleanTracks + '.csv'
            msdName = fN + suffix_globalMsd + '.csv'
            msdFitsName = fN + suffix_globalMsdFits
            
            df = pd.read_csv(os.path.join(dstDir, cleanTrackName), sep='\t')
            
            max_lagtime = int(max_Dt_s * FPS)
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
            
            dict_MSDfits = {
                'maxDt_s': max_Dt_s,
                'lowDt_upper': lowDt_upper, 'highDt_lower': highDt_lower,
                'D_linear': D_linear,
                'k_full': k_full, 'D_full': D_full,
                'k_lowDt': k_lowDt, 'D_lowDt': D_lowDt,
                'k_highDt': k_highDt, 'D_highDt': D_highDt,
                }
            
            ufun.dict2json(dict_MSDfits, dstDir, msdFitsName)


def compute_pairwise_MSD(df, PixPerUm, FPS, Nframes,
                         len_TRanges = 100, delta_TRanges = -1, 
                         dist_th_um = 4, max_Dt_s = 2.5):
    
    list_TRanges, list_Pairs = tbca.get_pairs_for_TRanges_Delaunay(
        df, PixPerUm, FPS, Nframes,
        len_TRanges = 100, 
        delta_TRanges = -1,
        dist_th_um = 4
        )
    
    pairMSD = []
    max_lagtime = int(max_Dt_s * FPS)
    
    for k, TRange in enumerate(list_TRanges):
        pairs = list_Pairs[k]
        df_pairs = tbca.get_relative_displacement_by_TRange(df, pairs, TRange)
        res_pair_emsd = tp.motion.emsd(df_pairs.rename(columns={'pair_id':'particle'}), 
                                       UmPerPix, FPS, max_lagtime=80).reset_index()
        pairMSD.append(res_pair_emsd)

    df_allTRange_pair_MSD = pd.concat([pairMSD[0]['lagt']] + [df['msd'] for df in pairMSD], axis=1)
    df_allTRange_pair_MSD.columns = ['lagt'] + [f'msd_{tr}' for tr in list_TRanges]
    median_msd = np.median(df_allTRange_pair_MSD.loc[:, [f'msd_{tr}' for tr in list_TRanges]].to_numpy(), 
                           axis = 1)

    df_allTRange_pair_MSD['msd_median'] = median_msd
    
    return(df_allTRange_pair_MSD)



#### Plotting functions

def Tpf_str2num(tpf_str):
    L = tpf_str.split('min')
    try:
        tpf_num = int(L[0])*60
        if len(L) > 1 and len(L[1]) > 0:
            tpf_num += int(L[1])
    except:
        tpf_num = 0
    return(tpf_num)


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


# %% 1. Tracking and MSD

# %%% 26-09-30_D1

# %%%% Settings

#### Paths    


mainDir = os.path.join(up.Path_IntraCellTracking, '26-09-30_FastAcq-Channel_Fec_NB-Yolk')
srcDir = os.path.join(mainDir, 'D1')
dstDir = os.path.join(mainDir, 'SPT_results')

tifNames = [
    '26-09-30_D1_PreF_C1_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PreF_C2_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PreF_C3_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PostF_4min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PostF_6min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PostF_10min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PostF_13min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PostF_18min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PostF_25min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PostF_30min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PostF_35min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PostF_40min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PostF_45min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PostF_52min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PostF_60min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PostF_65min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PostF_70min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    ]

tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
fileNames = [fN.split('.')[0] for fN in tifNames]

Dict_Files_Suffix = {
    'suffix_contour' : '_cellContour',
    'suffix_mask' : '_cellMask',
    'suffix_rawTracks' : '_TmTracks',
    'suffix_cleanTracks' : '_PyTracks',
    'suffix_globalMsd' : '_GlobalMsd',
    'suffix_globalMsdFits' : '_GlobalMsdFits',
    }

#### Settings

UmPerPix = cd.UmPerPix_60X_W1
Dict_Image_Settings = {
    'UmPerPix' : UmPerPix,
    'PixPerUm' : 1/UmPerPix,
    'FPS' : 20,
    }

Dict_TrackMate_Settings = {
    'IMG_UNITS' : 'PIX',
    'RADIUS_UM' : 0.8, 
    'THRESH_SPOT_QLT' : 0.25,
    'THRESH_LINK_UM' : 0.25, 
    'THRESH_MIN_DURATION' : 30,
    }

Dict_Analysis_Settings = {
    'mask_buffer_um' : 2.0,
    'edge_buffer_cutoff_um' : 3.0,
    'max_Dt_s' : 5,
    'lowDt_upper' : 0.5,
    'highDt_lower' : 1.0,
    }


# %%%% Run analysis

NByolk_analysis_sequence(mainDir, srcDir, dstDir, tifNames, Dict_Files_Suffix, 
                         Dict_Image_Settings, Dict_TrackMate_Settings, Dict_Analysis_Settings,
                         Do_Contours = False, Do_Tracking = False, Do_TrackCleaning = False, Do_MSD = True)
  

# %%%% Import MSD & plot
    
max_Dt_s = Dict_Analysis_Settings['max_Dt_s']
lowDt_upper = Dict_Analysis_Settings['lowDt_upper']
highDt_lower = Dict_Analysis_Settings['highDt_lower']

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.set_xscale('log')
ax.set_yscale('log')

dict_res = {
    'label'  : [],
    'D_full' : [],
    'k_full' : [],
    'D_lowDt': [],
    'k_lowDt': [],
    'D_highDt':[],
    'k_highDt':[],
    }

ColorList = list(sns.color_palette("husl", len(fileNames)))

for i in range(len(fileNames)):
    tP, fN = tifPaths[i], fileNames[i]
    cleanTrackName = fN + Dict_Files_Suffix['suffix_cleanTracks'] + '.csv'
    msdName = fN + Dict_Files_Suffix['suffix_globalMsd'] + '.csv'
    msdFitsName = fN + Dict_Files_Suffix['suffix_globalMsdFits']
    
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

    long_label = '_'.join(fN.split('_')[2:4])
    short_label = fN.split('_')[3]

    ax.plot(T, MSD, label=long_label, color=ColorList[i], 
            marker='o', markersize=3, alpha=0.85)
    
    dict_res['label'].append(short_label)
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
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
ax.grid()
# ax.set_xlim([0.4e-1, 0.6e1])
# ax.set_ylim([2e-3, 0.5e0])
ax.set_ylabel('MSD (µm²)')
ax.set_xlabel(r'$\Delta t$ (s)')

plt.show()


df_Diffusion = pd.DataFrame(dict_res)
df_Diffusion['Tpf_s'] = df_Diffusion['label'].apply(lambda x : Tpf_str2num(x))
df_Diffusion['Tpf_min'] = df_Diffusion['Tpf_s']/60

fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex = True, layout='compressed')
ax = axes[0]
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.D_full, ls='-', marker='o', label=r'All $\Delta t$')
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.D_lowDt, ls='-', marker='o', label=r'$\Delta t \leq 0.5s$')
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.D_highDt, ls='-', marker='o', label=r'$\Delta t \geq 1s$')
ax.set_ylabel(r'$D_{eff}\ (\mu m^2/s^\alpha)$')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1]
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.k_full, ls='-', marker='o', label=r'All $\Delta t$')
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.k_lowDt, ls='-', marker='o', label=r'$\Delta t \leq 0.5s$')
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.k_highDt, ls='-', marker='o', label=r'$\Delta t \geq 1s$')
ax.set_ylabel(r'$\alpha$')
ax.set_xticks(df_Diffusion['Tpf_min'].values)
ax.set_xticklabels(df_Diffusion['Tpf_min'].values.astype(int), rotation = 30)
ax.set_xlabel('Tpf (min)')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

plt.show()



# %%% 26-09-30_D2

# %%%% Settings

#### Paths
mainDir = os.path.join(up.Path_IntraCellTracking, '26-09-30_FastAcq-Channel_Fec_NB-Yolk')
srcDir = os.path.join(mainDir, 'D2')
dstDir = os.path.join(mainDir, 'SPT_results')
tifNames = [
    '26-09-30_D2-R_PreF_C3_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PreF_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_4min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_6min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_8min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_12min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_15min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_20min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_25min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_30min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_36min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_40min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_45min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_50min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_53min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_55min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_60min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_65min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    ]

tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
fileNames = [fN.split('.')[0] for fN in tifNames]

Dict_Files_Suffix = {
    'suffix_contour' : '_cellContour',
    'suffix_mask' : '_cellMask',
    'suffix_rawTracks' : '_TmTracks',
    'suffix_cleanTracks' : '_PyTracks',
    'suffix_globalMsd' : '_GlobalMsd',
    'suffix_globalMsdFits' : '_GlobalMsdFits',
    'suffix_globalPairMsd' : '_GlobalPairMsd',
    'suffix_globalPairMsdFits' : '_GlobalPairMsdFits',
    }


#### Settings
UmPerPix = cd.UmPerPix_60X_W1
Dict_Image_Settings = {
    'UmPerPix' : UmPerPix,
    'PixPerUm' : 1/UmPerPix,
    'FPS' : 20,
    }

Dict_TrackMate_Settings = {
    'IMG_UNITS' : 'PIX',
    'RADIUS_UM' : 0.8, 
    'THRESH_SPOT_QLT' : 0.25,
    'THRESH_LINK_UM' : 0.25, 
    'THRESH_MIN_DURATION' : 30,
    }

Dict_Analysis_Settings = {
    'mask_buffer_um' : 2.0,
    'edge_buffer_cutoff_um' : 3.0,
    'max_Dt_s' : 5,
    'lowDt_upper' : 0.5,
    'highDt_lower' : 1.0,
    }


# %%%% Run analysis

NByolk_analysis_sequence(mainDir, srcDir, dstDir, tifNames, Dict_Files_Suffix, 
                         Dict_Image_Settings, Dict_TrackMate_Settings, Dict_Analysis_Settings,
                         Do_Contours = False, Do_Tracking = False, Do_TrackCleaning = False, Do_MSD = True)


# %%%% Import MSD & plot
    
max_Dt_s = Dict_Analysis_Settings['max_Dt_s']
lowDt_upper = Dict_Analysis_Settings['lowDt_upper']
highDt_lower = Dict_Analysis_Settings['highDt_lower']

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.set_xscale('log')
ax.set_yscale('log')

dict_res = {
    'label'  : [],
    'D_full' : [],
    'k_full' : [],
    'D_lowDt': [],
    'k_lowDt': [],
    'D_highDt':[],
    'k_highDt':[],
    }

ColorList = list(sns.color_palette("husl", len(fileNames)))

for i in range(len(fileNames)):
    tP, fN = tifPaths[i], fileNames[i]
    cleanTrackName = fN + Dict_Files_Suffix['suffix_cleanTracks'] + '.csv'
    msdName = fN + Dict_Files_Suffix['suffix_globalMsd'] + '.csv'
    msdFitsName = fN + Dict_Files_Suffix['suffix_globalMsdFits']
    
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

    long_label = '_'.join(fN.split('_')[2:4])
    short_label = fN.split('_')[3]

    ax.plot(T, MSD, label=long_label, color=ColorList[i], 
            marker='o', markersize=3, alpha=0.85)
    
    dict_res['label'].append(short_label)
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
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
ax.grid()
# ax.set_xlim([0.4e-1, 0.6e1])
# ax.set_ylim([2e-3, 0.5e0])
ax.set_ylabel('MSD (µm²)')
ax.set_xlabel(r'$\Delta t$ (s)')

plt.show()


df_Diffusion = pd.DataFrame(dict_res)
df_Diffusion['Tpf_s'] = df_Diffusion['label'].apply(lambda x : Tpf_str2num(x))
df_Diffusion['Tpf_min'] = df_Diffusion['Tpf_s']/60

fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex = True, layout='compressed')
ax = axes[0]
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.D_full, ls='-', marker='o', label=r'All $\Delta t$')
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.D_lowDt, ls='-', marker='o', label=r'$\Delta t \leq 0.5s$')
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.D_highDt, ls='-', marker='o', label=r'$\Delta t \geq 1s$')
ax.set_ylabel(r'$D_{eff}\ (\mu m^2/s^\alpha)$')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1]
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.k_full, ls='-', marker='o', label=r'All $\Delta t$')
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.k_lowDt, ls='-', marker='o', label=r'$\Delta t \leq 0.5s$')
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.k_highDt, ls='-', marker='o', label=r'$\Delta t \geq 1s$')
ax.set_ylabel(r'$\alpha$')
ax.set_xticks(df_Diffusion['Tpf_min'].values)
ax.set_xticklabels(df_Diffusion['Tpf_min'].values.astype(int), rotation = 30)
ax.set_xlabel('Tpf (min)')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

plt.show()



# %%%% Compute pair-MSD

pm.setGraphicOptions(mode='screen')

for i in range(len(fileNames)):
    print(i)
    tP, fN = tifPaths[i], fileNames[i]
    cleanTrackName = fN + Dict_Files_Suffix['suffix_cleanTracks'] + '.csv'
    df = pd.read_csv(os.path.join(dstDir, cleanTrackName), sep='\t')
    
    PixPerUm = Dict_Image_Settings['PixPerUm']
    FPS = Dict_Image_Settings['FPS']
    max_Dt_s = 4
    
    shape, dtype = ufun.tiff_inspect(tP)
    Nframes = shape[0]   
    
    #### Compute Pair MSD
    df_allTRange_pair_MSD = compute_pairwise_MSD(df, PixPerUm, FPS, Nframes,
                                                 len_TRanges = 100, delta_TRanges = -1, 
                                                 dist_th_um = 4, max_Dt_s = 4)
    print(i)
    
    pairMsdName = fN + Dict_Files_Suffix['suffix_globalPairMsd'] + '.csv'
    df_allTRange_pair_MSD.to_csv(os.path.join(dstDir, pairMsdName), index=False, sep='\t')
    
    
    #### Fit Pair MSD
    T = df_allTRange_pair_MSD['lagt'].to_numpy()
    MSD = df_allTRange_pair_MSD['msd_median'].to_numpy()

    lowDt_upper = Dict_Analysis_Settings['lowDt_upper']
    highDt_lower = Dict_Analysis_Settings['highDt_lower']
    
    iLow = ufun.findFirst(lowDt_upper, T) + 1
    iHigh = ufun.findFirst(highDt_lower, T)
    
    parms, results = ufun.fitLineHuber(T, MSD, with_intercept = False)
    D_linear = parms[0]/8
    
    parms, results = ufun.fitLineHuber(np.log(T), np.log(MSD), with_intercept = True)
    b, a = parms
    k_full = a
    D_full = np.exp(b)/8

    parms, results = ufun.fitLineHuber(np.log(T[:iLow]), np.log(MSD[:iLow]), with_intercept = True)
    b, a = parms
    k_lowDt = a
    D_lowDt = np.exp(b)/8

    parms, results = ufun.fitLineHuber(np.log(T[iHigh:]), np.log(MSD[iHigh:]), with_intercept = True)
    b, a = parms
    k_highDt = a
    D_highDt = np.exp(b)/8
    
    dict_pairMSDfits = {
        'max_Dt_s': max_Dt_s,
        'lowDt_upper': lowDt_upper, 'highDt_lower': highDt_lower,
        'D_linear': D_linear,
        'k_full': k_full, 'D_full': D_full,
        'k_lowDt': k_lowDt, 'D_lowDt': D_lowDt,
        'k_highDt': k_highDt, 'D_highDt': D_highDt,
        }
    
    pairMsdFitsName = fN + Dict_Files_Suffix['suffix_globalPairMsdFits']
    ufun.dict2json(dict_pairMSDfits, dstDir, pairMsdFitsName)


# %%%% Import pair-MSD & plot
    
max_Dt_s = Dict_Analysis_Settings['max_Dt_s']
lowDt_upper = Dict_Analysis_Settings['lowDt_upper']
highDt_lower = Dict_Analysis_Settings['highDt_lower']

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax in axes:
    ax.set_xscale('log')
    ax.set_yscale('log')

dict_res_MSD = {
    'label'  : [],
    'D_linear' : [],
    'D_full' : [], 'k_full' : [],
    'D_lowDt': [], 'k_lowDt': [],
    'D_highDt':[], 'k_highDt':[],
    }

dict_res_pMSD = {
    'label'  : [],
    'D_linear' : [],
    'D_full' : [], 'k_full' : [],
    'D_lowDt': [], 'k_lowDt': [],
    'D_highDt':[], 'k_highDt':[],
    }

ColorList = list(sns.color_palette("husl", len(fileNames)))

for i in range(len(fileNames)):
    tP, fN = tifPaths[i], fileNames[i]
    cleanTrackName = fN + Dict_Files_Suffix['suffix_cleanTracks'] + '.csv'
    msdName = fN + Dict_Files_Suffix['suffix_globalMsd'] + '.csv'
    msdFitsName = fN + Dict_Files_Suffix['suffix_globalMsdFits']
    pairMsdName = fN + Dict_Files_Suffix['suffix_globalPairMsd'] + '.csv'
    pairMsdFitsName = fN + Dict_Files_Suffix['suffix_globalPairMsdFits']
    
    df = pd.read_csv(os.path.join(dstDir, cleanTrackName), sep='\t')
    
    # "Normal" MSD
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
    
    # Tc = (D_lowDt/D_highDt)**(1/(k_highDt-k_lowDt))

    long_label = '_'.join(fN.split('_')[2:4])
    short_label = fN.split('_')[3]
    
    ax = axes[0]
    ax.plot(T, MSD, label=long_label, color=ColorList[i], 
            marker='o', markersize=3, alpha=0.85)
    
    dict_res_MSD['label'].append(short_label)
    dict_res_MSD['D_linear'].append(D_linear)
    dict_res_MSD['D_full'].append(D_full)
    dict_res_MSD['k_full'].append(k_full)
    dict_res_MSD['D_lowDt'].append(D_lowDt)
    dict_res_MSD['k_lowDt'].append(k_lowDt)
    dict_res_MSD['D_highDt'].append(D_highDt)
    dict_res_MSD['k_highDt'].append(k_highDt)
    
    # Pair-MSD
    res_pair_emsd = pd.read_csv(os.path.join(dstDir, pairMsdName), sep='\t')
    pT, pMSD = res_pair_emsd['lagt'], res_pair_emsd['msd_median']
    dict_pairMSDfits = ufun.json2dict(dstDir, pairMsdFitsName)
    
    D_linear = dict_pairMSDfits['D_linear']
    k_full = dict_pairMSDfits['k_full']
    D_full = dict_pairMSDfits['D_full']
    k_lowDt = dict_pairMSDfits['k_lowDt']
    D_lowDt = dict_pairMSDfits['D_lowDt']
    k_highDt = dict_pairMSDfits['k_highDt']
    D_highDt = dict_pairMSDfits['D_highDt']
    
    # Tc = (D_lowDt/D_highDt)**(1/(k_highDt-k_lowDt))

    long_label = '_'.join(fN.split('_')[2:4])
    short_label = fN.split('_')[3]
    
    ax = axes[1]
    ax.plot(pT, pMSD, label=long_label, color=ColorList[i], 
            marker='o', markersize=3, alpha=0.85)
    
    dict_res_pMSD['label'].append(short_label)
    dict_res_pMSD['D_linear'].append(D_linear)
    dict_res_pMSD['D_full'].append(D_full)
    dict_res_pMSD['k_full'].append(k_full)
    dict_res_pMSD['D_lowDt'].append(D_lowDt)
    dict_res_pMSD['k_lowDt'].append(k_lowDt)
    dict_res_pMSD['D_highDt'].append(D_highDt)
    dict_res_pMSD['k_highDt'].append(k_highDt)
    
for ax in axes:
    Xp1 = np.array([1e-1, 5e-1])
    Xp2 = np.array([1, 2])
    ax.plot(Xp1, 12e-3*Xp1**0.5, color = 'gray', ls=':', label=r'$y \propto x^{1/2}$')
    ax.plot(Xp2, 0.15e-1*Xp2**1, color = 'gray', ls='--', label=r'$y \propto x^{1}$')
    ax.grid()
    # ax.set_xlim([0.4e-1, 0.6e1])
    # ax.set_ylim([2e-3, 0.5e0])
    ax.set_ylabel('MSD (µm²)')
    ax.set_xlabel(r'$\Delta t$ (s)')

ax = axes[0]
ax.set_ylabel('MSD (µm²)')

ax = axes[1]
ax.set_ylabel('pair-MSD (µm²)')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

plt.show()


# Time - D - alpha
df_Diffusion = pd.DataFrame(dict_res_MSD)
df_Diffusion['Tpf_s'] = df_Diffusion['label'].apply(lambda x : Tpf_str2num(x))
df_Diffusion['Tpf_min'] = df_Diffusion['Tpf_s']/60

df_pairDiffusion = pd.DataFrame(dict_res_pMSD)
df_pairDiffusion['Tpf_s'] = df_Diffusion['label'].apply(lambda x : Tpf_str2num(x))
df_pairDiffusion['Tpf_min'] = df_Diffusion['Tpf_s']/60

fig, axes = plt.subplots(2, 2, figsize=(12, 7), 
                         sharey='row', sharex = True, layout='compressed')

# "Normal" MSD
df = df_Diffusion
ax = axes[0, 0]
ax.set_title('Individual MSD')
ax.plot(df.Tpf_min, df.D_full, ls='-', marker='o', label=r'All $\Delta t$')
ax.plot(df.Tpf_min, df.D_lowDt, ls='-', marker='o', label=r'$\Delta t \leq 0.5s$')
ax.plot(df.Tpf_min, df.D_highDt, ls='-', marker='o', label=r'$\Delta t \geq 1s$')
ax.plot(df.Tpf_min, df.D_linear, ls='-', marker='o', label=r'Linear fit')
ax.set_ylabel(r'$D_{eff}\ (\mu m^2/s^\alpha)$')
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 0]
ax.plot(df.Tpf_min, df.k_full, ls='-', marker='o', label=r'All $\Delta t$')
ax.plot(df.Tpf_min, df.k_lowDt, ls='-', marker='o', label=r'$\Delta t \leq 0.5s$')
ax.plot(df.Tpf_min, df.k_highDt, ls='-', marker='o', label=r'$\Delta t \geq 1s$')
ax.set_ylabel(r'$\alpha$')
ax.set_xticks(df_Diffusion['Tpf_min'].values)
ax.set_xticklabels(df_Diffusion['Tpf_min'].values.astype(int), rotation = 30)
ax.set_xlabel('Tpf (min)')
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

# Pair MSD
df = df_pairDiffusion
ax = axes[0, 1]
ax.set_title('Pair-MSD')
ax.plot(df.Tpf_min, df.D_full, ls='-', marker='o', label=r'All $\Delta t$')
ax.plot(df.Tpf_min, df.D_lowDt, ls='-', marker='o', label=r'$\Delta t \leq 0.5s$')
ax.plot(df.Tpf_min, df.D_highDt, ls='-', marker='o', label=r'$\Delta t \geq 1s$')
ax.plot(df.Tpf_min, df.D_linear, ls='-', marker='o', label=r'Linear fit')
ax.set_ylabel(r'$D_{eff}\ (\mu m^2/s^\alpha)$')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 1]
ax.plot(df.Tpf_min, df.k_full, ls='-', marker='o', label=r'All $\Delta t$')
ax.plot(df.Tpf_min, df.k_lowDt, ls='-', marker='o', label=r'$\Delta t \leq 0.5s$')
ax.plot(df.Tpf_min, df.k_highDt, ls='-', marker='o', label=r'$\Delta t \geq 1s$')
ax.set_ylabel(r'$\alpha$')
ax.set_xticks(df_Diffusion['Tpf_min'].values)
ax.set_xticklabels(df_Diffusion['Tpf_min'].values.astype(int), rotation = 30)
ax.set_xlabel('Tpf (min)')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

plt.show()


# %%%% Make movie

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


def get_pairs_for_TRanges_Delaunay(df, PixPerUm, FPS, Nframes,
                                   len_TRanges = 200, delta_TRanges = -1,
                                   dist_th_um = 5):
    df.frame = df.frame.astype(int)
    df.particle = df.particle.astype(int)
    dist_th = dist_th_um * PixPerUm
    
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
    list_TRanges = [f'{fi}_{ff}' for fi, ff in zip(FI, FF)]
    list_Pairs = []
    
    #### !!!! The syntax below is really cool !
    grouped = df.groupby('particle')
    for pid, df_p in grouped:
        frames = df_p['frame'].to_numpy()
        x = df_p['x'].to_numpy()
        y = df_p['y'].to_numpy()

        pfi = frames.min()
        pff = frames.max()

        for fi, ff in zip(FI, FF):
            if pfi <= fi + 1 and ff <= pff:
                mask = (fi <= frames - 1) & (frames - 1 < ff)
                xm = np.median(x[mask])
                ym = np.median(y[mask])

                key = f'{fi}_{ff}'
                result = dict_TRanges2particles[key]
                result['pid'].append(int(pid))
                result['xm'].append(float(xm))
                result['ym'].append(float(ym))
    
    for k, TRange in enumerate(list_TRanges):
        df_parts = pd.DataFrame(dict_TRanges2particles[TRange])
        XY = np.array([df_parts['xm'].values[:],
                       df_parts['ym'].values[:]]).T
        
        tri = Delaunay(XY)
        edges_short, _ = tri_to_short_edges(tri, XY, dist_th)
        close_pairs = df_parts['pid'].values[edges_short]
        
        list_Pairs.append(np.array(close_pairs))    
            
    return(list_TRanges, list_Pairs)



def make_snapshot_of_pairs(df, SCALE, FPS, fi, ff, dist_th_um = 5):
    df.frame = df.frame.astype(int)
    df.particle = df.particle.astype(int)
    dist_th = dist_th_um * PixPerUm
    
    df_f = df[(df['frame']>fi) & (df['frame']<=ff)]
    
    valid_particles = []
    
    grouped = df_f.groupby('particle')
    for pid, df_p in grouped:
        if len(df_p) == (ff-fi):
            valid_particles.append(pid)
            
    valid_particles = np.array(valid_particles)
    df_f = df_f[df_f['particle'].apply(lambda x : x in valid_particles)]
    
    grouped = df_f.groupby('frame')
    for f, df_ff in grouped:
        XY = np.array([df_ff['x'].to_numpy(),
                       df_ff['y'].to_numpy()]).T
        
        tri = Delaunay(XY)
        edges_short, _ = tri_to_short_edges(tri, XY, dist_th)
        close_pairs = df_ff['pid'].to_numpy()[edges_short]
        
        #### !!!! TBD here !!
            
    


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





# %%% 26-07-29

# %%%% Settings

#### Paths    

mainDir = os.path.join(up.Path_IntraCellTracking, '26-07-29_FastAcq_Fec_NB-Yolk')
srcDir = os.path.join(mainDir, 'Crops')
dstDir = os.path.join(mainDir, 'SPT_results')

tifNames = [
            '26-07-29_PostF_2min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
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

Dict_Files_Suffix = {
    'suffix_contour' : '_cellContour',
    'suffix_mask' : '_cellMask',
    'suffix_rawTracks' : '_TmTracks',
    'suffix_cleanTracks' : '_PyTracks',
    'suffix_globalMsd' : '_GlobalMsd',
    'suffix_globalMsdFits' : '_GlobalMsdFits',
    }


#### Settings
UmPerPix = cd.UmPerPix_60X_W1
Dict_Image_Settings = {
    'UmPerPix' : UmPerPix,
    'PixPerUm' : 1/UmPerPix,
    'FPS' : 10,
    }

Dict_TrackMate_Settings = {
    'IMG_UNITS' : 'PIX',
    'RADIUS_UM' : 0.8, 
    'THRESH_SPOT_QLT' : 0.25,
    'THRESH_LINK_UM' : 0.4, # increased at 0.4um for 10Hz
    'THRESH_MIN_DURATION' : 30,
    }

Dict_Analysis_Settings = {
    'mask_buffer_um' : 2.0,
    'edge_buffer_cutoff_um' : 3.0,
    'max_Dt_s' : 5,
    'lowDt_upper' : 0.5,
    'highDt_lower' : 1.0,
    }



# %%% TBD
    
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
dict_TRanges2pairs_N = tbca.get_pairs_for_TRanges(df, SCALE, FPS, Nframes,
                          len_TRanges = 100, delta_TRanges = -1,
                          dist_th_um = 5)
print(f'Dt = {time.time()-top:.3f} s')

top = time.time()
dict_TRanges2pairs_D = tbca.get_pairs_for_TRanges_Delaunay(df, SCALE, FPS, Nframes,
                          len_TRanges = 100, delta_TRanges = -1,
                          dist_th_um = 5)
print(f'Dt = {time.time()-top:.3f} s')



# %%%% MSRD functions II




pm.setGraphicOptions(mode='screen')

TRanges = np.array(list(dict_TRanges2pairs_D.keys()))
dict_TRanges2pairMSD = {}

for TRange in TRanges:

    pairs = dict_TRanges2pairs_D[TRange]

    # top = time.time()
    df_pairs = tbca.get_relative_displacement_by_TRange(df, pairs, TRange)
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

# %%% Dev

# %%%% MSRD computation test

pm.setGraphicOptions(mode='screen')

# for i in range(len(fileNames)):
i = 13
tP, fN = tifPaths[i], fileNames[i]
cleanTrackName = fN + Dict_Files_Suffix['suffix_cleanTracks'] + '.csv'
df = pd.read_csv(os.path.join(dstDir, cleanTrackName), sep='\t')

PixPerUm = Dict_Image_Settings['PixPerUm']
FPS = Dict_Image_Settings['FPS']
shape, dtype = ufun.tiff_inspect(tP)
Nframes = shape[0]   
    

list_TRanges, list_Pairs = tbca.get_pairs_for_TRanges_Delaunay(
    df, PixPerUm, FPS, Nframes,
    len_TRanges = 100, 
    delta_TRanges = -1,
    dist_th_um = 4
    )

pairMSD = []
for k, TRange in enumerate(list_TRanges):
    pairs = list_Pairs[k]
    df_pairs = tbca.get_relative_displacement_by_TRange(df, pairs, TRange)
    res_pair_emsd = tp.motion.emsd(df_pairs.rename(columns={'pair_id':'particle'}), 
                                   UmPerPix, FPS, max_lagtime=80).reset_index()
    pairMSD.append(res_pair_emsd)

df_allTRange_pair_MSD = pd.concat([pairMSD[0]['lagt']] + [df['msd'] for df in pairMSD], axis=1)
df_allTRange_pair_MSD.columns = ['lagt'] + [f'msd_{tr}' for tr in list_TRanges]
median_msd = np.median(df_allTRange_pair_MSD.loc[:, [f'msd_{tr}' for tr in list_TRanges]].to_numpy(), 
                       axis = 1)
# mean_msd = np.mean(df_allTRange_pair_MSD.loc[:, [f'msd_{tr}' for tr in list_TRanges]].to_numpy(), 
#                        axis = 1)
df_allTRange_pair_MSD['msd_median'] = median_msd
# df_allTRange_pair_MSD['msd_mean'] = mean_msd

# Plot
fig, ax = plt.subplots(1, 1, figsize=(7 , 5))
ax.set_xscale('log')
ax.set_yscale('log')

for k, TRange in enumerate(list_TRanges):
    res_pair_emsd = pairMSD[k]
    ax.plot(res_pair_emsd.lagt, res_pair_emsd.msd, ls='', marker='.', label=TRange)
  
ax.plot(df_allTRange_pair_MSD['lagt'], df_allTRange_pair_MSD['msd_median'], 
        ls='', marker='.', color='k', label='Median')
# ax.plot(df_allTRange_pair_MSD['lagt'], df_allTRange_pair_MSD['msd_mean'], 
#         ls='', marker='.', color='w', mec='k', mew=0.2, label='Mean')
ax.grid()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          ncols = 2)

plt.show()
