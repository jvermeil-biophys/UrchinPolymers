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
# import re
# import cv2
# import pyautogui

import numpy as np
import pandas as pd
import trackpy as tp
import skimage as skm
import seaborn as sns
import matplotlib as mpl
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

from shapely.geometry import MultiPoint, Polygon
# from scipy.spatial import Delaunay # ConvexHull
# from scipy.signal import savgol_filter

import Libs.PlotMaker as pm
import Libs.UrchinPaths as up
import Libs.CalibrationData as cd
import Libs.UtilityFunctions as ufun
import Libs.ToolboxCytoplasmAnalysis as tbca
# import Libs.ToolboxStructureAnalysis as tbsa


#### Paths

FIG_DIR = os.path.join(up.Path_IntraCellTracking, 'SPT_figures')
RES_DIR = os.path.join(up.Path_IntraCellTracking, 'SPT_results')

Dict_rP = {
    'dir_segmentation': os.path.join(RES_DIR, 'segmentation'),
    'dir_rawTracks': os.path.join(RES_DIR, 'rawTracks'),
    'dir_cleanTracks': os.path.join(RES_DIR, 'cleanTracks'),
    'dir_msd': os.path.join(RES_DIR, 'MSD'),
    'dir_pairMsd': os.path.join(RES_DIR, 'pairMSD'),
    'dir_indivMsd': os.path.join(RES_DIR, 'indivMSD'),
    'dir_ROMsd': os.path.join(RES_DIR, 'RO-MSD'),
    }

Dict_pP = {
    'dir_trackFig': os.path.join(FIG_DIR, 'Tracking'),
    'dir_segmentFig': os.path.join(FIG_DIR, 'Segmentation'),
    'dir_driftFig': os.path.join(FIG_DIR, 'Drift correction'),
    }

df_cells_PATH = os.path.join(RES_DIR, 'SPT_Data_Cells.csv')
df_films_PATH = os.path.join(RES_DIR, 'SPT_Data_Films.csv')

DF_CELLS = pd.read_csv(df_cells_PATH)
DF_FILMS = pd.read_csv(df_films_PATH)


Dict_fSuf = {
    'suf_contour' : '_cellContour',
    'suf_mask' : '_cellMask',
    'suf_rawTracks' : '_TmTracks',
    'suf_cleanTracks' : '_PyTracks',
    'suf_cleanTracks_DriftCorr' : '_PyTracks2',
    'suf_drift' : '_XYdrift',
    'suf_globalMsd' : '_GlobalMsd',
    'suf_globalMsdFits' : '_GlobalMsdFits',
    'suf_globalPairMsd' : '_GlobalPairMsd',
    'suf_globalPairMsdFits' : '_GlobalPairMsdFits',
    'suf_indivMsd' : '_IndivMsd',
    'suf_indivMsdFits' : '_IndivMsdFits',
    'suf_globalROMsd' : '_GlobalRO-Msd',
    'suf_indivROMsd' : '_IndivRO-Msd',
    'suf_indivROMsdFits' : '_IndivRO-MsdFits',
    }



# %% Analysis functions

def autofill_SPT_Data(srcDir, tifNames, df_films_path):
    DF_FILMS = pd.read_csv(df_films_path)
    for tN in tifNames:
        tP = os.path.join(srcDir, tN)
        if tN not in DF_FILMS['tN'].to_numpy():
            idx = len(DF_FILMS)
            DF_FILMS.loc[idx, 'tN'] = tN
            
        if tN in DF_FILMS['tN'].to_numpy():
            idx = DF_FILMS[DF_FILMS['tN'] == tN].index[0]
            im_shape, im_type = ufun.tiff_inspect(tP)
            nT, nY, nX = im_shape
            DF_FILMS.loc[idx, 'n_frames'] = int(nT)
            DF_FILMS.loc[idx, 'im_width'] = int(nX)
            DF_FILMS.loc[idx, 'im_height'] = int(nY)
            
            img = ufun.load_stack_region(tP, time_indices=[nT//2])[0]
            QA, uiXY = tbca.point_cell_center(img)
            if QA == 'No center':
                xc, yc = np.nan, np.nan
            else:
                xc, yc = uiXY[0]
                xc, yc = round(xc), round(yc)
                
            DF_FILMS.loc[idx, 'x_center'] = xc
            DF_FILMS.loc[idx, 'y_center'] = yc
    
    int_cols = ['n_frames', 'im_width', 'im_height']
    for col in int_cols:
        DF_FILMS[col] = DF_FILMS[col].astype(int)
    
    DF_FILMS.to_csv(df_films_path, index=False, na_rep='NaN')
    

def NByolk_track_sequence(srcDir, tifNames, df_films_path, 
                          Dict_fSuf, Dict_rP, Dict_pP, 
                          Dict_TrackMate_Settings, 
                          Do_Contours = True, 
                          Do_Tracking = True, 
                          Do_TrackCleaning = True,):
    
    DF_FILMS = pd.read_csv(df_films_path)
    tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
    fileNames = [fN.split('.')[0] for fN in tifNames]
    
    for i in range(len(fileNames)):
        tP, tN, fN = tifPaths[i], tifNames[i], fileNames[i]
        print(pm.CYAN + f'\nFile {i+1:.0f}/{len(fileNames):.0f} - {fN}' + pm.NORMAL)
        
        idx = DF_FILMS[DF_FILMS['tN'] == tN].index[0]
        PixPerUm = DF_FILMS.loc[idx, 'PixPerUm']
        FPS = DF_FILMS.loc[idx, 'FPS']
        Segmentation_Mode = DF_FILMS.loc[idx, 'segmentation']
        Do_DriftCorrection = DF_FILMS.loc[idx, 'drift_correction']
        mask_buffer_um = DF_FILMS.loc[idx, 'mask_edge_cutoff_um']
        edge_buffer_cutoff_um = DF_FILMS.loc[idx, 'tracking_edge_cutoff_um']
        nT = int(DF_FILMS.loc[idx, 'n_frames'])
        nY = int(DF_FILMS.loc[idx, 'im_height'])
        nX = int(DF_FILMS.loc[idx, 'im_width'])
        
        
        #### Make and save cell contours and masks
        if Do_Contours:
            print(pm.BRIGHTORANGE + '1. Contours step' + pm.NORMAL)
            TT = range(0, nT, nT//100)
            img = ufun.load_stack_region(tP, time_indices=TT)
            
            Contour_cell, Mask_cell = tbca.make_NbYolkCell_contour_and_mask(
                img, tN, Dict_pP['dir_segmentFig'], PixPerUm,
                keep_only_largest = True, exclude_on_edges = True,
                PLOT = True, SHOWPLOT = False, SAVEPLOT = True
                )
            
            contourFile = fN + Dict_fSuf['suf_contour'] + '.npy'
            maskFile = fN + Dict_fSuf['suf_mask'] + '.npy'
            np.save(os.path.join(Dict_rP['dir_segmentation'], contourFile), Contour_cell)
            np.save(os.path.join(Dict_rP['dir_segmentation'], maskFile), Mask_cell)
    
    
        #### Run Trackmate
        if Do_Tracking:
            print(pm.BRIGHTORANGE + '2. Tracking step' + pm.NORMAL)
            rawTrackName = fN + Dict_fSuf['suf_rawTracks'] + '.xml'
            maskFile = fN + Dict_fSuf['suf_mask'] + '.npy'
            if Segmentation_Mode == 'cell-contour':
                Mask_cell = np.load(os.path.join(Dict_rP['dir_segmentation'], maskFile))
            else:
                Mask_cell = None
            
            tbca.pretreat_and_track_NbYolk(
                tP, rawTrackName, 
                Dict_rP['dir_rawTracks'], Dict_pP['dir_trackFig'], 
                PixPerUm, Dict_TrackMate_Settings = Dict_TrackMate_Settings,
                Mask_cell = Mask_cell, mask_buffer_um = mask_buffer_um,
                PLOT = True, SHOWPLOT = False, SAVEPLOT = True
                )
    
    
        #### Import & format tracks
        if Do_TrackCleaning:
            print(pm.BRIGHTORANGE + '3. Tracks formatting step' + pm.NORMAL)
            img_0 = ufun.load_stack_region(tP, time_indices=[0])[0]
            
            rawTrackName = fN + Dict_fSuf['suf_rawTracks'] + '.xml'
            cleanTrackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
            contourPath = os.path.join(Dict_rP['dir_segmentation'], fN + Dict_fSuf['suf_contour'] + '.npy')
            
            rawTracks = tbca.import_TrackMate_tracks(os.path.join(Dict_rP['dir_rawTracks'], rawTrackName))
            shape = [nY, nX]
            
            if Segmentation_Mode == 'cell-contour':
                Contour_cell = np.load(contourPath)
            else:
                Contour_cell = None
            
            df = tbca.rawTracks_2_cleanTracks(
                rawTracks, cleanTrackName, Dict_pP['dir_trackFig'], 
                PixPerUm, shape = shape, Contour_cell = Contour_cell, 
                edge_buffer_cutoff_um = edge_buffer_cutoff_um, nPoints_cuttoff = 30,
                RefImg = img_0, PLOT = True, SHOWPLOT = False, SAVEPLOT = True,
                )
            
            df.to_csv(os.path.join(Dict_rP['dir_cleanTracks'], cleanTrackName), index=False, sep = '\t')
            plt.close('all')
        
        
        #### Correct global drift in trajectory
        if Do_DriftCorrection:
            print(pm.BRIGHTORANGE + '3.5. Drift correction' + pm.NORMAL)                
            cleanTrackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
            df = pd.read_csv(os.path.join(dstDir, cleanTrackName), sep='\t')
        
            df2, Dxy_corr = tbca.correct_drift_in_Tracks(
                df, fN, tP, Dict_rP['dir_cleanTracks'], Dict_pP['dir_driftFig'], 
                PixPerUm, FPS, 
                PLOT = True, SHOWPLOT = False, SAVEPLOT = True, 
                export_image = False
                )
            
            driftArrayName = fN + Dict_fSuf['suf_drift']  + '.txt'
            np.savetxt(os.path.join(Dict_rP['dir_cleanTracks'], driftArrayName), Dxy_corr)
            
            dcTrackName = fN + Dict_fSuf['suf_cleanTracks_DriftCorr'] + '.csv'
            df2.to_csv(os.path.join(Dict_rP['dir_cleanTracks'], dcTrackName), sep='\t')
    
    
def NByolk_msd_compute(tifNames, df_films_path, df_res_name,
                       Dict_fSuf, Dict_rP, Dict_pP, 
                       Do_MSD = True, Do_PairMSD = True):
    
    DF_FILMS = pd.read_csv(df_films_path)
    
    fileNames = [fN.split('.')[0] for fN in tifNames]
    # tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
    
    for i in range(len(fileNames)):
        tN, fN = tifNames[i], fileNames[i]
        # tP = tifPaths[i]
        print(pm.CYAN + f'\nFile {i+1:.0f}/{len(fileNames):.0f} - {fN}' + pm.NORMAL)
        
        try:
            DF_RES = pd.read_csv(os.path.join(RES_DIR, df_res_name + '.csv')).reset_index(drop=True)
        except:
            DF_RES = pd.DataFrame({})
            
        if ('tN' not in DF_RES.columns) or (tN not in DF_RES['tN'].to_numpy()):
            DF_RES.loc[len(DF_RES), 'tN'] = tN
        
        idx = DF_FILMS[DF_FILMS['tN'] == tN].index[0]
        PixPerUm = DF_FILMS.loc[idx, 'PixPerUm']
        FPS = DF_FILMS.loc[idx, 'FPS']
        Use_DriftCorrected = DF_FILMS.loc[idx, 'drift_correction']
        max_Dt_s = DF_FILMS.loc[idx, 'msd_max_Dt_s']
        lowDt_upper = DF_FILMS.loc[idx, 'msd_lowDt_upper_s']
        highDt_lower = DF_FILMS.loc[idx, 'msd_highDt_lower_s']
        pairMSD_Dt_s = DF_FILMS.loc[idx, 'pMsd_max_Dt_s']
        pairMSD_dist_th_um = DF_FILMS.loc[idx, 'pMSD_dist_th_um']
    
    
        #### Import tracks, run trackpy.emsd, fit MSD
        if Do_MSD:
            print(pm.BRIGHTORANGE + '4. MSD compute & fit step' + pm.NORMAL)
            # rawTrackName = fN + Dict_fSuf['suf_rawTracks'] + '.xml'
            if Use_DriftCorrected:
                trackName = fN + Dict_fSuf['suf_cleanTracks_DriftCorr'] + '.csv'
            else:
                trackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
                
            msdName = fN + Dict_fSuf['suf_globalMsd'] + '.csv'
            msdFitsName = fN + Dict_fSuf['suf_globalMsdFits']
            
            df = pd.read_csv(os.path.join(Dict_rP['dir_cleanTracks'], trackName), sep='\t')
            msd_N_tracks = len(df['particle'].unique())
            
            max_lagtime = int(max_Dt_s * FPS)
            res_emsd = tp.motion.emsd(df, 1/PixPerUm, FPS, max_lagtime=max_lagtime).reset_index()
            res_emsd.to_csv(os.path.join(Dict_rP['dir_msd'], msdName), index=False, sep='\t')
            
            T, MSD = res_emsd['lagt'].values, res_emsd['msd'].values
            iLow = ufun.findFirst(lowDt_upper, T) + 1
            iHigh = ufun.findFirst(highDt_lower, T)

            parms, results = ufun.fitLineHuber(T, MSD, with_intercept = False)
            msd_D_lin = parms[0]/4
            
            parms, results = ufun.fitLineHuber(
                np.log(T), np.log(MSD), 
                with_intercept = True)
            b, a = parms
            msd_k_full = a
            msd_D_full = np.exp(b)/4
            
            if len(T[:iLow]) >= 3:
                parms, results = ufun.fitLineHuber(
                    np.log(T[:iLow]), np.log(MSD[:iLow]), 
                    with_intercept = True)
                b, a = parms
                msd_k_lowDt = a
                msd_D_lowDt = np.exp(b)/4
            else:
                msd_k_lowDt, msd_D_lowDt = np.nan, np.nan
            
            if len(T[iHigh:]) >= 6:
                parms, results = ufun.fitLineHuber(
                    T[iHigh:], MSD[iHigh:], 
                    with_intercept = False)
                msd_D_lin_highDt = parms[0]/4
                
                parms, results = ufun.fitLineHuber(
                    np.log(T[iHigh:]), np.log(MSD[iHigh:]), 
                    with_intercept = True)
                b, a = parms
                msd_k_highDt = a
                msd_D_highDt = np.exp(b)/4
            else:
                msd_k_highDt, msd_D_highDt = np.nan, np.nan
                
            msd_tc = (msd_D_lowDt/msd_D_highDt)**(1/(msd_k_highDt - msd_k_lowDt))
            
            dict_MSDfits = {
                # 'maxDt_s': max_Dt_s, 'lowDt_upper': lowDt_upper, 'highDt_lower': highDt_lower,
                'msd_N_tracks': msd_N_tracks, 'msd_tc': msd_tc, 
                'msd_D_lin': msd_D_lin, 'msd_D_lin_highDt': msd_D_lin_highDt,
                'msd_k_full': msd_k_full, 'msd_D_full': msd_D_full,
                'msd_k_lowDt': msd_k_lowDt, 'msd_D_lowDt': msd_D_lowDt,
                'msd_k_highDt': msd_k_highDt, 'msd_D_highDt': msd_D_highDt, 
                }
            
            ufun.dict2json(dict_MSDfits, Dict_rP['dir_msd'], msdFitsName)
            for k, v in dict_MSDfits.items():
                DF_RES.loc[(DF_RES['tN'] == tN), k] = v

    
        #### Import tracks, compute pairs, fit pair-MSD
        if Do_PairMSD:
            print(pm.BRIGHTORANGE + '5. pair-MSD compute & fit step' + pm.NORMAL)            
            if Use_DriftCorrected:
                trackName = fN + Dict_fSuf['suf_cleanTracks_DriftCorr'] + '.csv'
            else:
                trackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
            
            df = pd.read_csv(os.path.join(Dict_rP['dir_cleanTracks'], trackName), sep='\t')
            pmsd_N_tracks = len(df['particle'].unique())
            
            # Compute Pair MSD
            Nframes = np.max(df['frame'].to_numpy()) 
            len_TRanges = int(FPS * pairMSD_Dt_s)
            dist_th_um = pairMSD_dist_th_um
            df_allTRange_pair_MSD = tbca.compute_pairwise_MSD(
                df, PixPerUm, FPS, Nframes,
                len_TRanges = len_TRanges, delta_TRanges = -1, 
                dist_th_um = dist_th_um, max_Dt_s = max_Dt_s
                )
            
            pairMsdName = fN + Dict_fSuf['suf_globalPairMsd'] + '.csv'
            pairMsdFitsName = fN + Dict_fSuf['suf_globalPairMsdFits'] + '.json'
            
            df_allTRange_pair_MSD.to_csv(os.path.join(Dict_rP['dir_pairMsd'], pairMsdName), 
                                         index=False, sep='\t')
            
            # Fit Pair MSD
            T = df_allTRange_pair_MSD['lagt'].to_numpy()
            MSD = df_allTRange_pair_MSD['msd_median'].to_numpy()
            
            iLow = ufun.findFirst(lowDt_upper, T) + 1
            iHigh = ufun.findFirst(highDt_lower, T)
            
            parms, results = ufun.fitLineHuber(T, MSD, with_intercept = False)
            pmsd_D_lin = parms[0]/8
            
            parms, results = ufun.fitLineHuber(
                np.log(T), np.log(MSD), 
                with_intercept = True)
            b, a = parms
            pmsd_k_full = a
            pmsd_D_full = np.exp(b)/8
            
            if len(T[:iLow]) >= 3:
                parms, results = ufun.fitLineHuber(
                    np.log(T[:iLow]), np.log(MSD[:iLow]), 
                    with_intercept = True)
                b, a = parms
                pmsd_k_lowDt = a
                pmsd_D_lowDt = np.exp(b)/8
            else:
                pmsd_k_lowDt, pmsd_D_lowDt = np.nan, np.nan
            
            if len(T[iHigh:]) >= 6:
                parms, results = ufun.fitLineHuber(
                    T[iHigh:], MSD[iHigh:], 
                    with_intercept = False)
                pmsd_D_lin_highDt = parms[0]/8
                
                parms, results = ufun.fitLineHuber(
                    np.log(T[iHigh:]), np.log(MSD[iHigh:]), 
                    with_intercept = True)
                b, a = parms
                pmsd_k_highDt = a
                pmsd_D_highDt = np.exp(b)/8
            else:
                pmsd_k_highDt, pmsd_D_highDt = np.nan, np.nan
                
            pmsd_tc = (pmsd_D_lowDt/pmsd_D_highDt)**(1/(pmsd_k_highDt - pmsd_k_lowDt))
            
            dict_pairMSDfits = {
                # 'max_Dt_s': max_Dt_s, 'lowDt_upper': lowDt_upper, 'highDt_lower': highDt_lower,
                'pmsd_N_tracks': pmsd_N_tracks, 'pmsd_tc': pmsd_tc,
                'pmsd_D_lin': pmsd_D_lin, 'pmsd_D_lin_highDt': pmsd_D_lin_highDt,
                'pmsd_k_full': pmsd_k_full, 'pmsd_D_full': pmsd_D_full,
                'pmsd_k_lowDt': pmsd_k_lowDt, 'pmsd_D_lowDt': pmsd_D_lowDt,
                'pmsd_k_highDt': pmsd_k_highDt, 'pmsd_D_highDt': pmsd_D_highDt,
                }
            
            ufun.dict2json(dict_pairMSDfits, Dict_rP['dir_pairMsd'], pairMsdFitsName)
            for k, v in dict_pairMSDfits.items():
                DF_RES.loc[(DF_RES['tN'] == tN), k] = v
        
        #### End of the loop -> Save the dataframe
        DF_RES.to_csv(os.path.join(RES_DIR, df_res_name + '.csv'), index=False)
                
    #### End of the function
    pass
    

# %% Plotting functions

def Tpf_str2num(tpf_str):
    L = tpf_str.split('min')
    try:
        tpf_num = int(L[0])*60
        if len(L) > 1 and len(L[1]) > 0:
            tpf_num += int(L[1])
    except:
        tpf_num = 0
    return(tpf_num)


def Tpf_label2num(fileName_str):
    if 'min' in fileName_str:
        L = fileName_str.split('_')
        for s in L:
            if 'min' in s:
                tpf_num = Tpf_str2num(s)
                break
    else:
        tpf_num = 0
    return(tpf_num)


def import_and_plot_MSD(fileNames, #tifPaths,
                        Dict_fSuf, Dict_rP, Dict_pP,
                        fig_title = '',
                        # Use_DriftCorrected = False,
                        ):    
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    dict_res = {
        'label'  : [],
        'D_linear': [],
        'D_full' : [],
        'k_full' : [],
        'D_lowDt': [],
        'k_lowDt': [],
        'D_highDt':[],
        'k_highDt':[],
        'Tc':[],
        }
    
    ColorList = list(sns.color_palette("husl", len(fileNames)))
    
    for i in range(len(fileNames)):
        fN = fileNames[i]
        # tP = tifPaths[i]
        
        msdName = fN + Dict_fSuf['suf_globalMsd'] + '.csv'
        msdFitsName = fN + Dict_fSuf['suf_globalMsdFits'] + '.json'
        # if not Use_DriftCorrected:
        #     trackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
        # else:
        #     trackName = fN + Dict_fSuf['suf_cleanTracks_DriftCorr'] + '.csv'
        # df = pd.read_csv(os.path.join(dstDir, trackName), sep='\t')
        
        res_emsd = pd.read_csv(os.path.join(dstDir, msdName), sep='\t')
        T, MSD = res_emsd['lagt'], res_emsd['msd']
    
        long_label = '_'.join(fN.split('_')[:-1])
        short_label = '_'.join(fN.split('_')[2:4])
    
        ax.plot(T, MSD, label=short_label, color=ColorList[i], 
                marker='o', markersize=3, alpha=0.85)
        
        dict_MSDfits = ufun.json2dict(dstDir, msdFitsName)
        D_linear = dict_MSDfits['D_linear']
        k_full = dict_MSDfits['k_full']
        D_full = dict_MSDfits['D_full']
        k_lowDt = dict_MSDfits['k_lowDt']
        D_lowDt = dict_MSDfits['D_lowDt']
        k_highDt = dict_MSDfits['k_highDt']
        D_highDt = dict_MSDfits['D_highDt']
        
        Tc = (D_lowDt/D_highDt)**(1/(k_highDt-k_lowDt))
        
        dict_res['label'].append(long_label)
        dict_res['D_linear'].append(D_linear)
        dict_res['D_full'].append(D_full)
        dict_res['k_full'].append(k_full)
        dict_res['D_lowDt'].append(D_lowDt)
        dict_res['k_lowDt'].append(k_lowDt)
        dict_res['D_highDt'].append(D_highDt)
        dict_res['k_highDt'].append(k_highDt)
        dict_res['Tc'].append(Tc)
        
    Xp1 = np.array([1e-1, 5e-1])
    Xp2 = np.array([1, 2])
    ax.plot(Xp1, 12e-3*Xp1**0.5, color = 'gray', ls=':', label=r'$y \propto x^{1/2}$')
    ax.plot(Xp2, 0.15e-1*Xp2**1, color = 'gray', ls='--', label=r'$y \propto x^{1}$')
    ncols = 1 + (len(fileNames)//18)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
              fontsize=8, ncols=ncols)
    ax.grid()
    ax.set_ylabel('MSD (µm²)')
    ax.set_xlabel(r'$\Delta t$ (s)')
    # ax.set_xlim([0.4e-1, 0.6e1])
    # ax.set_ylim([2e-3, 0.5e0])
    if len(fig_title)>0:
        fig.suptitle(fig_title)
    
    df_Diffusion = pd.DataFrame(dict_res)
    df_Diffusion['Tpf_s'] = df_Diffusion['label'].apply(lambda x : Tpf_label2num(x))
    df_Diffusion['Tpf_min'] = df_Diffusion['Tpf_s']/60
    
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), 
                             sharex = True, layout='compressed')
    ax = axes[0]
    ax.plot(df_Diffusion.Tpf_min, df_Diffusion.D_full, ls='-', marker='o', label=r'All $\Delta t$')
    ax.plot(df_Diffusion.Tpf_min, df_Diffusion.D_lowDt, ls='-', marker='o', label=r'$\Delta t \leq 0.5s$')
    ax.plot(df_Diffusion.Tpf_min, df_Diffusion.D_highDt, ls='-', marker='o', label=r'$\Delta t \geq 1s$')
    ax.plot(df_Diffusion.Tpf_min, df_Diffusion.D_linear, ls='-', marker='o', label=r'Linear fit')
    ax.set_ylabel(r'$D_{eff}\ (\mu m^2/s^\alpha)$')
    ym, yM = ax.get_ylim()
    ax.set_ylim([0, yM])
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
    ym, yM = ax.get_ylim()
    ax.set_ylim([0, yM])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid()
    if len(fig_title)>0:
        fig.suptitle(fig_title)
    
    plt.show()
    
    return(df_Diffusion)



def import_and_plot_pairMSD(fileNames, Dict_fSuf, fig_title=''):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    dict_res_MSD = {
        'label'  : [], 'D_linear' : [],
        'D_full' : [], 'k_full' : [],
        'D_lowDt': [], 'k_lowDt': [],
        'D_highDt':[], 'k_highDt':[],
        'Tc' : [],
        }
    
    dict_res_pMSD = {
        'label'  : [], 'D_linear' : [],
        'D_full' : [], 'k_full' : [],
        'D_lowDt': [], 'k_lowDt': [],
        'D_highDt':[], 'k_highDt':[],
        'Tc' : [],
        }
    
    ColorList = list(sns.color_palette("husl", len(fileNames)))
    
    for i in range(len(fileNames)):
        fN = fileNames[i]
        msdName = fN + Dict_fSuf['suf_globalMsd'] + '.csv'
        msdFitsName = fN + Dict_fSuf['suf_globalMsdFits']
        pairMsdName = fN + Dict_fSuf['suf_globalPairMsd'] + '.csv'
        pairMsdFitsName = fN + Dict_fSuf['suf_globalPairMsdFits']
        
        # cleanTrackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
        # df = pd.read_csv(os.path.join(dstDir, cleanTrackName), sep='\t')
        
        # "Normal" MSD
        res_emsd = pd.read_csv(os.path.join(dstDir, msdName), sep='\t')
        T, MSD = res_emsd['lagt'], res_emsd['msd']
        dict_MSDfits = ufun.json2dict(dstDir, msdFitsName)
        long_label = '_'.join(fN.split('_')[:6])
        short_label = '_'.join(fN.split('_')[2:4])
        
        ax = axes[0]
        ax.plot(T, MSD, label=short_label, color=ColorList[i], 
                marker='o', markersize=3, alpha=0.85)
                
        D_linear = dict_MSDfits['D_linear']
        k_full = dict_MSDfits['k_full']
        D_full = dict_MSDfits['D_full']
        k_lowDt = dict_MSDfits['k_lowDt']
        D_lowDt = dict_MSDfits['D_lowDt']
        k_highDt = dict_MSDfits['k_highDt']
        D_highDt = dict_MSDfits['D_highDt']
        
        Tc = (D_lowDt/D_highDt)**(1/(k_highDt-k_lowDt))
        
        dict_res_MSD['label'].append(long_label)
        dict_res_MSD['Tc'].append(Tc)
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
        long_label = '_'.join(fN.split('_')[:6])
        short_label = '_'.join(fN.split('_')[2:4])
        
        ax = axes[1]
        ax.plot(pT, pMSD, label=short_label, color=ColorList[i], 
                marker='o', markersize=3, alpha=0.85)
        
        D_linear = dict_pairMSDfits['D_linear']
        k_full = dict_pairMSDfits['k_full']
        D_full = dict_pairMSDfits['D_full']
        k_lowDt = dict_pairMSDfits['k_lowDt']
        D_lowDt = dict_pairMSDfits['D_lowDt']
        k_highDt = dict_pairMSDfits['k_highDt']
        D_highDt = dict_pairMSDfits['D_highDt']
        
        Tc = (D_lowDt/D_highDt)**(1/(k_highDt-k_lowDt))

        dict_res_pMSD['label'].append(long_label)
        dict_res_pMSD['Tc'].append(Tc)
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
    ncols = 1 + (len(fileNames)//18)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
              fontsize=8, ncols = ncols)
    if len(fig_title)>0:
        fig.suptitle(fig_title)
    
    
    # Time - D - alpha
    df_Diffusion = pd.DataFrame(dict_res_MSD)
    df_Diffusion['Tpf_s'] = df_Diffusion['label'].apply(lambda x : Tpf_label2num(x))
    df_Diffusion['Tpf_min'] = df_Diffusion['Tpf_s']/60
    
    df_pairDiffusion = pd.DataFrame(dict_res_pMSD)
    df_pairDiffusion['Tpf_s'] = df_Diffusion['label'].apply(lambda x : Tpf_label2num(x))
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
    ym, yM = ax.get_ylim()
    ax.set_ylim([0, yM])
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
    ym, yM = ax.get_ylim()
    ax.set_ylim([0, yM])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid()
    if len(fig_title)>0:
        fig.suptitle(fig_title)
    
    plt.show()

    return(df_Diffusion, df_pairDiffusion)




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


def plot_MSD_HeatMap(df_grid, M_boxes, xy_col, parm_col,
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

# %% Other functions

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



# For Claude ! The function below is the one i want to complete
def NByolk_imsd_compute(tifNames, df_films_path, 
                       Dict_fSuf, Dict_rP, Dict_pP):
    
    DF_FILMS = pd.read_csv(df_films_path)
    
    fileNames = [fN.split('.')[0] for fN in tifNames]
    # tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
    
    for i in range(len(fileNames)):
        tN, fN = tifNames[i], fileNames[i]
        # tP = tifPaths[i]
        print(pm.CYAN + f'\nFile {i+1:.0f}/{len(fileNames):.0f} - {fN}' + pm.NORMAL)
        
        idx = DF_FILMS[DF_FILMS['tN'] == tN].index[0]
        PixPerUm = DF_FILMS.loc[idx, 'PixPerUm']
        FPS = DF_FILMS.loc[idx, 'FPS']
        Use_DriftCorrected = DF_FILMS.loc[idx, 'drift_correction']
        max_Dt_s = DF_FILMS.loc[idx, 'msd_max_Dt_s']
        lowDt_upper = DF_FILMS.loc[idx, 'msd_lowDt_upper_s']
        highDt_lower = DF_FILMS.loc[idx, 'msd_highDt_lower_s']
    
    
        #### Import tracks, run trackpy.imsd, fit MSD
        # rawTrackName = fN + Dict_fSuf['suf_rawTracks'] + '.xml'
        if Use_DriftCorrected:
            trackName = fN + Dict_fSuf['suf_cleanTracks_DriftCorr'] + '.csv'
        else:
            trackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
            
        df = pd.read_csv(os.path.join(Dict_rP['dir_cleanTracks'], trackName), sep='\t')
        
        #### A. Compute individual MSD 
        print(pm.BRIGHTORANGE + 'A. Compute individual MSD... ' + pm.NORMAL, end='')
        top = time.time()
        
        max_lagtime = int(max_Dt_s * FPS)
        res_imsd = tp.motion.imsd(df, 1/PixPerUm, FPS, max_lagtime=max_lagtime
            ).reset_index().rename(columns={'lag time [s]':'lagt'})
        
        imsdName     = fN + Dict_fSuf['suf_indivMsd']     + '.csv'
        res_imsd.to_csv(os.path.join(Dict_rP['dir_indivMsd'], imsdName),     index=False, sep='\t')
        print(f'{time.time()-top:.2f} s')
        
        #### B. Fit metrics
        print(pm.BRIGHTORANGE + 'B. Fit metrics... ' + pm.NORMAL, end='')
        top = time.time()
        
        imsd_rows = []
        T_all = res_imsd['lagt'].values  # common lag-time axis for all particles

        iLow = ufun.findFirst(lowDt_upper, T_all) + 1
        iHigh = ufun.findFirst(highDt_lower, T_all)

        for particle_id in res_imsd.columns:
            if particle_id == 'lagt':
                continue

            MSD = res_imsd[particle_id].values

            # Skip particles with too many NaN or zero values
            valid = np.isfinite(MSD) & (MSD > 0)
            if valid.sum() < 6:
                continue

            T = T_all[valid]
            MSD = MSD[valid]

            row = {'particle': particle_id}

            # --- Linear fit over full range (2D: MSD = 4·D·Δt) ---
            try:
                parms, _ = ufun.fitLineHuber(T, MSD, with_intercept=False)
                row['D_lin'] = parms[0] / 4
            except Exception:
                row['D_lin'] = np.nan

            # --- Log-log fit over full range ---
            try:
                parms, _ = ufun.fitLineHuber(np.log(T), np.log(MSD), with_intercept=True)
                b, a = parms
                row['k_full'] = a
                row['D_full'] = np.exp(b) / 4
            except Exception:
                row['k_full'], row['D_full'] = np.nan, np.nan

            # --- Log-log fit over low-Δt range ---
            # Re-compute valid mask restricted to low-Δt slice
            valid_low = np.isfinite(res_imsd[particle_id].values[:iLow]) & \
                        (res_imsd[particle_id].values[:iLow] > 0)
            T_low  = T_all[:iLow][valid_low]
            MSD_low = res_imsd[particle_id].values[:iLow][valid_low]

            if len(T_low) >= 3:
                try:
                    parms, _ = ufun.fitLineHuber(np.log(T_low), np.log(MSD_low), with_intercept=True)
                    b, a = parms
                    row['k_lowDt'] = a
                    row['D_lowDt'] = np.exp(b) / 4
                except Exception:
                    row['k_lowDt'], row['D_lowDt'] = np.nan, np.nan
            else:
                row['k_lowDt'], row['D_lowDt'] = np.nan, np.nan

            # --- Linear + log-log fits over high-Δt range ---
            valid_high = np.isfinite(res_imsd[particle_id].values[iHigh:]) & \
                         (res_imsd[particle_id].values[iHigh:] > 0)
            T_high   = T_all[iHigh:][valid_high]
            MSD_high = res_imsd[particle_id].values[iHigh:][valid_high]

            if len(T_high) >= 6:
                try:
                    parms, _ = ufun.fitLineHuber(T_high, MSD_high, with_intercept=False)
                    row['D_lin_highDt'] = parms[0] / 4
                except Exception:
                    row['D_lin_highDt'] = np.nan

                try:
                    parms, _ = ufun.fitLineHuber(np.log(T_high), np.log(MSD_high), with_intercept=True)
                    b, a = parms
                    row['k_highDt'] = a
                    row['D_highDt'] = np.exp(b) / 4
                except Exception:
                    row['k_highDt'], row['D_highDt'] = np.nan, np.nan
            else:
                row['D_lin_highDt'] = np.nan
                row['k_highDt'], row['D_highDt'] = np.nan, np.nan

            # --- Cross-over time ---
            try:
                row['tc'] = (row['D_lowDt'] / row['D_highDt']) ** \
                            (1 / (row['k_highDt'] - row['k_lowDt']))
            except Exception:
                row['tc'] = np.nan

            imsd_rows.append(row)

        df_imsd_fits = pd.DataFrame(imsd_rows)

        # Save the full imsd table (lag-times × particles) and the per-particle fit results
        imsdFitsName = fN + Dict_fSuf['suf_indivMsdFits'] + '.csv'
        df_imsd_fits.to_csv(os.path.join(Dict_rP['dir_indivMsd'], imsdFitsName), index=False, sep='\t')
        print(f'{time.time()-top:.2f} s')
        
        
# V2
def NByolk_ro_msd_compute(tifNames, df_films_path, df_res_name,
                          Dict_fSuf, Dict_rP, Dict_pP):
    
    DF_FILMS = pd.read_csv(df_films_path)
    
    fileNames = [fN.split('.')[0] for fN in tifNames]
    # tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
    
    for i in range(len(fileNames)):
        tN, fN = tifNames[i], fileNames[i]
        # tP = tifPaths[i]
        print(pm.CYAN + f'\nFile {i+1:.0f}/{len(fileNames):.0f} - {fN}' + pm.NORMAL)
        
        try:
            DF_RES = pd.read_csv(os.path.join(RES_DIR, df_res_name + '.csv')).reset_index(drop=True)
        except:
            DF_RES = pd.DataFrame({})
            
        if ('tN' not in DF_RES.columns) or (tN not in DF_RES['tN'].to_numpy()):
            DF_RES.loc[len(DF_RES), 'tN'] = tN
        
        idx = DF_FILMS[DF_FILMS['tN'] == tN].index[0]
        PixPerUm = DF_FILMS.loc[idx, 'PixPerUm']
        FPS = DF_FILMS.loc[idx, 'FPS']
        Use_DriftCorrected = DF_FILMS.loc[idx, 'drift_correction']
        max_Dt_s = DF_FILMS.loc[idx, 'msd_max_Dt_s']
        lowDt_upper = DF_FILMS.loc[idx, 'msd_lowDt_upper_s']
        highDt_lower = DF_FILMS.loc[idx, 'msd_highDt_lower_s']     
            
        Nx, Ny = DF_FILMS.loc[idx, 'im_width'], DF_FILMS.loc[idx, 'im_height']
        xC, yC = DF_FILMS.loc[idx, 'x_center'], DF_FILMS.loc[idx, 'y_center']
        if np.isnan(xC) or np.isnan(yC):
            xC, yC = Nx//2, Ny//2

        if Use_DriftCorrected:
            trackName = fN + Dict_fSuf['suf_cleanTracks_DriftCorr'] + '.csv'
        else:
            trackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
            
        df = pd.read_csv(os.path.join(Dict_rP['dir_cleanTracks'], trackName), sep='\t')
        msd_N_tracks = len(df['particle'].unique())
        
        max_lagtime = int(max_Dt_s * FPS)
        
        # For Claude ! COMPLETE CODE HERE
            # 1. Project all trajectories onto radial & orthoradial axes → df_radial, df_orthoradial
            # 2. Run tp.motion.imsd on each projected table (1D, pos_columns=['x'])
            # 3. Fit same metrics as NByolk_msd_compute (divide by 2: MSD_1D = 2·D·Δt)
            # 4. Compute contrast = (parm_r - parm_or) / (parm_r + parm_or) for each metric
            # 5. Collect everything into one per-particle DataFrame

        def fit_1d_msd(T_axis, MSD_values, iLow, iHigh):
            """
            Fit the standard set of MSD metrics on a 1-D MSD curve.
            Divides by 2 instead of 4 (MSD_1D = 2·D·Δt).
            Returns a dict of fit results.
            """
            res = {}
            valid = np.isfinite(MSD_values) & (MSD_values > 0)
            if valid.sum() < 6:
                for key in ['D_lin', 'k_full', 'D_full',
                            'k_lowDt', 'D_lowDt',
                            'D_lin_highDt', 'k_highDt', 'D_highDt', 'tc']:
                    res[key] = np.nan
                return res

            T   = T_axis[valid]
            MSD = MSD_values[valid]

            # Linear fit – full range
            try:
                parms, _ = ufun.fitLineHuber(T, MSD, with_intercept=False)
                res['D_lin'] = parms[0] / 2
            except Exception:
                res['D_lin'] = np.nan

            # Log-log fit – full range
            try:
                parms, _ = ufun.fitLineHuber(np.log(T), np.log(MSD), with_intercept=True)
                b, a = parms
                res['k_full'] = a
                res['D_full'] = np.exp(b) / 2
            except Exception:
                res['k_full'] = res['D_full'] = np.nan

            # Log-log fit – low-Δt range
            valid_low = np.isfinite(MSD_values[:iLow]) & (MSD_values[:iLow] > 0)
            T_low, MSD_low = T_axis[:iLow][valid_low], MSD_values[:iLow][valid_low]
            if len(T_low) >= 3:
                try:
                    parms, _ = ufun.fitLineHuber(np.log(T_low), np.log(MSD_low), with_intercept=True)
                    b, a = parms
                    res['k_lowDt'] = a
                    res['D_lowDt'] = np.exp(b) / 2
                except Exception:
                    res['k_lowDt'] = res['D_lowDt'] = np.nan
            else:
                res['k_lowDt'] = res['D_lowDt'] = np.nan

            # Linear + log-log fits – high-Δt range
            valid_high = np.isfinite(MSD_values[iHigh:]) & (MSD_values[iHigh:] > 0)
            T_high, MSD_high = T_axis[iHigh:][valid_high], MSD_values[iHigh:][valid_high]
            if len(T_high) >= 6:
                try:
                    parms, _ = ufun.fitLineHuber(T_high, MSD_high, with_intercept=False)
                    res['D_lin_highDt'] = parms[0] / 2
                except Exception:
                    res['D_lin_highDt'] = np.nan
                try:
                    parms, _ = ufun.fitLineHuber(np.log(T_high), np.log(MSD_high), with_intercept=True)
                    b, a = parms
                    res['k_highDt'] = a
                    res['D_highDt'] = np.exp(b) / 2
                except Exception:
                    res['k_highDt'] = res['D_highDt'] = np.nan
            else:
                res['D_lin_highDt'] = res['k_highDt'] = res['D_highDt'] = np.nan

            # Cross-over time
            try:
                res['tc'] = (res['D_lowDt'] / res['D_highDt']) ** \
                            (1 / (res['k_highDt'] - res['k_lowDt']))
            except Exception:
                res['tc'] = np.nan

            return res

        # Metric names for which we compute the contrast
        contrast_metrics = ['D_lin', 'k_full', 'D_full',
                             'k_lowDt', 'D_lowDt',
                             'D_lin_highDt', 'k_highDt', 'D_highDt', 'tc']

        # A. Build projected trajectory tables
        print(pm.BRIGHTORANGE + 'A. Build projected trajectory tables... ' + pm.NORMAL, end='')
        top = time.time()
        
        radial_rows      = []
        orthoradial_rows = []
        particle_meta    = {}  # stores per-particle geometry info

        for particle_id, traj in df.groupby('particle'):
            traj = traj.sort_values('frame').reset_index(drop=True)

            x_med = np.median(traj['x'].values)
            y_med = np.median(traj['y'].values)

            dx_ref = x_med - xC
            dy_ref = y_med - yC
            norm_ref = np.hypot(dx_ref, dy_ref)
            angle_ref = np.atan2(dy_ref, dx_ref)

            if norm_ref < 1e-9:
                continue  # particle sits on the centre, skip

            e_r  = np.array([dx_ref, dy_ref]) / norm_ref   # radial unit vector
            e_or = np.array([-dy_ref, dx_ref]) / norm_ref  # orthoradial unit vector (90° CCW)

            pos_r  = traj['x'].values * e_r[0]  + traj['y'].values * e_r[1]
            pos_or = traj['x'].values * e_or[0] + traj['y'].values * e_or[1]

            for frame, xr, xor in zip(traj['frame'].values, pos_r, pos_or):
                radial_rows.append({'frame': frame, 'x': xr,  'particle': particle_id})
                orthoradial_rows.append({'frame': frame, 'x': xor, 'particle': particle_id})

            particle_meta[particle_id] = {
                'x_med_pix': x_med,
                'y_med_pix': y_med,
                'r_ref_pix':  norm_ref,
                'angle_ref_rad':  angle_ref,
            }

        df_radial      = pd.DataFrame(radial_rows)
        df_orthoradial = pd.DataFrame(orthoradial_rows)
        print(f'{time.time()-top:.2f} s')
        
        #### B. Ensemble MSD R & OR
        print(pm.BRIGHTORANGE + 'B. Ensemble MSD R & OR... ' + pm.NORMAL, end='')
        top = time.time()
        
        # --- 1. Run tp.motion.emsd on each projected table ---
        res_emsd_r  = tp.motion.emsd(df_radial, 1/PixPerUm, FPS,
                              max_lagtime=max_lagtime,
                              pos_columns=['x']).reset_index()

        res_emsd_or = tp.motion.emsd(df_orthoradial, 1/PixPerUm, FPS,
                                      max_lagtime=max_lagtime,
                                      pos_columns=['x']).reset_index()
        
        # Common lag-time axis (both tables share the same one)
        T_axis = res_emsd_r['lagt'].values
        iLow   = ufun.findFirst(lowDt_upper,  T_axis) + 1
        iHigh  = ufun.findFirst(highDt_lower, T_axis)
        
        # --- 2 & 3. Fit metrics and compute contrasts, one row per particle ---
        MSD_r  = res_emsd_r['msd'].values
        MSD_or = res_emsd_or['msd'].values
        
        roMsdName = fN + Dict_fSuf['suf_globalROMsd'] + '.csv'
        ro_emsd_combined = res_emsd_r.copy()
        ro_emsd_combined.columns = ['lagt'] + [f'{c}_r'  for c in res_emsd_r.columns[1:]]
        ro_emsd_combined[[f'{c}_or' for c in res_emsd_or.columns[1:]]] = res_emsd_or.iloc[:, 1:].values
        ro_emsd_combined.to_csv(os.path.join(Dict_rP['dir_ROMsd'], roMsdName), index=False, sep='\t')

        fits_r  = fit_1d_msd(T_axis, MSD_r,  iLow, iHigh)
        fits_or = fit_1d_msd(T_axis, MSD_or, iLow, iHigh)

        df_res_row = {}

        for metric, val in fits_r.items():
            df_res_row[f'{metric}_r'] = val
        for metric, val in fits_or.items():
            df_res_row[f'{metric}_or'] = val

        for metric in contrast_metrics:
            v_r  = fits_r.get(metric, np.nan)
            v_or = fits_or.get(metric, np.nan)
            denom = v_r + v_or
            if denom != 0 and np.isfinite(denom):
                df_res_row[f'contrast_{metric}'] = (v_r - v_or) / denom
            else:
                df_res_row[f'contrast_{metric}'] = np.nan
                
        for k, v in df_res_row.items():
            DF_RES.loc[(DF_RES['tN'] == tN), k] = v
    
        #### -> Save the dataframe DF_RES
        DF_RES.to_csv(os.path.join(RES_DIR, df_res_name + '.csv'), index=False)
        print(f'{time.time()-top:.2f} s')
        
        #### C. Individual MSD R & OR
        print(pm.BRIGHTORANGE + 'C. Individual MSD R & OR... ' + pm.NORMAL, end='')
        top = time.time()
        
        # --- 1. Run tp.motion.imsd on each projected table ---
        res_imsd_r  = tp.motion.imsd(df_radial, 1/PixPerUm, FPS,
                                     max_lagtime=max_lagtime,
                                     pos_columns=['x']).reset_index().rename(columns={'lag time [s]':'lagt'})

        res_imsd_or = tp.motion.imsd(df_orthoradial, 1/PixPerUm, FPS,
                                     max_lagtime=max_lagtime,
                                     pos_columns=['x']).reset_index().rename(columns={'lag time [s]':'lagt'})

        # Save the full imsd tables (lag-times × particles) to disk
        roMsdName = fN + Dict_fSuf['suf_indivROMsd'] + '.csv'
        ro_imsd_combined = res_imsd_r.copy()
        ro_imsd_combined.columns = ['lagt'] + [f'{c}_r'  for c in res_imsd_r.columns[1:]]
        ro_imsd_combined[[f'{c}_or' for c in res_imsd_or.columns[1:]]] = res_imsd_or.iloc[:, 1:].values
        ro_imsd_combined.to_csv(os.path.join(Dict_rP['dir_ROMsd'], roMsdName), index=False, sep='\t')

        # Common lag-time axis (both tables share the same one)
        T_axis = res_imsd_r['lagt'].values
        iLow   = ufun.findFirst(lowDt_upper,  T_axis) + 1
        iHigh  = ufun.findFirst(highDt_lower, T_axis)

        # --- 2 & 3. Fit metrics and compute contrasts, one row per particle ---
        ro_rows = []

        for particle_id in particle_meta:
            if particle_id not in res_imsd_r.columns or particle_id not in res_imsd_or.columns:
                continue

            MSD_r  = res_imsd_r[particle_id].values
            MSD_or = res_imsd_or[particle_id].values

            fits_r  = fit_1d_msd(T_axis, MSD_r,  iLow, iHigh)
            fits_or = fit_1d_msd(T_axis, MSD_or, iLow, iHigh)

            row = {'particle': particle_id, **particle_meta[particle_id]}

            for metric, val in fits_r.items():
                row[f'{metric}_r'] = val
            for metric, val in fits_or.items():
                row[f'{metric}_or'] = val

            for metric in contrast_metrics:
                v_r  = fits_r.get(metric, np.nan)
                v_or = fits_or.get(metric, np.nan)
                denom = v_r + v_or
                if denom != 0 and np.isfinite(denom):
                    row[f'contrast_{metric}'] = (v_r - v_or) / denom
                else:
                    row[f'contrast_{metric}'] = np.nan

            ro_rows.append(row)

        df_ro_fits = pd.DataFrame(ro_rows)

        roMsdFitsName = fN + Dict_fSuf['suf_indivROMsdFits'] + '.csv'
        df_ro_fits.to_csv(os.path.join(Dict_rP['dir_ROMsd'], roMsdFitsName), index=False, sep='\t')
        print(f'{time.time()-top:.2f} s')
        

# %% 1. Tracking and MSD

# %%% 26-10-28_D2-s - Ctrl ?

# %%%% Settings

dateDir = os.path.join(up.Path_IntraCellTracking, '26-10-28_FastAcq-Channel_Fec_NB-Yolk_FM143-Endo')
srcDir = os.path.join(dateDir, 'D2-s')
os.listdir(srcDir)

tifNames = [
    '26-10-28_D2-s_PreF_0min_C1_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-28_D2-s_PreF_0min_C2_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-28_D2-s_PostF_2min_C3_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-28_D2-s_PostF_5min_C3_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-28_D2-s_PostF_8min_C3_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-28_D2-s_PostF_11min_C3_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-28_D2-s_PostF_15min_C3_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-28_D2-s_PostF_18min_C3_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-28_D2-s_PostF_24min_C3_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-28_D2-s_PostF_28min_C3_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-28_D2-s_PostF_35min_C3_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    ]

tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
fileNames = [fN.split('.')[0] for fN in tifNames]

# %%%% Run table autofill

autofill_SPT_Data(srcDir, tifNames, df_films_PATH)

# %%%% Run new Tracking

#### Settings
Dict_TrackMate_Settings = {
    'RADIUS_UM' : 0.8, 'THRESH_SPOT_QLT' : 0.25,
    'THRESH_LINK_UM' : 0.25, 'THRESH_MIN_DURATION' : 30,
    }

NByolk_track_sequence(
    srcDir, tifNames, df_films_PATH, 
    Dict_fSuf, Dict_rP, Dict_pP, 
    Dict_TrackMate_Settings, 
    Do_Contours = True, Do_Tracking = True, 
    Do_TrackCleaning = True, 
    )

# %%%% Run new MSD

df_res_name = 'SPT_Results_01'
NByolk_msd_compute(tifNames, df_films_PATH, df_res_name,
                    Dict_fSuf, Dict_rP, Dict_pP, 
                    Do_MSD = True, Do_PairMSD = True)

# %%% 26-10-07_D1-L - Noco

# %%%% Settings

#### Paths
dateDir = os.path.join(up.Path_IntraCellTracking, '26-10-07_FastAcq-Channel_Fec_NB-Yolk')
srcDir = os.path.join(dateDir, 'D1-L')

tifNames = [
    '26-10-07_D1-L_PreF_0min_C1_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PreF_0min_C2_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PreF_0min_C3_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_3min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tif',
    '26-10-07_D1-L_PostF_7min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_12min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_17min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_20min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_25min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_29min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_35min_noco_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_38min_noco_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_45min_nocoWashed_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_51min_nocoWashed2_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_55min_nocoWashed2_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_60min_nocoWashed2_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_65min_nocoWashed2_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_70min_nocoWashed2_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_75min_nocoWashed2_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_80min_nocoWashed2_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_85min_nocoWashed2_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_90min_nocoWashed3_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_95min_nocoWashed3_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_100min_nocoWashed3_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_124min_nocoWashed3_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    ]

tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
fileNames = [fN.split('.')[0] for fN in tifNames]

# %%%% Run table autofill

autofill_SPT_Data(srcDir, tifNames, df_films_PATH)

# %%%% Run new Tracking

#### Settings
Dict_TrackMate_Settings = {
    'RADIUS_UM' : 0.8, 'THRESH_SPOT_QLT' : 0.25,
    'THRESH_LINK_UM' : 0.25, 'THRESH_MIN_DURATION' : 30,
    }

NByolk_track_sequence(
    srcDir, tifNames, df_films_PATH, 
    Dict_fSuf, Dict_rP, Dict_pP, 
    Dict_TrackMate_Settings, 
    Do_Contours = True, Do_Tracking = True, 
    Do_TrackCleaning = True, 
    )

# %%%% Run new MSD

df_res_name = 'SPT_Results_01'
NByolk_msd_compute(tifNames, df_films_PATH, df_res_name,
                    Dict_fSuf, Dict_rP, Dict_pP, 
                    Do_MSD = True, Do_PairMSD = True)

# %%%% Run iMSD and R-OR MSD

NByolk_imsd_compute(tifNames, df_films_PATH,
                       Dict_fSuf, Dict_rP, Dict_pP)

df_res_name = 'SPT_Results_01'
NByolk_ro_msd_compute(tifNames, df_films_PATH, df_res_name,
                          Dict_fSuf, Dict_rP, Dict_pP)

# %%%% Import MSD & plot / Import pairMSD & plot

title = '26-10-07_D1-L_Noco'
df_diff = import_and_plot_MSD(fileNames, Dict_fSuf, fig_title=title)
df_diff, df_pairDiff = import_and_plot_pairMSD(fileNames, Dict_fSuf, fig_title=title)

for i in plt.get_fignums():
    plt.figure(i)
    plt.savefig(os.path.join(FIG_DIR, title + f'_MSDfig_{i:02d}.png'))


# %%%% Import tracks & analyse shape of explored zone

Use_DriftCorrected = False

mainDir = os.path.join(up.Path_IntraCellTracking, '26-10-07_FastAcq-Channel_Fec_NB-Yolk')
srcDir = os.path.join(mainDir, 'D1-L')
dstDir = os.path.join(mainDir, 'SPT_results')

tifNames = [
    # '26-10-07_D1-L_PreF_0min_C1_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-10-07_D1-L_PreF_0min_C2_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-10-07_D1-L_PreF_0min_C3_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-10-07_D1-L_PostF_3min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tif',
    # '26-10-07_D1-L_PostF_7min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_12min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-L_PostF_17min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-L_PostF_20min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-L_PostF_25min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_29min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-L_PostF_35min_noco_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_38min_noco_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-L_PostF_45min_nocoWashed_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-L_PostF_51min_nocoWashed2_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_55min_nocoWashed2_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-L_PostF_60min_nocoWashed2_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-L_PostF_65min_nocoWashed2_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-L_PostF_70min_nocoWashed2_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-L_PostF_75min_nocoWashed2_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-L_PostF_80min_nocoWashed2_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-L_PostF_85min_nocoWashed2_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-L_PostF_90min_nocoWashed3_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-L_PostF_95min_nocoWashed3_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-L_PostF_100min_nocoWashed3_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-L_PostF_124min_nocoWashed3_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    ]

tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
fileNames = [fN.split('.')[0] for fN in tifNames]

dict_res = {'label':[],
            'AngleDiffs':[]}

GEOM_DATA = []

df_g = pd.read_csv(os.path.join(srcDir, 'Cell_geometry.csv'), sep=',')


for i in range(len(tifNames)):
    print(i)
    tP, fN = tifPaths[i], fileNames[i]
    _, tN = os.path.split(tP)
    shape, dtype = ufun.tiff_inspect(tP)
    
    X_MTcenter = df_g.loc[df_g['tN']==tN, 'xc'].to_numpy()[0]
    Y_MTcenter = df_g.loc[df_g['tN']==tN, 'yc'].to_numpy()[0]

    title = '_'.join(fN.split('_')[1:3])
    rawTrackName = fN + Dict_fSuf['suf_rawTracks'] + '.xml'
    if Use_DriftCorrected:
        trackName = fN + Dict_fSuf['suf_driftCorrTracks'] + '.csv'
    else:
        trackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
        
    # msdName = fN + Dict_fSuf['suf_globalMsd'] + '.csv'
    # msdFitsName = fN + Dict_fSuf['suf_globalMsdFits']
    if np.isnan(X_MTcenter) or np.isnan(Y_MTcenter):
        contourName = fN + Dict_fSuf['suf_contour'] + '.npy'
        contourCell = np.load(os.path.join(srcDir, contourName))
        poly = Polygon(contourCell)
        xc, yc = poly.centroid.xy
        X_MTcenter, Y_MTcenter = int(xc[0]), int(yc[0])
    
    df = pd.read_csv(os.path.join(dstDir, trackName), sep='\t')
    
    # Lp = df['particle'].unique().astype(int)
    dict_geom = {'particle':[],
                 'np':[],
                 'xc':[],
                 'yc':[],
                 'theta':[],
                 'a':[],
                 'b':[],
                 'L':[],
                 'l':[],
                 'AR':[],
                 'phi':[],}
    
    grouped = df.groupby('particle')
    
    for p, df_p in grouped:
        points = np.array([[x, y] for (x, y) in zip(df_p.x, df_p.y)])
        
        MPoints = MultiPoint(points)
        Hull = MPoints.convex_hull
        xy_ch = np.array([xy for xy in Hull.exterior.coords[1:]])
        xc, yc = Hull.centroid.coords[0]
        
        out = ufun.fit_ellipse_fixedCenter(xy_ch[:, 0], xy_ch[:, 1], 
                                           xc, yc, mode='cartesian')
        _, _, a, b, phi = out
        theta = np.atan2(yc - Y_MTcenter, xc - X_MTcenter)
        theta_deg = theta * 180/np.pi
        dotprod = np.cos(theta)*np.cos(phi) + np.sin(theta)*np.sin(phi)
        if dotprod < 0:
            phi = phi - np.sign(phi) * np.pi
        phi_deg = phi * 180/np.pi
        L, l = max(a, b), min(a, b)
        AR = L/l
        
        dict_geom['particle'].append(p)
        dict_geom['np'].append(len(df_p))
        dict_geom['xc'].append(xc)
        dict_geom['yc'].append(yc)
        dict_geom['theta'].append(theta)
        dict_geom['a'].append(a)
        dict_geom['b'].append(b)
        dict_geom['L'].append(L)
        dict_geom['l'].append(l)
        dict_geom['AR'].append(AR)
        dict_geom['phi'].append(phi)
    
    df_geom_raw = pd.DataFrame(dict_geom)
    GEOM_DATA.append(df_geom_raw)
    
    df_geom = df_geom_raw[df_geom_raw['AR'] > 1.5]    
    df_geom['theta_bin'] = (df_geom['theta'].values * 18/np.pi).astype(int) * 10 + 5
    C = np.cos(df_geom['theta'].values) * np.cos(df_geom['phi'].values) + \
        np.sin(df_geom['theta'].values) * np.sin(df_geom['phi'].values)
    DA = np.acos(C)   
    
    dict_res['label'].append(title)
    dict_res['AngleDiffs'].append(DA)



pm.setGraphicOptions(mode='screen')
fig, axes = plt.subplots(2, len(tifNames), figsize=(len(tifNames)*3, 6),
                         layout='compressed')
colors = pm.cL_Set21

for i in range(len(tifNames)):
    print(i)
    tP, fN = tifPaths[i], fileNames[i]
    _, tN = os.path.split(tP)
    shape, dtype = ufun.tiff_inspect(tP)
    
    X_MTcenter = df_g.loc[df_g['tN']==tN, 'xc'].to_numpy()[0]
    Y_MTcenter = df_g.loc[df_g['tN']==tN, 'yc'].to_numpy()[0]
    # print(X_MTcenter, Y_MTcenter)

    title = '_'.join(fN.split('_')[1:4])
    rawTrackName = fN + Dict_fSuf['suf_rawTracks'] + '.xml'
    if Use_DriftCorrected:
        trackName = fN + Dict_fSuf['suf_driftCorrTracks'] + '.csv'
    else:
        trackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
        
    # msdName = fN + Dict_fSuf['suf_globalMsd'] + '.csv'
    # msdFitsName = fN + Dict_fSuf['suf_globalMsdFits']
    if np.isnan(X_MTcenter) or np.isnan(Y_MTcenter):
        contourName = fN + Dict_fSuf['suf_contour'] + '.npy'
        contourCell = np.load(os.path.join(srcDir, contourName))
        poly = Polygon(contourCell)
        xc, yc = poly.centroid.xy
        X_MTcenter, Y_MTcenter = int(xc[0]), int(yc[0])
    
    # df = pd.read_csv(os.path.join(dstDir, trackName), sep='\t')
    df_geom = GEOM_DATA[i]
    df_geom = df_geom[df_geom['AR'] > 1.5]
    grouped = df_geom.groupby('particle')
    
    ax = axes[0, i]
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    
    for p, df_gp in grouped:      
        xc, yc = df_gp['xc'].to_numpy()[0], df_gp['yc'].to_numpy()[0]
        a, b = df_gp['a'].to_numpy()[0], df_gp['b'].to_numpy()[0]
        phi = df_gp['phi'].to_numpy()[0]
        c = colors[p%len(colors)]       
        if p%10 == 0:
            aa = np.linspace(0, 2*np.pi, 360)
            xp, yp = ufun.get_ellipse_xy(xc, yc, a, b, phi, aa=aa)
            ax.plot(xp, yp, c=c, ls='-', lw=0.5)
    plt.show()
    
    C = np.cos(df_geom['theta'].values) * np.cos(df_geom['phi'].values) + \
        np.sin(df_geom['theta'].values) * np.sin(df_geom['phi'].values)
    DA = np.acos(C)
    
    ax = axes[1, i]
    ax.hist(DA, bins=20)
    ax.set_ylabel('N trajectories')
    ax.set_xlabel(r'$|\theta - \phi|$ (rad)')
    ax.set_xticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
    ax.set_xticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
    ax.grid()

plt.show()



# -----------------------------------------------------------------------------
# %%% 26-10-07_D1-s - ctrl

# %%%% Settings

#### Paths
dateDir = os.path.join(up.Path_IntraCellTracking, '26-10-07_FastAcq-Channel_Fec_NB-Yolk')
srcDir = os.path.join(dateDir, 'D1-s')

tifNames = [
    '26-10-07_D1-s_PreF_0min_C1_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PreF_0min_C2_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PreF_0min_C3_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PreF_0min_C3_5fps_Texp200ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_3min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_6min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_9min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_12min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_15min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_20min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_25min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_30min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_35min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_40min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_45min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_50min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_55min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_60min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_66min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_71min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_83min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    ]

tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
fileNames = [fN.split('.')[0] for fN in tifNames]


# %%%% Run table autofill

autofill_SPT_Data(srcDir, tifNames, df_films_PATH)

# %%%% Run new Tracking

#### Settings
Dict_TrackMate_Settings = {
    'RADIUS_UM' : 0.8, 'THRESH_SPOT_QLT' : 0.25,
    'THRESH_LINK_UM' : 0.25, 'THRESH_MIN_DURATION' : 30,
    }

NByolk_track_sequence(
    srcDir, tifNames, df_films_PATH, 
    Dict_fSuf, Dict_rP, Dict_pP, 
    Dict_TrackMate_Settings, 
    Do_Contours = True, Do_Tracking = True, 
    Do_TrackCleaning = True, 
    )

# %%%% Run new MSD

df_res_name = 'SPT_Results_01'
NByolk_msd_compute(tifNames, df_films_PATH, df_res_name,
                    Dict_fSuf, Dict_rP, Dict_pP, 
                    Do_MSD = True, Do_PairMSD = True)


# %%%% Import MSD & plot / Import pairMSD & plot

title = '26-10-07_D1-s'
df_diff = import_and_plot_MSD(fileNames, Dict_fSuf, fig_title=title)
df_diff, df_pairDiff = import_and_plot_pairMSD(fileNames, Dict_fSuf, fig_title=title)

for i in plt.get_fignums():
    plt.figure(i)
    plt.savefig(os.path.join(FIG_DIR, title + f'_MSDfig_{i:02d}.png'))
    
    
# %%%% Import tracks & analyse shape of explored zone

Use_DriftCorrected = False

mainDir = os.path.join(up.Path_IntraCellTracking, '26-10-07_FastAcq-Channel_Fec_NB-Yolk')
srcDir = os.path.join(mainDir, 'D1-s')
dstDir = os.path.join(mainDir, 'SPT_results')

tifNames = [
    # '26-10-07_D1-s_PreF_0min_C1_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-s_PreF_0min_C2_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-s_PreF_0min_C3_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-s_PreF_0min_C3_5fps_Texp200ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-s_PostF_3min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_6min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-s_PostF_9min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-s_PostF_12min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_15min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-s_PostF_20min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_25min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-s_PostF_30min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-s_PostF_35min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_40min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-s_PostF_45min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-s_PostF_50min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-s_PostF_55min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    '26-10-07_D1-s_PostF_60min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-s_PostF_66min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-s_PostF_71min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    # '26-10-07_D1-s_PostF_83min_C4_20fps_Texp50ms_L30p1_CSU642.ome.tf2',
    ]

tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
fileNames = [fN.split('.')[0] for fN in tifNames]

dict_res = {'label':[],
            'AngleDiffs':[]}

GEOM_DATA = []

df_g = pd.read_csv(os.path.join(srcDir, 'Cell_geometry.csv'), sep=',')


for i in range(len(tifNames)):
    print(i)
    tP, fN = tifPaths[i], fileNames[i]
    _, tN = os.path.split(tP)
    shape, dtype = ufun.tiff_inspect(tP)
    
    X_MTcenter = df_g.loc[df_g['tN']==tN, 'xc'].to_numpy()[0]
    Y_MTcenter = df_g.loc[df_g['tN']==tN, 'yc'].to_numpy()[0]

    title = '_'.join(fN.split('_')[1:3])
    rawTrackName = fN + Dict_fSuf['suf_rawTracks'] + '.xml'
    if Use_DriftCorrected:
        trackName = fN + Dict_fSuf['suf_driftCorrTracks'] + '.csv'
    else:
        trackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
        
    # msdName = fN + Dict_fSuf['suf_globalMsd'] + '.csv'
    # msdFitsName = fN + Dict_fSuf['suf_globalMsdFits']
    if np.isnan(X_MTcenter) or np.isnan(Y_MTcenter):
        contourName = fN + Dict_fSuf['suf_contour'] + '.npy'
        contourCell = np.load(os.path.join(srcDir, contourName))
        poly = Polygon(contourCell)
        xc, yc = poly.centroid.xy
        X_MTcenter, Y_MTcenter = int(xc[0]), int(yc[0])
    
    df = pd.read_csv(os.path.join(dstDir, trackName), sep='\t')
    
    # Lp = df['particle'].unique().astype(int)
    dict_geom = {'particle':[],
                 'np':[],
                 'xc':[],
                 'yc':[],
                 'theta':[],
                 'a':[],
                 'b':[],
                 'L':[],
                 'l':[],
                 'AR':[],
                 'phi':[],}
    
    grouped = df.groupby('particle')
    
    for p, df_p in grouped:
        points = np.array([[x, y] for (x, y) in zip(df_p.x, df_p.y)])
        
        MPoints = MultiPoint(points)
        Hull = MPoints.convex_hull
        xy_ch = np.array([xy for xy in Hull.exterior.coords[1:]])
        xc, yc = Hull.centroid.coords[0]
        
        out = ufun.fit_ellipse_fixedCenter(xy_ch[:, 0], xy_ch[:, 1], 
                                           xc, yc, mode='cartesian')
        _, _, a, b, phi = out
        theta = np.atan2(yc - Y_MTcenter, xc - X_MTcenter)
        theta_deg = theta * 180/np.pi
        dotprod = np.cos(theta)*np.cos(phi) + np.sin(theta)*np.sin(phi)
        if dotprod < 0:
            phi = phi - np.sign(phi) * np.pi
        phi_deg = phi * 180/np.pi
        L, l = max(a, b), min(a, b)
        AR = L/l
        
        dict_geom['particle'].append(p)
        dict_geom['np'].append(len(df_p))
        dict_geom['xc'].append(xc)
        dict_geom['yc'].append(yc)
        dict_geom['theta'].append(theta)
        dict_geom['a'].append(a)
        dict_geom['b'].append(b)
        dict_geom['L'].append(L)
        dict_geom['l'].append(l)
        dict_geom['AR'].append(AR)
        dict_geom['phi'].append(phi)
    
    df_geom_raw = pd.DataFrame(dict_geom)
    GEOM_DATA.append(df_geom_raw)
    
    df_geom = df_geom_raw[df_geom_raw['AR'] > 1.5]    
    df_geom['theta_bin'] = (df_geom['theta'].values * 18/np.pi).astype(int) * 10 + 5
    C = np.cos(df_geom['theta'].values) * np.cos(df_geom['phi'].values) + \
        np.sin(df_geom['theta'].values) * np.sin(df_geom['phi'].values)
    DA = np.acos(C)   
    
    dict_res['label'].append(title)
    dict_res['AngleDiffs'].append(DA)



pm.setGraphicOptions(mode='screen')
fig, axes = plt.subplots(2, len(tifNames), figsize=(len(tifNames)*3, 6),
                         layout='compressed')
colors = pm.cL_Set21

for i in range(len(tifNames)):
    print(i)
    tP, fN = tifPaths[i], fileNames[i]
    _, tN = os.path.split(tP)
    shape, dtype = ufun.tiff_inspect(tP)
    
    X_MTcenter = df_g.loc[df_g['tN']==tN, 'xc'].to_numpy()[0]
    Y_MTcenter = df_g.loc[df_g['tN']==tN, 'yc'].to_numpy()[0]
    # print(X_MTcenter, Y_MTcenter)

    title = '_'.join(fN.split('_')[1:4])
    rawTrackName = fN + Dict_fSuf['suf_rawTracks'] + '.xml'
    if Use_DriftCorrected:
        trackName = fN + Dict_fSuf['suf_driftCorrTracks'] + '.csv'
    else:
        trackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
        
    # msdName = fN + Dict_fSuf['suf_globalMsd'] + '.csv'
    # msdFitsName = fN + Dict_fSuf['suf_globalMsdFits']
    if np.isnan(X_MTcenter) or np.isnan(Y_MTcenter):
        contourName = fN + Dict_fSuf['suf_contour'] + '.npy'
        contourCell = np.load(os.path.join(srcDir, contourName))
        poly = Polygon(contourCell)
        xc, yc = poly.centroid.xy
        X_MTcenter, Y_MTcenter = int(xc[0]), int(yc[0])
    
    # df = pd.read_csv(os.path.join(dstDir, trackName), sep='\t')
    df_geom = GEOM_DATA[i]
    df_geom = df_geom[df_geom['AR'] > 1.5]
    grouped = df_geom.groupby('particle')
    
    ax = axes[0, i]
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    
    for p, df_gp in grouped:      
        xc, yc = df_gp['xc'].to_numpy()[0], df_gp['yc'].to_numpy()[0]
        a, b = df_gp['a'].to_numpy()[0], df_gp['b'].to_numpy()[0]
        phi = df_gp['phi'].to_numpy()[0]
        c = colors[p%len(colors)]       
        if p%10 == 0:
            aa = np.linspace(0, 2*np.pi, 360)
            xp, yp = ufun.get_ellipse_xy(xc, yc, a, b, phi, aa=aa)
            ax.plot(xp, yp, c=c, ls='-', lw=0.5)
    plt.show()
    
    C = np.cos(df_geom['theta'].values) * np.cos(df_geom['phi'].values) + \
        np.sin(df_geom['theta'].values) * np.sin(df_geom['phi'].values)
    DA = np.acos(C)
    
    ax = axes[1, i]
    ax.hist(DA, bins=20)
    ax.set_ylabel('N trajectories')
    ax.set_xlabel(r'$|\theta - \phi|$ (rad)')
    ax.set_xticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
    ax.set_xticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
    ax.grid()

plt.show()
    
    

# -----------------------------------------------------------------------------
# %%% 26-09-30_D1

# %%%% Settings

#### Paths    

dateDir = os.path.join(up.Path_IntraCellTracking, '26-09-30_FastAcq-Channel_Fec_NB-Yolk')
srcDir = os.path.join(dateDir, 'D1')

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


# %%%% Run table autofill

autofill_SPT_Data(srcDir, tifNames, df_films_PATH)

# %%%% Run new Tracking

#### Settings
Dict_TrackMate_Settings = {
    'RADIUS_UM' : 0.8, 'THRESH_SPOT_QLT' : 0.25,
    'THRESH_LINK_UM' : 0.25, 'THRESH_MIN_DURATION' : 30,
    }

NByolk_track_sequence(
    srcDir, tifNames, df_films_PATH, 
    Dict_fSuf, Dict_rP, Dict_pP, 
    Dict_TrackMate_Settings, 
    Do_Contours = True, Do_Tracking = True, 
    Do_TrackCleaning = True, 
    )

# %%%% Run new MSD

df_res_name = 'SPT_Results_01'
NByolk_msd_compute(tifNames, df_films_PATH, df_res_name,
                    Dict_fSuf, Dict_rP, Dict_pP, 
                    Do_MSD = True, Do_PairMSD = True)

# %%%% Run iMSD and R-OR MSD

NByolk_imsd_compute(tifNames, df_films_PATH,
                       Dict_fSuf, Dict_rP, Dict_pP)

df_res_name = 'SPT_Results_01'
NByolk_ro_msd_compute(tifNames, df_films_PATH, df_res_name,
                          Dict_fSuf, Dict_rP, Dict_pP)

# %%%% Import MSD & plot / Import pairMSD & plot

title = '26-09-30_D1'
df_diff = import_and_plot_MSD(fileNames, Dict_fSuf, fig_title=title)
df_diff, df_pairDiff = import_and_plot_pairMSD(fileNames, Dict_fSuf, fig_title=title)

for i in plt.get_fignums():
    plt.figure(i)
    plt.savefig(os.path.join(FIG_DIR, title + f'_MSDfig_{i:02d}.png'))

# %%%% Import tracks & analyse shape of explored zone

Use_DriftCorrected = False

mainDir = os.path.join(up.Path_IntraCellTracking, '26-09-30_FastAcq-Channel_Fec_NB-Yolk')
srcDir = os.path.join(mainDir, 'D1')
dstDir = os.path.join(mainDir, 'SPT_results')

tifNames = [
    # '26-09-30_D1_PreF_C1_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D1_PreF_C2_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D1_PreF_C3_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PostF_4min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D1_PostF_6min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PostF_10min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D1_PostF_13min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PostF_18min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D1_PostF_25min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PostF_30min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D1_PostF_35min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D1_PostF_40min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D1_PostF_45min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D1_PostF_52min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D1_PostF_60min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D1_PostF_65min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D1_PostF_70min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    ]

tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
fileNames = [fN.split('.')[0] for fN in tifNames]

dict_res = {'label':[],
            'AngleDiffs':[]}

GEOM_DATA = []

df_g = pd.read_csv(os.path.join(srcDir, 'Cell_geometry.csv'), sep=',')


for i in range(len(tifNames)):
    print(i)
    tP, fN = tifPaths[i], fileNames[i]
    _, tN = os.path.split(tP)
    shape, dtype = ufun.tiff_inspect(tP)
    
    X_MTcenter = df_g.loc[df_g['tN']==tN, 'xc'].to_numpy()[0]
    Y_MTcenter = df_g.loc[df_g['tN']==tN, 'yc'].to_numpy()[0]

    title = '_'.join(fN.split('_')[1:3])
    rawTrackName = fN + Dict_fSuf['suf_rawTracks'] + '.xml'
    if Use_DriftCorrected:
        trackName = fN + Dict_fSuf['suf_driftCorrTracks'] + '.csv'
    else:
        trackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
        
    # msdName = fN + Dict_fSuf['suf_globalMsd'] + '.csv'
    # msdFitsName = fN + Dict_fSuf['suf_globalMsdFits']
    if np.isnan(X_MTcenter) or np.isnan(Y_MTcenter):
        contourName = fN + Dict_fSuf['suf_contour'] + '.npy'
        contourCell = np.load(os.path.join(srcDir, contourName))
        poly = Polygon(contourCell)
        xc, yc = poly.centroid.xy
        X_MTcenter, Y_MTcenter = int(xc[0]), int(yc[0])
    
    df = pd.read_csv(os.path.join(dstDir, trackName), sep='\t')
    
    # Lp = df['particle'].unique().astype(int)
    dict_geom = {'particle':[],
                 'np':[],
                 'xc':[],
                 'yc':[],
                 'theta':[],
                 'a':[],
                 'b':[],
                 'L':[],
                 'l':[],
                 'AR':[],
                 'phi':[],}
    
    grouped = df.groupby('particle')
    
    for p, df_p in grouped:
        points = np.array([[x, y] for (x, y) in zip(df_p.x, df_p.y)])
        
        MPoints = MultiPoint(points)
        Hull = MPoints.convex_hull
        xy_ch = np.array([xy for xy in Hull.exterior.coords[1:]])
        xc, yc = Hull.centroid.coords[0]
        
        out = ufun.fit_ellipse_fixedCenter(xy_ch[:, 0], xy_ch[:, 1], 
                                           xc, yc, mode='cartesian')
        _, _, a, b, phi = out
        theta = np.atan2(yc - Y_MTcenter, xc - X_MTcenter)
        theta_deg = theta * 180/np.pi
        dotprod = np.cos(theta)*np.cos(phi) + np.sin(theta)*np.sin(phi)
        if dotprod < 0:
            phi = phi - np.sign(phi) * np.pi
        phi_deg = phi * 180/np.pi
        L, l = max(a, b), min(a, b)
        AR = L/l
        
        dict_geom['particle'].append(p)
        dict_geom['np'].append(len(df_p))
        dict_geom['xc'].append(xc)
        dict_geom['yc'].append(yc)
        dict_geom['theta'].append(theta)
        dict_geom['a'].append(a)
        dict_geom['b'].append(b)
        dict_geom['L'].append(L)
        dict_geom['l'].append(l)
        dict_geom['AR'].append(AR)
        dict_geom['phi'].append(phi)
    
    df_geom_raw = pd.DataFrame(dict_geom)
    GEOM_DATA.append(df_geom_raw)
    
    df_geom = df_geom_raw[df_geom_raw['AR'] > 1.5]    
    df_geom['theta_bin'] = (df_geom['theta'].values * 18/np.pi).astype(int) * 10 + 5
    C = np.cos(df_geom['theta'].values) * np.cos(df_geom['phi'].values) + \
        np.sin(df_geom['theta'].values) * np.sin(df_geom['phi'].values)
    DA = np.acos(C)   
    
    dict_res['label'].append(title)
    dict_res['AngleDiffs'].append(DA)



pm.setGraphicOptions(mode='screen')
fig, axes = plt.subplots(2, len(tifNames), figsize=(len(tifNames)*3, 6),
                         layout='compressed')
colors = pm.cL_Set21

for i in range(len(tifNames)):
    print(i)
    tP, fN = tifPaths[i], fileNames[i]
    _, tN = os.path.split(tP)
    shape, dtype = ufun.tiff_inspect(tP)
    
    X_MTcenter = df_g.loc[df_g['tN']==tN, 'xc'].to_numpy()[0]
    Y_MTcenter = df_g.loc[df_g['tN']==tN, 'yc'].to_numpy()[0]
    # print(X_MTcenter, Y_MTcenter)

    title = '_'.join(fN.split('_')[1:4])
    rawTrackName = fN + Dict_fSuf['suf_rawTracks'] + '.xml'
    if Use_DriftCorrected:
        trackName = fN + Dict_fSuf['suf_driftCorrTracks'] + '.csv'
    else:
        trackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
        
    # msdName = fN + Dict_fSuf['suf_globalMsd'] + '.csv'
    # msdFitsName = fN + Dict_fSuf['suf_globalMsdFits']
    if np.isnan(X_MTcenter) or np.isnan(Y_MTcenter):
        contourName = fN + Dict_fSuf['suf_contour'] + '.npy'
        contourCell = np.load(os.path.join(srcDir, contourName))
        poly = Polygon(contourCell)
        xc, yc = poly.centroid.xy
        X_MTcenter, Y_MTcenter = int(xc[0]), int(yc[0])
    
    # df = pd.read_csv(os.path.join(dstDir, trackName), sep='\t')
    df_geom = GEOM_DATA[i]
    df_geom = df_geom[df_geom['AR'] > 1.5]
    grouped = df_geom.groupby('particle')
    
    ax = axes[0, i]
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    
    for p, df_gp in grouped:      
        xc, yc = df_gp['xc'].to_numpy()[0], df_gp['yc'].to_numpy()[0]
        a, b = df_gp['a'].to_numpy()[0], df_gp['b'].to_numpy()[0]
        phi = df_gp['phi'].to_numpy()[0]
        c = colors[p%len(colors)]       
        if p%10 == 0:
            aa = np.linspace(0, 2*np.pi, 360)
            xp, yp = ufun.get_ellipse_xy(xc, yc, a, b, phi, aa=aa)
            ax.plot(xp, yp, c=c, ls='-', lw=0.5)
    plt.show()
    
    C = np.cos(df_geom['theta'].values) * np.cos(df_geom['phi'].values) + \
        np.sin(df_geom['theta'].values) * np.sin(df_geom['phi'].values)
    DA = np.acos(C)
    
    ax = axes[1, i]
    ax.hist(DA, bins=20)
    ax.set_ylabel('N trajectories')
    ax.set_xlabel(r'$|\theta - \phi|$ (rad)')
    ax.set_xticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
    ax.set_xticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
    ax.grid()

plt.show()

# %%%% DEV FUNCTION ! Import tracks & analyse shape of explored zone

Use_DriftCorrected = False

mainDir = os.path.join(up.Path_IntraCellTracking, '26-09-30_FastAcq-Channel_Fec_NB-Yolk')
srcDir = os.path.join(mainDir, 'D1')
dstDir = os.path.join(mainDir, 'SPT_results')

tifNames = [
    '26-09-30_D1_PostF_18min_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    ]

tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
fileNames = [fN.split('.')[0] for fN in tifNames]

dict_res = {'label':[],
            'AngleDiffs':[]}

GEOM_DATA = []

df_cellGeom = pd.read_csv(os.path.join(srcDir, 'Cell_geometry.csv'), sep=',')

i = 0
print(i)

tifPath, fileName = tifPaths[i], fileNames[i]

def getEllipses(fileName, tifPath, df_cellGeom, 
                Use_DriftCorrected, Dict_Files_Suffix):
    fN = fileName
    tP = tifPath
    _, tN = os.path.split(tP)
    # shape, dtype = ufun.tiff_inspect(tP)
    
    df_g = df_cellGeom
    X_MTcenter = df_g.loc[df_g['tN']==tN, 'xc'].to_numpy()[0]
    Y_MTcenter = df_g.loc[df_g['tN']==tN, 'yc'].to_numpy()[0]
    
    title = '_'.join(fN.split('_')[1:3])
    if Use_DriftCorrected:
        trackName = fN + Dict_Files_Suffix['suffix_driftCorrTracks'] + '.csv'
    else:
        trackName = fN + Dict_Files_Suffix['suffix_cleanTracks'] + '.csv'
        
    if np.isnan(X_MTcenter) or np.isnan(Y_MTcenter):
        contourName = fN + Dict_Files_Suffix['suffix_contour'] + '.npy'
        contourCell = np.load(os.path.join(srcDir, contourName))
        poly = Polygon(contourCell)
        xc, yc = poly.centroid.xy
        X_MTcenter, Y_MTcenter = int(xc[0]), int(yc[0])
    
    df = pd.read_csv(os.path.join(dstDir, trackName), sep='\t')
    
    dict_geom = {'particle':[], 'np':[],
                 'xc':[], 'yc':[],
                 'theta':[],
                 'a':[], 'b':[],
                 'L':[], 'l':[], 'AR':[],
                 'phi':[],}
    
    grouped = df.groupby('particle')
    
    for p, df_p in grouped:
        points = np.array([[x, y] for (x, y) in zip(df_p.x, df_p.y)])
        
        MPoints = MultiPoint(points)
        Hull = MPoints.convex_hull
        xy_ch = np.array([xy for xy in Hull.exterior.coords[1:]])
        xc, yc = Hull.centroid.coords[0]
        
        out = ufun.fit_ellipse_fixedCenter(xy_ch[:, 0], xy_ch[:, 1], 
                                           xc, yc, mode='cartesian')
        _, _, a, b, phi = out
        theta = np.atan2(yc - Y_MTcenter, xc - X_MTcenter)
        # theta_deg = theta * 180/np.pi
        dotprod = np.cos(theta)*np.cos(phi) + np.sin(theta)*np.sin(phi)
        if dotprod < 0:
            phi = phi - np.sign(phi) * np.pi
        # phi_deg = phi * 180/np.pi
        L, l = max(a, b), min(a, b)
        AR = L/l
        
        dict_geom['particle'].append(p)
        dict_geom['np'].append(len(df_p))
        dict_geom['xc'].append(xc)
        dict_geom['yc'].append(yc)
        dict_geom['theta'].append(theta)
        dict_geom['a'].append(a)
        dict_geom['b'].append(b)
        dict_geom['L'].append(L)
        dict_geom['l'].append(l)
        dict_geom['AR'].append(AR)
        dict_geom['phi'].append(phi)
    
    df_geom = pd.DataFrame(dict_geom)
    df_geom_f = df_geom[df_geom['AR'] > 1.5]    
    df_geom_f['theta_bin'] = (df_geom_f['theta'].values * 18/np.pi).astype(int) * 10 + 5
    
    C = np.cos(df_geom_f['theta'].values) * np.cos(df_geom_f['phi'].values) + \
        np.sin(df_geom_f['theta'].values) * np.sin(df_geom_f['phi'].values)
    DA = np.acos(C)   
    
    
    pm.setGraphicOptions(mode='screen')
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 7),
                             layout='compressed')
    colors = pm.cL_Set21
    ax = axes[0]
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    grouped = df_geom_f.groupby('particle')
    ax.plot([X_MTcenter], [Y_MTcenter], 'ko')
    for p, df_gp in grouped:
        xc, yc = df_gp['xc'].to_numpy()[0], df_gp['yc'].to_numpy()[0]
        a, b = df_gp['a'].to_numpy()[0], df_gp['b'].to_numpy()[0]
        phi = df_gp['phi'].to_numpy()[0]
        c = colors[p%len(colors)]       
        if p%10 == 0:
            aa = np.linspace(0, 2*np.pi, 360)
            xp, yp = ufun.get_ellipse_xy(xc, yc, a, b, phi, aa=aa)
            ax.plot(xp, yp, c=c, ls='-', lw=0.5)
            
    ax = axes[1]
    ax.hist(DA, bins=20)
    ax.set_ylabel('N trajectories')
    ax.set_xlabel(r'$|\theta - \phi|$ (rad)')
    ax.set_xticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
    ax.set_xticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
    ax.grid()
    plt.show()
    
    return(df_geom, DA, fig, axes)


df_geom, DA, fig, axes = getEllipses(fileName, tifPath, df_cellGeom, 
                                     Use_DriftCorrected, Dict_fSuf)


# -----------------------------------------------------------------------------
# %%% 26-09-30_D2

# %%%% Settings

#### Paths
mainDir = os.path.join(up.Path_IntraCellTracking, '26-09-30_FastAcq-Channel_Fec_NB-Yolk')
srcDir = os.path.join(mainDir, 'D2')
dstDir = os.path.join(mainDir, 'SPT_Results_2')
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


# %%%% Run table autofill

autofill_SPT_Data(srcDir, tifNames, df_films_PATH)

# %%%% Run new Tracking

#### Settings
Dict_TrackMate_Settings = {
    'RADIUS_UM' : 0.8, 'THRESH_SPOT_QLT' : 0.25,
    'THRESH_LINK_UM' : 0.25, 'THRESH_MIN_DURATION' : 30,
    }

NByolk_track_sequence(
    srcDir, tifNames, df_films_PATH, 
    Dict_fSuf, Dict_rP, Dict_pP, 
    Dict_TrackMate_Settings, 
    Do_Contours = True, Do_Tracking = True, 
    Do_TrackCleaning = True, 
    )

# %%%% Run new MSD

df_res_name = 'SPT_Results_01'
NByolk_msd_compute(tifNames, df_films_PATH, df_res_name,
                    Dict_fSuf, Dict_rP, Dict_pP, 
                    Do_MSD = True, Do_PairMSD = True)

# %%%% Run iMSD and R-OR MSD

NByolk_imsd_compute(tifNames, df_films_PATH,
                       Dict_fSuf, Dict_rP, Dict_pP)

df_res_name = 'SPT_Results_01'
NByolk_ro_msd_compute(tifNames, df_films_PATH, df_res_name,
                          Dict_fSuf, Dict_rP, Dict_pP)

# %%%% Import MSD & plot / Import pairMSD & plot

title = '26-09-30_D2'
df_diff = import_and_plot_MSD(fileNames, Dict_fSuf, fig_title=title)
df_diff, df_pairDiff = import_and_plot_pairMSD(fileNames, Dict_fSuf, fig_title=title)

# for i in plt.get_fignums():
#     plt.figure(i)
#     plt.savefig(os.path.join(figDir, title + f'_MSDfig_{i:02d}.png'))



# %%%% Import tracks & analyse shape of explored zone

Use_DriftCorrected = False

mainDir = os.path.join(up.Path_IntraCellTracking, '26-09-30_FastAcq-Channel_Fec_NB-Yolk')
srcDir = os.path.join(mainDir, 'D2')
dstDir = os.path.join(mainDir, 'SPT_results')

tifNames = [
    # '26-09-30_D2-R_PreF_C3_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D2-R_PreF_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D2-R_PostF_4min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_6min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D2-R_PostF_8min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D2-R_PostF_12min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_15min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D2-R_PostF_20min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_25min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D2-R_PostF_30min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_36min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D2-R_PostF_40min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D2-R_PostF_45min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PostF_50min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D2-R_PostF_53min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D2-R_PostF_55min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D2-R_PostF_60min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    # '26-09-30_D2-R_PostF_65min_C5_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    ]

tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
fileNames = [fN.split('.')[0] for fN in tifNames]

dict_res = {'label':[],
            'AngleDiffs':[]}

GEOM_DATA = []

df_g = pd.read_csv(os.path.join(srcDir, 'Cell_geometry.csv'), sep=',')


for i in range(len(tifNames)):
    print(i)
    tP, fN = tifPaths[i], fileNames[i]
    _, tN = os.path.split(tP)
    shape, dtype = ufun.tiff_inspect(tP)
    
    X_MTcenter = df_g.loc[df_g['tN']==tN, 'xc'].to_numpy()[0]
    Y_MTcenter = df_g.loc[df_g['tN']==tN, 'yc'].to_numpy()[0]

    title = '_'.join(fN.split('_')[1:3])
    rawTrackName = fN + Dict_fSuf['suf_rawTracks'] + '.xml'
    if Use_DriftCorrected:
        trackName = fN + Dict_fSuf['suf_driftCorrTracks'] + '.csv'
    else:
        trackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
        
    # msdName = fN + Dict_fSuf['suf_globalMsd'] + '.csv'
    # msdFitsName = fN + Dict_fSuf['suf_globalMsdFits']
    if np.isnan(X_MTcenter) or np.isnan(Y_MTcenter):
        contourName = fN + Dict_fSuf['suf_contour'] + '.npy'
        contourCell = np.load(os.path.join(srcDir, contourName))
        poly = Polygon(contourCell)
        xc, yc = poly.centroid.xy
        X_MTcenter, Y_MTcenter = int(xc[0]), int(yc[0])
    
    df = pd.read_csv(os.path.join(dstDir, trackName), sep='\t')
    
    # Lp = df['particle'].unique().astype(int)
    dict_geom = {'particle':[],
                 'np':[],
                 'xc':[],
                 'yc':[],
                 'theta':[],
                 'a':[],
                 'b':[],
                 'L':[],
                 'l':[],
                 'AR':[],
                 'phi':[],}
    
    grouped = df.groupby('particle')
    
    for p, df_p in grouped:
        points = np.array([[x, y] for (x, y) in zip(df_p.x, df_p.y)])
        
        MPoints = MultiPoint(points)
        Hull = MPoints.convex_hull
        xy_ch = np.array([xy for xy in Hull.exterior.coords[1:]])
        xc, yc = Hull.centroid.coords[0]
        
        out = ufun.fit_ellipse_fixedCenter(xy_ch[:, 0], xy_ch[:, 1], 
                                           xc, yc, mode='cartesian')
        _, _, a, b, phi = out
        theta = np.atan2(yc - Y_MTcenter, xc - X_MTcenter)
        theta_deg = theta * 180/np.pi
        dotprod = np.cos(theta)*np.cos(phi) + np.sin(theta)*np.sin(phi)
        if dotprod < 0:
            phi = phi - np.sign(phi) * np.pi
        phi_deg = phi * 180/np.pi
        L, l = max(a, b), min(a, b)
        AR = L/l
        
        dict_geom['particle'].append(p)
        dict_geom['np'].append(len(df_p))
        dict_geom['xc'].append(xc)
        dict_geom['yc'].append(yc)
        dict_geom['theta'].append(theta)
        dict_geom['a'].append(a)
        dict_geom['b'].append(b)
        dict_geom['L'].append(L)
        dict_geom['l'].append(l)
        dict_geom['AR'].append(AR)
        dict_geom['phi'].append(phi)
    
    df_geom_raw = pd.DataFrame(dict_geom)
    GEOM_DATA.append(df_geom_raw)
    
    df_geom = df_geom_raw[df_geom_raw['AR'] > 1.5]    
    df_geom['theta_bin'] = (df_geom['theta'].values * 18/np.pi).astype(int) * 10 + 5
    C = np.cos(df_geom['theta'].values) * np.cos(df_geom['phi'].values) + \
        np.sin(df_geom['theta'].values) * np.sin(df_geom['phi'].values)
    DA = np.acos(C)   
    
    dict_res['label'].append(title)
    dict_res['AngleDiffs'].append(DA)



pm.setGraphicOptions(mode='screen')
fig, axes = plt.subplots(2, len(tifNames), figsize=(len(tifNames)*3, 6),
                         layout='compressed')
colors = pm.cL_Set21

for i in range(len(tifNames)):
    print(i)
    tP, fN = tifPaths[i], fileNames[i]
    _, tN = os.path.split(tP)
    shape, dtype = ufun.tiff_inspect(tP)
    
    X_MTcenter = df_g.loc[df_g['tN']==tN, 'xc'].to_numpy()[0]
    Y_MTcenter = df_g.loc[df_g['tN']==tN, 'yc'].to_numpy()[0]
    # print(X_MTcenter, Y_MTcenter)

    title = '_'.join(fN.split('_')[1:4])
    rawTrackName = fN + Dict_fSuf['suf_rawTracks'] + '.xml'
    if Use_DriftCorrected:
        trackName = fN + Dict_fSuf['suf_driftCorrTracks'] + '.csv'
    else:
        trackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
        
    # msdName = fN + Dict_fSuf['suf_globalMsd'] + '.csv'
    # msdFitsName = fN + Dict_fSuf['suf_globalMsdFits']
    if np.isnan(X_MTcenter) or np.isnan(Y_MTcenter):
        contourName = fN + Dict_fSuf['suf_contour'] + '.npy'
        contourCell = np.load(os.path.join(srcDir, contourName))
        poly = Polygon(contourCell)
        xc, yc = poly.centroid.xy
        X_MTcenter, Y_MTcenter = int(xc[0]), int(yc[0])
    
    # df = pd.read_csv(os.path.join(dstDir, trackName), sep='\t')
    df_geom = GEOM_DATA[i]
    df_geom = df_geom[df_geom['AR'] > 1.5]
    grouped = df_geom.groupby('particle')
    
    ax = axes[0, i]
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    
    for p, df_gp in grouped:      
        xc, yc = df_gp['xc'].to_numpy()[0], df_gp['yc'].to_numpy()[0]
        a, b = df_gp['a'].to_numpy()[0], df_gp['b'].to_numpy()[0]
        phi = df_gp['phi'].to_numpy()[0]
        c = colors[p%len(colors)]       
        if p%10 == 0:
            aa = np.linspace(0, 2*np.pi, 360)
            xp, yp = ufun.get_ellipse_xy(xc, yc, a, b, phi, aa=aa)
            ax.plot(xp, yp, c=c, ls='-', lw=0.5)
    plt.show()
    
    C = np.cos(df_geom['theta'].values) * np.cos(df_geom['phi'].values) + \
        np.sin(df_geom['theta'].values) * np.sin(df_geom['phi'].values)
    DA = np.acos(C)
    
    ax = axes[1, i]
    ax.hist(DA, bins=20)
    ax.set_ylabel('N trajectories')
    ax.set_xlabel(r'$|\theta - \phi|$ (rad)')
    ax.set_xticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
    ax.set_xticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
    ax.grid()

plt.show()




# -----------------------------------------------------------------------------
# %%% 26-07-29

# %%%% Settings

#### Paths    

mainDir = os.path.join(up.Path_IntraCellTracking, '26-07-29_FastAcq_Fec_NB-Yolk')
srcDir = os.path.join(mainDir, 'Crops')
dstDir = os.path.join(mainDir, 'SPT_results')

tifNames = [
    '26-07-29_PreF_Pos2-1_20fps_Texp50ms_CSU642_crop.tif',
    '26-07-29_PreF_Pos2-2_20fps_Texp50ms_CSU642_crop.tif',
    '26-07-29_PostF_2min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
    '26-07-29_PostF_6min30_Pos11_10fps_Texp100ms_CSU642_crop.tif',
    '26-07-29_PostF_12min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
    '26-07-29_PostF_20min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
    '26-07-29_PostF_30min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
    '26-07-29_PostF_45min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
    '26-07-29_PostF_60min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
    '26-07-29_PostF_70min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
    '26-07-29_PostF_80min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
    ]

tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
fileNames = [fN.split('.')[0] for fN in tifNames]


# %%%% Run table autofill

autofill_SPT_Data(srcDir, tifNames, df_films_PATH)

# %%%% Run new Tracking

#### Settings
Dict_TrackMate_Settings = {
    'RADIUS_UM' : 0.8, 'THRESH_SPOT_QLT' : 0.25,
    'THRESH_LINK_UM' : 0.25, 'THRESH_MIN_DURATION' : 30,
    }

NByolk_track_sequence(
    srcDir, tifNames, df_films_PATH, 
    Dict_fSuf, Dict_rP, Dict_pP, 
    Dict_TrackMate_Settings, 
    Do_Contours = False, Do_Tracking = True, 
    Do_TrackCleaning = True, 
    )

# %%%% Run new MSD

df_res_name = 'SPT_Results_01'
NByolk_msd_compute(tifNames, df_films_PATH, df_res_name,
                    Dict_fSuf, Dict_rP, Dict_pP, 
                    Do_MSD = True, Do_PairMSD = True)


# -----------------------------------------------------------------------------
# %%% Charlotte's Fixed Cells

# %%%% Settings

#### Paths
mainDir = os.path.join(up.Path_IntraCellTracking, '2026_26-01_Fixed_NileBlue_CharlotteMallart')
srcDir = os.path.join(mainDir, 'CropedFilms')
dstDir = os.path.join(mainDir, 'SPT_results')
# print(os.listdir(srcDir))

tifNames = [
    '2026_2601_NileBlue_Fix_1-2.tif',
    '2026_2601_NileBlue_Fix_2-2.tif',
    '2026_2601_NileBlue_Fix_3-2.tif',
    '2026_2601_NileBlue_Fix_4-1.tif',
    '2026_2601_NileBlue_Fix_5-1.tif',
    '2026_2601_NileBlue_Fix_5-2.tif',
    '2026_2601_NileBlue_Fix_6-1.tif',
    '2026_2601_NileBlue_Fix_6-2.tif',
    '2026_2601_NileBlue_Fix_6-3.tif',
    ]

tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
fileNames = [fN.split('.')[0] for fN in tifNames]

# %%%% Run table autofill

autofill_SPT_Data(srcDir, tifNames, df_films_PATH)

# %%%% Run new Tracking

#### Settings
Dict_TrackMate_Settings = {
    'RADIUS_UM' : 0.8, 'THRESH_SPOT_QLT' : 0.25,
    'THRESH_LINK_UM' : 0.25, 'THRESH_MIN_DURATION' : 30,
    }

NByolk_track_sequence(
    srcDir, tifNames, df_films_PATH, 
    Dict_fSuf, Dict_rP, Dict_pP, 
    Dict_TrackMate_Settings, 
    Do_Contours = False, Do_Tracking = True, 
    Do_TrackCleaning = True, 
    )

# %%%% Run new MSD

df_res_name = 'SPT_Results_FixedCells'
NByolk_msd_compute(tifNames, df_films_PATH, df_res_name,
                    Dict_fSuf, Dict_rP, Dict_pP, 
                    Do_MSD = True, Do_PairMSD = True)

# %%%% Import MSD & plot / Import pairMSD & plot

df_diff = import_and_plot_MSD(fileNames, Dict_fSuf,)
df_diff, df_pairDiff = import_and_plot_pairMSD(fileNames, Dict_fSuf,)


# -----------------------------------------------------------------------------
# %% -----

# %% 2. Comparing across cells: Data & plots

# %%% Import tables

df_cells_PATH = os.path.join(RES_DIR, 'SPT_Data_Cells.csv')
df_films_PATH = os.path.join(RES_DIR, 'SPT_Data_Films.csv')
df_res_PATH = os.path.join(RES_DIR, 'SPT_Results_01.csv')

DF_CELLS = pd.read_csv(df_cells_PATH)
DF_FILMS = pd.read_csv(df_films_PATH)
DF_RES = pd.read_csv(df_res_PATH)
DF_M = pd.merge(DF_FILMS, DF_RES, on='tN', how='inner')

DF_M['Mid'] = DF_M['date'] + '_' + DF_M['manip']
DF_M['Cid'] = DF_M['date'] + '_' + DF_M['manip'] + '_' + DF_M['cell']
DF_CELLS['Mid'] = DF_CELLS['date'] + '_' + DF_CELLS['manip']
DF_CELLS['Cid'] = DF_CELLS['date'] + '_' + DF_CELLS['manip'] + '_' + DF_CELLS['cell']

DF_M = pd.merge(DF_M, DF_CELLS, on='Cid', how='left', suffixes=['', '_cell'])

# %%% Plot together


FILTERS = [
    (DF_M['treatment_cell'] == 'none'),
    ]
df_f = pm.filterDf(DF_M, FILTERS).reset_index(drop=True)


I_PHASES = {'aster-centration':0, 
          'interphase':1, 
          'mitosis':2}

for i in range(len(df_f)):
    row = df_f.loc[i, :]
    phase = row['phase']
    if phase == 'non-fertilized':
        tpf_norm = 0
    else:
        tpf = row['tpf']
        i_phase = I_PHASES[phase]
        tpf_i, tpf_f = row['T_'+phase+'_i'], row['T_'+phase+'_f']
        tpf_norm = i_phase + ((tpf - tpf_i)/(tpf_f - tpf_i))
    df_f.loc[i, 'tpf_norm'] = tpf_norm


fig, ax = plt.subplots(1, 1)

metric_T = 'tpf_norm'
metric_D = 'pmsd_D_highDt'
normalize = False

Mid_groups = df_f.groupby('Mid')
for mid, df_c in Mid_groups:
    if 'postF' in df_c['status'].to_numpy():
        tpf = df_c[metric_T].to_numpy()
        
        if normalize:
            FILTERS_ctrl = [
                (df_c['tpf'] == 0),
                ]
            df_cf = pm.filterDf(df_c, FILTERS_ctrl)
            D = (df_c[metric_D] / np.median(df_cf[metric_D])).to_numpy()
        
        else:
            D = df_c[metric_D].to_numpy()
        
        
        ax.plot(tpf, D, ls='-', marker='o', label=mid)

ax.legend()
plt.show()
















# %%% -






# %% -----
# %% Tests

# %%%% Claude written 2D smoother

"""
Gaussian-weighted spatial smoothing map.

For every point on a regular grid, computes the Gaussian-weighted average
of the k nearest input points (by Euclidean distance).

Inputs
------
data  : Nx3 array  [[x, y, v], ...]
sigma : float      standard deviation of the Gaussian kernel (same units as x/y)
k     : int        number of nearest neighbours to consider

Usage
-----
Run the script directly for a built-in demo, or import `smooth_map` and call
it with your own data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def smooth_map(
    data: np.ndarray,
    sigma: float,
    k: int,
    grid_resolution: int = 300,
    x_range: tuple | None = None,
    y_range: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a Gaussian-weighted average of the k nearest values on a 2-D grid.

    Parameters
    ----------
    data            : (N, 3) array — columns are x, y, v.
    sigma           : Gaussian kernel standard deviation.
    k               : Number of nearest neighbours to use per grid point.
    grid_resolution : Number of grid cells along each axis.
    x_range         : (x_min, x_max) — defaults to data extent + 5 % padding.
    y_range         : (y_min, y_max) — defaults to data extent + 5 % padding.

    Returns
    -------
    gx, gy : 2-D arrays of grid coordinates (output of np.meshgrid).
    gz     : 2-D array of smoothed values.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("data must be an (N, 3) array.")

    xy = data[:, :2]
    v  = data[:, 2]
    k  = min(k, len(data))          # can't ask for more neighbours than points

    # --- build grid ---
    def _padded(arr, pad=0.05):
        lo, hi = arr.min(), arr.max()
        margin = (hi - lo) * pad or 1.0   # handle degenerate case
        return lo - margin, hi + margin

    x_min, x_max = x_range or _padded(xy[:, 0])
    y_min, y_max = y_range or _padded(xy[:, 1])

    xs = np.linspace(x_min, x_max, grid_resolution)
    ys = np.linspace(y_min, y_max, grid_resolution)
    gx, gy = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])

    # --- k-nearest-neighbour lookup ---
    tree = cKDTree(xy)
    dists, idxs = tree.query(grid_pts, k=k)   # shape (M, k)

    # --- Gaussian weights ---
    if k == 1:                                 # query returns 1-D arrays
        dists = dists[:, np.newaxis]
        idxs  = idxs[:, np.newaxis]

    weights = np.exp(-0.5 * (dists / sigma) ** 2)   # shape (M, k)
    w_sum   = weights.sum(axis=1, keepdims=True)
    w_sum   = np.where(w_sum == 0, 1.0, w_sum)       # avoid division by zero

    values  = v[idxs]                                # shape (M, k)
    gz_flat = (weights * values).sum(axis=1) / w_sum.squeeze()
    gz      = gz_flat.reshape(gx.shape)

    return gx, gy, gz


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def plot_smooth_map(
    data: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    sigma: float,
    k: int,
    cmap: str = "viridis",
    show_points: bool = True,
) -> plt.Figure:
    """Render the smoothed map with the scatter overlay."""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.pcolormesh(gx, gy, gz, cmap=cmap, shading="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Smoothed value", fontsize=11)

    if show_points:
        sc = ax.scatter(
            data[:, 0], data[:, 1],
            c=data[:, 2], cmap=cmap,
            s=40, edgecolors="white", linewidths=0.5,
            vmin=gz.min(), vmax=gz.max(),
            zorder=3,
        )

    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)
    ax.set_title(f"Gaussian-weighted smooth map  (σ={sigma}, k={k})", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(40)

    # Three clusters with different mean values
    n = 60
    centers = [(1, 3, 10), (3, 3, 50), (2, 4, 90)]
    parts = []
    for cx, cy, cv in centers:
        xy_part = rng.normal(loc=[cx, cy], scale=1.0, size=(n, 2))
        v_part  = rng.normal(loc=cv, scale=20, size=(n, 1))
        parts.append(np.hstack([xy_part, v_part]))

    data = np.vstack(parts)

    # Parameters
    sigma = 1
    k     = 20

    gx, gy, gz = smooth_map(data, sigma=sigma, k=k, grid_resolution=400)
    fig = plot_smooth_map(data, gx, gy, gz, sigma=sigma, k=k, cmap="plasma")
    plt.show()

# %%%% Test Compute pair-MSD

pm.setGraphicOptions(mode='screen')

for i in range(len(fileNames)):
    print(i)
    tP, fN = tifPaths[i], fileNames[i]
    cleanTrackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
    df = pd.read_csv(os.path.join(dstDir, cleanTrackName), sep='\t')
    
    PixPerUm = Dict_Image_Settings['PixPerUm']
    FPS = Dict_Image_Settings['FPS']
    max_Dt_s = 4
    
    shape, dtype = ufun.tiff_inspect(tP)
    Nframes = shape[0]   
    
    #### Compute Pair MSD
    df_allTRange_pair_MSD = tbca.compute_pairwise_MSD(df, PixPerUm, FPS, Nframes,
                                                 len_TRanges = 100, delta_TRanges = -1, 
                                                 dist_th_um = 4, max_Dt_s = 4)
    print(i)
    
    pairMsdName = fN + Dict_fSuf['suf_globalPairMsd'] + '.csv'
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
    
    pairMsdFitsName = fN + Dict_fSuf['suf_globalPairMsdFits']
    ufun.dict2json(dict_pairMSDfits, dstDir, pairMsdFitsName)



# %%%% Test movie of triangulation

i = 0
print(i)
tP, fN = tifPaths[i], fileNames[i]
dirPlots = os.path.join(up.Path_Desktop, "Plots_film_triangles")
cleanTrackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
df = pd.read_csv(os.path.join(dstDir, cleanTrackName), sep='\t')
shape, _ = ufun.tiff_inspect(tP)
PixPerUm = cd.PixPerUm_60X_W1
FPS = 20
plot_frames = np.arange(1001, 1101, 2)
dist_th_um = 5
# tbca.make_film_of_pairs_V1(df, PixPerUm, FPS, shape, dirPlots,
#                           plot_frames, dist_th_um = 4)
tbca.make_film_of_pairs_V2(df, PixPerUm, FPS, shape, dirPlots, 
                          plot_frames, dist_th_um = 4)

# %%%% Test drift correction

i = 10
print(i)
tP, fN = tifPaths[i], fileNames[i]

cleanTrackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
df = pd.read_csv(os.path.join(dstDir, cleanTrackName), sep='\t')

PixPerUm = cd.PixPerUm_60X_W1
FPS = 20

df2 = correct_drift_in_Tracks(df, fN, tP, dstDir, PixPerUm, FPS, 
                              PLOT = True, SHOWPLOT = True, SAVEPLOT = False, 
                              export_image = False)

# %%%% Test drift correction 2

#### Paths
mainDir = os.path.join(up.Path_IntraCellTracking, 'TestsFilms_W1_60x_5fps_NonFertilized')
srcDir = os.path.join(mainDir, 'Films')
dstDir = os.path.join(mainDir, 'SPT_results')
tifNames = [
    '26-09-30_D2-R_PreF_C3_5fps_Texp50ms_L20p_CSU642.tif',
    '26-09-30_D2-R_PreF_C4_5fps_Texp50ms_L20p_CSU642.tif',
    'AVG_20261005_dish3_39C_image_1_CSU642.ome.tf2 - C=0-1.tif',
    ]

tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
fileNames = [fN.split('.')[0] for fN in tifNames]

Dict_fSuf = {
    'suf_contour' : '_cellContour',
    'suf_mask' : '_cellMask',
    'suf_rawTracks' : '_TmTracks',
    'suf_cleanTracks' : '_PyTracks',
    'suf_cleanTracks_DriftCorr' : '_PyTracks2',
    'suf_drift' : '_XYdrift',
    'suf_globalMsd' : '_GlobalMsd',
    'suf_globalMsdFits' : '_GlobalMsdFits',
    'suf_globalPairMsd' : '_GlobalPairMsd',
    'suf_globalPairMsdFits' : '_GlobalPairMsdFits',
    }


#### Settings
UmPerPix = cd.UmPerPix_60X_W1
Dict_Image_Settings = {
    'UmPerPix' : UmPerPix,
    'PixPerUm' : 1/UmPerPix,
    'FPS' : 20,
    }

Dict_TrackMate_Settings = {
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


i = 2
print(i)
tP, fN = tifPaths[i], fileNames[i]

cleanTrackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
df = pd.read_csv(os.path.join(dstDir, cleanTrackName), sep='\t')

PixPerUm = cd.PixPerUm_60X_W1
FPS = 20

df2, Dxy_corr = correct_drift_in_Tracks(df, fN, tP, dstDir, PixPerUm, FPS, 
                                        PLOT = True, SHOWPLOT = True, SAVEPLOT = False, 
                                        export_image = False)



# %%% Test drift correction

i = 3
print(i)
tP, fN = tifPaths[i], fileNames[i]

cleanTrackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
df = pd.read_csv(os.path.join(dstDir, cleanTrackName), sep='\t')

PixPerUm = cd.PixPerUm_60X_W1
FPS = 20

df2, Dxy_corr = correct_drift_in_Tracks(df, fN, tP, dstDir, PixPerUm, FPS, 
                              PLOT = True, SHOWPLOT = True, SAVEPLOT = False, 
                              export_image = False)

# %%% Test smoothed 2D-maps

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.stats.qmc import Halton

rng = np.random.default_rng()
xyobs = 2*Halton(2, seed=rng).random(400)
pobs = np.sum(xyobs, axis=1)*np.exp(-6*np.sum(xyobs**4, axis=1))

x1 = np.linspace(0, 2, 50)
xygrid = np.asarray(np.meshgrid(x1, x1, indexing='ij'))
xyflat = xygrid.reshape(2, -1).T     # make it a 2-D array
# pflat = RBFInterpolator(xyobs, pobs, kernel='gaussian', epsilon=2)(xyflat)
# pgrid = pflat.reshape(50, 50)

# fig, axes = plt.subplots(1, 3, figsize=(9, 3))
# ax = axes[0]
# # ax.pcolormesh(*xgrid, ygrid, vmin=-0.25, vmax=0.25, shading='gouraud')
# p = ax.scatter(*xyobs.T, c=pobs, s=20, ec='k', vmin=-0.25, vmax=0.25)

# ax = axes[1]
# ax.pcolormesh(*xygrid, pgrid, vmin=-0.25, vmax=0.25, shading='gouraud')
# # p = ax.scatter(*xobs.T, c=yobs, s=50, ec='k', vmin=-0.25, vmax=0.25)

# ax = axes[2]
# ax.pcolormesh(*xygrid, pgrid, vmin=-0.25, vmax=0.25, shading='gouraud')
# p = ax.scatter(*xyobs.T, c=pobs, s=20, ec='k', vmin=-0.25, vmax=0.25)

# fig.colorbar(p)
# plt.show()

df = pd.DataFrame({'x':xyobs[:,0],
                   'y':xyobs[:,1],
                   'p':pobs})

def gaussian_weights(yx0, Y, X, P, sigma=1):
    dx = X-yx0[1]
    dy = Y-yx0[0]
    D2 = (dx**2 + dy**2)**0.5
    w = np.exp(-D2/(2*sigma**2))
    wP = P*w
    wAvgP = np.sum(wP)/np.sum(w)
    return(wAvgP)

def make_smoothed_grid(X, Y, P, shape, Mx, My, sigma):
    NpixY, NpixX = shape
    xx = np.linspace(0, NpixX, Mx)
    yy = np.linspace(0, NpixY, My)
    yxgrid = np.asarray(np.meshgrid(yy, xx, indexing='ij'))
    yxflat = yxgrid.reshape(2, -1).T     # make it a 2-D array
    pflat = np.array([gaussian_weights(yx0, Y, X, P, sigma=sigma) for yx0 in yxflat])
    pgrid = pflat.reshape(My, Mx)
    return(pgrid, yxgrid)
    

X = df['x'].to_numpy()
Y = df['y'].to_numpy()
P = df['p'].to_numpy()
shape = [2, 2]
Mx, My = 20, 20
sigma = 0.1

pgrid2, yxgrid2 = make_smoothed_grid(X, Y, P, shape, Mx, My, sigma)


fig, axes = plt.subplots(1, 3, figsize=(9, 3))
ax = axes[0]
# ax.pcolormesh(*xgrid, ygrid, vmin=-0.25, vmax=0.25, shading='gouraud')
p = ax.scatter(*xyobs.T, c=pobs, s=20, ec='k', vmin=-0.25, vmax=0.25)

ax = axes[1]
ax.pcolormesh(*yxgrid2, pgrid2, vmin=-0.25, vmax=0.25, shading='gouraud')
# ax.imshow(pgrid2)
# p = ax.scatter(*xobs.T, c=yobs, s=50, ec='k', vmin=-0.25, vmax=0.25)

ax = axes[2]
ax.pcolormesh(*yxgrid2, pgrid2, vmin=-0.25, vmax=0.25, shading='gouraud')
# ax.imshow(pgrid2)
p = ax.scatter(*xyobs.T, c=pobs, s=20, ec='k', vmin=-0.25, vmax=0.25)

fig.colorbar(p)
plt.show()

# %%


# %% TBD
    
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
    cleanTrackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
    msdName = fN + Dict_fSuf['suf_globalMsd'] + '.csv'
    msdFitsName = fN + Dict_fSuf['suf_globalMsdFits']
    
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


# %% -----------------------

# %%% Dev

# %%%% MSRD computation test

pm.setGraphicOptions(mode='screen')

# for i in range(len(fileNames)):
i = 13
tP, fN = tifPaths[i], fileNames[i]
cleanTrackName = fN + Dict_fSuf['suf_cleanTracks'] + '.csv'
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
