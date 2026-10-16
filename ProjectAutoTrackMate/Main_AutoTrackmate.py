# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 16:50:16 2021
@authors: Joseph Vermeil

Main_AutoTrackmate.py - DESCRIPTION

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

# %% Imports & General Settings

#### Imports

import os
import sys
import time

import numpy as np
import pandas as pd
import trackpy as tp
# import seaborn as sns
# import matplotlib as mpl
import matplotlib.pyplot as plt

import Toolbox_AutoTrackmate as tbat

#### Settings

# Path_IntraCellTracking = "C:\\Users\\Utilisateur\\Desktop\\IntraCellTracking\\"
# Path_IntraCellTracking = "F:\\IntraCellTracking\\"
Path_IntraCellTracking = ".//Test_Dataset"


# %% Function for the all-in-one pipeline

def all_in_one_pipeline(srcDir, fileNames, tifPaths, dstDir,
                        Dict_TrackMate_Settings = None,
                        time_indices = None,
                        ROI_MODE = 'Square ROIs',
                        suffix_squareROI = '_listSquareROIs.json',
                        suffix_contour =  '_cellContour.npy',
                        suffix_mask = '_cellMask.npy',
                        suffix_tracks = '_Tracks.xml',
                        ):
    
    #### STEP 1 - ROIs
    print(tbat.BRIGHTORANGE + f'\n\n1. Defining Tracking ROIs\nROI mode = {ROI_MODE}\n' + tbat.NORMAL)
    for i in range(len(fileNames)):
        tifPath, fN = tifPaths[i], fileNames[i]
        fileROIs = fN + suffix_squareROI
        contourFile = fN + suffix_contour
        maskFile = fN + suffix_mask
        
        top = time.time()
        print(tbat.CYAN + f'File {i+1:.0f}/{len(fileNames):.0f} - {fN}...' + tbat.NORMAL, end='')
        sys.stdout.flush()    
        
        shape, dtype = tbat.tiff_inspect(tifPath)
        nT = shape[0]
        TT = range(0, nT, nT//100)
        img = tbat.load_stack_region(tifPath, time_indices=TT)
        
        if ROI_MODE == 'Square ROIs':
            Contour_cell, Mask_cell = tbat.make_NbYolkCell_contour_and_mask(img, PixPerUm,
                                                                            buffer_um = 0.0,
                                                                            mode = 'dark_background', 
                                                                            PLOT = False)
            
            Contour_pronuc, Mask_pronuc = tbat.make_NbYolkPronucleus_contour_and_mask(img, PixPerUm,
                                                                                      Contour_cell, Mask_cell,
                                                                                      mode = 'dark_background', 
                                                                                      buffer_pix = 3,
                                                                                      PLOT = False)
            
            list_rois = tbat.get_N_random_square_ROIs_in_cell(Mask_cell, L_ROIs, N_ROIs, 
                                                              excluded_mask = Mask_pronuc,
                                                              force_distribution = True,
                                                              seed=42, PLOT = False)
            
            tbat.list2json(list_rois, os.path.join(srcDir, fileROIs))
        
        
        elif ROI_MODE == 'Whole cell':
            Contour_cell, Mask_cell = tbat.make_NbYolkCell_contour_and_mask(img, PixPerUm,
                                                                            buffer_um = 0.0,
                                                                            mode = 'dark_background', 
                                                                            PLOT = False)
            
            np.save(os.path.join(srcDir, contourFile), Contour_cell)
            np.save(os.path.join(srcDir, maskFile), Mask_cell)
            
        print(tbat.GREEN + f' Done in {time.time()-top:.3f} s' + tbat.NORMAL, end='\n')
        sys.stdout.flush()
    
    #### STEP 2 - TrackMate
    print(tbat.BRIGHTORANGE + '\n\n2. Running Trackmate\n' + tbat.NORMAL)
    for i in range(len(fileNames)):
        tifPath, fN = tifPaths[i], fileNames[i]
        
        top = time.time()
        print(tbat.CYAN + f'\nFile {i+1:.0f}/{len(fileNames):.0f} - {fN}...' + tbat.NORMAL)
        
        if ROI_MODE == 'Square ROIs':
            fileROIs = fN + suffix_squareROI
            list_rois = tbat.json2list(os.path.join(srcDir, fileROIs))
            
            for k, ROI in enumerate(list_rois):
                rawTrackName = fN + f'_SqROI-0{k+1:.0f}' + suffix_tracks
                
                tbat.pretreat_and_track_NbYolk(tifPath, rawTrackName, dstDir, PixPerUm, 
                                          Dict_TrackMate_Settings = Dict_TrackMate_Settings,
                                          time_indices = time_indices, ROI_MODE = ROI_MODE, ROI = ROI,
                                          Mask_cell = None, mask_buffer_um = 0.0,
                                          correct_on_edges_buffer_um = 0.5,
                                          PLOT = True, SHOWPLOT = False, SAVEPLOT = True)
                
        elif ROI_MODE == 'Whole cell':
            maskFile = fN + suffix_mask
            Mask_cell = np.load(os.path.join(srcDir, maskFile))
            rawTrackName = fN + '_WholeCell' + suffix_tracks
            
            tbat.pretreat_and_track_NbYolk(tifPath, rawTrackName, dstDir, PixPerUm, 
                                           Dict_TrackMate_Settings = Dict_TrackMate_Settings,
                                           time_indices = time_indices, ROI_MODE = ROI_MODE, ROI = None,
                                           Mask_cell = Mask_cell, mask_buffer_um = 1.0,
                                           correct_on_edges_buffer_um = 1.0,
                                           PLOT = True, SHOWPLOT = False, SAVEPLOT = True)
    
        print(tbat.GREEN + f'\nDone in {time.time()-top:.3f} s' + tbat.NORMAL)






# %% Day of expt

# %%% Paths & Setting


# %%% Get ROIs

# %%% Run Trackmate

# %% ---






# %% EXAMPLE - 26-09-30
# These can be ran on the test dataset

Path_IntraCellTracking = ".//Test_Dataset"

# %%% Paths & Settings

mainDir = os.path.join(Path_IntraCellTracking, 'Films_W1_60x_5fps_NonFertilized')

srcDir = os.path.join(mainDir, 'Films')
dstDir = os.path.join(mainDir, 'SPT_results')
# Useful line : os.listdir(srcDir)

tifNames = [
    '26-09-30_D2-R_PreF_C3_5fps_Texp50ms_L20p_CSU642.tif',
    '26-09-30_D2-R_PreF_C4_5fps_Texp50ms_L20p_CSU642.tif',
    'AVG_20261005_dish3_39C_image_1_CSU642.ome.tf2 - C=0-1.tif',
    ]


tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
fileNames = [fN.split('.')[0] for fN in tifNames]

for tP in tifPaths:
    test = os.path.isfile(tP)
    if not test:
        print(tbat.RED + f'Error: {tP} does not exist' + tbat.NORMAL)

# Image properties
UmPerPix = tbat.UmPerPix_60X_W1
PixPerUm = 1/UmPerPix

ROI_MODE = 'Whole cell' # 'Square ROIs' | 'Whole cell'
# Explanation:
# In ROI_MODE = 'Whole cell', the step 1. will segment the whole cell 
# and save a mask & a contour of its shape. All particles within will be tracked
# in step 2.
# In ROI_MODE = 'Square ROIs', the step 1 will create N non-overlapping 
# square ROIs of size L µm, excluding the pronucleus. 
# Step 2 will track particles within these ROI, resulting in N .xml files.


# Parameters for square ROIs
L_ROIs_um = 15 # Size of the ROIs in µm
L_ROIs = int(round(L_ROIs_um/UmPerPix)) # Size of the ROIs in pixels
N_ROIs = 3 # Number of ROIs


# Parameters for TrackMate
Dict_TrackMate_Settings = {
    'RADIUS_UM' : 0.8, 
    'THRESH_SPOT_QLT' : 0.25,
    'THRESH_LINK_UM' : 0.25, 
    'THRESH_MIN_DURATION' : 30,
    }

# File suffix
suffix_squareROI = '_listSquareROIs.json'
suffix_contour =  '_cellContour.npy'
suffix_mask = '_cellMask.npy'
suffix_TmTracks = '_TmTracks.xml'
suffix_PyTracks = '_PyTracks.csv'

# %%% All-in-one pipeline

all_in_one_pipeline(srcDir, fileNames, tifPaths, dstDir,
                    Dict_TrackMate_Settings = Dict_TrackMate_Settings,
                    # ROI_MODE = ROI_MODE,
                    # suffix_squareROI = suffix_squareROI,
                    # suffix_contour =  suffix_contour,
                    # suffix_mask = suffix_mask,
                    # suffix_tracks = suffix_tracks,
                    )


# %%% OR : Step-by-step pipeline

# %%%% 1. Get ROIs


print(tbat.BRIGHTORANGE + f'\n\n1. Defining Tracking ROIs\nROI mode = {ROI_MODE}\n' + tbat.NORMAL)
for i in range(len(fileNames)):
    tifPath, fN = tifPaths[i], fileNames[i]
    fileROIs = fN + suffix_squareROI
    contourFile = fN + suffix_contour
    maskFile = fN + suffix_mask
    
    top = time.time()
    print(tbat.CYAN + f'File {i+1:.0f}/{len(fileNames):.0f} - {fN}...' + tbat.NORMAL, end='')
    sys.stdout.flush()    
    
    shape, dtype = tbat.tiff_inspect(tifPath)
    nT = shape[0]
    TT = range(0, nT, nT//100)
    img = tbat.load_stack_region(tifPath, time_indices=TT)
    
    if ROI_MODE == 'Square ROIs':
        Contour_cell, Mask_cell = tbat.make_NbYolkCell_contour_and_mask(img, PixPerUm,
                                                                        buffer_um = 0.0,
                                                                        mode = 'dark_background', 
                                                                        keep_only_largest = True,
                                                                        exclude_on_edges = True,
                                                                        PLOT = False)
        
        Contour_pronuc, Mask_pronuc = tbat.make_NbYolkPronucleus_contour_and_mask(img, PixPerUm,
                                                                                  Contour_cell, Mask_cell,
                                                                                  mode = 'dark_background', 
                                                                                  buffer_pix = 3,
                                                                                  PLOT = False)
        
        list_rois = tbat.get_N_random_square_ROIs_in_cell(Mask_cell, L_ROIs, N_ROIs, 
                                                          excluded_mask = Mask_pronuc,
                                                          force_distribution = True,
                                                          seed=42, PLOT = False)
        
        tbat.list2json(list_rois, os.path.join(srcDir, fileROIs))
    
    
    elif ROI_MODE == 'Whole cell':
        Contour_cell, Mask_cell = tbat.make_NbYolkCell_contour_and_mask(img, PixPerUm,
                                                                        buffer_um = 0.0,
                                                                        mode = 'dark_background', 
                                                                        PLOT = False)
        
        np.save(os.path.join(srcDir, contourFile), Contour_cell)
        np.save(os.path.join(srcDir, maskFile), Mask_cell)
        
    print(tbat.GREEN + f' Done in {time.time()-top:.3f} s' + tbat.NORMAL, end='\n')
    sys.stdout.flush()
    

# %%%% 2. Run Trackmate


print(tbat.BRIGHTORANGE + '\n\n2. Running Trackmate' + tbat.NORMAL)
for i in range(len(fileNames)):
    tifPath, fN = tifPaths[i], fileNames[i]
    
    top = time.time()
    print(tbat.CYAN + f'\nFile {i+1:.0f}/{len(fileNames):.0f} - {fN}...' + tbat.NORMAL)
    
    if ROI_MODE == 'Square ROIs':
        fileROIs = fN + suffix_squareROI
        list_rois = tbat.json2list(os.path.join(srcDir, fileROIs))
        
        for k, ROI in enumerate(list_rois):
            rawTrackName = fN + f'_SqROI-0{k+1:.0f}' + suffix_TmTracks
            
            tbat.pretreat_and_track_NbYolk(tifPath, rawTrackName, dstDir, PixPerUm, 
                                      Dict_TrackMate_Settings = Dict_TrackMate_Settings,
                                      time_indices = None, ROI_MODE = ROI_MODE, ROI = ROI,
                                      Mask_cell = None, mask_buffer_um = 0.0,
                                      correct_on_edges_buffer_um = 0.5,
                                      PLOT = True, SHOWPLOT = False, SAVEPLOT = True)
            
    elif ROI_MODE == 'Whole cell':
        maskFile = fN + suffix_mask
        Mask_cell = np.load(os.path.join(srcDir, maskFile))
        rawTrackName = fN + '_WholeCell' + suffix_TmTracks
        
        tbat.pretreat_and_track_NbYolk(tifPath, rawTrackName, dstDir, PixPerUm, 
                                       Dict_TrackMate_Settings = Dict_TrackMate_Settings,
                                       time_indices = None, ROI_MODE = ROI_MODE, ROI = None,
                                       Mask_cell = Mask_cell, mask_buffer_um = 1.0,
                                       correct_on_edges_buffer_um = 1.0,
                                       PLOT = True, SHOWPLOT = False, SAVEPLOT = True)

    print(tbat.GREEN + f'\nDone in {time.time()-top:.3f} s' + tbat.NORMAL)
    

# %%%% 3. From Tracks in Trackmate format to Python format

print(tbat.BRIGHTORANGE + '\n\n3. Tracks formatting step\n' + tbat.NORMAL)
for i in range(len(fileNames)): #len(fileNames)
    tP, fN = tifPaths[i], fileNames[i]
    print(tbat.CYAN + f'File {i+1:.0f}/{len(fileNames):.0f} - {fN}...' + tbat.NORMAL, end='')
    sys.stdout.flush()
    top = time.time()
    
    img_0 = tbat.load_stack_region(tP, time_indices=[0])[0]
    
    TmTracksName = fN + '_WholeCell' + suffix_TmTracks
    PyTracksName = fN + suffix_PyTracks
    contourPath = os.path.join(srcDir, fN + suffix_contour)
    
    TmTracks = tbat.import_TrackMate_tracks(os.path.join(dstDir, TmTracksName))
    Contour_cell = np.load(contourPath)

    tbat.rawTracks_2_cleanTracks(TmTracks, dstDir, PyTracksName,
                                 Contour_cell, PixPerUm,
                                 edge_buffer_cutoff_um = 0, nPoints_cuttoff = 30,
                                 RefImg = img_0, PLOT = True, SHOWPLOT = False, SAVEPLOT = True,
                                )
    
    print(tbat.GREEN + f' Done in {time.time()-top:.3f} s' + tbat.NORMAL)
    sys.stdout.flush()
    
    
# %%%% Take a df and correct the drift

def correct_drift_in_Tracks(df, fN, dstDir, PixPerUm, FPS, 
                            PLOT = False, SHOWPLOT = False, SAVEPLOT = False, 
                            export_image = False):
    
    grouped = df.groupby('frame')
    list_df_f = [(f, df_f) for f, df_f in grouped]
    nF = len(list_df_f)
    Dxy = np.zeros((nF-1, 2))
    for (f1, df_f1), (f2, df_f2) in zip(list_df_f[:-1], list_df_f[1:]):
        df_f_m = pd.merge(df_f1, df_f2, on='particle', how='inner', suffixes=('_1', '_2'))
        df_f_m['dx'] = df_f_m['x_2'] - df_f_m['x_1']
        df_f_m['dy'] = df_f_m['y_2'] - df_f_m['y_1']
        Dxy[f1-1, 0] = np.mean(df_f_m['dx'].to_numpy()) #/ PixPerUm
        Dxy[f1-1, 1] = np.mean(df_f_m['dy'].to_numpy()) #/ PixPerUm
    
    sum_Dxy = np.array([np.sum(Dxy[:k, :], axis=0) for k in range(1, nF)])
    
    sDxy = np.zeros_like(Dxy)
    sDxy = savgol_filter(Dxy, 5, 3, axis=0)
    sum_sDxy = np.array([np.sum(sDxy[:k, :], axis=0) for k in range(1, nF)])
    
    if PLOT:
        if not SHOWPLOT:
            plt.ioff()
        fig, axes = plt.subplots(2, 1, figsize=(7, 7), layout='compressed')
        axes[0].set_ylabel('dx, dy')
        axes[1].set_ylabel('Smoothed dx, dy')
        
        ax = axes[0]
        ax.plot(np.arange(1, nF)*FPS, Dxy[:, 0]/PixPerUm, 'b-')
        ax.plot(np.arange(1, nF)*FPS, sum_Dxy[:, 0]/PixPerUm, ls='-', color='cyan')
        ax.plot(np.arange(1, nF)*FPS, Dxy[:, 1]/PixPerUm, 'r-')
        ax.plot(np.arange(1, nF)*FPS, sum_Dxy[:, 1]/PixPerUm, ls='-', color='orange')
        ax.grid()
        
        ax = axes[1]
        ax.plot(np.arange(1, nF)*FPS, sDxy[:, 0], 'b-')
        ax.plot(np.arange(1, nF)*FPS, sum_sDxy[:, 0], ls='-', color='cyan')
        ax.plot(np.arange(1, nF)*FPS, sDxy[:, 1], 'r-')
        ax.plot(np.arange(1, nF)*FPS, sum_sDxy[:, 0], ls='-', color='orange')
        ax.grid()
        ax.set_xlabel('Frame No.')
        
        if SAVEPLOT:
            figName = fN + '_DriftCorr.png'
            figPath = os.path.join(dstDir, figName)
            fig.savefig(figPath)
        
        if SHOWPLOT:
            plt.show()
        if not SHOWPLOT:
            plt.ion()

    if export_image:
        nT = nF
        img = tbat.load_stack_region(tP, time_indices=[i for i in range(nT)])
        tf_img = np.zeros_like(img)
        tf_img[0, :, :] = img[0, :, :]
        # tf_img = []
        # tf_img.append(img[0, :, :])
        for t in range(nT-1): # nF-1
            img_t = img[t+1, :, :]
            tform = skm.transform.EuclideanTransform(translation=(-sum_sDxy[t, 0], -sum_sDxy[t, 1]))
            tf_img[t+1] = skm.transform.warp(img_t, tform.inverse, preserve_range=True)
            # tf_img.append(skm.transform.warp(img_t, tform.inverse, preserve_range=True))
        tf_img = np.array(tf_img)
            
        fpath = os.path.join(dstDir, fN + '_DriftCorr.tif')
        skm.io.imsave(fpath, tf_img)





for i in range(len(fileNames)):
    tP, fN = tifPaths[i], fileNames[i]
    PyTracksName = fN + suffix_PyTracks
    df = pd.read_csv(os.path.join(dstDir, PyTracksName), sep='\t')
    
    
    
    

    
    
# %%%% A test: compute the cell drift image per image

tifNames = [
    '26-09-30_D2-R_PreF_C3_5fps_Texp50ms_L20p_CSU642.tif',
    '26-09-30_D2-R_PreF_C4_5fps_Texp50ms_L20p_CSU642.tif',
    'AVG_20261005_dish3_39C_image_1_CSU642.ome.tf2 - C=0-1.tif',
    ]


tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
fileNames = [fN.split('.')[0] for fN in tifNames]

list_FPS = [20, 20, 1/0.45]
list_PixPerUm = [tbat.PixPerUm_60X_W1, tbat.PixPerUm_60X_W1, tbat.PixPerUm_60X_W1, ]
max_Dt_s = 7.5
list_df = []

# print(tbat.BRIGHTORANGE + '\n\n3. Tracks formatting step\n' + tbat.NORMAL)
for i in range(len(fileNames)): #len(fileNames)
    tP, fN = tifPaths[i], fileNames[i]
    FPS = list_FPS[i]
    PyTracksName = fN + suffix_PyTracks
    df = pd.read_csv(os.path.join(dstDir, PyTracksName), sep='\t')
    list_df.append(df)
    
    # max_lagtime = int(max_Dt_s * FPS)
    # res_emsd = tp.motion.emsd(df, UmPerPix, FPS, max_lagtime=max_lagtime).reset_index()
    # res_emsd.to_csv(os.path.join(dstDir, msdName), index=False, sep='\t')

from scipy.signal import savgol_filter
import skimage as skm

# %%%% Next step

fig, axes = plt.subplots(2, len(list_df), figsize=(4*len(list_df), 6), 
                         sharey=True, layout='compressed')
axes[0, 0].set_ylabel('dx, dy')
axes[1, 0].set_ylabel('Smoothed dx, dy')

for i in range(len(list_df)): # len(list_df)
    tP, fN = tifPaths[i], fileNames[i]
    df = list_df[i]
    PixPerUm = list_PixPerUm[i]
    grouped = df.groupby('frame')
    list_df_f = [(f, df_f) for f, df_f in grouped]
    nF = len(list_df_f)
    Dxy = np.zeros((nF-1, 2))
    for (f1, df_f1), (f2, df_f2) in zip(list_df_f[:-1], list_df_f[1:]):
        df_f_m = pd.merge(df_f1, df_f2, on='particle', how='inner', suffixes=('_1', '_2'))
        df_f_m['dx'] = df_f_m['x_2'] - df_f_m['x_1']
        df_f_m['dy'] = df_f_m['y_2'] - df_f_m['y_1']
        Dxy[f1-1, 0] = np.mean(df_f_m['dx'].to_numpy()) #/ PixPerUm
        Dxy[f1-1, 1] = np.mean(df_f_m['dy'].to_numpy()) #/ PixPerUm
    
    sDxy = np.zeros_like(Dxy)
    sDxy[:, 0] = savgol_filter(Dxy[:, 0], 5, 3)
    sDxy[:, 1] = savgol_filter(Dxy[:, 1], 5, 3)
    
    ax = axes[0, i]
    ax.plot(np.arange(1, len(list_df_f)), Dxy[:, 0], 'b-')
    ax.plot(np.arange(1, len(list_df_f)), [np.sum(Dxy[:k, 0]) for k in range(1, nF)], ls='-', color='cyan')
    ax.plot(np.arange(1, len(list_df_f)), Dxy[:, 1], 'r-')
    ax.plot(np.arange(1, len(list_df_f)), [np.sum(Dxy[:k, 1]) for k in range(1, nF)], ls='-', color='orange')

    ax.grid()
    # ax.set_xlabel('Frame No.')
    
    ax = axes[1, i]
    ax.plot(np.arange(1, len(list_df_f)), sDxy[:, 0], 'b-')
    ax.plot(np.arange(1, len(list_df_f)), [np.sum(sDxy[:k, 0]) for k in range(1, nF)], ls='-', color='cyan')
    ax.plot(np.arange(1, len(list_df_f)), sDxy[:, 1], 'r-')
    ax.plot(np.arange(1, len(list_df_f)), [np.sum(sDxy[:k, 1]) for k in range(1, nF)], ls='-', color='orange')

    ax.grid()
    ax.set_xlabel('Frame No.')
    
    sum_sDxy = np.array([np.sum(sDxy[:k, :], axis=0) for k in range(1, nF)])
    
    nT = nF
    # nT = 100
    
    img = tbat.load_stack_region(tP, time_indices=[i for i in range(nT)])
    tf_img = np.zeros_like(img)
    tf_img[0, :, :] = img[0, :, :]
    # tf_img = []
    # tf_img.append(img[0, :, :])
    for t in range(nT-1): # nF-1
        print(t+1)
        img_t = img[t+1, :, :]
        tform = skm.transform.EuclideanTransform(translation=(-sum_sDxy[t, 0], -sum_sDxy[t, 1]))
        tf_img[t+1] = skm.transform.warp(img_t, tform.inverse, preserve_range=True)
        # tf_img.append(skm.transform.warp(img_t, tform.inverse, preserve_range=True))
        
    tf_img = np.array(tf_img)
        
    testDir = os.path.join(mainDir, 'Tests')
    fpath = os.path.join(testDir, fN + '_trans.tif')
    skm.io.imsave(fpath, tf_img)
    
    
plt.show()


    
# %%%% 3. From Tracks in Python format to MSD and pairMSD
    
# %% ---


