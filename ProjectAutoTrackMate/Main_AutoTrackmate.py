# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:40:52 2026

@author: josep
"""

# %% Imports & General Settings

#### Imports

import os

import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib as mpl
# import matplotlib.pyplot as plt

import Toolbox_AutoTrackmate as tbat

#### Settings

Path_IntraCellTracking = "C:\\Users\\josep\\Desktop\\Seafile\\DownloadedFromSeafile\\IntraCellTracking\\"


# %% TEMPLATE Day of expt

# %%% Paths

# %%% Get ROIs

# %%% Run Trackmate

# %% ---


# %% Day of expt

# %%% Paths & Settings

mainDir = os.path.join(Path_IntraCellTracking, '26-09-30_FastAcq-Channel_Fec_NB-Yolk')

srcDir = os.path.join(mainDir, 'D2')
dstDir = os.path.join(mainDir, 'SPT_results')
tifNames = [
    '26-09-30_D2-R_PreF_C3_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    '26-09-30_D2-R_PreF_C4_20fps_Texp50ms_L20p1_CSU642.ome.tf2',
    ]

tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
fileNames = [fN.split('.')[0] for fN in tifNames]

# Image properties
UmPerPix = tbat.UmPerPix_60X_W1
PixPerUm = 1/UmPerPix

ROI_MODE = 'Square ROIs' # 'Square ROIs' | 'Whole cell'

# Parameters for square ROIs
L_ROIs = int(15/UmPerPix)
N_ROIs = 3

# Parameters for TrackMate
Dict_TrackMate_Settings = {
    'RADIUS_UM' : 0.8, 
    'THRESH_SPOT_QLT' : 0.25,
    'THRESH_LINK_UM' : 0.25, 
    'THRESH_MIN_DURATION' : 30,
    }

# File suffix
suffix_squareROI = '_listSquareROIs.txt'
suffix_contour =  '_cellContour.npy'
suffix_mask = '_cellMask.npy'


# %%% Get ROIs

# i = 0
for i in range(len(fileNames)):
    tifPath, fN = tifPaths[i], fileNames[i]
    fileROIs = fN + suffix_squareROI
    contourFile = fN + suffix_contour
    maskFile = fN + suffix_mask
    
    shape, dtype = tbat.tiff_inspect(tifPath)
    nT = shape[0]
    TT = range(0, nT, nT//100)
    img = tbat.load_stack_region(tifPath, time_indices=TT)
    
    if ROI_MODE == 'Square ROIs':
        Contour_cell, Mask_cell = tbat.make_NbYolkCell_contour_and_mask(img, PixPerUm,
                                                                        buffer_um = 0.0,
                                                                        mode = 'dark_background', 
                                                                        PLOT = True)
        
        Contour_pronuc, Mask_pronuc = tbat.make_NbYolkPronucleus_contour_and_mask(img, PixPerUm,
                                                                                  Contour_cell, Mask_cell,
                                                                                  mode = 'dark_background', 
                                                                                  buffer_pix = 3,
                                                                                  PLOT = True)
        
        list_rois = tbat.get_N_random_square_ROIs_in_cell(Mask_cell, L_ROIs, N_ROIs, 
                                                          excluded_mask = Mask_pronuc,
                                                          force_distribution = True,
                                                          seed=42, PLOT = False)
        
        np.savetxt(os.path.join(srcDir, fileROIs), list_rois, fmt='%.0f')
    
    
    elif ROI_MODE == 'Whole cell':
        Contour_cell, Mask_cell = tbat.make_NbYolkCell_contour_and_mask(img, PixPerUm,
                                                                        buffer_um = 0.0,
                                                                        mode = 'dark_background', 
                                                                        PLOT = True)
        
        np.save(os.path.join(srcDir, contourFile), Contour_cell)
        np.save(os.path.join(srcDir, maskFile), Mask_cell)

# %%% Run Trackmate

# i = 0
for i in range(len(fileNames)):
    tifPath, fN = tifPaths[i], fileNames[i]
    
    if ROI_MODE == 'Square ROIs':
        fileROIs = fN + suffix_squareROI
        list_rois = np.loadtxt(os.path.join(srcDir, fileROIs))
        
        for k, ROI in enumerate(list_rois):
            rawTrackName = fN + f'_sqROI-0{k:.0f}.xml'
            
            tbat.pretreat_and_track_NbYolk(tifPath, rawTrackName, dstDir, PixPerUm, 
                                      Dict_TrackMate_Settings = Dict_TrackMate_Settings,
                                      ROI_MODE = ROI_MODE, ROI = ROI,
                                      Mask_cell = None, mask_buffer_um = 0.0,
                                      correct_on_edges_buffer_um = 0.5,
                                      PLOT = True, SHOWPLOT = True, SAVEPLOT = True)
            
    elif ROI_MODE == 'Whole cell':
        Mask_cell = np.load(os.path.join(srcDir, maskFile))
        
        tbat.pretreat_and_track_NbYolk(tifPath, rawTrackName, dstDir, PixPerUm, 
                                       Dict_TrackMate_Settings = Dict_TrackMate_Settings,
                                       ROI_MODE = ROI_MODE, ROI = None,
                                       Mask_cell = Mask_cell, mask_buffer_um = 2.0,
                                       correct_on_edges_buffer_um = 0.5,
                                       PLOT = True, SHOWPLOT = True, SAVEPLOT = True)


# %% ---


