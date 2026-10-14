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
# import pandas as pd
# import seaborn as sns
# import matplotlib as mpl
# import matplotlib.pyplot as plt

import Toolbox_AutoTrackmate as tbat

#### Settings

# Path_IntraCellTracking = "C:\\Users\\Utilisateur\\Desktop\\IntraCellTracking\\"
# Path_IntraCellTracking = "F:\\IntraCellTracking\\"
Path_IntraCellTracking = ".//Test_Dataset"


# %% Function for the all-in-one pipeline

def all_in_one_pipeline(srcDir, fileNames, tifPaths, dstDir,
                        Dict_TrackMate_Settings = None,
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
                                          ROI_MODE = ROI_MODE, ROI = ROI,
                                          Mask_cell = None, mask_buffer_um = 0.0,
                                          correct_on_edges_buffer_um = 0.5,
                                          PLOT = True, SHOWPLOT = False, SAVEPLOT = True)
                
        elif ROI_MODE == 'Whole cell':
            maskFile = fN + suffix_mask
            Mask_cell = np.load(os.path.join(srcDir, maskFile))
            rawTrackName = fN + '_WholeCell' + suffix_tracks
            
            tbat.pretreat_and_track_NbYolk(tifPath, rawTrackName, dstDir, PixPerUm, 
                                           Dict_TrackMate_Settings = Dict_TrackMate_Settings,
                                           ROI_MODE = ROI_MODE, ROI = None,
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
tifNames = [
    '26-09-30_D2-R_PreF_C3_5fps_Texp50ms_L20p_CSU642.tif',
    # '26-09-30_D2-R_PreF_C4_5fps_Texp50ms_L20p_CSU642.tif',
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

ROI_MODE = 'Square ROIs' # 'Square ROIs' | 'Whole cell'
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
suffix_tracks = '_Tracks.xml'

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


print(tbat.BRIGHTORANGE + '\n\n2. Running Trackmate\n' + tbat.NORMAL)
for i in range(len(fileNames)):
    tifPath, fN = tifPaths[i], fileNames[i]
    
    top = time.time()
    print(tbat.CYAN + f'\nFile {i+1:.0f}/{len(fileNames):.0f} - {fN}...' + tbat.NORMAL)
    
    if ROI_MODE == 'Square ROIs':
        fileROIs = fN + suffix_squareROI
        list_rois = tbat.json2list(os.path.join(srcDir, fileROIs))
        
        for k, ROI in enumerate(list_rois):
            rawTrackName = fN + f'_SqROI-0{k+1:.0f}_Tracks.xml'
            
            tbat.pretreat_and_track_NbYolk(tifPath, rawTrackName, dstDir, PixPerUm, 
                                      Dict_TrackMate_Settings = Dict_TrackMate_Settings,
                                      ROI_MODE = ROI_MODE, ROI = ROI,
                                      Mask_cell = None, mask_buffer_um = 0.0,
                                      correct_on_edges_buffer_um = 0.5,
                                      PLOT = True, SHOWPLOT = False, SAVEPLOT = True)
            
    elif ROI_MODE == 'Whole cell':
        maskFile = fN + suffix_mask
        Mask_cell = np.load(os.path.join(srcDir, maskFile))
        rawTrackName = fN + '_WholeCell_Tracks.xml'
        
        tbat.pretreat_and_track_NbYolk(tifPath, rawTrackName, dstDir, PixPerUm, 
                                       Dict_TrackMate_Settings = Dict_TrackMate_Settings,
                                       ROI_MODE = ROI_MODE, ROI = None,
                                       Mask_cell = Mask_cell, mask_buffer_um = 1.0,
                                       correct_on_edges_buffer_um = 1.0,
                                       PLOT = True, SHOWPLOT = False, SAVEPLOT = True)

    print(tbat.GREEN + f'\nDone in {time.time()-top:.3f} s' + tbat.NORMAL)
    
    
# %% ---


