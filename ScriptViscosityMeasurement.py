# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 13:54:28 2025

@author: Joseph
"""

# %% 1. Imports

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import statsmodels.api as sm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

import os
import json
import colorsys

from scipy import interpolate, optimize

import PlotMaker as pm
import UrchinPaths as up
import MagnetsCalibrationsConstants as mcc
import Toolbox1_CalibrateMagnet_MeasureViscosity as tb1

# %% 2. Helper functions

# %%% Graphic settings

# colorListMpl = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
#                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# colorListSns = ['#66c2a5',  '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854','#ffd92f', 
#                 '#e5c494', '#b3b3b3', '#e41a1c', '#377eb8', '#4daf4a',
#                 '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

# colorListSns2 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', 
#                  '#a65628', '#f781bf', '#66c2a5',  '#fc8d62', '#8da0cb', 
#                  '#e78ac3', '#a6d854','#ffd92f', '#e5c494', '#b3b3b3']

# def setGraphicOptions(mode = 'screen', colorList = colorListSns):
#     if mode == 'screen':
#         SMALLER_SIZE = 11
#         SMALL_SIZE = 13
#         MEDIUM_SIZE = 16
#         BIGGER_SIZE = 20
#     if mode == 'screen_big':
#         SMALLER_SIZE = 12
#         SMALL_SIZE = 14
#         MEDIUM_SIZE = 18
#         BIGGER_SIZE = 22
#     elif mode == 'print':
#         SMALLER_SIZE = 8
#         SMALL_SIZE = 10
#         MEDIUM_SIZE = 11
#         BIGGER_SIZE = 12
        
#     plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#     plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#     plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
#     plt.rc('xtick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
#     plt.rc('ytick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
#     plt.rc('legend', fontsize=SMALLER_SIZE)    # legend fontsize
#     plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
#     mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colorList) 
    
    
# def lighten_color(color, factor=1.0):
#     """
#     Source : https://gist.github.com/ihincks/6a420b599f43fcd7dbd79d56798c4e5a
#     and : https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
#     Lightens the given color by multiplying (1-luminosity) by the given amount.
#     Input can be matplotlib color string, hex string, or RGB tuple.

#     Examples:
#     >> lighten_color('g', 0.3)
#     >> lighten_color('#F034A3', 0.6)
#     >> lighten_color((.3,.55,.1), 0.5)
#     """
    
#     try:
#         c = mpl.colors.cnames[color]
#     except:
#         c = color
#     c = colorsys.rgb_to_hls(*mpl.colors.to_rgb(c))
#     new_c = colorsys.hls_to_rgb(c[0], max(0, min(1, factor * c[1])), c[2])
#     return(new_c)



# %% 2. Run an analysis

# %%% Empty template 

parms_pL = [
        39603.33040969049,
        -2.0162526263553215
    ]

D2F_pL = lambda x : powerLaw(x, *parms_pL)

D2F_func = D2F_pL

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = ''

# saveDir is the directory where the data and the plots will be saved
saveDir = ''

expLabel = ''            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0                   # Bead radius, µm - here MyOne Dynabeads
visco = 0                # Medium viscosity, mPa.s - here 75% Gly at 20.6°C
SCALE = 0                # Microscope scale, µm/pixel
FPS = 0                  # Frame per second, 1/s

filesInfo = []

#### Film 1
fI = {}
fI['fileName'] = ''
fI['FPS'] = FPS
fI['MagX'], fI['MagY'], fI['MagR'] = 0, 0, 0 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

#### Run the analysis
runAnalysis(mainDir, SCALE, Rb, D2F_func, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)

# %%% 26/04/09

# Source
# E:\AnalysisPulls\26-03-20_UVonCytoplasmAndBeads_CalibMagnetJN\Calib_MagnetJN_20X_Gly75p_MyOne_Capi01
# MyOne_Glycerol75%_magnetJN_capi_fitData.json
parms_2exp = [
        7.805139888548116,
        102.27510392873741,
        1.2304327867498293,
        270.4339259099587
    ]
D2F_2exp = lambda x : doubleExpo(x, *parms_2exp)
parms_pL = [
        20963.176438241888,
        -1.785737243995788
    ]

D2F_pL = lambda x : powerLaw(x, *parms_pL)

D2F_func = D2F_pL

# %%%% Capillary 1 Control

path = up.Path_AnalysisPulls + '/26-04-09_ViscoInCapillaries'

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = path + '/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = path

expLabel = 'TestUV_before_26-04-09'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

filesInfo = []

#### Film 1
# fI = {}
# fI['fileName'] = '26-04-09_Gly80p_MyOne_HPMA-100mM_I2959-10mM_P1_noUV_Tracks.xml'
# fI['FPS'] = 5
# fI['MagX'], fI['MagY'], fI['MagR'] =  127.5,  472.5, 155 * 0.5 
# fI['CropX'], fI['CropY'] = 0, 0 
# filesInfo.append(fI)


#### Film 5
fI = {}
fI['fileName'] = '26-04-09_Gly80p_MyOne_HPMA-100mM_I2959-10mM_P5_noUV_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  126.5,  516.5, 167 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)


#### Run the analysis
runAnalysis(mainDir, SCALE, Rb, D2F_func, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)

# %%%% Capillary 1 UV 0.03A 5min

path = up.Path_AnalysisPulls + '/26-04-09_ViscoInCapillaries'

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = path + '/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = path

expLabel = 'UV-0A03-5min'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

filesInfo = []

#### Film 2
fI = {}
fI['fileName'] = '26-04-09_Gly80p_MyOne_HPMA-100mM_I2959-10mM_P2_UV-0A03-5min_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  146.5,  461.5, 161 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)


#### Run the analysis
runAnalysis(mainDir, SCALE, Rb, D2F_func, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)

# %%%% Capillary 1 UV 0.1A 5min

path = up.Path_AnalysisPulls + '/26-04-09_ViscoInCapillaries'

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = path + '/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = path

expLabel = 'UV-0A1-5min'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

filesInfo = []


#### Film 3
fI = {}
fI['fileName'] = '26-04-09_Gly80p_MyOne_HPMA-100mM_I2959-10mM_P3_UV-0A1-5min_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  113,  514, 168 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)


#### Run the analysis
runAnalysis(mainDir, SCALE, Rb, D2F_func, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)


# %%%% Capillary 1 UV 0.2A 5min

path = up.Path_AnalysisPulls + '/26-04-09_ViscoInCapillaries'

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = path + '/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = path

expLabel = 'UV-0A2-5min'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

filesInfo = []

#### Film 4
fI = {}
fI['fileName'] = '26-04-09_Gly80p_MyOne_HPMA-100mM_I2959-10mM_P4_UV-0A2-5min_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  120,  513, 168 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

#### Run the analysis
runAnalysis(mainDir, SCALE, Rb, D2F_func, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)




# %%% 26/03/17

# Source
# C:\Users\Utilisateur\Desktop\AnalysisPulls\26-01-07_Calib_MagnetJingAude\26-01-07_20x_MyOneGly75p\Results
# MyOne_Glycerol75%_magnetJX_capi_fitData.json
parms_2exp = [
        5.99012502563366,
        100.84369389492127,
        0.11435249326662983,
        2101.719709808641
    ]
D2F_2exp = lambda x : doubleExpo(x, *parms_2exp)
parms_pL = [
        39603.33040969049,
        -2.0162526263553215
    ]
D2F_pL = lambda x : powerLaw(x, *parms_pL)

D2F_func = D2F_pL

# %%%% Capillary 1 Control

path = up.Path_WorkingData + '/LeicaData/26-03-18_UVonCapillaryBulk'

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = path + '/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = path

expLabel = '26-03-18_TestUV_before'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

filesInfo = []

#### Film 1
fI = {}
fI['fileName'] = 'Capi01_noUV_P1_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  126,  415, 168 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

#### Film 2
fI = {}
fI['fileName'] = 'Capi01_noUV_P2_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  107,  360, 168 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

#### Film 3
fI = {}
fI['fileName'] = 'Capi01_noUV_P3_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  101,  353, 168 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)


#### Run the analysis
runAnalysis(mainDir, SCALE, Rb, D2F_func, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)

# %%%% Capillary 1 UV 600mW 1min

path = up.Path_WorkingData + '/LeicaData/26-03-18_UVonCapillaryBulk'

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = path + '/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = path

expLabel = 'UV-600mW-1min'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

filesInfo = []

#### Film 4
fI = {}
fI['fileName'] = 'Capi01_UV-0A36-1min_P1_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  115,  392, 168 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

#### Film 5
fI = {}
fI['fileName'] = 'Capi01_UV-0A36-1min_P2_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  103,  441, 168 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

#### Run the analysis
runAnalysis(mainDir, SCALE, Rb, D2F_func, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)



# %%%% Capillary 2 Control

path = up.Path_WorkingData + '/LeicaData/26-03-18_UVonCapillaryBulk'

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = path + '/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = path

expLabel = 'TestUV_before'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

filesInfo = []

#### Film 6
fI = {}
fI['fileName'] = 'Capi02_noUV_P1_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  110,  431, 168 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

#### Film 7
fI = {}
fI['fileName'] = 'Capi02_noUV_P2_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  119,  395, 168 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

#### Film 8
fI = {}
fI['fileName'] = 'Capi02_noUV_P3_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  119,  403, 168 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)


#### Run the analysis
runAnalysis(mainDir, SCALE, Rb, D2F_func, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)

# %%%% Capillary 2 UV 120mW 5min

path = up.Path_WorkingData + '/LeicaData/26-03-18_UVonCapillaryBulk'

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = path + '/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = path

expLabel = 'UV-120mW-5min'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

filesInfo = []

#### Film 9
fI = {}
fI['fileName'] = 'Capi02_UV-0A08-5min_P2_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  100,  392, 168 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

#### Run the analysis
runAnalysis(mainDir, SCALE, Rb, D2F_func, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)


# %%%% Capillary 2 UV 1800mW 1min

path = up.Path_WorkingData + '/LeicaData/26-03-18_UVonCapillaryBulk'

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = path + '/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = path

expLabel = 'UV-1800mW-1min'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

filesInfo = []

#### Film 10
fI = {}
fI['fileName'] = 'Capi02_UV-1A08-1min_P1_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  115,  391, 168 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)


#### Run the analysis
runAnalysis(mainDir, SCALE, Rb, D2F_func, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)



# %%% 26/01/07

# Source
# C:\Users\Utilisateur\Desktop\AnalysisPulls\26-01-07_Calib_MagnetJingAude\26-01-07_20x_MyOneGly75p\Results
# MyOne_Glycerol75%_magnetJX_capi_fitData.json
parms_2exp = [
        5.99012502563366,
        100.84369389492127,
        0.11435249326662983,
        2101.719709808641
    ]
D2F_2exp = lambda x : doubleExpo(x, *parms_2exp)
parms_pL = [
        39603.33040969049,
        -2.0162526263553215
    ]
D2F_pL = lambda x : powerLaw(x, *parms_pL)

D2F_func = D2F_pL

# %%%% Control

path = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/26-01-07_TestUV_MagnetJingAude/BeforeUV'

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = path + '/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = path

expLabel = 'TestUV_before'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.451                # Microscope scale, µm/pixel

filesInfo = []

#### Film 1
fI = {}
fI['fileName'] = '26-01-07_20x_MyOneGly75p_BeforeUV_CropInv_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] = 140.5, 375.5, 147 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

#### Run the analysis
runAnalysis(mainDir, SCALE, Rb, D2F_func, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)

# %%%% UV 1A 10 min

path = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/26-01-07_TestUV_MagnetJingAude/AfterUV_1A_10min'

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = path + '/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = path

expLabel = 'TestUV_after'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.451                # Microscope scale, µm/pixel

filesInfo = []

#### Film 1
fI = {}
fI['fileName'] = '26-01-07_20x_MyOneGly75p_AfterUV_CropInv_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] = 108.5, 375.5, 161 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

#### Run the analysis
runAnalysis(mainDir, SCALE, Rb, D2F_func, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)


