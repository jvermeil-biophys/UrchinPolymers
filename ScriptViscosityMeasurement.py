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

import Libs.PlotMaker as pm
import Libs.UrchinPaths as up
import Libs.UtilityFunctions as ufun
import Libs.MagnetsCalibrationsConstants as mcc
import Libs.ToolboxCalibVisco as tbcv


# %% 2. Run an analysis

# %%% Empty template 

magnet, beads, funcType = 'magnet_JX', 'MyOne', 'power law'
D2F_func = mcc.getMagnet_D2F(magnet, beads, funcType)

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
tbcv.runViscoAnalysis(mainDir, SCALE, Rb, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)

# %%% 26/04/09

# Source
# E:\AnalysisPulls\26-03-20_UVonCytoplasmAndBeads_CalibMagnetJN\Calib_MagnetJN_20X_Gly75p_MyOne_Capi01
# MyOne_Glycerol75%_magnetJN_capi_fitData.json

magnet, beads, funcType = 'magnet_JN', 'MyOne', 'power law'
D2F_func = mcc.getMagnet_D2F(magnet, beads, funcType)

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
# runAnalysis(mainDir, SCALE, Rb, D2F_func, filesInfo, 
#                saveDir, expLabel, saveResults, savePlots)
tbcv.runViscoAnalysis(mainDir, SCALE, Rb, D2F_func, filesInfo, 
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


