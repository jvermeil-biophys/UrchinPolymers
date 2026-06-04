# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 13:54:28 2025

@author: Joseph
"""

# %% 1. Imports

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

import Libs.PlotMaker as pm
import Libs.UrchinPaths as up
import Libs.UtilityFunctions as ufun
import Libs.ToolboxCalibVisco as tbcv
import Libs.MagnetsCalibrationsConstants as mcc



# %% 2. Run an analysis

# %% Empty template 

magnet, beads, funcType = 'magnet_JX', 'MyOne', 'power law'
D2F_func = mcc.getMagnet_D2F(magnet, beads, funcType)
Mag_dX0 = mcc.getMagnet_dX0(magnet, beads)

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

#### Film 2
fI = {}
fI['fileName'] = ''
fI['FPS'] = FPS
fI['MagX'], fI['MagY'], fI['MagR'] = 0, 0, 0 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

#### Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)

# %% ------


# %% 26-05-21 

magnet, beads, funcType = 'magnet_JV01', 'MyOne', 'power law'
D2F_func = mcc.getMagnet_D2F(magnet, beads, funcType)
Mag_dX0 = mcc.getMagnet_dX0(magnet, beads)

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = up.Path_AnalysisPulls + '26-05-21_ViscoInCapillaries/Tracks'

# saveDir is the directory where the data and the plots will be saved

saveDir = up.Path_AnalysisPulls + '26-05-21_ViscoInCapillaries/ResultsVisco'



# %%% M1 HPMA-200mM I2959-25mM 

# 26-05-21 Magnet Positions
# MyOne beads in Glycerol mix
# M1 HPMA-200mM I2959-25mM 

saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel



#### No UV
# Get infos
filesInfo = []
expLabel = '26-05-21_Magnet-JV01_MyOne_Gly80%_M1_HPMA200mM_PI25mM_UVnone'  
# The label for this condition - used as a prefix for saved data and plots

fI = {}
fI['fileName'] = '26-05-21_M1_Gly80p_HPMA-200mM_I2959-25mM_UV-none_P1_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 400, 506, 140 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-21_M1_Gly80p_HPMA-200mM_I2959-25mM_UV-none_P2_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 407, 504, 140 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# fI = {}
# fI['fileName'] = '26-05-21_M1_Gly80p_HPMA-200mM_I2959-25mM_UV-none_P3_Tracks.xml'
# fI['FPS'] = 2
# fI['MagX'], fI['MagY'], fI['MagR'] = 407, 504, 140 * 0.5
# fI['CropX'], fI['CropY'] = 0, 0 
# filesInfo.append(fI)

# fI = {}
# fI['fileName'] = '26-05-21_M1_Gly80p_HPMA-200mM_I2959-25mM_UV-none_P8_Tracks.xml'
# fI['FPS'] = 2
# fI['MagX'], fI['MagY'], fI['MagR'] = 534, 506, 140 * 0.5
# fI['CropX'], fI['CropY'] = 0, 0 
# filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots, 
                      low_cut = 0.75, high_cut = 1.25)


#### UV 5min 0.2A
# Get infos
filesInfo = []
expLabel = '26-05-21_Magnet-JV01_MyOne_Gly80%_M1_HPMA200mM_PI25mM_UV5min0A2'  
# The label for this condition - used as a prefix for saved data and plots

fI = {}
fI['fileName'] = '26-05-21_M1_Gly80p_HPMA-200mM_I2959-25mM_UV-0A2-5min_P4_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 504, 513, 140 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# fI = {}
# fI['fileName'] = '26-05-21_M1_Gly80p_HPMA-200mM_I2959-25mM_UV-0A2-5min_P6_Tracks.xml'
# fI['FPS'] = 2
# fI['MagX'], fI['MagY'], fI['MagR'] = 508, 499, 140 * 0.5
# fI['CropX'], fI['CropY'] = 0, 0 
# filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots, 
                      low_cut = 0.75, high_cut = 1.25)


#### UV 1min 0.4A
# Get infos
filesInfo = []
expLabel = '26-05-21_Magnet-JV01_MyOne_Gly80%_M1_HPMA200mM_PI25mM_UV1min0A4'  
# The label for this condition - used as a prefix for saved data and plots

fI = {}
fI['fileName'] = '26-05-21_M1_Gly80p_HPMA-200mM_I2959-25mM_UV-0A4-1min_P5_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 503, 517, 140 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# fI = {}
# fI['fileName'] = '26-05-21_M1_Gly80p_HPMA-200mM_I2959-25mM_UV-0A4-1min_P7_Tracks.xml'
# fI['FPS'] = 2
# fI['MagX'], fI['MagY'], fI['MagR'] = 511, 496, 140 * 0.5
# fI['CropX'], fI['CropY'] = 0, 0 
# filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots, 
                      low_cut = 0.75, high_cut = 1.25)

# %%% M2 HPMA-100mM I2959-25mM  

# 26-05-21 Magnet Positions
# MyOne beads in Glycerol mix
# M2 HPMA-100mM I2959-25mM


saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel


#### No UV
# Get infos
filesInfo = []
expLabel = '26-05-21_Magnet-JV01_MyOne_Gly80%_M2_HPMA100mM_PI25mM_UVnone'  
# The label for this condition - used as a prefix for saved data and plots

fI = {}
fI['fileName'] = '26-05-21_M2_Gly80p_HPMA-100mM_I2959-25mM_UV-none_P1_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 418, 524, 140 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-21_M2_Gly80p_HPMA-100mM_I2959-25mM_UV-none_P2_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 415, 528, 140 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-21_M2_Gly80p_HPMA-100mM_I2959-25mM_UV-none_P3_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 407, 516, 140 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots, 
                      low_cut = 0.75, high_cut = 1.25)


#### UV 5min 0.2A
# Get infos
filesInfo = []
expLabel = '26-05-21_Magnet-JV01_MyOne_Gly80%_M2_HPMA100mM_PI25mM_UV5min0A2'  
# The label for this condition - used as a prefix for saved data and plots

fI = {}
fI['fileName'] = '26-05-21_M2_Gly80p_HPMA-100mM_I2959-25mM_UV-0A2-5min_P4_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 467, 525, 140 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-21_M2_Gly80p_HPMA-100mM_I2959-25mM_UV-0A2-5min_P5_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 479, 530, 140 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots, 
                      low_cut = 0.75, high_cut = 1.25)


#### UV 1min 0.4A
# Get infos
filesInfo = []
expLabel = '26-05-21_Magnet-JV01_MyOne_Gly80%_M2_HPMA100mM_PI25mM_UV1min0A4'  
# The label for this condition - used as a prefix for saved data and plots

fI = {}
fI['fileName'] = '26-05-21_M2_Gly80p_HPMA-100mM_I2959-25mM_UV-0A4-1min_P6_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 539, 496, 140 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-21_M2_Gly80p_HPMA-100mM_I2959-25mM_UV-0A4-1min_P7_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 524, 491, 140 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots, 
                      low_cut = 0.75, high_cut = 1.25)


# %%% M3 HPMA-300mM I2959-25mM

# 26-05-21 Magnet Positions
# MyOne beads in Glycerol mix
# M3 HPMA-300mM I2959-25mM

saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

#### No UV
# Get infos
filesInfo = []
expLabel = '26-05-21_Magnet-JV01_MyOne_Gly80%_M3_HPMA300mM_PI25mM_UVnone'  
# The label for this condition - used as a prefix for saved data and plots

fI = {}
fI['fileName'] = '26-05-21_M3_Gly80p_HPMA-300mM_I2959-25mM_UV-none_P1_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 537, 504, 136 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-21_M3_Gly80p_HPMA-300mM_I2959-25mM_UV-none_P2_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 537, 527, 142 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-21_M3_Gly80p_HPMA-300mM_I2959-25mM_UV-none_P3_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 528, 521, 142 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)


#### UV 5min 0.2A
# Get infos
filesInfo = []
expLabel = '26-05-21_Magnet-JV01_MyOne_Gly80%_M3_HPMA300mM_PI25mM_UV5min0A2'  
# The label for this condition - used as a prefix for saved data and plots

fI = {}
fI['fileName'] = '26-05-21_M3_Gly80p_HPMA-300mM_I2959-25mM_UV-0A2-5min_P4_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 523, 492, 142 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-21_M3_Gly80p_HPMA-300mM_I2959-25mM_UV-0A2-5min_P5_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 532, 500, 142 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)


# %% ------



# %% 26-05-11 

magnet, beads, funcType = 'magnet_JV01', 'MyOne', 'power law'
D2F_func = mcc.getMagnet_D2F(magnet, beads, funcType)
Mag_dX0 = mcc.getMagnet_dX0(magnet, beads)

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = up.Path_AnalysisPulls + '26-05-11_ViscoInCapillaries/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = up.Path_AnalysisPulls + '26-05-11_ViscoInCapillaries/ResultsVisco'



# %%% M1 TBD NaSS-500mM noPI 

# 26-05-11 Magnet Positions
# MyOne beads in Glycerol mix
# M1 NaSS-500mM noPI

saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel


#### No UV


# %%% M2 NaSS-300mM I2959-25mM 

# 26-05-11 Magnet Positions
# MyOne beads in Glycerol mix
# M2 NaSS-300mM I2959-25mM

saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel


#### No UV
# Get infos
filesInfo = []
expLabel = '26-05-11_Magnet-JV01_MyOne_Gly80%_M2_NaSS300mM-PI25mM-noUV'  
# The label for this condition - used as a prefix for saved data and plots

# fI = {}
# fI['fileName'] = '26-05-11_M2_Gly80p_NaSS-300mM_I2959-25mM_P1_Tracks.xml'
# fI['FPS'] = 2
# fI['MagX'], fI['MagY'], fI['MagR'] = 507, 526, 148 * 0.5
# fI['CropX'], fI['CropY'] = 0, 0 
# filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-11_M2_Gly80p_NaSS-300mM_I2959-25mM_P2_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 497, 534, 148 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# fI = {}
# fI['fileName'] = '26-05-11_M2_Gly80p_NaSS-300mM_I2959-25mM_P5_Tracks.xml'
# fI['FPS'] = 2
# fI['MagX'], fI['MagY'], fI['MagR'] =  474, 551, 148 * 0.5
# fI['CropX'], fI['CropY'] = 0, 0 
# filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)

#### UV 0.2A 5min
# Get infos
filesInfo = []
expLabel = '26-05-11_Magnet-JV01_MyOne_Gly80%_M2_NaSS300mM-PI25mM-UV5min0A2'  
# The label for this condition - used as a prefix for saved data and plots


fI = {}
fI['fileName'] = '26-05-11_M2_Gly80p_NaSS-300mM_I2959-25mM_UV-0A2-5min_P3_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 495, 537, 148 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots,
                      low_cut = 0.65, high_cut = 1.20)


#### UV 0.4A 1min
# Get infos
filesInfo = []
expLabel = '26-05-11_Magnet-JV01_MyOne_Gly80%_M2_NaSS300mM-PI25mM-UV1min0A4'  
# The label for this condition - used as a prefix for saved data and plots

fI = {}
fI['fileName'] = '26-05-11_M2_Gly80p_NaSS-300mM_I2959-25mM_UV-0A4-1min_P4_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] =  483, 536, 148 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots, 
                      low_cut = 0.55, high_cut = 1.75)


# %%% M3 NaSS-200mM I2959-25mM

# 26-05-11 Magnet Positions
# MyOne beads in Glycerol mix
# M3 NaSS-200mM I2959-25mM

saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

#### No UV
# Get infos
filesInfo = []
expLabel = '26-05-11_Magnet-JV01_MyOne_Gly80%_M3_NaSS200mM-PI25mM-noUV'  
# The label for this condition - used as a prefix for saved data and plots

fI = {}
fI['fileName'] = '26-05-21_M3_Gly80p_NaSS-200mM_I2959-25mM_UV-none_P1_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 537, 504, 136 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-21_M3_Gly80p_NaSS-200mM_I2959-25mM_UV-none_P2_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 537, 527, 142 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-21_M3_Gly80p_NaSS-200mM_I2959-25mM_UV-none_P3_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 528, 521, 142 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)




#### UV 0.2A 5min
# Get infos
filesInfo = []
expLabel = '26-05-11_Magnet-JV01_MyOne_Gly80%_M3_NaSS200mM-PI25mM-UV5min0A2'  
# The label for this condition - used as a prefix for saved data and plots

fI = {}
fI['fileName'] = '26-05-11_M3_Gly80p_NaSS-200mM_I2959-25mM_UV-0A2-5min_P3_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 491.5, 557.5, 143 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-11_M3_Gly80p_NaSS-200mM_I2959-25mM_UV-0A2-5min_P6_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 480.5, 557.5, 143 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)



#### UV 0.4A 1min
# Get infos
filesInfo = []
expLabel = '26-05-11_Magnet-JV01_MyOne_Gly80%_M3_NaSS200mM-PI25mM-UV1min0A4'  
# The label for this condition - used as a prefix for saved data and plots

fI = {}
fI['fileName'] = '26-05-11_M3_Gly80p_NaSS-200mM_I2959-25mM_UV-0A4-1min_P4_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 496.5, 564.5, 143 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)






# %%% M4 NaSS-500mM I2959-25mM 

# 26-05-11 Magnet Positions
# MyOne beads in Glycerol mix
# M4 NaSS-500mM I2959-25mM

saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel


#### No UV
# Get infos
filesInfo = []
expLabel = '26-05-11_Magnet-JV01_MyOne_Gly80%_M4_NaSS500mM-PI25mM-noUV'  
# The label for this condition - used as a prefix for saved data and plots

fI = {}
fI['fileName'] = '26-05-11_M4_Gly80p_NaSS-500mM_I2959-25mM_P1_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 399, 539, 146 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-11_M4_Gly80p_NaSS-500mM_I2959-25mM_P2_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 499, 550, 146 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-11_M4_Gly80p_NaSS-500mM_I2959-25mM_P3_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 510, 547, 146 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)


#### UV 5min 0.2A
# Get infos
filesInfo = []
expLabel = '26-05-11_Magnet-JV01_MyOne_Gly80%_M4_NaSS500mM-PI25mM-UV5min0A2'  
# The label for this condition - used as a prefix for saved data and plots

fI = {}
fI['fileName'] = '26-05-11_M4_Gly80p_NaSS-500mM_I2959-25mM_UV-0A2-5min_P4_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 508, 541, 146 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)


#### UV 1min 0.4A
# Get infos
filesInfo = []
expLabel = '26-05-11_Magnet-JV01_MyOne_Gly80%_M4_NaSS500mM-PI25mM-UV1min0A4'  
# The label for this condition - used as a prefix for saved data and plots

fI = {}
fI['fileName'] = '26-05-11_M4_Gly80p_NaSS-500mM_I2959-25mM_UV-0A4-1min_P5_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 508, 540, 146 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)




# %%% M5 NaSS-100mM I2959-25mM

# 26-05-11 Magnet Positions
# MyOne beads in Glycerol mix
# M5 NaSS-100mM I2959-25mM

saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

#### No UV

# Get infos
filesInfo = []
expLabel = '26-05-11_Magnet-JV01_MyOne_Gly80%_M5_NaSS100mM-PI25mM-noUV'  
# The label for this condition - used as a prefix for saved data and plots

fI = {}
fI['fileName'] = '26-05-11_M5_Gly80p_NaSS-100mM_I2959-25mM_P1_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 510.5, 564.5, 143 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-11_M5_Gly80p_NaSS-100mM_I2959-25mM_P2_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 510.5, 562.5, 143 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# fI = {}
# fI['fileName'] = '26-05-11_M5_Gly80p_NaSS-100mM_I2959-25mM_P7_Tracks.xml'
# fI['FPS'] = 2
# fI['MagX'], fI['MagY'], fI['MagR'] = 405, 544, 148 * 0.5
# fI['CropX'], fI['CropY'] = 0, 0 
# filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots,
                      low_cut = 0.65, high_cut = 1.15)


#### UV 0.2A 5min

# Get infos
filesInfo = []
expLabel = '26-05-11_Magnet-JV01_MyOne_Gly80%_M5_NaSS100mM-PI25mM-UV5min0A2'  
# The label for this condition - used as a prefix for saved data and plots

fI = {}
fI['fileName'] = '26-05-11_M5_Gly80p_NaSS-100mM_I2959-25mM_UV-0A2-5min_P3_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 491.5, 557.5, 143 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-11_M5_Gly80p_NaSS-100mM_I2959-25mM_UV-0A2-5min_P4_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 480.5, 557.5, 143 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots,
                      low_cut = 0.65, high_cut = 1.15)


#### UV 0.4A 1min

# Get infos
filesInfo = []
expLabel = '26-05-11_Magnet-JV01_MyOne_Gly80%_M5_NaSS100mM-PI25mM-UV1min0A4'  
# The label for this condition - used as a prefix for saved data and plots

# fI = {}
# fI['fileName'] = '26-05-11_M5_Gly80p_NaSS-100mM_I2959-25mM_UV-0A4-1min_P5_Tracks.xml'
# fI['FPS'] = 2
# fI['MagX'], fI['MagY'], fI['MagR'] = 496.5, 564.5, 143 * 0.5
# fI['CropX'], fI['CropY'] = 0, 0 
# filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-11_M5_Gly80p_NaSS-100mM_I2959-25mM_UV-0A4-1min_P6_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 496, 559, 148 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots,
                      low_cut = 0.65, high_cut = 1.15)


# %%% M6 NaSS-500mM No PI

# 26-05-11 Magnet Positions
# MyOne beads in Glycerol mix
# M6 NaSS-500mM No PI
          
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel


#### No UV

# Get infos
filesInfo = []
expLabel = '26-05-11_Magnet-JV01_MyOne_Gly80%_M6_NaSS500mM-noPI-noUV'  
# The label for this condition - used as a prefix for saved data and plots

fI = {}
fI['fileName'] = '26-05-11_M6_Gly80p_NaSS-500mM_P2_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] =  392.5, 557.5, 139 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-11_M6_Gly80p_NaSS-500mM_P4_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 424, 566, 144 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-11_M6_Gly80p_NaSS-500mM_P6_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] =  317, 564, 148 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)


#### UV 0.2A 5min

# Get infos
filesInfo = []
expLabel = '26-05-11_Magnet-JV01_MyOne_Gly80%_M6_NaSS500mM-noPI-UV5min0A2'  
# The label for this condition - used as a prefix for saved data and plots

fI = {}
fI['fileName'] = '26-05-11_M6_Gly80p_NaSS-500mM_UV-0A2-5min_P3_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] = 369, 559, 142 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-11_M6_Gly80p_NaSS-500mM_UV-0A2-5min_P5_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] =  385, 582, 142 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)

#### UV 0.4A 1min

# Get infos
filesInfo = []
expLabel = '26-05-11_Magnet-JV01_MyOne_Gly80%_M6_NaSS500mM-noPI-UV1min0A4'  
# The label for this condition - used as a prefix for saved data and plots

fI = {}
fI['fileName'] = '26-05-11_M6_Gly80p_NaSS-500mM_UV-0A4-1min_P7_Tracks.xml'
fI['FPS'] = 2
fI['MagX'], fI['MagY'], fI['MagR'] =  412.5, 574.5, 143 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# Run the analysis
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)





# %% ------

# %% 26-05-07 

magnet, beads, funcType = 'magnet_JV01', 'MyOne', 'power law'
D2F_func = mcc.getMagnet_D2F(magnet, beads, funcType)
Mag_dX0 = mcc.getMagnet_dX0(magnet, beads)

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = up.Path_AnalysisPulls + '26-05-07_ViscoInCapillaries/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = up.Path_AnalysisPulls + '26-05-07_ViscoInCapillaries/ResultsVisco'

# %%% M2 Gly80%

expLabel = '26-05-07_Magnet-JV01_MyOne_Glycerol80%_M2_P2'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

filesInfo = []

fI = {}
fI['fileName'] = '26-05-07_M2_Gly80p_P2_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  518.5, 578.5, 143 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)


#### Run the analysis

tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)

# %%% M3 - noUV

expLabel = '26-05-07_Magnet-JV01_MyOne_Gly80%_M3_HPMA300-PI50-noUV_P6'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

filesInfo = []

# fI = {}
# fI['fileName'] = '26-05-07_M3_Gly80p_HPMAg-300mM_I2959-50mM_UV-none_P1_Tracks.xml'
# fI['FPS'] = 5
# fI['MagX'], fI['MagY'], fI['MagR'] =  575.5, 597.5, 137 * 0.5
# fI['CropX'], fI['CropY'] = 0, 0 
# filesInfo.append(fI)

# fI = {}
# fI['fileName'] = '26-05-07_M3_Gly80p_HPMAg-300mM_I2959-50mM_UV-none_P2_Tracks.xml'
# fI['FPS'] = 5
# fI['MagX'], fI['MagY'], fI['MagR'] = 548.5, 621.5, 133 * 0.5
# fI['CropX'], fI['CropY'] = 0, 0 
# filesInfo.append(fI)


fI = {}
fI['fileName'] = '26-05-07_M3_Gly80p_HPMAg-300mM_I2959-50mM_UV-none_P6_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] = 504.5, 587.5, 133 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)


#### Run the analysis

tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)

# %%% M3 - UV 0.2A 5min

expLabel = '26-05-07_Magnet-JV01_MyOne_Gly80%_M3_HPMA300-PI50-UV0A2-5min'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

filesInfo = []

fI = {}
fI['fileName'] = '26-05-07_M3_Gly80p_HPMAg-300mM_I2959-50mM_UV-0A2-5min_P3_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] = 501.5, 619.5, 139 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-07_M3_Gly80p_HPMAg-300mM_I2959-50mM_UV-0A2-5min_P4_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] = 473, 556, 152 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)


#### Run the analysis

tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)

# %%% M3 - UV 0.2A 10min

expLabel = '26-05-07_Magnet-JV01_MyOne_Gly80%_M3_HPMA300-PI50-UV0A2-10min'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

filesInfo = []

fI = {}
fI['fileName'] = '26-05-07_M3_Gly80p_HPMAg-300mM_I2959-50mM_UV-0A2-10min_P7_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] = 510, 580, 138 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-07_M3_Gly80p_HPMAg-300mM_I2959-50mM_UV-0A2-10min_P8_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] = 504, 581, 148 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)


#### Run the analysis

tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)


# %%% M3 - UV 1A 5min

expLabel = '26-05-07_Magnet-JV01_MyOne_Gly80%_M3_HPMA300-PI50-UV1A-5min'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

filesInfo = []

fI = {}
fI['fileName'] = '26-05-07_M3_Gly80p_HPMAg-300mM_I2959-50mM_UV-1A-5min_P5_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  415.5, 560.5, 135 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)


#### Run the analysis

tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)

# %%% M4 - noUV

expLabel = '26-05-07_Magnet-JV01_MyOne_Gly80%_M4_HPMA300-PI100-noUV'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

filesInfo = []

fI = {}
fI['fileName'] = '26-05-07_M4_Gly80p_HPMAg-300mM_I2959-100mM_UV-none_P1_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  424, 588, 144 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-07_M4_Gly80p_HPMAg-300mM_I2959-100mM_UV-none_P3_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] = 431, 569, 152 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)



#### Run the analysis

tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)


# %%% M4 - UV 0.2A 5min

expLabel = '26-05-07_Magnet-JV01_MyOne_Gly80%_M4_HPMA300-PI100-UV-0A2-5min'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

filesInfo = []

fI = {}
fI['fileName'] = '26-05-07_M4_Gly80p_HPMAg-300mM_I2959-100mM_UV-0A2-5min_P2_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] = 430, 570, 148 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

fI = {}
fI['fileName'] = '26-05-07_M4_Gly80p_HPMAg-300mM_I2959-100mM_UV-0A2-5min_P4_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] = 438, 522, 138 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)


#### Run the analysis

tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)



# %% ------

# %% 26-04-30 - test with calib data
#### (kind of a circular reasonning, i know)

magnet, beads, funcType = 'magnet_JV01', 'MyOne', 'power law'
D2F_func = mcc.getMagnet_D2F(magnet, beads, funcType)
Mag_dX0 = mcc.getMagnet_dX0(magnet, beads)

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = up.Path_AnalysisPulls + '26-04-30_CalibMagnet_JV01_and_JN/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = up.Path_AnalysisPulls + '26-04-30_CalibMagnet_JV01_and_JN/Test_MeasVisco'

# %%% Film 1

expLabel = '26-04-30_Magnet-JV01_MyOne_GlycerolX%_M1_P1'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

filesInfo = []

fI = {}
fI['fileName'] = '26-04-30_M1_Gly80p_Magnet-JV01_capi01_P1_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  368, 496, 146 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)


#### Run the analysis

tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)


# %%% Film 2

expLabel = '26-04-30_Magnet-JV01_MyOne_GlycerolX%_M1_P2'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

filesInfo = []

fI = {}
fI['fileName'] = '26-04-30_M1_Gly80p_Magnet-JV01_capi01_P2_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  331.5, 506.5, 149 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)


#### Run the analysis

tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)

# %%% Film 3

expLabel = '26-04-30_Magnet-JV01_MyOne_GlycerolX%_M1_P3'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

filesInfo = []

fI = {}
fI['fileName'] = '26-04-30_M1_Gly80p_Magnet-JV01_capi01_P3_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  354.5, 486.5, 147 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)


#### Run the analysis

tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)

# %%% Film 4

expLabel = '26-04-30_Magnet-JV01_MyOne_GlycerolX%_M2_P1'            # The label for this condition - used as a prefix for saved data and plots
saveResults = True       # If you want to export results as a .json file
savePlots = True         # If you want to save the plots as a .png file
Rb = 0.5                   # Bead radius, µm - here MyOne Dynabeads
SCALE = 0.461                # Microscope scale, µm/pixel

filesInfo = []


fI = {}
fI['fileName'] = '26-04-30_M2_Gly75p_Magnet-JV01_capi02_P1_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] =  490, 522, 150 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)


#### Run the analysis

tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)

# %% 26-04-09

# Source
# E:\AnalysisPulls\26-03-20_UVonCytoplasmAndBeads_CalibMagnetJN\Calib_MagnetJN_20X_Gly75p_MyOne_Capi01
# MyOne_Glycerol75%_magnetJN_capi_fitData.json

magnet, beads, funcType = 'magnet_JN', 'MyOne', 'power law'
D2F_func = mcc.getMagnet_D2F(magnet, beads, funcType)

# %%% Capillary 1 Control

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
# tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
#                saveDir, expLabel, saveResults, savePlots)
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
                      saveDir, expLabel, saveResults, savePlots)

# %%% Capillary 1 UV 0.03A 5min

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
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)

# %%% Capillary 1 UV 0.1A 5min

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
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
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
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)




# %% 26-03-18

# Source
# C:\Users\Utilisateur\Desktop\AnalysisPulls\26-01-07_Calib_MagnetJingAude\26-01-07_20x_MyOneGly75p\Results
# MyOne_Glycerol75%_magnetJX_capi_fitData.json
# parms_2exp = [
#         5.99012502563366,
#         100.84369389492127,
#         0.11435249326662983,
#         2101.719709808641
#     ]
# D2F_2exp = lambda x : mcc.doubleExpo(x, *parms_2exp)
# parms_pL = [
#         39603.33040969049,
#         -2.0162526263553215
#     ]
# D2F_pL = lambda x : mcc.powerLaw(x, *parms_pL)

# D2F_func = D2F_pL

magnet, beads, funcType = 'magnet_JX', 'MyOne', 'power law'
D2F_func = mcc.getMagnet_D2F(magnet, beads, funcType)
Mag_dX0 = mcc.getMagnet_dX0(magnet, beads)

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = up.Path_AnalysisPulls + '26-03-18_ViscoInCapillaries/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = up.Path_AnalysisPulls + '26-03-18_ViscoInCapillaries/NewMeasVisco'

# %%%% Capillary 1 Control

expLabel = '26-03-18_M1_TestUV_before'            # The label for this condition - used as a prefix for saved data and plots
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
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)

# %%%% Capillary 1 UV 600mW 1min


expLabel = '26-03-18_M1_UV-600mW-1min'            # The label for this condition - used as a prefix for saved data and plots
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
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)



# %%%% Capillary 2 Control


expLabel = '26-03-18_M2_TestUV_before'            # The label for this condition - used as a prefix for saved data and plots
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
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)

# %%%% Capillary 2 UV 120mW 5min


expLabel = '26-03-18_M2_UV-120mW-5min'            # The label for this condition - used as a prefix for saved data and plots
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
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)


# %%%% Capillary 2 UV 1800mW 1min

expLabel = '26-03-18_M2_UV-1800mW-1min'            # The label for this condition - used as a prefix for saved data and plots
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
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)



# %% 26-01-07

# Source
# C:\Users\Utilisateur\Desktop\AnalysisPulls\26-01-07_Calib_MagnetJingAude\26-01-07_20x_MyOneGly75p\Results
# MyOne_Glycerol75%_magnetJX_capi_fitData.json
parms_2exp = [
        5.99012502563366,
        100.84369389492127,
        0.11435249326662983,
        2101.719709808641
    ]
D2F_2exp = lambda x : mcc.doubleExpo(x, *parms_2exp)
parms_pL = [
        39603.33040969049,
        -2.0162526263553215
    ]
D2F_pL = lambda x : mcc.powerLaw(x, *parms_pL)

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
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
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
tbcv.runSimpleViscoAnalysis(mainDir, SCALE, Rb, Mag_dX0, D2F_func, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)


