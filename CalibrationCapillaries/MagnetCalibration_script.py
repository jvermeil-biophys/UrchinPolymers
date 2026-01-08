# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 15:40:23 2025

@author: Utilisateur
"""

# %% 1. Imports 

from MagnetCalibration_main import runCalibration, compareCalibrations


# %% 2. Empty template 

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


#### Run the calibration
runCalibration(mainDir, SCALE, Rb, visco, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)


# %% 3. Example 

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = './ExampleData/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = './ExampleData/Results_python'

expLabel = 'MyOne_Glycerol75%' # The label for this condition - used as a prefix for saved data and plots
saveResults = True             # If you want to export results as a .json file
savePlots = True               # If you want to save the plots as a .png file
Rb = 1 * 0.5                   # Bead radius, µm - here MyOne Dynabeads
visco = 53.3                   # Medium viscosity, mPa.s - here 75% Gly at 20.6°C
SCALE = 0.451                  # Microscope scale, µm/pixel
FPS = 5                        # Frame per second, 1/s

filesInfo = []

#### Film 1
fI = {}
fI['fileName'] = '25-11-19_Capi04_FilmBF_5fps_1_CropInv_Tracks.xml'
fI['FPS'] = FPS
fI['MagX'], fI['MagY'], fI['MagR'] = 154, 497, 234 * 0.5 
fI['CropX'], fI['CropY'] = 790, 0 
filesInfo.append(fI)

#### Film 2
fI = {}
fI['fileName'] = '25-11-19_Capi04_FilmBF_5fps_2_CropInv_Tracks.xml'
fI['FPS'] = FPS
fI['MagX'], fI['MagY'], fI['MagR'] = 140, 551, 232 * 0.5 
fI['CropX'], fI['CropY'] = 715, 1
filesInfo.append(fI)

#### Film 4
fI = {}
fI['fileName'] = '25-11-19_Capi04_FilmBF_5fps_4_CropInv_Tracks.xml'
fI['FPS'] = FPS
fI['MagX'], fI['MagY'], fI['MagR'] = 149, 610, 238 * 0.5 
fI['CropX'], fI['CropY'] = 723, 0
filesInfo.append(fI)


#### Run the calibration
runCalibration(mainDir, SCALE, Rb, visco, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)


# %% 4. Joseph

# %%%  25-12-05 - Capillaries with thinner walls

path = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/25-12_DynabeadsInCapillaries_CalibrationsTests'

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = path + '/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = path

# %%%% Capi01


# 25-12-05
# Capilariy 01 - ID=500um, w=100um
# MyOne Beads in Glycerol 80%
# T = 20.8°C

expLabel = 'MyOne_Glycerol80%_CapiW100um' # The label for this condition - used as a prefix for saved data and plots
saveResults = True             # If you want to export results as a .json file
savePlots = True               # If you want to save the plots as a .png file
Rb = 1 * 0.5                   # Bead radius, µm - here MyOne Dynabeads
visco = 86.8                   # Medium viscosity, mPa.s - here 75% Gly at 20.6°C
SCALE = 0.451                  # Microscope scale, µm/pixel
FPS = 5                        # Frame per second, 1/s

filesInfo = []

# Film 1
fI = {}
fI['fileName'] = '25-12-05_Capi01_20x_FilmBF_5fps_1_CropInvSub_Tracks.xml'
fI['FPS'] = FPS
fI['MagX'], fI['MagY'], fI['MagR'] = 141.5, 571.5, 239 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# Film 2
fI = {}
fI['fileName'] = '25-12-05_Capi01_20x_FilmBF_5fps_2_CropInv_Tracks.xml'
fI['FPS'] = FPS
fI['MagX'], fI['MagY'], fI['MagR'] = 155.5, 600.5, 241 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# # Film 4
# fI = {}
# fI['fileName'] = '25-12-05_Capi01_20x_FilmBF_5fps_4_CropInv_Tracks.xml'
# fI['MagX'], fI['MagY'], fI['MagR'] = 200.5, 554.5, 241 * 0.5 
# fI['CropX'], fI['CropY'] = 0, 0 
# filesInfo.append(fI)



#### Run the calibration
runCalibration(mainDir, SCALE, Rb, visco, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)

# %%%% Capi02


# 25-12-05
# Capilariy 02 - ID=1000um x 100um, w=70um
# MyOne Beads in Glycerol 80%
# T = 20.8°C

expLabel = 'MyOne_Glycerol80%_CapiW70um' # The label for this condition - used as a prefix for saved data and plots
saveResults = True             # If you want to export results as a .json file
savePlots = True               # If you want to save the plots as a .png file
Rb = 1 * 0.5                   # Bead radius, µm - here MyOne Dynabeads
visco = 86.8                   # Medium viscosity, mPa.s - here 75% Gly at 20.6°C
SCALE = 0.451                  # Microscope scale, µm/pixel
FPS = 5                        # Frame per second, 1/s

filesInfo = []

# Film 1
fI = {}
fI['fileName'] = '25-12-05_Capi02_20x_FilmBF_5fps_1_CropInvSub_Tracks.xml'
fI['FPS'] = FPS
fI['MagX'], fI['MagY'], fI['MagR'] = 145, 488, 234 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

# Film 3
fI = {}
fI['fileName'] = '25-12-05_Capi02_20x_FilmBF_5fps_3_CropInvSub_Tracks.xml'
fI['FPS'] = FPS
fI['MagX'], fI['MagY'], fI['MagR'] = 168, 626, 252 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)



#### Run the calibration
runCalibration(mainDir, SCALE, Rb, visco, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)


# %%%% Compare

srcDir = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/25-12_DynabeadsInCapillaries_CalibrationsTests/Results'
saveDir = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/25-12_DynabeadsInCapillaries_CalibrationsTests/'

labelList = ['MyOne_Glycerol80%', 'MyOne_Glycerol80%_CapiW100um', 'MyOne_Glycerol80%_CapiW70um']

compareCalibrations(srcDir, labelList = labelList, 
                    savePlots = True, saveDir = saveDir,
                    showRawData = True, show2ExpFits = True, showPlFits = False)

# %%% ----------------------------

# %%%  25-12-18 - Jessica's Magnet

path = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/25-12_JessicaMagnet'

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = path + '/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = path

# %%%% Capi01


# 25-12-18
# Jessica Ng's magnet
# Capilariy 01 - ID=500um, w=100um
# MyOne Beads in Glycerol 80%
# T = 22.9°C

expLabel = 'MyOne_Glycerol80%_magnetJN_capi' # The label for this condition - used as a prefix for saved data and plots
saveResults = True             # If you want to export results as a .json file
savePlots = True               # If you want to save the plots as a .png file
Rb = 1 * 0.5                   # Bead radius, µm - here MyOne Dynabeads
visco = 75.897                 # Medium viscosity, mPa.s - here 80% Gly at 22.9°C
SCALE = 0.451                  # Microscope scale, µm/pixel

filesInfo = []

#### Film 1
fI = {}
fI['fileName'] = '25-12-18_20x_FastBFGFP_1_CropInv_Tracks.xml'
fI['FPS'] = 1/0.575
fI['MagX'], fI['MagY'], fI['MagR'] = 151, 360, 174 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

#### Film 2
fI = {}
fI['fileName'] = '25-12-18_20x_FastBFGFP_2_CropInv_Tracks.xml'
fI['FPS'] = 1/0.5945
fI['MagX'], fI['MagY'], fI['MagR'] = 170.500, 381.500, 183 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)



#### Run the calibration
runCalibration(mainDir, SCALE, Rb, visco, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)


# %%%% Droplet01


# 25-12-18
# Jessica Ng's magnet
# Capilariy 01 - ID=500um, w=100um
# MyOne Beads in Glycerol 80%
# T = 22.9°C

expLabel = 'MyOne_Glycerol80%_magnetJN_droplet' # The label for this condition - used as a prefix for saved data and plots
saveResults = True             # If you want to export results as a .json file
savePlots = True               # If you want to save the plots as a .png file
Rb = 1 * 0.5                   # Bead radius, µm - here MyOne Dynabeads
visco = 75.897                 # Medium viscosity, mPa.s - here 80% Gly at 22.9°C
SCALE = 0.451                  # Microscope scale, µm/pixel

filesInfo = []

#### Film 1
fI = {}
fI['fileName'] = '25-12-18_20x_FastBFGFP_Droplet01_1_CropInv_Tracks.xml'
fI['FPS'] = 1/0.636
fI['MagX'], fI['MagY'], fI['MagR'] = 180, 411, 174 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)




#### Run the calibration
runCalibration(mainDir, SCALE, Rb, visco, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)


# %%%% Compare

srcDir = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/25-12_JessicaMagnet/Results'
saveDir = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/25-12_JessicaMagnet/'

labelList = ['MyOne_Glycerol80%_magnetJN_capi', 'MyOne_Glycerol80%_magnetJN_droplet']

compareCalibrations(srcDir, labelList = labelList, 
                    savePlots = True, saveDir = saveDir,
                    showRawData = True, show2ExpFits = True, showPlFits = True)

# %%% ----------------------------

# %%%  26-01-07 - Jing's Magnet



# %%%% Capi01 - MyOne, Gly75%

path = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/26-01-07_Calib_MagnetJingAude/26-01-07_20x_MyOneGly75p'

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = path + '/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = path

# 26-01-07
# Jing Xie's magnet
# Capillariy 01 - ID=500um, w=100um
# MyOne Beads in Glycerol 75%
# T = 21°C

expLabel = 'MyOne_Glycerol75%_magnetJX_capi' # The label for this condition - used as a prefix for saved data and plots
saveResults = True             # If you want to export results as a .json file
savePlots = True               # If you want to save the plots as a .png file
Rb = 1 * 0.5                   # Bead radius, µm - here MyOne Dynabeads
visco = 52.05               # Medium viscosity, mPa.s - here 75% Gly at 21°C
SCALE = 0.451                  # Microscope scale, µm/pixel

filesInfo = []

#### Film 1
fI = {}
fI['fileName'] = '26-01-07_20x_MyOneGly75p_Capillary_1_CropInv_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] = 136.5, 414.5, 155 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

#### Film 3
fI = {}
fI['fileName'] = '26-01-07_20x_MyOneGly75p_Capillary_3_CropInv_Tracks.xml'
fI['FPS'] = 5
fI['MagX'], fI['MagY'], fI['MagR'] = 108.5, 384.5, 147 * 0.5 
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)



#### Run the calibration
runCalibration(mainDir, SCALE, Rb, visco, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)

# %%%% Capi02 - M270, Gly75%

path = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/26-01-07_Calib_MagnetJingAude/26-01-07_20x_M270Gly75p'

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = path + '/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = path

# 26-01-07
# Jing Xie's magnet
# Capillariy 02 - ID=500um, w=100um
# M270 Beads in Glycerol 75%
# T = 21°C

expLabel = 'M270_Glycerol75%_magnetJX_capi' # The label for this condition - used as a prefix for saved data and plots
saveResults = True             # If you want to export results as a .json file
savePlots = True               # If you want to save the plots as a .png file
Rb = 2.7 * 0.5                   # Bead radius, µm - here M270 Dynabeads
visco = 52.05              # Medium viscosity, mPa.s - here 80% Gly at 22.9°C
SCALE = 0.451                  # Microscope scale, µm/pixel

filesInfo = []

#### Film 1
fI = {}
fI['fileName'] = '26-01-07_20x_M270Gly75p_Capillary_10fps_1_CropInv_Tracks.xml'
fI['FPS'] = 10
fI['MagX'], fI['MagY'], fI['MagR'] = 161, 380, 148 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)

#### Film 2
fI = {}
fI['fileName'] = '26-01-07_20x_M270Gly75p_Capillary_10fps_2_CropInv_Tracks.xml'
fI['FPS'] = 10
fI['MagX'], fI['MagY'], fI['MagR'] = 109, 446, 152 * 0.5
fI['CropX'], fI['CropY'] = 0, 0 
filesInfo.append(fI)



#### Run the calibration
runCalibration(mainDir, SCALE, Rb, visco, filesInfo, 
               saveDir, expLabel, saveResults, savePlots)


# %%%% Compare

import numpy as np
import matplotlib.pyplot as plt

import MagnetCalibration_main as MC

def doubleExpo(x, A, k1, B, k2):
    return(A*np.exp(-x/k1) + B*np.exp(-x/k2))

def powerLaw(x, A, k):
    return(A*(x**k))

# Rb = 0.5
visco_JV = 52
visco_JX = 86
XX = np.linspace(100, 450, 1000)

# JV MyOne
# Rb = 0.5
# V_popt_2exp_JVcal = [11.585197879152066, 101.44485565618078, 0.2472776065364326, 1630.137844431001]
# F_popt_2exp_JVcal = [5.6777251762487335, 101.44592253961258, 0.12117095765999744, 1630.7823632583463]
# V2F_JV = 6 * np.pi * visco_JV*1e-3 * Rb*1e-6 * 1e12
# VV_JV = doubleExpo(XX, *V_popt_2exp_JVcal)
# FF_JV = doubleExpo(XX, *F_popt_2exp_JVcal)

# JV M270
Rb = 2.7/2
V_popt_2exp_JVcal = [90.09764127526009, 65.90677826700784, 12.334214812334038, 214.40692675365054]
F_popt_2exp_JVcal = [119.21587409989525, 65.9084742068066, 16.320212861019403, 214.41166616406406]
F_popt_pL_JVcal = [948854.73, -2.123674]
Drag_JV = 6 * np.pi * visco_JV*1e-3 * Rb*1e-6 * 1e12
VV_JV = doubleExpo(XX, *V_popt_2exp_JVcal)
FF_JV = doubleExpo(XX, *F_popt_2exp_JVcal)
FF_JV2 = powerLaw(XX, *F_popt_pL_JVcal)

# JX Calib
Rb = 2.7/2
V_popt_2exp_JXcal = [332, 33.16, 33.13, 115.5]
Drag_JX = 6 * np.pi * visco_JX*1e-3 * Rb*1e-6 * 1e12
VV_JX = doubleExpo(XX, *V_popt_2exp_JXcal)
FF_JX = VV_JX * 1e-6 * Drag_JX


#### Plot the fits
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig = fig
color_V = MC.colorListMpl[0]
color_F = MC.colorListSns[0]

ax = axes[0]
ax.plot(XX, FF_JX, ls='-', color='darkorange', label = 'Calib Jing')
ax.plot(XX, FF_JV, ls='-', color='r', label = 'Calib Joseph - 2Expo')
ax.plot(XX, FF_JV2, ls='-', color='purple', label = 'Calib Joseph - PowerLaw')
# ax.plot(D_plot, V_fit_pLS, 'c-', label = V_label_pLS)
ax.grid()
ax.set_xlim([0, 1.1 * max(XX)])
ax.set_ylim([0, 1.2 * max(FF_JX)])
ax.set_xlabel('d [µm]')
ax.set_ylabel('F [pN]')
# ax.legend()

ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(XX, FF_JX, ls='-', color='darkorange', label = 'Calib Jing')
ax.plot(XX, FF_JV, ls='-', color='r', label = 'Calib Joseph - 2Expo')
ax.plot(XX, FF_JV2, ls='-', color='purple', label = 'Calib Joseph - PowerLaw')
# ax.plot(D_plot, V_fit_pLS, 'c-', label = V_label_pLS)
ax.grid()
ax.set_xlim([50, 2000])
ax.set_ylim([0.01, 100])
ax.set_xlabel('d [µm]')
ax.set_ylabel('F [pN]')
ax.legend(title='Fit on F(d)',
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

MainTitle = 'Calibration Data, JX vs JV'
fig.suptitle(MainTitle)
fig.tight_layout()

plt.show()
