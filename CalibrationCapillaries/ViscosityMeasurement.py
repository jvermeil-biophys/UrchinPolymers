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


# %% 2. Helper functions

# %%% Graphic settings

colorListMpl = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

colorListSns = ['#66c2a5',  '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854','#ffd92f', 
                '#e5c494', '#b3b3b3', '#e41a1c', '#377eb8', '#4daf4a',
                '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

colorListSns2 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', 
                 '#a65628', '#f781bf', '#66c2a5',  '#fc8d62', '#8da0cb', 
                 '#e78ac3', '#a6d854','#ffd92f', '#e5c494', '#b3b3b3']

def setGraphicOptions(mode = 'screen', colorList = colorListSns):
    if mode == 'screen':
        SMALLER_SIZE = 11
        SMALL_SIZE = 13
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 20
    if mode == 'screen_big':
        SMALLER_SIZE = 12
        SMALL_SIZE = 14
        MEDIUM_SIZE = 18
        BIGGER_SIZE = 22
    elif mode == 'print':
        SMALLER_SIZE = 8
        SMALL_SIZE = 10
        MEDIUM_SIZE = 11
        BIGGER_SIZE = 12
        
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALLER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colorList) 
    
    
def lighten_color(color, factor=1.0):
    """
    Source : https://gist.github.com/ihincks/6a420b599f43fcd7dbd79d56798c4e5a
    and : https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    
    try:
        c = mpl.colors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mpl.colors.to_rgb(c))
    new_c = colorsys.hls_to_rgb(c[0], max(0, min(1, factor * c[1])), c[2])
    return(new_c)



# %%% Fitting routines & functions


#### Fitting routines
def fitLine(X, Y):
    """
    returns: results.params, results \n
    Y=a*X+b ; params[0] = b,  params[1] = a
    
    NB:
        R2 = results.rsquared \n
        ci = results.conf_int(alpha=0.05) \n
        CovM = results.cov_params() \n
        p = results.pvalues \n
    
    This is how one should compute conf_int:
        bse = results.bse \n
        dist = stats.t \n
        alpha = 0.05 \n
        q = dist.ppf(1 - alpha / 2, results.df_resid) \n
        params = results.params \n
        lower = params - q * bse \n
        upper = params + q * bse \n
    """
    
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit()
    params = results.params 
#     print(dir(results))
    return(results.params, results)


def fitLineHuber(X, Y, with_wlm_results = False):
    """
    returns: results.params, results \n
    Y=a*X+b ; params[0] = b,  params[1] = a
    
    NB:
        R2 = results.rsquared \n
        ci = results.conf_int(alpha=0.05) \n
        CovM = results.cov_params() \n
        p = results.pvalues \n
    
    This is how one should compute conf_int:
        bse = results.bse \n
        dist = stats.t \n
        alpha = 0.05 \n
        q = dist.ppf(1 - alpha / 2, results.df_resid) \n
        params = results.params \n
        lower = params - q * bse \n
        upper = params + q * bse \n
    """
    
    X = sm.add_constant(X)
    model = sm.RLM(Y, X, M=sm.robust.norms.HuberT())
    results = model.fit()
    params = results.params
    
    if not with_wlm_results:
        out = (results.params, results)
    else:
        weights = results.weights
        w_model = sm.WLS(Y, X, weights)
        w_results = w_model.fit()
        out = (results.params, results, w_results)
    return(out)


#### Good models to fit the Force-Distance function
def doubleExpo(x, A, k1, B, k2):
    return(A*np.exp(-x/k1) + B*np.exp(-x/k2))

def powerLaw(x, A, k):
    return(A*(x**k))


# %%% File save & load


def dict2json(d, dirPath, fileName):
    for k in d.keys():
        obj = d[k]
        if isinstance(obj, np.ndarray):
            d[k] = d[k].tolist()
        else:
            pass
    with open(os.path.join(dirPath, fileName + '.json'), 'w') as fp:
        json.dump(d, fp, indent=4)
        
        
def json2dict(dirPath, fileName):
    with open(os.path.join(dirPath, fileName + '.json'), 'r') as fp:
        d = json.load(fp)
    for k in d.keys():
        obj = d[k]
        if isinstance(obj, list):
            d[k] = np.array(d[k])
        else:
            pass
    return(d)


def listOfdict2json(L, dirPath, fileName):
    for d in L:
        for k in d.keys():
            obj = d[k]
            if isinstance(obj, np.ndarray):
                d[k] = d[k].tolist()
            else:
                pass
    with open(os.path.join(dirPath, fileName + '.json'), 'w') as fp:
        json.dump(L, fp, indent=4)
      
        
def json2listOfdict(dirPath, fileName):
    with open(os.path.join(dirPath, fileName + '.json'), 'r') as fp:
        L = json.load(fp)
    for d in L:
        for k in d.keys():
            obj = d[k]
            if isinstance(obj, list):
                d[k] = np.array(d[k])
            else:
                pass
    return(L)


def importTrackMateTracks(filepath):
    """
    Parse a TrackMate XML file and return list of tracks.
    Each track: numpy array [t, x, y].
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    tracks = []
    for particle in root.findall('particle'):
        L = []
        for detection in particle.iter("detection"):
            # print(detection)
            # ID = int(spot.attrib["ID"])
            t = float(detection.attrib["t"])
            x = float(detection.attrib["x"])
            y = float(detection.attrib["y"])
            L.append([t, x, y])
        tracks.append(np.array(L))
    return(tracks)

# %% Clean Tracks of weird points when importing them

def cleanRawTrack(track):
    track_valid = True
    cleaned_track = track
    N = len(track)
    X = track[:,1]
    Y = track[:,2]
    mX, MX = X==min(X), X==max(X)
    mY, MY = Y==min(Y), Y==max(Y)
    NmX, NMX, NmY, NMY = sum(mX), sum(MX), sum(mY), sum(MY)
    max_saturation = max(NmX, NMX, NmY, NMY)
    if max_saturation >= 0.8*N or (N - max_saturation) < 20:
        track_valid = False
    elif max_saturation > 2:
        i = np.argmax([NmX, NMX, NmY, NMY])
        filter_array = ~np.array([mX, MX, mY, MY][i])
        cleaned_track = cleaned_track[filter_array]
    return(cleaned_track, track_valid)


def cleanAllRawTracks(all_tracks):
    clean_tracks = []
    for track in all_tracks:
        cleaned_track, track_valid = cleanRawTrack(track)
        if track_valid:
            clean_tracks.append(cleaned_track)
    return(clean_tracks)


# %% 3. Core functions

def tracks_pretreatment(all_tracks, SCALE, FPS, 
                        MagX, MagY, MagR, Rb, CropX, CropY,
                        D2F_func):
    tracks_data = []
    MagX *= SCALE
    MagY *= SCALE
    MagR *= SCALE
    CropX *= SCALE
    CropY *= SCALE
    all_tracks = cleanAllRawTracks(all_tracks)

    for i, track in enumerate(all_tracks):
        #### Conversion in um and sec
        T = track[:, 0] * (1/FPS)
        X = track[:, 1] * SCALE
        Y = track[:, 2] * SCALE
                
        tracks_data.append({'T':T, 'Xraw':X, 'Yraw':Y})
        
        #### Origin as the magnet center
        X2, Y2 = (X+CropX)-MagX, MagY-(Y+CropY)
        # NB: inversion of Y so that the trajectories 
        # shown by matplotlib look like the Fiji ones
        medX2, medY2 = np.median(X2), np.median(Y2)
        tracks_data[i].update({'X':X2, 'Y':Y2,
                               'medX':medX2, 'medY':medY2})
        #### Rotate the trajectory by its own angle
        parms, res, wlm_res = fitLineHuber(X2, Y2, with_wlm_results=True)
        b_fit, a_fit = parms
        r2 = wlm_res.rsquared
        theta = np.atan(a_fit)
        # rotation_mat = np.array([[np.cos(-theta), -np.sin(-theta)],
        #                          [np.sin(-theta),  np.cos(-theta)]])
        # rotated_XY = np.vstack((X2, Y2)).T @ rotation_mat.T
        # X3, Y3 = rotated_XY[:,0], rotated_XY[:,1]
        tracks_data[i].update({'a_fit':a_fit, 'b_fit':b_fit, 'r2_fit':r2,
                               'theta':theta,
                               # 'X3':X3, 'Y3':Y3,
                               })
        #### Rotate the trajectory by its angle with the magnet
        phi = np.atan(medY2/medX2)
        delta = theta-phi # delta is the angle between the traj fit & strait line to the magnet
        # rotation_mat = np.array([[np.cos(-phi), -np.sin(-phi)],
        #                          [np.sin(-phi),  np.cos(-phi)]])
        # rotated_XY = np.vstack((X2, Y2)).T @ rotation_mat.T
        # X4, Y4 = rotated_XY[:,0], rotated_XY[:,1]
        tracks_data[i].update({'phi':phi, 'delta':delta,
                               # 'X4':X4, 'Y4':Y4,
                               })
        #### Compute distances [Several possible definitions]
        D2 = np.array(((X2**2 + Y2**2)**0.5) - MagR) # Seems like the best one
        # D3 = np.array(X3 - MagR)
        # D4 = np.array(X4 - MagR) # Note: D4 == D2, almost
        # tracks_data[i].update({'D2':D2, 'D3':D3, 'D4':D4,
        #                        })
        
        #### Smooth D with splines and compute velocity V
        # Chose distance definition
        D = D2 
        # Make spline
        spline_D = interpolate.make_splrep(T, D, s=6)
        spline_V = spline_D.derivative(nu=1)
        # Compute V and chose definition
        V_spline = np.abs(spline_V(T))
        # V_savgol = dxdt(D, T, kind="savitzky_golay", 
        #                 left=10, right=10, order=3)
        # V_kalman = dxdt(D, T, kind="kalman", alpha=1)
        V = V_spline # Seems to be the best one
        
        # Compute the force, ie the viscous drag for given dynamic viscosity in mPa.s
        # USING THE CALIBRATION FUNCTION
        F = D2F_func(D)
        
        #
        medD, medV, medF = np.median(D), np.median(V), np.median(F)
        tracks_data[i].update({'D':D, 'V':V, 'F':F,
                               'medD':medD, 'medV':medV, 'medF':medF
                               })
    return(tracks_data)



def tracks_analysis(tracks_data, Rb = 0.5, expLabel = '',
                    saveResults = True, savePlots = True, saveDir = '',
                    return_fig = 0):
    
    MagR = 60 # µm - Typical Diameter
    
    #### 1.1 First filter
    tracks_data_f1 = []
    for track in tracks_data:
        crit1 = (np.abs(track['delta']*180/np.pi) < 30)
        crit2 = (np.abs(track['r2_fit'] > 0.10))
        # crit3 = not ((np.median(track['D']) < 300) and (np.median(track['V']) < 0.1))
        # bypass1 = (np.min(track['X']) < 300)
        if (crit1 and crit2): #  and crit3
            tracks_data_f1.append(track)
        
    fig1, axes1 = plt.subplots(1, 3, figsize = (24, 8))
    fig = fig1
    #### 1.2 Plot all trajectories
    ax = axes1[0]
    ax.set_title(f'All tracks, N = {len(tracks_data)}')
    for track in tracks_data:
        X, Y = track['X'], track['Y']
        ax.plot(X, Y)
    
    #### 1.3 Plot first filter
    ax = axes1[1]
    ax.set_title(f'First filter, N = {len(tracks_data_f1)}')
    for track in tracks_data_f1:
        X, Y = track['X'], track['Y']
        ax.plot(X, Y)

    for ax in axes1[:2]:
        circle1 = plt.Circle((0, 0), MagR, color='dimgrey')
        ax.add_patch(circle1)
        # ax.axvspan(wall_L, wall_R, color='lightgray', zorder=0)
        ax.set_xlim([0, 800])
        ax.set_ylim([-400, +400])
        ax.grid()
        ax.axis('equal')
        ax.set_xlabel('X [µm]')
        ax.set_ylabel('Y [µm]')
        
        
    #### 2.1 Second Filter
    all_medD = np.array([track['medD'] for track in tracks_data_f1])
    all_medV = np.array([track['medV'] for track in tracks_data_f1])
    all_D = np.concat([track['D'] for track in tracks_data_f1])
    all_V = np.concat([track['V'] for track in tracks_data_f1])
    
    D_plot = np.linspace(1, 5000, 500)

    # Double Expo
    V_popt_2exp, V_pcov_2exp = optimize.curve_fit(doubleExpo, all_D, all_V, 
                           p0 = [1000, 50, 100, 1000], 
                           bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
    V_fit_2exp = doubleExpo(D_plot, *V_popt_2exp)
    label_2exp = r'$\bf{A \cdot exp(-x/k_1) + B \cdot exp(-x/k_2)}$' + '\n'
    label_2exp += '$A$ = {:.2e} | $k_1$ = {:.2f}\n$B$ = {:.2f} | $k_2$ = {:.2e}'.format(*V_popt_2exp)

    # Power Law
    V_popt_pL, V_pcov_pL = optimize.curve_fit(powerLaw, all_D, all_V, 
                           p0 = [1000, -2], 
                           bounds=([0, -10], [np.inf, 0]))
    V_fit_pL = powerLaw(D_plot, *V_popt_pL)
    V_label_pL = r'$\bf{A \cdot x^k}$' + '\n'
    V_label_pL += '$A$ = {:.2e} | $k$ = {:.2f}'.format(*V_popt_pL)

    expected_medV = powerLaw(all_medD, *V_popt_pL)
    ratio_fitV = all_medV/expected_medV
    high_cut = 1.45
    low_cut = 0.55

    tracks_data_f2 = []
    removed_tracks = []
    for i, track in enumerate(tracks_data_f1):
        if (ratio_fitV[i] > low_cut) and (ratio_fitV[i] < high_cut):
            tracks_data_f2.append(track)
        else:
            removed_tracks.append(track)
            
    all_D = np.concat([track['D'] for track in tracks_data_f2])
    all_V = np.concat([track['V'] for track in tracks_data_f2])
    all_F = np.concat([track['F'] for track in tracks_data_f2])
    all_removedD = np.concat([track['D'] for track in removed_tracks])
    all_removedV = np.concat([track['V'] for track in removed_tracks])
    
    #### 2.2 Plot second filter
    fig, ax = fig1, axes1[2]
    ax.plot(all_D, all_V, ls='', marker='.', alpha=0.05)
    ax.plot(D_plot, V_fit_pL, ls='-.', c='darkred', lw=2.0, label = 'Naive fit')
    ax.plot(D_plot, V_fit_pL*high_cut, 
            ls='-.', c='green', lw=1.25, label = f'High cut = {high_cut}')
    ax.plot(D_plot, V_fit_pL*low_cut,  
            ls='-.', c='blue', lw=1.25, label = f'Low cut = {low_cut}')
    ax.plot(all_removedD, all_removedV, ls='', marker='.', alpha=0.05)
    MD, MV = max(all_D), max(all_V)
    ax.set_xlim([0, 1.1*MD])
    ax.set_ylim([0, 1.2*MV])
    ax.grid()
    ax.legend()
    ax.set_xlabel('D [µm]')
    ax.set_ylabel('V [µm/s]')
    ax.set_title(f'Second Filter, N = {len(tracks_data_f2)}')
    
    
    #### 3.1 Final fits
    # Global visco calculation
    [a, b] = np.polyfit(all_V, all_F, 1)
    visco_global = a / (6*np.pi*Rb)
    print(visco_global)
    
    # Calculation per trajectory
    visco_list = []
    for track in tracks_data_f2:
        tV = track['V']
        tF = track['F']
        [ta, tb] = np.polyfit(tV, tF, 1)
        visco = ta / (6*np.pi*Rb)
        visco_list.append(visco)
    
    # Plots
    V_plot = np.linspace(0, np.max(all_V), 100)
    fig2, axes2 = plt.subplots(1, 2)
    
    ax = axes2[0]
    ax.plot(all_V, all_F, ls='', marker='.', alpha=0.05)
    ax.plot(V_plot, a*V_plot + b, ls='-', c='darkred', lw=1.5, 
            label = f'Fit y=ax+b,\n a = {a:.2f}, b = {b:.2f}')

    ax = axes2[1]
    sns.swarmplot(ax=ax, x=['']*len(visco_list), y=visco_list)
    
    plt.show()
    
    
# %% 4. Main functions

def runAnalysis(mainDir, SCALE, Rb, D2F_func, filesInfo, 
                saveDir, expLabel, saveResults, savePlots):
    
    setGraphicOptions(mode = 'screen_big', colorList = colorListMpl)
    
    tracks_data = [];
    Nfiles = len(filesInfo)

    # 1. Import all the files data & run the pretreatment
    for i in range(Nfiles):
        fI = filesInfo[i]
        fileName = fI['fileName']
        filePath = os.path.join(mainDir, fileName)
        FPS = fI['FPS']
        MagX, MagY, MagR = fI['MagX'], fI['MagY'], fI['MagR']
        CropX, CropY = fI['CropX'], fI['CropY']
        all_tracks = importTrackMateTracks(filePath)
        tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                MagX, MagY, MagR, Rb, CropX, CropY,
                                D2F_func)
    # 2. Run analysis
    tracks_analysis(tracks_data, Rb, expLabel, saveResults, savePlots, saveDir)


        


# %% 5. Run an analysis

# %%% Empty template 

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


