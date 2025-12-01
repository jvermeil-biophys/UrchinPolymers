# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 13:54:28 2025

@author: Joseph
"""

# %% 1. Imports

import numpy as np
import pandas as pd
import matplotlib as mpl
import statsmodels.api as sm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

import os
import json
import colorsys

from scipy import interpolate, optimize


# %% 2. Subfunctions

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


# %% 3. Main functions


# %%% Analyse tracks

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


def tracks_pretreatment(all_tracks, SCALE, FPS, 
                        MagX, MagY, MagR, Rb, visco,
                        CropXY = [0, 0]):
    tracks_data = []
    MagX *= SCALE
    MagY *= SCALE
    MagR *= SCALE
    CropXY = np.array(CropXY)*SCALE
    all_tracks = cleanAllRawTracks(all_tracks)

    for i, track in enumerate(all_tracks):
        #### Conversion in um and sec
        T = track[:, 0] * (1/FPS)
        X = track[:, 1] * SCALE
        Y = track[:, 2] * SCALE
                
        tracks_data.append({'T':T, 'Xraw':X, 'Yraw':Y})
        
        #### Origin as the magnet center
        X2, Y2 = (X+CropXY[0])-MagX, MagY-(Y+CropXY[1])
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
        F = 6 * np.pi * visco*1e-3 * Rb*1e-6 * V*1e-6 * 1e12 # pN
        #
        medD, medV, medF = np.median(D), np.median(V), np.median(F)
        tracks_data[i].update({'D':D, 'V':V, 'F':F,
                               'medD':medD, 'medV':medV, 'medF':medF
                               })
    return(tracks_data)



def tracks_analysis(tracks_data, expLabel = '', 
                    saveResults = True, savePlots = True, saveDir = '.',
                    return_fig = 0):
    
    MagR = 60 # µm - Diamètre typique
    
    #### 1.1 First filter
    tracks_data_f1 = []
    for track in tracks_data:
        crit1 = (np.abs(track['delta']*180/np.pi) < 25)
        crit2 = (np.abs(track['r2_fit'] > 0.80))
        bypass1 = (np.min(track['X']) < 300)
        if (crit1 and crit2) or bypass1:
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
    D_plot = np.linspace(1, 5000, 500)

    #### 3.2 Velocity
    # Double Expo
    V_popt_2exp, V_pcov_2exp = optimize.curve_fit(doubleExpo, all_D, all_V, 
                           p0 = [1000, 50, 100, 1000], 
                           bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
    V_fit_2exp = doubleExpo(D_plot, *V_popt_2exp)
    V_label_2exp = r'$\bf{A \cdot exp(-x/k_1) + B \cdot exp(-x/k_2)}$' + '\n'
    V_label_2exp += '$A$ = {:.2e} | $k_1$ = {:.2f}\n$B$ = {:.2f} | $k_2$ = {:.2e}'.format(*V_popt_2exp)

    # Power Law
    V_popt_pL, V_pcov_pL = optimize.curve_fit(powerLaw, all_D, all_V, 
                           p0 = [1000, -2], 
                           bounds=([0, -10], [np.inf, 0]))
    V_fit_pL = powerLaw(D_plot, *V_popt_pL)
    V_label_pL = r'$\bf{A \cdot x^k}$' + '\n'
    V_label_pL += '$A$ = {:.2e} | $k$ = {:.2f}'.format(*V_popt_pL)

    #### 3.3 Force
    # Double Expo
    F_popt_2exp, F_pcov_2exp = optimize.curve_fit(doubleExpo, all_D, all_F, 
                           p0 = [1000, 50, 100, 1000], 
                           bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
    F_fit_2exp = doubleExpo(D_plot, *F_popt_2exp)
    F_label_2exp = r'$\bf{A \cdot exp(-x/k_1) + B \cdot exp(-x/k_2)}$' + '\n'
    F_label_2exp += '$A$ = {:.2e} | $k_1$ = {:.2f}\n$B$ = {:.2f} | $k_2$ = {:.2e}'.format(*F_popt_2exp)

    # Power Law
    F_popt_pL, F_pcov_pL = optimize.curve_fit(powerLaw, all_D, all_F, 
                           p0 = [1000, -2], 
                           bounds=([0, -10], [np.inf, 0]))
    F_fit_pL = powerLaw(D_plot, *F_popt_pL)
    F_label_pL = r'$\bf{A \cdot x^k}$' + '\n'
    F_label_pL += '$A$ = {:.2e} | $k$ = {:.2f}'.format(*F_popt_pL)


    #### 3.4 Plot the clean fits
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    fig = fig2
    color_V = colorListMpl[0]
    color_F = colorListSns[0]
    ax = axes2[0,0]
    ax.plot(all_D, all_V, ls='', marker='.', alpha=0.01, c = color_V)
    ax.plot(D_plot, V_fit_2exp, 'r-', label = label_2exp)
    ax.plot(D_plot, V_fit_pL, ls='-', color='darkorange', label = V_label_pL)
    # ax.plot(D_plot, V_fit_pLS, 'c-', label = V_label_pLS)
    ax.grid()
    ax.set_xlim([0, 1.1 * max(all_D)])
    ax.set_ylim([0, 1.2 * max(all_V)])
    ax.set_xlabel('d [µm]')
    ax.set_ylabel('v [µm/s]')
    # ax.legend()

    ax = axes2[0,1]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(all_D, all_V, ls='', marker='.', alpha=0.01, c = color_V)
    ax.plot(D_plot, V_fit_2exp, 'r-', label = V_label_2exp)
    ax.plot(D_plot, V_fit_pL, ls='-', color='darkorange', label = V_label_pL)
    # ax.plot(D_plot, V_fit_pLS, 'c-', label = V_label_pLS)
    ax.grid()
    ax.set_xlim([50, 5000])
    ax.set_ylim([0.01, 100])
    ax.set_xlabel('d [µm]')
    ax.set_ylabel('v [µm/s]')
    ax.legend(title='Fit on V(d)',
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    ax = axes2[1,0]
    ax.plot(all_D, all_F, ls='', marker='.', alpha=0.01, c = color_F)
    ax.plot(D_plot, F_fit_2exp, 'r-', label = F_label_2exp)
    ax.plot(D_plot, F_fit_pL, ls='-', color='darkorange', label = F_label_pL)
    # ax.plot(D_plot, V_fit_pLS, 'c-', label = V_label_pLS)
    ax.grid()
    ax.set_xlim([0, 1.1 * max(all_D)])
    ax.set_ylim([0, 1.2 * max(all_F)])
    ax.set_xlabel('d [µm]')
    ax.set_ylabel('F [pN]')
    # ax.legend()

    ax = axes2[1,1]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(all_D, all_F, ls='', marker='.', alpha=0.01, c = color_F)
    ax.plot(D_plot, F_fit_2exp, 'r-', label = F_label_2exp)
    ax.plot(D_plot, F_fit_pL, ls='-', color='darkorange', label = F_label_pL)
    # ax.plot(D_plot, V_fit_pLS, 'c-', label = V_label_pLS)
    ax.grid()
    ax.set_xlim([50, 5000])
    ax.set_ylim([0.01, 100])
    ax.set_xlabel('d [µm]')
    ax.set_ylabel('F [pN]')
    ax.legend(title='Fit on F(d)',
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    MainTitle = 'Calibration Data'
    if expLabel != '':
        MainTitle = MainTitle + ' - ' + expLabel
    fig.suptitle(MainTitle)
    fig.tight_layout()
    
    plt.show()
    
    if savePlots:
        fig1.savefig(os.path.join(saveDir, expLabel + '_Traj.png'), dpi=400)
        fig2.savefig(os.path.join(saveDir, expLabel + '_Fits.png'), dpi=400)
        
    if saveResults:
        dictResults = {'V_popt_2exp':V_popt_2exp,
                       'V_popt_pL':V_popt_pL,
                       'F_popt_2exp':F_popt_2exp,
                       'F_popt_pL':F_popt_pL,
                       'all_D':all_D,
                       'all_V':all_V,
                       'all_F':all_F,
                       }
        listOfdict2json(tracks_data_f2, saveDir, expLabel+'_allTracksData')
        dict2json(dictResults, saveDir, expLabel+'_fitData')     
        
    if return_fig == 1:
        return(fig1, axes1)
    elif return_fig == 2:
        return(fig2, axes2)



# %%% Compare analysis


def examineCalibration(srcDir, labelList = [], 
                       savePlots = True, saveDir = '.',
                       return_fig = 0):
    dataList = []
    supTitle = ''
    saveTitle = ''
    for lab in labelList:
        fitData = json2dict(srcDir, lab + '_fitData')
        dataList.append(fitData)
        supTitle += (lab + ' vs. ')
        saveTitle += (lab + '-v-')
    supTitle = supTitle[:-5]
    saveTitle = saveTitle[:-3]
    
    #### Initialize the plots
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 8))
    fig = fig1
    
    for i, data in enumerate(dataList):
        lab = labelList[i]
        color_V = colorListMpl[i]
        color_F = colorListMpl[i+len(dataList)]
        color_fitV = lighten_color(color_V, factor=0.6)
        color_fitF = lighten_color(color_F, factor=0.6)
        D_plot = np.linspace(1, 5000, 500)
        all_D, all_V, all_F = data['all_D'], data['all_V'], data['all_F']
        
        #### Velocity
        # Double Expo
        V_popt_2exp = data['V_popt_2exp']
        V_fit_2exp = doubleExpo(D_plot, *V_popt_2exp)
        V_label_2exp = r'$\it{V\ fit\ double\ expo}$'
        # V_label_2exp = r'$\bf{A \cdot exp(-x/k_1) + B \cdot exp(-x/k_2)}$' + '\n'
        V_label_2exp += '\n' + '$A$ = {:.2e} | $k_1$ = {:.2f}\n$B$ = {:.2f} | $k_2$ = {:.2e}'.format(*V_popt_2exp)

        # Power Law
        V_popt_pL = data['V_popt_pL']
        V_fit_pL = powerLaw(D_plot, *V_popt_pL)
        V_label_pL = r'$\it{V\ fit\ power\ law}$'
        # V_label_pL = r'$\bf{A \cdot x^k}$' + '\n'
        V_label_pL += '\n' + '$A$ = {:.2e} | $k$ = {:.2f}'.format(*V_popt_pL)

        #### Force
        # Double Expo
        F_popt_2exp = data['F_popt_2exp']
        F_fit_2exp = doubleExpo(D_plot, *F_popt_2exp)
        F_label_2exp = r'$\it{F\ fit\ double\ expo}$'
        # F_label_2exp = r'$\bf{A \cdot exp(-x/k_1) + B \cdot exp(-x/k_2)}$' + '\n'
        F_label_2exp += '\n' + '$A$ = {:.2e} | $k_1$ = {:.2f}\n$B$ = {:.2f} | $k_2$ = {:.2e}'.format(*F_popt_2exp)

        # Power Law
        F_popt_pL = data['F_popt_pL']
        F_fit_pL = powerLaw(D_plot, *F_popt_pL)
        F_label_pL = r'$\it{F\ fit\ power\ law}$'
        # F_label_pL = r'$\bf{A \cdot x^k}$' + '\n'
        F_label_pL += '\n' + '$A$ = {:.2e} | $k$ = {:.2f}'.format(*F_popt_pL)
        
        alpha = min(1, max(0.01, 800/len(all_D)))
    
        ax = axes1[0,0]
        ax.plot(all_D, all_V, ls='', marker='.', alpha=alpha, c = color_V)
        ax.plot(D_plot, V_fit_2exp, ls='--', color=color_fitV, label = V_label_2exp)
        ax.plot(D_plot, V_fit_pL, ls=':', color=color_fitV, label = V_label_pL)
        ax.set_xlim([0, 1.1 * max(all_D)])
        ax.set_ylim([0, 1.2 * max(all_V)])
        ax.set_xlabel('d [µm]')
        ax.set_ylabel('v [µm/s]')
    
        ax = axes1[0,1]
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(all_D, all_V, ls='', marker='.', alpha=alpha, c = color_V)
        ax.plot([], [], ls='', marker='.', alpha=1, c = color_V, label =lab)
        ax.plot(D_plot, V_fit_2exp, ls='--', color=color_fitV, label = V_label_2exp)
        ax.plot(D_plot, V_fit_pL, ls=':', color=color_fitV, label = V_label_pL)
        ax.set_xlim([50, 5000])
        ax.set_ylim([0.01, 100])
        ax.set_xlabel('d [µm]')
        ax.set_ylabel('v [µm/s]')
    
        ax = axes1[1,0]
        ax.plot(all_D, all_F, ls='', marker='.', alpha=alpha, c = color_F)
        ax.plot(D_plot, F_fit_2exp, ls='--', color=color_fitF, label = F_label_2exp)
        ax.plot(D_plot, F_fit_pL, ls=':', color=color_fitF, label = F_label_pL)        
        ax.set_xlim([0, 1.1 * max(all_D)])
        ax.set_ylim([0, 1.2 * max(all_F)])
        ax.set_xlabel('d [µm]')
        ax.set_ylabel('F [pN]')
        
    
        ax = axes1[1,1]
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(all_D, all_F, ls='', marker='.', alpha=alpha, c = color_F)
        ax.plot([], [], ls='', marker='.', alpha=1, c = color_F, label = lab)
        ax.plot(D_plot, F_fit_2exp, ls='--', color=color_fitF, label = F_label_2exp)
        ax.plot(D_plot, F_fit_pL, ls=':', color=color_fitF, label = F_label_pL)
        ax.set_xlim([50, 5000])
        ax.set_ylim([0.01, 100])
        ax.set_xlabel('d [µm]')
        ax.set_ylabel('F [pN]')
        
        
    titles = ['Velocities', 'Forces']
    for ax, title in zip(axes1[:,1], titles):    
        ax.legend(title=title,
                  title_fontproperties={'weight':'bold'},
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
        
    for ax in axes1.flatten():
        ax.grid(axis='both')

    MainTitle = 'Calibration Data - ' + supTitle
    fig.suptitle(MainTitle)
    fig.tight_layout()
    
    plt.show()
    
    if savePlots:
        fig1.savefig(os.path.join(saveDir, 'Compare_' + saveTitle + '.png'), dpi=500)
        
    if return_fig == 1:
        return(fig1, axes1)
    


# %% 4. Run the analysis

# %%% 4.1 - Example : MyOne - Gly75%

# %%%% (i) Define directories & constant

setGraphicOptions(mode = 'screen_big', colorList = colorListMpl)

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/'
mainDir += '25-11_DynabeadsInCapillaries_CalibrationsTests/Tracks'

# saveDir is the directory where the data and the plots will be saved
saveDir = 'C:/Users/josep/Desktop/Seafile/AnalysisPulls/25-11_DynabeadsInCapillaries_CalibrationsTests/'

# Beads - MyOne
Rb = 1 * 0.5
# Medium - 75% Gly at 20.6°C
visco = 53.3 # mPa.s
# Microscope
SCALE = 0.451
FPS = 5


# %%%% (ii) Import & apply pretreatment on several files

# Initialize the list of all tracks data
tracks_data = [] # Do not modify this line

#### Film 1

fileName = '25-11-19_Capi04_FilmBF_5fps_1_CropInv_Tracks.xml'
filePath = os.path.join(mainDir, fileName)
# Magnet
MagX, MagY = 154, 497
MagR = 234 * 0.5 
# Crop
# CropX and CropY are useful if you perfomed tracking on a croped image, where the magnet was no longer visible
# If you did not, leave them at 0
# If you did, they are equal to the top left corner coordinates (in the original image) of your cropping rectangle
CropX, CropY = 790, 0 


all_tracks = importTrackMateTracks(filePath)
tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                   MagX, MagY, MagR, Rb, visco,
                                   CropXY = [CropX, CropY])

#### Film 2
fileName = '25-11-19_Capi04_FilmBF_5fps_2_CropInv_Tracks.xml'
filePath = os.path.join(mainDir, fileName)
# Magnet
MagX, MagY = 140 
MagY = 551 
MagR = 232 * 0.5 
# Crop
CropX, CropY = 715, 1

all_tracks = importTrackMateTracks(filePath)
tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                   MagX, MagY, MagR, Rb, visco,
                                   CropXY = [CropX, CropY])

#### Film 4
fileName = '25-11-19_Capi04_FilmBF_5fps_4_CropInv_Tracks.xml'
filePath = os.path.join(mainDir, fileName)
# Magnet
MagX, MagY = 149, 610
MagR = 238 * 0.5 
# Crop
CropX = 723, 0

all_tracks = importTrackMateTracks(filePath)
tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                   MagX, MagY, MagR, Rb, visco,
                                   CropXY = [CropX, CropY])

# NB: now tracks_data contains all the data from the 3 files

# %%%% (iii) Run the analysis, make plots and save the data

# NB: the parameter 'expLabel' will be used as a prefix for all the files saved by this function

tracks_analysis(tracks_data, expLabel = 'MyOne_Glycerol75%', 
                saveResults = True, savePlots = True, saveDir = saveDir)


# %%% ---------------------



# %%% 4.2 - Run your own calibration !

# %%%% (i) Define directories & constant

setGraphicOptions(mode = 'screen_big', colorList = colorListMpl)

# mainDir is the directory containing the track files (.xml from TrackMate)
mainDir = ''

# saveDir is the directory where the data and the plots will be saved
saveDir = ''

# Beads - MyOne
Rb = 1 * 0.5
# Medium - 75% Gly at 20.6°C
visco = 53.3 # mPa.s
# Microscope
SCALE = 0.451
FPS = 5


# %%%% (ii) Import & apply pretreatment on several files

# Initialize the list of all tracks data
tracks_data = [] # Do not modify this line

#### Film 1

fileName = ''
filePath = os.path.join(mainDir, fileName)
# Magnet
MagX, MagY = 0, 0
MagR = 0 * 0.5 
# Crop
CropX, CropY = 0, 0 

all_tracks = importTrackMateTracks(filePath)
tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                   MagX, MagY, MagR, Rb, visco,
                                   CropXY = [CropX, CropY])

#### Film 2
fileName = ''
filePath = os.path.join(mainDir, fileName)
# Magnet
MagX, MagY = 0 
MagY = 0 
MagR = 0 * 0.5 
# Crop
CropX, CropY = 0, 0

all_tracks = importTrackMateTracks(filePath)
tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                   MagX, MagY, MagR, Rb, visco,
                                   CropXY = [CropX, CropY])


# NB: now tracks_data contains all the data from the N files

# %%%% (iii) Run the analysis, make plots and save the data

# NB: the parameter 'expLabel' will be used as a prefix for all the files saved by this function

tracks_analysis(tracks_data, expLabel = 'my_label', 
                saveResults = True, savePlots = True, saveDir = saveDir)


# %%% ---------------------








