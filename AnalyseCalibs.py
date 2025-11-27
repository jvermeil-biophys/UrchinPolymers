# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 15:13:32 2025

@author: Utilisateur
"""

# %% 1. Imports

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import statsmodels.api as sm
import matplotlib.pyplot as plt

import os
import re
import json
import random
import matplotlib


from scipy import interpolate, optimize, signal
from matplotlib.gridspec import GridSpec


#### Local Imports

import PlotMaker as pm
import UtilityFunctions as ufun



# %% 2. Subfunctions

# %%% Fitting functions

def doubleExpo(x, A, k1, B, k2):
    return(A*np.exp(-x/k1) + B*np.exp(-x/k2))

def powerLaw(x, A, k):
    return(A*(x**k))

def powerLawShifted(x, A, k, x0):
    return(A*((x-x0)**k))


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


# %%% Analyse tracks - without flow

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

# def ApplyFlowCorrection(tracks_data, df_flowCorr):
#     df = df_flowCorr
#     a = np.polyfit(np.arange(len(df)), df['x'].values)
#     print(a)


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
                               'medX2':medX2, 'medY2':medY2})
        #### Rotate the trajectory by its own angle
        parms, res, wlm_res = ufun.fitLineHuber(X2, Y2, with_wlm_results=True)
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
                    flowCorrection = False, flowCorrectionPath = '',
                    saveResults = True, savePlots = True, saveDir = '.'):
    
    MagR = 150/2
    
    #### First filter
    tracks_data_f1 = []
    for track in tracks_data:
        crit1 = (np.abs(track['delta']*180/np.pi) < 25)
        crit2 = (np.abs(track['r2_fit'] > 0.80))
        bypass1 = (np.min(track['X']) < 300)
        if (crit1 and crit2) or bypass1:
            tracks_data_f1.append(track)
        
    fig1, axes1 = plt.subplots(1, 3, figsize = (24, 8))
    fig = fig1
    #### Plot all trajectories
    ax = axes1[0]
    ax.set_title(f'All tracks, N = {len(tracks_data)}')
    for track in tracks_data:
        X, Y = track['X'], track['Y']
        ax.plot(X, Y)
    
    #### Plot first filter
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

    #### Second Filter
    all_medD = np.array([track['medD'] for track in tracks_data_f1])
    all_medV = np.array([track['medV'] for track in tracks_data_f1])
    all_D = np.concat([track['D'] for track in tracks_data_f1])
    all_V = np.concat([track['V'] for track in tracks_data_f1])
    
    # fig2, ax2 = plt.subplots(1, 1, figsize = (8, 6))
    # fig, ax = fig2, ax2
    # # ax.plot(all_medD, all_medV, ls='', marker='.')
    # ax.plot(all_D, all_V, ls='', marker='.', alpha=0.05)
    # ax.grid()
    # ax.set_xlabel('D [µm]')
    # ax.set_ylabel('V [µm/s]')
    
    
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

    # fig21, ax21 = plt.subplots(1,1)
    # fig, ax = fig21, ax21
    # ax.plot(all_medD, ratio_fitV, ls='', marker='.', alpha=0.3)
    # ax.axhline(high_cut, color = 'k', ls='-.')
    # ax.axhline(low_cut, color = 'k', ls='-.')
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
    
    #### Plot second filter
    # fig22, ax22 = plt.subplots(1, 1, figsize = (8, 6))
    # fig, ax = fig22, ax22
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
            
    
    #### In case of flow correction
    if flowCorrection:
        df_fC = pd.read_csv(flowCorrectionPath, header = 3, sep='\t', #skiprows=2,
                            on_bad_lines='skip', encoding='utf_8',
                            names=['x', 'y', 'u', 'v', 'vector_type']) # 'utf_16_le'
        # Original column names
        # x [px]	y [px]	u [px/frame]	v [px/frame]	Vector type [-]
        
        tracks_data_f3 = ApplyFlowCorrection(tracks_data_f2, df_fC)
    
    
        
    
    
    
    
    #### Final fits
    D_plot = np.linspace(1, 5000, 500)

    #### Velocity
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

    #### Force
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


    #### Plot the clean fits
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    fig = fig2
    color_V = pm.colorList10[0]
    color_F = pm.cL_Set21[0]
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
        # json2listOfdict(path, fileName)



# %%% Compare analysis

def examineCalibration(srcDir, labelList = [], 
                       savePlots = True, saveDir = '.'):
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
        color_V = pm.colorList40[10+i]
        color_F = pm.colorList40[10+i+len(dataList)]
        color_fitV = pm.lighten_color(color_V, factor=0.6)
        color_fitF = pm.lighten_color(color_F, factor=0.6)
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
    
        ax = axes1[0,0]
        ax.plot(all_D, all_V, ls='', marker='.', alpha=0.01, c = color_V)
        ax.plot(D_plot, V_fit_2exp, ls='--', color=color_fitV, label = V_label_2exp)
        ax.plot(D_plot, V_fit_pL, ls=':', color=color_fitV, label = V_label_pL)
        ax.set_xlim([0, 1.1 * max(all_D)])
        ax.set_ylim([0, 1.2 * max(all_V)])
        ax.set_xlabel('d [µm]')
        ax.set_ylabel('v [µm/s]')
    
        ax = axes1[0,1]
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(all_D, all_V, ls='', marker='.', alpha=0.01, c = color_V)
        ax.plot([], [], ls='', marker='.', alpha=1, c = color_V, label =lab)
        ax.plot(D_plot, V_fit_2exp, ls='--', color=color_fitV, label = V_label_2exp)
        ax.plot(D_plot, V_fit_pL, ls=':', color=color_fitV, label = V_label_pL)
        # ax.plot([], [], ls='', marker='.', alpha=1, c = 'w', label = '\n')
        ax.set_xlim([50, 5000])
        ax.set_ylim([0.01, 100])
        ax.set_xlabel('d [µm]')
        ax.set_ylabel('v [µm/s]')
        # ax.legend(title='Fit on V(d)',
        #           loc="center left",
        #           bbox_to_anchor=(1, 0, 0.5, 1))
    
        ax = axes1[1,0]
        ax.plot(all_D, all_F, ls='', marker='.', alpha=0.01, c = color_F)
        ax.plot(D_plot, F_fit_2exp, ls='--', color=color_fitF, label = F_label_2exp)
        ax.plot(D_plot, F_fit_pL, ls=':', color=color_fitF, label = F_label_pL)        
        ax.set_xlim([0, 1.1 * max(all_D)])
        ax.set_ylim([0, 1.2 * max(all_F)])
        ax.set_xlabel('d [µm]')
        ax.set_ylabel('F [pN]')
        
    
        ax = axes1[1,1]
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(all_D, all_F, ls='', marker='.', alpha=0.01, c = color_F)
        ax.plot([], [], ls='', marker='.', alpha=1, c = color_F, label = lab)
        ax.plot(D_plot, F_fit_2exp, ls='--', color=color_fitF, label = F_label_2exp)
        ax.plot(D_plot, F_fit_pL, ls=':', color=color_fitF, label = F_label_pL)
        # ax.plot([], [], ls='', marker='.', alpha=1, c = 'w', label = '\n')
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
    

# %%% Analyse tracers


def tracerTracks_pretreatment(all_tracks, SCALE, FPS, 
                        MagX, MagY, MagR, 
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
                               'medX2':medX2, 'medY2':medY2})
        
        # #### Rotate the trajectory by its own angle
        # parms, res, wlm_res = ufun.fitLineHuber(X2, Y2, with_wlm_results=True)
        # b_fit, a_fit = parms
        # r2 = wlm_res.rsquared
        # theta = np.atan(a_fit)

        # tracks_data[i].update({'a_fit':a_fit, 'b_fit':b_fit, 'r2_fit':r2,
        #                        'theta':theta,
        #                        })
        
        # #### Rotate the trajectory by its angle with the magnet
        # phi = np.atan(medY2/medX2)
        # delta = theta-phi # delta is the angle between the traj fit & strait line to the magnet
        # tracks_data[i].update({'phi':phi, 'delta':delta,
        #                        })
        
        #### Compute distance & velocity analogue to tracks
        D = np.array(((X2**2 + Y2**2)**0.5) - MagR)
        
        #### Smooth D with splines and compute velocity V
        # Make spline
        spline_D = interpolate.make_splrep(T, D, s=6)
        spline_V2mag = spline_D.derivative(nu=1)
        # Compute V and chose definition
        V2mag = np.abs(spline_V2mag(T))

        #
        medD, medV2mag = np.median(D), np.median(V2mag)
        tracks_data[i].update({'D':D, 'V2mag':V2mag,
                               'medD':medD, 'medV2mag':medV2mag,
                               })
        
        #### Savitsky-Golay - smooth & derive velocities in X and Y
        window_length = len(T)//2
        polyorder = 3
        X3 = signal.savgol_filter(X2, window_length, polyorder, deriv=0, delta=(1/FPS), mode='interp')
        Y3 = signal.savgol_filter(Y2, window_length, polyorder, deriv=0, delta=(1/FPS), mode='interp')
        U3 = signal.savgol_filter(X2, window_length, polyorder, deriv=1, delta=(1/FPS), mode='interp')
        V3 = signal.savgol_filter(Y2, window_length, polyorder, deriv=1, delta=(1/FPS), mode='interp')
        tracks_data[i].update({'X3':X3, 'Y3':Y3,
                               'U3':U3, 'V3':V3,
                               })
        
        #### Spline - smooth & derive velocities in X and Y
        # splineXY, u = interpolate.make_splprep([X2, Y2], s=50)
        # X3, Y3 = splineXY(np.linspace(0, 1, len(T)))
        s = len(T)*2
        splineX = interpolate.make_splrep(T, X2, s=s)
        splineY = interpolate.make_splrep(T, Y2, s=s)
        splineU = interpolate.make_splrep(T, X2, s=s).derivative(nu=1)
        splineV = interpolate.make_splrep(T, Y2, s=s).derivative(nu=1)
        X4, Y4 = splineX(T), splineY(T)
        U4, V4 = splineU(T), splineV(T)
        tracks_data[i].update({'X4':X4, 'Y4':Y4,
                               'U4':U4, 'V4':V4,
                               })
        
                
    return(tracks_data)


def tracerTracks_analysis(tracks_data, df_flowCorr, 
                          SCALE, FPS, 
                          MagX, MagY, MagR):

    #### Manage the flow correction data
    df_fC = df_flowCorr
    
    df_fC['x2'] = (df_fC['x']-MagX) * SCALE
    df_fC['y2'] = (MagY-df_fC['y']) * SCALE
    df_fC['u2'] = df_fC['u'] * (FPS*SCALE)
    df_fC['v2'] = df_fC['v'] * ((-1)*FPS*SCALE)
    
    X_fC = df_fC['x2'].unique()
    step_fC = np.polyfit(np.arange(len(X_fC)), X_fC, 1)[0]
    df_fC['xi'] = df_fC['x2']//step_fC
    df_fC['yi'] = df_fC['y2']//step_fC
    list_xi_fC = df_fC['xi'].unique().astype(int)
    list_yi_fC = df_fC['yi'].unique().astype(int)
    print(list_xi_fC)
    print(list_yi_fC)

    #### First plots
    fig1, axes1 = plt.subplots(1, 3, figsize = (24, 8))
    fig = fig1
    for ax in axes1:
        circle1 = plt.Circle((0, 0), MagR*SCALE, color='dimgrey')
        ax.add_patch(circle1)
        # ax.axvspan(wall_L, wall_R, color='lightgray', zorder=0)
        ax.set_xlim([0, 800])
        ax.set_ylim([-400, +400])
        ax.grid()
        ax.axis('equal')
        ax.set_xlabel('X [µm]')
        ax.set_ylabel('Y [µm]')
    
    #### Plot raw trajectories
    ax = axes1[0]
    ax.set_title(f'All tracks, N = {len(tracks_data)}')
    for track in tracks_data:
        X, Y = track['X'], track['Y']
        ax.plot(X, Y)
        
    #### Plot smoothed trajectories
    ax = axes1[1]
    ax.set_title('Smooth tracks, Savitsky-Golay')
    for track in tracks_data:
        X, Y = track['X3'], track['Y3']
        ax.plot(X, Y)
        
    #### Plot smoothed trajectories with vectors from PIV
    ax = axes1[2]
    ax.set_title('Smooth tracks & vectors from PIV')
    for track in tracks_data:
        X, Y = track['X3'], track['Y3']
        ax.plot(X, Y, lw=1.0)
    ax.quiver(df_fC['x2'], df_fC['y2'], df_fC['u2'], df_fC['v2'], zorder=6)
   
        
    plt.show()
    
    #### Plot the different definition of the speed
    # fig2, axes2 = plt.subplots(2, 2, figsize = (12, 12), sharey=True)
    # fig = fig2
    # for i, var in enumerate(['U3', 'V3', 'U4', 'V4']):
    #     ax = axes2.flatten()[i]
    #     ax.set_title(var + '(t)')
    #     for track in tracks_data[:10]:
    #         T, v = track['T'], track[var]
    #         ax.plot(T, v)
    
    #### Examine only a specific windows
    ROIs = []
    
    height, width = 20, 20
    x1, y1 = 140, -10
    x2, y2 = x1+width, y1+height
    ROI = (x1, y1, x2, y2, height, width)
    ROIs.append(ROI)
    
    height, width = 40, 40
    x1, y1 = 300, 150
    x2, y2 = x1+width, y1+height
    ROI = (x1, y1, x2, y2, height, width)
    ROIs.append(ROI)
    
    height, width = 40, 40
    x1, y1 = 150, -150
    x2, y2 = x1+width, y1+height
    ROI = (x1, y1, x2, y2, height, width)
    ROIs.append(ROI)
    
    height, width = 50, 50
    x1, y1 = -20, 160
    x2, y2 = x1+width, y1+height
    ROI = (x1, y1, x2, y2, height, width)
    ROIs.append(ROI)
    

    fig3 = plt.figure(layout="constrained", figsize = (18, 8))
    fig = fig3

    gs = GridSpec(2, 4, figure=fig)
    ax0 = fig.add_subplot(gs[:, :2])
    # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
    ax1 = fig.add_subplot(gs[0, 2])
    ax2 = fig.add_subplot(gs[0, 3])
    ax3 = fig.add_subplot(gs[1, 2])
    ax4 = fig.add_subplot(gs[1, 3])
    axes3 = [ax0, ax1, ax2, ax3, ax4]
    
    #### Plot trajectories in ROIs
    ax = axes3[0]
    ax.set_xlim([0, 800])
    ax.set_ylim([-400, +400])
    ax.grid()
    ax.axis('equal')
    ax.set_xlabel('X [µm]')
    ax.set_ylabel('Y [µm]')
    circle1 = plt.Circle((0, 0), MagR*SCALE, color='dimgrey')
    ax.add_patch(circle1)
    
    for i, ROI in enumerate(ROIs):
        color = pm.cL_Set12[i]
        
        ax = axes3[0]
        (x1, y1, x2, y2, height, width) = ROI
        rect = plt.Rectangle(xy=(x1, y1), width=width, height=height, 
                             facecolor='None', edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        ax = axes3[i+1]
        ax.grid()
        ax.plot([], [], 'ko', label = r'$V_x$')
        ax.plot([], [], 'k^', label = r'$V_y$')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
            
        xm, ym = (x1+x2)*0.5, (y1+y2)*0.5
        xm_i, ym_i = (xm+step_fC/2)//step_fC, (ym+step_fC/2)//step_fC
        print(int(xm_i), int(ym_i))
        
        valid_ROI = (xm_i in list_xi_fC) and (ym_i in list_yi_fC) 
        if valid_ROI:
            print(f'ROI {i+1} ok !')
            row_loc = (df_fC['xi']==xm_i) & (df_fC['yi']==ym_i)
            U_fC = float(df_fC.loc[row_loc, 'u2'].values[0])
            V_fC = float(df_fC.loc[row_loc, 'v2'].values[0])
            
            
            ax.axhline(U_fC, ls='--', label=r'$V_x$ from PIV')
            ax.axhline(V_fC, ls=':', label=r'$V_y$ from PIV')
        # print(VV)
    
    
    for track in tracks_data:
        X, Y = track['X3'], track['Y3']
        ax = axes3[0]
        ax.plot(X, Y, alpha = 0.3)
        
        for i, ROI in enumerate(ROIs):               
            (x1, y1, x2, y2, height, width) = ROI
            filterX = (x1 <= X) & (X <= x2)
            filterY = (y1 <= Y) & (Y <= y2)
            filterXY = filterX & filterY
            
            Xf, Yf = X[filterXY], Y[filterXY]
            Tf, Uf, Vf = track['T'][filterXY], track['U3'][filterXY], track['V3'][filterXY]
            
            if len(Tf) > 0:
                ax = axes3[0]
                ax.plot(Xf, Yf)
                
                ax = axes3[i+1]
                ax.plot(np.median(Tf), np.median(Uf), ls='', marker='o')
                ax.plot(np.median(Tf), np.median(Vf), ls='', marker='^')
                
            
            
                
    for ax in [axes3[2], axes3[4]]:
        ax.legend(loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))

                

        
    #### Plot smoothed trajectories
    # ax = axes3[1]
    # ax.set_title(f'Smooth tracks, Savitsky-Golay')
    # for track in tracks_data:
    #     X, Y = track['X3'], track['Y3']
    #     ax.plot(X, Y)

        
    plt.show()
    
    return(df_fC)
    



# mainDir = 'E:/WorkingData/LeicaData/25-11-19/25-11-19_Droplet01_Gly80p_MyOne'
# mainDir += '/25-11-19_Droplet01_20x_FilmFluo_3spf_3/'

mainDir = 'C:/Users/josep/Desktop/Seafile/AnalysisPulls/'
mainDir += '25-11-19_Droplet01_Gly80p_MyOne/'
mainDir += '25-11-19_Droplet01_20x_FilmFluo_3spf_3/'

fileName = '20x_FilmFluo_3spf_3_CropTracers_MinusMED_Tracks.xml'
filePath = os.path.join(mainDir, fileName)

SCALE = 0.451
FPS = (1/3)

# Magnet
MagX = 229
MagY = 702
MagR = 250 * 0.5
# Crop
CropX = 0
CropY = 0

all_tracks = ufun.importTrackMateTracks(filePath)
tracks_data = tracerTracks_pretreatment(all_tracks, SCALE, FPS, 
                                   MagX, MagY, MagR,
                                   CropXY = [CropX, CropY])

# flowCorrectionPath = 'E:\WorkingData/LeicaData/25-11-19/25-11-19_Droplet01_Gly80p_MyOne/'
# flowCorrectionPath += '25-11-19_Droplet01_20x_FilmFluo_3spf_3/Res01_PIVlab.txt'

flowCorrectionFile = 'Res01_PIVlab.txt'
flowCorrectionPath = os.path.join(mainDir, flowCorrectionFile)

df_fC = pd.read_csv(flowCorrectionPath, header = 3, sep='\t', #skiprows=2,
                    on_bad_lines='skip', encoding='utf_8',
                    names=['x', 'y', 'u', 'v', 'vector_type']).dropna(subset=['u']) # 'utf_16_le'

# df_fC = df_flowCorr
# X_fC = df_fC['x'].unique()
# step_fC = np.polyfit(np.arange(len(X_fC)), X_fC, 1)[0]
# df_fC['xi'] = df_fC['x']//step_fC
# df_fC['yi'] = df_fC['y']//step_fC
# list_xi_fC = df_fC['xi'].unique().astype(int)
# list_yi_fC = df_fC['yi'].unique().astype(int)

df_fC2 = tracerTracks_analysis(tracks_data, df_fC, 
                      SCALE, FPS, 
                      MagX, MagY, MagR)


# %%% Analyse tracks - with flow


def tracks_pretreatment_wFlow(all_tracks, SCALE, FPS, 
                        MagX, MagY, MagR, Rb, visco,
                        CropXY = [0, 0]):
    tracks_data = []
    MagX *= SCALE
    MagY *= SCALE
    MagR *= SCALE
    CropXY = np.array(CropXY)*SCALE
    # all_tracks = cleanAllRawTracks(all_tracks)
    all_tracks = [track for track in all_tracks if track.shape[0] >= 10]

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
                               'medX2':medX2, 'medY2':medY2})
        #### Rotate the trajectory by its own angle
        parms, res, wlm_res = ufun.fitLineHuber(X2, Y2, with_wlm_results=True)
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



def tracks_analysis_wFlow(tracks_data, df_flowCorr, expLabel = '', 
                    flowCorrection = False, flowCorrectionPath = '',
                    saveResults = True, savePlots = True, saveDir = '.'):
    
    MagR = 150/2
    
    #### Manage the flow correction data
    df_fC = df_flowCorr
    
    df_fC['x2'] = (df_fC['x']-MagX) * SCALE
    df_fC['y2'] = (MagY-df_fC['y']) * SCALE
    df_fC['u2'] = df_fC['u'] * (FPS*SCALE)
    df_fC['v2'] = df_fC['v'] * ((-1)*FPS*SCALE)
    
    X_fC = df_fC['x2'].unique()
    step_fC = np.polyfit(np.arange(len(X_fC)), X_fC, 1)[0]
    df_fC['xi'] = df_fC['x2']//step_fC
    df_fC['yi'] = df_fC['y2']//step_fC
    list_xi_fC = df_fC['xi'].unique().astype(int)
    list_yi_fC = df_fC['yi'].unique().astype(int)
    print(list_xi_fC)
    print(list_yi_fC)
    
    #### First filter
    tracks_data_f1 = []
    for track in tracks_data:
        crit1 = (np.abs(track['delta']*180/np.pi) < 25)
        crit2 = (np.abs(track['r2_fit'] > 0.80))
        bypass1 = (np.min(track['X']) < 300)
        if (crit1 and crit2) or bypass1:
            tracks_data_f1.append(track)
        
    fig1, axes1 = plt.subplots(1, 3, figsize = (24, 8))
    fig = fig1
    #### Plot all trajectories
    ax = axes1[0]
    ax.set_title(f'All tracks, N = {len(tracks_data)}')
    for track in tracks_data:
        X, Y = track['X'], track['Y']
        ax.plot(X, Y)
    
    # #### Plot first filter
    # ax = axes1[1]
    # ax.set_title(f'First filter, N = {len(tracks_data_f1)}')
    # for track in tracks_data_f1:
    #     X, Y = track['X'], track['Y']
    #     ax.plot(X, Y)

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

    # #### Second Filter
    # all_medD = np.array([track['medD'] for track in tracks_data_f1])
    # all_medV = np.array([track['medV'] for track in tracks_data_f1])
    # all_D = np.concat([track['D'] for track in tracks_data_f1])
    # all_V = np.concat([track['V'] for track in tracks_data_f1])
    
    # # fig2, ax2 = plt.subplots(1, 1, figsize = (8, 6))
    # # fig, ax = fig2, ax2
    # # # ax.plot(all_medD, all_medV, ls='', marker='.')
    # # ax.plot(all_D, all_V, ls='', marker='.', alpha=0.05)
    # # ax.grid()
    # # ax.set_xlabel('D [µm]')
    # # ax.set_ylabel('V [µm/s]')
    
    
    # D_plot = np.linspace(1, 5000, 500)

    # # Double Expo
    # V_popt_2exp, V_pcov_2exp = optimize.curve_fit(doubleExpo, all_D, all_V, 
    #                        p0 = [1000, 50, 100, 1000], 
    #                        bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
    # V_fit_2exp = doubleExpo(D_plot, *V_popt_2exp)
    # label_2exp = r'$\bf{A \cdot exp(-x/k_1) + B \cdot exp(-x/k_2)}$' + '\n'
    # label_2exp += '$A$ = {:.2e} | $k_1$ = {:.2f}\n$B$ = {:.2f} | $k_2$ = {:.2e}'.format(*V_popt_2exp)

    # # Power Law
    # V_popt_pL, V_pcov_pL = optimize.curve_fit(powerLaw, all_D, all_V, 
    #                        p0 = [1000, -2], 
    #                        bounds=([0, -10], [np.inf, 0]))
    # V_fit_pL = powerLaw(D_plot, *V_popt_pL)
    # V_label_pL = r'$\bf{A \cdot x^k}$' + '\n'
    # V_label_pL += '$A$ = {:.2e} | $k$ = {:.2f}'.format(*V_popt_pL)

    # expected_medV = powerLaw(all_medD, *V_popt_pL)
    # ratio_fitV = all_medV/expected_medV
    # high_cut = 1.45
    # low_cut = 0.55

    # # fig21, ax21 = plt.subplots(1,1)
    # # fig, ax = fig21, ax21
    # # ax.plot(all_medD, ratio_fitV, ls='', marker='.', alpha=0.3)
    # # ax.axhline(high_cut, color = 'k', ls='-.')
    # # ax.axhline(low_cut, color = 'k', ls='-.')
    # tracks_data_f2 = []
    # removed_tracks = []
    # for i, track in enumerate(tracks_data_f1):
    #     if (ratio_fitV[i] > low_cut) and (ratio_fitV[i] < high_cut):
    #         tracks_data_f2.append(track)
    #     else:
    #         removed_tracks.append(track)
            
    # all_D = np.concat([track['D'] for track in tracks_data_f2])
    # all_V = np.concat([track['V'] for track in tracks_data_f2])
    # all_F = np.concat([track['F'] for track in tracks_data_f2])
    # all_removedD = np.concat([track['D'] for track in removed_tracks])
    # all_removedV = np.concat([track['V'] for track in removed_tracks])   
    
    # #### Plot second filter
    # # fig22, ax22 = plt.subplots(1, 1, figsize = (8, 6))
    # # fig, ax = fig22, ax22
    # fig, ax = fig1, axes1[2]
    # ax.plot(all_D, all_V, ls='', marker='.', alpha=0.05)
    # ax.plot(D_plot, V_fit_pL, ls='-.', c='darkred', lw=2.0, label = 'Naive fit')
    # ax.plot(D_plot, V_fit_pL*high_cut, 
    #         ls='-.', c='green', lw=1.25, label = f'High cut = {high_cut}')
    # ax.plot(D_plot, V_fit_pL*low_cut,  
    #         ls='-.', c='blue', lw=1.25, label = f'Low cut = {low_cut}')
    # ax.plot(all_removedD, all_removedV, ls='', marker='.', alpha=0.05)
    # MD, MV = max(all_D), max(all_V)
    # ax.set_xlim([0, 1.1*MD])
    # ax.set_ylim([0, 1.2*MV])
    # ax.grid()
    # ax.legend()
    # ax.set_xlabel('D [µm]')
    # ax.set_ylabel('V [µm/s]')
    # ax.set_title(f'Second Filter, N = {len(tracks_data_f2)}')
            
    
    # #### In case of flow correction
    # if flowCorrection:
    #     df_fC = pd.read_csv(flowCorrectionPath, header = 3, sep='\t', #skiprows=2,
    #                         on_bad_lines='skip', encoding='utf_8',
    #                         names=['x', 'y', 'u', 'v', 'vector_type']) # 'utf_16_le'
    #     # Original column names
    #     # x [px]	y [px]	u [px/frame]	v [px/frame]	Vector type [-]
        
    #     tracks_data_f3 = ApplyFlowCorrection(tracks_data_f2, df_fC)
    
    
        
    
    
    
    
    # #### Final fits
    # D_plot = np.linspace(1, 5000, 500)

    # #### Velocity
    # # Double Expo
    # V_popt_2exp, V_pcov_2exp = optimize.curve_fit(doubleExpo, all_D, all_V, 
    #                        p0 = [1000, 50, 100, 1000], 
    #                        bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
    # V_fit_2exp = doubleExpo(D_plot, *V_popt_2exp)
    # V_label_2exp = r'$\bf{A \cdot exp(-x/k_1) + B \cdot exp(-x/k_2)}$' + '\n'
    # V_label_2exp += '$A$ = {:.2e} | $k_1$ = {:.2f}\n$B$ = {:.2f} | $k_2$ = {:.2e}'.format(*V_popt_2exp)

    # # Power Law
    # V_popt_pL, V_pcov_pL = optimize.curve_fit(powerLaw, all_D, all_V, 
    #                        p0 = [1000, -2], 
    #                        bounds=([0, -10], [np.inf, 0]))
    # V_fit_pL = powerLaw(D_plot, *V_popt_pL)
    # V_label_pL = r'$\bf{A \cdot x^k}$' + '\n'
    # V_label_pL += '$A$ = {:.2e} | $k$ = {:.2f}'.format(*V_popt_pL)

    # #### Force
    # # Double Expo
    # F_popt_2exp, F_pcov_2exp = optimize.curve_fit(doubleExpo, all_D, all_F, 
    #                        p0 = [1000, 50, 100, 1000], 
    #                        bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
    # F_fit_2exp = doubleExpo(D_plot, *F_popt_2exp)
    # F_label_2exp = r'$\bf{A \cdot exp(-x/k_1) + B \cdot exp(-x/k_2)}$' + '\n'
    # F_label_2exp += '$A$ = {:.2e} | $k_1$ = {:.2f}\n$B$ = {:.2f} | $k_2$ = {:.2e}'.format(*F_popt_2exp)

    # # Power Law
    # F_popt_pL, F_pcov_pL = optimize.curve_fit(powerLaw, all_D, all_F, 
    #                        p0 = [1000, -2], 
    #                        bounds=([0, -10], [np.inf, 0]))
    # F_fit_pL = powerLaw(D_plot, *F_popt_pL)
    # F_label_pL = r'$\bf{A \cdot x^k}$' + '\n'
    # F_label_pL += '$A$ = {:.2e} | $k$ = {:.2f}'.format(*F_popt_pL)


    # #### Plot the clean fits
    # fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    # fig = fig2
    # color_V = pm.colorList10[0]
    # color_F = pm.cL_Set21[0]
    # ax = axes2[0,0]
    # ax.plot(all_D, all_V, ls='', marker='.', alpha=0.01, c = color_V)
    # ax.plot(D_plot, V_fit_2exp, 'r-', label = label_2exp)
    # ax.plot(D_plot, V_fit_pL, ls='-', color='darkorange', label = V_label_pL)
    # # ax.plot(D_plot, V_fit_pLS, 'c-', label = V_label_pLS)
    # ax.grid()
    # ax.set_xlim([0, 1.1 * max(all_D)])
    # ax.set_ylim([0, 1.2 * max(all_V)])
    # ax.set_xlabel('d [µm]')
    # ax.set_ylabel('v [µm/s]')
    # # ax.legend()

    # ax = axes2[0,1]
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.plot(all_D, all_V, ls='', marker='.', alpha=0.01, c = color_V)
    # ax.plot(D_plot, V_fit_2exp, 'r-', label = V_label_2exp)
    # ax.plot(D_plot, V_fit_pL, ls='-', color='darkorange', label = V_label_pL)
    # # ax.plot(D_plot, V_fit_pLS, 'c-', label = V_label_pLS)
    # ax.grid()
    # ax.set_xlim([50, 5000])
    # ax.set_ylim([0.01, 100])
    # ax.set_xlabel('d [µm]')
    # ax.set_ylabel('v [µm/s]')
    # ax.legend(title='Fit on V(d)',
    #           loc="center left",
    #           bbox_to_anchor=(1, 0, 0.5, 1))

    # ax = axes2[1,0]
    # ax.plot(all_D, all_F, ls='', marker='.', alpha=0.01, c = color_F)
    # ax.plot(D_plot, F_fit_2exp, 'r-', label = F_label_2exp)
    # ax.plot(D_plot, F_fit_pL, ls='-', color='darkorange', label = F_label_pL)
    # # ax.plot(D_plot, V_fit_pLS, 'c-', label = V_label_pLS)
    # ax.grid()
    # ax.set_xlim([0, 1.1 * max(all_D)])
    # ax.set_ylim([0, 1.2 * max(all_F)])
    # ax.set_xlabel('d [µm]')
    # ax.set_ylabel('F [pN]')
    # # ax.legend()

    # ax = axes2[1,1]
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.plot(all_D, all_F, ls='', marker='.', alpha=0.01, c = color_F)
    # ax.plot(D_plot, F_fit_2exp, 'r-', label = F_label_2exp)
    # ax.plot(D_plot, F_fit_pL, ls='-', color='darkorange', label = F_label_pL)
    # # ax.plot(D_plot, V_fit_pLS, 'c-', label = V_label_pLS)
    # ax.grid()
    # ax.set_xlim([50, 5000])
    # ax.set_ylim([0.01, 100])
    # ax.set_xlabel('d [µm]')
    # ax.set_ylabel('F [pN]')
    # ax.legend(title='Fit on F(d)',
    #           loc="center left",
    #           bbox_to_anchor=(1, 0, 0.5, 1))

    # MainTitle = 'Calibration Data'
    # if expLabel != '':
    #     MainTitle = MainTitle + ' - ' + expLabel
    # fig.suptitle(MainTitle)
    # fig.tight_layout()
    
    # plt.show()
    
    # if savePlots:
    #     fig1.savefig(os.path.join(saveDir, expLabel + '_Traj.png'), dpi=400)
    #     fig2.savefig(os.path.join(saveDir, expLabel + '_Fits.png'), dpi=400)
    
    # if saveResults:
    #     dictResults = {'V_popt_2exp':V_popt_2exp,
    #                    'V_popt_pL':V_popt_pL,
    #                    'F_popt_2exp':F_popt_2exp,
    #                    'F_popt_pL':F_popt_pL,
    #                    'all_D':all_D,
    #                    'all_V':all_V,
    #                    'all_F':all_F,
    #                    }
    #     listOfdict2json(tracks_data_f2, saveDir, expLabel+'_allTracksData')
    #     dict2json(dictResults, saveDir, expLabel+'_fitData')
    #     # json2listOfdict(path, fileName)






# mainDir = 'E:/WorkingData/LeicaData/25-11-19/25-11-19_Droplet01_Gly80p_MyOne'
# mainDir += '/25-11-19_Droplet01_20x_FilmFluo_3spf_3/'

mainDir = 'C:/Users/josep/Desktop/Seafile/AnalysisPulls/'
mainDir += '25-11-19_Droplet01_Gly80p_MyOne/'
mainDir += '25-11-19_Droplet01_20x_FilmFluo_3spf_3/'

fileName = '20x_FilmFluo_3spf_3_CropMagbeads_Tracks.xml'
filePath = os.path.join(mainDir, fileName)

SCALE = 0.451
FPS = (1/3)
# Beads
Rb = 1 * 0.5
# Medium - °C 20.6 - %Gly 75
visco = 87.9 # mPa.s

# Magnet
MagX = 229
MagY = 702
MagR = 250 * 0.5
# Crop
CropX = 0
CropY = 0

all_tracks = ufun.importTrackMateTracks(filePath)
tracks_data = tracks_pretreatment_wFlow(all_tracks, SCALE, FPS, 
                                   MagX, MagY, MagR, Rb, visco,
                                   CropXY = [CropX, CropY])

# flowCorrectionPath = 'E:\WorkingData/LeicaData/25-11-19/25-11-19_Droplet01_Gly80p_MyOne/'
# flowCorrectionPath += '25-11-19_Droplet01_20x_FilmFluo_3spf_3/Res01_PIVlab.txt'

flowCorrectionFile = 'Res01_PIVlab.txt'
flowCorrectionPath = os.path.join(mainDir, flowCorrectionFile)

df_fC = pd.read_csv(flowCorrectionPath, header = 3, sep='\t', #skiprows=2,
                    on_bad_lines='skip', encoding='utf_8',
                    names=['x', 'y', 'u', 'v', 'vector_type']).dropna(subset=['u']) # 'utf_16_le'

# df_fC = df_flowCorr
# X_fC = df_fC['x'].unique()
# step_fC = np.polyfit(np.arange(len(X_fC)), X_fC, 1)[0]
# df_fC['xi'] = df_fC['x']//step_fC
# df_fC['yi'] = df_fC['y']//step_fC
# list_xi_fC = df_fC['xi'].unique().astype(int)
# list_yi_fC = df_fC['yi'].unique().astype(int)

df_fC2 = tracks_analysis_wFlow(tracks_data, df_fC, expLabel = '', 
                    flowCorrection = False, flowCorrectionPath = '',
                    saveResults = False, savePlots = False, saveDir = '.')
















# %% 3. Analyse capillaries 25-11-19 & 25-11-21

# saveDir = 'C:/Users/josep/Desktop/Seafile/DownloadedFromSeafile/25-11-19+21'
# saveDir = 'E:/WorkingData/LeicaData/25-11-19/'
# saveDir = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/25-11_DynabeadsInCapillaries_CalibrationsTests/'
saveDir = 'C:/Users/josep/Desktop/Seafile/AnalysisPulls/25-11_DynabeadsInCapillaries_CalibrationsTests/'


# %%% 3.1 MyOne - Gly75%

# %%%% IMPORT

# mainDir = 'E:/WorkingData/LeicaData/25-11-19/Capillaire04_Gly75p_MyOne/'
# mainDir = 'C:/Users/josep/Desktop/Seafile/DownloadedFromSeafile/'
# mainDir += '25-11-19+21/25-11-19_Capillaire04_Gly75p_MyOne/'
mainDir = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/'
mainDir += '25-11_DynabeadsInCapillaries_CalibrationsTests/Tracks'
tracks_data = []

# Common data

# Beads
Rb = 1 * 0.5
# Medium
# °C 20.6
# %Gly 75
visco = 53.3 # mPa.s

#### Film 1

fileName = '25-11-19_Capi04_FilmBF_5fps_1_CropInv_Tracks.xml'
filePath = os.path.join(mainDir, fileName)

SCALE = 0.451
FPS = 5
# Magnet
MagX = 154 
MagY = 497 
MagR = 234 * 0.5 
# Crop
CropX = 790 
CropY = 0 

all_tracks = ufun.importTrackMateTracks(filePath)
tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                   MagX, MagY, MagR, Rb, visco,
                                   CropXY = [CropX, CropY])

#### Film 2

fileName = '25-11-19_Capi04_FilmBF_5fps_2_CropInv_Tracks.xml'
filePath = os.path.join(mainDir, fileName)

SCALE = 0.451
FPS = 5
# Magnet
MagX = 140 
MagY = 551 
MagR = 232 * 0.5 
# Crop
CropX = 715 
CropY = 1 

all_tracks = ufun.importTrackMateTracks(filePath)
tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                   MagX, MagY, MagR, Rb, visco,
                                   CropXY = [CropX, CropY])

#### Film 4

fileName = '25-11-19_Capi04_FilmBF_5fps_4_CropInv_Tracks.xml'
filePath = os.path.join(mainDir, fileName)

SCALE = 0.451
FPS = 5
# Magnet
MagX = 149 
MagY = 610 
MagR = 238 * 0.5 
# Crop
CropX = 723 
CropY = 0 

all_tracks = ufun.importTrackMateTracks(filePath)
tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                   MagX, MagY, MagR, Rb, visco,
                                   CropXY = [CropX, CropY])

# %%%% ANALYZE

tracks_analysis(tracks_data, expLabel = 'MyOne_Glycerol75%', 
                saveResults = True, savePlots = True, saveDir = saveDir)

# %%%% ---------------------




# %%% 3.2 MyOne - Gly80%

# %%%% IMPORT

#### PARTIE 1 - 25-11-19

# mainDir = 'E:/WorkingData/LeicaData/25-11-19/Capillaire04_Gly75p_MyOne/'
# mainDir = 'C:/Users/josep/Desktop/Seafile/DownloadedFromSeafile/'
# mainDir += '25-11-19+21/25-11-19_Capillaire01_Gly80p_MyOne/'
mainDir = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/'
mainDir += '25-11_DynabeadsInCapillaries_CalibrationsTests/Tracks'
tracks_data = []

# Common data

# Beads
Rb = 1 * 0.5
# Medium - °C 20.6 - %Gly 75
visco = 87.9 # mPa.s

#### Film 3
fileName = '25-11-19_Capi01_FilmBF_5fps_3_CropInv_Tracks.xml'
filePath = os.path.join(mainDir, fileName)

SCALE = 0.451
FPS = 5
# Magnet
MagX = 163
MagY = 488
MagR = 240
# Crop
CropX = 824
CropY = 0

all_tracks = ufun.importTrackMateTracks(filePath)
tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                   MagX, MagY, MagR, Rb, visco,
                                   CropXY = [CropX, CropY])


#### Film 4
fileName = '25-11-19_Capi01_FilmBF_5fps_4_CropInv_Tracks.xml'
filePath = os.path.join(mainDir, fileName)

SCALE = 0.451
FPS = 5
# Magnet
MagX = 156
MagY = 435
MagR = 236
# Crop
CropX = 821
CropY = 0

all_tracks = ufun.importTrackMateTracks(filePath)
tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                   MagX, MagY, MagR, Rb, visco,
                                   CropXY = [CropX, CropY])




#### PARTIE 2 - 25-11-21

# mainDir = 'E:/WorkingData/LeicaData/25-11-21/Capillaire01_Gly80p_MyOne/'
# mainDir = 'C:/Users/josep/Desktop/Seafile/DownloadedFromSeafile/'
# mainDir += '25-11-19+21/25-11-21_Capillaire01_Gly80p_MyOne/'
mainDir = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/'
mainDir += '25-11_DynabeadsInCapillaries_CalibrationsTests/Tracks'

# Beads
Rb = 1 * 0.5
# Medium - °C 20.6 - %Gly 80
visco = 87.9 # mPa.s

#### Film 1

# fileName = 'FilmBF_5fps_1_CropInv_Tracks.xml'
# filePath = os.path.join(mainDir, fileName)

# SCALE = 0.451
# FPS = 5
# # Magnet
# MagX = 295.5 
# MagY = 1003.5 
# MagR = 259 * 0.5 
# # Crop
# CropX = 892 
# CropY = 288 
# # Beads
# Rb = 1 * 0.5
# # Medium
# # °C 20.6
# # %Gly 80
# visco = 87.9 # mPa.s


# all_tracks = ufun.importTrackMateTracks(filePath)
# tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                   # MagX, MagY, MagR, Rb, visco,
                                   # CropXY = [CropX, CropY])

#### Film 2

# fileName = 'FilmBF_5fps_2_CropInv_Tracks.xml'
# filePath = os.path.join(mainDir, fileName)

# SCALE = 0.451
# FPS = 5
# # Magnet
# MagX = 293.5 
# MagY = 1000.5 
# MagR = 261 * 0.5 
# # Crop
# CropX = 856 
# CropY = 332 
# # Beads
# Rb = 1 * 0.5
# # Medium
# # °C 20.6
# # %Gly 80
# visco = 87.9 # mPa.s

# all_tracks = ufun.importTrackMateTracks(filePath)
# tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                   # MagX, MagY, MagR, Rb, visco,
                                   # CropXY = [CropX, CropY])

#### Film 3

fileName = '25-11-21_Capi01_20x_FilmBF_5fps_3_CropInv_Tracks.xml'
filePath = os.path.join(mainDir, fileName)

SCALE = 0.451
FPS = 5
# Magnet
MagX = 293.5 
MagY = 1000.5 
MagR = 261 * 0.5 
# Crop
CropX = 856 
CropY = 332 


all_tracks = ufun.importTrackMateTracks(filePath)
tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                   MagX, MagY, MagR, Rb, visco,
                                   CropXY = [CropX, CropY])

#### Film 4

fileName = '25-11-21_Capi01_20x_FilmBF_2-5fps_4_CropInv_Tracks.xml'
filePath = os.path.join(mainDir, fileName)

SCALE = 0.451
FPS = 2.5
# Magnet
MagX = 299.5 
MagY = 994.5 
MagR = 263 * 0.5 
# Crop
CropX = 872 
CropY = 344 
# Beads
Rb = 1 * 0.5
# Medium - °C 20.6 - %Gly 80
visco = 87.9 # mPa.s

all_tracks = ufun.importTrackMateTracks(filePath)
tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                   MagX, MagY, MagR, Rb, visco,
                                   CropXY = [CropX, CropY])

# %%%% ANALYZE

tracks_analysis(tracks_data, expLabel = 'MyOne_Glycerol80%', 
                saveResults = True, savePlots = True, saveDir = saveDir)


# %%%% ---------------------




# %%% 3.3 M270 - Gly75%

# %%%% IMPORT

# mainDir = 'E:/WorkingData/LeicaData/25-11-19/Capillaire02_Gly75p_M270/'
# mainDir = 'C:/Users/josep/Desktop/Seafile/DownloadedFromSeafile/'
# mainDir += '25-11-19+21/25-11-19_Capillaire02_Gly75p_M270/'
mainDir = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/'
mainDir += '25-11_DynabeadsInCapillaries_CalibrationsTests/Tracks'
tracks_data = []

# Beads
Rb = 2.7 * 0.5
# Medium
# °C 20.6
# %Gly 75
visco = 53.3 # mPa.s


#### Film 1

fileName = '25-11-19_Capi02_FilmBF_5fps_1_CropInv_Tracks.xml'
filePath = os.path.join(mainDir, fileName)

SCALE = 0.451
FPS = 5
# Magnet
MagX = 166.5 
MagY = 608.5 
MagR = 249.0 * 0.5 
# Crop
CropX = 920 
CropY = 0 

all_tracks = ufun.importTrackMateTracks(filePath)
tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                   MagX, MagY, MagR, Rb, visco,
                                   CropXY = [CropX, CropY])

#### Film 2

fileName = '25-11-19_Capi02_FilmBF_5fps_2_CropInv_Tracks.xml'
filePath = os.path.join(mainDir, fileName)

SCALE = 0.451
FPS = 5
# Magnet
MagX = 164 
MagY = 572 
MagR = 244 * 0.5 
# Crop
CropX = 876 
CropY = 0 

all_tracks = ufun.importTrackMateTracks(filePath)
tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                   MagX, MagY, MagR, Rb, visco,
                                   CropXY = [CropX, CropY])

#### Film 3

fileName = '25-11-19_Capi02_FilmBF_5fps_3_CropInv_Tracks.xml'
filePath = os.path.join(mainDir, fileName)

SCALE = 0.451
FPS = 5
# Magnet
MagX = 164.5 
MagY = 601.5 
MagR = 247 * 0.5 
# Crop
CropX = 800 
CropY = 0 

all_tracks = ufun.importTrackMateTracks(filePath)
tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                   MagX, MagY, MagR, Rb, visco,
                                   CropXY = [CropX, CropY])



# %%%% ANALYZE

tracks_analysis(tracks_data, expLabel = 'M270_Glycerol75%', 
                saveResults = True, savePlots = True, saveDir = saveDir)


# %%%% ---------------------

# %%% 3.4 M270 - Gly80%

# %%%% IMPORT

# mainDir = 'E:/WorkingData/LeicaData/25-11-19/Capillaire03_Gly80p_M270/'
# mainDir = 'C:/Users/josep/Desktop/Seafile/DownloadedFromSeafile/'
# mainDir += '25-11-19+21/25-11-19_Capillaire03_Gly80p_M270/'
mainDir = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/'
mainDir += '25-11_DynabeadsInCapillaries_CalibrationsTests/Tracks'
tracks_data = []

# Beads
Rb = 2.7 * 0.5
# Medium
# °C 20.6
# %Gly 80
visco = 87.9 # mPa.s



#### Film 1

fileName = '25-11-19_Capi03_FilmBF_5fps_1_CropInv_Tracks.xml'
filePath = os.path.join(mainDir, fileName)

SCALE = 0.451
FPS = 5
# Magnet
MagX = 163.5 
MagY = 512.5 
MagR = 231 * 0.5 
# Crop
CropX = 800 
CropY = 0 

all_tracks = ufun.importTrackMateTracks(filePath)
tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                   MagX, MagY, MagR, Rb, visco,
                                   CropXY = [CropX, CropY])

#### Film 2

fileName = '25-11-19_Capi03_FilmBF_5fps_2_CropInv_Tracks.xml'
filePath = os.path.join(mainDir, fileName)

SCALE = 0.451
FPS = 5
# Magnet
MagX = 170 
MagY = 576 
MagR = 222 * 0.5 
# Crop
CropX = 859 
CropY = 0 

all_tracks = ufun.importTrackMateTracks(filePath)
tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                   MagX, MagY, MagR, Rb, visco,
                                   CropXY = [CropX, CropY])

#### Film 3

fileName = '25-11-19_Capi03_FilmBF_5fps_3_CropInv_Tracks.xml'
filePath = os.path.join(mainDir, fileName)

SCALE = 0.451
FPS = 5
# Magnet
MagX = 168.5 
MagY = 665.5 
MagR = 239 * 0.5 
# Crop
CropX = 810 
CropY = 134 

all_tracks = ufun.importTrackMateTracks(filePath)
tracks_data += tracks_pretreatment(all_tracks, SCALE, FPS, 
                                   MagX, MagY, MagR, Rb, visco,
                                   CropXY = [CropX, CropY])

# %%%% ANALYZE

tracks_analysis(tracks_data, expLabel = 'M270_Glycerol80%', 
                saveResults = True, savePlots = True, saveDir = saveDir)

# %%%% ---------------------

# %%% 3.5 Compare the different analysis for a given bead type
    

# srcDir = 'C:/Users/josep/Desktop/Seafile/DownloadedFromSeafile/25-11-19+21'
# srcDir = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/25-11_DynabeadsInCapillaries_CalibrationsTests/Results'
# dstDir = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/25-11_DynabeadsInCapillaries_CalibrationsTests/'
srcDir = 'C:/Users/josep/Desktop/Seafile/AnalysisPulls/25-11_DynabeadsInCapillaries_CalibrationsTests/Results'
dstDir = 'C:/Users/josep/Desktop/Seafile/AnalysisPulls/25-11_DynabeadsInCapillaries_CalibrationsTests/'


labelList = ['MyOne_Glycerol75%', 'MyOne_Glycerol80%']
examineCalibration(srcDir, labelList = labelList,
                   savePlots = True, saveDir = dstDir)

labelList = ['M270_Glycerol75%', 'M270_Glycerol80%']
examineCalibration(srcDir, labelList = labelList,
                   savePlots = True, saveDir = dstDir)

# labelList = ['MyOne_Glycerol75%', 'M270_Glycerol75%']
# examineCalibration(srcDir, labelList = labelList)

# labelList = ['MyOne_Glycerol80%', 'M270_Glycerol80%']
# examineCalibration(srcDir, labelList = labelList)




# %%% ---------------------------------------------------------------------------------------------















# %% 4. Analyse droplets

# %%% First movie

# mainDir = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/'
# mainDir += '25-11_DynabeadsInCapillaries_CalibrationsTests/Tracks'
# mainDir = 'E:/WorkingData/LeicaData/25-11-19/25-11-19_Droplet01_Gly80p_MyOne'
# mainDir += '/25-11-19_Droplet01_20x_FilmFluo_3spf_3/'
mainDir = 'C:/Users/josep/Desktop/Seafile/AnalysisPulls/'
mainDir += '25-11-19_Droplet01_Gly80p_MyOne/'
mainDir += '25-11-19_Droplet01_20x_FilmFluo_3spf_3/'

# %%%% Look at the tracer motion

fileName = '20x_FilmFluo_3spf_3_CropTracers_MinusMED_Tracks.xml'
filePath = os.path.join(mainDir, fileName)

SCALE = 0.451
FPS = 5

# Magnet
MagX = 229
MagY = 702
MagR = 250 * 0.5
# Crop
CropX = 0
CropY = 0

all_tracks = ufun.importTrackMateTracks(filePath)
tracks_data = tracerTracks_pretreatment(all_tracks, SCALE, FPS, 
                                        MagX, MagY, MagR, 
                                        CropXY = [0, 0])

tracerTracks_analysis(tracks_data, MagR = MagR*SCALE)

# %%%% Test ApplyFlowCorrection()

def ApplyFlowCorrection(tracks_data, df_flowCorr):
    df = df_flowCorr
    XX = df['x'].unique()
    step = np.polyfit(np.arange(len(XX)), XX, 1)[0]
    df['xi'] = df['x']//step
    df['yi'] = df['y']//step
    list_xi = df['xi'].unique().astype(int)
    list_yi = df['yi'].unique().astype(int)
    
    corrected_tracks_data = []
    
    for track in tracks_data[:1]:
        valid_track = True
        track['xi'] = ((track['Xraw']/0.451)//step).astype(int)
        track['yi'] = ((track['Yraw']/0.451)//step).astype(int)
        # print(track['Xraw'], track['Yraw'])
        valid_xi = np.array([(xi in list_xi) for xi in track['xi']])
        valid_yi = np.array([(yi in list_yi) for yi in track['yi']])
        valid_point = (valid_xi & valid_yi)
        valid_idx = [i for i in range(len(valid_point)) if valid_point[i]]
        # print(valid_idx)
        
        # track['T_fc'] = track['T']
        # track['D_fc'] = track['D']
        # track['V_fc'] = track['V']
        track['T_fc'] = track['T'][valid_point]
        track['D_fc'] = track['D'][valid_point]
        track['V_fc'] = track['V'][valid_point]
        
        phi = np.pi+track['phi']
        vect_dir = np.array([np.cos(phi), np.sin(phi)]).T
        # print(vect_dir)
        
        UU = [float(df.loc[(df['xi']==track['xi'][i]) & (df['yi']==track['yi'][i]), 'u'].values[0]) for i in valid_idx]
        # print(UU)
        VV = [float(df.loc[(df['xi']==track['xi'][i]) & (df['yi']==track['yi'][i]), 'v'].values[0]) for i in valid_idx]
        # print(VV)
        UUVV = np.array([UU, VV]).T
        proj_UUVV = UUVV @ vect_dir
        
        track['V_fc2'] = track['V_fc']-proj_UUVV
        print(track['V_fc2'])
        
        # track['T_fc'] = track['T_fc'][valid_point]
        # track['D_fc'] = track['D_fc'][valid_point]
        # track['V_fc'] = track['V_fc'][valid_point]
        
        if valid_track:
            corrected_tracks_data.append(track)
        
    return(df)

#### test 
flowCorrectionPath = 'E:\WorkingData/LeicaData/25-11-19/25-11-19_Droplet01_Gly80p_MyOne/'
flowCorrectionPath += '25-11-19_Droplet01_20x_FilmFluo_3spf_3/Res01_PIVlab.txt'
df_fC = pd.read_csv(flowCorrectionPath, header = 3, sep='\t', #skiprows=2,
                    on_bad_lines='skip', encoding='utf_8',
                    names=['x', 'y', 'u', 'v', 'vector_type']) # 'utf_16_le'
# Original column names
# x [px]	y [px]	u [px/frame]	v [px/frame]	Vector type [-]

output = ApplyFlowCorrection(tracks_data, df_fC)


# %%%% Test ExaminePIV()

def ExaminePIV(df_flowBeads, df_flowTracers):
    df_fB = df_flowBeads
    df_fT = df_flowTracers
    for df in [df_fB, df_fT]:
        X = df['x'].unique()
        step = np.polyfit(np.arange(len(X)), X, 1)[0]
        df['xi'] = df['x']//step
        df['yi'] = df['y']//step
    list_xi = df_fT['xi'].unique().astype(int)
    list_yi = df_fT['yi'].unique().astype(int)
    
    corrected_tracks_data = []
    
    for track in tracks_data[:1]:
        valid_track = True
        track['xi'] = ((track['Xraw']/0.451)//step).astype(int)
        track['yi'] = ((track['Yraw']/0.451)//step).astype(int)
        # print(track['Xraw'], track['Yraw'])
        valid_xi = np.array([(xi in list_xi) for xi in track['xi']])
        valid_yi = np.array([(yi in list_yi) for yi in track['yi']])
        valid_point = (valid_xi & valid_yi)
        valid_idx = [i for i in range(len(valid_point)) if valid_point[i]]
        # print(valid_idx)
        
        # track['T_fT'] = track['T']
        # track['D_fT'] = track['D']
        # track['V_fT'] = track['V']
        track['T_fT'] = track['T'][valid_point]
        track['D_fT'] = track['D'][valid_point]
        track['V_fT'] = track['V'][valid_point]
        
        phi = np.pi+track['phi']
        vect_dir = np.array([np.cos(phi), np.sin(phi)]).T
        # print(vect_dir)
        
        UU = [float(df_fT.loc[(df_fT['xi']==track['xi'][i]) & (df_fT['yi']==track['yi'][i]), 'u'].values[0]) for i in valid_idx]
        # print(UU)
        VV = [float(df_fT.loc[(df_fT['xi']==track['xi'][i]) & (df_fT['yi']==track['yi'][i]), 'v'].values[0]) for i in valid_idx]
        # print(VV)
        UUVV = np.array([UU, VV]).T
        proj_UUVV = UUVV @ vect_dir
        
        track['V_fT2'] = track['V_fT']-proj_UUVV
        print(track['V_fT2'])
        
        # track['T_fT'] = track['T_fT'][valid_point]
        # track['D_fT'] = track['D_fT'][valid_point]
        # track['V_fT'] = track['V_fT'][valid_point]
        
        if valid_track:
            corrected_tracks_data.append(track)
        
    return(df_fT)

#### test 
flowCorrectionPath = 'E:\WorkingData/LeicaData/25-11-19/25-11-19_Droplet01_Gly80p_MyOne/'
flowCorrectionPath += '25-11-19_Droplet01_20x_FilmFluo_3spf_3/Res01_PIVlab.txt'
df_fC = pd.read_csv(flowCorrectionPath, header = 3, sep='\t', #skiprows=2,
                    on_bad_lines='skip', encoding='utf_8',
                    names=['x', 'y', 'u', 'v', 'vector_type']) # 'utf_16_le'
# Original column names
# x [px]	y [px]	u [px/frame]	v [px/frame]	Vector type [-]

output = ApplyFlowCorrection(tracks_data, df_fC)



# %%% First movie

mainDir = 'E:/WorkingData/LeicaData/25-11-19/Droplet01_Gly80p_MyOne/20x_FilmFluo_3spf_3'
fileName = '20x_FilmFluo_3spf_3_CropTracers_MinusMED_MedandGausFilter_Tracks.xml'
filePath = os.path.join(mainDir, fileName)

SCALE = 0.451
FPS = 1/3
MagX = 233.5 * SCALE
MagY = 704.5 * SCALE
MagR = (241 / 2) * SCALE
Rb = 0.25 # µm

all_tracks_t1 = ufun.importTrackMateTracks(filePath)
tracks_data_t1 = tracerTracks_pretreatment(all_tracks_t1,
                                    SCALE, FPS, 
                                    MagX, MagY, MagR, Rb)

tracks_data = tracks_data_t1

# %%%% First filter

# tracks_data_f1 = []
# for track in tracks_data:
#     crit1 = (np.abs(track['delta']*180/np.pi) < 15)
#     crit2 = (np.abs(track['r2'] > 0.85))
#     bypass1 = (np.min(track['X2'] < 300))
#     if (crit1 and crit2) or bypass1:
#         tracks_data_f1.append(track)
    
tracks_data_f1 = tracks_data

fig1, axes1 = plt.subplots(1, 2, figsize = (20, 12), 
                           sharey=True)
fig = fig1
ax = axes1[0]
ax.set_title(f'All tracks, N = {len(tracks_data)}')
for track in tracks_data:
    X, Y = track['X2'][8:12], track['Y2'][8:12]
    ax.plot(X, Y, ls='-', lw=0.5, marker = '.', ms=2)
    
ax = axes1[1]
ax.set_title(f'Smoothed tracks, N = {len(tracks_data_f1)}')
for track in tracks_data_f1:
    X, Y = track['X3'][8:12], track['Y3'][8:12]
    
    ax.plot(X, Y, ls='-', lw=0.5, marker = '.', ms=2)

for ax in axes1:
    circle1 = plt.Circle((0, 0), MagR, color='dimgrey')
    ax.add_patch(circle1)
    # ax.axvspan(wall_L, wall_R, color='lightgray', zorder=0)
    ax.grid()
    ax.axis('equal')
    ax.set_xlabel('X [µm]')
    ax.set_ylabel('Y [µm]')
    ax.set_xlim([-100, 600])
    ax.set_ylim([-400, 400])
    
plt.show()

# %%%% Second filter

fig2, ax2 = plt.subplots(1, 1, figsize = (8, 6))
fig, ax = fig2, ax2
all_medD = np.array([track['medD'] for track in tracks_data_f1])
all_medV = np.array([track['medV'] for track in tracks_data_f1])
all_D = np.concat([track['D'] for track in tracks_data_f1])
all_V = np.concat([track['V'] for track in tracks_data_f1])
# ax.plot(all_medD, all_medV, ls='', marker='.')
ax.plot(all_D, all_V, ls='', marker='.', alpha=0.05)
ax.grid()
ax.set_xlabel('D [µm]')
ax.set_ylabel('V [µm/s]')
    
plt.show()

D_plot = np.linspace(1, 5000, 500)

#### Double Expo
V_popt_2exp, V_pcov_2exp = optimize.curve_fit(doubleExpo, all_D, all_V, 
                       p0 = [1000, 50, 100, 1000], 
                       bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
V_fit_2exp = doubleExpo(D_plot, *V_popt_2exp)
label_2exp = r'$\bf{A \cdot exp(-x/k_1) + B \cdot exp(-x/k_2)}$' + '\n'
label_2exp += '$A$ = {:.2e} | $k_1$ = {:.2f}\n$B$ = {:.2f} | $k_2$ = {:.2e}'.format(*V_popt_2exp)

#### Power Law
V_popt_pL, V_pcov_pL = optimize.curve_fit(powerLaw, all_D, all_V, 
                       p0 = [1000, -2], 
                       bounds=([0, -10], [np.inf, 0]))
V_fit_pL = powerLaw(D_plot, *V_popt_pL)
V_label_pL = r'$\bf{A \cdot x^k}$' + '\n'
V_label_pL += '$A$ = {:.2e} | $k$ = {:.2f}'.format(*V_popt_pL)

expected_medV = powerLaw(all_medD, *V_popt_pL)
ratio_fitV = all_medV/expected_medV
high_cut = 1.7
low_cut = 0.5

fig21, ax21 = plt.subplots(1,1)
fig, ax = fig21, ax21
ax.plot(all_medD, ratio_fitV, ls='', marker='.', alpha=0.3)
ax.axhline(high_cut, color = 'k', ls='-.')
ax.axhline(low_cut, color = 'k', ls='-.')
plt.show()

tracks_data_f2 = []
removed_tracks = []
for i, track in enumerate(tracks_data_f1):
    if (ratio_fitV[i] > low_cut) and (ratio_fitV[i] < high_cut):
        tracks_data_f2.append(track)
    else:
        removed_tracks.append(track)
        
all_D = np.concat([track['D'] for track in tracks_data_f2])
all_V = np.concat([track['V'] for track in tracks_data_f2])
all_removedD = np.concat([track['D'] for track in removed_tracks])
all_removedV = np.concat([track['V'] for track in removed_tracks])   

     
fig22, ax22 = plt.subplots(1, 1, figsize = (8, 6))
fig, ax = fig22, ax22
# ax.plot(all_medD, all_medV, ls='', marker='.')
ax.plot(all_D, all_V, ls='', marker='.', alpha=0.05)
ax.plot(all_removedD, all_removedV, ls='', marker='.', alpha=0.05)
ax.grid()
ax.set_xlabel('D [µm]')
ax.set_ylabel('V [µm/s]')
        




# %% 11. Tests

# %%% Save a dict in .json

import json
import pickle

testPath = 'C:/Users/josep/Desktop/UrchinPolymers'
testFile = 'DataTestJson'
testDict = {'number':0,
            'string':'a',
            'list':[i for i in range(5)],
            'array':np.arange(5),
            'dict':{'key':'value'}}
testList = [testDict] * 3


def dict2json(d, path, fileName):
    for k in d.keys():
        obj = d[k]
        if isinstance(obj, np.ndarray):
            d[k] = d[k].tolist()
        else:
            pass
    with open(os.path.join(testPath, fileName + '.json'), 'w') as fp:
        json.dump(d, fp, indent=4)
        
def json2dict(path, fileName):
    with open(os.path.join(testPath, fileName + '.json'), 'r') as fp:
        d = json.load(fp)
    for k in d.keys():
        obj = d[k]
        if isinstance(obj, list):
            d[k] = np.array(d[k])
        else:
            pass
    return(d)


def listOfdict2json(L, path, fileName):
    for d in L:
        for k in d.keys():
            obj = d[k]
            if isinstance(obj, np.ndarray):
                d[k] = d[k].tolist()
            else:
                pass
    with open(os.path.join(testPath, fileName + '.json'), 'w') as fp:
        json.dump(L, fp, indent=4)
        
def json2listOfdict(path, fileName):
    with open(os.path.join(testPath, fileName + '.json'), 'r') as fp:
        L = json.load(fp)
    for d in L:
        for k in d.keys():
            obj = d[k]
            if isinstance(obj, list):
                d[k] = np.array(d[k])
            else:
                pass
    return(L)

# dict2json(testDict, testPath, testFile)
# dict2 = json2dict(testPath, testFile)
# listOfdict2json(testList, testPath, testFile)
# List2 = json2listOfdict(testPath, testFile)


# testPath = 'C:/Users/josep/Desktop/UrchinPolymers'
# testFile = 'DataTestPickle'
# testDict = {'number':0,
#             'string':'a',
#             'list':[i for i in range(5)],
#             'array':np.arange(5),
#             'dict':{'key':'value'}}

# def dict2pckl(d, path, fileName):
#     with open(os.path.join(testPath, fileName + '.pkl'), 'w') as fp:
#         pickle.dump(d, fp, sort_keys=True, indent=4)
        
# def pckl2dict(path, fileName):
#     with open(os.path.join(testPath, fileName + '.pkl'), 'r') as fp:
#         data = pickle.load(fp)
#     return(data)

# dict2pckl(testDict, testPath, testFile)
# dict3 = pckl2dict(testPath, testFile)

# %%% Test cleaning of TXY

def cleanRawTracks(all_tracks):
    clean_tracks = []
    for track in all_tracks:
        cleaned_track, track_valid = cleanRawTrack(track)
        if track_valid:
            clean_tracks.append(cleaned_track)
    return(clean_tracks)
            
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

testTrack1 = all_tracks[1]
testTrack48 = all_tracks[48]

cleaned_track1, track_valid1 = cleanRawTrack(testTrack1)
cleaned_track48, track_valid48 = cleanRawTrack(testTrack48)

# %%% New test





# %% 101. First scripts

# %%% Make a calibration curve by myself

# %%%% Pretreatment, as a function

def tracks_pretreatment_V0(all_tracks,
                        SCALE, FPS, 
                        MagX, MagY, MagR, Rb):

    tracks_data = []
    
    for i, track in enumerate(all_tracks):
        #### Conversion in um and sec
        T = track[:, 0] * (1/FPS)
        X = track[:, 1] * SCALE
        Y = track[:, 2] * SCALE
        tracks_data.append({'T':T, 'X':X, 'Y':Y})
        #### Origin as the magnet center
        X2, Y2 = X-MagX, MagY-Y
        # NB: inversion of Y so that the plt trajectories look like the Fiji ones
        medX2, medY2 = np.median(X2), np.median(Y2)
        tracks_data[i].update({'X2':X2, 'Y2':Y2,
                               'medX2':medX2, 'medY2':medY2})
        #### Rotate the trajectory by its own angle
        parms, res, wlm_res = ufun.fitLineHuber(X2, Y2, with_wlm_results=True)
        b_fit, a_fit = parms
        r2 = wlm_res.rsquared
        theta = np.atan(a_fit)
        rotation_mat = np.array([[np.cos(-theta), -np.sin(-theta)],
                                 [np.sin(-theta),  np.cos(-theta)]])
        rotated_XY = np.vstack((X2, Y2)).T @ rotation_mat.T
        X3, Y3 = rotated_XY[:,0], rotated_XY[:,1]
        tracks_data[i].update({'a_fit':a_fit, 'b_fit':b_fit, 'r2':r2,
                               'theta':theta,
                               'X3':X3, 'Y3':Y3})
        #### Rotate the trajectory by its angle with the magnet
        phi = np.atan(medY2/medX2)
        delta = theta-phi
        rotation_mat = np.array([[np.cos(-phi), -np.sin(-phi)],
                                 [np.sin(-phi),  np.cos(-phi)]])
        rotated_XY = np.vstack((X2, Y2)).T @ rotation_mat.T
        X4, Y4 = rotated_XY[:,0], rotated_XY[:,1]
        tracks_data[i].update({'phi':phi, 'delta':delta,
                               'X4':X4, 'Y4':Y4})
        #### Compute distances
        D2 = np.array(((X2**2 + Y2**2)**0.5) - MagR)
        D3 = np.array(X3 - MagR)
        D4 = np.array(X4 - MagR) # Note: D4 == D2
        tracks_data[i].update({'D2':D2, 'D3':D3, 'D4':D4,
                               })
        
        #### Compute splines
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
        V = V_spline
        #
        medD, medV = np.median(D), np.median(V)
        tracks_data[i].update({'D':D, 'V':V, 
                               'medD':medD, 'medV':medV,
                               })
        
    return(tracks_data)
        
        # fig2, ax2 = plt.subplots(1, 1, figsize = (8, 6))
        # fig, ax = fig2, ax2
        # all_medX2 = [track['medX2'] for track in tracks_data]
        # all_delta = [track['delta']*180/np.pi for track in tracks_data]
        # ax.plot(all_medX2, all_delta, ls='', marker='.')
        # ax.grid()
        # ax.set_xlabel('median(X) [µm]')
        # ax.set_ylabel(r'$\theta - \phi$ [°]')
            
        # plt.show()
        
        
        # fig3, axes3 = plt.subplots(1, 2, figsize = (8, 12))
        # fig = fig3
        # for track in tracks_data:
        #     if np.abs(track['delta']*180/np.pi) < 10:
        #         X2, Y2 = track['X2'], track['Y2']
        #         D2 = np.array(((X2**2 + Y2**2)**0.5) - MagR)
        #         X3 = track['X3']
        #         D3 = np.array(X3 - MagR)
        #         X4 = track['X4']
        #         D4 = np.array(X4 - MagR)
        #         ax = axes3[0]
        #         ax.plot(X2, D3/D2, ls='', marker='.')
        #         ax = axes3[1]
        #         ax.plot(X2, D4/D2, ls='', marker='.')
                
        # for ax in axes3:
        #     ax.set_ylim([1-10e-3, 1+10e-3])
        #     ax.grid()
            
        # plt.show()
        
        # fig4, axes4 = plt.subplots(1, 4, figsize = (20, 6), 
        #                            sharey=True)
        # fig = fig1
        # for i in range(len(axes4)):
        #     ax = axes4[i]
        #     track = tracks_data[20+i]
        #     D = track['D']
        #     ax.plot(D, track['V_spline'])
        #     ax.plot(D, track['V_savgol'])
        #     ax.plot(D, track['V_kalman'])

# %%%% Import Data 1

# mainDir = 'C://Users//Utilisateur//Desktop//AnalysisPulls//Tracks//25-10-11'
mainDir = 'C://Users//josep//Desktop//Seafile//AnalysisPulls//Tracks//25-10-11'
fileName = 'BeadPulling10_10fps_InvCroped_Tracks.xml'
filePath = os.path.join(mainDir, fileName)

SCALE = 0.451
FPS = 10
MagX = 150.5 * SCALE
MagY = 396.5 * SCALE
MagR = (223 / 2) * SCALE
Rb = 0.5 # µm

wall_L = 307*SCALE - MagX
wall_R = (307+280)*SCALE - MagX
all_tracks1 = ufun.importTrackMateTracks(filePath)
tracks_data1 = tracks_pretreatment(all_tracks1,
                        SCALE, FPS, 
                        MagX, MagY, MagR, Rb)

# %%%% Import Data 2

# mainDir = 'C://Users//Utilisateur//Desktop//AnalysisPulls//Tracks//25-10-11'
mainDir = 'C://Users//josep//Desktop//Seafile//AnalysisPulls//Tracks//25-10-11'
fileName = 'BeadPulling05_5fps_Inv_Tracks.xml'
filePath = os.path.join(mainDir, fileName)

SCALE = 0.451
FPS = 5
MagX = 168.5 * SCALE
MagY = 262.5 * SCALE
MagR = (217 / 2) * SCALE
Rb = 0.5 # µm


wall_L = 408*SCALE - MagX
wall_R = (408+288)*SCALE - MagX
all_tracks2 = ufun.importTrackMateTracks(filePath)
tracks_data2 = tracks_pretreatment(all_tracks2,
                        SCALE, FPS, 
                        MagX, MagY, MagR, Rb)

# %%%% Import Data 3

# mainDir = 'C://Users//Utilisateur//Desktop//AnalysisPulls//Tracks//25-10-11'
mainDir = 'C://Users//josep//Desktop//Seafile//AnalysisPulls//Tracks//25-10-11'
fileName = 'BeadPulling03_5fps_Inv_Tracks.xml'
filePath = os.path.join(mainDir, fileName)

SCALE = 0.451
FPS = 5
MagX = 129 * SCALE
MagY = 376 * SCALE
MagR = (234 / 2) * SCALE
Rb = 0.5 # µm

wall_L = 365*SCALE - MagX
wall_R = (365+290)*SCALE - MagX
all_tracks3 = ufun.importTrackMateTracks(filePath)
tracks_data3 = tracks_pretreatment(all_tracks3,
                        SCALE, FPS, 
                        MagX, MagY, MagR, Rb)

# %%%% Concatenate all tracks

# tracks_data = tracks_data1 + tracks_data2 + tracks_data3

# tracks_data = tracks_data2 + tracks_data3

tracks_data = tracks_data1

# %%%% First filter

tracks_data_f1 = []
for track in tracks_data:
    crit1 = (np.abs(track['delta']*180/np.pi) < 15)
    crit2 = (np.abs(track['r2'] > 0.85))
    bypass1 = (np.min(track['X2'] < 300))
    if (crit1 and crit2) or bypass1:
        tracks_data_f1.append(track)
    

fig1, axes1 = plt.subplots(2, 1, figsize = (10, 14), 
                           sharex=True)
fig = fig1
ax = axes1[0]
ax.set_title(f'All tracks, N = {len(tracks_data)}')
for track in tracks_data:
    X, Y = track['X2'], track['Y2']
    ax.plot(X, Y)
    
ax = axes1[1]
ax.set_title(f'Validated tracks, N = {len(tracks_data_f1)}')
for track in tracks_data_f1:
    X, Y = track['X2'], track['Y2']
    ax.plot(X, Y)

for ax in axes1:
    circle1 = plt.Circle((0, 0), MagR, color='dimgrey')
    ax.add_patch(circle1)
    # ax.axvspan(wall_L, wall_R, color='lightgray', zorder=0)
    ax.grid()
    ax.axis('equal')
    ax.set_xlabel('X [µm]')
    ax.set_ylabel('Y [µm]')
    ax.set_xlim([0, ax.get_xlim()[1]])
    
plt.show()

# %%%% Second filter

fig2, ax2 = plt.subplots(1, 1, figsize = (8, 6))
fig, ax = fig2, ax2
all_medD = np.array([track['medD'] for track in tracks_data_f1])
all_medV = np.array([track['medV'] for track in tracks_data_f1])
all_D = np.concat([track['D'] for track in tracks_data_f1])
all_V = np.concat([track['V'] for track in tracks_data_f1])
# ax.plot(all_medD, all_medV, ls='', marker='.')
ax.plot(all_D, all_V, ls='', marker='.', alpha=0.05)
ax.grid()
ax.set_xlabel('D [µm]')
ax.set_ylabel('V [µm/s]')
    
plt.show()

D_plot = np.linspace(1, 5000, 500)

#### Double Expo
V_popt_2exp, V_pcov_2exp = optimize.curve_fit(doubleExpo, all_D, all_V, 
                       p0 = [1000, 50, 100, 1000], 
                       bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
V_fit_2exp = doubleExpo(D_plot, *V_popt_2exp)
label_2exp = r'$\bf{A \cdot exp(-x/k_1) + B \cdot exp(-x/k_2)}$' + '\n'
label_2exp += '$A$ = {:.2e} | $k_1$ = {:.2f}\n$B$ = {:.2f} | $k_2$ = {:.2e}'.format(*V_popt_2exp)

#### Power Law
V_popt_pL, V_pcov_pL = optimize.curve_fit(powerLaw, all_D, all_V, 
                       p0 = [1000, -2], 
                       bounds=([0, -10], [np.inf, 0]))
V_fit_pL = powerLaw(D_plot, *V_popt_pL)
V_label_pL = r'$\bf{A \cdot x^k}$' + '\n'
V_label_pL += '$A$ = {:.2e} | $k$ = {:.2f}'.format(*V_popt_pL)

expected_medV = powerLaw(all_medD, *V_popt_pL)
ratio_fitV = all_medV/expected_medV
high_cut = 1.7
low_cut = 0.5

fig21, ax21 = plt.subplots(1,1)
fig, ax = fig21, ax21
ax.plot(all_medD, ratio_fitV, ls='', marker='.', alpha=0.3)
ax.axhline(high_cut, color = 'k', ls='-.')
ax.axhline(low_cut, color = 'k', ls='-.')
plt.show()

tracks_data_f2 = []
removed_tracks = []
for i, track in enumerate(tracks_data_f1):
    if (ratio_fitV[i] > low_cut) and (ratio_fitV[i] < high_cut):
        tracks_data_f2.append(track)
    else:
        removed_tracks.append(track)
        
all_D = np.concat([track['D'] for track in tracks_data_f2])
all_V = np.concat([track['V'] for track in tracks_data_f2])
all_removedD = np.concat([track['D'] for track in removed_tracks])
all_removedV = np.concat([track['V'] for track in removed_tracks])   

     
fig22, ax22 = plt.subplots(1, 1, figsize = (8, 6))
fig, ax = fig22, ax22
# ax.plot(all_medD, all_medV, ls='', marker='.')
ax.plot(all_D, all_V, ls='', marker='.', alpha=0.05)
ax.plot(all_removedD, all_removedV, ls='', marker='.', alpha=0.05)
ax.grid()
ax.set_xlabel('D [µm]')
ax.set_ylabel('V [µm/s]')
        
# %%%% Final fits

D_plot = np.linspace(1, 5000, 500)

# Double Expo
V_popt_2exp, V_pcov_2exp = optimize.curve_fit(doubleExpo, all_D, all_V, 
                       p0 = [1000, 50, 100, 1000], 
                       bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
V_fit_2exp = doubleExpo(D_plot, *V_popt_2exp)
label_2exp = r'$\bf{A \cdot exp(-x/k_1) + B \cdot exp(-x/k_2)}$' + '\n'
label_2exp += '$A$ = {:.2e} | $k_1$ = {:.2f}\n$B$ = {:.2f} | $k_2$ = {:.2e}'.format(*V_popt_2exp)

#### Power Law
V_popt_pL, V_pcov_pL = optimize.curve_fit(powerLaw, all_D, all_V, 
                       p0 = [1000, -2], 
                       bounds=([0, -10], [np.inf, 0]))
V_fit_pL = powerLaw(D_plot, *V_popt_pL)
V_label_pL = r'$\bf{A \cdot x^k}$' + '\n'
V_label_pL += '$A$ = {:.2e} | $k$ = {:.2f}'.format(*V_popt_pL)

# def powerLawShifted(x, A, k, x0):
#     return(A*((x-x0)**k))

# V_popt_pLS, V_pcov_pLS = optimize.curve_fit(powerLawShifted, all_D, all_V, 
#                        p0 = [1e6, -7, -600], 
#                        bounds=([0, -10, -1000], [np.inf, 0, 0]),
#                        maxfev = 10000) # 
# V_fit_pLS = powerLawShifted(D_plot, *V_popt_pLS)
# V_label_pLS = r'$\bf{A \cdot (x - x_0)^k}$' + '\n'
# V_label_pLS += '$A$ = {:.2e} | $k$ = {:.2f}\n$x_0$ = {:.2f}'.format(*V_popt_pLS)


fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes[0]
ax.plot(all_D, all_V, ls='', marker='.', alpha=0.01)
ax.plot(D_plot, V_fit_2exp, 'r-', label = label_2exp)
ax.plot(D_plot, V_fit_pL, ls='-', color='darkorange', label = V_label_pL)
# ax.plot(D_plot, V_fit_pLS, 'c-', label = V_label_pLS)
ax.grid()
ax.set_xlim([0, 1000])
ax.set_ylim([0, 5])
ax.set_xlabel('d [µm]')
ax.set_ylabel('v [µm/s]')
# ax.legend()

ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(all_D, all_V, ls='', marker='.', alpha=0.01)
ax.plot(D_plot, V_fit_2exp, 'r-', label = label_2exp)
ax.plot(D_plot, V_fit_pL, ls='-', color='darkorange', label = V_label_pL)
# ax.plot(D_plot, V_fit_pLS, 'c-', label = V_label_pLS)
ax.grid()
ax.set_xlim([50, 5000])
ax.set_ylim([0.01, 10])
ax.set_xlabel('d [µm]')
ax.set_ylabel('v [µm/s]')
ax.legend(loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

fig.suptitle("Joseph's calibration data")
fig.tight_layout()
plt.show()

# V_popt_2exp = [1138.18, 37.2887, 2.01482, 296.243]
# V_popt_2exp = [2683, 30.33, 2.440, 314.54]
V_popt_2exp = [255.07,  45.81,   1.42, 401.66]
# V_popt_pL = [310320, -2.19732]


# %%% Plotting Maribel's calibration

mainDir = "C:/Users/josep/Desktop/Seafile/DownloadedFromSeafile"
fileName = 'dist-speed_ALL.txt'
filePath = os.path.join(mainDir, fileName)

df = pd.read_csv(filePath, sep=' ', names=['Id', 'D', 'V'])
all_D = df['D'].values
all_V = df['V'].values

D_plot = np.linspace(1, 5000, 500)

def doubleExpo(x, A, k1, B, k2):
    return(A*np.exp(-x/k1) + B*np.exp(-x/k2))

V_popt_2exp, V_pcov_2exp = optimize.curve_fit(doubleExpo, all_D, all_V, 
                       p0 = [1000, 50, 100, 1000], 
                       bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
V_fit_2exp = doubleExpo(D_plot, *V_popt_2exp)
label_2exp = r'$\bf{A \cdot exp(-x/k_1) + B \cdot exp(-x/k_2)}$' + '\n'
label_2exp += '$A$ = {:.2e} | $k_1$ = {:.2f}\n$B$ = {:.2f} | $k_2$ = {:.2e}'.format(*V_popt_2exp)


def powerLaw(x, A, k):
    return(A*(x**k))

V_popt_pL, V_pcov_pL = optimize.curve_fit(powerLaw, all_D, all_V, 
                       p0 = [1000, -2], 
                       bounds=([0, -10], [np.inf, 0]))
V_fit_pL = powerLaw(D_plot, *V_popt_pL)
V_label_pL = r'$\bf{A \cdot x^k}$' + '\n'
V_label_pL += '$A$ = {:.2e} | $k$ = {:.2f}'.format(*V_popt_pL)


# def powerLawShifted(x, A, k, x0):
#     return(A*((x-x0)**k))

# V_popt_pLS, V_pcov_pLS = optimize.curve_fit(powerLawShifted, all_D, all_V, 
#                        p0 = [1e6, -2.5, 0], 
#                        maxfev = 10000) # bounds=([0, -10, -200], [np.inf, 0, 200])
# V_fit_pLS = powerLawShifted(D_plot, *V_popt_pLS)
# V_label_pLS = r'$\bf{A \cdot (x - x_0)^k}$' + '\n'
# V_label_pLS += '$A$ = {:.2e} | $k$ = {:.2f}\n$x_0$ = {:.2f}'.format(*V_popt_pLS)



fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes[0]
ax.plot(all_D, all_V, ls='', marker='.', alpha=0.3)
ax.plot(D_plot, V_fit_2exp, 'r-', label = label_2exp)
ax.plot(D_plot, V_fit_pL, ls='-', color='darkorange', label = V_label_pL)
# ax.plot(D_plot, V_fit_pLS, 'c-', label = V_label_pLS)
ax.grid()
ax.set_xlim([0, 1000])
ax.set_ylim([0, 30])
ax.set_xlabel('d [µm]')
ax.set_ylabel('v [µm/s]')
# ax.legend()

ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(all_D, all_V, ls='', marker='.', alpha=0.3)
ax.plot(D_plot, V_fit_2exp, 'r-', label = label_2exp)
ax.plot(D_plot, V_fit_pL, ls='-', color='darkorange', label = V_label_pL)
# ax.plot(D_plot, V_fit_pLS, 'c-', label = V_label_pLS)
ax.grid()
ax.set_xlim([50, 5000])
ax.set_ylim([0.1, 1000])
ax.set_xlabel('d [µm]')
ax.set_ylabel('v [µm/s]')
ax.legend(loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

fig.suptitle("Maribel's calibration data")
fig.tight_layout()
plt.show()