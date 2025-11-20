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
import random
import matplotlib


from scipy import interpolate, optimize
from skimage import io, filters, exposure, measure, transform, util, color
from matplotlib.gridspec import GridSpec
from datetime import date, datetime
from copy import deepcopy

#### Local Imports

import GraphicStyles as gs
import UtilityFunctions as ufun

# %%

t = np.arange(0, 1.1, .1)
x = np.sin(2*np.pi*t)
y = np.cos(2*np.pi*t)
tck, u = interpolate.splprep([x, y], s=0)
unew = np.arange(0, 1.01, 0.01)
out = interpolate.splev(unew, tck)
plt.figure()
plt.plot(x, y, 'x', out[0], out[1], np.sin(2*np.pi*unew), np.cos(2*np.pi*unew), x, y, 'b')
plt.legend(['Linear', 'Cubic Spline', 'True'])
plt.axis([-1.05, 1.05, -1.05, 1.05])
plt.title('Spline of parametrically-defined curve')
plt.show()

# %% 2. Subfunctions

def doubleExpo(x, A, k1, B, k2):
    return(A*np.exp(-x/k1) + B*np.exp(-x/k2))

def powerLaw(x, A, k):
    return(A*(x**k))

def powerLawShifted(x, A, k, x0):
    return(A*((x-x0)**k))

def tracerTracks_pretreatment(all_tracks,
                        SCALE, FPS, 
                        MagX, MagY, MagR, Rb):

    tracks_data = []
    
    for i, track in enumerate(all_tracks):
        #### Conversion in um and sec
        T = track[:, 0] * (1/FPS)
        X = track[:, 1] #* SCALE
        Y = track[:, 2] #* SCALE
        tracks_data.append({'T':T, 'X':X, 'Y':Y})
        #### Origin as the magnet center
        X2, Y2 = X-MagX, MagY-Y
        # NB: inversion of Y so that the plt trajectories look like the Fiji ones
        medX2, medY2 = np.median(X2), np.median(Y2)
        tracks_data[i].update({'X2':X2, 'Y2':Y2,
                               'medX2':medX2, 'medY2':medY2})

        #### Compute splines
        # Make spline
        # spline = interpolate.make_interp_spline(T, np.array([X2, Y2]).T, k=2)
        # XY_spline = spline(T).T
        # X3, Y3 = XY_spline[0], XY_spline[1]
        
        spline, u = interpolate.make_splprep([X2, Y2], s=50)
        X3, Y3 = spline(np.linspace(0, 1, len(T)))
        
        tracks_data[i].update({'X3':X3, 'Y3':Y3,
                               })
        # spline_V = spline_D.derivative(nu=1)
        # # Compute V and chose definition
        # V_spline = np.abs(spline_V(T))
        # # V_savgol = dxdt(D, T, kind="savitzky_golay", 
        # #                 left=10, right=10, order=3)
        # # V_kalman = dxdt(D, T, kind="kalman", alpha=1)
        # V = V_spline
        # #
        # medD, medV = np.median(D), np.median(V)
        # tracks_data[i].update({'D':D, 'V':V, 
        #                        'medD':medD, 'medV':medV,
        #                        })
        
    return(tracks_data)

# %% 3. Analyse tracers

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
popt_2exp, pcov_2exp = optimize.curve_fit(doubleExpo, all_D, all_V, 
                       p0 = [1000, 50, 100, 1000], 
                       bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
V_fit_2exp = doubleExpo(D_plot, *popt_2exp)
label_2exp = r'$\bf{A \cdot exp(-x/k_1) + B \cdot exp(-x/k_2)}$' + '\n'
label_2exp += '$A$ = {:.2e} | $k_1$ = {:.2f}\n$B$ = {:.2f} | $k_2$ = {:.2e}'.format(*popt_2exp)

#### Power Law
popt_pL, pcov_pL = optimize.curve_fit(powerLaw, all_D, all_V, 
                       p0 = [1000, -2], 
                       bounds=([0, -10], [np.inf, 0]))
V_fit_pL = powerLaw(D_plot, *popt_pL)
label_pL = r'$\bf{A \cdot x^k}$' + '\n'
label_pL += '$A$ = {:.2e} | $k$ = {:.2f}'.format(*popt_pL)

expected_medV = powerLaw(all_medD, *popt_pL)
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
popt_2exp, pcov_2exp = optimize.curve_fit(doubleExpo, all_D, all_V, 
                       p0 = [1000, 50, 100, 1000], 
                       bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
V_fit_2exp = doubleExpo(D_plot, *popt_2exp)
label_2exp = r'$\bf{A \cdot exp(-x/k_1) + B \cdot exp(-x/k_2)}$' + '\n'
label_2exp += '$A$ = {:.2e} | $k_1$ = {:.2f}\n$B$ = {:.2f} | $k_2$ = {:.2e}'.format(*popt_2exp)

#### Power Law
popt_pL, pcov_pL = optimize.curve_fit(powerLaw, all_D, all_V, 
                       p0 = [1000, -2], 
                       bounds=([0, -10], [np.inf, 0]))
V_fit_pL = powerLaw(D_plot, *popt_pL)
label_pL = r'$\bf{A \cdot x^k}$' + '\n'
label_pL += '$A$ = {:.2e} | $k$ = {:.2f}'.format(*popt_pL)

expected_medV = powerLaw(all_medD, *popt_pL)
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
popt_2exp, pcov_2exp = optimize.curve_fit(doubleExpo, all_D, all_V, 
                       p0 = [1000, 50, 100, 1000], 
                       bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
V_fit_2exp = doubleExpo(D_plot, *popt_2exp)
label_2exp = r'$\bf{A \cdot exp(-x/k_1) + B \cdot exp(-x/k_2)}$' + '\n'
label_2exp += '$A$ = {:.2e} | $k_1$ = {:.2f}\n$B$ = {:.2f} | $k_2$ = {:.2e}'.format(*popt_2exp)

#### Power Law
popt_pL, pcov_pL = optimize.curve_fit(powerLaw, all_D, all_V, 
                       p0 = [1000, -2], 
                       bounds=([0, -10], [np.inf, 0]))
V_fit_pL = powerLaw(D_plot, *popt_pL)
label_pL = r'$\bf{A \cdot x^k}$' + '\n'
label_pL += '$A$ = {:.2e} | $k$ = {:.2f}'.format(*popt_pL)

# def powerLawShifted(x, A, k, x0):
#     return(A*((x-x0)**k))

# popt_pLS, pcov_pLS = optimize.curve_fit(powerLawShifted, all_D, all_V, 
#                        p0 = [1e6, -7, -600], 
#                        bounds=([0, -10, -1000], [np.inf, 0, 0]),
#                        maxfev = 10000) # 
# V_fit_pLS = powerLawShifted(D_plot, *popt_pLS)
# label_pLS = r'$\bf{A \cdot (x - x_0)^k}$' + '\n'
# label_pLS += '$A$ = {:.2e} | $k$ = {:.2f}\n$x_0$ = {:.2f}'.format(*popt_pLS)


fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes[0]
ax.plot(all_D, all_V, ls='', marker='.', alpha=0.01)
ax.plot(D_plot, V_fit_2exp, 'r-', label = label_2exp)
ax.plot(D_plot, V_fit_pL, ls='-', color='darkorange', label = label_pL)
# ax.plot(D_plot, V_fit_pLS, 'c-', label = label_pLS)
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
ax.plot(D_plot, V_fit_pL, ls='-', color='darkorange', label = label_pL)
# ax.plot(D_plot, V_fit_pLS, 'c-', label = label_pLS)
ax.grid()
ax.set_xlim([50, 5000])
ax.set_ylim([0.01, 100])
ax.set_xlabel('d [µm]')
ax.set_ylabel('v [µm/s]')
ax.legend(loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

fig.suptitle("Joseph's calibration data")
fig.tight_layout()
plt.show()

# popt_2exp = [1138.18, 37.2887, 2.01482, 296.243]
# popt_2exp = [2683, 30.33, 2.440, 314.54]
popt_2exp = [255.07,  45.81,   1.42, 401.66]
# popt_pL = [310320, -2.19732]


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

popt_2exp, pcov_2exp = optimize.curve_fit(doubleExpo, all_D, all_V, 
                       p0 = [1000, 50, 100, 1000], 
                       bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
V_fit_2exp = doubleExpo(D_plot, *popt_2exp)
label_2exp = r'$\bf{A \cdot exp(-x/k_1) + B \cdot exp(-x/k_2)}$' + '\n'
label_2exp += '$A$ = {:.2e} | $k_1$ = {:.2f}\n$B$ = {:.2f} | $k_2$ = {:.2e}'.format(*popt_2exp)


def powerLaw(x, A, k):
    return(A*(x**k))

popt_pL, pcov_pL = optimize.curve_fit(powerLaw, all_D, all_V, 
                       p0 = [1000, -2], 
                       bounds=([0, -10], [np.inf, 0]))
V_fit_pL = powerLaw(D_plot, *popt_pL)
label_pL = r'$\bf{A \cdot x^k}$' + '\n'
label_pL += '$A$ = {:.2e} | $k$ = {:.2f}'.format(*popt_pL)


# def powerLawShifted(x, A, k, x0):
#     return(A*((x-x0)**k))

# popt_pLS, pcov_pLS = optimize.curve_fit(powerLawShifted, all_D, all_V, 
#                        p0 = [1e6, -2.5, 0], 
#                        maxfev = 10000) # bounds=([0, -10, -200], [np.inf, 0, 200])
# V_fit_pLS = powerLawShifted(D_plot, *popt_pLS)
# label_pLS = r'$\bf{A \cdot (x - x_0)^k}$' + '\n'
# label_pLS += '$A$ = {:.2e} | $k$ = {:.2f}\n$x_0$ = {:.2f}'.format(*popt_pLS)



fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes[0]
ax.plot(all_D, all_V, ls='', marker='.', alpha=0.3)
ax.plot(D_plot, V_fit_2exp, 'r-', label = label_2exp)
ax.plot(D_plot, V_fit_pL, ls='-', color='darkorange', label = label_pL)
# ax.plot(D_plot, V_fit_pLS, 'c-', label = label_pLS)
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
ax.plot(D_plot, V_fit_pL, ls='-', color='darkorange', label = label_pL)
# ax.plot(D_plot, V_fit_pLS, 'c-', label = label_pLS)
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