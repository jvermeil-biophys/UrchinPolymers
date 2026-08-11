# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:33:46 2026

@author: Joseph Vermeil

UtilityFunctions.py - contains all kind of small functions used by CortExplore programs, 
to be imported with "import UtilityFunctions as ufun" and call with "ufun.my_function".
Joseph Vermeil, 2026

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

# %% Imports

import os
import re
import time

import numpy as np
import pandas as pd
import trackpy as tp
import skimage as skm
import seaborn as sns
import matplotlib as mpl
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

import shapely
from shapely.ops import polylabel
from shapely.plotting import plot_polygon, plot_points # , plot_line

from PIL import Image, ImageDraw
from scipy import signal # stats #, optimize, interpolate, 

import Libs.PlotMaker as pm
import Libs.UrchinPaths as up
import Libs.CalibrationData as cd
import Libs.UtilityFunctions as ufun
import Libs.ToolboxCytoplasmAnalysis as tbca


# %% Film BF

mainDir = 'C://Users//josep//Desktop//Seafile//DownloadedFromSeafile//IntraCellTracking//26-06-19_FastAcq_BF'

# %%% 1. DDM 

# %%%% 1.1 Settings

mainDir = 'C://Users//josep//Desktop//Seafile//DownloadedFromSeafile//IntraCellTracking//26-06-19_FastAcq_BF'
srcDir = os.path.join(mainDir, 'Crops')
tifNames = ['FilmBF_fastAcq_4000f_200Hz_C3_Crop.tif', 'FilmBF_fastAcq_4000f_10Hz_C3_Crop.tif']
tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]

dstDir = os.path.join(mainDir, 'DDM_results')

PixPerUm = cd.PixPerUm_40X_Leica
frequencies = [200, 10]
nbimages = 4000
pointsPerDecade = 15
maxNCouples = 100 #10 for fast evaluation, 300 for accurate analysis

ddmFileNames = []
dtFileNames = []
for fN, f in zip(tifNames, frequencies):
    ddmFileNames.append('_'.join(fN.split('_')[:-1]) + f'_Nc{maxNCouples:.0f}_DDM.npy')
    dtFileNames.append('_'.join(fN.split('_')[:-1]) + f'_Nc{maxNCouples:.0f}_dt.npy')


# %%%% 1.2 Compute

idts = tbca.logSpaced(nbimages, pointsPerDecade)
dts = [idts/float(freq) for freq in frequencies]

DDMs = []
for p in tifPaths:
    print(f'\n\nAnalyzing {os.path.split(p)}...')
    DDM = tbca.ddm(tbca.ImageStack(p), idts, maxNCouples)
    DDMs.append(DDM)
    
for ddmN, dtN, D, dt in zip(ddmFileNames, dtFileNames, DDMs, dts):
    np.save(os.path.join(dstDir, ddmN), D)
    np.save(os.path.join(dstDir, dtN), dt)


# %%%% 1.3 Merge

srcDir = os.path.join(mainDir, 'DDM_results')
DDMs = [np.load(os.path.join(srcDir, fN)) for fN in ddmFileNames]
dts = [np.load(os.path.join(srcDir, fN)) for fN in dtFileNames]


frequencies = [200, 10]

DDMMerge, dtMerge = tbca.mergeDDM(DDMs, dts, frequencies)

# %%%% 1.4 Plot


N_pix = 256
L_um = N_pix*PixPerUm
dq = 2*np.pi / L_um
qmax = 10


for k in range(len(DDMs)):
    DDM_plot = DDMs[k]
    dt_plot = dts[k]
    QQ = np.arange(1, 1+len(DDM_plot[0,:]))*dq

    # Test the merging for `q=100`
    Q_plot = [10, 30, 100]
    for q in Q_plot:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.plot(dts[0], DDMs[0][:,q],'o', label='200 Hz')
        ax.plot(dts[1], DDMs[1][:,q],'s', label='10 Hz')
        ax.plot(dtMerge, DDMMerge[:,q], label='Merged')
        # ax.plot(dtMerge, DDM_plot[:,100]*2, 'o', label='Shifted merged')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(r'$\mathcal{D}$')
        ax.set_xlabel(r'$\Delta t\,(s)$')
        # ax.axvline(dts[0][boundary], color='r')
        # ax.axvline(dts[0][boundary+overlap0], color='b')
        # ax.axvline(dts[1][overlap1], color='g')
        ax.legend(loc='lower right')
        ax.set_title(f'q = {q:.0f}')
        plt.show()
    
    (Ndt, Nq) = DDM_plot.shape
    fig, axes = plt.subplots(2, 1, figsize = (5, 8))
    ax = axes[0]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$q\ (\mu m^{-1})$')
    ax.set_ylabel('$D$')
    for i in range(0, Ndt, 10):
        ax.plot(QQ, DDM_plot[i,:], marker='.', ls='',
                color = mpl.cm.autumn(i/Ndt))
        ax.axvline(dq, color='gray', ls='-', alpha=0.7)
        ax.axvline(qmax, color='gray', ls='-', alpha=0.7)
        
    fig.colorbar(plt.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=np.min(dt_plot), vmax=np.max(dt_plot)), 
                                       cmap="autumn"),
                 ax=ax, label=r"$\Delta t$")
        
    
    ax = axes[1]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\Delta t\ (s)$')
    ax.set_ylabel('$D$')
    for j in range(0, Nq, 25):
        ax.plot(dt_plot, DDM_plot[:,j], marker='.', ls='',
                color = mpl.cm.winter(j/Nq))
        
    fig.colorbar(plt.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=np.min(QQ), vmax=np.max(QQ)), 
                                       cmap="winter"),
                 ax=ax, label="$q$")
    
    plt.show()
    
    # Now we can export the merged results (adapt to the folder you want to save to)
    
    # np.save('DDM_fastAcq_C3.npy', DDM_plot)
    # np.save('dt_fastAcq_C3.npy', dtMerge)
    
    ApB_est = np.median(DDM_plot[-4:,:], axis=0)
    B_est = np.median(DDM_plot[:6,:], axis=0)
    B_best = 4.1e10
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax = axes[0]
    ax.plot(QQ, ApB_est, 'r.')
    ax.axvline(dq, color='gray', ls='-', alpha=0.7)
    ax.axvline(qmax, color='gray', ls='-', alpha=0.7)
    ax = axes[1]
    ax.plot(QQ, B_est,'k.')
    ax.axvline(dq, color='gray', ls='-', alpha=0.7)
    ax.axvline(qmax, color='gray', ls='-', alpha=0.7)
    ax = axes[2]
    ax.plot(QQ, ApB_est-B_est,'b.')
    ax.plot(QQ, ApB_est-B_best,'g.')
    ax.axvline(dq, color='gray', ls='-', alpha=0.7)
    ax.axvline(qmax, color='gray', ls='-', alpha=0.7)
    
    plt.show()


# %% New film

mainDir = 'C:\\Users\\Joseph\\Desktop\\IntraCellTracking\\26-07-29_FastAcq_NBYolk-Fecondation'
srcDir = os.path.join(mainDir, 'Crops')
tifName = '26-07-29_PostF_2min_Pos11_10fps_Texp100ms1_CSU642_crop.tif'
tifPath = os.path.join(srcDir, tifName)

xmlName = tifName.split('.')[0] + '_PyTracks.xml'
xmlPath = os.path.join(srcDir, xmlName)

dstDir = os.path.join(mainDir, 'SPT_results')

tbca.pretreatAndTrack2(tifPath, dstDir)
Tracks = tbca.importTrackMateTracks(xmlPath)

# %%% Analyse tracks - Test pipeline

mainDir = 'C:\\Users\\Joseph\\Desktop\\IntraCellTracking\\26-07-29_FastAcq_NBYolk-Fecondation'

tifDir = os.path.join(mainDir, 'Crops')
tifName = '26-07-29_PostF_2min_Pos11_10fps_Texp100ms1_CSU642_crop.tif'
tifPath = os.path.join(tifDir, tifName)

xmlDir = os.path.join(mainDir, 'SPT_results')
xmlName = tifName.split('.')[0] + '_PyTracks.xml'
xmlPath = os.path.join(xmlDir, xmlName)

SCALE = 1/cd.SCALE_60X_W1
FPS = 10

Tracks = tbca.importTrackMateTracks(xmlPath)
Np = len(Tracks)
    
column_names = ['frame', 'x', 'y', 'particle']
all_tracks = []
for i, track in enumerate(Tracks):
    nT = len(track)
    if nT >= 30:
        track = np.concat((track, np.ones((len(track[:,0]), 1), dtype=int) * (i+1)), axis = 1)
        track[:,0] = track[:,0].astype(int) + 1
        all_tracks.append(track)

concat_tracks = np.concat(all_tracks, axis = 0)
df = pd.DataFrame({column_names[k] : concat_tracks[:,k] for k in range(len(column_names))})


# %%%

#### Run msd
res_emsd = tp.motion.emsd(df, SCALE, FPS, max_lagtime=40).reset_index()
T, MSD = res_emsd['lagt'], res_emsd['msd']

parms, results = ufun.fitLineHuber(T, MSD, with_intercept = False)
D = parms.values[0]/4

parms, results = ufun.fitLineHuber(np.log(T), np.log(MSD), with_intercept = True)
b, a = parms
k_full = a
D_full = np.exp(b)/4

print(k_full, D_full)

parms, results = ufun.fitLineHuber(np.log(T[:4]), np.log(MSD[:4]), with_intercept = True)
b, a = parms
k_fast = a
D_fast = np.exp(b)/4

print(k_fast, D_fast)

parms, results = ufun.fitLineHuber(np.log(T[-15:]), np.log(MSD[-15:]), with_intercept = True)
b, a = parms
k_slow = a
D_slow = np.exp(b)/4

print(k_slow, D_slow)

Tc = (D_fast/D_slow)**(1/(k_slow-k_fast))

# %%%

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_xscale('log')
ax.set_yscale('log')

Xp = np.array([1e-3, 1e3])

ax.plot(T, MSD, 'wo', mec='k')
# ax.plot(Xp, 4*D_full*(Xp**k_full), ls='-', color=pm.cL_Set21[0], mec='k', label='Full curve')
ax.plot(Xp, 4*D_fast*(Xp**k_fast), ls='-', color=pm.cL_Set21[1], mec='k', 
        label=f'First 4 pts\n$\\alpha$={k_fast:.2f}')
ax.plot(Xp, 4*D_slow*(Xp**k_slow), ls='-', color=pm.cL_Set21[2], mec='k', 
        label=f'Last 15 pts\n$\\alpha$={k_slow:.2f}')
ax.axvline(Tc, color='gray', lw=1.5, label=f'$T_c$ = {Tc:.2f}')
ax.legend()
ax.grid()
ax.set_xlim([0.5e-1, 2e1])
ax.set_ylim([0.5e-2, 2e0])
ax.set_ylabel('MSD (um²)')
ax.set_xlabel('T (s)')
plt.show()

# %%%
#### Run imsd -> Might be useful for SEM computation
res_imsd = tp.motion.imsd(df, SCALE, FPS, max_lagtime=30).reset_index()

# %%%

fig, ax = plt.subplots(1, 1)
ax.set_xscale('log')
ax.set_yscale('log')
for p in range(1, Np+1):
    T, MSD = res_imsd['lag time [s]'].values, res_imsd[p].values
    ax.plot(T, MSD, lw=0.5)
ax.set_xlabel('$\\Delta t$ (s)')
ax.set_ylabel('MSD ($\\mu m^2$)')
plt.show()

list_D = []
list_k = []

T= res_imsd['lag time [s]'].values
for p in range(1, Np+1):
    MSD = res_imsd[p].values
    try:
        parms, results = ufun.fitLineHuber(np.log(T), np.log(MSD), with_intercept = True)
        b, a = parms
        k_nl = a
        D_nl = np.exp(b)/4
        list_k.append(k_nl)
        list_D.append(D_nl)
    except:
        pass


# for p in range(1, Np+1):
#     T, MSD = res_imsd['lag time [s]'].values[-10:], res_imsd[p].values[-10:]
#     parms, results = ufun.fitLineHuber(np.log(T), np.log(MSD), with_intercept = True)
#     b, a = parms
#     k_nl = a
#     D_nl = np.exp(b)/4
#     if D_nl < 0.2 and D_nl > 0 and k_nl > 0:
#        list_k.append(k_nl)
#        list_D.append(D_nl)
    
# %%%

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
ax = axes[0]
# sns.swarmplot(ax=ax, y=list_D)
ax.hist(list_D, bins=60, color='gray', zorder=3)
# ax.set_xlim([-0.005, 0.18])
ax.set_xlabel('D ($\\mu m^2/s$)')
ax.set_ylabel('N')

ax = axes[1]
# sns.swarmplot(ax=ax, y=list_k)
ax.hist(list_k, bins=60, color='gray', zorder=3)
ax.set_xlabel('exponent $\\alpha$')
ax.set_ylabel('N')

ax = axes[2]
sns.scatterplot(ax=ax, x=list_k, y=list_D, alpha=0.01)
ax.set_xlabel('exponent $\\alpha$')
ax.set_ylabel('D ($\\mu m^2/s$)')

plt.show()




# %% ------------------------------------------



# %% First test

#### Arguments

# tifPath = "F:\\WorkingData\\26-06-19_FastAcq\\FilmBF_fastAcq_4000f_10Hz_C1.tif"
# tifPath = "C:\\Users\\josep\\Desktop\\Seafile\\AnalysisPulls\\" + \
#           "26-06-19_FastAcq\\FilmBF_fastAcq_4000f_10Hz_C1.tif"
srcDir = "C:\\Users\\josep\\Desktop\\Seafile\\AnalysisPulls\\" + \
          "26-06-10_Test-NileBlueYolk\\M1_40x-WI\\"
dstDir = srcDir

# tifName = "26-06-10_TestNileBlueYolk_C2_10fps_1min_L50p.tif"
tifName = "26-06-10_TestNileBlueYolk_C1_4fps_5min_L20p.tif"
tifPath = os.path.join(srcDir, tifName)   

xmlName = tifName.split('.')[0] + '_PyTracks.xml'
xmlPath = os.path.join(srcDir, xmlName)
    
# PtImage = PretreatImageForTrackMate(tifPath)
# tif_file = ij.py.to_java(PtImage)

tbca.pretreatAndTrack(tifPath, dstDir)
Tracks = tbca.importTrackMateTracks(xmlPath)


# %%% Analyse tracks - Test pipeline

# tifPath = "F:\\WorkingData\\26-06-19_FastAcq\\FilmBF_fastAcq_4000f_10Hz_C1.tif"
# tifPath = "C:\\Users\\josep\\Desktop\\Seafile\\AnalysisPulls\\" + \
#           "26-06-19_FastAcq\\FilmBF_fastAcq_4000f_10Hz_C1.tif"
srcDir = "C:\\Users\\josep\\Desktop\\Seafile\\AnalysisPulls\\" + \
          "26-06-10_Test-NileBlueYolk\\M1_40x-WI\\"
dstDir = srcDir

tifName = "26-06-10_TestNileBlueYolk_C2_10fps_1min_L50p.tif"
# tifName = "26-06-10_TestNileBlueYolk_C1_4fps_5min_L20p.tif"
tifPath = os.path.join(srcDir, tifName)   

xmlName = tifName.split('.')[0] + '_PyTracks.xml'
xmlPath = os.path.join(srcDir, xmlName)

SCALE = cd.SCALE_40X_X1
FPS = 10

Tracks = tbca.importTrackMateTracks(xmlPath)
Np = len(Tracks)
    
column_names = ['frame', 'x', 'y', 'particle']
all_tracks = []
for i, track in enumerate(Tracks):
    nT = len(track)
    if nT >= 30:
        track = np.concat((track, np.ones((len(track[:,0]), 1), dtype=int) * (i+1)), axis = 1)
        track[:,0] = track[:,0].astype(int) + 1
        all_tracks.append(track)

concat_tracks = np.concat(all_tracks, axis = 0)
df = pd.DataFrame({column_names[k] : concat_tracks[:,k] for k in range(len(column_names))})

# %%% Compute Self-ISF

from scipy.special import jv

dt = 1
q = 1

def dr(track, t0, dt):
    return(((track[t0,1]-track[t0+dt,1])**2 + (track[t0,2]-track[t0+dt,2])**2)**0.5)

def Fs(q, dt, Tracks):
    F = np.sum([np.sum([jv(0, q * dr(track, t0, dt)) for track in Tracks]) for t0 in range(31-dt)])
    return(F)

Fs(q, dt, Tracks)

DT = np.arange(20)

Fs_05 = np.array([Fs(0.5, dt, Tracks) for dt in DT])
Fs_1 = np.array([Fs(1, dt, Tracks) for dt in DT])
Fs_2 = np.array([Fs(2, dt, Tracks) for dt in DT])
Fs_4 = np.array([Fs(4, dt, Tracks) for dt in DT])
# Fs_8 = np.array([Fs(8, dt, Tracks) for dt in DT])

Fs_05 /= Fs_05[0]
Fs_1 /= Fs_1[0]
Fs_2 /= Fs_2[0]
Fs_4 /= Fs_4[0]

fig, ax = plt.subplots(1, 1)
# ax.set_xscale('log')
# ax.set_yscale('log')
ax.plot((0.5**2)*DT/FPS, Fs_05, label = 'q = 0.5')
ax.plot((1**2)*DT/FPS, Fs_1, label = 'q = 1')
ax.plot((2**2)*DT/FPS, Fs_2, label = 'q = 2')
ax.plot((4**2)*DT/FPS, Fs_4, label = 'q = 4')
# ax.plot(DT/FPS, Fs_8, label = 'q = 8')
# ax.set_xlim([1e-2, 5e0])
ax.grid()
ax.legend()
plt.show()


# %%%

import trackpy as tp

#### Run msd
res_emsd = tp.motion.emsd(df, SCALE, FPS, max_lagtime=40).reset_index()
T, MSD = res_emsd['lagt'], res_emsd['msd']

parms, results = ufun.fitLineHuber(T, MSD, with_intercept = False)
D = parms.values[0]/4

parms, results = ufun.fitLineHuber(np.log(T), np.log(MSD), with_intercept = True)
b, a = parms
k_full = a
D_full = np.exp(b)/4

print(k_full, D_full)

parms, results = ufun.fitLineHuber(np.log(T[:4]), np.log(MSD[:4]), with_intercept = True)
b, a = parms
k_fast = a
D_fast = np.exp(b)/4

print(k_fast, D_fast)

parms, results = ufun.fitLineHuber(np.log(T[-15:]), np.log(MSD[-15:]), with_intercept = True)
b, a = parms
k_slow = a
D_slow = np.exp(b)/4

print(k_slow, D_slow)

Tc = (D_fast/D_slow)**(1/(k_slow-k_fast))

# %%%

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_xscale('log')
ax.set_yscale('log')

Xp = np.array([1e-3, 1e3])

ax.plot(T, MSD, 'wo', mec='k')
# ax.plot(Xp, 4*D_full*(Xp**k_full), ls='-', color=pm.cL_Set21[0], mec='k', label='Full curve')
ax.plot(Xp, 4*D_fast*(Xp**k_fast), ls='-', color=pm.cL_Set21[1], mec='k', 
        label=f'First 4 pts\n$\\alpha$={k_fast:.2f}')
ax.plot(Xp, 4*D_slow*(Xp**k_slow), ls='-', color=pm.cL_Set21[2], mec='k', 
        label=f'Last 15 pts\n$\\alpha$={k_slow:.2f}')
ax.axvline(Tc, color='gray', lw=1.5, label=f'$T_c$ = {Tc:.2f}')
ax.legend()
ax.grid()
ax.set_xlim([0.5e-1, 2e1])
ax.set_ylim([0.5e-2, 2e0])
plt.show()

# %%%
#### Run imsd -> Might be useful for SEM computation
res_imsd = tp.motion.imsd(df, SCALE, FPS, max_lagtime=30).reset_index()

# %%%

fig, ax = plt.subplots(1, 1)
ax.set_xscale('log')
ax.set_yscale('log')
for p in range(1, Np+1):
    T, MSD = res_imsd['lag time [s]'].values, res_imsd[p].values
    ax.plot(T, MSD, lw=0.5)
ax.set_xlabel('$\\Delta t$ (s)')
ax.set_ylabel('MSD ($\\mu m^2$)')
plt.show()

list_D = []
list_k = []

for p in range(1, Np+1):
    T, MSD = res_imsd['lag time [s]'].values, res_imsd[p].values
    parms, results = ufun.fitLineHuber(np.log(T), np.log(MSD), with_intercept = True)
    b, a = parms
    k_nl = a
    D_nl = np.exp(b)/4
    list_k.append(k_nl)
    list_D.append(D_nl)


# for p in range(1, Np+1):
#     T, MSD = res_imsd['lag time [s]'].values[-10:], res_imsd[p].values[-10:]
#     parms, results = ufun.fitLineHuber(np.log(T), np.log(MSD), with_intercept = True)
#     b, a = parms
#     k_nl = a
#     D_nl = np.exp(b)/4
#     if D_nl < 0.2 and D_nl > 0 and k_nl > 0:
#        list_k.append(k_nl)
#        list_D.append(D_nl)
    
# %%%

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
ax = axes[0]
# sns.swarmplot(ax=ax, y=list_D)
ax.hist(list_D, bins=60, color='gray', zorder=3)
ax.set_xlim([-0.005, 0.18])
ax.set_xlabel('D ($\\mu m^2/s$)')
ax.set_ylabel('N')

ax = axes[1]
# sns.swarmplot(ax=ax, y=list_k)
ax.hist(list_k, bins=60, color='gray', zorder=3)
ax.set_xlabel('exponent $\\alpha$')
ax.set_ylabel('N')

ax = axes[2]
sns.scatterplot(ax=ax, x=list_k, y=list_D, alpha=0.1)
ax.set_xlabel('exponent $\\alpha$')
ax.set_ylabel('D ($\\mu m^2/s$)')

plt.show()

