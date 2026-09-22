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
import cv2

import numpy as np
import pandas as pd
import trackpy as tp
import skimage as skm
import seaborn as sns
import matplotlib as mpl
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from scipy.interpolate import make_splrep

from shapely.geometry import MultiPoint, Polygon

import Libs.PlotMaker as pm
import Libs.UrchinPaths as up
import Libs.CalibrationData as cd
import Libs.UtilityFunctions as ufun
import Libs.ToolboxCytoplasmAnalysis as tbca
import Libs.ToolboxStructureAnalysis as tbsa



# %% -----------------------



# %% Film MT

# mainDir = 'C://Users//josep//Desktop//Seafile//DownloadedFromSeafile//IntraCellTracking//26-06-19_FastAcq_BF'
mainDir = "C://Users//Joseph//Desktop//FilmsFromNishant"


# %%% 1. DDM 

# %%%% 1.1 Settings

mainDir = "C://Users//Joseph//Desktop//FilmsFromNishant"
srcDir = mainDir
dstDir = os.path.join(mainDir, 'DDM_results')

tifNames = ['1.5xMT-1.5xKin-2prcnt-pluronic_2.tif',
            ]

tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]



UmPerPix = 1
frequencies = [1/30] * len(tifNames)
nbimages = 211
pointsPerDecade = 15
maxNCouples = 20 #10 for fast evaluation, 300 for accurate analysis

N_pix = 2048
L_um = N_pix*UmPerPix
print(f'Pixel size = {UmPerPix:.3f} µm',
      f'Optical resol = {0.647/(2*1.2):.3f} µm') # Lambda / 2.NA
dL = min(UmPerPix, 0.647/(2*1.2))
dq = 2*np.pi / L_um
qmin = 5*dq
qmax = ((2*np.pi) / (2*dL)) * 0.4  # 11.7
qmax = 0.5

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





# %%%% 1.3 Import

srcDir = os.path.join(mainDir, 'DDM_results')
DDMs = [np.load(os.path.join(srcDir, fN)) for fN in ddmFileNames]
dts = [np.load(os.path.join(srcDir, fN)) for fN in dtFileNames]


frequencies = [10] * len(DDMs)
QQ_raw = np.arange(1, 1+DDMs[0].shape[1])*dq

valid_iQ, valid_Q = [], []
for iq in range(len(QQ_raw)):
    q = QQ_raw[iq]
    if q >= qmin and q < qmax:
        valid_Q.append(q)
        valid_iQ.append(iq)

QQ = np.array(valid_Q)
iQ = np.array(valid_iQ)


# %%%% 1.3.x Plot typical images

pm.setGraphicOptions(mode = 'screen')

idts = tbca.logSpaced(nbimages, pointsPerDecade)
dts = [idts/float(freq) for freq in frequencies]

Nstep = 20
maxNCouples=20


for p in tifPaths:
    print(f'\n\nPlotting for {os.path.split(p)}...')
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), layout='compressed')
    
    stack = tbca.ImageStack(p) #, convert_to_8bits=True)
    
    ax = axes[0, 0]
    i = 0
    ax.imshow(stack[i], 'gray')
    ax.set_title(f'Frame no {i+1:.0f}')
    ax = axes[1, 0]
    j = Nstep-1
    ax.imshow(stack[j], 'gray')
    ax.set_title(f'Frame no {j+1:.0f}')
    ax = axes[2, 0]
    ax.imshow(stack[j] - stack[i].astype(float), 'gray')
    ax.set_title(r'$\Delta I$ for ' + f'f{j+1:.0f} and f{i+1:.0f}')
    
    F1, F2, F3 = (Nstep)-1, (Nstep*5)-1, (Nstep*10)-1
    I_0_N = np.fft.fftshift(tbca.spectrumDiff(stack[0], stack[(Nstep)-1]))
    I_0_10N = np.fft.fftshift(tbca.spectrumDiff(stack[0], stack[(Nstep*5)-1]))
    I_0_100N = np.fft.fftshift(tbca.spectrumDiff(stack[0], stack[(Nstep*10)-1]))
    V1, V2, V3 = np.percentile(I_0_N, 99), np.percentile(I_0_10N, 99), np.percentile(I_0_100N, 99)
    axes[0, 1].imshow(I_0_N, 'hot', vmin=0, vmax=V1)
    axes[0, 1].set_title(r'$TF[\Delta I]$ ' + f'for f0 and f{F1:.0f}')
    axes[1, 1].imshow(I_0_10N, 'hot', vmin=0, vmax=V2)
    axes[1, 1].set_title(r'$TF[\Delta I]$ ' + f'for f0 and f{F2:.0f}')
    axes[2, 1].imshow(I_0_100N, 'hot', vmin=0, vmax=V3)
    axes[2, 1].set_title(r'$TF[\Delta I]$ ' + f'for f0 and f{F3:.0f}')
    # print(f"{np.percentile(I_0_N, 99):.2e}")
    # print(f"{np.percentile(I_0_10N, 99):.2e}")
    # print(f"{np.percentile(I_0_100N, 99):.2e}")
    
    S1, S2, S3 = Nstep//5, Nstep, Nstep*5
    J_0_N10   = tbca.timeAveraged(stack, Nstep//5, maxNCouples=maxNCouples)
    J_0_N  = tbca.timeAveraged(stack, Nstep, maxNCouples=maxNCouples)
    J_0_10N = tbca.timeAveraged(stack, Nstep*5, maxNCouples=maxNCouples)
    V1, V2, V3 = np.percentile(J_0_N10, 99), np.percentile(J_0_N, 99), np.percentile(J_0_10N, 99)
    axes[0, 2].imshow(np.fft.fftshift(J_0_N10), 'hot', vmin=0, vmax=V1)
    axes[0, 2].set_title(r'$TF[\Delta I]$ for $\Delta t$ = ' + f'{S1:.0f}f')
    axes[1, 2].imshow(np.fft.fftshift(J_0_N), 'hot', vmin=0, vmax=V2)
    axes[1, 2].set_title(r'$TF[\Delta I]$ for $\Delta t$ = ' + f'{S2:.0f}f')
    axes[2, 2].imshow(np.fft.fftshift(J_0_10N), 'hot', vmin=0, vmax=V3)
    axes[2, 2].set_title(r'$TF[\Delta I]$ for $\Delta t$ = ' + f'{S3:.0f}f')
    
    ra = tbca.RadialAverager(stack.shape[1:])
    for ax in axes[:, 3]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'q ($rad\cdot px^{-1}$)')
        ax.set_ylabel(r'$D(q, \Delta t)$')
        
    axes[0, 3].plot(ra(J_0_N10), 'b-')
    axes[0, 3].set_title(r'$RA$ for $\Delta t$ = ' + f'{S1:.0f}f')
    axes[1, 3].plot(ra(J_0_N), 'b-')
    axes[1, 3].set_title(r'$RA$ for $\Delta t$ = ' + f'{S2:.0f}f')
    axes[2, 3].plot(ra(J_0_10N), 'b-')
    axes[2, 3].set_title(r'$RA$ for $\Delta t$ = ' + f'{S3:.0f}f')
    
    figfile = f'{os.path.split(p)[-1]}'[:-4] + '_summary.png'
    figpath = os.path.join(srcDir, figfile)
    fig.suptitle(f'Plotting for {os.path.split(p)[-1]}')
    fig.savefig(figpath, dpi=500, )
    plt.show()


# %%%% 1.4 Plot the structure matrix D

pm.setGraphicOptions(mode='screen')

for ii in range(len(DDMs)):
    fN = tifNames[ii]
    DDM_plot = DDMs[ii]
    dt_plot = dts[ii]
    
    (Ndt, Nq) = (len(dt_plot), len(QQ))
    fig, axes = plt.subplots(2, 1, figsize = (4, 7), layout='compressed')
    fig.suptitle(f'{fN}')
    
    # QQ_plot = np.arange(1, 1+Nq)*dq
    ax = axes[0]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$q\ (rad\cdot px^{-1})$')
    ax.set_xlim([1e-2, 1])
    ax.set_ylabel('$D$')
    for i in range(0, Ndt, 5):
        ax.plot(QQ, DDM_plot[i,iQ], marker='.', ls='',
                color = mpl.cm.autumn(i/Ndt))
        ax.axvline(qmin, color='gray', ls='-', alpha=0.7)
        ax.axvline(qmax, color='gray', ls='-', alpha=0.7)
        
    fig.colorbar(plt.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=np.min(dt_plot), 
                                                               vmax=np.max(dt_plot)), 
                                       cmap="autumn"),
                 ax=ax, label=r"$\Delta t$")
        
    
    ax = axes[1]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\Delta t\ (s)$')
    ax.set_ylabel('$D$')
    for j in iQ[::30]:
        ax.plot(dt_plot, DDM_plot[:,j], marker='.', ls='',
                color = mpl.cm.winter(j/Nq))
        
    fig.colorbar(plt.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=np.min(QQ), vmax=np.max(QQ)), 
                                       cmap="winter"),
                 ax=ax, label="$q$")
    
    figName = '.'.join(fN.split('.')[:-1])
    print(figName)
    figfile = f'{figName}' + '_matrixD.png'
    figpath = os.path.join(srcDir, figfile)
    fig.savefig(figpath, dpi=500, )
    
    plt.show()



# %%%% Plot estimates of A and B

DDM_plot = DDMs[0][:, iQ]
dt_plot = dts[0]

ApB_est = np.median(DDM_plot[-4:,:], axis=0)
B_est = np.median(DDM_plot[:6,:], axis=0)

fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
for ax in axes:
    ax.set_xscale('log')
    ax.set_yscale('log')
ax = axes[0]
ax.plot(QQ, ApB_est, 'r.')
# ax.axvline(dq, color='gray', ls='-', alpha=0.7)
ax.axvline(qmax, color='gray', ls='-', alpha=0.7)
ax = axes[1]
ax.plot(QQ, B_est,'k.')
# ax.axvline(dq, color='gray', ls='-', alpha=0.7)
ax.axvline(qmax, color='gray', ls='-', alpha=0.7)
ax = axes[2]
ax.plot(QQ, ApB_est-B_est,'b.')

# ax.axvline(dq, color='gray', ls='-', alpha=0.7)
ax.axvline(qmax, color='gray', ls='-', alpha=0.7)

plt.show()


# %%%% 1.5 Use a model to fit A, B and get f (Brownian case)

AA, BB, GG = [], [], []
kk_G = []

for ii in range(len(DDMs)): # len(DDMs)
    fN = tifNames[ii]
    DDM_fit = DDMs[ii]
    dt_fit = dts[ii]
    
    def simple_brownian_model(dt, A, B, G):
        D = A * (1 - np.exp(-G*dt)) + B
        return(D)
    
    ApB_est = np.median(DDM_fit[-4:, :], axis=0)
    B_est = np.min(DDM_fit[:5, :], axis=0)
    A_est = ApB_est - B_est
    
    A_est = A_est[iQ]
    B_est = B_est[iQ]
    
    
    list_A, list_B, list_G = [], [], []
    
    FORCE_B = True
    
    forced_B = [np.percentile(B_est, 0)/10000000] * len(QQ)
    
    # forced_B = B_est
    
    # MB = np.median(B_est[:3])
    # mB = np.median(B_est[-3:])
    # MQ = np.max(QQ)
    # mQ = np.min(QQ)
    # k = (np.log(MB)-np.log(mB)) / (np.log(mQ)-np.log(MQ))
    # A = mB / (MQ**k)
    # forced_B = [A * q**k for q in QQ]
    
    # logQQ = np.log(QQ)
    # logBest = np.log(B_est)
    # p_fitted = np.polynomial.Polynomial.fit(logQQ, logBest, deg=2)
    # B_smooth = [np.exp(p_fitted(q)) for q in logQQ]
    # forced_B = B_smooth
    
    # fig, ax = plt.subplots(1, 1, figsize=(4, 3), sharey=True)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.plot(QQ, B_est, 'r.')
    # ax.plot(QQ, forced_B, 'k--')
    # ax.axvline(qmax, color='gray', ls='-', alpha=0.7)
    # plt.show()
    
    fig, axes = plt.subplots(1, 3, figsize = (12, 5))
    
    for iq in iQ:
        jq = iq - min(iQ)
        q = QQ[jq]        
        D = DDM_fit[:,iq]
        dt = dt_fit
        
        if not FORCE_B:
            # some initial parameter values - must be within bounds
            initB = np.median(DDM_fit[:5, iq], axis=0)/10
            initA = max(0, np.median(DDM_fit[-4:, iq], axis=0) - initB)
            initG = 0.05
            
            initialParameters = [initA, initB, initG]
            
            # bounds on parameters - initial parameters must be within these
            lowerBounds = (0, 0, 0) # 0.8*np.min(B_est)
            upperBounds = (np.inf, np.inf, np.inf)
            parameterBounds = [lowerBounds, upperBounds]
            
            params, covM = curve_fit(simple_brownian_model, dt, D, 
                                     p0=initialParameters, bounds = parameterBounds, maxfev = 140000)
            
            A, B, G = params[0], params[1], params[2]
            list_A.append(A)
            list_B.append(B)
            list_G.append(G)
        
        else:
            # some initial parameter values - must be within bounds
            B_set = forced_B[jq]
            
            def simple_brownian_model_forced_B(dt, A, G):
                D = A * (1 - np.exp(-G*dt)) + B_set
                return(D)
            
            initA = max(0, np.median(DDM_fit[-4:, iq], axis=0) - B_set)
            initG = 0.005
                   
            initialParameters = [initA, initG]
            print('----------')
            print(f'q = {q:.2e} ' + r'$(rad\cdot px^{-1})$')
            print(f'{initA:.2e}', f'{B_set:.2e}', f'{initG:.1f}')
            
            # bounds on parameters - initial parameters must be within these
            lowerBounds = (0, 0)
            upperBounds = (np.inf, np.inf)
            parameterBounds = [lowerBounds, upperBounds]
            
            params, covM = curve_fit(simple_brownian_model_forced_B, dt, D, 
                                     p0=initialParameters, bounds = parameterBounds, maxfev = 200000)
            
            print(f'{params[0]:.2e}', f'{B_set:.2e}', f'{params[1]:.6f}')
            
            A, B, G = params[0], B_set, params[1]
            list_A.append(A)
            list_B.append(B)
            list_G.append(G)
            
            
        if iq%20 == 0:
            ax = axes[0]
            ax.set_xscale('log')
            ax.set_yscale('log')
            D = DDM_fit[:, iq]
            ax.plot(dt_fit, D, ls='', marker='o', label=f'q={q:.1f}')
            
            D_fit = simple_brownian_model(dt_fit, A, B, G)
            ax.plot(dt_fit, D_fit, ls='-', marker='', color='k')
            plt.show()
    
    list_A = np.array(list_A)
    list_B = np.array(list_B)
    list_G = np.array(list_G)
    AA.append(list_A)
    BB.append(list_B)
    GG.append(list_G)
    
    valid = (QQ < np.inf) & (QQ > 0)
    
    X, Y = np.log(QQ[valid]), np.log(list_G[valid])
    params, results = ufun.fitLineHuber(X, Y)
    (p1, p2) = params
    k = p2
    A = np.exp(p1)
    
    kk_G.append(k)
    
    ax = axes[0]
    # ax.set_ylim([1e8, 1e13])
    ax.legend(fontsize=8)
      
    
    ax = axes[1]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(QQ, list_A, ls='', marker='.', label='A(q)')
    ax.plot(QQ, list_B, ls='', marker='.', label='B(q)')
    ax.set_xlim([1e-2, 1])
    # ax.set_ylim([1e8, 1e13])
    ax.legend(fontsize=8)
    
    ax = axes[2]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(QQ, list_G, 'k.', label=r'$\Gamma$(q)')
    ax.plot(QQ, A * QQ**k, 'r-', label=f'k = {k:.2f}')
    # ax.set_ylim([1e-3, 1e0])
    ax.legend(fontsize=8)
    
    # fig.suptitle(f'T dev = {fN.split('_')[2]}')
    # figName = '_'.join(fN.split('_')[:5])
    # print(figName)
    # figfile = f'{figName}' + '_BrownianFit.png'
    # figpath = os.path.join(srcDir, figfile)
    # fig.savefig(figpath, dpi=500, )
    
    figName = '.'.join(fN.split('.')[:-1])
    print(figName)
    figfile = f'{figName}' + '_BrownianFit.png'
    figpath = os.path.join(srcDir, figfile)
    fig.savefig(figpath, dpi=500, )

    
    plt.show()



    

# %%%% 1.6 Plot the fit



for ii in range(len(DDMs)): # len(DDMs)
    fN = tifNames[ii]
    DDM_fit = DDMs[ii]
    dt_fit = dts[ii]
    list_A = AA[ii]
    list_B = BB[ii]
    list_G = GG[ii]
    k_G = kk_G[ii]
    
    idx = slice(10, len(iQ), 20)
    
    # fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # ax = ax
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # cmap = mpl.cm.plasma
    
    # k = 0
    
    # for iq in iQ[idx]:
    #     jq = iq - min(iQ)
    #     q = QQ[iq]
    #     A = list_A[jq]
    #     B = list_B[jq]
    #     G = list_G[jq]
        
    #     D = DDM_fit[:, iq]
    #     color = cmap(k/len(iQ[idx]))
    #     k += 1
    #     ax.plot(dt_fit, D, ls='', marker='o', color = color, label=f'q={q:.1f}')
        
    #     D_fit = simple_brownian_model(dt_fit, A, B, G)
    #     ax.plot(dt_fit, D_fit, ls='-', marker='', color = color, label='fit')
    
    # ax.legend()
    # ax.grid()
    # plt.show()
    
    
    
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout='compressed')
    cmap = mpl.cm.viridis
    
    k = 0
    
    for iq in iQ[idx]:
        jq = iq - min(iQ)
        q = QQ[iq]
        A = list_A[jq]
        B = list_B[jq]
        G = list_G[jq]
        
        D = DDM_fit[:, iq]
        color = cmap(k/len(iQ[idx]))
        k += 1
        fR = 1 - ((D-B)/A)
        fR_fit = np.exp(-G*dt)
        
        ax = axes[0]
        ax.plot(dt_fit, fR, ls='', marker='o', color = color, ms=3,
                label=f'q = {q:.1f}' + r'$\mu m^{-1}$')
        ax.plot(dt_fit, fR_fit, ls='-', marker='', color = color, lw=1,
                label=r'Fit, $\Gamma$' + f'={G:.1e}' + r'$s^{-1}$')
        
        ax = axes[1]
        ax.plot(dt_fit*q*q, fR, ls='', marker='o', color = color, ms=3)
        ax.plot(dt_fit*q*q, fR_fit, ls='-', marker='', color = color, lw=1)
        
        ax = axes[2]
        ax.plot(dt_fit*(q**k_G), fR, ls='', marker='o', color = color, ms=3)
        ax.plot(dt_fit*(q**k_G), fR_fit, ls='-', marker='', color = color, lw=1)
    
    for ax in axes:
        ax.set_xscale('log')
        ax.set_ylabel('ACF')
        
        ax.grid()
    
    axes[0].set_xlabel(r'$\Delta t$ (s)')
    axes[1].set_xlabel(r'$\Delta t \cdot q^2$ (s/um²)')
    axes[2].set_xlabel(r'$\Delta t \cdot q^k\ (s/um^k)$ ' + f'k={k_G:.1f}')
    fig.legend(fontsize = 7, loc='outside right center')
    
    plt.show()
    # fig.suptitle(f'T dev = {fN.split('_')[2]}', fontsize=11)
    # figName = '_'.join(fN.split('_')[:5])
    # print(figName)
    # figfile = f'{figName}' + '_ACF_Brownian.png'
    # figpath = os.path.join(srcDir, figfile)
    # fig.savefig(figpath, dpi=500, )
    
    figName = '.'.join(fN.split('.')[:-1])
    print(figName)
    figfile = f'{figName}' + '_ACF_Brownian.png'
    figpath = os.path.join(srcDir, figfile)
    fig.savefig(figpath, dpi=500, )
    
    
    
    
    
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # for ax in axes:
    #     ax.set_xscale('log')
    #     ax.set_yscale('log')
        
    # cmap = mpl.cm.plasma
    
    # # idx = slice(0, len(valid_iQ), 10)
    # k = 0
    
    # list_MSD_exp = []
    # list_MSD_fit = []
    
    # for iq in iQ[idx]:
    #     jq = iq - min(iQ)
    #     q = QQ[iq]
    #     A = list_A[jq]
    #     B = list_B[jq]
    #     G = list_G[jq]
        
    #     D = DDM_fit[:, iq]
    #     color = cmap(jq/(len(iQ)))
        
    #     fR = 1 - ((D-B)/A)
    #     fR_fit = np.exp(-G*dt)
        
    #     MSD_exp = -(4/q**2) * np.log(fR)
    #     MSD_fit = -(4/q**2) * np.log(fR_fit)
        
    #     list_MSD_exp.append(MSD_exp)
    #     list_MSD_fit.append(MSD_fit)
        
    #     if jq%10==0:
    #         ax = axes[0]
    #         ax.plot(dt, MSD_exp, ls='', marker='o', color = color)
    #         ax.plot(dt, MSD_fit, ls='-', marker='', color = color)
        
    #     k += 1
        
    # list_MSD_exp = np.array(list_MSD_exp)
    # list_MSD_fit = np.array(list_MSD_fit)
    
    # avg_MSD_exp = np.nanmean(list_MSD_exp, axis=0)
    # avg_MSD_fit = np.nanmean(list_MSD_fit, axis=0)
    
    # ax = axes[1]
    # ax.plot(dt, avg_MSD_exp, ls='', marker='o', color = 'k')
    # ax.plot(dt, avg_MSD_fit, ls='-', marker='', color = 'k')
    
    # for ax in axes:
    #     ax.legend()
    #     ax.grid()
    
    
    

# %%%% 1.6.ii Plot the fit, cont'd

pm.setGraphicOptions(mode='screen')
cmap = mpl.cm.viridis

iq = 13
jq = iq - min(iQ)
q = QQ[jq]

fig, ax = plt.subplots(1, 1, layout='compressed')
ax.set_xscale('log')
ax.set_ylabel('ACF')
ax.set_xlabel(r'$\Delta t$ (s)')
ax.grid()

for ii in range(len(DDMs)):
    DDM = DDMs[ii]
    dt = dts[ii]
    name = ddmFileNames[ii]
    list_A = AA[ii]
    list_B = BB[ii]
    list_G = GG[ii]
    
    D = DDM[:, iq]
    
    A = list_A[jq]
    B = list_B[jq]
    G = list_G[jq]
    
    fR = 1 - ((D-B)/A)
    fR_fit = np.exp(-G*dt)
    
    color = cmap((ii+0.5)/len(DDMs))
    Tdev = name.split('_')[2]
    
    ax.set_title(f'q = {q:.1f}' + r'$\mu m^{-1}$')
    ax.plot(dt, fR, ls='', marker='o', color = color, ms=3,
            label=f'tpf = {Tdev}')
    ax.plot(dt, fR_fit, ls='-', marker='', color = color, lw=1,
            label=r'$\Gamma$' + f'={G:.1e}' + r'$s^{-1}$')
    
fig.legend(fontsize = 9, loc='outside right center',
           title='Time Post Fertil.')
    
plt.show()

figfile = f'ACF_Brownian_allTdev_q-{q*1000:.0f}' + '.png'
figpath = os.path.join(srcDir, figfile)
fig.savefig(figpath, dpi=500, )


# %%%% 1.6.iii Plot the fit, cont'd

pm.setGraphicOptions(mode='screen')
cmap = mpl.cm.viridis
cmap = mpl.cm.BrBG

def Tpf_str2num(tpf_str):
    L = tpf_str.split('min')
    tpf_num = int(L[0])*60
    if len(L) > 1 and len(L[1]) > 0:
        tpf_num += int(L[1])
    return(tpf_num)    
    
fig, ax = plt.subplots(1, 1, layout='compressed')
# ax.set_xscale('log')
ax.set_ylabel(r'$q \ (\mu m^{-1})$')
ax.set_xlabel('Tpf (min)')
# ax.grid()

TT = 1/np.array(GG)
T_min, T_max = np.min(TT), np.max(TT)
norm_TT = (lambda x : (np.log(x)-np.log(T_min))/(np.log(T_max)-np.log(T_min)))

for ii in range(len(DDMs)):
    DDM = DDMs[ii]
    dt = dts[ii]
    name = ddmFileNames[ii]
    list_A = AA[ii]
    list_B = BB[ii]
    list_G = GG[ii]
    list_T = TT[ii, :]
    
    Tpf_str = name.split('_')[2]
    Tpf_num = Tpf_str2num(Tpf_str)
    
    for iq in iQ:
    
        D = DDM[:, iq]
        
        jq = iq - min(iQ)
        q = QQ[jq]
        A = list_A[jq]
        B = list_B[jq]
        G = list_G[jq]
        T = list_T[jq]
        
        fR = 1 - ((D-B)/A)
        fR_fit = np.exp(-dt/T)
        
        color = cmap(norm_TT(T))
    
    
        # ax.set_title(f'q = {q:.1f}' + r'$\mu m^{-1}$')
        ax.scatter(Tpf_num/60, q, s=12, color = color, alpha=0.8)

fig.colorbar(plt.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=np.min(TT), 
                                                           vmax=np.max(TT)), 
                                   cmap=cmap),
             ax=ax, label=r"$\tau=1/\Gamma \ (s)$")
    
plt.show()

# figfile = f'ACF_Brownian_allTdev_q-{q*1000:.0f}' + '.png'
# figpath = os.path.join(srcDir, figfile)
# fig.savefig(figpath, dpi=500, )

# %%%% 1.X Use a model to fit Beta, A, B and get f (Brownian + Exp Beta case)

AA, BB, GG, BetaBeta = [], [], [], []

for ii in range(len(DDMs)): # len(DDMs)
    fN = tifNames[ii]
    DDM_fit = DDMs[ii]
    dt_fit = dts[ii]
    
    def expo_brownian_model(dt, A, B, G, Beta):
        D = A * (1 - np.exp(-(G*dt)**Beta)) + B
        return(D)
    
    ApB_est = np.median(DDM_fit[-4:, :], axis=0)
    B_est = np.min(DDM_fit[:5, :], axis=0)
    A_est = ApB_est - B_est
    
    A_est = A_est[iQ]
    B_est = B_est[iQ]
    
    
    list_A, list_B, list_G, list_Beta = [], [], [], []
    
    FORCE_B = True
    
    forced_B = [np.percentile(B_est, 3)] * len(QQ)
    
    MB = np.median(B_est[:3])
    mB = np.median(B_est[-3:])
    MQ = np.max(QQ)
    mQ = np.min(QQ)
    k = (np.log(MB)-np.log(mB)) / (np.log(mQ)-np.log(MQ))
    A = mB / (MQ**k)
    forced_B = [A * q**k for q in QQ]
    
    # logQQ = np.log(QQ)
    # logBest = np.log(B_est)
    # p_fitted = np.polynomial.Polynomial.fit(logQQ, logBest, deg=2)
    # B_smooth = [np.exp(p_fitted(q)) for q in logQQ]
    # forced_B = B_smooth
    
    # fig, ax = plt.subplots(1, 1, figsize=(4, 3), sharey=True)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.plot(QQ, B_est, 'r.')
    # ax.plot(QQ, forced_B, 'k--')
    # ax.axvline(qmax, color='gray', ls='-', alpha=0.7)
    # plt.show()
    
    fig, axes = plt.subplots(1, 3, figsize = (12, 5))
    
    for iq in iQ:
        jq = iq - min(iQ)
        q = QQ[jq]        
        D = DDM_fit[:,iq]
        dt = dt_fit
        
        if not FORCE_B:
            # some initial parameter values - must be within bounds
            initB = np.median(DDM_fit[:5,iq], axis=0)
            initA = np.median(DDM_fit[-4:,iq], axis=0) - initB
            initG = 1
            initBeta = 0.5
            
            initialParameters = [initA, initB, initG, initBeta]
            
            # bounds on parameters - initial parameters must be within these
            lowerBounds = (0, 0.8*np.min(B_est), 0, 0.0)
            upperBounds = (np.inf, np.inf, np.inf, 1.0)
            parameterBounds = [lowerBounds, upperBounds]
            
            params, covM = curve_fit(expo_brownian_model, dt, D, 
                                     p0=initialParameters, bounds = parameterBounds)
            
            A, B, G, Beta = params[0], params[1], params[2], params[3]
            list_A.append(A)
            list_B.append(B)
            list_G.append(G)
            list_Beta.append(Beta)
        
        else:
            # some initial parameter values - must be within bounds
            B_set = forced_B[jq]
            def expo_brownian_model_forced_B(dt, A, G, Beta):
                D = A * (1 - np.exp(-(G*dt)**Beta)) + B_set
                return(D)
            
            initA = np.median(DDM_fit[-4:,iq], axis=0) - B_set
            initG = 1
            initBeta = 0.5
            
            initialParameters = [initA, initG, initBeta]
            
            # bounds on parameters - initial parameters must be within these
            lowerBounds = (0, 0, 0)
            upperBounds = (np.inf, np.inf, 1)
            parameterBounds = [lowerBounds, upperBounds]
            
            params, covM = curve_fit(expo_brownian_model_forced_B, dt, D, 
                                     p0=initialParameters, bounds = parameterBounds, maxfev = 140000)
            
            A, B, G, Beta = params[0], B_set, params[1], params[2]
            list_A.append(A)
            list_B.append(B)
            list_G.append(G)
            list_Beta.append(Beta)
            
            
        if iq%10 == 0:
            ax = axes[0]
            ax.set_xscale('log')
            ax.set_yscale('log')
            D = DDM_fit[:, iq]
            ax.plot(dt_fit, D, ls='', marker='o', label=f'q={q:.1f}')
            
            D_fit = expo_brownian_model(dt_fit, A, B, G, Beta)
            ax.plot(dt_fit, D_fit, ls='-', marker='', color='k')
            plt.show()
    
    list_A = np.array(list_A)
    list_B = np.array(list_B)
    list_G = np.array(list_G)
    list_Beta = np.array(list_Beta)
    AA.append(list_A)
    BB.append(list_B)
    GG.append(list_G)
    BetaBeta.append(list_Beta)
    
    valid = (QQ < 10) & (QQ > 2)
    
    X, Y = np.log(QQ[valid]), np.log(list_G[valid])
    params, results = ufun.fitLineHuber(X, Y)
    (p1, p2) = params
    k = p2
    A = np.exp(p1)
    
    ax = axes[0]
    ax.set_ylim([1e8, 1e13])
    ax.legend(fontsize=8)
      
    
    ax = axes[1]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(QQ, list_A, ls='', marker='.', label='A(q)')
    ax.plot(QQ, list_B, ls='', marker='.', label='B(q)')
    ax.set_ylim([1e8, 1e13])
    ax.legend(fontsize=8)
    
    ax = axes[2]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(QQ, list_G, 'k.', label=r'$\Gamma$(q)')
    ax.plot(QQ, A * QQ**k, 'r-', label=f'k = {k:.2f}')
    ax.set_ylim([1e-3, 1e0])
    ax.legend(fontsize=8)
    
    fig.suptitle(f'T dev = {fN.split('_')[2]}')
    figName = '_'.join(fN.split('_')[:5])
    print(figName)
    figfile = f'{figName}' + '_BrownianBetaFit.png'
    figpath = os.path.join(srcDir, figfile)
    fig.savefig(figpath, dpi=500, )
    
    plt.show()




# %%%% 1.X Plot the fit


for ii in range(len(DDMs)): # len(DDMs)
    fN = tifNames[ii]
    DDM_fit = DDMs[ii]
    dt_fit = dts[ii]
    list_A = AA[ii]
    list_B = BB[ii]
    list_G = GG[ii]
    list_Beta = BetaBeta[ii]
    
    idx = slice(10, len(iQ), 10)
    
    # fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # ax = ax
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # cmap = mpl.cm.plasma
    
    # k = 0
    
    # for iq in iQ[idx]:
    #     jq = iq - min(iQ)
    #     q = QQ[iq]
    #     A = list_A[jq]
    #     B = list_B[jq]
    #     G = list_G[jq]
        
    #     D = DDM_fit[:, iq]
    #     color = cmap(k/len(iQ[idx]))
    #     k += 1
    #     ax.plot(dt_fit, D, ls='', marker='o', color = color, label=f'q={q:.1f}')
        
    #     D_fit = simple_brownian_model(dt_fit, A, B, G)
    #     ax.plot(dt_fit, D_fit, ls='-', marker='', color = color, label='fit')
    
    # ax.legend()
    # ax.grid()
    # plt.show()
    
    
    
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), layout='compressed')
    cmap = mpl.cm.viridis
    
    k = 0
    
    for iq in iQ[idx]:
        jq = iq - min(iQ)
        q = QQ[iq]
        A = list_A[jq]
        B = list_B[jq]
        G = list_G[jq]
        Beta = list_Beta[jq]
        
        D = DDM_fit[:, iq]
        color = cmap(k/len(iQ[idx]))
        k += 1
        fR = 1 - ((D-B)/A)
        fR_fit = np.exp(-(G*dt)**Beta)
        
        ax = axes[0]
        ax.plot(dt_fit, fR, ls='', marker='o', color = color, ms=3,
                label=f'q = {q:.1f}' + r'$\mu m^{-1}$')
        ax.plot(dt_fit, fR_fit, ls='-', marker='', color = color, lw=1,
                label=r'Fit, $\Gamma$' + f'={G:.1e}' + r'$s^{-1}$')
        
        
    ax = axes[1]
    ax.set_xscale('log')
    ax.set_ylabel(r'Exponent $\beta$')
    ax.scatter(QQ, list_Beta, ls='', marker='o', s=5, 
                   c=(QQ-min(QQ))/(max(QQ)-min(QQ)), cmap=cmap)
    
    for ax in axes:
        ax.grid()
    
    axes[0].set_xlabel(r'$\Delta t$ (s)')
    axes[0].set_xscale('log')
    axes[0].set_ylabel('ACF')
    
    axes[1].set_xlabel(r'q ($\mu m^{-1}$)')
    fig.legend(fontsize = 7, loc='outside right center')
    
    plt.show()
    fig.suptitle(f'T dev = {fN.split('_')[2]}', fontsize=11)
    figName = '_'.join(fN.split('_')[:5])
    print(figName)
    figfile = f'{figName}' + '_ACF_BrownianBeta.png'
    figpath = os.path.join(srcDir, figfile)
    fig.savefig(figpath, dpi=500, )
    
    
    
    
    
    
    



# %%%% 1.7 Use a model to fit A, B and get f (Brownian + Ballistic case)

DDM_fit = DDMMerge
dt_fit = dtMerge

def brownian_plus_ballistic_model(dt, A, B, tD, tB, Z):
    theta = dt/((Z+1) * tB)
    F = np.exp(-dt/tD) * np.sin(Z*np.atan(theta))/(Z*theta * (1+theta**2)**(Z/2))
    D = A * (1 - F) + B
    return(D)

def brownian_plus_ballistic_model_fixed_Z(dt, A, B, tD, tB):
    Z = 2
    theta = dt/((Z+1) * tB)
    F = np.exp(-dt/tD) * np.sin(Z*np.atan(theta))/(Z*theta * (1+theta**2)**(Z/2))
    D = A * (1 - F) + B
    return(D)


FORCE_B = False

forced_B = [np.percentile(B_est, 3)] * len(QQ)

MB = np.median(B_est[:5])
mB = np.median(B_est[-5:])
MQ = np.max(QQ)
mQ = np.min(QQ)
k = (np.log(MB)-np.log(mB)) / (np.log(mQ)-np.log(MQ))
A = mB / (MQ**k)
forced_B = [A * q**k for q in QQ]

logQQ = np.log(QQ)
logBest = np.log(B_est)
p_fitted = np.polynomial.Polynomial.fit(logQQ, logBest, deg=5)
B_smooth = [np.exp(p_fitted(q)) for q in logQQ]
forced_B = B_smooth


list_A, list_B, list_tD, list_tB, list_Z = [], [], [], [], []

# FORCE_B = False
# forced_B = [np.percentile(B_est, 3)] * len(QQ)


for iq in iQ:
    jq = iq - min(iQ)
    q = QQ[jq]        
    D = DDM_fit[:,iq]
    dt = dt_fit
    
    if not FORCE_B:
        # some initial parameter values - must be within bounds
        initB = np.median(DDM_fit[:3,iq], axis=0)
        initA = np.median(DDM_fit[-4:,iq], axis=0) - initB
        inittD = 1/(0.01*q*q)
        inittB = 1/(0.1*q)
        initZ = 10
        
        initialParameters = [initA, initB, inittD, inittB, initZ]
        
        # bounds on parameters - initial parameters must be within these
        lowerBounds = (0, 0, 0, 0, 0)
        upperBounds = (np.inf, np.inf, np.inf, np.inf, np.inf)
        parameterBounds = [lowerBounds, upperBounds]
        
        params, covM = curve_fit(brownian_plus_ballistic_model, dt, D, 
                                 p0=initialParameters, bounds = parameterBounds)
        
        A, B, tD, tB, Z = params[0], params[1], params[2], params[3], params[4]
        list_A.append(A)
        list_B.append(B)
        list_tD.append(tD)
        list_tB.append(tB)
        list_Z.append(Z)
        
    else:
        # some initial parameter values - must be within bounds
        # B_set = forced_B[jq]
        # def model_forced_B(dt, A, tD, tB, Z):
        #     B = B_set
        #     return(brownian_plus_ballistic_model(dt, A, B, tD, tB, Z))
        
        # initA = np.median(DDM_fit[-4:,iq], axis=0) - initB
        # inittD = 1/(0.01*q*q)
        # inittB = 1/(0.1*q)
        # initZ = 2
        
        # initialParameters = [initA, inittD, inittB, initZ]
        
        # # bounds on parameters - initial parameters must be within these
        # lowerBounds = (0, 0, 0, 0)
        # upperBounds = (np.inf, np.inf, np.inf, np.inf)
        # parameterBounds = [lowerBounds, upperBounds]
        
        # params, covM = curve_fit(model_forced_B, dt, D, 
        #                          p0=initialParameters, bounds = parameterBounds)
        
        # A, tD, tB, Z = params[0], params[1], params[2], params[3]
        # list_A.append(A)
        # list_B.append(B_set)
        # list_tD.append(tD)
        # list_tB.append(tB)
        # list_Z.append(Z)
        
        B_set = forced_B[jq]
        def model_forced_B(dt, A, tD, tB):
            B = B_set
            return(brownian_plus_ballistic_model_fixed_Z(dt, A, B, tD, tB))
        
        initA = np.median(DDM_fit[-4:,iq], axis=0) - initB
        inittD = 1/(0.01*q*q)
        inittB = 1/(0.1*q)
        
        initialParameters = [initA, inittD, inittB]
        
        # bounds on parameters - initial parameters must be within these
        lowerBounds = (0, 0, 0)
        upperBounds = (np.inf, np.inf, np.inf)
        parameterBounds = [lowerBounds, upperBounds]
        
        params, covM = curve_fit(model_forced_B, dt, D, 
                                 p0=initialParameters, bounds = parameterBounds)
        
        A, tD, tB = params[0], params[1], params[2]
        list_A.append(A)
        list_B.append(B_set)
        list_tD.append(tD)
        list_tB.append(tB)
        list_Z.append(2)
        
        
    if iq%10 == 0:
        fig, ax = plt.subplots(1, 1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        D = DDM_fit[:, iq]
        ax.plot(dt_fit, D, ls='', marker='o', label=f'q={q:.1f}')
        
        D_fit = simple_brownian_model(dt_fit, A, B, G)
        ax.plot(dt_fit, D_fit, ls='-', marker='', label='fit')
        plt.show()


X, Y = np.log(QQ), np.log(list_tD)
params, results = ufun.fitLineHuber(X, Y)
(p1, p2) = params
kD = p2
AD = np.exp(p1)

X, Y = np.log(QQ), np.log(list_tB)
params, results = ufun.fitLineHuber(X, Y)
(p1, p2) = params
kB = p2
AB = np.exp(p1)

  
fig, axes = plt.subplots(1, 2, figsize = (8, 5))
ax = axes[0]
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(QQ, list_A, ls='', marker='.')
ax.plot(QQ, list_B, ls='', marker='.')

ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(QQ, list_tD, 'k.')
ax.plot(QQ, AD * QQ**kD, 'r-')

ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(QQ, list_tB, 'b.')
ax.plot(QQ, AB * QQ**kB, 'c-')

plt.show()

    
# %%%% 1.8 Use a model to fit A, B and get f (Brownian + Ballistic Fraction case)

AA, BB, TDTD, TBTB, AlphaAlpha, ZZ = [], [], [], []

for ii in range(len(DDMs)): # len(DDMs)
    fN = tifNames[ii]
    DDM_fit = DDMs[ii]
    dt_fit = dts[ii]
    
    def expo_brownian_model(dt, A, B, G, Beta):
        D = A * (1 - np.exp(-(G*dt)**Beta)) + B
        return(D)
    
    ApB_est = np.median(DDM_fit[-4:, :], axis=0)
    B_est = np.min(DDM_fit[:5, :], axis=0)
    A_est = ApB_est - B_est
    
    A_est = A_est[iQ]
    B_est = B_est[iQ]
    
    
    list_A, list_B, list_G, list_Beta = [], [], [], []
    
    FORCE_B = True
    
    forced_B = [np.percentile(B_est, 3)] * len(QQ)
    
    MB = np.median(B_est[:3])
    mB = np.median(B_est[-3:])
    MQ = np.max(QQ)
    mQ = np.min(QQ)
    k = (np.log(MB)-np.log(mB)) / (np.log(mQ)-np.log(MQ))
    A = mB / (MQ**k)
    forced_B = [A * q**k for q in QQ]

def brownian_plus_ballisticFrac_model(dt, A, B, tD, tB, alpha, Z):
    theta = dt/((Z+1) * tB)
    P = np.sin(Z*np.atan(theta))/(Z*theta * (1+theta**2)**(Z/2))
    F = np.exp(-dt/tD) * ((1-alpha) + alpha*P)
    D = A * (1 - F) + B
    return(D)

# def brownian_plus_ballisticFrac_model(dt, A, B, tD, tB, Z):
#     alpha = 0.1
#     theta = dt/((Z+1) * tB)
#     P = np.sin(Z*np.atan(theta))/(Z*theta * (1+theta**2)**(Z/2))
#     F = np.exp(-dt/tD) * ((1-alpha) + alpha*P)
#     D = A * (1 - F) + B
#     return(D)




# FORCE_B = False
# forced_B = [np.percentile(B_est, 3)] * len(QQ)

RERUN_WITH_FIXED_B = True



list_A, list_B, list_tD, list_tB, list_alpha, list_Z = [], [], [], [], [], []

for iq in iQ:
    jq = iq - min(iQ)
    q = QQ[jq]        
    D = DDM_fit[:,iq]
    dt = dt_fit
    
    # some initial parameter values - must be within bounds
    initB = np.median(DDM_fit[:3,iq], axis=0)
    initA = np.median(DDM_fit[-4:,iq], axis=0) - initB
    inittD = 1/(0.005*q*q)
    inittB = 1/(0.1*q)
    initalpha = 0.1
    initZ = 2
    
    initialParameters = [initA, initB, inittD, inittB, initalpha, initZ]
    
    # bounds on parameters - initial parameters must be within these
    lowerBounds = (0, 0, 0, 0, 0, 0)
    upperBounds = (np.inf, np.inf, np.inf, np.inf, 1, np.inf)
    parameterBounds = [lowerBounds, upperBounds]
    
    params, covM = curve_fit(brownian_plus_ballisticFrac_model, dt, D, 
                             p0=initialParameters, bounds = parameterBounds, maxfev = 140000)

    
    A, B, tD, tB, alpha, Z = params[0], params[1], params[2], params[3], params[4], params[5]
    list_A.append(A)
    list_B.append(B)
    list_tD.append(tD)
    list_tB.append(tB)
    list_alpha.append(alpha)
    list_Z.append(Z)
        
        
        
    if iq%10 == 0:
        fig, ax = plt.subplots(1, 1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        D = DDM_fit[:, iq]
        ax.plot(dt_fit, D, ls='', marker='o')
        
        D_fit = brownian_plus_ballisticFrac_model(dt, A, B, tD, tB, alpha, Z)
        ax.plot(dt_fit, D_fit, ls='-', marker='', label='fit')
        ax.set_title(f'q = {q:.1f}')
        plt.show()

X, Y = np.log(QQ), np.log(list_B)
params, results = ufun.fitLineHuber(X, Y)
(p1, p2) = params
k_B = p2
A_B = np.exp(p1)

X, Y = np.log(QQ), np.log(list_tD)
params, results = ufun.fitLineHuber(X, Y)
(p1, p2) = params
kD = p2
AD = np.exp(p1)

X, Y = np.log(QQ), np.log(list_tB)
params, results = ufun.fitLineHuber(X, Y)
(p1, p2) = params
kB = p2
AB = np.exp(p1)

  
fig, axes = plt.subplots(1, 2, figsize = (8, 5))
ax = axes[0]
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(QQ, list_A, ls='', marker='.')
ax.plot(QQ, list_B, ls='', marker='.')
ax.plot(QQ, A_B * QQ**k_B, ls='-')

ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(QQ, list_tD, 'k.')
ax.plot(QQ, AD * QQ**kD, 'r-')

ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(QQ, list_tB, 'b.')
ax.plot(QQ, AB * QQ**kB, 'c-')

plt.show()



# %%%% 1.9 Plot the fit

DDM_fit = DDMMerge
dt_fit = dtMerge
idx = slice(10, len(iQ), 10)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax = ax
ax.set_xscale('log')
ax.set_yscale('log')
cmap = mpl.cm.plasma

k = 0

for iq in iQ[idx]:
    jq = iq - min(iQ)
    q = QQ[iq]
    A = list_A[jq]
    B = list_B[jq]
    tD = list_tD[jq]
    tB = list_tB[jq]
    alpha = list_alpha[jq]
    Z = list_Z[jq]
    
    D = DDM_fit[:, iq]
    color = cmap(k/len(iQ[idx]))
    k += 1
    ax.plot(dt_fit, D, ls='', marker='o', color = color, label=f'q={q:.1f}')
    
    D_fit = brownian_plus_ballisticFrac_model(dt, A, B, tD, tB, alpha, Z)
    ax.plot(dt_fit, D_fit, ls='-', marker='', color = color, label='fit')

ax.legend()
ax.grid()
plt.show()




fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax in axes:
    ax.set_xscale('log')
    ax.legend()
    ax.grid()
cmap = mpl.cm.viridis

k = 0

for iq in iQ[idx]:
    jq = iq - min(iQ)
    q = QQ[iq]
    A = list_A[jq]
    B = list_B[jq]
    G = list_G[jq]
    
    D = DDM_fit[:, iq]
    color = cmap(k/len(iQ[idx]))
    k += 1
    fR = 1 - ((D-B)/A)
    fR_fit = np.exp(-G*dt)
    
    ax = axes[0]
    ax.plot(dt_fit, fR, ls='', marker='o', color = color, label=f'q = {q:.3f}')
    ax.plot(dt_fit, fR_fit, ls='-', marker='', color = color, label=f'fit, G = {G:.1e}')
    
    ax = axes[1]
    ax.plot(dt_fit*q*q, fR, ls='', marker='o', color = color, label=f'q = {q:.3f}')
    ax.plot(dt_fit*q*q, fR_fit, ls='-', marker='', color = color, label=f'fit, G = {G:.1e}')


    
plt.show()







fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax in axes:
    ax.set_xscale('log')
    ax.set_yscale('log')
    
cmap = mpl.cm.plasma

# idx = slice(0, len(valid_iQ), 10)
k = 0

list_MSD_exp = []
list_MSD_fit = []

for iq in iQ[idx]:
    jq = iq - min(iQ)
    q = QQ[iq]
    A = list_A[jq]
    B = list_B[jq]
    G = list_G[jq]
    
    D = DDM_fit[:, iq]
    color = cmap(jq/(len(iQ)))
    
    fR = 1 - ((D-B)/A)
    fR_fit = np.exp(-G*dt)
    
    MSD_exp = -(4/q**2) * np.log(fR)
    MSD_fit = -(4/q**2) * np.log(fR_fit)
    
    list_MSD_exp.append(MSD_exp)
    list_MSD_fit.append(MSD_fit)
    
    if jq%10==0:
        ax = axes[0]
        ax.plot(dt, MSD_exp, ls='', marker='o', color = color)
        ax.plot(dt, MSD_fit, ls='-', marker='', color = color)
    
    k += 1
    
list_MSD_exp = np.array(list_MSD_exp)
list_MSD_fit = np.array(list_MSD_fit)

avg_MSD_exp = np.nanmean(list_MSD_exp, axis=0)
avg_MSD_fit = np.nanmean(list_MSD_fit, axis=0)

ax = axes[1]
ax.plot(dt, avg_MSD_exp, ls='', marker='o', color = 'k')
ax.plot(dt, avg_MSD_fit, ls='-', marker='', color = 'k')

for ax in axes:
    ax.legend()
    ax.grid()
    

plt.show()

