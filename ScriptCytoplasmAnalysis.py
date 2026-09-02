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

import numpy as np
import pandas as pd
import trackpy as tp
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull

import Libs.PlotMaker as pm
import Libs.UrchinPaths as up
import Libs.CalibrationData as cd
import Libs.UtilityFunctions as ufun
import Libs.ToolboxCytoplasmAnalysis as tbca
import Libs.ToolboxStructureAnalysis as tbsa


# %% Test the proper TF

t = np.linspace(0, 2 * np.pi, 1024)
A = np.array([np.sin(10*t)]).T
B = np.ones((1, 1024))
# B = np.array([np.cos(10*t)])
# data2d = np.sin(t)[:, np.newaxis] * np.cos(t)[np.newaxis, :]
im = A @ B

tf_im = np.fft.fft2(im)

tf_im_shift = np.fft.fftshift(tf_im)

fig, axes = plt.subplots(1, 3)
ax = axes[0]
ax.imshow(im, cmap='gray')

ax = axes[1]
ax.imshow(np.abs(tf_im), cmap='gray')

ax = axes[2]
ax.imshow(np.abs(tf_im_shift), cmap='gray')

A = np.sin(10*t)

tf_A = np.fft.fft(A)

tf_A_shift = np.fft.fftshift(tf_A)

fig, axes = plt.subplots(1, 3)
ax = axes[0]
ax.plot(t, A)

ax = axes[1]
ax.plot(np.arange(len(A)), np.abs(tf_A))

ax = axes[2]
ax.plot(np.arange(len(A)), np.abs(tf_A_shift))

plt.show()

# %% Film BF

# mainDir = 'C://Users//josep//Desktop//Seafile//DownloadedFromSeafile//IntraCellTracking//26-06-19_FastAcq_BF'
mainDir = os.path.join(up.Path_IntraCellTracking, '26-06-19_FastAcq_BF')



# %%% 1. DDM 

# %%%% 1.1 Settings
mainDir = os.path.join(up.Path_IntraCellTracking, '26-06-19_FastAcq_BF')
srcDir = os.path.join(mainDir, 'Crops')
tifNames = ['FilmBF_fastAcq_4000f_200Hz_C3_crop.tif', 'FilmBF_fastAcq_4000f_10Hz_C3_crop.tif']
tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]

dstDir = os.path.join(mainDir, 'DDM_results')

UmPerPix = cd.UmPerPix_40X_Leica
frequencies = [200, 10]
nbimages = 4000
pointsPerDecade = 15
maxNCouples = 100 #10 for fast evaluation, 300 for accurate analysis

N_pix = 256
L_um = N_pix*UmPerPix
dq = 2*np.pi / L_um
qmin = 0.5
qmax = 10 # 11.7

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



# %%%% 1.3 Import & Merge

srcDir = os.path.join(mainDir, 'DDM_results')
DDMs = [np.load(os.path.join(srcDir, fN)) for fN in ddmFileNames]
dts = [np.load(os.path.join(srcDir, fN)) for fN in dtFileNames]


frequencies = [200, 10]

DDMMerge, dtMerge = tbca.mergeDDM(DDMs, dts, frequencies)

N_pix = 256
L_um = N_pix*UmPerPix
dq = 2*np.pi / L_um
qmin = 0.5
qmax = 10 # 11.7
QQ_raw = np.arange(1, 1+len(DDMMerge[0,:]))*dq

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

Nstep = 40

DDMs = []
for p in tifPaths:
    print(f'\n\nPlotting for {os.path.split(p)}...')
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), layout='compressed')
    
    stack = tbca.ImageStack(p)
    
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
    ax.set_title(r'$\Delta I$ for ' + f'F {j+1:.0f} and F {i+1:.0f}')
    
    I_0_N = np.fft.fftshift(tbca.spectrumDiff(stack[0], stack[Nstep-1]))
    I_0_10N = np.fft.fftshift(tbca.spectrumDiff(stack[0], stack[Nstep*10-1]))
    I_0_100N = np.fft.fftshift(tbca.spectrumDiff(stack[0], stack[Nstep*100-1]))
    V1, V2, V3 = np.percentile(I_0_N, 99), np.percentile(I_0_10N, 99), np.percentile(I_0_100N, 99)
    axes[0, 1].imshow(I_0_N, 'hot', vmin=0, vmax=V1)
    axes[0, 1].set_title(f'Frame no {Nstep*1:.0f}')
    axes[1, 1].imshow(I_0_10N, 'hot', vmin=0, vmax=V2)
    axes[1, 1].set_title(f'Frame no {Nstep*10:.0f}')
    axes[2, 1].imshow(I_0_100N, 'hot', vmin=0, vmax=V3)
    axes[2, 1].set_title(f'Frame no {Nstep*100:.0f}')
    # print(f"{np.percentile(I_0_N, 99):.2e}")
    # print(f"{np.percentile(I_0_10N, 99):.2e}")
    # print(f"{np.percentile(I_0_100N, 99):.2e}")
    
    J_0_N10   = tbca.timeAveraged(stack, Nstep//10, maxNCouples=100)
    J_0_N  = tbca.timeAveraged(stack, Nstep, maxNCouples=100)
    J_0_10N = tbca.timeAveraged(stack, Nstep*10, maxNCouples=100)
    V1, V2, V3 = np.percentile(J_0_N10, 99), np.percentile(J_0_N, 99), np.percentile(J_0_10N, 99)
    axes[0, 2].imshow(np.fft.fftshift(J_0_N10), 'hot', vmin=0, vmax=V1)
    axes[0, 2].set_title(r'$TF[\Delta I]$ for $\Delta t$ = ' + f'{Nstep//10:.0f}f')
    axes[1, 2].imshow(np.fft.fftshift(J_0_N), 'hot', vmin=0, vmax=V2)
    axes[1, 2].set_title(r'$TF[\Delta I]$ for $\Delta t$ = ' + f'{Nstep*1:.0f}f')
    axes[2, 2].imshow(np.fft.fftshift(J_0_10N), 'hot', vmin=0, vmax=V3)
    axes[2, 2].set_title(r'$TF[\Delta I]$ for $\Delta t$ = ' + f'{Nstep*10:.0f}f')
    
    ra = tbca.RadialAverager(stack.shape[1:])
    for ax in axes[:, 3]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'q ($px^{-1}$)')
        ax.set_ylabel(r'$D(q, \Delta t)$')
        
    axes[0, 3].plot(ra(J_0_N10), 'b-')
    axes[0, 3].set_title(r'$RA$ for $\Delta t$ = ' + f'{Nstep//10:.0f}f')
    axes[1, 3].plot(ra(J_0_N), 'b-')
    axes[1, 3].set_title(r'$RA$ for $\Delta t$ = ' + f'{Nstep*1:.0f}f')
    axes[2, 3].plot(ra(J_0_10N), 'b-')
    axes[2, 3].set_title(r'$RA$ for $\Delta t$ = ' + f'{Nstep*10:.0f}f')
    
    figfile = f'{os.path.split(p)[-1]}'[:-4] + '_summary.png'
    figpath = os.path.join(srcDir, figfile)
    fig.suptitle(f'Plotting for {os.path.split(p)[-1]}')
    fig.savefig(figpath, dpi=500, )
    plt.show()



# %%%% 1.3.y Plot the merge

# Test the merging
iQ_plot = [50]
Q_plot = [QQ[i] for i in iQ_plot]
for iq, q in zip(iQ_plot, Q_plot):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(dts[0], DDMs[0][:, iQ_plot],'o', label='200 Hz')
    ax.plot(dts[1], DDMs[1][:, iQ_plot],'s', label='10 Hz')
    ax.plot(dtMerge, DDMMerge[:, iQ_plot], label='Merged')
    ax.plot(dtMerge, DDMMerge[:, iQ_plot]*5, 'o', label='Shifted merged')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$\mathcal{D}$')
    ax.set_xlabel(r'$\Delta t\,(s)$')
    ax.legend(loc='lower right')
    ax.set_title(f'q = {q:.3f}')
    
plt.show()

# %%%% Plot the structure matrix D

DDM_plot = DDMMerge
dt_plot = dtMerge

(Ndt, Nq) = DDM_plot.shape
fig, axes = plt.subplots(2, 1, figsize = (5, 8))
# QQ_plot = np.arange(1, 1+Nq)*dq
ax = axes[0]
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$q\ (\mu m^{-1})$')
ax.set_ylabel('$D$')
for i in range(0, Ndt, 10):
    ax.plot(QQ, DDM_plot[i,iQ], marker='.', ls='',
            color = mpl.cm.autumn(i/Ndt))
    ax.axvline(qmin, color='gray', ls='-', alpha=0.7)
    ax.axvline(qmax, color='gray', ls='-', alpha=0.7)
    
fig.colorbar(plt.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=np.min(dt_plot), vmax=np.max(dt_plot)), 
                                   cmap="autumn"),
             ax=ax, label=r"$\Delta t$")
    

ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\Delta t\ (s)$')
ax.set_ylabel('$D$')
for j in iQ[::10]:
    ax.plot(dt_plot, DDM_plot[:,j], marker='.', ls='',
            color = mpl.cm.winter(j/Nq))
    
fig.colorbar(plt.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=np.min(QQ), vmax=np.max(QQ)), 
                                   cmap="winter"),
             ax=ax, label="$q$")

plt.show()

# %%%% Plot estimates of A and B

DDM_plot = DDMMerge[:, iQ]
dt_plot = dtMerge

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

DDM_fit = DDMMerge
dt_fit = dtMerge

def simple_brownian_model(dt, A, B, G):
    D = A * (1 - np.exp(-G*dt)) + B
    return(D)

ApB_est = np.median(DDM_fit[-4:,:], axis=0)
B_est = np.median(DDM_fit[:5,:], axis=0)
A_est = ApB_est - B_est

A_est = A_est[iQ]
B_est = B_est[iQ]


list_A, list_B, list_G = [], [], []

FORCE_B = True

forced_B = [np.percentile(B_est, 3)] * len(QQ)

MB = np.median(B_est[:5])
mB = np.median(B_est[-5:])
MQ = np.max(QQ)
mQ = np.min(QQ)
k = (np.log(MB)-np.log(mB)) / (np.log(mQ)-np.log(MQ))
A = mB / (MQ**k)
forced_B = [A * q**k for q in QQ]

# forced_B = B_est

# logQQ = np.log(QQ)
# logBest = np.log(B_est)
# p_fitted = np.polynomial.Polynomial.fit(logQQ, logBest, deg=5)
# B_smooth = [np.exp(p_fitted(q)) for q in logQQ]
# forced_B = B_smooth

# fig, ax = plt.subplots(1, 1, figsize=(4, 3), sharey=True)
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax = ax
# ax.plot(QQ, B_est, 'r.')
# ax.plot(QQ, forced_B, 'k--')
# ax.axvline(qmax, color='gray', ls='-', alpha=0.7)

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
        
        initialParameters = [initA, initB, initG]
        
        # bounds on parameters - initial parameters must be within these
        lowerBounds = (0, np.min(B_est), 0)
        upperBounds = (np.inf, np.inf, np.inf)
        parameterBounds = [lowerBounds, upperBounds]
        
        params, covM = curve_fit(simple_brownian_model, dt, D, 
                                 p0=initialParameters, bounds = parameterBounds)
        
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
        
        initA = np.median(DDM_fit[-4:,iq], axis=0) - B_set
        initG = 1
               
        initialParameters = [initA, initG]
        
        # bounds on parameters - initial parameters must be within these
        lowerBounds = (0, 0)
        upperBounds = (np.inf, np.inf)
        parameterBounds = [lowerBounds, upperBounds]
        
        params, covM = curve_fit(simple_brownian_model_forced_B, dt, D, 
                                 p0=initialParameters, bounds = parameterBounds)
        
        A, B, G = params[0], B_set, params[1]
        list_A.append(A)
        list_B.append(B)
        list_G.append(G)
        
        
    if iq%10 == 0:
        ax = axes[0]
        ax.set_xscale('log')
        ax.set_yscale('log')
        D = DDM_fit[:, iq]
        ax.plot(dt_fit, D, ls='', marker='o', label=f'q={q:.1f}')
        
        D_fit = simple_brownian_model(dt_fit, A, B, G)
        ax.plot(dt_fit, D_fit, ls='-', marker='', color='k')
        plt.show()
        
# smoothA = savgol_filter(list_A, 9, 3)
# smoothB = savgol_filter(list_B, 19, 1)
# coeffs = np.polyfit(valid_Q, list_B, 3)
# smoothB = np.sum([coeffs[k]*np.array(valid_Q)**(len(coeffs)-k) for k in range(len(coeffs))], axis=0)

# list_A = smoothA
# list_B = smoothB

# smoothG = []
# for k, iq in enumerate(valid_iQ):
#     q = QQ[iq]
        
#     D = DDM_fit[:, iq]
#     dt = dt_fit
    
#     # some initial parameter values - must be within bounds
#     setA = list_A[k]
#     setB = list_B[k]
#     initG = 1
    
#     def simple_brownian_model_setAB(dt, G):
#         D = setA * (1 - np.exp(-G*dt)) + setB
#         return(D)
    
#     initialParameters = [initG]
    
#     # bounds on parameters - initial parameters must be within these
#     lowerBounds = (0)
#     upperBounds = (np.inf)
#     parameterBounds = [lowerBounds, upperBounds]
    
#     params, covM = curve_fit(simple_brownian_model_setAB, dt, D, 
#                              p0=initialParameters, bounds = parameterBounds)
    
#     # if params[1] < 1e8:
#     #     params[1] = initB
    
#     smoothG.append(params[0])

# list_G = smoothG

list_A = np.array(list_A)
list_B = np.array(list_B)
list_G = np.array(list_G)

valid = (QQ < 7) & (QQ > 3)

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
ax.plot(QQ, B_est, 'k.', alpha=0.4, label='B(q) - estimated')
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

fig.suptitle('Bright Field DDM_merged')
figName = '_'.join(fN.split('_')[:5])
print(figName)
figfile = f'{figName}' + '_BrownianFit.png'
figpath = os.path.join(srcDir, figfile)
fig.savefig(figpath, dpi=500, )

plt.show()
    

# %%%% 1.6 Plot the fit

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
    G = list_G[jq]
    
    D = DDM_fit[:, iq]
    color = cmap(k/len(iQ[idx]))
    k += 1
    ax.plot(dt_fit, D, ls='', marker='o', color = color, label=f'q={q:.1f}')
    
    D_fit = simple_brownian_model(dt_fit, A, B, G)
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


FORCE_B = True

forced_B = [np.percentile(B_est, 3)] * len(QQ)

MB = np.median(B_est[:5])
mB = np.median(B_est[-5:])
MQ = np.max(QQ)
mQ = np.min(QQ)
k = (np.log(MB)-np.log(mB)) / (np.log(mQ)-np.log(MQ))
A = mB / (MQ**k)
forced_B = [A * q**k for q in QQ]

# logQQ = np.log(QQ)
# logBest = np.log(B_est)
# p_fitted = np.polynomial.Polynomial.fit(logQQ, logBest, deg=5)
# B_smooth = [np.exp(p_fitted(q)) for q in logQQ]
# forced_B = B_smooth


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

DDM_fit = DDMMerge
dt_fit = dtMerge

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

fig, axes = plt.subplots(1, 4, figsize = (16, 5))

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
        ax = axes[0]
        ax.set_xscale('log')
        ax.set_yscale('log')
        D = DDM_fit[:, iq]
        ax.plot(dt_fit, D, ls='', marker='o', label=f'q={q:.1f}')
        
        D_fit = brownian_plus_ballisticFrac_model(dt, A, B, tD, tB, alpha, Z)
        ax.plot(dt_fit, D_fit, ls='-', marker='', color='k')
        plt.show()



X, Y = np.log(QQ), np.log(list_B)
params, results = ufun.fitLineHuber(X, Y)
(p1, p2) = params
k_B = p2
A_B = np.exp(p1)

valid = (QQ < 7) & (QQ > 3)

X, Y = np.log(QQ[valid]), np.log(np.array(list_tD)[valid])
params, results = ufun.fitLineHuber(X, Y)
(p1, p2) = params
kD = p2
AD = np.exp(p1)

X, Y = np.log(QQ[valid]), np.log(np.array(list_tB)[valid])
params, results = ufun.fitLineHuber(X, Y)
(p1, p2) = params
kB = p2
AB = np.exp(p1)

  
ax = axes[0]
ax.set_ylim([1e8, 1e13])
ax.legend(fontsize=8)
ax.set_xlabel(r'$\Delta t$ (s)')
  


ax = axes[1]
ax.set_xlabel(r'$q\ (\mu m)$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(QQ, list_A, ls='', marker='.', label='A(q)')
ax.plot(QQ, B_est, 'k.', alpha=0.4, label='B(q) - estimated')
ax.plot(QQ, list_B, ls='', marker='.', label='B(q)')
ax.plot(QQ, A_B * QQ**k_B, ls='-')
ax.set_ylim([1e8, 1e13])
ax.legend(fontsize=8)


ax = axes[2]
ax.set_xlabel(r'$q\ (\mu m)$')
ax.set_ylabel(r'Characteristic Times (s)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(QQ, list_tD, 'k.', label=r'$\tau_D(q)$')
ax.plot(QQ, AD * QQ**kD, 'r-', label=r'Fit $\tau_D(q)$ ' + f'\nk={kD:.2f}')

ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(QQ, list_tB, 'b.', label=r'$\tau_B$')
ax.plot(QQ, AB * QQ**kB, 'c-', label=r'Fit $\tau_B(q)$ ' + f'\nk={kB:.2f}')
ax.legend()

ax = axes[3]
ax.set_xlabel(r'$q\ (\mu m)$')
ax.set_xscale('log')
ax.plot(QQ, list_alpha, 'b.', label=r'$\alpha$')
ax.legend()

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










# %%% 



# %% -----------------------



# %% Film NB-Yolk

# mainDir = 'C://Users//josep//Desktop//Seafile//DownloadedFromSeafile//IntraCellTracking//26-06-19_FastAcq_BF'
mainDir = os.path.join(up.Path_IntraCellTracking, '26-07-29_FastAcq_Fec_NB-Yolk')


# %%% 1. DDM 

# %%%% 1.1 Settings

mainDir = os.path.join(up.Path_IntraCellTracking, '26-07-29_FastAcq_Fec_NB-Yolk')
srcDir = os.path.join(mainDir, 'Crops')
tifNames = ['26-07-29_PostF_2min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_6min30_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_12min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_30min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_45min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_70min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_80min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_100min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_120min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
             ]

tifNames = ['26-07-29_PostF_2min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_6min30_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_12min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_30min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_45min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_70min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_80min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_100min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_120min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
             ]
tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]

dstDir = os.path.join(mainDir, 'DDM_results')

UmPerPix = cd.UmPerPix_60X_W1
frequencies = [10] * len(tifNames)
nbimages = 2000
pointsPerDecade = 15
maxNCouples = 300 #10 for fast evaluation, 300 for accurate analysis

N_pix = 512
L_um = N_pix*UmPerPix
print(f'Pixel size = {UmPerPix:.3f} µm',
      f'Optical resol = {0.647/(2*1.2):.3f} µm') # Lambda / 2.NA
dL = min(UmPerPix, 0.647/(2*1.2))
dq = 2*np.pi / L_um
qmin = 5*dq
qmax = ((2*np.pi) / (2*dL)) * 0.4  # 11.7

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
    
    I_0_N = np.fft.fftshift(tbca.spectrumDiff(stack[0], stack[Nstep-1]))
    I_0_10N = np.fft.fftshift(tbca.spectrumDiff(stack[0], stack[Nstep*10-1]))
    I_0_100N = np.fft.fftshift(tbca.spectrumDiff(stack[0], stack[Nstep*100-1]))
    V1, V2, V3 = np.percentile(I_0_N, 99), np.percentile(I_0_10N, 99), np.percentile(I_0_100N, 99)
    axes[0, 1].imshow(I_0_N, 'hot', vmin=0, vmax=V1)
    axes[0, 1].set_title(r'$TF[\Delta I]$ ' + f'for f0 and f{Nstep*1:.0f}')
    axes[1, 1].imshow(I_0_10N, 'hot', vmin=0, vmax=V2)
    axes[1, 1].set_title(r'$TF[\Delta I]$ ' + f'for f0 and f{Nstep*10:.0f}')
    axes[2, 1].imshow(I_0_100N, 'hot', vmin=0, vmax=V3)
    axes[2, 1].set_title(r'$TF[\Delta I]$ ' + f'for f0 and f{Nstep*100:.0f}')
    # print(f"{np.percentile(I_0_N, 99):.2e}")
    # print(f"{np.percentile(I_0_10N, 99):.2e}")
    # print(f"{np.percentile(I_0_100N, 99):.2e}")
    
    J_0_N10   = tbca.timeAveraged(stack, Nstep//10, maxNCouples=100)
    J_0_N  = tbca.timeAveraged(stack, Nstep, maxNCouples=100)
    J_0_10N = tbca.timeAveraged(stack, Nstep*10, maxNCouples=100)
    V1, V2, V3 = np.percentile(J_0_N10, 99), np.percentile(J_0_N, 99), np.percentile(J_0_10N, 99)
    axes[0, 2].imshow(np.fft.fftshift(J_0_N10), 'hot', vmin=0, vmax=V1)
    axes[0, 2].set_title(r'$TF[\Delta I]$ for $\Delta t$ = ' + f'{Nstep//10:.0f}f')
    axes[1, 2].imshow(np.fft.fftshift(J_0_N), 'hot', vmin=0, vmax=V2)
    axes[1, 2].set_title(r'$TF[\Delta I]$ for $\Delta t$ = ' + f'{Nstep*1:.0f}f')
    axes[2, 2].imshow(np.fft.fftshift(J_0_10N), 'hot', vmin=0, vmax=V3)
    axes[2, 2].set_title(r'$TF[\Delta I]$ for $\Delta t$ = ' + f'{Nstep*10:.0f}f')
    
    ra = tbca.RadialAverager(stack.shape[1:])
    for ax in axes[:, 3]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'q ($px^{-1}$)')
        ax.set_ylabel(r'$D(q, \Delta t)$')
        
    axes[0, 3].plot(ra(J_0_N10), 'b-')
    axes[0, 3].set_title(r'$RA$ for $\Delta t$ = ' + f'{Nstep//10:.0f}f')
    axes[1, 3].plot(ra(J_0_N), 'b-')
    axes[1, 3].set_title(r'$RA$ for $\Delta t$ = ' + f'{Nstep*1:.0f}f')
    axes[2, 3].plot(ra(J_0_10N), 'b-')
    axes[2, 3].set_title(r'$RA$ for $\Delta t$ = ' + f'{Nstep*10:.0f}f')
    
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
    fig.suptitle(f'T dev = {fN.split('_')[2]}')
    
    # QQ_plot = np.arange(1, 1+Nq)*dq
    ax = axes[0]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$q\ (\mu m^{-1})$')
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
    for j in iQ[::10]:
        ax.plot(dt_plot, DDM_plot[:,j], marker='.', ls='',
                color = mpl.cm.winter(j/Nq))
        
    fig.colorbar(plt.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=np.min(QQ), vmax=np.max(QQ)), 
                                       cmap="winter"),
                 ax=ax, label="$q$")
    
    figName = '_'.join(fN.split('_')[:5])
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
    
    forced_B = [np.percentile(B_est, 3)] * len(QQ)
    
    forced_B = B_est
    
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
            
            initialParameters = [initA, initB, initG]
            
            # bounds on parameters - initial parameters must be within these
            lowerBounds = (0, 0, 0) # 0.8*np.min(B_est)
            upperBounds = (np.inf, np.inf, np.inf)
            parameterBounds = [lowerBounds, upperBounds]
            
            params, covM = curve_fit(simple_brownian_model, dt, D, 
                                     p0=initialParameters, bounds = parameterBounds)
            
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
            
            initA = np.median(DDM_fit[-4:,iq], axis=0) - B_set
            initG = 1
                   
            initialParameters = [initA, initG]
            
            # bounds on parameters - initial parameters must be within these
            lowerBounds = (0, 0)
            upperBounds = (np.inf, np.inf)
            parameterBounds = [lowerBounds, upperBounds]
            
            params, covM = curve_fit(simple_brownian_model_forced_B, dt, D, 
                                     p0=initialParameters, bounds = parameterBounds)
            
            A, B, G = params[0], B_set, params[1]
            list_A.append(A)
            list_B.append(B)
            list_G.append(G)
            
            
        if iq%10 == 0:
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
    
    valid = (QQ < 10) & (QQ > 2)
    
    X, Y = np.log(QQ[valid]), np.log(list_G[valid])
    params, results = ufun.fitLineHuber(X, Y)
    (p1, p2) = params
    k = p2
    A = np.exp(p1)
    
    kk_G.append(k)
    
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
    
    # fig.suptitle(f'T dev = {fN.split('_')[2]}')
    # figName = '_'.join(fN.split('_')[:5])
    # print(figName)
    # figfile = f'{figName}' + '_BrownianFit.png'
    # figpath = os.path.join(srcDir, figfile)
    # fig.savefig(figpath, dpi=500, )
    
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
    fig.suptitle(f'T dev = {fN.split('_')[2]}', fontsize=11)
    figName = '_'.join(fN.split('_')[:5])
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


# %%% 1.5. Brownian fit as a function

def fitBrownianModel(DDM, dt, QQ, iQ, fN):
    DDM_fit = DDM
    dt_fit = dt
    
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
            
            initialParameters = [initA, initB, initG]
            
            # bounds on parameters - initial parameters must be within these
            lowerBounds = (0, 0.8*np.min(B_est), 0)
            upperBounds = (np.inf, np.inf, np.inf)
            parameterBounds = [lowerBounds, upperBounds]
            
            params, covM = curve_fit(simple_brownian_model, dt, D, 
                                     p0=initialParameters, bounds = parameterBounds)
            
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
            
            initA = np.median(DDM_fit[-4:,iq], axis=0) - B_set
            initG = 1
                   
            initialParameters = [initA, initG]
            
            # bounds on parameters - initial parameters must be within these
            lowerBounds = (0, 0)
            upperBounds = (np.inf, np.inf)
            parameterBounds = [lowerBounds, upperBounds]
            
            params, covM = curve_fit(simple_brownian_model_forced_B, dt, D, 
                                     p0=initialParameters, bounds = parameterBounds)
            
            A, B, G = params[0], B_set, params[1]
            list_A.append(A)
            list_B.append(B)
            list_G.append(G)
            
            
        if iq%10 == 0:
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
    figfile = f'{figName}' + '_BrownianFit.png'
    figpath = os.path.join(srcDir, figfile)
    fig.savefig(figpath, dpi=500, )
    
    plt.show()
    
    return(list_A, list_B, list_G)







# %%% 2. Tracking and MSD

# %%%% Settings

mainDir = os.path.join(up.Path_IntraCellTracking, '26-07-29_FastAcq_Fec_NB-Yolk')
srcDir = os.path.join(mainDir, 'Crops')
dstDir = os.path.join(mainDir, 'SPT_results')


tifNames = ['26-07-29_PostF_2min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_6min30_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_12min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_30min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_45min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_70min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_80min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_100min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_120min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
             ]
# tifNames = ['26-07-29_PostF_2min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
#              ]
tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
xmlNames = [tifName.split('.')[0] + '_PyTracks.xml' for tifName in tifNames]
xmlPaths = [os.path.join(dstDir, xmlName)  for xmlName in xmlNames]

dfNames = [tifName.split('.')[0] + '_PyTracks.csv' for tifName in tifNames]
jsonNames = [tifName.split('.')[0] + '_MSDfit.json' for tifName in tifNames]
msdNames = [tifName.split('.')[0] + '_MSD.csv' for tifName in tifNames]

UmPerPix = cd.UmPerPix_60X_W1
SCALE = 1/UmPerPix
nbimages = 2000
FPS = 10

N_pix = 512
L_um = N_pix*UmPerPix
# print(f'Pixel size = {UmPerPix:.3f} µm',
#       f'Optical resol = {0.647/(2*1.2):.3f} µm') # Lambda / 2.NA
# dL = min(UmPerPix, 0.647/(2*1.2))
# dq = 2*np.pi / L_um 
# qmin = 5*dq
# qmax = ((2*np.pi) / (2*dL)) * 0.4  # 11.7










# %%%% Run Trackmate

for tifPath, xmlName in zip(tifPaths, xmlNames):
    tbca.pretreatAndTrack_CropedYolk(tifPath, xmlName, dstDir,
                                     PLOT = True, SAVEPLOT = True)


# %%%% Import & format tracks

for ii in range(len(xmlPaths)):
    xmlPath = xmlPaths[ii]
    dfName = dfNames[ii]
    
    Tracks = tbca.importTrackMateTracks(xmlPath)
    
    Np = len(Tracks)
        
    column_names = ['frame', 'x', 'y', 'particle']
    all_tracks = []
    for i, track in enumerate(Tracks):
        nT = len(track)
        # test_x_sat = ((np.max(track[:, 1]) - np.min(track[:, 1])) < 1)
        # test_y_sat = ((np.max(track[:, 2]) - np.min(track[:, 2])) < 1)
        test_x_sat = ((np.max(track[:, 1]) == (N_pix-1)) or (np.min(track[:, 1]) == 0))
        test_y_sat = ((np.max(track[:, 2]) == (N_pix-1)) or (np.min(track[:, 2]) == 0))
        if (not test_x_sat) and (not test_y_sat) and (nT >= 30):
            track = np.concat((track, np.ones((len(track[:,0]), 1), dtype=int) * (i+1)), axis = 1)
            track[:, 0] = track[:, 0].astype(int) + 1
            all_tracks.append(track)
    
    concat_tracks = np.concat(all_tracks, axis = 0)
    df = pd.DataFrame({column_names[k] : concat_tracks[:,k] for k in range(len(column_names))})
    df.to_csv(os.path.join(dstDir, dfName), index=False, sep = '\t')
    

# %%%% Import tracks & run msd


for ii in range(len(dfNames)):
    dfName = dfNames[ii]
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    jsonName = jsonNames[ii]
    msdName = msdNames[ii]
    
    res_emsd = tp.motion.emsd(df, UmPerPix, FPS, max_lagtime=40).reset_index()
    res_emsd.to_csv(os.path.join(dstDir, msdName), index=False, sep='\t')
    
    T, MSD = res_emsd['lagt'], res_emsd['msd']
    
    parms, results = ufun.fitLineHuber(T, MSD, with_intercept = False)
    D_linear = parms.values[0]/4
    
    parms, results = ufun.fitLineHuber(np.log(T), np.log(MSD), with_intercept = True)
    b, a = parms
    k_full = a
    D_full = np.exp(b)/4

    parms, results = ufun.fitLineHuber(np.log(T[:4]), np.log(MSD[:4]), with_intercept = True)
    b, a = parms
    k_f4 = a
    D_f4 = np.exp(b)/4

    parms, results = ufun.fitLineHuber(np.log(T[-15:]), np.log(MSD[-15:]), with_intercept = True)
    b, a = parms
    k_l15 = a
    D_l15 = np.exp(b)/4
    
    dict_MSDfits = {'D_linear': D_linear,
                'k_full': k_full,
                'D_full': D_full,
                'k_f4': k_f4,
                'D_f4': D_f4,
                'k_l15': k_l15,
                'D_l15': D_l15,}
    
    ufun.dict2json(dict_MSDfits, dstDir, jsonName)


# %%%% Import MSD & plot

# for ii in range(len(msdNames)):
#     df = pd.read_csv(os.path.join(dstDir, dfNames[ii]), sep='\t')
#     res_emsd = pd.read_csv(os.path.join(dstDir, msdNames[ii]), sep='\t')
#     T, MSD = res_emsd['lagt'], res_emsd['msd']
    
#     dict_MSDfits = ufun.json2dict(dstDir, jsonNames[ii])
#     D_linear = dict_MSDfits['D_linear']
#     k_full = dict_MSDfits['k_full']
#     D_full = dict_MSDfits['D_full']
#     k_f4 = dict_MSDfits['k_f4']
#     D_f4 = dict_MSDfits['D_f4']
#     k_l15 = dict_MSDfits['k_l15']
#     D_l15 = dict_MSDfits['D_l15']
    
#     Tc = (D_f4/D_l15)**(1/(k_l15-k_f4))
    
#     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#     ax.set_xscale('log')
#     ax.set_yscale('log')

#     Xp = np.array([1e-2, 1e2])

#     ax.plot(T, MSD, 'wo', mec='k')
#     # ax.plot(Xp, 4*D_full*(Xp**k_full), ls='-', color=pm.cL_Set21[0], mec='k', label='Full curve')
#     ax.plot(Xp, 4*D_f4*(Xp**k_f4), ls='-', color=pm.cL_Set21[1], mec='k', 
#             label=f'First 4 pts\n$\\alpha$ = {k_f4:.2f}')
#     ax.plot(Xp, 4*D_l15*(Xp**k_l15), ls='-', color=pm.cL_Set21[2], mec='k', 
#             label=f'Last 15 pts\n$\\alpha$ = {k_l15:.2f}')
#     ax.axvline(Tc, color='gray', lw=1.5, label=f'$T_c$ = {Tc:.2f}')
#     ax.legend()
#     ax.grid()
#     ax.set_xlim([0.5e-1, 2e1])
#     ax.set_ylim([0.5e-3, 2e0])
#     ax.set_ylabel('MSD (um²)')
#     ax.set_xlabel('T (s)')
#     plt.show()
    
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_xscale('log')
ax.set_yscale('log')

for ii in range(len(msdNames)):
    df = pd.read_csv(os.path.join(dstDir, dfNames[ii]), sep='\t')
    res_emsd = pd.read_csv(os.path.join(dstDir, msdNames[ii]), sep='\t')
    T, MSD = res_emsd['lagt'], res_emsd['msd']
    
    dict_MSDfits = ufun.json2dict(dstDir, jsonNames[ii])
    D_linear = dict_MSDfits['D_linear']
    k_full = dict_MSDfits['k_full']
    D_full = dict_MSDfits['D_full']
    k_f4 = dict_MSDfits['k_f4']
    D_f4 = dict_MSDfits['D_f4']
    k_l15 = dict_MSDfits['k_l15']
    D_l15 = dict_MSDfits['D_l15']
    
    Tc = (D_f4/D_l15)**(1/(k_l15-k_f4))  

    

    ax.plot(T, MSD, ls='', marker='o', color=pm.cL_Set21[ii], alpha=0.5,
            markersize=4, label=msdNames[ii].split('_')[2])
    

Xp1 = np.array([1e-1, 5e-1])
Xp2 = np.array([1, 2])
ax.plot(Xp1, 9e-3*Xp1**0.5, color = 'gray', ls=':', label='$x^{1/2}$')
ax.plot(Xp2, 0.15e-1*Xp2**1, color = 'gray', ls='--', label='$x^{1}$')
ax.legend(edgecolor='None')
ax.grid()
ax.set_xlim([0.8e-1, 0.5e1])
ax.set_ylim([2e-3, 0.4e0])
ax.set_ylabel('MSD (um²)')
ax.set_xlabel('T (s)')
plt.show()

# %%%% Import tracks & analyse shape of explored zone

pm.setGraphicOptions(mode='screen')
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim([0, 250])
ax.set_ylim([0, 50])

colors = pm.cL_Set21

for ii in range(0, 1):   
    dfName = dfNames[ii]
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    Lp = df['particle'].unique().astype(int)
    for j in Lp[:3]:
        c = colors[j%len(colors)]
        df_j = df[df['particle']==j]
        # x, y = 
        points = np.array([[x, y] for (x, y) in zip(df_j.x, df_j.y)])
        hull = ConvexHull(points)
        hull_xy = [[float(points[i, 0]) for i in hull.vertices],
                   [float(points[i, 1]) for i in hull.vertices]]
        # ax.plot(points[:,0], points[:,1], 
        #         c=c, ls='', marker='.')
        Hx_plot = hull_xy[0] + [hull_xy[0][0]]
        Hy_plot = hull_xy[1] + [hull_xy[1][0]]
        ax.plot(Hx_plot, Hy_plot,
                c=c, marker='.', ls='', ) #ls='-', lw=1)
        
        [M, m, cx, cy, phi] = ufun.fitEllipse(np.array(hull_xy[0]), 
                                              np.array(hull_xy[1]))
        print(m)

        theta = np.linspace(0, 2*np.pi, 360)
        xp = cx + M*np.cos(phi)*np.cos(theta) - m*np.sin(phi)*np.sin(theta)
        yp = cy + M*np.cos(phi)*np.sin(theta) + m*np.sin(phi)*np.cos(theta)
        ax.plot(xp, yp, 'k-', lw=0.5)
        
        # Ell = mpl.patches.Ellipse(xy=(50, 20),
        #         width=10, height=5,
        #         angle=1 * (180/np.pi),
        #         facecolor='None', edgecolor='k', lw=0.5)
        # ax.add_patch(Ell)
        
plt.show()

# %%%% Import tracks & run pcf2d

pm.setGraphicOptions(mode='screen')

for ii in range(5, 10):   
    dfName = dfNames[ii]
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    title = '_'.join(dfName.split('_')[:5])
    
    for jj in [1000]:
        df_j = df[df['frame'] == jj]
        title_j = title + f' - Frame no {jj:.0f}'
        fName_j = title + f'_Fn{jj:.0f}_PCF.png'
        XX, YY = df_j.x*UmPerPix, df_j.y*UmPerPix
        XXYY = np.array([XX, YY]).T
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='compressed')
        ax = axes[0]
        ax.axis('equal')
        ax.plot(XX, YY, ls='', marker='.')
        ax.set_xlabel(r'$x\ (\mu m)$')
        ax.set_ylabel(r'$y\ (\mu m)$')
        ax.set_xlim([0, 512*UmPerPix])
        ax.set_ylim([0, 512*UmPerPix])
        
        array_positions = XXYY
        bins_distances = np.arange(0, 15, 0.2)
        
        out = tbsa.pcf2d(array_positions, bins_distances, 
                  coord_border=None, coord_holes=None, fast_method=False,
                  show_timing=False, plot=False, full_output=False)
        
        (g_of_r_normalized, radii) = out
        N_of_r_normalized = 2*np.pi * np.array([np.sum(radii[:k]*g_of_r_normalized[:k]) for k in range(len(g_of_r_normalized))])
        
        ax = axes[1]
        ax.plot(radii, g_of_r_normalized, color=pm.cL_Set2[0])
        ax.set_xlabel(r'$r\ (\mu m)$')
        ax.set_ylabel(r'$G(r)$')
        # ax.grid()
        ax.axhline(1, linestyle=':', color='gray')

        
        ax = axes[2]
        ax.set_xscale('log')
        ax.set_yscale('log')
        idx_start_fit = 30
        

        ax.plot(radii[:], N_of_r_normalized[:],
                'k.', label='Data')
        
        xfit = np.log(radii[idx_start_fit:])
        yfit = np.log(N_of_r_normalized[idx_start_fit:])
        parms, res = ufun.fitLineHuber(xfit, yfit)
        b, a = parms
        k, A = a, np.exp(b)
        xx = radii[idx_start_fit:]
        ax.plot(xx, A*xx**k, lw=1.5,
                label=r'Fit $y=Ax^k$' + f'\nk={k:.2f}')
        ax.legend()
        ax.grid()
        ax.set_xlabel(r'$r\ (\mu m)$')
        ax.set_ylabel(r'$N(r)$')
        
        fig.suptitle(title_j)
        figpath = os.path.join(dstDir, fName_j)
        fig.savefig(figpath, dpi=500, )
        
        
    
# %%% Compare MSD for DDM and SPT

#### PATHS

mainDir = 'C:\\Users\\Joseph\\Desktop\\IntraCellTracking\\26-07-29_FastAcq_NBYolk-Fecondation'
srcDir = os.path.join(mainDir, 'Crops')
dstSPTDir = os.path.join(mainDir, 'SPT_results')
dstDDMDir = os.path.join(mainDir, 'DDM_results')

tifNames = ['26-07-29_PostF_2min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_6min30_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_12min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_30min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_45min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_70min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_80min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_100min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_120min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
             ]
tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
xmlNames = [tifName.split('.')[0] + '_PyTracks.xml' for tifName in tifNames]
xmlPaths = [os.path.join(dstDir, xmlName)  for xmlName in xmlNames]

#### SETTINGS

UmPerPix = cd.UmPerPix_60X_W1
SCALE = 1/UmPerPix

nbimages = 2000
FPS = 10
frequencies = [10] * len(tifNames)

maxNCouples = 300
N_pix = 512
L_um = N_pix*UmPerPix
print(f'Pixel size = {UmPerPix:.3f} µm',
      f'Optical resol = {0.647/(2*1.2):.3f} µm') # Lambda / 2.NA
dL = min(UmPerPix, 0.647/(2*1.2))
dq = 2*np.pi / L_um
qmin = 5*dq
qmax = ((2*np.pi) / (2*dL)) * 0.4  # 11.7

#### MORE PATHS

ddmFileNames = []
dtFileNames = []
for fN, f in zip(tifNames, frequencies):
    ddmFileNames.append('_'.join(fN.split('_')[:-1]) + f'_Nc{maxNCouples:.0f}_DDM.npy')
    dtFileNames.append('_'.join(fN.split('_')[:-1]) + f'_Nc{maxNCouples:.0f}_dt.npy')
    
frequencies = [10] * len(DDMs)

#### RUN

for ii in [0, 2, 4]:
    print(fN.split('_')[2])
    fN = tifNames[ii]
    DDMname = ddmFileNames[ii]
    dtname = dtFileNames[ii]
    DDM = np.load(os.path.join(dstDDMDir, DDMname))
    dt = np.load(os.path.join(dstDDMDir, dtname))
    
    dict_MSDfits = ufun.json2dict(dstDir, jsonNames[ii])
    res_emsd = pd.read_csv(os.path.join(dstDir, msdNames[ii]), sep='\t')
    
    QQ_raw = np.arange(1, 1+DDM.shape[1])*dq
    
    valid_iQ, valid_Q = [], []
    for iq in range(len(QQ_raw)):
        q = QQ_raw[iq]
        if q >= qmin and q < qmax:
            valid_Q.append(q)
            valid_iQ.append(iq)
    
    QQ = np.array(valid_Q)
    iQ = np.array(valid_iQ)
    
    #### DDM
    
    AA, BB, GG = fitBrownianModel(DDM, dt, QQ, iQ, fN)
    
    
    #### MSD
    
    T, MSD = res_emsd['lagt'], res_emsd['msd']
    
    D_linear = dict_MSDfits['D_linear']
    k_full = dict_MSDfits['k_full']
    D_full = dict_MSDfits['D_full']
    k_f4 = dict_MSDfits['k_f4']
    D_f4 = dict_MSDfits['D_f4']
    k_l15 = dict_MSDfits['k_l15']
    D_l15 = dict_MSDfits['D_l15']
    Tc = (D_f4/D_l15)**(1/(k_l15-k_f4))  
    
    #### PLOT
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.grid()
    fig.suptitle(fN.split('_')[2])
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    Xp = np.array([1e-3, 1e3])
    
    ax.plot(T, MSD, 'wo', mec='k', label='MSD from SPT')
    # ax.plot(Xp, 4*D_full*(Xp**k_full), ls='-', color=pm.cL_Set21[0], mec='k', label='Full curve')
    # ax.plot(Xp, 4*D_f4*(Xp**k_f4), ls='-', color=pm.cL_Set21[1], mec='k', 
    #         label=f'First 4 pts\n$\\alpha$ = {k_f4:.2f}')
    # ax.plot(Xp, 4*D_l15*(Xp**k_l15), ls='-', color=pm.cL_Set21[2], mec='k', 
    #         label=f'Last 15 pts\n$\\alpha$ = {k_l15:.2f}')
    # ax.axvline(Tc, color='gray', lw=1.5, label=f'$T_c$ = {Tc:.2f}')
    ax.legend()
    ax.grid()
    ax.set_xlim([0.5e-1, 2e1])
    ax.set_ylim([0.5e-3, 2e0])
    ax.set_ylabel('MSD (um²)')
    ax.set_xlabel('T (s)')
    plt.show()
    
    
    idx = slice(10, len(iQ), 10)
    cmap = mpl.cm.plasma
    
    # idx = slice(0, len(valid_iQ), 10)
    
    list_MSD_exp = []
    list_MSD_fit = []
    
    for iq in iQ[idx]:
        jq = iq - min(iQ)
        q = QQ[iq]
        A = AA[jq]
        B = BB[jq]
        G = GG[jq]
        
        D = DDM[:, iq]
        color = cmap(jq/(len(iQ)))
        
        fR = 1 - ((D-B)/A)
        fR_fit = np.exp(-G*dt)
        
        MSD_exp = -(4/q**2) * np.log(fR)
        MSD_fit = -(4/q**2) * np.log(fR_fit)
        
        list_MSD_exp.append(MSD_exp)
        list_MSD_fit.append(MSD_fit)
        
        
    list_MSD_exp = np.array(list_MSD_exp)
    list_MSD_fit = np.array(list_MSD_fit)
    
    avg_MSD_exp = np.nanmean(list_MSD_exp, axis=0)
    avg_MSD_fit = np.nanmean(list_MSD_fit, axis=0)
    
    ax = ax
    ax.plot(dt, avg_MSD_exp, ls='', marker='o', color = 'k', label='MSD from DDM')
    # ax.plot(dt, avg_MSD_fit, ls='-', marker='', color = 'k')
    ax.legend()
    ax.grid()
        
    
    plt.show()





# %% -----------------------





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
#   
# list_D.append(D_nl)
    
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

