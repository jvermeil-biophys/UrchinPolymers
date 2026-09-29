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
import matplotlib as mpl
import statsmodels.api as sm
import matplotlib.pyplot as plt


from scipy.optimize import curve_fit


# import Libs.PlotMaker as pm
# import Libs.UrchinPaths as up
# import Libs.CalibrationData as cd
# import Libs.UtilityFunctions as ufun
import ToolboxDDM as tbDDM

# %% Helper functions

def fitLineHuber(X, Y, with_wlm_results = False, with_intercept = True):
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
    if with_intercept:
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


# %% -----------------------

# %%% 1. DDM 

# %%%% 1.1 Settings

# Paths
mainDir = "C://Users//Joseph//Desktop//FilmsFromNishant"
srcDir = mainDir # source directory for films
dstDir = os.path.join(mainDir, 'DDM_results') # destination directory for results

# Names and paths of tif images
tifNames = [
            '1.5xMT-1.5xKin-2prcnt-pluronic_2.tif',
            ]

tifPaths = [
            os.path.join(srcDir, tifName) for tifName in tifNames
            ]


# Metadata of the films
UmPerPix = 1
frequencies = [1/30] * len(tifNames)
nbimages = 211
N_pix = 2048

# Settings of the DDM analysis
pointsPerDecade = 15
maxNCouples = 20 
#maxNCouples = 10 for fast evaluation, increase it for a more accurate analysis

# Compute resolution in the direct space (dL) and Fourrier space (dq)
L_um = N_pix*UmPerPix
dL = UmPerPix
dq = 2*np.pi / L_um

# Set qmin and qmax (range in which the analysis is done)
qmin = 5*dq
qmax = ((2*np.pi) / (2*dL)) * 0.4  # 11.7
qmax = 0.5 # Force a value of qmax


# Set path for results files
ddmFileNames = []
dtFileNames = []
for fN, f in zip(tifNames, frequencies):
    ddmFileNames.append('_'.join(fN.split('_')[:-1]) + f'_Nc{maxNCouples:.0f}_DDM.npy')
    dtFileNames.append('_'.join(fN.split('_')[:-1]) + f'_Nc{maxNCouples:.0f}_dt.npy')


# %%%% 1.2 Compute DDM and save results

idts = tbDDM.logSpaced(nbimages, pointsPerDecade)
dts = [idts/float(freq) for freq in frequencies]

DDMs = []
for p in tifPaths:
    print(f'\n\nAnalyzing {os.path.split(p)}...')
    DDM = tbDDM.ddm(tbDDM.ImageStack(p), idts, maxNCouples)
    DDMs.append(DDM)
    
for ddmN, dtN, D, dt in zip(ddmFileNames, dtFileNames, DDMs, dts):
    np.save(os.path.join(dstDir, ddmN), D)
    np.save(os.path.join(dstDir, dtN), dt)


# %%%% 1.3 Load DDM results

srcDir = os.path.join(mainDir, 'DDM_results')
DDMs = [np.load(os.path.join(srcDir, fN)) for fN in ddmFileNames]
dts = [np.load(os.path.join(srcDir, fN)) for fN in dtFileNames]

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

idts = tbDDM.logSpaced(nbimages, pointsPerDecade)
dts = [idts/float(freq) for freq in frequencies]

Nstep = 20
maxNCouples=20

for p in tifPaths:
    print(f'\n\nPlotting for {os.path.split(p)}...')
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), layout='compressed')
    
    stack = tbDDM.ImageStack(p) #, convert_to_8bits=True)
    
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
    I_0_N = np.fft.fftshift(tbDDM.spectrumDiff(stack[0], stack[(Nstep)-1]))
    I_0_10N = np.fft.fftshift(tbDDM.spectrumDiff(stack[0], stack[(Nstep*5)-1]))
    I_0_100N = np.fft.fftshift(tbDDM.spectrumDiff(stack[0], stack[(Nstep*10)-1]))
    v1, v2, v3 = np.percentile(I_0_N, 1), np.percentile(I_0_10N, 1), np.percentile(I_0_100N, 1)
    V1, V2, V3 = np.percentile(I_0_N, 99), np.percentile(I_0_10N, 99), np.percentile(I_0_100N, 99)
    axes[0, 1].imshow(I_0_N, 'hot', norm=mpl.colors.LogNorm(vmin=v1, vmax=V1))
    axes[0, 1].set_title(r'$TF[\Delta I]$ ' + f'for f0 and f{F1:.0f}')
    axes[1, 1].imshow(I_0_10N, 'hot', norm=mpl.colors.LogNorm(vmin=v2, vmax=V2, ))
    axes[1, 1].set_title(r'$TF[\Delta I]$ ' + f'for f0 and f{F2:.0f}')
    axes[2, 1].imshow(I_0_100N, 'hot', norm=mpl.colors.LogNorm(vmin=v3, vmax=V3, ))
    axes[2, 1].set_title(r'$TF[\Delta I]$ ' + f'for f0 and f{F3:.0f}')
    # print(f"{np.percentile(I_0_N, 99):.2e}")
    # print(f"{np.percentile(I_0_10N, 99):.2e}")
    # print(f"{np.percentile(I_0_100N, 99):.2e}")
    
    S1, S2, S3 = Nstep//5, Nstep, Nstep*5
    J_0_N10   = tbDDM.timeAveraged(stack, Nstep//5, maxNCouples=maxNCouples)
    J_0_N  = tbDDM.timeAveraged(stack, Nstep, maxNCouples=maxNCouples)
    J_0_10N = tbDDM.timeAveraged(stack, Nstep*5, maxNCouples=maxNCouples)
    v1, v2, v3 = np.percentile(J_0_N10, 1), np.percentile(J_0_N, 1), np.percentile(J_0_10N, 1)
    V1, V2, V3 = np.percentile(J_0_N10, 99), np.percentile(J_0_N, 99), np.percentile(J_0_10N, 99)
    axes[0, 2].imshow(np.fft.fftshift(J_0_N10), 'hot', norm=mpl.colors.LogNorm(vmin=v1, vmax=V1, ))
    axes[0, 2].set_title(r'$TF[\Delta I]$ for $\Delta t$ = ' + f'{S1:.0f}f')
    axes[1, 2].imshow(np.fft.fftshift(J_0_N), 'hot', norm=mpl.colors.LogNorm(vmin=v2, vmax=V2, ))
    axes[1, 2].set_title(r'$TF[\Delta I]$ for $\Delta t$ = ' + f'{S2:.0f}f')
    axes[2, 2].imshow(np.fft.fftshift(J_0_10N), 'hot', norm=mpl.colors.LogNorm(vmin=v3, vmax=V3, ))
    axes[2, 2].set_title(r'$TF[\Delta I]$ for $\Delta t$ = ' + f'{S3:.0f}f')
    
    ra = tbDDM.RadialAverager(stack.shape[1:])
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
    
    #### SAVE
    figfile = f'{os.path.split(p)[-1]}'[:-4] + '_summary.png'
    figpath = os.path.join(srcDir, figfile)
    fig.suptitle(f'Plotting for {os.path.split(p)[-1]}')
    fig.savefig(figpath, dpi=500, )
    plt.show()


# %%%% 1.4 Plot the structure matrix D

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



# %%%% 1.5 Plot estimates of A and B

DDM_plot = DDMs[0][:, iQ]
dt_plot = dts[0]

ApB_est = np.median(DDM_plot[-4:,:], axis=0)
B_est = np.median(DDM_plot[:6,:], axis=0)

fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
for ax in axes:
    ax.set_xscale('log')
    ax.set_yscale('log')
ax = axes[0]
ax.set_title('A(q) + B(q)')
ax.plot(QQ, ApB_est, 'r.')
ax.axvline(qmin, color='gray', ls='-', alpha=0.7)
ax.axvline(qmax, color='gray', ls='-', alpha=0.7)
ax = axes[1]
ax.set_title('B(q)')
ax.plot(QQ, B_est,'k.')
ax.axvline(qmin, color='gray', ls='-', alpha=0.7)
ax.axvline(qmax, color='gray', ls='-', alpha=0.7)
ax = axes[2]
ax.set_title('A(q)')
ax.plot(QQ, ApB_est-B_est,'b.')
ax.axvline(qmin, color='gray', ls='-', alpha=0.7)
ax.axvline(qmax, color='gray', ls='-', alpha=0.7)

plt.show()


# %%%% 1.6 Use a model to fit A, B and get f (Brownian case)

AA, BB, GG = [], [], []
kk_G = []
AA_G = []

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
    
    # If the fit for B is bullshit,
    # set this to true to force B to a given set of values
    FORCE_B = True
    
    # Option 1
    forced_B = [np.percentile(B_est, 0)/10000000] * len(QQ)
    
    # Option 2
    # forced_B = B_est
    
    # Option 3
    # MB = np.median(B_est[:3])
    # mB = np.median(B_est[-3:])
    # MQ = np.max(QQ)
    # mQ = np.min(QQ)
    # k = (np.log(MB)-np.log(mB)) / (np.log(mQ)-np.log(MQ))
    # A = mB / (MQ**k)
    # forced_B = [A * q**k for q in QQ]
    
    # Option 4
    # logQQ = np.log(QQ)
    # logBest = np.log(B_est)
    # p_fitted = np.polynomial.Polynomial.fit(logQQ, logBest, deg=2)
    # B_smooth = [np.exp(p_fitted(q)) for q in logQQ]
    # forced_B = B_smooth
    
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
    params, results = fitLineHuber(X, Y)
    (p1, p2) = params
    k = p2
    A = np.exp(p1)
    
    kk_G.append(k)
    AA_G.append(A)
    
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
    
    figName = '.'.join(fN.split('.')[:-1])
    print(figName)
    figfile = f'{figName}' + '_BrownianFit.png'
    figpath = os.path.join(srcDir, figfile)
    fig.savefig(figpath, dpi=500, )

    
    plt.show()



    

# %%%% 1.7 Plot the fit



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
    
    figName = '.'.join(fN.split('.')[:-1])
    print(figName)
    figfile = f'{figName}' + '_ACF_Brownian.png'
    figpath = os.path.join(srcDir, figfile)
    fig.savefig(figpath, dpi=500, )
    
    
    