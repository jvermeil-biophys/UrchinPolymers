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



# %% Utility functions

def Tpf_str2num(tpf_str):
    L = tpf_str.split('min')
    tpf_num = int(L[0])*60
    if len(L) > 1 and len(L[1]) > 0:
        tpf_num += int(L[1])
    return(tpf_num)

def distribute_in_boxes(df, L, M, 
                        str_Id = 'Id', str_X = 'X', str_Y = 'Y'):
    box_size = L / M
    
    # Compute box indices
    cols = np.floor(df[str_X] / box_size).astype(int)
    rows = np.floor(df[str_Y] / box_size).astype(int)
    
    # Handle points exactly at X=N or Y=N
    cols = np.minimum(cols, M - 1)
    rows = np.minimum(rows, M - 1)
    
    # Group IDs by box
    result = (
        df.assign(row=rows, col=cols)
          .groupby(["row", "col"])[str_Id]
          .apply(list)
          .to_dict()
    )
    
    return(result)


def df_2_heatmap(df, ax, parmCol='k', boxCol='Bxy', y_ascending=False,
                 cmap="viridis", annotate=False, colorScale='linear'):
    """
    Plot k values on an M x M grid.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns parmCol and BoxCol.
        BoxCol contains tuples (x, y).
    M : int
        Grid size.
    cmap : str
        Matplotlib colormap name.
    annotate : bool
        Whether to display k values in cells.
    """
    
    M = np.max(np.array(df[boxCol].tolist())) + 1
    
    # Extract coordinates
    coords = pd.DataFrame(
        df[boxCol].tolist(),
        columns=["x", "y"],
        index=df.index
    )

    data = pd.concat(
        [coords, df[parmCol]],
        axis=1
    )

    # Convert to matrix
    matrix = data.pivot(
        index="y",
        columns="x",
        values=parmCol
    )

    # Ensure all M x M boxes are represented
    matrix = matrix.reindex(
        index=range(M),
        columns=range(M)
    )

    # Plot
    if colorScale == 'linear':
        sns.heatmap(
            matrix,
            cmap=cmap,
            square=True,
            annot=annotate,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": parmCol,
                      "shrink": .5},
            ax=ax
        )
    elif colorScale == 'log':
        sns.heatmap(
            matrix,
            cmap=cmap,
            square=True,
            annot=annotate,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": parmCol,
                      "shrink": .5},
            ax=ax,
            norm=mpl.colors.LogNorm(
                vmin=matrix.min().min(),
                vmax=matrix.max().max()
                )        
        )
    
    if y_ascending:
        ax.invert_yaxis()

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Heatmap of {parmCol}")

    return(ax)







def autocorr_fft(x):
    N = len(x)
    F = np.fft.fft(x, n = 2*N)  # 2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real  # now we have the autocorrelation in convention B
    n = N * np.ones(N) - np.arange(0, N) # divide res(m) by (N-m)
    return(res / n)  # this is the autocorrelation in convention A

def msd_fft_trackpyStyle(traj, mpp, fps, max_lagtime=100, pos_columns=['x', 'y']):
    """
    https://github.com/hadim/Public-Notebooks/blob/master/Code/Quick_MSD/notebook.ipynb
    """
    
    r = traj[pos_columns].values
    r *= mpp

    t = traj['frame']

    max_lagtime = min(max_lagtime, len(t))  # checking to be safe
    lagtimes = 1 + np.arange(max_lagtime - 1)    

    N = len(r)

    D = np.square(r).sum(axis=1) 
    D = np.append(D, 0)
    S2 = sum([autocorr_fft(r[:, i]) for i in range(len(pos_columns))])

    Q = 2 * D.sum()
    S1 = np.zeros(max_lagtime)

    for m in range(max_lagtime):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)

    msd = S1 - 2 * S2[:max_lagtime]
    msd = msd[1:]

    lagt = lagtimes / fps

    results = pd.DataFrame(np.array([msd, lagt]).T, columns=['msd', 'lagt'])
    results.index = 1 + np.arange(max_lagtime - 1)
    results.index.name = 'lagt'
    
    return(results)


def msd_fft_1D(pos, mpp, fps, max_lagtime=100):
    """
    https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft/34222273#34222273
    """
    
    r = pos
    r *= mpp

    t = np.arange(len(pos))

    max_lagtime = min(max_lagtime, len(t))  # checking to be safe
    lagtimes = 1 + np.arange(max_lagtime)    

    N = len(r)

    D = np.square(r)
    D = np.append(D, 0)
    S2 = sum([autocorr_fft(r[:])])

    Q = 2 * D.sum()
    S1 = np.zeros(max_lagtime+1)

    for m in range(max_lagtime+1):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)

    msd = S1 - 2 * S2[:max_lagtime+1]
    msd = msd[1:]

    lagt = lagtimes / fps

    results = pd.DataFrame(np.array([msd, lagt]).T, columns=['msd', 'lagt'])
    results.index = 1 + np.arange(max_lagtime)
    results.index.name = 'lagt'
    
    return(results)


# %% Test MSD computation

mainDir = os.path.join(up.Path_IntraCellTracking, '26-07-29_FastAcq_Fec_NB-Yolk')
srcDir = os.path.join(mainDir, 'Crops')
dstDir = os.path.join(mainDir, 'SPT_results')

tifNames = ['26-07-29_PostF_2min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
             ]

tifPaths = [os.path.join(srcDir, tifName) for tifName in tifNames]
dfNames = [tifName.split('.')[0] + '_PyTracks.csv' for tifName in tifNames]

UmPerPix = cd.UmPerPix_60X_W1
SCALE = 1/UmPerPix
nbimages = 2000
FPS = 10

N_pix = 512
C_pix = np.median(np.arange(N_pix)) # Center (pixels)
L_um = N_pix*UmPerPix


dfName = dfNames[0]

df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
PIDs = df.particle.unique().astype(int)

for p in PIDs[:3]:
    
    df_p = df[df['particle'] == p]
    
    MSD_tpRes = tp.motion.msd(df_p, UmPerPix, FPS, max_lagtime=30, detail=True, pos_columns=None)

    MSD_1DRes = msd_fft_1D(df_p.x, UmPerPix, FPS, max_lagtime=30)

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

DDM_fit = DDMs[0]
dt_fit = dts[0]

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

DDM_fit = DDMs[0]
dt_fit = dts[0]
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
            '26-07-29_PostF_20min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_30min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_45min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_60min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            '26-07-29_PostF_70min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
            ]
#             '26-07-29_PostF_80min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
#             '26-07-29_PostF_100min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
#             '26-07-29_PostF_120min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
#              ]


# tifNames = [
#             '26-07-29_PostF_20min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
#             '26-07-29_PostF_60min_Pos11_10fps_Texp100ms_CSU642_crop.tif',
#             ]

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
C_pix = np.median(np.arange(N_pix)) # Center (pixels)
L_um = N_pix*UmPerPix
# print(f'Pixel size = {UmPerPix:.3f} µm',
#       f'Optical resol = {0.647/(2*1.2):.3f} µm') # Lambda / 2.NA
# dL = min(UmPerPix, 0.647/(2*1.2))
# dq = 2*np.pi / L_um 
# qmin = 5*dq
# qmax = ((2*np.pi) / (2*dL)) * 0.4  # 11.7




max_lagtime = 50
lowDt_upper = 0.5
highDt_lower = 1.0





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
    
    res_emsd = tp.motion.emsd(df, UmPerPix, FPS, max_lagtime=max_lagtime).reset_index()
    res_emsd.to_csv(os.path.join(dstDir, msdName), index=False, sep='\t')
    
    T, MSD = res_emsd['lagt'].values, res_emsd['msd'].values
    iLow = ufun.findFirst(lowDt_upper, T) + 1
    iHigh = ufun.findFirst(highDt_lower, T)
    
    parms, results = ufun.fitLineHuber(T, MSD, with_intercept = False)
    D_linear = parms[0]/4
    
    parms, results = ufun.fitLineHuber(np.log(T), np.log(MSD), with_intercept = True)
    b, a = parms
    k_full = a
    D_full = np.exp(b)/4

    parms, results = ufun.fitLineHuber(np.log(T[:iLow]), np.log(MSD[:iLow]), with_intercept = True)
    b, a = parms
    k_lowDt = a
    D_lowDt = np.exp(b)/4

    parms, results = ufun.fitLineHuber(np.log(T[iHigh:]), np.log(MSD[iHigh:]), with_intercept = True)
    b, a = parms
    k_highDt = a
    D_highDt = np.exp(b)/4
    
    dict_MSDfits = {'D_linear': D_linear,
                    'k_full': k_full,
                    'D_full': D_full,
                    'k_lowDt': k_lowDt,
                    'D_lowDt': D_lowDt,
                    'k_highDt': k_highDt,
                    'D_highDt': D_highDt,}
    
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
#     k_lowDt = dict_MSDfits['k_lowDt']
#     D_lowDt = dict_MSDfits['D_lowDt']
#     k_highDt = dict_MSDfits['k_highDt']
#     D_highDt = dict_MSDfits['D_highDt']
    
#     Tc = (D_lowDt/D_highDt)**(1/(k_highDt-k_lowDt))
    
#     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#     ax.set_xscale('log')
#     ax.set_yscale('log')

#     Xp = np.array([1e-2, 1e2])

#     ax.plot(T, MSD, 'wo', mec='k')
#     # ax.plot(Xp, 4*D_full*(Xp**k_full), ls='-', color=pm.cL_Set21[0], mec='k', label='Full curve')
#     ax.plot(Xp, 4*D_lowDt*(Xp**k_lowDt), ls='-', color=pm.cL_Set21[1], mec='k', 
#             label=f'First 4 pts\n$\\alpha$ = {k_lowDt:.2f}')
#     ax.plot(Xp, 4*D_highDt*(Xp**k_highDt), ls='-', color=pm.cL_Set21[2], mec='k', 
#             label=f'Last 15 pts\n$\\alpha$ = {k_highDt:.2f}')
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

dict_res = {'label':[],
            'D_full':[],
            'k_full':[],
            'D_lowDt':[],
            'k_lowDt':[],
            'D_highDt':[],
            'k_highDt':[],
            }

for ii in range(len(msdNames)):
    df = pd.read_csv(os.path.join(dstDir, dfNames[ii]), sep='\t')
    res_emsd = pd.read_csv(os.path.join(dstDir, msdNames[ii]), sep='\t')
    T, MSD = res_emsd['lagt'], res_emsd['msd']
    
    dict_MSDfits = ufun.json2dict(dstDir, jsonNames[ii])
    D_linear = dict_MSDfits['D_linear']
    k_full = dict_MSDfits['k_full']
    D_full = dict_MSDfits['D_full']
    k_lowDt = dict_MSDfits['k_lowDt']
    D_lowDt = dict_MSDfits['D_lowDt']
    k_highDt = dict_MSDfits['k_highDt']
    D_highDt = dict_MSDfits['D_highDt']
    
    Tc = (D_lowDt/D_highDt)**(1/(k_highDt-k_lowDt))  

    label = msdNames[ii].split('_')[2]

    ax.plot(T, MSD, label=label, color=pm.cL_Set21[ii], 
            # ls='-', lw = 1, alpha=0.85,)
            marker='o', markersize=3, alpha=0.85)
    
    dict_res['label'].append(label)
    dict_res['D_full'].append(D_full)
    dict_res['k_full'].append(k_full)
    dict_res['D_lowDt'].append(D_lowDt)
    dict_res['k_lowDt'].append(k_lowDt)
    dict_res['D_highDt'].append(D_highDt)
    dict_res['k_highDt'].append(k_highDt)
    

Xp1 = np.array([1e-1, 5e-1])
Xp2 = np.array([1, 2])
ax.plot(Xp1, 12e-3*Xp1**0.5, color = 'gray', ls=':', label=r'$y \propto x^{1/2}$')
ax.plot(Xp2, 0.15e-1*Xp2**1, color = 'gray', ls='--', label=r'$y \propto x^{1}$')
ax.legend(edgecolor='None', title='Tpf')
ax.grid()
ax.set_xlim([0.8e-1, 0.6e1])
ax.set_ylim([2e-3, 0.5e0])
ax.set_ylabel('MSD (µm²)')
ax.set_xlabel(r'$\Delta t$ (s)')

plt.show()


df_Diffusion = pd.DataFrame(dict_res)
df_Diffusion['Tpf_s'] = df_Diffusion['label'].apply(lambda x : Tpf_str2num(x))
df_Diffusion['Tpf_min'] = df_Diffusion['Tpf_s']/60

fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex = True, layout='compressed')
ax = axes[0]
# ax.plot(np.arange(len(df_Diffusion)), df_Diffusion.D_full, ls='-', marker='o', label=r'$D_{full}$')
# ax.plot(np.arange(len(df_Diffusion)), df_Diffusion.D_lowDt, ls='-', marker='o', label=r'$D_{\Delta t \leq 0.5s}$')
# ax.plot(np.arange(len(df_Diffusion)), df_Diffusion.D_highDt, ls='-', marker='o', label=r'$D_{\Delta t \geq 1s}$')
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.D_full, ls='-', marker='o', label=r'All $\Delta t$')
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.D_lowDt, ls='-', marker='o', label=r'$\Delta t \leq 0.5s$')
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.D_highDt, ls='-', marker='o', label=r'$\Delta t \geq 1s$')
ax.set_ylabel(r'$D_{eff}\ (\mu m^2/s^\alpha)$')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1]
# ax.plot(np.arange(len(df_Diffusion)), df_Diffusion.k_full, ls='-', marker='o', label=r'$\alpha_{full}$')
# ax.plot(np.arange(len(df_Diffusion)), df_Diffusion.k_lowDt, ls='-', marker='o', label=r'$\alpha_{\Delta t \leq 0.5s}$')
# ax.plot(np.arange(len(df_Diffusion)), df_Diffusion.k_highDt, ls='-', marker='o', label=r'$\alpha_{\Delta t \geq 1s}$')
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.k_full, ls='-', marker='o', label=r'All $\Delta t$')
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.k_lowDt, ls='-', marker='o', label=r'$\Delta t \leq 0.5s$')
ax.plot(df_Diffusion.Tpf_min, df_Diffusion.k_highDt, ls='-', marker='o', label=r'$\Delta t \geq 1s$')
ax.set_ylabel(r'$\alpha$')
ax.set_xticks(df_Diffusion['Tpf_min'].values)
ax.set_xticklabels(df_Diffusion['Tpf_min'].values, rotation = 20)
ax.set_xlabel('Tpf (min)')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

plt.show()

# %%%% Import tracks -> run imsd

IMSD = []

for ii in range(len(dfNames)): # len(dfNames)
    dfName = dfNames[ii]
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    df.particle = df.particle.astype(int)
    
    res_imsd = tp.motion.imsd(df, UmPerPix, FPS, max_lagtime=50).reset_index()
    IMSD.append(res_imsd)



# %%%% From imsd -> Diffusion Map

DF_PART_MSD = []

for ii in range(len(dfNames)): # range(len(dfNames)) [2]
    im = ufun.load_stack_region(tifPaths[ii], time_indices=[0])[0]
    dfName = dfNames[ii]
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    df.particle = df.particle.astype(int)
    
    res_imsd = IMSD[ii]
    dt = res_imsd.loc[:, 'lag time [s]']
    
    dict_particle_MSD = {
                        'Pid':[],
                        'Xc':[],
                        'Yc':[],
                        'theta':[],
                        'fmin':[],
                        'fmax':[],
                        'D_lin':[],
                        'D_full':[],
                        'k_full':[],
                        }
    
    # fig, ax = plt.subplots(1, 1)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    
    Pid_list = df.particle.unique()
    for p in Pid_list[:]:
        dfp = df[df['particle']==p]
        msd = res_imsd.loc[:, p].dropna()
        # ax.plot(dt, msd, 'o', markersize=4, alpha=0.02, 
        #         color='navy', mec='None')
                
        parms, results = ufun.fitLineHuber(dt[:len(msd)], msd, 
                                           with_intercept = False)
        D_linear = parms.values[0]/4
        
        parms, results = ufun.fitLineHuber(np.log(dt[:len(msd)]), np.log(msd), 
                                           with_intercept = True)
        b, a = parms
        k_full = a
        D_full = np.exp(b)/4
        
        fmin, fmax = np.min(dfp.frame), np.max(dfp.frame)
        X, Y = dfp.x, dfp.y
        XY = np.array([X, Y]).T
        CvxHull = MultiPoint(XY).convex_hull
        Xc, Yc = CvxHull.centroid.coords[0]
        theta = np.atan2(Yc - C_pix, Xc - C_pix)
        
        dict_particle_MSD['Pid'].append(p)
        dict_particle_MSD['Xc'].append(Xc)
        dict_particle_MSD['Yc'].append(Yc)
        dict_particle_MSD['theta'].append(theta) # *180/np.pi
        dict_particle_MSD['fmin'].append(fmin)
        dict_particle_MSD['fmax'].append(fmax)
        dict_particle_MSD['D_lin'].append(D_linear)
        dict_particle_MSD['D_full'].append(D_full)
        dict_particle_MSD['k_full'].append(k_full)
        
    df_particle_MSD = pd.DataFrame(dict_particle_MSD)
    DF_PART_MSD.append(df_particle_MSD)


#### SAVE

tableNames = [tifName.split('.')[0] + '_partTrajData.csv' for tifName in tifNames]
for ii in range(len(dfNames)):
    df_particle_MSD = DF_PART_MSD[ii]
    df_particle_MSD.to_csv(os.path.join(dstDir, tableNames[ii]), sep=';', index=False)


# %%%% Plot the Map

df_centers = pd.read_csv(os.path.join(srcDir, 'OrganizingCenters.csv'), sep=';')
tableNames = [tifName.split('.')[0] + '_partTrajData.csv' for tifName in tifNames]

for ii in range(len(dfNames)): #
    try:
        t = nbimages//2
        im = ufun.load_stack_region(tifPaths[ii], time_indices=[t])[0]
        NoImg=False
    except:
        NoImg=True
    df_particle_MSD = pd.read_csv(os.path.join(dstDir, tableNames[ii]), sep=';')
    label = msdNames[ii].split('_')[2]
    
    M_boxes = 15
    L_box = N_pix/M_boxes
    
    Xc, Yc = df_centers.loc[ii, 'xc'], df_centers.loc[ii, 'yc']

    df_particle_MSD['Xb'] = (df_particle_MSD['Xc'].values//L_box).astype(int)
    df_particle_MSD['Yb'] = (df_particle_MSD['Yc'].values//L_box).astype(int)

    df_particle_MSD['Bxy'] = [(x,y) for x, y in zip(df_particle_MSD['Xb'], df_particle_MSD['Yb'])]

    grouped = df_particle_MSD.groupby('Bxy')
    df_grid_MSD = grouped.agg({'Pid':'count',
                               'D_lin':'mean',
                               'D_full':'mean',
                               'k_full':'mean',
                               }).rename(columns={'Pid':'count'}).reset_index()

    df_grid_MSD = df_grid_MSD[df_grid_MSD['count'] >= 5]

    lims = np.linspace(0, N_pix-1, (M_boxes+1))
    fig, axes = plt.subplots(2, 2, figsize = (8, 6), layout='compressed')
    axes_f = axes.flatten()

    ax = axes_f[0]
    if not NoImg:
        vmin, vmax = np.percentile(im, 0.5), np.percentile(im, 99.5)
        ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
        ax.hlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
        ax.vlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
        ax.plot(Xc, Yc, 'ro', markersize=3)
    ax.set_xlim([0, 511])
    ax.set_ylim([0, 511])
    ax.set_title(f'Tpf {label} - tiled img')

    df_2_heatmap(df_grid_MSD, axes_f[1], parmCol='k_full', boxCol='Bxy', 
                 y_ascending=True, cmap="viridis", annotate=False, colorScale='linear')

    df_2_heatmap(df_grid_MSD, axes_f[2], parmCol='D_lin', boxCol='Bxy', 
                 y_ascending=True, cmap="viridis", annotate=False, colorScale='log')

    df_2_heatmap(df_grid_MSD, axes_f[3], parmCol='D_full', boxCol='Bxy', 
                 y_ascending=True, cmap="viridis", annotate=False, colorScale='log')


    plt.show()
    





# %%%% Plot the points

df_centers = pd.read_csv(os.path.join(srcDir, 'OrganizingCenters.csv'), sep=';')
tableNames = [tifName.split('.')[0] + '_partTrajData.csv' for tifName in tifNames]

for ii in range(len(dfNames)): #
    t = nbimages//2
    im = ufun.load_stack_region(tifPaths[ii], time_indices=[t])[0]
    df_particle_MSD = pd.read_csv(os.path.join(dstDir, tableNames[ii]), sep=';')
    label = msdNames[ii].split('_')[2]


    lims = np.linspace(0, N_pix-1, (M_boxes+1))
    fig, axes = plt.subplots(2, 2, figsize = (10, 8), layout='compressed')
    axes_f = axes.flatten()
    ax = axes_f[0]
    vmin, vmax = np.percentile(im, 0.5), np.percentile(im, 99.5)
    ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
    # ax.hlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
    # ax.vlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
    ax.set_xlim([0, 511])
    ax.set_ylim([0, 511])
    ax.set_title('Original image')
    
    ax = axes_f[1]
    parm = 'k_full' # 'D_full', 'k_full'
    v_high = np.percentile(df_particle_MSD[parm], 98)
    df_f = df_particle_MSD[df_particle_MSD[parm] < v_high]
    
    g = ax.scatter(df_f['Xc'], df_f['Yc'], 
                   c=df_f[parm], cmap='viridis',
                   s = 8, alpha = 1, edgecolor='None',
                   norm=mpl.colors.Normalize(),
                   )
    cbar = fig.colorbar(g)
    ax.set_xlim([0, 511])
    ax.set_ylim([0, 511])
    ax.set_title(parm)
    
    
    ax = axes_f[2]
    parm = 'D_lin' # 'D_full', 'k_full'
    v_high = np.percentile(df_particle_MSD[parm], 98)
    df_f = df_particle_MSD[df_particle_MSD[parm] < v_high]
    
    g = ax.scatter(df_f['Xc'], df_f['Yc'], 
                   c=df_f[parm], cmap='PuRd',
                   s = 8, alpha = 1, edgecolor='None',
                   norm=mpl.colors.LogNorm(),
                   )
    cbar = fig.colorbar(g)
    ax.set_xlim([0, 511])
    ax.set_ylim([0, 511])
    ax.set_title(parm)
    
    
    ax = axes_f[3]
    parm = 'D_full' # 'D_full', 'k_full'
    v_high = np.percentile(df_particle_MSD[parm], 98)
    df_f = df_particle_MSD[df_particle_MSD[parm] < v_high]
    
    g = ax.scatter(df_f['Xc'], df_f['Yc'], 
                   c=df_f[parm], cmap='BuPu',
                   s = 8, alpha = 1, edgecolor='None',
                   norm=mpl.colors.LogNorm(),
                   )
    cbar = fig.colorbar(g)
    ax.set_xlim([0, 511])
    ax.set_ylim([0, 511])
    ax.set_title(parm)
    
    # Remove the legend and add a colorbar
    
    
    plt.show()
        


    
# %%%% Radial / OrthoRadial


max_lagtime = 30
lagT = np.arange(1, 31) / FPS

iLow = ufun.findFirst(lowDt_upper, lagT) + 1
iHigh = ufun.findFirst(highDt_lower, lagT)

df_centers = pd.read_csv(os.path.join(srcDir, 'OrganizingCenters.csv'), sep=';')

tableNames = [tifName.split('.')[0] + '_partTrajData_RandOR.csv' for tifName in tifNames]


for ii in range(len(dfNames)): # len(dfNames)
    print(ii)
    X_MTcenter, Y_MTcenter = df_centers.loc[ii, 'xc'], df_centers.loc[ii, 'yc']
    
    dfName = dfNames[ii]
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    df.particle = df.particle.astype(int)
    
    dict_particle_MSD = {
                        'Pid':[],
                        'Xc':[],
                        'Yc':[],
                        'theta':[],
                        'fmin':[],
                        'fmax':[],
                        'D_r_lin':[],
                        'D_r_full':[],
                        'k_r_full':[],
                        'D_r_highDt':[],
                        'k_r_highDt':[],
                        'D_r_lowDt':[],
                        'k_r_lowDt':[],
                        'D_or_lin':[],
                        'D_or_full':[],
                        'k_or_full':[],
                        'D_or_highDt':[],
                        'k_or_highDt':[],
                        'D_or_lowDt':[],
                        'k_or_lowDt':[],
                        }

    Pid_list = df.particle.unique()
    for p in Pid_list[:]:
        dfp = df[df['particle']==p]
        
        # msd = res_imsd.loc[:, p]
        # # ax.plot(dt, msd, 'o', markersize=4, alpha=0.02, 
        # #         color='navy', mec='None')
                
        # parms, results = ufun.fitLineHuber(dt, msd, with_intercept = False)
        # D_linear = parms.values[0]/4
        
        # parms, results = ufun.fitLineHuber(np.log(dt), np.log(msd), 
        #                                    with_intercept = True)
        # b, a = parms
        # k_full = a
        # D_full = np.exp(b)/4
        
        fmin, fmax = np.min(dfp.frame), np.max(dfp.frame)
        X, Y = dfp.x, dfp.y
        XY = np.array([X, Y]).T
        CvxHull = MultiPoint(XY).convex_hull
        Xc, Yc = CvxHull.centroid.coords[0]
        theta = np.atan2(Yc - Y_MTcenter, Xc - X_MTcenter)
        
        RM = np.array([[np.cos(theta), - np.sin(theta)], 
                       [np.sin(theta),   np.cos(theta)]])
        XY_center = np.array([X_MTcenter, Y_MTcenter])
        XY_r = ((XY - XY_center) @ RM) + XY_center
        
        # PLOT A ROTATED TRAJECTORY
        # fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))
        # ax = axes[0]
        # ax.set_aspect('equal', adjustable='box')
        # ax.plot(XY[:,0]-C_pix, XY[:,1]-C_pix, lw=1)
        # ax.plot(XY_r[:,0]-C_pix, XY_r[:,1]-C_pix, lw=1)
        # ax.set_xlim([-C_pix, C_pix])
        # ax.set_ylim([-C_pix, C_pix])
        # ax.grid()
        
        # ax = axes[1]
        # ax.set_aspect('equal', adjustable='box')
        # ax.plot(XY[:,0]-C_pix, XY[:,1]-C_pix, lw=1)
        # ax.grid()
        
        # ax = axes[2]
        # ax.set_aspect('equal', adjustable='box')
        # ax.plot(XY_r[:,0]-C_pix, XY_r[:,1]-C_pix, lw=1)
        # ax.grid()
        
        # plt.show()
        
        # Radial / Orthoradial coordinates
        Rd, ORd = XY_r[:, 0], XY_r[:, 1]
        
        # MSD 1D for each
        # lagT
        MSD_r = (msd_fft_1D(Rd, UmPerPix, FPS, max_lagtime=max_lagtime))['msd']
        MSD_or = (msd_fft_1D(ORd, UmPerPix, FPS, max_lagtime=max_lagtime))['msd']
        
        
        # Fits for D and k
        # Radial
        parms, results = ufun.fitLineHuber(lagT, MSD_r, with_intercept = False)
        D_r_linear = parms.values[0]/4
        
        parms, results = ufun.fitLineHuber(np.log(lagT), np.log(MSD_r), 
                                           with_intercept = True)
        b, a = parms
        k_r_full = a
        D_r_full = np.exp(b)/4
        
        parms, results = ufun.fitLineHuber(np.log(lagT[iHigh:]), np.log(MSD_r[iHigh:]), 
                                           with_intercept = True)
        b, a = parms
        k_r_highDt = a
        D_r_highDt = np.exp(b)/4
        
        parms, results = ufun.fitLineHuber(np.log(lagT[:iLow]), np.log(MSD_r[:iLow]), 
                                           with_intercept = True)
        b, a = parms
        k_r_lowDt = a
        D_r_lowDt = np.exp(b)/4
        
        
        # OrthoRadial
        parms, results = ufun.fitLineHuber(lagT, MSD_or, with_intercept = False)
        D_or_linear = parms.values[0]/4
        
        parms, results = ufun.fitLineHuber(np.log(lagT), np.log(MSD_or), 
                                           with_intercept = True)
        b, a = parms
        k_or_full = a
        D_or_full = np.exp(b)/4
        
        parms, results = ufun.fitLineHuber(np.log(lagT[iHigh:]), np.log(MSD_or[iHigh:]), 
                                           with_intercept = True)
        b, a = parms
        k_or_highDt = a
        D_or_highDt = np.exp(b)/4
        
        parms, results = ufun.fitLineHuber(np.log(lagT[:iLow]), np.log(MSD_or[:iLow]), 
                                           with_intercept = True)
        b, a = parms
        k_or_lowDt = a
        D_or_lowDt = np.exp(b)/4
        
        
        # Save
        dict_particle_MSD['Pid'].append(p)
        dict_particle_MSD['Xc'].append(Xc)
        dict_particle_MSD['Yc'].append(Yc)
        dict_particle_MSD['theta'].append(theta) # *180/np.pi
        dict_particle_MSD['fmin'].append(fmin)
        dict_particle_MSD['fmax'].append(fmax)
        dict_particle_MSD['D_r_lin'].append(D_r_linear)
        dict_particle_MSD['D_r_full'].append(D_r_full)
        dict_particle_MSD['k_r_full'].append(k_r_full)
        dict_particle_MSD['D_r_highDt'].append(D_r_highDt)
        dict_particle_MSD['k_r_highDt'].append(k_r_highDt)
        dict_particle_MSD['D_r_lowDt'].append(D_r_lowDt)
        dict_particle_MSD['k_r_lowDt'].append(k_r_lowDt)
        dict_particle_MSD['D_or_lin'].append(D_or_linear)
        dict_particle_MSD['D_or_full'].append(D_or_full)
        dict_particle_MSD['k_or_full'].append(k_or_full)
        dict_particle_MSD['D_or_highDt'].append(D_or_highDt)
        dict_particle_MSD['k_or_highDt'].append(k_or_highDt)
        dict_particle_MSD['D_or_lowDt'].append(D_or_lowDt)
        dict_particle_MSD['k_or_lowDt'].append(k_or_lowDt)
        
        
        
    df_particle_MSD_CylCoo = pd.DataFrame(dict_particle_MSD)
    df_particle_MSD_CylCoo.to_csv(os.path.join(dstDir, tableNames[ii]), sep=';', index=False)

    
    # dict_boxes = distribute_in_boxes(dict_particle_MSD, N_pix, M_boxes, 
    #                     str_Id = 'Pid', str_X = 'Xc', str_Y = 'Yc')
    # L = [len(dict_boxes[k]) for k in dict_boxes.keys()]

    

    # plt.show()
    
    
# %%%% Plot the Map

df = df_particle_MSD_CylCoo

M_boxes = 15
L_box = N_pix/M_boxes

df['Xb'] = (df['Xc'].values//L_box).astype(int)
df['Yb'] = (df['Yc'].values//L_box).astype(int)

df['Bxy'] = [(x,y) for x, y in zip(df['Xb'], df['Yb'])]

grouped = df.groupby('Bxy')
df_grid_MSD = grouped.agg({'Pid':'count',
                           'D_r_lin':'mean',
                           'D_r_full':'mean',
                           'k_r_full':'mean',
                           'D_or_lin':'mean',
                           'D_or_full':'mean',
                           'k_or_full':'mean',
                           }).rename(columns={'Pid':'count'}).reset_index()

df_grid_MSD = df_grid_MSD[df_grid_MSD['count'] >= 10]

lims = np.linspace(0, N_pix-1, (M_boxes+1))
fig, axes = plt.subplots(2, 3, figsize = (12, 8), layout='compressed')
axes_f = axes.flatten()

ax = axes_f[0]
vmin, vmax = np.percentile(im, 0.5), np.percentile(im, 99.5)
ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
ax.hlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
ax.vlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
ax.set_xlim([0, 511])
ax.set_ylim([0, 511])
ax.set_title('Tiled image')

df_2_heatmap(df_grid_MSD, axes_f[1], parmCol='k_r_full', boxCol='Bxy', 
             y_ascending=True, cmap="viridis", annotate=False, colorScale='linear')

df_2_heatmap(df_grid_MSD, axes_f[2], parmCol='k_or_full', boxCol='Bxy', 
             y_ascending=True, cmap="viridis", annotate=False, colorScale='linear')

df_2_heatmap(df_grid_MSD, axes_f[4], parmCol='D_r_full', boxCol='Bxy', 
             y_ascending=True, cmap="viridis", annotate=False, colorScale='log')

df_2_heatmap(df_grid_MSD, axes_f[5], parmCol='D_or_full', boxCol='Bxy', 
             y_ascending=True, cmap="viridis", annotate=False, colorScale='log')


plt.show()


# lims = np.linspace(0, N_pix-1, (M_boxes+1))
# fig, axes = plt.subplots(2, 3, figsize = (12, 8), layout='compressed')
# axes_f = axes.flatten()

# ax = axes_f[0]
# vmin, vmax = np.percentile(im, 0.5), np.percentile(im, 99.5)
# ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
# ax.hlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
# ax.vlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
# ax.set_xlim([0, 511])
# ax.set_ylim([0, 511])
# ax.set_title('Tiled image')

# df_2_heatmap(df_grid_MSD, axes_f[1], parmCol='k_r_full', boxCol='Bxy', 
#              y_ascending=True, cmap="viridis", annotate=False, colorScale='linear')

# df_2_heatmap(df_grid_MSD, axes_f[2], parmCol='k_or_full', boxCol='Bxy', 
#              y_ascending=True, cmap="viridis", annotate=False, colorScale='linear')

# df_2_heatmap(df_grid_MSD, axes_f[4], parmCol='D_r_full', boxCol='Bxy', 
#              y_ascending=True, cmap="viridis", annotate=False, colorScale='log')

# df_2_heatmap(df_grid_MSD, axes_f[5], parmCol='D_or_full', boxCol='Bxy', 
#              y_ascending=True, cmap="viridis", annotate=False, colorScale='log')


# plt.show()

# %%%% Plot the points

pm.setGraphicOptions(mode='print')

df_centers = pd.read_csv(os.path.join(srcDir, 'OrganizingCenters.csv'), sep=';')

tableNames = [tifName.split('.')[0] + '_partTrajData_RandOR.csv' for tifName in tifNames]

ii = 7

# for ii in range(len(dfNames)): # len(dfNames)
#     print(ii)
X_MTcenter, Y_MTcenter = df_centers.loc[ii, 'xc'], df_centers.loc[ii, 'yc']

df_particle_MSD_CylCoo = pd.read_csv(os.path.join(dstDir, tableNames[ii]), sep=';')
# dfName = dfNames[ii]
# df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
# df.particle = df.particle.astype(int)
    

df = df_particle_MSD_CylCoo

lims = np.linspace(0, N_pix-1, (M_boxes+1))
fig, axes = plt.subplots(1, 4, figsize = (12, 4), layout='compressed')
axes_f = axes.flatten()

ax = axes_f[0]
vmin, vmax = np.percentile(im, 0.5), np.percentile(im, 99.5)
ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
# ax.hlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
# ax.vlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
ax.set_xlim([0, 511])
ax.set_ylim([0, 511])
ax.set_title('Original image')

ax = axes_f[1]
parm1 = 'D_r_full' # 'D_r_full', 'k_r_full'
v_high1 = np.percentile(df[parm1], 98)
df_f = df[df[parm1] < v_high1]

g = ax.scatter(df_f['Xc'], df_f['Yc'], 
               c=df_f[parm1], cmap='viridis',
               s = 5, alpha = 1, edgecolor='None',
               norm=mpl.colors.Normalize(), # LogNorm
               )
cbar = fig.colorbar(g)
ax.set_xlim([0, 511])
ax.set_ylim([0, 511])
ax.set_title(parm1)


ax = axes_f[2]
parm2 = 'D_or_full' # 'D_or_full', 'k_or_full'
v_high2 = np.percentile(df[parm2], 98)
df_f = df[df[parm2] < v_high2]

g = ax.scatter(df_f['Xc'], df_f['Yc'], 
               c=df_f[parm2], cmap='PuRd',
               s = 5, alpha = 1, edgecolor='None',
               norm=mpl.colors.Normalize(),
               )
cbar = fig.colorbar(g)
ax.set_xlim([0, 511])
ax.set_ylim([0, 511])
ax.set_title(parm2)


ax = axes_f[3]
df_f = df[(df[parm1] < v_high1) & (df[parm2] < v_high2)]
df_f['delta'] = df_f[parm1] - df_f[parm2] 
g = ax.scatter(df_f['Xc'], df_f['Yc'], 
               c=df_f['delta'], cmap='BuPu',
               s = 5, alpha = 1, edgecolor='None',
               norm=mpl.colors.Normalize(),
               )
cbar = fig.colorbar(g)
ax.set_xlim([0, 511])
ax.set_ylim([0, 511])
ax.set_title(f'{parm1} - {parm2}')

# Remove the legend and add a colorbar

plt.show()



fig, axes = plt.subplots(3, 3, figsize = (9, 7), layout='compressed', sharex='col')
axes_f = axes.flatten()

ax = axes_f[0]
vmin, vmax = np.percentile(im, 0.5), np.percentile(im, 99.5)
ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
# ax.hlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
# ax.vlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
ax.set_xlim([0, 511])
ax.set_ylim([0, 511])
ax.set_title('Original image')

ax = axes_f[1]
parm1 = 'D_r_full' # 'D_r_full', 'k_r_full'
v_high1 = np.percentile(df[parm1], 98)
df_f = df[df[parm1] < v_high1]
ax.hist(df_f[parm1].values, bins=60, alpha=0.4, label=parm1)

parm2 = 'D_or_full' # 'D_or_full', 'k_or_full'
v_high2 = np.percentile(df[parm2], 98)
df_f = df[df[parm2] < v_high2]
ax.hist(df_f[parm2].values, bins=60, alpha=0.4, label=parm2)
ax.legend()
ax.set_title(f'{parm1} & {parm2}')

ax = axes_f[2]
df_f = df[(df[parm1] < v_high1) & (df[parm2] < v_high2)]
df_f['delta'] = df_f[parm1] - df_f[parm2] 
ax.hist(df_f['delta'].values, bins=60)
ax.axvline(0, color='gray', ls='-', lw=0.75, alpha=0.8)
ax.set_title(f'{parm1} - {parm2}')


ax = axes_f[3+1]
parm1 = 'D_r_lowDt' # 'D_r_full', 'k_r_full'
v_high1 = np.percentile(df[parm1], 98)
df_f = df[df[parm1] < v_high1]
ax.hist(df_f[parm1].values, bins=60, alpha=0.4, label=parm1)

parm2 = 'D_or_lowDt' # 'D_or_full', 'k_or_full'
v_high2 = np.percentile(df[parm2], 98)
df_f = df[df[parm2] < v_high2]
ax.hist(df_f[parm2].values, bins=60, alpha=0.4, label=parm2)
ax.legend()
ax.set_title(f'{parm1} & {parm2}')

ax = axes_f[3+2]
df_f = df[(df[parm1] < v_high1) & (df[parm2] < v_high2)]
df_f['delta'] = df_f[parm1] - df_f[parm2] 
ax.hist(df_f['delta'].values, bins=60)
ax.axvline(0, color='gray', ls='-', lw=0.75, alpha=0.8)
ax.set_title(f'{parm1} - {parm2}')


ax = axes_f[6+1]
parm1 = 'D_r_highDt' # 'D_r_full', 'k_r_full'
v_high1 = np.percentile(df[parm1], 98)
df_f = df[df[parm1] < v_high1]
ax.hist(df_f[parm1].values, bins=60, alpha=0.4, label=parm1)

parm2 = 'D_or_highDt' # 'D_or_full', 'k_or_full'
v_high2 = np.percentile(df[parm2], 98)
df_f = df[df[parm2] < v_high2]
ax.hist(df_f[parm2].values, bins=60, alpha=0.4, label=parm2)
ax.legend()
ax.set_title(f'{parm1} & {parm2}')

ax = axes_f[6+2]
df_f = df[(df[parm1] < v_high1) & (df[parm2] < v_high2)]
df_f['delta'] = df_f[parm1] - df_f[parm2] 
ax.hist(df_f['delta'].values, bins=60)
ax.axvline(0, color='gray', ls='-', lw=0.75, alpha=0.8)
ax.set_title(f'{parm1} - {parm2}')


plt.show()
    



fig, axes = plt.subplots(3, 3, figsize = (9, 7), layout='compressed', sharex='col')
axes_f = axes.flatten()

ax = axes_f[0]
vmin, vmax = np.percentile(im, 0.5), np.percentile(im, 99.5)
ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
# ax.hlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
# ax.vlines(lims, 0, N_pix, linestyle=':', color='w', lw=0.75)
ax.set_xlim([0, 511])
ax.set_ylim([0, 511])
ax.set_title('Original image')

ax = axes_f[1]
parm1 = 'k_r_full' # 'D_r_full', 'k_r_full'
v_high1 = np.percentile(df[parm1], 98)
df_f = df[df[parm1] < v_high1]
ax.hist(df_f[parm1].values, bins=60, alpha=0.4, label=parm1)

parm2 = 'k_or_full' # 'D_or_full', 'k_or_full'
v_high2 = np.percentile(df[parm2], 98)
df_f = df[df[parm2] < v_high2]
ax.hist(df_f[parm2].values, bins=60, alpha=0.4, label=parm2)
ax.legend()
ax.set_title(f'{parm1} & {parm2}')

ax = axes_f[2]
df_f = df[(df[parm1] < v_high1) & (df[parm2] < v_high2)]
df_f['delta'] = df_f[parm1] - df_f[parm2] 
ax.hist(df_f['delta'].values, bins=60)
ax.axvline(0, color='gray', ls='-', lw=0.75, alpha=0.8)
ax.set_title(f'{parm1} - {parm2}')


ax = axes_f[3+1]
parm1 = 'k_r_lowDt' # 'D_r_full', 'k_r_full'
v_high1 = np.percentile(df[parm1], 98)
df_f = df[df[parm1] < v_high1]
ax.hist(df_f[parm1].values, bins=60, alpha=0.4, label=parm1)

parm2 = 'k_or_lowDt' # 'D_or_full', 'k_or_full'
v_high2 = np.percentile(df[parm2], 98)
df_f = df[df[parm2] < v_high2]
ax.hist(df_f[parm2].values, bins=60, alpha=0.4, label=parm2)
ax.legend()
ax.set_title(f'{parm1} & {parm2}')

ax = axes_f[3+2]
df_f = df[(df[parm1] < v_high1) & (df[parm2] < v_high2)]
df_f['delta'] = df_f[parm1] - df_f[parm2] 
ax.hist(df_f['delta'].values, bins=60)
ax.axvline(0, color='gray', ls='-', lw=0.75, alpha=0.8)
ax.set_title(f'{parm1} - {parm2}')


ax = axes_f[6+1]
parm1 = 'k_r_highDt' # 'D_r_full', 'k_r_full'
v_high1 = np.percentile(df[parm1], 98)
df_f = df[df[parm1] < v_high1]
ax.hist(df_f[parm1].values, bins=60, alpha=0.4, label=parm1)

parm2 = 'k_or_highDt' # 'D_or_full', 'k_or_full'
v_high2 = np.percentile(df[parm2], 98)
df_f = df[df[parm2] < v_high2]
ax.hist(df_f[parm2].values, bins=60, alpha=0.4, label=parm2)
ax.legend()
ax.set_title(f'{parm1} & {parm2}')

ax = axes_f[6+2]
df_f = df[(df[parm1] < v_high1) & (df[parm2] < v_high2)]
df_f['delta'] = df_f[parm1] - df_f[parm2] 
ax.hist(df_f['delta'].values, bins=60)
ax.axvline(0, color='gray', ls='-', lw=0.75, alpha=0.8)
ax.set_title(f'{parm1} - {parm2}')


plt.show()
    






# %%%% Dev MRSD (Pairwise MSD)

pm.setGraphicOptions('print')

Nframe_per_chunk = 100
Nc = nbimages//Nframe_per_chunk
Fi = np.arange(0, 2000, step = Nframe_per_chunk)
Ff = Fi + Nframe_per_chunk
Dict_TRanges = {}

for ii in [2]: # len(dfNames)
    dfName = dfNames[ii]
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    df.particle = df.particle.astype(int)
    df.frame = df.frame.astype(int)
    
    PIDs = df.particle.unique()
    dict_fi_ff = {}
    
    for pid in PIDs:
        pfi = np.min(df[df['particle'] == pid]['frame'].values) - 1
        pff = np.max(df[df['particle'] == pid]['frame'].values) - 1
        dict_fi_ff[pid] = [pfi, pff]
    
    
for ii in [2]: # len(dfNames)
    dfName = dfNames[ii]
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    df.particle = df.particle.astype(int)
    df.frame = df.frame.astype(int)
    
    PIDs = df.particle.unique()
    for jj in range(Nc):
        fi, ff = Fi[jj], Ff[jj]
        TRange = []
        # print('bounds', fi, ff)
        
        for pid in PIDs[:]:
            pfi, pff = dict_fi_ff[pid]
            if (pfi <= fi) and (ff-1 <= pff):
                # print(pfi, pff)
                TRange.append(pid)
                
        Dict_TRanges[fi] = TRange
        
ncols = 5
nrows = ((Nc - 1)//ncols) + 1

fig, axes = plt.subplots(nrows, ncols, figsize=(2.0*ncols, 2.0*nrows))
axes_f = axes.flatten()

for k in range(Nc):
    print(k)
    ax = axes_f[k]
    fi, ff = Fi[k], Ff[k]
    TRange = Dict_TRanges[fi]
    df_TRange = df[df['particle'].apply(lambda x : x in TRange)]
    sns.lineplot(ax=ax,
                 data=df_TRange, x=df_TRange.x, y=df_TRange.y,
                 hue=df_TRange.particle, palette=pm.cL_Set21,
                 ls='-', lw=1, 
                 legend=False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # for p in df_chunk.particle.unique():
    #     df_chunk_p = df_chunk[df_chunk['particle'] == p]
    #     ax.plot(df_chunk_p.x, df_chunk_p.y, ls='-', lw=1)



for k in range(Nc):
    print(k)
    ax = axes_f[k]
    fi, ff = Fi[k], Ff[k]
    TRange = Dict_TRanges[fi]
    df_TRange = df[df['particle'].apply(lambda x : x in TRange)]
    

# %%%% MSRD functions



def get_pairs_for_TRanges(df, SCALE, FPS, Nframes,
                          len_TRanges = 200, delta_TRanges = -1,
                          dist_th_um = 5):
    df.frame = df.frame.astype(int)
    df.particle = df.particle.astype(int)
    dist_th = dist_th_um * SCALE
    
    if delta_TRanges < 0:
        delta_TRanges = len_TRanges
    FI = np.arange(0, Nframes, step=delta_TRanges)
    FF = FI + len_TRanges
    valid = (FF <= Nframes)
    if valid[-1]:
        pass
    else:
        i_stop = ufun.findFirst(True, (FF>Nframes))
        FI = FI[:i_stop]
        FF = FF[:i_stop]
    
    dict_TRanges2particles = {f'{fi}_{ff}':{'pid':[],'xm':[],'ym':[]} \
                              for fi, ff in zip(FI, FF)}
    dict_TRanges2pairs = {f'{fi}_{ff}':[] for fi, ff in zip(FI, FF)}
    
    PIDs = df.particle.unique()
    for pid in PIDs:
        pfi = np.min(df[df['particle'] == pid]['frame'].values) - 1
        pff = np.max(df[df['particle'] == pid]['frame'].values) - 1
        
        for fi, ff in zip(FI, FF):
            if (pfi <= fi) and (ff-1 <= pff):
                xm = np.median(df[df['particle'] == pid]['x'].values)
                ym = np.median(df[df['particle'] == pid]['y'].values)
                dict_TRanges2particles[f'{fi}_{ff}']['pid'].append(pid)
                dict_TRanges2particles[f'{fi}_{ff}']['xm'].append(xm)
                dict_TRanges2particles[f'{fi}_{ff}']['ym'].append(ym)
    
    for TRange in dict_TRanges2particles.keys():
        df_parts = pd.DataFrame(dict_TRanges2particles[TRange])
        listPairs = []
        while len(df_parts)>1:
            p1 = df_parts['pid'].values[0]
            XY1 = np.array([df_parts['xm'].values[0],
                            df_parts['ym'].values[0]])
            XYothers = np.array([df_parts['xm'].values[1:],
                                 df_parts['ym'].values[1:]]).T
            dists = np.power((np.sum((XYothers - XY1)**2, axis=1)), 0.5)
            min_d = np.min(dists)
            if min_d > dist_th:
                idx_to_drop = df_parts[(df_parts["pid"] == p1)].index
                df_parts.drop(axis=0, index=idx_to_drop, inplace=True)
            else:
                idx_min = np.argmin(dists) + 1
                p2 = df_parts['pid'].values[idx_min]
                listPairs.append((p1, p2))
                idx_to_drop = df_parts[(df_parts["pid"] == p1) | (df_parts["pid"] == p2)].index
                df_parts.drop(axis=0, index=idx_to_drop, inplace=True)
                # except:
                #     print(df_parts)
                
        dict_TRanges2pairs[TRange] = np.array(listPairs)
            
    return(dict_TRanges2pairs)







# Test run
dfName = dfNames[2]
df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')

UmPerPix = cd.UmPerPix_60X_W1
SCALE = 1/UmPerPix
Nframes = 2000
FPS = 10

dict_TRanges2pairs = get_pairs_for_TRanges(df, SCALE, FPS, Nframes,
                      len_TRanges = 100, delta_TRanges = -1,
                      dist_th_um = 5)




# %%% 3. Tracking and structure analysis

# %%%% Import tracks & analyse shape of explored zone

idx_films = [2, 3]

pm.setGraphicOptions(mode='screen')
fig, axes = plt.subplots(2, len(idx_films), figsize=(len(idx_films)*3, 6),
                         layout='compressed')
colors = pm.cL_Set21

dict_res = {'label':[],
            'AngleDiffs':[]}

GEOM_DATA = []

for k, ii in enumerate(idx_films):
    dfName = dfNames[ii]
    title = '_'.join(dfName.split('_')[1:3])
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    Lp = df['particle'].unique().astype(int)
    dict_geom = {'particle':[],
                 'np':[],
                 'xc':[],
                 'yc':[],
                 'theta':[],
                 'L':[],
                 'l':[],
                 'AR':[],
                 'phi':[],}
    
    ax = axes[0, k]
    ax.set_xlim([0, 511])
    ax.set_ylim([0, 511])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    
    for j in Lp[:]:
        c = colors[j%len(colors)]
        df_j = df[df['particle']==j]
        # x, y = 
        points = np.array([[x, y] for (x, y) in zip(df_j.x, df_j.y)])
        
        MPoints = MultiPoint(points)
        Hull = MPoints.convex_hull
        xy_ch = np.array([xy for xy in Hull.exterior.coords[1:]])
        xc, yc = Hull.centroid.coords[0]
        
        out = ufun.fit_ellipse_fixedCenter(xy_ch[:, 0], xy_ch[:, 1], 
                                           xc, yc, mode='cartesian')
        _, _, a, b, phi = out
        
        if j%10 == 0:
            aa = np.linspace(0, 2*np.pi, 360)
            xp, yp = ufun.get_ellipse_xy(xc, yc, a, b, phi, aa=aa)
            # ax.plot(xy_ch[:, 0], xy_ch[:, 1],
            #         c=c, marker='.', ls='', ) #ls='-', lw=1)
            ax.plot(xp, yp, c=c, ls='-', lw=0.5)
        
        
        
        theta = np.atan2(yc-255, xc-255)
        theta_deg = theta * 180/np.pi
        
        dotprod = np.cos(theta)*np.cos(phi) + np.sin(theta)*np.sin(phi)
        if dotprod < 0:
            phi = phi - np.sign(phi) * np.pi
            
        phi_deg = phi * 180/np.pi
        # print(phi_deg, theta_deg)
        
        L, l = max(a, b), min(a, b)
        AR = L/l
        
        dict_geom['particle'].append(j)
        dict_geom['np'].append(xc)
        dict_geom['xc'].append(xc)
        dict_geom['yc'].append(yc)
        dict_geom['theta'].append(theta)
        dict_geom['L'].append(L)
        dict_geom['l'].append(l)
        dict_geom['AR'].append(AR)
        dict_geom['phi'].append(phi)
        
        # # create a minimum rotated rectangle containing all the points
        # points = np.array(hull_xy).T 
        # multipoints = MultiPoint(points)
        # polygon = multipoints.minimum_rotated_rectangle
        # Xr, Yr = polygon.exterior.coords.xy
        # ax.plot(Xr, Yr, 'k-', lw=0.5)
        
        # d1 = ((Xr[0]-Xr[1])**2 + (Yr[0]-Yr[1])**2)**0.5
        # d2 = ((Xr[2]-Xr[1])**2 + (Yr[2]-Yr[1])**2)**0.5
        # # print(d1, d2)
        # L, l = max(d1, d2), min(d1, d2)
        # AR = L/l
        
        # V1 = (Xr[1]-Xr[0], Yr[1]-Yr[0])
        # theta = np.atan2(V1[1], V1[0])
        # print(theta*180/np.pi
    
    plt.show()
    
    
    df_geom = pd.DataFrame(dict_geom)
    
    GEOM_DATA.append(df_geom)
    
    df_geom['theta_bin'] = (df_geom['theta'].values * 18/np.pi).astype(int) * 10 + 5
    
    C = np.cos(df_geom['theta'].values) * np.cos(df_geom['phi'].values) + np.sin(df_geom['theta'].values) * np.sin(df_geom['phi'].values)
    DA = np.acos(C)
    
    ax = axes[1, k]
    ax.hist(DA, bins=40)
    ax.set_ylabel('N trajectories')
    ax.set_xlabel(r'$|\theta - \phi|$ (rad)')
    ax.set_xticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
    ax.set_xticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
    ax.grid()
    
    
    dict_res['label'].append(title)
    dict_res['AngleDiffs'].append(DA)

plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(6, 6),
#                          layout='compressed')
# for i, L in enumerate(dict_res['label']):
#     ax.hist(dict_res['AngleDiffs'][i], bins=30, 
#             density=True, alpha=1, label=L, histtype='step')
# ax.legend()
# ax.set_ylabel('Frequency')
# ax.set_xlabel(r'$|\theta - \phi|$ (rad)')
# plt.show()


# %%%% Plot the geometry

df_geom = pd.DataFrame(dict_geom)
df_geom['theta_bin'] = (df_geom['theta'].values * 18/np.pi).astype(int) * 10 + 5

C = np.cos(df_geom['theta'].values) * np.cos(df_geom['phi'].values) + np.sin(df_geom['theta'].values) * np.sin(df_geom['phi'].values)
DA = np.acos(C)

fig, ax = plt.subplots(1, 1, layout='compressed')
ax.hist(DA, bins=40)
ax.set_ylabel('N trajectories')
ax.set_xlabel(r'$|\theta - \phi|$ (rad)')
ax.set_xticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
ax.set_xticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
ax.grid()
plt.show()

# %%%% Plot the inferred center ?

df_geom = GEOM_DATA[0]

im = ufun.load_stack_region(tifPaths[2], time_indices=[1000])[0]

Filters = [(df_geom['AR'] > 4)]

df_f = pm.filterDf(df_geom, Filters).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes[0]
ax.set_aspect('equal', adjustable='box')
ax.set_xlim([0, 511])
ax.set_ylim([0, 511])
ax.imshow(im, cmap='gray')

ax = axes[1]
ax.set_aspect('equal', adjustable='box')
ax.set_xlim([0, 511])
ax.set_ylim([0, 511])
ax.scatter(df_f['xc'], df_f['yc'], s=6, marker='.', color='c', alpha = 0.05)

for i in range(len(df_f)):
    xc, yc, phi = df_f.loc[i, 'xc'], df_f.loc[i, 'yc'], df_f.loc[i, 'phi']
    S = np.tan(phi)
    ax.axline((xc, yc), slope=S, linestyle='-', color='c', alpha = 0.04, lw=10)


plt.show()



# %%%% Import tracks & run pcf2d

pm.setGraphicOptions(mode='screen')

for ii in [2]:   
    dfName = dfNames[ii]
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    title = '_'.join(dfName.split('_')[:5])
    
    for jj in [1750]:
        df_j = df[df['frame'] == jj+1]
        im = ufun.load_stack_region(tifPaths[ii], time_indices=[jj])[0]
        title_j = title + f' - Frame no {jj+1:.0f}'
        fName_j = title + f'_Fn{jj+1:.0f}_PCF.png'
        XX, YY = df_j.x*UmPerPix, df_j.y*UmPerPix
        XXYY = np.array([XX, YY]).T
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), layout='compressed')
        axes_f = axes.flatten()
        
        #### EQUALIZE
        p1, p99 = np.percentile(im, (1, 99))
        im_pt = skm.exposure.rescale_intensity(im, in_range=(p1, p99))
        
        ax = axes_f[0]
        ax.set_aspect('equal', adjustable='box')
        ax.imshow(im_pt, cmap='gray')
        ax.plot(XX/UmPerPix, YY/UmPerPix, ls='', marker='.',  markersize=2, color='red')
        ax.set_xlabel(r'$x\ (px)$')
        ax.set_ylabel(r'$y\ (px)$')
        ax.set_xlim([0, 511])
        ax.set_ylim([0, 511])
        
        ax = axes_f[1]
        ax.set_aspect('equal', adjustable='box')
        ax.plot(XX, YY, ls='', marker='.')
        ax.set_xlabel(r'$x\ (\mu m)$')
        ax.set_ylabel(r'$y\ (\mu m)$')
        ax.set_xlim([0, 511*UmPerPix])
        ax.set_ylim([0, 511*UmPerPix])
        
        array_positions = XXYY
        bins_distances = np.arange(0, 15, 0.2)
        
        out = tbsa.pcf2d(array_positions, bins_distances, 
                  coord_border=None, coord_holes=None, fast_method=False,
                  show_timing=False, plot=False, full_output=False)
        
        (g_of_r_normalized, radii) = out
        N_of_r_normalized = 2*np.pi * np.array([np.sum(radii[:k]*g_of_r_normalized[:k]) for k in range(len(g_of_r_normalized))])
        
        ax = axes_f[2]
        ax.plot(radii, g_of_r_normalized, color=pm.cL_Set2[0])
        ax.set_xlabel(r'$r\ (\mu m)$')
        ax.set_ylabel(r'$G(r)$')
        # ax.grid()
        ax.axhline(1, linestyle=':', color='gray')

        
        ax = axes_f[3]
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
        
        fig.suptitle(title_j + ' - points spatial stats')
        
        # fig.suptitle(title_j)
        # figpath = os.path.join(dstDir, fName_j)
        # fig.savefig(figpath, dpi=500, )
        

# %%%% Import image, treat and & run pcf2d

pm.setGraphicOptions(mode='screen')

for ii in [2]:   
    dfName = dfNames[ii]
    df = pd.read_csv(os.path.join(dstDir, dfName), sep='\t')
    title = '_'.join(dfName.split('_')[:5])
    
    for jj in [1750]:
        # df_j = df[df['frame'] == jj+1]
        im = ufun.load_stack_region(tifPaths[ii], time_indices=[jj])[0]
        title_j = title + f' - F {jj+1:.0f}'
        fName_j = title + f'_Fn{jj+1:.0f}_PCF.png'
        # XX, YY = df_j.x*UmPerPix, df_j.y*UmPerPix
        # XXYY = np.array([XX, YY]).T
        
        #### EQUALIZE
        p1, p99 = np.percentile(im, (1, 99))
        im_pt = skm.exposure.rescale_intensity(im, in_range=(p1, p99))   
            
        #### FILTER
        k = 3
        im_pt = cv2.medianBlur(im_pt, k)
        
        #### TOP_HAT
        filterSize = (12, 12)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
        im_pt = cv2.morphologyEx(im_pt, cv2.MORPH_TOPHAT, kernel)
        p1, p99 = np.percentile(im_pt, (1, 99))
        im_pt = skm.exposure.rescale_intensity(im_pt, in_range=(p1, p99))

        #### BINARIZE
        th = skm.filters.threshold_li(im_pt)
        im_bin = (im_pt >= th)
        k = 3
        im_bin = ndi.binary_opening(im_bin)
        
        # PLOT PRETREATMENT
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='compressed')
        axes_f = axes.flatten()
        ax = axes_f[0]
        ax.imshow(im, cmap='gray')
        ax.set_xlabel(r'$x\ (px)$')
        ax.set_ylabel(r'$y\ (px)$')
        
        ax = axes_f[1]
        ax.imshow(im_pt, cmap='gray')
        ax.set_xlabel(r'$x\ (px)$')
        ax.set_ylabel(r'$y\ (px)$')
        
        ax = axes_f[2]
        ax.imshow(im_bin, cmap='gray')
        ax.set_xlabel(r'$x\ (px)$')
        ax.set_ylabel(r'$y\ (px)$')
        
        fig.suptitle(title_j + ' - pretreatment')
        plt.show()
        
        
        # COMPUTE SPATIAL STATS
        M = 150
        r_values = np.linspace(1, 300, 300)
        selected_points, all_points = tbsa.sample_white_pixels(im_bin, M=M, seed=40)
        
        K_local, K_global = tbsa.ripley_K_sampled_references(selected_points,
                                                             all_points,
                                                             im_bin.shape,
                                                             r_values)
        L_global = (K_global/np.pi)**0.5
        
        spline_Kg = make_splrep(r_values, K_global, s=6)
        Kg_splinified = spline_Kg(r_values)
        spline_Kg_der = spline_Kg.derivative(nu=1)
        G = spline_Kg_der(r_values) * 1/(2*np.pi*r_values)
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), layout='compressed')
        axes_f = axes.flatten()
        ax = axes_f[0]
        ax.imshow(im_bin, cmap='gray')
        ax.plot(
            selected_points[:,0], selected_points[:,1],
            ls='', marker='.',  markersize=3, color='red',
        )
        ax.set_xlabel(r'$x\ (px)$')
        ax.set_ylabel(r'$y\ (px)$')
        
        ax = axes_f[1]
        for i in range(M):
            ax.plot(r_values, K_local[i, :], alpha=0.05)
        ax.grid()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Radius r")
        ax.set_ylabel("Local Ripley K(r)")
        
        ax = axes_f[2]
        ax.plot(r_values, K_global, ls='', marker='.', label='K(r)')
        ax.plot(r_values, Kg_splinified, linewidth=1.5, ls='-', label='spline rep', color='k')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Radius r")
        ax.set_ylabel("Ripley K(r)")
        idx_start_fit = 30
        xfit = np.log(r_values[idx_start_fit:])
        yfit = np.log(K_global[idx_start_fit:])
        parms, res = ufun.fitLineHuber(xfit, yfit)
        b, a = parms
        k, A = a, np.exp(b)
        xx = r_values[idx_start_fit:]
        ax.plot(xx, A*xx**k, lw=1.5, ls='--', color='red',
                label=r'Fit $y=Ax^k$' + f'\nk={k:.2f}')        
        ax.legend()
        ax.grid()
        

        ax = axes_f[3]
        ax.axhline(1, linestyle='--', color='gray', alpha=1)
        ax.plot(r_values, G, linewidth=2, ls='-', label='g(r)')
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_xlabel("Radius r")
        ax.set_ylabel(r"$g(r)$")
        
        # ax = axes_f[4]
        # ax.plot(r_values, L_global - r_values, linewidth=2, label='K(r)')
        # # ax.set_xscale('log')
        # # ax.set_yscale('log')
        # ax.set_xlabel("Radius r")
        # ax.set_ylabel(r"$\sqrt{K(r)/\pi} - r$")

        fig.suptitle(title_j + ' - spatial stats')
        plt.show()

        
        # array_positions = XXYY
        # bins_distances = np.arange(0, 15, 0.2)
        
        # out = tbsa.pcf2d(array_positions, bins_distances, 
        #           coord_border=None, coord_holes=None, fast_method=False,
        #           show_timing=False, plot=False, full_output=False)
        
        # (g_of_r_normalized, radii) = out
        # N_of_r_normalized = 2*np.pi * np.array([np.sum(radii[:k]*g_of_r_normalized[:k]) for k in range(len(g_of_r_normalized))])
        
        # ax = axes_f[2]
        # ax.plot(radii, g_of_r_normalized, color=pm.cL_Set2[0])
        # ax.set_xlabel(r'$r\ (\mu m)$')
        # ax.set_ylabel(r'$G(r)$')
        # # ax.grid()
        # ax.axhline(1, linestyle=':', color='gray')

        
        # ax = axes_f[3]
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        # idx_start_fit = 30
        

        # ax.plot(radii[:], N_of_r_normalized[:],
        #         'k.', label='Data')
        
        # xfit = np.log(radii[idx_start_fit:])
        # yfit = np.log(N_of_r_normalized[idx_start_fit:])
        # parms, res = ufun.fitLineHuber(xfit, yfit)
        # b, a = parms
        # k, A = a, np.exp(b)
        # xx = radii[idx_start_fit:]
        # ax.plot(xx, A*xx**k, lw=1.5,
        #         label=r'Fit $y=Ax^k$' + f'\nk={k:.2f}')
        # ax.legend()
        # ax.grid()
        # ax.set_xlabel(r'$r\ (\mu m)$')
        # ax.set_ylabel(r'$N(r)$')
        
        # fig.suptitle(title_j)
        # figpath = os.path.join(dstDir, fName_j)
        # fig.savefig(figpath, dpi=500, )
        
        
# %%%% Import FAKE image, treat and & run pcf2d

pm.setGraphicOptions(mode='screen')

srcDir = os.path.join(up.Path_IntraCellTracking, '26-09-11_FakeImages')
fileNames = ['dots.tiff', 'branches.tiff', 'clustered_dots.tiff']
filePaths = [os.path.join(srcDir, fN) for fN in fileNames]

for ii in range(len(fileNames)):   
    fN = fileNames[ii]
    fP = filePaths[ii]
    
    # df_j = df[df['frame'] == jj+1]
    im = skm.io.imread(fP, as_gray=True)
    im = skm.util.img_as_ubyte(im)
    title_j = f'{fN}'
    fName_j = f'{fN}_KRF.png'
    # XX, YY = df_j.x*UmPerPix, df_j.y*UmPerPix
    # XXYY = np.array([XX, YY]).T
    
    #### EQUALIZE
    # p1, p99 = np.percentile(im, (1, 99))
    # im_pt = skm.exposure.rescale_intensity(im, in_range=(p1, p99))   
        
    #### FILTER
    # k = 3
    # im_pt = cv2.medianBlur(im_pt, k)
    
    #### TOP_HAT
    # filterSize = (12, 12)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    # im_pt = cv2.morphologyEx(im_pt, cv2.MORPH_TOPHAT, kernel)
    # p1, p99 = np.percentile(im_pt, (1, 99))
    # im_pt = skm.exposure.rescale_intensity(im_pt, in_range=(p1, p99))

    #### BINARIZE
    # th = skm.filters.threshold_li(im_pt)
    # im_bin = (im_pt >= th)
    im_bin = ndi.binary_opening(im)
    
    
    M = 150

    selected_points, all_points = tbsa.sample_white_pixels(im_bin, M=M, seed=40)
    
    r_values = np.linspace(1, 300, 300)
    
    K_local, K_global = tbsa.ripley_K_sampled_references(selected_points,
                                                         all_points,
                                                         im_bin.shape,
                                                         r_values)
    L_global = (K_global/np.pi)**0.5
    
    spline_Kg = make_splrep(r_values, K_global, s=6)
    Kg_splinified = spline_Kg(r_values)
    spline_Kg_der = spline_Kg.derivative(nu=1)
    G = spline_Kg_der(r_values) * 1/(2*np.pi*r_values)
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), layout='compressed')
    axes_f = axes.flatten()
    ax = axes_f[0]
    ax.imshow(im_bin, cmap='gray')
    ax.plot(
        selected_points[:,0], selected_points[:,1],
        ls='', marker='.',  markersize=3, color='red',
    )
    ax.set_xlabel(r'$x\ (\mu m)$')
    ax.set_ylabel(r'$y\ (\mu m)$')
    
    ax = axes_f[1]
    for i in range(M):
        ax.plot(r_values, K_local[i, :], alpha=0.05)
    ax.grid()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Radius r")
    ax.set_ylabel("Local Ripley K(r)")
    
    ax = axes_f[2]
    ax.plot(r_values, K_global, linewidth=2, label='K(r)')
    ax.plot(r_values, Kg_splinified, linewidth=1, ls='--', label='spline rep')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Radius r")
    ax.set_ylabel("Ripley K(r)")
    
    idx_start_fit = 30
    xfit = np.log(r_values[idx_start_fit:])
    yfit = np.log(K_global[idx_start_fit:])
    parms, res = ufun.fitLineHuber(xfit, yfit)
    b, a = parms
    k, A = a, np.exp(b)
    xx = r_values[idx_start_fit:]
    ax.plot(xx, A*xx**k, lw=1.5,
            label=r'Fit $y=Ax^k$' + f'\nk={k:.2f}')
    
    ax.legend()
    ax.grid()
    
    
    
    # ax = axes_f[3]
    # ax.plot(r_values, L_global - r_values, linewidth=2, label='K(r)')
    # # ax.set_xscale('log')
    # # ax.set_yscale('log')
    # ax.set_xlabel("Radius r")
    # ax.set_ylabel(r"$\sqrt{K(r)/\pi} - r$")
    
    ax = axes_f[3]
    ax.axhline(1, linestyle=':', color='gray', alpha=0.7)
    ax.plot(r_values, G, linewidth=2, ls='-', label='g(r)')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel("Radius r")
    ax.set_ylabel(r"$g(r)$")
    
    
    
    
    

    plt.show()
        
        
    
# %%% 10. Compare MSD for DDM and SPT

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



