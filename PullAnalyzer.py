# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 21:49:47 2025

@author: Team Minc, adapted in Python by Joseph Vermeil
"""


# %% 1. Imports

import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from copy import deepcopy
from scipy.io import savemat
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import make_splrep

# Local imports
import PlotMaker as pm
import UtilityFunctions as ufun
pm.setGraphicOptions(mode = 'screen', palette = 'Set2', colorList = pm.colorList10)


# %% 2. Subfunctions

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


def trackSelection(tracks, mode="longest"):
    """
    Select one track.
    mode="longest": pick the track with most frames
    """
    if mode == "longest":
        lengths = [len(tr) for tr in tracks]
        idx = int(np.argmax(lengths))
        return(tracks[idx])
    else:
        print('Unspecified track selection')
        # Fallback: first track
        return(tracks[0])


def fitLine(X, Y, with_intercept = True):
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
        model = sm.OLS(Y, X)
    else:
        model = sm.OLS(Y, X, hasconst = False)  
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


def fitPower(X, Y):
    """
    returns: results.params, results \n
    Y=A*X^k ; params[0] = b,  params[1] = a
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
    X2, Y2 = np.log(X), np.log(Y)
    X2 = sm.add_constant(X2)
    model = sm.OLS(Y2, X2)
    results = model.fit()
    raw_params = results.params
    pow_params = [np.exp(raw_params[0]), raw_params[1]]
    return(pow_params, raw_params, results)


def fitPowerHuber(X, Y, with_wlm_results = False):
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
    X2, Y2 = np.log(X), np.log(Y)
    X2 = sm.add_constant(X2)
    model = sm.RLM(Y2, X2, M=sm.robust.norms.HuberT())
    results = model.fit()
    raw_params = results.params
    pow_params = [np.exp(raw_params[0]), raw_params[1]]
    return(pow_params, raw_params, results)


def get_R2(Ymeas, Ymodel):
    meanY = np.mean(Ymeas)
    meanYarray = meanY*np.ones(len(Ymeas))
    SST = np.sum((Ymeas-meanYarray)**2)
    SSR = np.sum((Ymodel-Ymeas)**2)
    if pd.isnull(SST) or pd.isnull(SSR) or SST == 0:
        R2 = np.nan
    else:
        R2 = 1 - (SSR/SST)
    return(R2)


def Df2Dict(df):
    N = len(df)
    d = df.to_dict(orient='list')
    if N == 1:
        for k in d.keys():
            v = d[k]
            d[k] = v[0]
    return(d)

# d1 = {'a':[1, 2, 3],
#       'b':[1, 2, 3],
#       'c':[1, 2, 3],
#       }

# df1 = pd.DataFrame(d1)
# d11 = Df2Dict(df1)

# d2 = {'a':[1],
#       'b':[2],
#       'c':[3],
#       }

# df2 = pd.DataFrame(d2)
# d22 = Df2Dict(df2)

def guess_init_parms_jeffrey(t, dx_n):
    n_low = 10
    n_high = len(dx_n)//3
    parms, res = fitLine(t[n_high:], dx_n[n_high:])
    b1, a1 = parms
    parms, res = fitLine(t[:n_low], dx_n[:n_low], with_intercept = False)
    a2 = parms[0]

    k1_i = max(1/b1, 0)
    gamma1_i = max(1/(a2-a1), 0)
    gamma2_i = max(1/a1, 0)
    
    return(k1_i, gamma1_i, gamma2_i)


def guess_init_parms_relax(t, dx_n):
    n_low = 10
    n_high = len(dx_n)//3

    b1 = np.median(dx_n[n_high:])
    
    parms, res = fitLine(t[:n_low], dx_n[:n_low])
    b2, a2 = parms

    a_i = b1
    tau_i = -(1-b1)/a2
    
    return(a_i, tau_i)

#### Physics

def doubleExpo(x, A, k1, B, k2):
    return(A*np.exp(-x/k1) + B*np.exp(-x/k2))

def powerLaw(x, A, k):
    return(A*(x**k))

def powerLawShifted(x, A, k, x0):
    return(A*((x-x0)**k))

def jeffrey(t, k1, g1, g2):
    return(t/g2 + (1/k1)*(1-np.exp(-t*k1/g1)))

def relax(t, a, tau):
    return((1-a)*(np.exp(-t/tau)) + a)


#### Magnets

MagnetDict = {
    'magnet_MA_MyOne':{
        "F_popt_2exp": [
            6.695570103435238,
            104.27091113638627,
            0.12369116542822649,
            25350957.444718868
        ],
        "F_popt_pL": [
            23630.751398753007,
            -1.8872124731536342
        ],
    },

    'magnet_MA_M270':{
        "F_popt_2exp": [
            141.59732873683043,
            72.16444694994189,
            12.460618814764926,
            272.91475019440617
        ],
        "F_popt_pL": [
            1482165.648896691,
            -2.166374314607223
        ],
    },

    'magnet_JX_MyOne':{
        "F_popt_2exp": [
            5.99012502563366,
            100.84369389492127,
            0.11435249326662983,
            2101.719709808641
        ],
        "F_popt_pL": [
            39603.33040969049,
            -2.0162526263553215
        ],
    },
    
    'magnet_JX_M270':{
        "F_popt_2exp": [
            119.21587409989525,
            65.9084742068066,
            16.320212861019403,
            214.41166616406406
        ],
        "F_popt_pL": [
            948854.7339381203,
            -2.1236746283466172
        ],
    },
}

# %% 3. Step-by-step prototype

# %%% 3. Parameters

# %%%% 3.1. P1

date = '25-09-04'
# pull_id = '25-09-04_M1-D7-P4'
# pull_id = '25-09-04_M1-D7-P6'
# pull_id = '25-09-04_M2-D3-P1'
pull_id = '25-09-04_M2-D4-P1'

# directory = os.path.join("C:/Users/Utilisateur/Desktop/Data From Maribel")
# df_manips = pd.read_csv(os.path.join(directory, 'md_manips.csv'))
# md_table = pd.read_csv(os.path.join(directory, 'md_pulls.csv'))
# directory = os.path.join(directory, 'Analysis')

# tracks_name = pull_id + '_Tracks.xml'
# output_folder = pull_id + '_Results'
# if not os.path.exists(os.path.join(directory, output_folder)):
#     os.makedirs(os.path.join(directory, output_folder))


# frame_initPull = md_table.loc[md_table['id'] == pull_id, 'frame_initPull'].values[0]
# frame_endPull = md_table.loc[md_table['id'] == pull_id, 'frame_endPull'].values[0]
# magnet_x = md_table.loc[md_table['id'] == pull_id, 'magnet_x'].values[0]
# magnet_y = md_table.loc[md_table['id'] == pull_id, 'magnet_y'].values[0]

# pixel_size = 0.451  # µm
# # pixel_size = 0.909  # µm
# time_stp = 0.5      # s
# magnet_radius = pixel_size * md_table.loc[md_table['id'] == pull_id, 'magnet_diameter'].values[0] / 2  # µm
# bead_radius = pixel_size * md_table.loc[md_table['id'] == pull_id, 'bead_diameter'].values[0] / 2   # µm


#### For Maribel's film
directory = os.path.join("C:/Users/Utilisateur/Desktop/Data From Maribel")
tracks_name = '10x_100CE_4fps_DIC__1_Tracks.xml'
output_folder = 'Outputs'
if not os.path.exists(os.path.join(directory, output_folder)):
    os.makedirs(os.path.join(directory, output_folder))
pixel_size = 0.909  # µm
time_stp = 0.25      # s
frame_initPull = 24 # 25
frame_endPull = 41
magnet_x = 694.032
magnet_y = 915.625
magnet_radius = pixel_size * 112 / 2
bead_radius = pixel_size * 34 / 2   # µm

# Fit options
start1 = [80, 100, 1000]   # initial [k, gamma1, gamma2]
start2 = [0.9, 25]      # initial [a, tau]

# %%%% 3.2. P2

# Viscosity of glycerol 80% v/v glycerol/water at 21°C [Pa.s]
viscosity_glycerol = 0.0857  
# Magnet function distance (µm) to velocity (µm/s) [expected velocity in glycerol]
mag_d2v = lambda x: 80.23*np.exp(-x/47.49) + 1.03*np.exp(-x/22740.0)
# Speed at 200 mu as um/s
v_interp = mag_d2v(200)
# Aggregate force coefficient (c) in f=cR^3
force_coeff = 0.3663 
# Magnet function distance (µm) to force (pN)
mag_d2f = lambda x: force_coeff*(bead_radius**3)*mag_d2v(x)/v_interp # If beads are agarose
# mag_d2f = lambda x: 6*np.pi*viscosity_glycerol*mag_d2v(x)*bead_radius # Function for MyOnes (calib beads)
                      

# Fit options
start1 = [30, 200, 600]   # initial [k, gamma1, gamma2]
start2 = [0.5, 4]      # initial [a, tau]


# %%% 4. Load data

tracks = importTrackMateTracks(os.path.join(directory, tracks_name))
track = trackSelection(tracks, mode = 'longest')

X, Y, t = track[:,1], track[:,2], track[:,0]

initPullTime = np.where(t == (frame_initPull - 1))[0][0]
finPullTime = np.where(t == (frame_endPull - 1))[0][0]


# %%% 5. Pretreatment

# %%%% --- Rotate track ---
theta = np.arctan2(Y[initPullTime] - Y[finPullTime],
                   X[initPullTime] - X[finPullTime])
rotation_mat = np.array([[np.cos(-theta), -np.sin(-theta)],
                         [np.sin(-theta),  np.cos(-theta)]])
coords = np.vstack((X, Y)).T @ rotation_mat.T
x_rot, y_rot = coords[:,0], coords[:,1]

x_shift = (-x_rot[initPullTime:] + np.max(x_rot[initPullTime:])) * pixel_size

# %%%% --- Pulling phase ---
pull_index = np.arange(initPullTime, finPullTime+1)
pull_length = len(pull_index)
xp, yp = x_rot[pull_index], y_rot[pull_index] # for plots

d = np.stack([(magnet_x - X[pull_index]) * pixel_size,
              (magnet_y - Y[pull_index]) * pixel_size], axis=1)
dist = np.linalg.norm(d, axis=1) - magnet_radius
pull_force = mag_d2f(dist)

tpulling = (t[pull_index] - t[initPullTime]) * time_stp
dx_pulling = x_shift[pull_index - initPullTime]
dx_pulling_n = dx_pulling / pull_force

# %%%% Release phase

release_index = np.arange(finPullTime, len(track))
release_length = len(release_index)

trelease = (t[release_index] - t[finPullTime+1]) * time_stp
dx_release = x_shift[release_index - initPullTime]
dx_release_n = dx_release / x_shift[pull_length]

# %%% 6. Fit model

# %%%% Option 1. --- Viscous model ---

# params, results = fitLine(tpulling[:], dx_pulling_n[:])
# # params, results = fitLineHuber(tpulling, dx_pulling_n)

# gamma = params[1]
# visco = 1/(6*np.pi*bead_radius*gamma)
# R2 = results.rsquared

# print(gamma, visco, R2)

# %%%% --- Figures ---

# os.chdir(os.path.join(directory, output_folder))

# fig1, axes1 = plt.subplots(1, 2, figsize = (10,5))
# ax = axes1[0]
# ax.plot(X, Y, ".-")
# ax.axis("equal")
# ax.grid(axis='both')
# ax.set_title("Original track")

# ax = axes1[1]
# ax.plot(x_rot, y_rot, ".-")
# ax.plot(xp, yp, "k.-", linewidth=2)
# ax.axis("equal")
# ax.grid(axis='both')
# ax.set_title("Rotated track")
# plt.savefig("trajectories.png")

# fig2, axes2 = plt.subplots(1, 1, figsize = (5,5))
# ax = axes2
# ax.plot(tpulling, dx_pulling_n, "s")
# xfit = np.linspace(0, tpulling[-1], 100)
# yfit = params[0] + params[1]*xfit
# ax.plot(xfit, yfit, "r-", label=r'$\eta$ = ' + f'{visco*1000:.2f} mPa.s')
# ax.set_xlabel("t [s]")
# ax.set_ylabel("dx/f [µm/pN]")
# ax.grid(axis='both')
# ax.legend(fontsize = 11)
# plt.savefig("fits.jpg")

# plt.show()

# %%% -----------

# %%% Option 2. --- Jeffreys model ---

# %%%% --- Fit pulling phase ---

def jeffrey_model(params, x):
    k, gamma1, gamma2 = params
    return (1 - np.exp(-k*x/gamma1))/k + x/gamma2

obj1 = lambda params: np.linalg.norm(jeffrey_model(params, tpulling) - dx_pulling_n)
res1 = minimize(obj1, start1, method="Nelder-Mead", tol=1e-10,
                options={"maxfev": 500})
k, gamma1, gamma2 = res1.x
tempo1, tempo2 = gamma1/k, gamma2/k

# %%%% --- Fit release phase ---

def exp_fit(params, x):
    a, tau = params
    return (1-a)*np.exp(-x/tau) + a

obj2 = lambda params: np.linalg.norm(exp_fit(params, trelease) - dx_release_n)
res2 = minimize(obj2, start2, method="Nelder-Mead", tol=1e-7,
                options={"maxfev": 1000})
a, tau = res2.x

# %%%% --- Figures ---

fig1, axes1 = plt.subplots(1, 2, figsize=(8,5))
ax = axes1[0]
ax.plot(X, Y, ".-")
ax.axis("equal")
ax.set_title("Original track")
ax.grid()

ax = axes1[1]
ax.plot(x_rot, y_rot, ".-")
ax.plot(xp, yp, "k.-", linewidth=2)
ax.axis("equal")
ax.set_title("Rotated track")
ax.grid()
plt.tight_layout()
# plt.savefig("trajectories.jpg")

fig2, axes2 = plt.subplots(1, 2, figsize=(8,5))
ax = axes2[0]
label1 = "Fitting Jeffrey's model\n" + r"$\frac{1}{k}(1 - exp(-k.x/\gamma_1)) + x/\gamma_2$" + "\n"
label1+= r"$k$ = " + f"{k:.2f}\n"
label1+= r"$\gamma_1$ = " + f"{gamma1:.2f}\n"
label1+= r"$\gamma_2$ = " + f"{gamma2:.2f}"
ax.plot(tpulling, dx_pulling_n, "s")
ax.plot(np.linspace(0, tpulling[-1], 1000),
         jeffrey_model([k,gamma1,gamma2], np.linspace(0, tpulling[-1], 1000)), "r-",
         label=label1)
ax.legend()
ax.grid()
ax.set_xlabel("t [s]"); plt.ylabel("dx/f [µm/pN]")

ax = axes2[1]
label2 = "Fitting Viscoel Relax\n" + r"$(1-a). exp(-x/\tau ) + a$" + "\n"
label2+= r"$a$ = " + f"{a:.2f}\n"
label2+= r"$\tau$ = " + f"{tau:.2f}"
ax.plot(trelease, dx_release_n, "s")
ax.plot(np.linspace(0, trelease[-1], 1000),
         exp_fit([a,tau], np.linspace(0, trelease[-1], 1000)), ls="-", c='darkorange',
         label=label2)
ax.legend()
ax.grid()
ax.set_xlabel("t [s]"); plt.ylabel("Normalized displacement")
ax.set_ylim([0, 1.5])
plt.tight_layout()
# plt.savefig("fits.jpg")


# %%%% --- Save results ---

os.makedirs(os.path.join(directory, output_folder), exist_ok=True)
os.chdir(os.path.join(directory, output_folder))

# Excel saving
with pd.ExcelWriter(f"Table.xlsx", engine="openpyxl") as writer:
    pulling_force_df = pd.DataFrame(np.column_stack([tpulling, dx_pulling, pull_force]),
                                    columns=["t[s]", "dx.pulling", "Pulling Force [pN]"])
    pulling_force_df.to_excel(writer, sheet_name="PullingPhase1", startrow=1, startcol=1, index=False)

    pulling_curves_df = pd.DataFrame(np.column_stack([tpulling, dx_pulling_n]),
                                      columns=["t[s]", "dx/f[µm/pN]"])
    pulling_curves_df.to_excel(writer, sheet_name="PullingPhase2", startrow=1, startcol=1, index=False)
    pd.DataFrame([[k, gamma1, gamma2, tempo1, tempo2]],
                  columns=["k", "gamma1", "gamma2", "tempo1", "tempo2"]).to_excel(
                      writer, sheet_name="PullingPhase2", startrow=1, startcol=3, index=False)

    release_curves_df = pd.DataFrame(np.column_stack([trelease, dx_release_n]),
                                      columns=["t[s]", "dx/dx(0)"])
    release_curves_df.to_excel(writer, sheet_name="ReleasePhase", startrow=1, startcol=1, index=False)
    pd.DataFrame([[a, tau]], columns=["a", "tau"]).to_excel(
        writer, sheet_name="ReleasePhase", startrow=1, startcol=3, index=False)

# Save .mat
savemat(f"Table.mat", {"k": k, "gamma1": gamma1, "gamma2": gamma2,
                                "a": a, "tau": tau})

# %%%% --- Figures ---

plt.figure()
plt.subplot(1,2,1)
plt.plot(X, Y, ".-")
plt.axis("equal")
plt.title("Original track")

plt.subplot(1,2,2)
plt.plot(x_rot, y_rot, ".-")
plt.plot(xp, yp, "k.-", linewidth=2)
plt.axis("equal")
plt.title("Rotated track")
plt.savefig("trajectories.jpg")

plt.figure()
plt.subplot(1,2,1)
plt.plot(tpulling, dx_pulling_n, "s")
plt.plot(np.linspace(0, tpulling[-1], 1000),
          jeffrey_model([k,gamma1,gamma2], np.linspace(0, tpulling[-1], 1000)), "r-")
plt.xlabel("t [s]"); plt.ylabel("dx/f [µm/pN]")

plt.subplot(1,2,2)
plt.plot(trelease, dx_release_n, "s")
plt.plot(np.linspace(0, trelease[-1], 1000),
          exp_fit([a,tau], np.linspace(0, trelease[-1], 1000)), "r-")
plt.xlabel("t [s]"); plt.ylabel("Normalized displacement")
plt.ylim([0, 1.5])
plt.savefig("fits.jpg")

print(k, gamma1, gamma2, a, tau)

# %% ---------------







# %% 11. Analysis as a function

def pullAnalyzer_multiFiles(mainDir, date, prefix_id,
                            analysisDir, tracksDir, resultsDir, plotsDir,
                            fits = ['newton'], calibFuncType='PowerLaw',
                            resultsFileName = 'results',
                            Redo = False, PLOT = True, SHOW = False):
    
    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)
    if not os.path.exists(plotsDir):
        os.makedirs(plotsDir)
    df_manips = pd.read_csv(os.path.join(analysisDir, 'MainExperimentalConditions.csv'))
    df_pulls = pd.read_csv(os.path.join(analysisDir, date + '_ExperimentalConditions.csv'))
    listTrackFiles = [f for f in os.listdir(tracksDir) if ('Track' in f) and (f.endswith('.xml'))]
    listTrackIds = ['_'.join(string.split('_')[:6]) for string in listTrackFiles]
    listPullIds = ['_'.join(string.split('_')[:5]) for string in listTrackFiles]
    dict_TrackIds2File = {tid : tf for (tid, tf) in zip(listTrackIds, listTrackFiles)}
    dict_TrackIds2PullId = {tid : pid for (tid, pid) in zip(listTrackIds, listPullIds)}
    try:
        df_res = pd.read_csv(os.path.join(resultsDir, resultsFileName + '.csv'))
        already_analyzed = df_res['pulltrack_id'].values
    except:
        df_res = pd.DataFrame({})
        already_analyzed = []
    
    prefix_id_manip = ('_').join(prefix_id.split('_')[:1 + ('_' in prefix_id)])
    listManips = [m for m in df_manips['id'] if m.startswith(prefix_id_manip)]
    
    id_cols = ['track_id']
    co_cols = ['type', 'system', 'solution', 'bead type', 'bead radius', 'treatment', 'magnet']
    # co_cols = ['type', 'solution', 'bead type', 'bead radius', 'treatment', 'magnet']
    
    results_cols = []
    M_results_cols = ['globalError', 
                   'median instant speed', 'mean speed', 'median force', 
                   'R min', 'R max', 'theta']
    results_cols += M_results_cols
    
    if 'newton' in fits:
        N_results_cols = ['N_fit_error', 'N_viscosity', 'N_R2']
        results_cols += N_results_cols

    if 'jeffrey' in fits:
        J_results_cols = ['J_fit_error', 'J_k', 'J_gamma1', 'J_gamma2', 'J_tempo1', 'J_tempo2', 'J_R2']
        R_results_cols = ['R_fit_error', 'R_a', 'R_tau', 'R_R2']
        results_cols += J_results_cols 
        results_cols += R_results_cols

    results_dict = {c:[] for c in id_cols}
    results_dict.update({c:[] for c in results_cols})
    results_dict.update({c:[] for c in co_cols})
    
    for M_id in listManips:
        co_dict = df_manips[df_manips['id'] == M_id].to_dict('list')
        listTracks = [t for t in listTrackIds if \
                      t.startswith(prefix_id) and (Redo or (t not in already_analyzed))]
        for T_id in listTracks:
            trackFileName = dict_TrackIds2File[T_id]
            tracks = importTrackMateTracks(os.path.join(tracksDir, trackFileName))
            track = trackSelection(tracks, mode = 'longest')
            
            P_id = dict_TrackIds2PullId[T_id]
            dict_pull = Df2Dict(df_pulls[df_pulls['id'] == P_id])
            
            magnet_id = co_dict['magnet'][0]
            bead_type = co_dict['bead type'][0]
            pixel_size = df_pulls.loc[df_pulls['id']==P_id, 'film_pixel_size'].values[0]
            bead_radius = df_pulls.loc[df_pulls['id']==P_id, 'b_r'].values[0]
            
            results_dict['track_id'].append(T_id)
            results_dict['type'].append(co_dict['type'][0])
            results_dict['system'].append(co_dict['system'][0])
            results_dict['solution'].append(co_dict['solution'][0])
            results_dict['treatment'].append(co_dict['treatment'][0])
            results_dict['bead type'].append(bead_type)
            results_dict['bead radius'].append(bead_radius)
            results_dict['magnet'].append(magnet_id)
            
            case_mag = magnet_id + '_' + bead_type
            if calibFuncType == 'PowerLaw':
                parms_type = 'F_popt_pL'
                parms = MagnetDict[case_mag][parms_type]
                mag_d2f = (lambda x : powerLaw(x, *parms))
            elif calibFuncType == 'DoubleExpo':
                parms_type = 'F_popt_2exp'
                parms = MagnetDict[case_mag][parms_type]
                mag_d2f = (lambda x : doubleExpo(x, *parms))
            
            Track_Rd = {}
            if 'newton' in fits:
                N_Rd, N_error = pullAnalyzer(track, T_id, dict_pull, mag_d2f,
                                             mode = 'newton', 
                                             PLOT = PLOT, SHOW = SHOW, plotsDir = plotsDir)
                Track_Rd.update(N_Rd)
                
            if 'jeffrey' in fits:
                J_Rd, J_error = pullAnalyzer(track, T_id, dict_pull, mag_d2f,
                                             mode = 'jeffrey', 
                                             PLOT = PLOT, SHOW = SHOW, plotsDir = plotsDir)
                Track_Rd.update(J_Rd)
            
            for col in results_cols:
                try:
                    results_dict[col].append(Track_Rd[col])
                except:
                    results_dict[col].append(np.nan)
            
                
    # for k in results_dict.keys():
    #     print(k, len(results_dict[k]))
    
    new_df_res = pd.DataFrame(results_dict)
    df_res = pd.concat([df_res,new_df_res]).drop_duplicates(subset='track_id', keep='last').reset_index(drop=True)
            
    df_res.to_csv(os.path.join(resultsDir, resultsFileName + '.csv'), index=False)
    return(df_res)




def pullAnalyzer(track, track_id, dict_pull, mag_d2f,
                 mode = 'newton', 
                 PLOT = True, SHOW = False, plotsDir = ''):
    if SHOW:
        plt.ion()
    else:
        plt.ioff()
        
    error = False 

    #### 1. Parameters
    # track_id = track_id
    pull_id = dict_pull['id']
    frame_initPull = dict_pull['mag_fi']
    frame_endPull = dict_pull['mag_ff']
    pixel_size = dict_pull['film_pixel_size']  # µm
    mag_x = dict_pull['mag_x']*pixel_size
    mag_y = dict_pull['mag_y']*pixel_size
    film_dt = dict_pull['film_dt']/1000      # s
    mag_r = dict_pull['mag_r']*pixel_size # µm
    b_r = dict_pull['b_r'] # µm
    
    X, Y = track[:,1] * pixel_size, track[:,2] * pixel_size
    t_idx, t = track[:,0], track[:,0]*film_dt
    
    initPullTime = np.where(t_idx == (frame_initPull - 1))[0][0]
    finPullTime = np.where(t_idx == (frame_endPull - 1))[0][0]
    
    #### 2. Initialize results
    resultsDict = {}
    
    #### 4. Pretreatment
    # --- Rotate track ---
    theta = np.arctan2(Y[initPullTime] - Y[finPullTime],
                       X[initPullTime] - X[finPullTime])
    rotation_mat = np.array([[np.cos(-theta), -np.sin(-theta)],
                             [np.sin(-theta),  np.cos(-theta)]])
    coords = np.vstack((X, Y)).T @ rotation_mat.T
    x_rot, y_rot = coords[:,0], coords[:,1]

    x_shift = (-x_rot[initPullTime:] + np.max(x_rot[initPullTime:]))

    # --- Pulling phase ---
    pull_index = np.arange(initPullTime, finPullTime+1)
    pull_length = len(pull_index)
    xp, yp = x_rot[pull_index], y_rot[pull_index] # for plots
    d = np.stack([(mag_x - X[pull_index]),
                  (mag_y - Y[pull_index])], axis=1)
    XY = np.copy(d)
    XY[:,0] -= mag_x
    XY[:,1] -= mag_y
    
    #### Definition of the distance and the force
    dist = np.linalg.norm(d, axis=1) - (mag_r) # Original behaviour
    pull_force = mag_d2f(dist)
    tpulling = (t[pull_index] - t[initPullTime])
    dx_pulling = x_shift[pull_index - initPullTime]
    dx_pulling_n = dx_pulling / pull_force
    
    #### Filters
    if mode == 'newton':
        Filter1 = ((5 < tpulling) & (tpulling < 20))
    elif mode == 'jeffrey':
        Filter1 = (tpulling < 15)
    # Filter2 = (np.abs(dx_pulling[0] - dx_pulling[:]) <= 100) # keep only first 100 µm of movement
    
    filterPull = Filter1
    if np.sum(filterPull) == 0:
        error = True
    #     output = ()
    # filterPull = np.ones_like(dx_pulling).astype(bool)
    
    # --- Release phase ---
    try:
        release_index = np.arange(finPullTime, len(track))
        release_length = len(release_index)
        trelease = (t[release_index] - t[finPullTime+1])
        dx_release = x_shift[release_index - initPullTime]
        dx_release_n = dx_release / x_shift[pull_length]
        error_release = False
    except:
        print(track_id)
        print('Could not analyze the release phase !')
        error_release = True

    # --- Measures ---
    if not error:
        # M_results_cols = ['globalError', 'median instant speed', 'mean speed', 'median force', 
        #                'R min', 'R max', 'theta']
        med_instant_speed = np.median((dx_pulling[filterPull][1:]-dx_pulling[filterPull][:-1])/film_dt) # µm/s
        mean_speed = (dx_pulling[filterPull][0] - dx_pulling[filterPull][-1]) / (tpulling[filterPull][0] - tpulling[filterPull][-1]) # µm/s
        med_force = np.median(pull_force[filterPull]) # pN
        r_min, r_max = min(dist), max(dist)
        resultsDict['median instant speed'] = med_instant_speed
        resultsDict['mean speed'] = mean_speed
        resultsDict['median force'] = med_force
        resultsDict['R min'] = r_min
        resultsDict['R max'] = r_max
        resultsDict['theta'] = theta
    else:
        resultsDict['globalError'] = True
        resultsDict['median instant speed'] = np.nan
        resultsDict['mean speed'] = np.nan
        resultsDict['median force'] = np.nan
        resultsDict['R min'] = np.nan
        resultsDict['R max'] = np.nan
        resultsDict['theta'] = np.nan
        
    
    
    #### 5. Fit Model
    if mode == 'newton':
        # N_results_cols = ['N_fit_error', 'N_viscosity', 'N_R2']
        try:
            params, results = fitLine(tpulling[filterPull], dx_pulling_n[filterPull])
            slope = params[1]
            gamma = 1/slope
            visco = gamma/(6*np.pi*b_r)
            R2 = results.rsquared
            
            resultsDict['N_fit_error'] = False
            resultsDict['N_viscosity'] = visco
            resultsDict['N_R2'] = R2
            
        except:
            resultsDict['N_fit_error'] = True
            resultsDict['N_viscosity'] = np.nan
            resultsDict['N_R2'] = np.nan
        
    elif mode == 'jeffrey':
        # J_results_cols = ['J_fit_error', 'J_k', 'J_gamma1', 'J_gamma2', 'J_tempo1', 'J_tempo2', 'J_R2']
        # R_results_cols = ['R_fit_error', 'R_a', 'R_tau', 'R_R2']
        
        # Pulling phase
        try:
            J_start = list(guess_init_parms_jeffrey(tpulling[filterPull], dx_pulling_n[filterPull]))
        
            
            def jeffrey_model(params, x):
                k, gamma1, gamma2 = params
                return (1 - np.exp(-k*x/gamma1))/k + x/gamma2
            
            def jeffrey_model_constraint(params, x):
                k, gamma1, r = params
                gamma2 = 3*gamma1 + r**2
                return (1 - np.exp(-k*x/gamma1))/k + x/gamma2
            
            #### Fit pulling phase --- V1
            # obj1 = lambda params: np.linalg.norm(jeffrey_model(params, tpulling) - dx_pulling_n)
            # res1 = minimize(obj1, start1, method="Nelder-Mead", tol=1e-10,
            #                 options={"maxfev": 1000})
            # k, gamma1, gamma2 = res1.x
            # # tempo1, tempo2 = gamma1/k, gamma2/k
            # Ymeas = dx_pulling_n
            # Yfit = jeffrey_model((k, gamma1, gamma2), tpulling)
            # R2_p = get_R2(Ymeas, Yfit) 
            
            #### Fit pulling phase --- V2
            g1_i = J_start[1]
            g2_i = J_start[2]
            r_i = max(1e-6, (g2_i - 3*g1_i))**0.5
            J_start[2] = r_i
            
            obj1 = lambda params: np.linalg.norm(jeffrey_model_constraint(params, tpulling[filterPull]) - dx_pulling_n[filterPull])
            res1 = minimize(obj1, J_start, method="Nelder-Mead", tol=1e-10,
                            options={"maxfev": 1000})
            k, gamma1, r = res1.x
            gamma2 = 3*gamma1 + r**2
            Ymeas = dx_pulling_n[filterPull]
            Yfit = jeffrey_model((k, gamma1, gamma2), tpulling[filterPull])
            J_R2 = get_R2(Ymeas, Yfit) 
            
            if min(k, gamma1, gamma2) <= 0:
                raise Exception("Jeffrey fit error: null or negative parameter")
            
            # J_results_cols = ['J_fit_error', 'J_k', 'J_gamma1', 'J_gamma2', 'J_tempo1', 'J_tempo2', 'J_R2']
            resultsDict['J_fit_error'] = False
            resultsDict['J_k'] = k
            resultsDict['J_gamma1'] = gamma1
            resultsDict['J_gamma2'] = gamma2
            resultsDict['J_tempo1'] = gamma1/k
            resultsDict['J_tempo2'] = gamma2/k
            resultsDict['J_R2'] = J_R2
        
        except:
            resultsDict['J_fit_error'] = True
            resultsDict['J_k'] = np.nan
            resultsDict['J_gamma1'] = np.nan
            resultsDict['J_gamma2'] = np.nan
            resultsDict['J_tempo1'] = np.nan
            resultsDict['J_tempo2'] = np.nan
            resultsDict['J_R2'] = np.nan



        #### Fit release phase
        try:
            R_start = list(guess_init_parms_relax(trelease, dx_release_n))
            
            def exp_fit(params, x):
                a, tau = params
                return (1-a)*np.exp(-x/tau) + a
    
            obj2 = lambda params: np.linalg.norm(exp_fit(params, trelease) - dx_release_n)
            res2 = minimize(obj2, R_start, method="Nelder-Mead", tol=1e-7,
                            options = {"maxfev": 1000})
            a, tau = res2.x
            Ymeas = dx_release_n
            Yfit = exp_fit((a, tau), trelease)
            R_R2 = get_R2(Ymeas, Yfit) 
            
            # R_results_cols = ['R_fit_error', 'R_a', 'R_tau', 'R_R2']
            resultsDict['R_fit_error'] = False
            resultsDict['R_a'] = a
            resultsDict['R_tau'] = tau
            resultsDict['R_R2'] = R_R2
        
        except:
            resultsDict['R_fit_error'] = True
            resultsDict['R_a'] = np.nan
            resultsDict['R_tau'] = np.nan
            resultsDict['R_R2'] = np.nan
    
    
    #### 6. Figures
    if PLOT:
        fig1, axes1 = plt.subplots(1, 2, figsize = (10,5))
        ax = axes1[0]
        ax.plot(X, Y, ".-")
        ax.axis("equal")
        ax.grid(axis='both')
        ax.set_title("Original track")
        
        ax = axes1[1]
        ax.plot(x_rot, y_rot, ".-")
        ax.plot(xp, yp, "k.-", linewidth=2)
        ax.axis("equal")
        ax.grid(axis='both')
        ax.set_title("Rotated track")
        fig1.tight_layout()
        
        fig1.savefig(os.path.join(plotsDir, track_id + "_Trajectories.png"))
            
    
    if mode == 'newton' and PLOT:
        fig2, axes2 = plt.subplots(1, 1, figsize = (5,5))
        ax = axes2
        axbis = axes2.twinx()
        axbis.plot(tpulling, dist, 'g--', zorder=4)
        axbis.set_ylabel('Distance from magnet tip [µm]')
        ax.plot(tpulling[filterPull], dx_pulling_n[filterPull], 
                ls='', marker='o', color='darkturquoise', markersize=5, zorder=5)
        ax.plot(tpulling[~filterPull], dx_pulling_n[~filterPull], 
                ls='', marker='o', color='lightblue', markersize=5, zorder=4)
        if not resultsDict['N_fit_error']:
            xfit = np.linspace(0, tpulling[-1], 100)
            yfit = params[0] + params[1]*xfit
            ax.plot(xfit, yfit, "r-", label=r'$\eta$ = ' + f'{visco:.2f} Pa.s', zorder=6)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("dx/f [µm/pN]")
        ax.grid(axis='both')
        ax.legend(fontsize = 11).set_zorder(6)
        fig2.suptitle(pull_id, fontsize=12)
        fig2.tight_layout()
        
        fig2.savefig(os.path.join(plotsDir, track_id + "_NewtonFits.png"))
            
    
    if mode == 'jeffrey' and PLOT:   
        fig2, axes2 = plt.subplots(1, 2, figsize=(8,5))
        ax = axes2[0]
        ax.plot(tpulling[filterPull], dx_pulling_n[filterPull], 
                ls='', marker='o', color='darkturquoise', markersize=5, zorder=5)
        ax.plot(tpulling[~filterPull], dx_pulling_n[~filterPull], 
                ls='', marker='o', color='lightblue', markersize=5, zorder=4)
        if not resultsDict['J_fit_error']:
            label1 = "Fitting Jeffrey's model\n" + r"$\frac{1}{k}(1 - exp(-k.t/\gamma_1)) + t/\gamma_2$" + "\n"
            label1+= r"$k$ = " + f"{k:.2f}\n"
            label1+= r"$\gamma_1$ = " + f"{gamma1:.2f}\n"
            label1+= r"$\gamma_2$ = " + f"{gamma2:.2f}"
            tp_fit = np.linspace(0, tpulling[-1], 200)
            dx_n_fit = jeffrey_model([k, gamma1, gamma2], tp_fit)
            ax.plot(tp_fit, dx_n_fit, "r-",
                     label=label1, zorder=6)
        ax.legend()
        ax.grid()
        ax.set_xlabel("t [s]")
        ax.set_ylabel("dx/f [µm/pN]")
    
    
        ax = axes2[1]
        if not error_release:
            ax.plot(trelease, dx_release_n, 
                    ls='', marker='o', color='limegreen')
            if not resultsDict['R_fit_error']:
                label2 = "Fitting Viscoel Relax\n" + r"$(1-a). exp(-t/\tau ) + a$" + "\n"
                label2+= r"$a$ = " + f"{a:.2f}\n"
                label2+= r"$\tau$ = " + f"{tau:.2f}"
                tr_fit = np.linspace(0, trelease[-1], 200)
                dx_release_fit = exp_fit([a,tau], tr_fit)
                ax.plot(tr_fit, dx_release_fit, 
                        ls="-", c='darkorange', label=label2)
        ax.legend()
        ax.grid()
        ax.set_xlabel("t [s]")
        ax.set_ylabel("Normalized displacement")
        ax.set_ylim([0, 1.2])
        plt.tight_layout()
        fig2.savefig(os.path.join(plotsDir, track_id + "_JeffreyFits.png"))
        
        
    if SHOW:
        plt.show()
    else:
        plt.close('all')
            
    plt.ion()
    
    resultsDict['globalError'] = error
    
    #### 7. Output
    return(resultsDict, error)



def pullAnalyzer_compareTracks(list_tracks, list_track_ids, list_dict_pull, list_mag_d2f,
                                 mode = 'newton', 
                                 PLOT = True, SHOW = False, plotsDir = '', plotFileTitle = ''):
    if SHOW:
        plt.ion()
    else:
        plt.ioff()
        
    N = len(list_tracks)
    
    if mode == 'newton':
        fig, axes = plt.subplots(1, N, figsize=(N*5, 5), sharex='row', sharey='row')
        
    elif mode == 'jeffrey':
        fig, axes = plt.subplots(2, N, figsize=(N*5, 8), sharex='row', sharey='row')
    
    for it in range(N):
        track = list_tracks[it]
        track_id = list_track_ids[it]
        dict_pull = list_dict_pull[it]
        mag_d2f = list_mag_d2f[it]
        print(it, track_id)
        
        error = False 
    
        #### 1. Parameters
        # track_id = track_id
        pull_id = dict_pull['id']
        frame_initPull = dict_pull['mag_fi']
        frame_endPull = dict_pull['mag_ff']
        pixel_size = dict_pull['film_pixel_size']  # µm
        mag_x = dict_pull['mag_x']*pixel_size
        mag_y = dict_pull['mag_y']*pixel_size
        film_dt = dict_pull['film_dt']/1000      # s
        mag_r = dict_pull['mag_r']*pixel_size # µm
        b_r = dict_pull['b_r'] # µm
        
        X, Y = track[:,1] * pixel_size, track[:,2] * pixel_size
        t_idx, t = track[:,0], track[:,0]*film_dt
        
        initPullTime = np.where(t_idx == (frame_initPull - 1))[0][0]
        finPullTime = np.where(t_idx == (frame_endPull - 1))[0][0]
        
        #### 2. Initialize results
        resultsDict = {}
        
        #### 4. Pretreatment
        # --- Rotate track ---
        theta = np.arctan2(Y[initPullTime] - Y[finPullTime],
                           X[initPullTime] - X[finPullTime])
        rotation_mat = np.array([[np.cos(-theta), -np.sin(-theta)],
                                 [np.sin(-theta),  np.cos(-theta)]])
        coords = np.vstack((X, Y)).T @ rotation_mat.T
        x_rot, y_rot = coords[:,0], coords[:,1]
    
        x_shift = (-x_rot[initPullTime:] + np.max(x_rot[initPullTime:]))
    
        # --- Pulling phase ---
        pull_index = np.arange(initPullTime, finPullTime+1)
        pull_length = len(pull_index)
        xp, yp = x_rot[pull_index], y_rot[pull_index] # for plots
        d = np.stack([(mag_x - X[pull_index]),
                      (mag_y - Y[pull_index])], axis=1)
        XY = np.copy(d)
        XY[:,0] -= mag_x
        XY[:,1] -= mag_y
        
        #### Definition of the distance and the force
        dist = np.linalg.norm(d, axis=1) - (mag_r) # Original behaviour
        pull_force = mag_d2f(dist)
        tpulling = (t[pull_index] - t[initPullTime])
        dx_pulling = x_shift[pull_index - initPullTime]
        dx_pulling_n = dx_pulling / pull_force
        
        #### Filters
        if mode == 'newton':
            Filter1 = ((10 < tpulling) & (tpulling < 200))
        elif mode == 'jeffrey':
            Filter1 = (tpulling < 200)
        # Filter2 = (np.abs(dx_pulling[0] - dx_pulling[:]) <= 100) # keep only first 100 µm of movement
        
        filterPull = Filter1
        if np.sum(filterPull) == 0:
            error = True
        #     output = ()
        # filterPull = np.ones_like(dx_pulling).astype(bool)
        
        # --- Release phase ---
        try:
            release_index = np.arange(finPullTime, len(track))
            release_length = len(release_index)
            trelease = (t[release_index] - t[finPullTime+1])
            dx_release = x_shift[release_index - initPullTime]
            dx_release_n = dx_release / x_shift[pull_length]
            error_release = False
        except:
            print(track_id)
            print('Could not analyze the release phase !')
            error_release = True
    
        # --- Measures ---
        if not error:
            # M_results_cols = ['globalError', 'median instant speed', 'mean speed', 'median force', 
            #                'R min', 'R max', 'theta']
            med_instant_speed = np.median((dx_pulling[filterPull][1:]-dx_pulling[filterPull][:-1])/film_dt) # µm/s
            mean_speed = (dx_pulling[filterPull][0] - dx_pulling[filterPull][-1]) / (tpulling[filterPull][0] - tpulling[filterPull][-1]) # µm/s
            med_force = np.median(pull_force[filterPull]) # pN
            r_min, r_max = min(dist), max(dist)
            resultsDict['median instant speed'] = med_instant_speed
            resultsDict['mean speed'] = mean_speed
            resultsDict['median force'] = med_force
            resultsDict['R min'] = r_min
            resultsDict['R max'] = r_max
            resultsDict['theta'] = theta
        else:
            resultsDict['globalError'] = True
            resultsDict['median instant speed'] = np.nan
            resultsDict['mean speed'] = np.nan
            resultsDict['median force'] = np.nan
            resultsDict['R min'] = np.nan
            resultsDict['R max'] = np.nan
            resultsDict['theta'] = np.nan
            
        
        
        #### 5. Fit Model
        if mode == 'newton':
            # N_results_cols = ['N_fit_error', 'N_viscosity', 'N_R2']
            try:
                params, results = fitLine(tpulling[filterPull], dx_pulling_n[filterPull])
                slope = params[1]
                gamma = 1/slope
                visco = gamma/(6*np.pi*b_r)
                R2 = results.rsquared
                
                resultsDict['N_fit_error'] = False
                resultsDict['N_viscosity'] = visco
                resultsDict['N_R2'] = R2
                
            except:
                resultsDict['N_fit_error'] = True
                resultsDict['N_viscosity'] = np.nan
                resultsDict['N_R2'] = np.nan
            
        elif mode == 'jeffrey':
            # J_results_cols = ['J_fit_error', 'J_k', 'J_gamma1', 'J_gamma2', 'J_tempo1', 'J_tempo2', 'J_R2']
            # R_results_cols = ['R_fit_error', 'R_a', 'R_tau', 'R_R2']
            
            # Pulling phase
            try:
                J_start = list(guess_init_parms_jeffrey(tpulling[filterPull], dx_pulling_n[filterPull]))
            
                
                def jeffrey_model(params, x):
                    k, gamma1, gamma2 = params
                    return (1 - np.exp(-k*x/gamma1))/k + x/gamma2
                
                def jeffrey_model_constraint(params, x):
                    k, gamma1, r = params
                    gamma2 = 3*gamma1 + r**2
                    return (1 - np.exp(-k*x/gamma1))/k + x/gamma2
                
                #### Fit pulling phase --- V1
                # obj1 = lambda params: np.linalg.norm(jeffrey_model(params, tpulling) - dx_pulling_n)
                # res1 = minimize(obj1, start1, method="Nelder-Mead", tol=1e-10,
                #                 options={"maxfev": 1000})
                # k, gamma1, gamma2 = res1.x
                # # tempo1, tempo2 = gamma1/k, gamma2/k
                # Ymeas = dx_pulling_n
                # Yfit = jeffrey_model((k, gamma1, gamma2), tpulling)
                # R2_p = get_R2(Ymeas, Yfit) 
                
                #### Fit pulling phase --- V2
                g1_i = J_start[1]
                g2_i = J_start[2]
                r_i = max(1e-6, (g2_i - 3*g1_i))**0.5
                J_start[2] = r_i
                
                obj1 = lambda params: np.linalg.norm(jeffrey_model_constraint(params, tpulling[filterPull]) - dx_pulling_n[filterPull])
                res1 = minimize(obj1, J_start, method="Nelder-Mead", tol=1e-10,
                                options={"maxfev": 1000})
                k, gamma1, r = res1.x
                gamma2 = 3*gamma1 + r**2
                Ymeas = dx_pulling_n[filterPull]
                Yfit = jeffrey_model((k, gamma1, gamma2), tpulling[filterPull])
                J_R2 = get_R2(Ymeas, Yfit) 
                
                if min(k, gamma1, gamma2) <= 0:
                    raise Exception("Jeffrey fit error: null or negative parameter")
                
                # J_results_cols = ['J_fit_error', 'J_k', 'J_gamma1', 'J_gamma2', 'J_tempo1', 'J_tempo2', 'J_R2']
                resultsDict['J_fit_error'] = False
                resultsDict['J_k'] = k
                resultsDict['J_gamma1'] = gamma1
                resultsDict['J_gamma2'] = gamma2
                resultsDict['J_tempo1'] = gamma1/k
                resultsDict['J_tempo2'] = gamma2/k
                resultsDict['J_R2'] = J_R2
            
            except:
                resultsDict['J_fit_error'] = True
                resultsDict['J_k'] = np.nan
                resultsDict['J_gamma1'] = np.nan
                resultsDict['J_gamma2'] = np.nan
                resultsDict['J_tempo1'] = np.nan
                resultsDict['J_tempo2'] = np.nan
                resultsDict['J_R2'] = np.nan
    
    
    
            #### Fit release phase
            try:
                R_start = list(guess_init_parms_relax(trelease, dx_release_n))
                
                def exp_fit(params, x):
                    a, tau = params
                    return (1-a)*np.exp(-x/tau) + a
        
                obj2 = lambda params: np.linalg.norm(exp_fit(params, trelease) - dx_release_n)
                res2 = minimize(obj2, R_start, method="Nelder-Mead", tol=1e-7,
                                options = {"maxfev": 1000})
                a, tau = res2.x
                Ymeas = dx_release_n
                Yfit = exp_fit((a, tau), trelease)
                R_R2 = get_R2(Ymeas, Yfit) 
                
                # R_results_cols = ['R_fit_error', 'R_a', 'R_tau', 'R_R2']
                resultsDict['R_fit_error'] = False
                resultsDict['R_a'] = a
                resultsDict['R_tau'] = tau
                resultsDict['R_R2'] = R_R2
            
            except:
                resultsDict['R_fit_error'] = True
                resultsDict['R_a'] = np.nan
                resultsDict['R_tau'] = np.nan
                resultsDict['R_R2'] = np.nan
    
    
        #### 6. Figures
        if mode == 'newton' and PLOT:
            ax = axes[it]
            axbis = ax.twinx()
            axbis.plot(tpulling, dist, 'g--', zorder=4)
            axbis.set_ylabel('Distance from magnet tip [µm]')
            ax.plot(tpulling[filterPull], dx_pulling_n[filterPull], 
                    ls='', marker='o', color='darkturquoise', markersize=5, zorder=5)
            ax.plot(tpulling[~filterPull], dx_pulling_n[~filterPull], 
                    ls='', marker='o', color='lightblue', markersize=5, zorder=4)
            if not resultsDict['N_fit_error']:
                xfit = np.linspace(0, tpulling[-1], 100)
                yfit = params[0] + params[1]*xfit
                ax.plot(xfit, yfit, "r-", 
                        label=r'$\eta$ = ' + f'{visco:.2f} Pa.s', zorder=6)
            ax.set_xlabel("t [s]")
            ax.set_ylabel("dx/f [µm/pN]")
            ax.grid(axis='both')
            ax.legend(fontsize = 11).set_zorder(6)
            ax.set_title(track_id, fontsize=10)
        
                
        
        if mode == 'jeffrey' and PLOT:   
            ax = axes[0, it]
            ax.plot(tpulling[filterPull], dx_pulling_n[filterPull], 
                    ls='', marker='o', color='darkturquoise', markersize=5, zorder=5)
            ax.plot(tpulling[~filterPull], dx_pulling_n[~filterPull], 
                    ls='', marker='o', color='lightblue', markersize=5, zorder=4)
            if not resultsDict['J_fit_error']:
                label1 = "Fitting Jeffrey's model\n" + r"$\frac{1}{k}(1 - exp(-k.t/\gamma_1)) + t/\gamma_2$" + "\n"
                label1+= r"$k$ = " + f"{k:.2f}\n"
                label1+= r"$\gamma_1$ = " + f"{gamma1:.2f}\n"
                label1+= r"$\gamma_2$ = " + f"{gamma2:.2f}"
                tp_fit = np.linspace(0, tpulling[-1], 200)
                dx_n_fit = jeffrey_model([k, gamma1, gamma2], tp_fit)
                ax.plot(tp_fit, dx_n_fit, "r-",
                         label=label1, zorder=6)
            ax.legend()
            ax.grid()
            ax.set_xlabel("t [s]")
            if it==0:
                ax.set_ylabel("dx/f [µm/pN]")
            ax.set_title(track_id, fontsize=10)
        
            ax = axes[1, it]
            if not error_release:
                ax.plot(trelease, dx_release_n, 
                        ls='', marker='o', color='limegreen')
                if not resultsDict['R_fit_error']:
                    label2 = "Fitting Viscoel Relax\n" + r"$(1-a). exp(-t/\tau ) + a$" + "\n"
                    label2+= r"$a$ = " + f"{a:.2f}\n"
                    label2+= r"$\tau$ = " + f"{tau:.2f}"
                    tr_fit = np.linspace(0, trelease[-1], 200)
                    dx_release_fit = exp_fit([a,tau], tr_fit)
                    ax.plot(tr_fit, dx_release_fit, 
                            ls="-", c='darkorange', label=label2)
            ax.legend()
            ax.grid()
            ax.set_xlabel("t [s]")
            if it==0:
                ax.set_ylabel("Normalized displacement")
            ax.set_ylim([0, 1.2])
        

    plt.tight_layout()
    fig.savefig(os.path.join(plotsDir, plotFileTitle + '_' + mode + "_comparedFits.png"))
    
    if SHOW:
        plt.show()
    else:
        plt.close('all')
            
    plt.ion()
    
    resultsDict['globalError'] = error
    
    #### 7. Output
    return(resultsDict, error)




# %% 12. Run the functions

# %%% ... on many files

# mainDir = os.path.join("C:/Users/Utilisateur/Desktop/") # Ordi IJM
# mainDir = os.path.join("C:/Users/josep/Desktop/Seafile") # Ordi perso
mainDir = os.path.join("C:/Users/Joseph/Desktop/") # Ordi LJP
date = '26-01-27'
subfolder = date + '_BeadTracking'


analysisDir = os.path.join(mainDir, 'AnalysisPulls') # where the csv tables are
tracksDir = os.path.join(analysisDir, subfolder, 'Tracks') # where the tracks are
resultsDir = os.path.join(analysisDir, subfolder, 'Results')
plotsDir = os.path.join(analysisDir, subfolder, 'Plots')

# cell = '_M1_C1_Pa0_P3'

prefix_id = '26-01-27' # + cell # used to select a subset of the track files if needed

Results = pullAnalyzer_multiFiles(mainDir, date, prefix_id,
                                    analysisDir, tracksDir, resultsDir, plotsDir,
                                    fits = ['newton', 'jeffrey'], calibFuncType='PowerLaw',
                                    resultsFileName = date + '_BeadsPulling',
                                    Redo = True, PLOT = True, SHOW = False)

# plt.close('all')

# %%% ... on one file

mainDir = os.path.join("C:/Users/Utilisateur/Desktop/")
date = '26-01-14'
subfolder = date + '_BeadTracking'

# analysisDir = os.path.join(mainDir, 'AnalysisPulls') # where the csv tables are
# tracksDir = os.path.join(analysisDir, 'Tracks', date) # where the tracks are
# resultsDir = os.path.join(analysisDir, 'Results')
# plotsDir = os.path.join(analysisDir, 'Plots', date)

analysisDir = os.path.join(mainDir, 'AnalysisPulls') # where the csv tables are
tracksDir = os.path.join(analysisDir, subfolder, 'Tracks') # where the tracks are
resultsDir = os.path.join(analysisDir, subfolder, 'Results')
plotsDir = os.path.join(analysisDir, subfolder, 'Plots')

# P_id = '25-09-19_M1_D5_P1_B1'
# P_id = '25-09-19_M1_D9_P1_B1'
# P_id = '25-09-19_M1_D8_P1_B1'
# P_id = '25-09-19_M1_D7_P1_B1'
# P_id = '25-09-19_M1_D6_P2_B1'
P_id = '26-01-14_M1_C4_Pa1_P2'
# P_id = '25-09-19_M2_D4_P1_B1' # used to select a subset of the track files if needed

df_manips = pd.read_csv(os.path.join(analysisDir, 'MainExperimentalConditions.csv'))
df_pulls = pd.read_csv(os.path.join(analysisDir, date + '_ExperimentalConditions.csv'))
df_id = df_pulls[df_pulls['id'] == P_id]
dict_pull = Df2Dict(df_pulls[df_pulls['id'] == P_id])
trackFileNames = [f for f in os.listdir(tracksDir) if f.startswith(P_id)]

tracks = [importTrackMateTracks(os.path.join(tracksDir, fN)) for fN in trackFileNames]
tracks = [trackSelection(T, mode = 'longest') for T in tracks]
beads_id = [int(fN[len(P_id)-1:len(P_id)+0]) for fN in trackFileNames]

# fit_V = 'LangevinFun'
# fit_V = 'DoubleExpo'

F_popt_pL = [39603.33040969049,
             -2.0162526263553215]
mag_d2f = lambda x : powerLaw(x, *F_popt_pL)

for track in tracks:
    output, dx_pulling_n, t, error = pullAnalyzer(track, dict_pull, mag_d2f,
                                 mode = 'jeffrey', 
                                 PLOT = True, SHOW = True, plotsDir = plotsDir)
    
    (k, gamma1, gamma2, R2_p, speed_med, force_med, a, tau, R2_r, theta, r_min, r_max, XY) = output









# %% ---------------

# %% Plot some stuff


# %%% Dataframe

# dirData = os.path.join("C:/Users/Utilisateur/Desktop/AnalysisPulls/") # Ordi IJM
# dirData = os.path.join("C:/Users/josep/Desktop/Seafile/AnalysisPulls/") # Ordi perso
dirData = os.path.join("C:/Users/Joseph/Desktop/AnalysisPulls/") # Ordi LJP
# dirData = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/26-01-14_BeadTracking/Results'
# dirData = 'C:/Users/Utilisateur/Desktop/AnalysisPulls/'
# fileName = '26-01-14_BeadsPulling_15s.csv'
fileName = 'AllResults_BeadsPulling.csv'
filePath = os.path.join(dirData, fileName)

specif = '_26-01-27_NoI2959'

df = pd.read_csv(filePath)

df['date'] = df['track_id'].apply(lambda x : x.split('_')[0])
df['Lc'] = 6*np.pi*df['bead radius']
df['J_modulus'] = df['J_k'] / df['Lc']
df['J_visco1'] = df['J_gamma1'] / df['Lc']
df['J_visco2'] = df['J_gamma2'] / df['Lc']

Filters = [(df['J_gamma2'] < 500),
           (df['J_fit_error'] == False),
           (df['J_R2'] >= 0.7),
           (df['date'] == '26-01-27'),
           (df['cell_id'] != '26-01-14_M1_C1'),
           ]
GlobalFilter = np.ones_like(Filters[0]).astype(bool)
for F in Filters:
    GlobalFilter = GlobalFilter & F

df_f = df[GlobalFilter]

savePath = dirData

# %%% Plots by pull

metrics = ['J_k', 'J_gamma1', 'J_gamma2', 'N_viscosity', ]
metric_names = ['$k_1$ (pN/µm)', '$\gamma_1$ (pN.s/µm)', '$\gamma_2$ (pN.s/µm)', '$\eta_N$ (Pa.s)']
metric_dict = {m:mn for (m,mn) in zip(metrics, metric_names)}

fig, axes = plt.subplots(2, 2, figsize=(8,5))
axes_f = axes.flatten()

for k in range(4):
    ax = axes_f[k]
    metric = metrics[k]
    
    medians = [np.median(df_f.loc[df_f['Pa']==i, metric]) for i in [0, 1]]
    
    for i in [0, 1]:
        HW = 0.35
        ax.plot([i-HW, i+HW], [medians[i], medians[i]], ls='--', lw=2, c='dimgrey')
    
    sns.swarmplot(data = df_f, ax=ax, 
                  x = 'Pa', y = metric, hue = 'cell_id',
                  size = 6)

    ax.set_xlim([-0.5, 1.5])
    ax.set_xticks([0, 1], ['Ctrl', '+UV'])
    ax.set_xlabel('')
    ax.set_ylabel(metric_dict[metric])
    yM = ax.get_ylim()[1]
    ax.set_ylim([0, 1.25*yM])
    ax.grid(axis='y')
    
    for i in [0, 1]:
        ax.text(i, 1.1*yM, f'{medians[i]:.2f}', ha='center', size = 10, style='italic', c='dimgrey')
    
    if k == 1:
        ax.legend(title='Cell IDs',
                  loc="upper left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
    else:
        ax.legend().set_visible(False)

fig.suptitle('All pulls & beads' + specif, fontsize=16)
fig.tight_layout()
plt.show()

fig.savefig(savePath + '/res_allPulls' + specif + '.png', dpi=500)


# %%% Plots by cell

metrics = ['J_k', 'J_gamma1', 'J_gamma2', 'N_viscosity', ]
group = df_f.groupby(['cell_id', 'Pa'])
agg_dict = {m:'mean' for m in metrics}
df_fg = group.agg(agg_dict)
df_fg = df_fg.reset_index()
df_fg = df_fg[df_fg['cell_id'] != '26-01-14_M1_C1']

list_cell_id = df_fg['cell_id'].values
for metric in agg_dict.keys():
    A_norm = np.array([df_fg.loc[((df_fg['Pa']==0) & (df_fg['cell_id']==cid)), metric] for cid in list_cell_id]).T
    A_norm = A_norm[0]
    df_fg[metric + '_norm'] = df_fg[metric] / A_norm

metric_names = ['$k_1$ (pN/µm)', '$\gamma_1$ (pN.s/µm)', '$\gamma_2$ (pN.s/µm)', '$\eta_N$ (Pa.s)']
metric_dict = {m:mn for (m,mn) in zip(metrics, metric_names)}

fig, axes = plt.subplots(2, 2)
axes_f = axes.flatten()

for k in range(4):
    ax = axes_f[k]
    metric = metrics[k]
    
    for cid, c in zip(df_fg['cell_id'].unique(), pm.colorList10):
        val0 = df_fg.loc[((df_fg['Pa']==0) & (df_fg['cell_id']==cid)), metric].values[0]
        val1 = df_fg.loc[((df_fg['Pa']==1) & (df_fg['cell_id']==cid)), metric].values[0]
        ax.plot(0, val0, 'o', c=c, zorder = 5)
        ax.plot(1, val1, 'o', c=c, zorder = 5)
        ax.plot([0, 1], [val0, val1], ls='-', c='dimgray', zorder = 4)
        
    
    
    ax.set_xlim([-0.5, 1.5])
    ax.set_xticks([0, 1], ['Ctrl', '+UV'])
    ax.set_xlabel('')
    ax.set_ylabel(metric_dict[metric])
    yM = ax.get_ylim()[1]
    ax.set_ylim([0, 1.1*yM])
    
    ax.grid(axis='y')
        
fig.suptitle('Average values by cell' + specif, fontsize=16)
fig.tight_layout()
plt.show()
    
fig.savefig(savePath + '/res_avgByCell' + specif + '.png', dpi=500)
    
    
metric_names_2 = ['$k_1$ (ratio)', '$\gamma_1$ (ratio)', '$\gamma_2$ (ratio)', '$\eta_N$ (ratio)']
metric_dict_2 = {m+'_norm':mn for (m,mn) in zip(metrics, metric_names_2)}

fig, axes = plt.subplots(2, 2)
axes_f = axes.flatten()

for k in range(4):
    ax = axes_f[k]
    metric = metrics[k] + '_norm'
    
    for cid, c in zip(df_fg['cell_id'].unique(), pm.colorList10):
        val0 = df_fg.loc[((df_fg['Pa']==0) & (df_fg['cell_id']==cid)), metric].values[0]
        val1 = df_fg.loc[((df_fg['Pa']==1) & (df_fg['cell_id']==cid)), metric].values[0]
        ax.plot(0, val0, 'o', c=c, zorder = 5)
        ax.plot(1, val1, 'o', c=c, zorder = 5)
        ax.plot([0, 1], [val0, val1], ls='-', c='dimgray', zorder = 4)
    
    ax.set_xlim([-0.5, 1.5])
    ax.set_xticks([0, 1], ['Ctrl', '+UV'])
    ax.set_xlabel('')
    ax.set_ylabel(metric_dict_2[metric])
    yM = ax.get_ylim()[1]
    ax.set_ylim([0, 1.1*yM])
    
    ax.grid(axis='y')

fig.suptitle('Average values by cell, normalized' + specif, fontsize=16)    
fig.tight_layout()
plt.show()
    
fig.savefig(savePath + '/res_avgByCell_norm' + specif + '.png', dpi=500)
    
    
    

# %%% Data for one bead across pullings

Filters = [
           (df_f['cell_id'] == '26-01-27_M1_C1'),
           (df_f['bead'] == 1),
           ]

iUV = 3 # nPulls_beforeUV

GlobalFilter = np.ones_like(Filters[0]).astype(bool)
for F in Filters:
    GlobalFilter = GlobalFilter & F

df_1c1b = df_f[GlobalFilter]
df_1c1b['Idx Pull'] = np.arange(1, 1+len(df_1c1b))

fig, ax = plt.subplots(1, 1, figsize = (5,4))


ax.plot(df_1c1b['Idx Pull'], df_1c1b['J_visco2'], 'o', markerfacecolor = 'None', label='$\eta_2$ (Jeffrey)')
ax.plot(df_1c1b['Idx Pull'], df_1c1b['N_viscosity'], 'o', markerfacecolor = 'None', label='$\eta_N$  (Newton)')
ax.axvspan(iUV + 0.25, iUV + 0.75, color='indigo', zorder=5)
ymid = np.mean(ax.get_ylim())
ax.text(iUV + 0.5, ymid, 'UV', c='w', style='italic', size=10, ha='center', va='center', zorder=6)

ax.set_xlabel('Pull No.')
ax.set_ylabel('$\eta$ (Pa.s)')

ax.grid(axis='y')
ax.legend()

fig.suptitle('Repeated pulls on 26-01-27_M1_C1 Bead n°1', fontsize = 12)
fig.tight_layout()
plt.show()



fig.savefig(savePath + '/RepPull_v2vN_26-01-27_M1_C1_B1.png', dpi=500)

# ------

fig, ax = plt.subplots(1, 1, figsize = (5.5,4))
axbis = ax.twinx()

ax.plot(df_1c1b['Idx Pull'], df_1c1b['J_visco1'], 'o', color=pm.colorList10[0], markerfacecolor = 'None')
ax.plot([], [], 'o', color=pm.colorList10[1], markerfacecolor = 'None', label='k')
axbis.plot(df_1c1b['Idx Pull'], df_1c1b['J_k'], 'o', color=pm.colorList10[1], markerfacecolor = 'None')
ax.axvspan(iUV + 0.25, iUV + 0.75, color='indigo', zorder=5)
ymid = np.mean(ax.get_ylim())
ax.text(iUV + 0.5, ymid, 'UV', c='w', style='italic', size=10, ha='center', va='center', zorder=6)

ax.set_xlabel('Pull No.')
ax.set_ylabel('$\eta_1$ (Pa.s)', color=pm.colorList10[0])
axbis.set_ylabel('$k_1$ (pN/µm)', color=pm.colorList10[1])

ax.grid(axis='y')
# ax.legend() 

fig.suptitle('Repeated pulls on 26-01-27_M1_C1 Bead n°1', fontsize = 12)
fig.tight_layout()
plt.show()


fig.savefig(savePath + '/RepPull_k1v1_26-01-27_M1_C1_B1.png', dpi=500)


# %%% Compare before / after UV on an example

#### Parms

mainDir = os.path.join("C:/Users/Utilisateur/Desktop/")
date = '26-01-14'
subfolder = date + '_BeadTracking'

analysisDir = os.path.join(mainDir, 'AnalysisPulls') # where the csv tables are
tracksDir = os.path.join(analysisDir, subfolder, 'Tracks') # where the tracks are
resultsDir = os.path.join(analysisDir, subfolder, 'Results')
plotsDir = os.path.join(analysisDir, subfolder, 'Plots')

calibFuncType = 'PowerLaw'
PLOT, SHOW = True, True

listTrackFiles = [
    '26-01-14_M1_C3_Pa0_P2_B1_Tracks.xml',
    '26-01-14_M1_C3_Pa1_P1_B1_Tracks.xml',
    ]


#### Data
if not os.path.exists(resultsDir):
    os.makedirs(resultsDir)
if not os.path.exists(plotsDir):
    os.makedirs(plotsDir)
    
df_manips = pd.read_csv(os.path.join(analysisDir, 'MainExperimentalConditions.csv'))
df_pulls = pd.read_csv(os.path.join(analysisDir, date + '_ExperimentalConditions.csv'))

listTrackIds = ['_'.join(string.split('_')[:6]) for string in listTrackFiles]
listPullIds = ['_'.join(string.split('_')[:5]) for string in listTrackFiles]
listManipIds = ['_'.join(string.split('_')[:2]) for string in listTrackFiles]
list_co_dict = [df_manips[df_manips['id'] == mid].to_dict('list') for mid in listManipIds]

dict_TrackIds2File = {tid : tf for (tid, tf) in zip(listTrackIds, listTrackFiles)}
dict_TrackIds2PullId = {tid : pid for (tid, pid) in zip(listTrackIds, listPullIds)}

list_tracks, list_track_ids, list_dict_pull, list_mag_d2f = [], [], [], []

for it, T_id in enumerate(listTrackIds):
    trackFileName = dict_TrackIds2File[T_id]
    tracks = importTrackMateTracks(os.path.join(tracksDir, trackFileName))
    track = trackSelection(tracks, mode = 'longest')
    
    P_id = dict_TrackIds2PullId[T_id]
    dict_pull = Df2Dict(df_pulls[df_pulls['id'] == P_id])
    co_dict = list_co_dict[it]
    
    magnet_id = co_dict['magnet'][0]
    bead_type = co_dict['bead type'][0]
    pixel_size = df_pulls.loc[df_pulls['id']==P_id, 'film_pixel_size'].values[0]
    bead_radius = df_pulls.loc[df_pulls['id']==P_id, 'b_r'].values[0]
    
    case_mag = magnet_id + '_' + bead_type
    if calibFuncType == 'PowerLaw':
        parms_type = 'F_popt_pL'
        parms = MagnetDict[case_mag][parms_type]
        mag_d2f = (lambda x : powerLaw(x, *parms))
    elif calibFuncType == 'DoubleExpo':
        parms_type = 'F_popt_2exp'
        parms = MagnetDict[case_mag][parms_type]
        mag_d2f = (lambda x : doubleExpo(x, *parms))
    
    list_tracks.append(track)
    list_track_ids.append(T_id)
    list_dict_pull.append(dict_pull)
    list_mag_d2f.append(mag_d2f)
        
#### Run the function

pfTitle = '26-01-14_M1_C3_ctrl-vs-UV'

N_Rd, N_error = pullAnalyzer_compareTracks(list_tracks, list_track_ids, list_dict_pull, list_mag_d2f,
                             mode = 'newton', 
                             PLOT = PLOT, SHOW = SHOW, plotsDir = resultsDir, plotFileTitle = pfTitle)

J_Rd, J_error = pullAnalyzer_compareTracks(list_tracks, list_track_ids, list_dict_pull, list_mag_d2f,
                             mode = 'jeffrey', 
                             PLOT = PLOT, SHOW = SHOW, plotsDir = resultsDir, plotFileTitle = pfTitle)




















# %% ---------------

a = np.arange(100, 105)
for i, x in enumerate(a):
    print(i, x)


# %% 101. Tests & Legacy

# %%% Test of Jeffrey fitting

def jeffrey_model(params, x):
    k, gamma1, gamma2 = params
    return (1 - np.exp(-k*x/gamma1))/k + x/gamma2

tt = np.arange(0, 10, 0.25)
trueParms = [200, 100, 500]
true_xx = jeffrey_model(trueParms, tt)
xx = 0, true_xx * (1 + 0.1*(1 - 2*np.random.rand(len(tt)))) + 0.005*(1 - 2*np.random.rand(len(tt)))

fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.plot(tt, xx, marker='o', markersize=4, ls='', mew=0.75, fillstyle='none', mec='k')
ax.plot(tt, true_xx, 'c-')
plt.show()

start1 = [5, 100, 500]
obj1 = lambda params: np.linalg.norm(jeffrey_model(params, tt) - xx)
res1 = minimize(obj1, start1, method="Nelder-Mead", tol=1e-10,
                options={"maxfev": 500})
k, gamma1, gamma2 = res1.x
fitParms = [k, gamma1, gamma2]
fit_xx = jeffrey_model(fitParms, tt)

tau = gamma1/k
print(tau)
print(np.array(fitParms).astype(int))

ax.plot(tt, fit_xx, 'r-')

ax.grid()
plt.show()

# Ymeas = dx_pulling_n
# Yfit = jeffrey_model((k, gamma1, gamma2), tpulling)
# R2_p = get_R2(Ymeas, Yfit) 



# %%% Test of .xml parsing
filepath = os.path.join(directory, tracks_name)

tree = ET.parse(filepath)
root = tree.getroot()

# Extract spots into dictionary: {ID: (t, x, y)}

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
    
# for spot in root.findall(".//Spot"):
#     # ID = int(spot.attrib["ID"])
#     t = float(spot.attrib["FRAME"])
#     x = float(spot.attrib["POSITION_X"])
#     y = float(spot.attrib["POSITION_Y"])
#     spots_dict[ID] = (t, x, y)

# for track in root.findall(".//Track"):
#     edges = []
#     for edge in track.findall("Edge"):
#         src = int(edge.attrib["SPOT_SOURCE_ID"])
#         tgt = int(edge.attrib["SPOT_TARGET_ID"])
#         edges.append((src, tgt))

    # Collect unique spot IDs from this track
#     spot_ids = set([src for src, _ in edges] + [tgt for _, tgt in edges])
#     track_points = [spots_dict[sid] for sid in spot_ids if sid in spots_dict]
#     track_points = np.array(sorted(track_points, key=lambda p: p[0]))  # sort by time
#     tracks.append(track_points)

# md = {}  # could store metadata here if needed

# %%% Test of initial parameter guessing

N = 200
t = np.linspace(0, 30, num = N)
parms0 = [0.1, 10, 50]
dx1 = jeffrey(t, *parms0) 
dx1 = dx1 * (1 + 0.03*np.random.normal(loc=0, scale=2, size=N)) 
dx1 = dx1 + 0.02*np.random.normal(loc=0, scale=2, size=N)

fig, ax = plt.subplots(1, 1)
ax.plot(t, dx1, 'co', ms=2)

guess = guess_init_parms_jeffrey(t, dx1)

print(guess)


N = 200
t = np.linspace(0, 30, num = N)
parms1 = [0.7, 10]
dx2 = relax(t, *parms1) 
dx2 = dx2 * (1 + 0.01*np.random.normal(loc=0, scale=2, size=N)) 
dx2 = dx2 + 0.01*np.random.normal(loc=0, scale=2, size=N)

fig, ax = plt.subplots(1, 1)
ax.plot(t, dx2, 'co', ms=2)

guess = guess_init_parms_relax(t, dx2)

print(guess)



# %%% 101.a - V1 of function to analyze multiple files

# def pullAnalyzer_multiFiles(mainDir, date, prefix_id,
#                             analysisDir, tracksDir, resultsDir, plotsDir,
#                             mode = 'newton', resultsFileName = 'results',
#                             Redo = False, PLOT = True, SHOW = False):
#     if not os.path.exists(resultsDir):
#         os.makedirs(resultsDir)
#     if not os.path.exists(plotsDir):
#         os.makedirs(plotsDir)
#     df_manips = pd.read_csv(os.path.join(analysisDir, 'md_manips.csv'))
#     df_pulls = pd.read_csv(os.path.join(analysisDir, date + '_md_pulls.csv'))
#     listTrackFiles = [f for f in os.listdir(tracksDir) if ('Track' in f) and (f.endswith('.xml'))]
#     listTrackIds = ['_'.join(string.split('_')[:5]) for string in listTrackFiles]
#     dictTrackIds = {tid : tf for (tid, tf) in zip(listTrackIds, listTrackFiles)}
#     try:
#         df_res = pd.read_csv(os.path.join(resultsDir, resultsFileName + '.csv'))
#         already_analyzed = df_res['pull_id'].values
#     except:
#         df_res = pd.DataFrame({})
#         already_analyzed = []
    
#     prefix_id_manip = ('_').join(prefix_id.split('_')[:1 + ('_' in prefix_id)])
#     listManips = [m for m in df_manips['id'] if m.startswith(prefix_id_manip)]
    
#     id_cols = ['pull_id']
#     co_cols = ['type', 'solution', 'bead type', 'bead radius', 'treatment', 'magnet']
#     if mode == 'newton':
#         result_cols = ['fit_mode', 'viscosity', 'R2', 'median speed', 'median force']
#     elif mode == 'jeffrey':
#         result_cols = ['fit_mode', 'k', 'gamma1', 'gamma2', 'tempo1', 'tempo2', 'R2_p', 
#                        'median pull speed', 'median pull force',
#                        'a', 'tau', 'R2_r']
        
#     results_dict = {c:[] for c in id_cols}
#     results_dict.update({c:[] for c in result_cols})
#     results_dict.update({c:[] for c in co_cols})
    
#     for M_id in listManips:
#         co_dict = df_manips[df_manips['id'] == M_id].to_dict('list')
#         listPulls = [p for p in df_pulls['id'] if p.startswith(M_id) and (Redo or (p not in already_analyzed))]
#         for P_id in listPulls:
#             try:
#                 trackFileName = dictTrackIds[P_id]
#                 tracks = importTrackMateTracks(os.path.join(tracksDir, trackFileName))
#             except:
#                 pass
            
#             track = trackSelection(tracks, mode = 'longest')
            
#             output, error = pullAnalyzer(track, df_pulls, P_id,
#                                          mode = 'newton', 
#                                          PLOT = PLOT, SHOW = SHOW)
#             if not error:
#                 pixel_size = df_pulls.loc[df_pulls['id']==P_id, 'pixel_size'].values[0]
#                 bead_radius = pixel_size * df_pulls.loc[df_pulls['id']==P_id, 'bead_diameter'].values[0]/2
#                 results_dict['pull_id'].append(P_id)
#                 results_dict['type'].append(co_dict['type'][0])
#                 results_dict['solution'].append(co_dict['solution'][0])
#                 results_dict['bead type'].append(co_dict['bead type'][0])
#                 results_dict['bead radius'].append(bead_radius)
#                 results_dict['treatment'].append(co_dict['treatment'][0])
#                 results_dict['magnet'].append(co_dict['magnet'][0])
                
#                 if mode == 'newton':
#                     visco, R2, speed_med, force_med = output
#                     results_dict['fit_mode'].append('newton')
#                     results_dict['viscosity'].append(visco*1000)
#                     results_dict['R2'].append(R2)
#                     results_dict['median speed'].append(speed_med)
#                     results_dict['median force'].append(force_med)
                    
#                 elif mode == 'jeffrey':
#                     k, gamma1, gamma2, R2_p, speed_med, force_med, a, tau, R2_r = output
#                     results_dict['fit_mode'].append('jeffrey')
#                     results_dict['k'].append(visco*1000)
#                     results_dict['gamma1'].append(gamma1)
#                     results_dict['gamma2'].append(gamma2)
#                     results_dict['tempo1'].append(gamma1/k)
#                     results_dict['tempo2'].append(gamma2/k)
#                     results_dict['R2_p'].append(R2_p)
#                     results_dict['median pull speed'].append(speed_med)
#                     results_dict['median pull force'].append(force_med)
#                     results_dict['a'].append(a)
#                     results_dict['tau'].append(tau)
#                     results_dict['R2_r'].append(R2_r)
                
#     new_df_res = pd.DataFrame(results_dict)
#     df_res = pd.concat([df_res,new_df_res]).drop_duplicates(subset='pull_id', keep='last').reset_index(drop=True)
    
#     df_res.to_csv(os.path.join(resultsDir, resultsFileName + '.csv'), index=False)
#     return(df_res)
                
            
# def pullAnalyzer(track, df_pulls, pull_id,
#                  mode = 'newton',
#                  PLOT = True, SHOW = False):
#     if SHOW:
#         plt.ion()
#     else:
#         plt.ioff()
        
#     error = False
        
#     #### 1. Paths
#     # analysisDir = os.path.join(mainDir, 'Analysis')
#     # df_pulls = pd.read_csv(os.path.join(mainDir, 'md_pulls.csv'))
#     # tracks_name = pull_id + '_Tracks.xml'
#     # output_folder = pull_id + '_Results'
#     # if not os.path.exists(os.path.join(analysisDir, output_folder)):
#     #     os.makedirs(os.path.join(analysisDir, output_folder))

#     #### 2. Parameters
#     frame_initPull = df_pulls.loc[df_pulls['id'] == pull_id, 'frame_initPull'].values[0]
#     frame_endPull = df_pulls.loc[df_pulls['id'] == pull_id, 'frame_endPull'].values[0]
#     magnet_x = df_pulls.loc[df_pulls['id'] == pull_id, 'magnet_x'].values[0]
#     magnet_y = df_pulls.loc[df_pulls['id'] == pull_id, 'magnet_y'].values[0]
#     pixel_size = df_pulls.loc[df_pulls['id'] == pull_id, 'pixel_size'].values[0]  # µm
#     time_stp = df_pulls.loc[df_pulls['id'] == pull_id, 'time_stp'].values[0]      # s
#     magnet_radius = pixel_size * df_pulls.loc[df_pulls['id'] == pull_id, 'magnet_diameter'].values[0] / 2  # µm
#     bead_radius = df_pulls.loc[df_pulls['id'] == pull_id, 'bead_diameter'].values[0] * pixel_size / 2 # µm
    
#     # Viscosity of glycerol 80% v/v glycerol/water at 21°C [Pa.s]
#     viscosity_glycerol = 0.0857  
#     # Magnet function distance (µm) to velocity (µm/s) [expected velocity in glycerol]
#     mag_d2v = lambda x: 80.23*np.exp(-x/47.49) + 1.03*np.exp(-x/22740.0)
#     # Speed at 200 mu as um/s
#     v_interp = mag_d2v(200)
#     # Aggregate force coefficient (c) in f=cR^3
#     force_coeff = 0.3663 
#     # Magnet function distance (µm) to force (pN)
#     mag_d2f = lambda x: force_coeff*(bead_radius**3)*mag_d2v(x)/v_interp # If beads are agarose
#     # mag_d2f = lambda x: 6*np.pi*viscosity_glycerol*mag_d2v(x)*bead_radius # Function for MyOnes (calib beads)
                          
    
#     #### 3. Load data
#     # try:
#     #     tracks = importTrackMateTracks(os.path.join(analysisDir, tracks_name))
#     # except:
#     #     error, output = True, ()
#     #     return(output, error)
#     # track = trackSelection(tracks, mode = 'longest')
    
#     X, Y, t = track[:,1], track[:,2], track[:,0]
    
#     if mode == 'newton': 
#     # Tackle the case where a bead comes in the field / disapear within the magnet frames
#     # NB : this can be relevant only in the purely newtonian viscous case 
#     # (constant viscosity as single parm)
#         if min(t) > (frame_initPull - 1):
#             initPullTime = 0
#         else:
#             initPullTime = np.where(t == (frame_initPull - 1))[0][0]
#         if max(t) <= (frame_endPull - 1):
#             finPullTime = len(t)-1
#         else:
#             finPullTime = np.where(t == (frame_endPull - 1))[0][0]
            
#     else:
#         initPullTime = np.where(t == (frame_initPull - 1))[0][0]
#         finPullTime = np.where(t == (frame_endPull - 1))[0][0]
    
#     #### 4. Pretreatment
#     # --- Rotate track ---
#     theta = np.arctan2(Y[initPullTime] - Y[finPullTime],
#                        X[initPullTime] - X[finPullTime])
#     rotation_mat = np.array([[np.cos(-theta), -np.sin(-theta)],
#                              [np.sin(-theta),  np.cos(-theta)]])
#     coords = np.vstack((X, Y)).T @ rotation_mat.T
#     x_rot, y_rot = coords[:,0], coords[:,1]

#     x_shift = (-x_rot[initPullTime:] + np.max(x_rot[initPullTime:])) * pixel_size

#     # --- Pulling phase ---
#     pull_index = np.arange(initPullTime, finPullTime+1)
#     pull_length = len(pull_index)
#     xp, yp = x_rot[pull_index], y_rot[pull_index] # for plots

#     d = np.stack([(magnet_x - X[pull_index]) * pixel_size,
#                   (magnet_y - Y[pull_index]) * pixel_size], axis=1)
    
#     #### Attempt at correcting the effective attractive point
#     # dist = np.linalg.norm(d, axis=1) - magnet_radius # Original behaviour
#     dist = np.linalg.norm(d, axis=1) + (magnet_radius*pixel_size) 
#     pull_force = mag_d2f(dist)
    
#     tpulling = (t[pull_index] - t[initPullTime]) * time_stp
#     dx_pulling = x_shift[pull_index - initPullTime]
#     dx_pulling_n = dx_pulling / pull_force
    
#     #### Filters
#     Filter1 = (dist > 500) # bead further than 550 µm from magnet center
#     Filter2 = (tpulling >= 2) # remove the first 2 seconds of filming
#     GlobalFilter = Filter1 & Filter2
#     # if np.sum(GlobalFilter) == 0:
#     #     error = True
#     #     output = ()

#     # --- Release phase ---
#     try:
#         release_index = np.arange(finPullTime, len(track))
#         release_length = len(release_index)
    
#         trelease = (t[release_index] - t[finPullTime+1]) * time_stp
#         dx_release = x_shift[release_index - initPullTime]
#         dx_release_n = dx_release / x_shift[pull_length]
#     except:
#         pass
    
#     # --- Measures ---
#     speed_med = np.median((dx_pulling[GlobalFilter][1:]-dx_pulling[GlobalFilter][:-1])/time_stp) # µm/s
#     force_med = np.median(pull_force[GlobalFilter]) # pN
    
#     #### 5. Fit Model
#     if mode == 'newton':
#         try:
#             params, results = fitLine(tpulling[GlobalFilter], dx_pulling_n[GlobalFilter])
#             # params, results = fitLineHuber(tpulling, dx_pulling_n)
#             gamma = params[1]
#             visco = 1/(6*np.pi*bead_radius*gamma)
#             R2 = results.rsquared
#             output = (visco, R2, speed_med, force_med)
#         except:
#             error = True
#             output = ()
#             return(output, error)
        
#     elif mode == 'jeffrey':
#         try:
#             # Fit options
#             # Code a way to input these
#             start1 = [2, 50, 900]   # initial [k, gamma1, gamma2]
#             start2 = [0.9, 25] 
            
#             #### Fit pulling phase
#             def jeffrey_model(params, x):
#                 k, gamma1, gamma2 = params
#                 return (1 - np.exp(-k*x/gamma1))/k + x/gamma2
    
#             obj1 = lambda params: np.linalg.norm(jeffrey_model(params, tpulling) - dx_pulling_n)
#             res1 = minimize(obj1, start1, method="Nelder-Mead", tol=1e-10,
#                             options={"maxfev": 500})
#             k, gamma1, gamma2 = res1.x
#             # tempo1, tempo2 = gamma1/k, gamma2/k
            
#             Ymeas = dx_pulling_n
#             Yfit = jeffrey_model((k, gamma1, gamma2), tpulling)
#             R2_p = get_R2(Ymeas, Yfit) 
    
#             #### Fit release phase
#             def exp_fit(params, x):
#                 a, tau = params
#                 return (1-a)*np.exp(-x/tau) + a
    
#             obj2 = lambda params: np.linalg.norm(exp_fit(params, trelease) - dx_release_n)
#             res2 = minimize(obj2, start2, method="Nelder-Mead", tol=1e-7,
#                             options = {"maxfev": 1000})
#             a, tau = res2.x
            
#             Ymeas = dx_release_n
#             Yfit = exp_fit((a, tau), trelease)
#             R2_r = get_R2(Ymeas, Yfit) 
            
#             output = (k, gamma1, gamma2, R2_p, speed_med, force_med, a, tau, R2_r)
    
#         except:
#             error = True
#             output = ()
#             return(output, error)
    
#     #### 6. Figures
#     if mode == 'newton' and PLOT:
#         fig1, axes1 = plt.subplots(1, 2, figsize = (10,5))
#         ax = axes1[0]
#         ax.plot(X, Y, ".-")
#         ax.axis("equal")
#         ax.grid(axis='both')
#         ax.set_title("Original track")
    
#         ax = axes1[1]
#         ax.plot(x_rot, y_rot, ".-")
#         ax.plot(xp, yp, "k.-", linewidth=2)
#         ax.axis("equal")
#         ax.grid(axis='both')
#         ax.set_title("Rotated track")
#         fig1.tight_layout()
#         fig1.savefig(os.path.join(plotsDir, pull_id + "_trajectories.png"))
    
#         fig2, axes2 = plt.subplots(1, 1, figsize = (5,5))
#         ax = axes2
#         axbis = axes2.twinx()
#         axbis.plot(tpulling, dist, 'g--', zorder=4)
#         axbis.set_ylabel('Distance from magnet center [µm]')
#         ax.plot(tpulling[GlobalFilter], dx_pulling_n[GlobalFilter], 'o', color='darkturquoise', markersize=5, zorder=5)
#         ax.plot(tpulling[~GlobalFilter], dx_pulling_n[~GlobalFilter], 'o', color='lightblue', markersize=5, zorder=4)
#         xfit = np.linspace(0, tpulling[-1], 100)
#         yfit = params[0] + params[1]*xfit
#         ax.plot(xfit, yfit, "r-", label=r'$\eta$ = ' + f'{visco*1000:.2f} mPa.s', zorder=6)
#         ax.set_xlabel("t [s]")
#         ax.set_ylabel("dx/f [µm/pN]")
#         ax.grid(axis='both')
#         ax.legend(fontsize = 11).set_zorder(6)
#         fig2.suptitle(pull_id, fontsize=12)
        
#         fig2.tight_layout()
#         fig2.savefig(os.path.join(plotsDir, pull_id + "_fits.png"))
        
#         if SHOW:
#             plt.show()
            
#         plt.ion()
    
#     elif mode == 'jeffrey' and PLOT:
#         fig1, axes1 = plt.subplots(1, 2, figsize=(8,5))
#         ax = axes1[0]
#         ax.plot(X, Y, ".-")
#         ax.axis("equal")
#         ax.set_title("Original track")
#         ax.grid()
    
#         ax = axes1[1]
#         ax.plot(x_rot, y_rot, ".-")
#         ax.plot(xp, yp, "k.-", linewidth=2)
#         ax.axis("equal")
#         ax.set_title("Rotated track")
#         ax.grid()
#         plt.tight_layout()
#         # plt.savefig("trajectories.jpg")
#         fig1.savefig(os.path.join(analysisDir, pull_id + "_trajectories.png"))
    
#         fig2, axes2 = plt.subplots(1, 2, figsize=(8,5))
#         ax = axes2[0]
#         label1 = "Fitting Jeffrey's model\n" + r"$\frac{1}{k}(1 - exp(-k.x/\gamma_1)) + x/\gamma_2$" + "\n"
#         label1+= r"$k$ = " + f"{k:.2f}\n"
#         label1+= r"$\gamma_1$ = " + f"{gamma1:.2f}\n"
#         label1+= r"$\gamma_2$ = " + f"{gamma2:.2f}"
#         ax.plot(tpulling, dx_pulling_n, "s")
#         ax.plot(np.linspace(0, tpulling[-1], 1000),
#                  jeffrey_model([k,gamma1,gamma2], np.linspace(0, tpulling[-1], 1000)), "r-",
#                  label=label1)
#         ax.legend()
#         ax.grid()
#         ax.set_xlabel("t [s]"); plt.ylabel("dx/f [µm/pN]")
    
#         ax = axes2[1]
#         label2 = "Fitting Viscoel Relax\n" + r"$(1-a). exp(-x/\tau ) + a$" + "\n"
#         label2+= r"$a$ = " + f"{a:.2f}\n"
#         label2+= r"$\tau$ = " + f"{tau:.2f}"
#         ax.plot(trelease, dx_release_n, "s")
#         ax.plot(np.linspace(0, trelease[-1], 1000), exp_fit([a,tau], np.linspace(0, trelease[-1], 1000)), 
#                 ls="-", c='darkorange', label=label2)
#         ax.legend()
#         ax.grid()
#         ax.set_xlabel("t [s]")
#         ax.set_ylabel("Normalized displacement")
#         ax.set_ylim([0, 1.5])
#         plt.tight_layout()
#         fig2.savefig(os.path.join(analysisDir, pull_id + "_fits.png"))
        
#         if SHOW:
#             plt.show()
            
#         plt.ion()
        
#     #### 7. Output
#     return(output, error


# %%% 101.b - V2 of function to analyze multiple files

def pullAnalyzer_multiFiles(mainDir, date, prefix_id,
                            analysisDir, tracksDir, resultsDir, plotsDir,
                            mode = 'newton', resultsFileName = 'results',
                            Redo = False, PLOT = True, SHOW = False):
    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)
    if not os.path.exists(plotsDir):
        os.makedirs(plotsDir)
    df_manips = pd.read_csv(os.path.join(analysisDir, 'md_manips.csv'))
    df_pulls = pd.read_csv(os.path.join(analysisDir, date + '_md_pulls.csv'))
    listTrackFiles = [f for f in os.listdir(tracksDir) if ('Track' in f) and (f.endswith('.xml'))]
    listTrackIds = ['_'.join(string.split('_')[:5]) for string in listTrackFiles]
    dictTrackIds = {tid : tf for (tid, tf) in zip(listTrackIds, listTrackFiles)}
    try:
        df_res = pd.read_csv(os.path.join(resultsDir, resultsFileName + '.csv'))
        already_analyzed = df_res['pull_id'].values
    except:
        df_res = pd.DataFrame({})
        already_analyzed = []
    
    prefix_id_manip = ('_').join(prefix_id.split('_')[:1 + ('_' in prefix_id)])
    listManips = [m for m in df_manips['id'] if m.startswith(prefix_id_manip)]
    
    id_cols = ['pull_id']
    co_cols = ['type', 'solution', 'bead type', 'bead radius', 'treatment', 'magnet']
    if mode == 'newton':
        result_cols = ['fit_mode', 'viscosity', 'R2', 
                       'median speed', 'median force', 
                       'R min', 'R max', 'theta']
        all_XY = []
    elif mode == 'jeffrey':
        result_cols = ['fit_mode', 'k', 'gamma1', 'gamma2', 'tempo1', 'tempo2', 'R2_p', 
                       'median pull speed', 'median pull force',
                       'a', 'tau', 'R2_r']
        # TODO!
        
    results_dict = {c:[] for c in id_cols}
    results_dict.update({c:[] for c in result_cols})
    results_dict.update({c:[] for c in co_cols})
    
    for M_id in listManips:
        co_dict = df_manips[df_manips['id'] == M_id].to_dict('list')
        listPulls = [p for p in df_pulls['id'] if p.startswith(M_id) and (Redo or (p not in already_analyzed))]
        for P_id in listPulls:
            try:
                trackFileName = dictTrackIds[P_id]
                tracks = importTrackMateTracks(os.path.join(tracksDir, trackFileName))
            except:
                pass
            
            track = trackSelection(tracks, mode = 'longest')
            dict_pull = Df2Dict(df_pulls[df_pulls['id'] == P_id])
            
            output, error = pullAnalyzer(track, dict_pull, 
                                         mode = 'newton', 
                                         PLOT = PLOT, SHOW = SHOW, plotsDir = plotsDir)
            if not error:
                pixel_size = df_pulls.loc[df_pulls['id']==P_id, 'pixel_size'].values[0]
                bead_radius = pixel_size * df_pulls.loc[df_pulls['id']==P_id, 'bead_diameter'].values[0]/2
                results_dict['pull_id'].append(P_id)
                results_dict['type'].append(co_dict['type'][0])
                results_dict['solution'].append(co_dict['solution'][0])
                results_dict['bead type'].append(co_dict['bead type'][0])
                results_dict['bead radius'].append(bead_radius)
                results_dict['treatment'].append(co_dict['treatment'][0])
                results_dict['magnet'].append(co_dict['magnet'][0])
                
                if mode == 'newton':
                    visco, R2, speed_med, force_med, theta, r_min, r_max, XY = output
                    results_dict['fit_mode'].append('newton')
                    results_dict['viscosity'].append(visco*1000)
                    results_dict['R2'].append(R2)
                    results_dict['median speed'].append(speed_med)
                    results_dict['median force'].append(force_med)
                    results_dict['theta'].append(theta)
                    results_dict['R min'].append(r_min)
                    results_dict['R max'].append(r_max)
                    all_XY.append(XY)
                    
                elif mode == 'jeffrey':
                    k, gamma1, gamma2, R2_p, speed_med, force_med, a, tau, R2_r, theta, r_min, r_max, XY = output
                    results_dict['fit_mode'].append('jeffrey')
                    results_dict['k'].append(k)
                    results_dict['gamma1'].append(gamma1)
                    results_dict['gamma2'].append(gamma2)
                    results_dict['tempo1'].append(gamma1/k)
                    results_dict['tempo2'].append(gamma2/k)
                    results_dict['R2_p'].append(R2_p)
                    results_dict['median pull speed'].append(speed_med)
                    results_dict['median pull force'].append(force_med)
                    results_dict['theta'].append(theta)
                    results_dict['R min'].append(r_min)
                    results_dict['R max'].append(r_max)
                    results_dict['a'].append(a)
                    results_dict['tau'].append(tau)
                    results_dict['R2_r'].append(R2_r)
                
    new_df_res = pd.DataFrame(results_dict)
    df_res = pd.concat([df_res,new_df_res]).drop_duplicates(subset='pull_id', keep='last').reset_index(drop=True)
    
    if mode == 'newton':
        selected_XY = deepcopy(all_XY)
        # selected_XY = [xy for xy in all_XY if np.abs(np.median(xy[:,1])) > 75]
        # selected_XY = [xy for xy in selected_XY if len(xy) > 75]
        fits_XY = []
        for xy in selected_XY:
            params, results = fitLineHuber(xy[:,0], xy[:,1])
            a, b = params[1], params[0]
            x0 = -b/a
            fits_XY.append([a, b, x0])
        fits_XY = np.array(fits_XY)
        
        fig, ax = plt.subplots(1, 1, figsize = (8, 8))
        ax.axis('equal')
        mag_radius = 335*0.451/2
        circle1 = plt.Circle((-mag_radius, 0), mag_radius, color='grey')
        ax.add_patch(circle1)
        for xy, p in zip(selected_XY, fits_XY):
            ax.plot(xy[:,0], xy[:,1], 'k-', lw=2)
            a, b, x0 = p
            xfit = np.arange(-250, 600)
            yfit = a*xfit + b
            ax.plot(xfit, yfit, ls='-', color = 'red', lw=1, zorder=5)
            ax.plot(x0, 0, 'c^', mec = 'dimgrey', lw=0.5, zorder=6)
        ax.plot(np.median(fits_XY[:,2]), 0, 'co', mec = 'k', lw=0.5, zorder=7)
        ax.plot(np.mean(fits_XY[:,2]), 0, 'cs', mec = 'k', lw=0.5, zorder=7)
        ax.grid()
        plt.show()
        
    df_res.to_csv(os.path.join(resultsDir, resultsFileName + '.csv'), index=False)
    return(df_res, all_XY)




def pullAnalyzer(track, dict_pull,
                 mode = 'newton', fit_V = 'DoubleExpo',
                 PLOT = True, SHOW = False, plotsDir = ''):
    if SHOW:
        plt.ion()
    else:
        plt.ioff()
        
    error = False
        
    #### 1. Paths
    # analysisDir = os.path.join(mainDir, 'Analysis')
    # df_pulls = pd.read_csv(os.path.join(mainDir, 'md_pulls.csv'))
    # tracks_name = pull_id + '_Tracks.xml'
    # output_folder = pull_id + '_Results'
    # if not os.path.exists(os.path.join(analysisDir, output_folder)):
    #     os.makedirs(os.path.join(analysisDir, output_folder))

    #### 2. Parameters
    pull_id = dict_pull['id']
    frame_initPull = dict_pull['frame_initPull']
    frame_endPull = dict_pull['frame_endPull']
    pixel_size = dict_pull['pixel_size']  # µm
    magnet_x = dict_pull['magnet_x']*pixel_size
    magnet_y = dict_pull['magnet_y']*pixel_size
    time_stp = dict_pull['time_stp']      # s
    magnet_radius = dict_pull['magnet_diameter']*pixel_size / 2  # µm
    bead_radius = dict_pull['bead_diameter'] * pixel_size / 2 # µm
    X, Y, t = track[:,1] * pixel_size, track[:,2] * pixel_size, track[:,0]
    
    # Viscosity of glycerol 80% v/v glycerol/water at 21°C [Pa.s]
    viscosity_glycerol = 0.0857
    
    # Viscosity of glycerol 75% v/v glycerol/water at 20°C [Pa.s]
    viscosity_glycerol = 0.055270
    
    # Magnet function distance (µm) to velocity (µm/s) [expected velocity in glycerol]
    # parms_2exp = [80.23, 47.49, 1.03, 22740.0] # Maribel Calib
    # parms_2exp = [146, 107, 0.93, 6.38e7] # Maribel Calib 2
    # parms_2exp = [1138.18, 37.2887, 2.01482, 296.243] # Joseph Calib
    # parms_2exp = [255.07,  45.81,   1.42, 401.66] # Joseph Calib Better
    parms_2exp = [2683, 30.33, 2.440, 314.54] # Joseph Calib Good
    mag_d2v_2exp = lambda x: doubleExpo(x, *parms_2exp)
    
    parms_langevin = [2.75379e+10, 6.87603e+21, -124.896]
    # parms_langevin = [3.56905e+11, 3.76308e+25, -140]
    # parms_langevin = [1.7435e+10, 4.65859e+19, -100] 
    # parms_langevin = [6.46079e+10, 1.09389e+21, 0]
    
    parms_powerLaw = [310320, -2.19732]
    mag_d2v_powerLaw = lambda x: powerLaw(x, *parms_powerLaw)
    
    if fit_V == 'DoubleExpo': # Original behavior
        mag_d2v = mag_d2v_2exp
    elif fit_V == 'PowerLaw': # Test
        mag_d2v = mag_d2v_powerLaw
        
    # Speed at 200 mu as um/s
    v_interp = mag_d2v(200)
    # Aggregate force coefficient (c) in f=cR^3
    force_coeff = 0.3663 
    # Magnet function distance (µm) to force (pN)
    mag_d2f = lambda x: force_coeff*(bead_radius**3)*mag_d2v(x)/v_interp # If beads are agarose
    # mag_d2f = lambda x: 6*np.pi*viscosity_glycerol*mag_d2v(x)*bead_radius # Function for MyOnes (calib beads)
                          
    
    #### 3. Load data
    # try:
    #     tracks = importTrackMateTracks(os.path.join(analysisDir, tracks_name))
    # except:
    #     error, output = True, ()
    #     return(output, error)
    # track = trackSelection(tracks, mode = 'longest')
        
    if mode == 'newton': 
    # Tackle the case where a bead comes in the field / disapear within the magnet frames
    # NB : this can be relevant only in the purely newtonian viscous case 
    # (constant viscosity as single parm)
        if min(t) > (frame_initPull - 1):
            initPullTime = 0
        else:
            initPullTime = np.where(t == (frame_initPull - 1))[0][0]
        if max(t) <= (frame_endPull - 1):
            finPullTime = len(t)-1
        else:
            finPullTime = np.where(t == (frame_endPull - 1))[0][0]
            
    else:
        initPullTime = np.where(t == (frame_initPull - 1))[0][0]
        finPullTime = np.where(t == (frame_endPull - 1))[0][0]
    
    #### 4. Pretreatment
    # --- Rotate track ---
    theta = np.arctan2(Y[initPullTime] - Y[finPullTime],
                       X[initPullTime] - X[finPullTime])
    rotation_mat = np.array([[np.cos(-theta), -np.sin(-theta)],
                             [np.sin(-theta),  np.cos(-theta)]])
    coords = np.vstack((X, Y)).T @ rotation_mat.T
    x_rot, y_rot = coords[:,0], coords[:,1]

    x_shift = (-x_rot[initPullTime:] + np.max(x_rot[initPullTime:]))

    # --- Pulling phase ---
    pull_index = np.arange(initPullTime, finPullTime+1)
    pull_length = len(pull_index)
    xp, yp = x_rot[pull_index], y_rot[pull_index] # for plots

    d = np.stack([(magnet_x - X[pull_index]),
                  (magnet_y - Y[pull_index])], axis=1)
    XY = np.copy(d)
    XY = XY * (-1)
    XY[:,0] -=  magnet_radius
    
    #### Definition of the distance and the force
    dist = np.linalg.norm(d, axis=1) - (magnet_radius) # Original behaviour
    # dist = np.linalg.norm(d, axis=1) + (magnet_radius) # Test
    pull_force = mag_d2f(dist)
    
    tpulling = (t[pull_index] - t[initPullTime]) * time_stp
    dx_pulling = x_shift[pull_index - initPullTime]
    dx_pulling_n = dx_pulling / pull_force
    
    #### Filters
    Filter1 = tpulling > 0.5
    Filter2 = (np.abs(dx_pulling[0] - dx_pulling[:]) <= 60) # keep only first 60 µm of movement
    GlobalFilter = Filter1 & Filter2
    if np.sum(GlobalFilter) == 0:
        error = True
        output = ()
    
    
    # --- Release phase ---
    try:
        release_index = np.arange(finPullTime, len(track))
        release_length = len(release_index)
    
        trelease = (t[release_index] - t[finPullTime+1]) * time_stp
        dx_release = x_shift[release_index - initPullTime]
        dx_release_n = dx_release / x_shift[pull_length]
    except:
        pass
    
    # --- Measures ---
    speed_med = np.median((dx_pulling[GlobalFilter][1:]-dx_pulling[GlobalFilter][:-1])/time_stp) # µm/s
    force_med = np.median(pull_force[GlobalFilter]) # pN
    r_min, r_max = min(dist), max(dist)
    
    #### 5. Fit Model
    if mode == 'newton' and np.sum(GlobalFilter) > 0:
        # try:
        params, results = fitLine(tpulling[GlobalFilter], dx_pulling_n[GlobalFilter])
        # params, results = fitLineHuber(tpulling, dx_pulling_n)
        gamma = params[1]
        visco = 1/(6*np.pi*bead_radius*gamma)
        R2 = results.rsquared
        output = (visco, R2, speed_med, force_med, theta, r_min, r_max, XY)
        # except:
        #     error = True
        #     output = ()
        #     return(output, error)
        
    elif mode == 'jeffrey':
        try:
            # Fit options
            #### TODO!
            # Code a way to input these
            start1 = [2, 50, 900]   # initial [k, gamma1, gamma2]
            start2 = [0.9, 25] 
            
            #### Fit pulling phase
            def jeffrey_model(params, x):
                k, gamma1, gamma2 = params
                return (1 - np.exp(-k*x/gamma1))/k + x/gamma2
    
            obj1 = lambda params: np.linalg.norm(jeffrey_model(params, tpulling) - dx_pulling_n)
            res1 = minimize(obj1, start1, method="Nelder-Mead", tol=1e-10,
                            options={"maxfev": 500})
            k, gamma1, gamma2 = res1.x
            # tempo1, tempo2 = gamma1/k, gamma2/k
            
            Ymeas = dx_pulling_n
            Yfit = jeffrey_model((k, gamma1, gamma2), tpulling)
            R2_p = get_R2(Ymeas, Yfit) 
    
            #### Fit release phase
            def exp_fit(params, x):
                a, tau = params
                return (1-a)*np.exp(-x/tau) + a
    
            obj2 = lambda params: np.linalg.norm(exp_fit(params, trelease) - dx_release_n)
            res2 = minimize(obj2, start2, method="Nelder-Mead", tol=1e-7,
                            options = {"maxfev": 1000})
            a, tau = res2.x
            
            Ymeas = dx_release_n
            Yfit = exp_fit((a, tau), trelease)
            R2_r = get_R2(Ymeas, Yfit) 
            
            output = (k, gamma1, gamma2, R2_p, speed_med, force_med, a, tau, R2_r, theta, r_min, r_max, XY)
    
        except:
            error = True
            output = ()
            return(output, error)
    
    #### 6. Figures
    if mode == 'newton' and PLOT:
        fig1, axes1 = plt.subplots(1, 2, figsize = (10,5))
        ax = axes1[0]
        ax.plot(X, Y, ".-")
        ax.axis("equal")
        ax.grid(axis='both')
        ax.set_title("Original track")
    
        ax = axes1[1]
        ax.plot(x_rot, y_rot, ".-")
        ax.plot(xp, yp, "k.-", linewidth=2)
        ax.axis("equal")
        ax.grid(axis='both')
        ax.set_title("Rotated track")
        fig1.tight_layout()
        
        fig1.savefig(os.path.join(plotsDir, pull_id + "_trajectories.png"))
    
    
        fig2, axes2 = plt.subplots(1, 1, figsize = (5,5))
        ax = axes2
        axbis = axes2.twinx()
        axbis.plot(tpulling, dist, 'g--', zorder=4)
        axbis.set_ylabel('Distance from magnet center [µm]')
        ax.plot(tpulling[GlobalFilter], dx_pulling_n[GlobalFilter], 'o', color='darkturquoise', markersize=5, zorder=5)
        ax.plot(tpulling[~GlobalFilter], dx_pulling_n[~GlobalFilter], 'o', color='lightblue', markersize=5, zorder=4)
        xfit = np.linspace(0, tpulling[-1], 100)
        yfit = params[0] + params[1]*xfit
        ax.plot(xfit, yfit, "r-", label=r'$\eta$ = ' + f'{visco*1000:.2f} mPa.s', zorder=6)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("dx/f [µm/pN]")
        ax.grid(axis='both')
        ax.legend(fontsize = 11).set_zorder(6)
        fig2.suptitle(pull_id, fontsize=12)
        fig2.tight_layout()
        
        fig2.savefig(os.path.join(plotsDir, pull_id + "_fits.png"))
        
        # i_select = np.abs(dist - dist[0]) < 50
        
        # fig3, axes3 = plt.subplots(1, 1, figsize = (5,5))
        # ax = axes3
        # axbis = axes3.twinx()
        # dist_spline = make_splrep(tpulling, dist, s=len(t)/2)
        # ax.plot(tpulling, dist, 'go', ms = 3, zorder=4)
        # ax.plot(tpulling, dist_spline(tpulling), 'k-', zorder=5)
        # ax.set_xlabel("t [s]")
        # ax.set_ylabel("X [µm]")
        # axbis.plot(tpulling, pull_force, 'r--', zorder=5)
        # axbis.set_ylabel('Pulling force [pN]')
        # ax.grid(axis='both')
        # ax.legend(fontsize = 11).set_zorder(6)
        # fig3.suptitle(pull_id, fontsize=12)
        # fig3.tight_layout()
        
        # fig3.savefig(os.path.join(plotsDir, pull_id + "_fits.png"))
        
        # fig4, axes4 = plt.subplots(1, 1, figsize = (5,5))
        # ax = axes4
        
        # dist_spline = make_splrep(tpulling, dist, s=len(t)/2)
        # speed_spline = dist_spline.derivative(nu=1)
        # speed = np.abs(speed_spline(tpulling))
        # parms, results = fitLine(speed[i_select], pull_force[i_select])
        # a, b = parms[1], parms[0]
        # eta2 = a/(6*np.pi*bead_radius)
        # xfit = np.linspace(0, max(speed), 100)
        # yfit = b + a*xfit
        
        # ax.plot(speed, pull_force, 'go', ms = 3, zorder=4)
        # ax.plot(speed[i_select], pull_force[i_select], 'co', ms = 3, zorder=4)
        # ax.plot(xfit, yfit, 'r--', label=r'$\eta$ = ' + f'{eta2*1000:.2f} mPa.s', zorder=5)
        # ax.set_xlabel("V [µm/s]")
        # ax.set_ylabel("F [pN]")
        # ax.grid(axis='both')
        # ax.legend(fontsize = 11).set_zorder(6)
        # fig4.suptitle(pull_id, fontsize=12)
        # fig4.tight_layout()
        
        # fig5, axes5 = plt.subplots(1, 1, figsize = (5,5))
        # ax = axes5
        # axbis = axes5.twinx()
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        # axbis.set_yscale('log')
        # dist_spline = make_splrep(tpulling, dist, s=len(t)/2)
        # speed_spline = dist_spline.derivative(nu=1)
        # speed = np.abs(speed_spline(tpulling))
        # # ax.plot(tpulling, dist, 'go', ms = 3, zorder=4)
        # ax.plot(dist, speed, 'k-', zorder=5)
        # ax.set_xlabel("X [µm]")
        # ax.set_ylabel("V [µm/s]")
        # axbis.plot(dist, pull_force, 'r-', zorder=5)
        # ax.plot(dist, (dist/dist[0])**(-7), 'b-', zorder=5)
        # ax.plot(dist, (dist/dist[0])**(-4), 'g-', zorder=5)
        # ax.plot(dist, (dist/dist[0])**(-2), 'c-', zorder=5)
        # axbis.set_ylabel('Pulling force [pN]')
        # ax.grid(axis='both')
        # ax.legend(fontsize = 11).set_zorder(6)
        # fig5.suptitle(pull_id, fontsize=12)
        # fig5.tight_layout()
        
        fig6, axes6 = plt.subplots(1, 1, figsize = (5,5))
        fig, ax = fig6, axes6
        ax.set_xscale('log')
        ax.set_yscale('log')
        dist_spline = make_splrep(tpulling, dist, s=len(t)/2)
        speed_spline = dist_spline.derivative(nu=1)
        speed = np.abs(speed_spline(tpulling))
        parms, raw_parms, results = fitPower(dist, speed)
        A, k = parms
        d_plot = np.linspace(min(dist), max(dist), 100)
        v_plot = A*(d_plot**k)
        ax.plot(dist, speed, 'wo', mec='k', ms = 4, zorder=5)
        ax.plot(d_plot, v_plot, 'r-', zorder=6, label=f'k = {k:.2f}')
        ax.set_xlabel("X [µm]")
        ax.set_ylabel("V [µm/s]")
        ax.set_xlim([100, 1000])
        ax.set_ylim([0.1, 15])
        ax.grid(axis='both', which='both')
        ax.legend(fontsize = 11).set_zorder(6)
        fig.suptitle(pull_id, fontsize=12)
        fig.tight_layout()
        
        if SHOW:
            plt.show()
            
        plt.ion()
    
    elif mode == 'jeffrey' and PLOT:
        fig1, axes1 = plt.subplots(1, 2, figsize=(8,5))
        ax = axes1[0]
        ax.plot(X, Y, ".-")
        ax.axis("equal")
        ax.set_title("Original track")
        ax.grid()
    
        ax = axes1[1]
        ax.plot(x_rot, y_rot, ".-")
        ax.plot(xp, yp, "k.-", linewidth=2)
        ax.axis("equal")
        ax.set_title("Rotated track")
        ax.grid()
        plt.tight_layout()
        # plt.savefig("trajectories.jpg")
        fig1.savefig(os.path.join(analysisDir, pull_id + "_trajectories.png"))
    
        fig2, axes2 = plt.subplots(1, 2, figsize=(8,5))
        ax = axes2[0]
        label1 = "Fitting Jeffrey's model\n" + r"$\frac{1}{k}(1 - exp(-k.x/\gamma_1)) + x/\gamma_2$" + "\n"
        label1+= r"$k$ = " + f"{k:.2f}\n"
        label1+= r"$\gamma_1$ = " + f"{gamma1:.2f}\n"
        label1+= r"$\gamma_2$ = " + f"{gamma2:.2f}"
        ax.plot(tpulling, dx_pulling_n, "s")
        ax.plot(np.linspace(0, tpulling[-1], 1000),
                 jeffrey_model([k,gamma1,gamma2], np.linspace(0, tpulling[-1], 1000)), "r-",
                 label=label1)
        ax.legend()
        ax.grid()
        ax.set_xlabel("t [s]"); plt.ylabel("dx/f [µm/pN]")
    
        ax = axes2[1]
        label2 = "Fitting Viscoel Relax\n" + r"$(1-a). exp(-x/\tau ) + a$" + "\n"
        label2+= r"$a$ = " + f"{a:.2f}\n"
        label2+= r"$\tau$ = " + f"{tau:.2f}"
        ax.plot(trelease, dx_release_n, "s")
        ax.plot(np.linspace(0, trelease[-1], 1000), exp_fit([a,tau], np.linspace(0, trelease[-1], 1000)), 
                ls="-", c='darkorange', label=label2)
        ax.legend()
        ax.grid()
        ax.set_xlabel("t [s]")
        ax.set_ylabel("Normalized displacement")
        ax.set_ylim([0, 1.5])
        plt.tight_layout()
        fig2.savefig(os.path.join(analysisDir, pull_id + "_fits.png"))
        
    if SHOW:
        plt.show()
    else:
        plt.close('all')
            
    plt.ion()
        
    #### 7. Output
    return(output, error)

