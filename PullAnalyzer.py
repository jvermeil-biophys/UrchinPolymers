# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 21:49:47 2025

@author: Team Minc, adapted in Python by Joseph Vermeil
"""


# %% 1. Imports

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from scipy.interpolate import make_splrep

from scipy.io import savemat
from scipy.optimize import minimize

from copy import deepcopy


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

def Langevin(x):
    # if x < 0.05:
    #     return(np.tanh(x/3))
    # else:
    return((1/np.tanh(x)) - (1/x))

def New_D2V(x, A, B, x0):
    return(A * Langevin(B/(x-x0)**3) * 1/(x-x0)**4)

# def New_D2V(x, A, x0):
#     return(A*np.exp(-x/x0))


# %% 3. Parameters


# %%% 3.1. P1

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

# %%% 3.2. P2

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


# %% 4. Load data

tracks = importTrackMateTracks(os.path.join(directory, tracks_name))
track = trackSelection(tracks, mode = 'longest')

X, Y, t = track[:,1], track[:,2], track[:,0]

initPullTime = np.where(t == (frame_initPull - 1))[0][0]
finPullTime = np.where(t == (frame_endPull - 1))[0][0]


# %% 5. Pretreatment

# %%% --- Rotate track ---
theta = np.arctan2(Y[initPullTime] - Y[finPullTime],
                   X[initPullTime] - X[finPullTime])
rotation_mat = np.array([[np.cos(-theta), -np.sin(-theta)],
                         [np.sin(-theta),  np.cos(-theta)]])
coords = np.vstack((X, Y)).T @ rotation_mat.T
x_rot, y_rot = coords[:,0], coords[:,1]

x_shift = (-x_rot[initPullTime:] + np.max(x_rot[initPullTime:])) * pixel_size

# %%% --- Pulling phase ---
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

# %%% Release phase

release_index = np.arange(finPullTime, len(track))
release_length = len(release_index)

trelease = (t[release_index] - t[finPullTime+1]) * time_stp
dx_release = x_shift[release_index - initPullTime]
dx_release_n = dx_release / x_shift[pull_length]

# %% 6. Fit model

# %%% Option 1. --- Viscous model ---

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




# %% 11. A function to analyze multiple files

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
                    results_dict['k'].append(visco*1000)
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
        selected_XY = [xy for xy in selected_XY if len(xy) > 75]
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
    # Magnet function distance (µm) to velocity (µm/s) [expected velocity in glycerol]
    mag_d2v = lambda x: 80.23*np.exp(-x/47.49) + 1.03*np.exp(-x/22740.0)
    new_d2v = lambda x: New_D2V(x, 2.75379e+10, 6.87603e+21, -124.896)
    # new_d2v = lambda x: New_D2V(x, 3.56905e+11, 3.76308e+25, -140)
    # new_d2v = lambda x: New_D2V(x, 1.7435e+10, 4.65859e+19, -100)
    # new_d2v = lambda x: New_D2V(x, 6.46079e+10, 1.09389e+21, 0)
    
    if fit_V == 'DoubleExpo': # Original behavior
        mag_d2v = mag_d2v
    elif fit_V == 'LangevinFun': # Test
        mag_d2v = new_d2v
        
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
    # Filter1 = (dist > 500) # bead further than 550 µm from magnet center
    # Filter1 = (tpulling >= 1) & (tpulling <= 10)
    Filter1 = tpulling >= 0.5
    # Filter2 = (tpulling <= 15) # remove the first 2 seconds of filming
    Filter2 = (dist <= 350) & (dist >= 250)
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
        ax.plot(tpulling[GlobalFilter], dx_pulling_n[GlobalFilter], "o", color='darkturquoise', markersize=5, zorder=5)
        ax.plot(tpulling[~GlobalFilter], dx_pulling_n[~GlobalFilter], "o", color='lightblue', markersize=5, zorder=4)
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
        
        i_select = np.abs(dist - dist[0]) < 50
        
        fig3, axes3 = plt.subplots(1, 1, figsize = (5,5))
        ax = axes3
        axbis = axes3.twinx()
        dist_spline = make_splrep(tpulling, dist, s=len(t)/2)
        ax.plot(tpulling, dist, 'go', ms = 3, zorder=4)
        ax.plot(tpulling, dist_spline(tpulling), 'k-', zorder=5)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("X [µm]")
        axbis.plot(tpulling, pull_force, 'r--', zorder=5)
        axbis.set_ylabel('Pulling force [pN]')
        ax.grid(axis='both')
        ax.legend(fontsize = 11).set_zorder(6)
        fig3.suptitle(pull_id, fontsize=12)
        fig3.tight_layout()
        
        # fig3.savefig(os.path.join(plotsDir, pull_id + "_fits.png"))
        
        fig4, axes4 = plt.subplots(1, 1, figsize = (5,5))
        ax = axes4
        
        dist_spline = make_splrep(tpulling, dist, s=len(t)/2)
        speed_spline = dist_spline.derivative(nu=1)
        speed = np.abs(speed_spline(tpulling))
        parms, results = fitLine(speed[i_select], pull_force[i_select])
        a, b = parms[1], parms[0]
        eta2 = a/(6*np.pi*bead_radius)
        xfit = np.linspace(0, max(speed), 100)
        yfit = b + a*xfit
        
        ax.plot(speed, pull_force, 'go', ms = 3, zorder=4)
        ax.plot(speed[i_select], pull_force[i_select], 'co', ms = 3, zorder=4)
        ax.plot(xfit, yfit, 'r--', label=r'$\eta$ = ' + f'{eta2*1000:.2f} mPa.s', zorder=5)
        ax.set_xlabel("V [µm/s]")
        ax.set_ylabel("F [pN]")
        ax.grid(axis='both')
        ax.legend(fontsize = 11).set_zorder(6)
        fig4.suptitle(pull_id, fontsize=12)
        fig4.tight_layout()
        
        fig5, axes5 = plt.subplots(1, 1, figsize = (5,5))
        dist2 = dist
        ax = axes5
        axbis = axes5.twinx()
        ax.set_xscale('log')
        ax.set_yscale('log')
        axbis.set_yscale('log')
        dist_spline = make_splrep(tpulling, dist2, s=len(t)/2)
        speed_spline = dist_spline.derivative(nu=1)
        speed = np.abs(speed_spline(tpulling))
        # ax.plot(tpulling, dist2, 'go', ms = 3, zorder=4)
        ax.plot(dist2, speed, 'k-', zorder=5)
        ax.set_xlabel("X [µm]")
        ax.set_ylabel("V [µm/s]")
        axbis.plot(dist2, pull_force, 'r-', zorder=5)
        ax.plot(dist2, (dist2/dist2[0])**(-7), 'b-', zorder=5)
        ax.plot(dist2, (dist2/dist2[0])**(-4), 'g-', zorder=5)
        ax.plot(dist2, (dist2/dist2[0])**(-2), 'c-', zorder=5)
        axbis.set_ylabel('Pulling force [pN]')
        ax.grid(axis='both')
        ax.legend(fontsize = 11).set_zorder(6)
        fig5.suptitle(pull_id, fontsize=12)
        fig5.tight_layout()
        
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
            
        plt.ion()
        
    #### 7. Output
    return(output, error)


# %% 12. Run the function

# %%% ... on many files

mainDir = os.path.join("C:/Users/Utilisateur/Desktop/")
date = '25-09-19'


analysisDir = os.path.join(mainDir, 'AnalysisPulls') # where the csv tables are
tracksDir = os.path.join(analysisDir, 'Tracks', date) # where the tracks are
resultsDir = os.path.join(analysisDir, 'Results')
plotsDir = os.path.join(analysisDir, 'Plots', date)

prefix_id = '25-09-19' # used to select a subset of the track files if needed

res, all_XY = pullAnalyzer_multiFiles(mainDir, date, prefix_id,
                        analysisDir, tracksDir, resultsDir, plotsDir,
                        mode = 'newton', resultsFileName = date + '_NaSS_results_TEST',
                        Redo = True, PLOT = False, SHOW = False)

# plt.close('all')

# %%% ... on one file

mainDir = os.path.join("C:/Users/Utilisateur/Desktop/")
date = '25-09-19'


analysisDir = os.path.join(mainDir, 'AnalysisPulls') # where the csv tables are
tracksDir = os.path.join(analysisDir, 'Tracks', date) # where the tracks are
resultsDir = os.path.join(analysisDir, 'Results')
plotsDir = os.path.join(analysisDir, 'Plots', date)

P_id = '25-09-19_M1_D5_P1_B1'
# P_id = '25-09-19_M2_D4_P1_B1' # used to select a subset of the track files if needed

df_manips = pd.read_csv(os.path.join(analysisDir, 'md_manips.csv'))
df_pulls = pd.read_csv(os.path.join(analysisDir, date + '_md_pulls.csv'))
dict_pull = Df2Dict(df_pulls[df_pulls['id'] == P_id])
trackFileName = [f for f in os.listdir(tracksDir) if f.startswith(P_id)][0]

tracks = importTrackMateTracks(os.path.join(tracksDir, trackFileName))
track = trackSelection(tracks, mode = 'longest')

# fit_V = 'LangevinFun'
fit_V = 'DoubleExpo'

output, error = pullAnalyzer(track, dict_pull,
                                mode = 'newton', fit_V = fit_V,
                                PLOT = True, SHOW = True, plotsDir = plotsDir)

visco, R2, speed_med, force_med, theta, r_min, r_max, XY = output




# %% ---------------

# %% 101. Tests & Legacy

# %%% Little test fitting

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


# %%% Test step 4.
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


# %%% Older version of the function

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
#         # TODO!
        
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
#             #### TODO!
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
#         ax.plot(tpulling[GlobalFilter], dx_pulling_n[GlobalFilter], "o", color='darkturquoise', markersize=5, zorder=5)
#         ax.plot(tpulling[~GlobalFilter], dx_pulling_n[~GlobalFilter], "o", color='lightblue', markersize=5, zorder=4)
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
#     return(output, error)