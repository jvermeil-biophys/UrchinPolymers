# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 10:00:29 2026

@author: Utilisateur
"""


# %% Imports

import os
import re
import time

import numpy as np
import pandas as pd
import skimage as skm
import seaborn as sns
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

import shapely
from shapely.ops import polylabel
from shapely.plotting import plot_polygon, plot_points # , plot_line

from trackpy.motion import msd, imsd, emsd
from PIL import Image, ImageDraw
from scipy import signal # stats #, optimize, interpolate, 

import PlotMaker as pm
import UrchinPaths as up
import UtilityFunctions as ufun

# %% Settings

SCALE = 0.222
FPS = 1


# %% Functions

#### Helper functions

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

    


def find_cell_inner_circle(img, 
                           binarize = False, k_th = 1.0,
                           zero_padding = 0,
                           PLOT=False):
    """
    On a picture of a cell in fluo, find the approximate position of the cell.
    The idea is to fit the largest circle which can be contained in a mask of the cell.
    It uses the library shapely, and more precisely the function polylabel,
    to find the "pole of inaccessibility" of the cell mask.
    See : https://github.com/mapbox/polylabel and https://sites.google.com/site/polesofinaccessibility/
    Interesting topic !
    """
    if binarize:
        th1 = skm.filters.threshold_otsu(img) * k_th
        # img_min = ndi.binary_fill_holes(img_min)
        # img_min = ndi.binary_closing(img_min, iterations=5)
        img_bin = (img < th1)
        img_bin = ndi.binary_opening(img_bin, iterations = 2)
    else:
        img_bin = img
    img_label, num_features = ndi.label(img_bin)
    df = pd.DataFrame(skm.measure.regionprops_table(img_label, img, properties = ['label', 'area']))
    df = df.sort_values(by='area', ascending=False)
    i_label = df.label.values[0]
    img_rawCell = (img_label == i_label)
    if zero_padding > 0:
        pad_width = zero_padding
        img_rawCell = np.pad(img_rawCell, pad_width, mode='constant')
        if PLOT:
            img = np.pad(img, pad_width, mode='constant')
    img_rawCell = ndi.binary_fill_holes(img_rawCell)
    
    # [contour_rawCell] = skm.measure.find_contours(img_rawCell, 0.5)
    FoundContours = skm.measure.find_contours(img_rawCell, 0.5)
    if len(FoundContours) == 1:
        contour_rawCell = FoundContours[0]
    else:
        L = [len(c) for c in FoundContours]
        im = np.argmax(L)
        contour_rawCell = FoundContours[im]    

    polygon_cell = shapely.Polygon(contour_rawCell[:, ::-1])
    center = polylabel(polygon_cell, tolerance=1)
    exterior_ring_cell = shapely.get_exterior_ring(polygon_cell)
    R = shapely.distance(center, exterior_ring_cell)
    circle = center.buffer(R)
    
    X, Y = list(center.coords)[0]
    X = X - zero_padding
    Y = Y - zero_padding
    
    if PLOT:
        fig, axes = plt.subplots(1,2, figsize = (8,4))
        ax = axes[0]
        ax.imshow(img_rawCell, cmap='gray')
        ax = axes[1]
        ax.imshow(img, cmap='gray')
        
        for ax in axes:
            plot_polygon(circle, ax=ax, add_points=False, color='green')
            plot_points(center, ax=ax, color='green', alpha=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        fig.tight_layout()
        plt.show()
    
    Y, X = round(Y), round(X)
    # mask =  img_rawCell
    return((Y, X), R)



def contour_to_mask(shape, contour):
    """
    Adapted from https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask
    """
    nY, nX = shape
    poly = np.round(contour[:,::-1], 0).astype(int).tolist()
    poly = [(p[0], p[1]) for p in poly[:]] + [(poly[0][0], poly[0][1])]
    Im0 = Image.new('L', (nX, nY), 0)
    ImageDraw.Draw(Im0).polygon(poly, outline=1, fill=1)
    mask = np.array(Im0).astype(bool)
    return(mask)




def get_Pa_value(df, manip_id, Pa):
    df['manip_id'] = df['date'] + '_' + df['manip']
    dff = df[df['manip_id'] == manip_id]
    Pa = str(Pa)
    L_irr = []
    L_Dt = []
    L_pow = []
    for n in Pa:
        Irr = dff.loc[dff['Pa']==int(n), 'irradiance'].values[0]
        Dt = dff.loc[dff['Pa']==int(n), 'duration'].values[0]
        L_irr.append(Irr)
        L_Dt.append(Dt)
        L_pow.append(Irr*Dt)
    return(np.array(L_irr), np.array(L_Dt), np.array(L_pow))



def get_reasonable_inner_cell_contour(img, PLOT = False):
    nT, nY, nX = img.shape
    img_min = np.min(img, axis = 0)
    
    (Yc, Xc), Rc = find_cell_inner_circle(img_min, binarize = True, 
                                          zero_padding = 10,
                                          PLOT=PLOT)
    Angles = np.linspace(0, 2*np.pi, 360)
    Xcontour = Xc + Rc*np.cos(Angles)
    Ycontour = Yc + Rc*np.sin(Angles)
    contour = np.array([Ycontour, Xcontour]).T
    if PLOT:
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(img_min, cmap='gray')
        axes[0].plot(contour[:,1], contour[:,0], 'r-')
        mask = contour_to_mask([nY, nX], contour)
        axes[1].imshow(img[0]*mask, cmap='gray')
        plt.show()
    return(contour)


def get_numbers_following_text(text, target, output = 'integer'):
    if output == 'integer':
        m = re.search(r''+target, text)
        m_num = re.search(r'[\d\.]+', text[m.end():m.end()+10])
        res = int(text[m.end():m.end()+10][m_num.start():m_num.end()])
    elif output == 'string':
        m = re.search(r''+target, text)
        m_num = re.search(r'[\d\.-]+', text[m.end():m.end()+10])
        res = str(text[m.end():m.end()+10][m_num.start():m_num.end()])
    return(res)
    


def check_if_file_has_tracks(fileName, srcDir):
    fN_root = fileName.split('.')[0]
    fN_contour = fN_root + '_Tracks.xml'
    has_contours = os.path.isfile(os.path.join(srcDir, fN_contour))
    return(has_contours)






#### Main functions


def compute_acor(image, mask, window_length, FPS,
                 EQUALIZE = True, PLOT = False):
    if EQUALIZE:
        for t in range(image.shape[0]):
            p1, p99 = np.percentile(image[t].flatten()[mask.flatten()], (1, 99))
            image[t] = skm.exposure.rescale_intensity(image[t], in_range=(p1, p99))
    
    if PLOT:
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(image[0]*mask, cmap = 'gray')
        axes[1].imshow(image[-1]*mask, cmap = 'gray')
        plt.show()
    
    short_len = window_length
    long_len = image.shape[0] - short_len + 1
    image_acor = np.zeros((long_len, image.shape[1], image.shape[2]))
    
    Zero_std_found = False
    
    image_mean = np.mean(image, axis=0)
    image_std = np.std(image, axis=0)
    non_zero_std = (image_std > 0)
    mask_2 = (mask & non_zero_std)
    image_normalized = (image - image_mean) / (image_std + (1-mask_2))
    
    for i in range(image.shape[1]):
        for j in range(image.shape[2]):
            if mask_2[i, j]:
                acor = signal.correlate(image_normalized[:,i,j], 
                                        image_normalized[:short_len,i,j], 
                                        mode="valid")
                acor = acor / acor[0]
                image_acor[:, i, j] = acor
                    
    total_acor = np.zeros(long_len)
    lags = np.arange(long_len) * (1/FPS)
    for t in range(len(total_acor)):
        total_acor[t] = np.mean(image_acor[t].flatten()[mask.flatten()])
    
    if PLOT:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(mask, cmap='gray')
        plt.show()
        fig, ax = plt.subplots(1, 1)
        ax.plot(lags, total_acor)
        plt.show()
        
    return(total_acor, image_acor)



def analyse_cells_ACF(imagePathList, df_Pa, SCALE, FPS):
    res_dict = {
                'id':[],
                'pos_id':[],
                'cell_id':[],
                'long_cell_id':[],
                'Pa':[],
                'Pa_total_power':[],
                'Pa_irradiance':[],
                'Pa_dt':[],
                't_50p':[],
                't_33p':[],
                't_25p':[],
                't_20p':[],
                't_0p':[],
                }
    ACF_dict = {}
    
    get_long_cell_id = lambda x : '_'.join(x.split('_')[:3] + [x.split('_')[-1]])
    
    print(pm.BLUE + 'Starting ACF analysis' + pm.NORMAL)
    
    for p in imagePathList:
        T0 = time.time()
        
        # Ids
        _, fN = os.path.split(p)
        print(pm.GREEN + f'Analysing {fN}' + pm.NORMAL)
        full_id = '_'.join(fN.split('_')[:5])
        manip_id = '_'.join(fN.split('_')[:2])
        long_cell_id = get_long_cell_id(full_id)
        pos_id = get_numbers_following_text(fN, '_Pos', output='string')
        cell_id = get_numbers_following_text(fN, '_C')
        Pa = get_numbers_following_text(fN, '_Pa')
        Irr, DT, Pow = get_Pa_value(df_Pa, manip_id, Pa) # mW/cm2 ; mJ/cm2
        str_irr = '_'.join(Irr.astype(str))
        str_dt = '_'.join(DT.astype(str))
        total_power = np.sum(Pow)/1000 # J/cm2
        
        # Get image and mask       
        image_raw = skm.io.imread(p)
        image = skm.util.img_as_float32(image_raw)
        shape = image.shape

        single_contour = get_reasonable_inner_cell_contour(image, PLOT = False)
        single_mask = contour_to_mask([shape[1], shape[2]], single_contour)
        single_mask = ndi.binary_erosion(single_mask, iterations = 50)
        
        window_length = shape[0]//3
        
        # ACF function
        total_acor, image_acor = compute_acor(image, single_mask, 
                                              window_length, FPS,
                                              EQUALIZE = True, PLOT = False)
        
        ACF_dict[full_id] = total_acor
        
        timescales = ['t_50p', 't_33p', 't_25p', 't_20p', 't_0p']
        th_timescales = [1/2, 1/3, 1/4, 1/5, 0]
        dict_timescales = {k:0 for k in timescales}
        # print(timescales)
        # print(th_timescales)
        # print(dict_timescales)
        
        interp_factor = 10
        total_acor_interp = ufun.resize_1Dinterp(total_acor, 
                                                 fx=interp_factor)
        
        for ts, th in zip(timescales, th_timescales):
            test = (total_acor_interp < th)
            t = ufun.findFirst(1, test)/interp_factor
            dict_timescales[ts] = t
        
        res_dict['id'].append(full_id)
        res_dict['pos_id'].append(pos_id)
        res_dict['cell_id'].append(cell_id)
        res_dict['long_cell_id'].append(long_cell_id)
        res_dict['Pa'].append(Pa)
        res_dict['Pa_total_power'].append(total_power)
        res_dict['Pa_irradiance'].append(str_irr)
        res_dict['Pa_dt'].append(str_dt)
        for ts in dict_timescales.keys():
            res_dict[ts].append(dict_timescales[ts])
        
        Dt = time.time() - T0
        print(f'Done in Dt = {Dt:.4f}')
        
    res_df = pd.DataFrame(res_dict)
        
    return(res_df, ACF_dict)
        
        
        

def analyse_white_blobs_MSD(trackPathList, df_Pa, SCALE, FPS):
    res_dict = {
                'id':[],
                'pos_id':[],
                'cell_id':[],
                'Pa':[],
                'Pa_total_power':[],
                'Pa_irradiance':[],
                'Pa_dt':[],
                'D':[],
                'k_nl':[],
                'D_nl':[],
                }
    tables_dict = {}
    MSD_dict = {}

    print(pm.BLUE + 'Starting MSD analysis' + pm.NORMAL)    
    
    for p in trackPathList:
        T0 = time.time()
        
        # Ids
        _, fN = os.path.split(p)
        print(pm.GREEN + f'Analysing {fN}' + pm.NORMAL)
        
        full_id = '_'.join(fN.split('_')[:5])
        manip_id = '_'.join(fN.split('_')[:2])
        pos_id = get_numbers_following_text(fN, '_Pos')
        cell_id = get_numbers_following_text(fN, '_C')
        Pa = get_numbers_following_text(fN, '_Pa')
        Irr, DT, Pow = get_Pa_value(df_Pa, manip_id, Pa) # mW/cm2 ; mJ/cm2
        str_irr = '_'.join(Irr.astype(str))
        str_dt = '_'.join(DT.astype(str))
        total_power = np.sum(Pow)/1000 # J/cm2
        
        # MSD
        Tracks = importTrackMateTracks(p)
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
        tables_dict[full_id] = df
        
        #### Run imsd -> Might be useful for SEM computation
        # res_imsd = imsd(df, SCALE, FPS).reset_index()
    
        #### Run msd
        res_emsd = emsd(df, SCALE, FPS, max_lagtime=40).reset_index()
        T, MSD = res_emsd['lagt'], res_emsd['msd']
        MSD_dict[full_id] = np.array([T, MSD]).T
        
        parms, results = ufun.fitLineHuber(T, MSD, with_intercept = False)
        D = parms.values[0]/4
        
        parms, results = ufun.fitLineHuber(np.log(T), np.log(MSD), with_intercept = True)
        b, a = parms
        k_nl = a
        D_nl = np.exp(b)/4
        
        res_dict['id'].append(full_id)
        res_dict['pos_id'].append(pos_id)
        res_dict['cell_id'].append(cell_id)
        res_dict['Pa'].append(Pa)
        res_dict['Pa_total_power'].append(total_power)
        res_dict['Pa_irradiance'].append(str_irr)
        res_dict['Pa_dt'].append(str_dt)
        res_dict['D'].append(D)
        res_dict['k_nl'].append(k_nl)
        res_dict['D_nl'].append(D_nl)
        
        Dt = time.time() - T0
        print(f'Done in Dt = {Dt:.4f}')
        
    res_df = pd.DataFrame(res_dict)
        
    return(res_df, MSD_dict)



#### Post-analysis functions

# Corrections / modifications of the result tables

def add_irr_and_pow_to_table(df, df_Pa):
    list_id = df['id'].values
    L_pow = []
    L_Irr = []
    L_Dt = []
    for i in list_id:
        manip_id = '_'.join(i.split('_')[:2])
        Pa = df.loc[df['id']==i, 'Pa'].values[0]
        irr, dt, power = get_Pa_value(df_Pa, manip_id, Pa)
        str_irr = '_'.join(irr.astype(str))
        str_dt = '_'.join(dt.astype(str))
        power = int(np.sum(power)/1000)
        L_pow.append(power)
        L_Irr.append(str_irr)
        L_Dt.append(str_dt)
    df['Pa_total_power'] = L_pow
    df['Pa_irradiance'] = L_Irr
    df['Pa_dt'] = L_Dt
    return(df)



# %% --------



# %% Analysis Scripts


# %%% Main script for ACF 

SCALE = 0.222
FPS = 1
file2id = lambda x : '_'.join(x.split('_')[:5])

redo_all_files = True
 
dirPath = up.Path_AnalysisPulls + "/26-02-11_UVonCytoplasmAndBeads"
listAllFiles = os.listdir(dirPath)
listTifFiles = [f for f in listAllFiles if ((f.endswith('.tif')) and ('Film5min' in f))]

PaTableName = "MainIrradianceConditions.csv"
PaTablePath = os.path.join(up.Path_AnalysisPulls, PaTableName)
df_Pa = pd.read_csv(PaTablePath, sep=',')

if not redo_all_files:
    ACF_res_df = pd.read_csv(os.path.join(dirPath, 'Results', 'results_ACF.csv'))
    ACF_dict = ufun.json2dict(dirPath + '/Results', 'ACF_dict')
    for col in ACF_res_df.columns:
        if 'Unnamed' in col:
            ACF_res_df = ACF_res_df.drop(labels = col, axis = 1)
    analyzed_id = ACF_res_df['id'].unique()
    listTifFiles = [f for f in listTifFiles if file2id(f) not in analyzed_id]

listTifPaths = [os.path.join(dirPath, f) for f in listTifFiles]

new_ACF_res_df, new_ACF_dict = analyse_cells_ACF(listTifPaths[:], df_Pa, 
                                                 SCALE, FPS)

if redo_all_files:
    ACF_res_df = new_ACF_res_df
    ACF_dict = new_ACF_dict
if not redo_all_files:
    ACF_res_df = pd.concat([ACF_res_df, new_ACF_res_df], axis=0)
    ACF_dict.update(new_ACF_dict)

ufun.dict2json(ACF_dict, dirPath + '/Results', 'ACF_dict')
ACF_res_df.to_csv(os.path.join(dirPath, 'Results', 'results_ACF.csv'), index=False)




# %%% Script for MSD


SCALE = 0.222
FPS = 1

redo_all_files = True

dirPath = up.Path_AnalysisPulls + "/26-02-09_UVonCytoplasm"
listAllFiles = os.listdir(dirPath)
listTifFiles = [f for f in listAllFiles if f.endswith('.tif')]

PaTableName = "MainIrradianceConditions.csv"
PaTablePath = os.path.join(up.Path_AnalysisPulls, PaTableName)
df_Pa = pd.read_csv(PaTablePath)

if not redo_all_files:
    MSD_res_df = pd.read_csv(os.path.join(dirPath, 'Results', 'results_MSD.csv'))
    for col in ACF_res_df.columns:
        if 'Unnamed' in col:
            MSD_res_df = MSD_res_df.drop(label = col, axis = 1)
    analyzed_id = MSD_res_df['id'].unique()
    listTifFiles = [f for f in listAllFiles if file2id(f) not in analyzed_id]


filesWithTracks = {fN:True for fN in listTifFiles}
for fN in listTifFiles:
    check = check_if_file_has_tracks(fN, dirPath)
    filesWithTracks[fN] = (check)
listTifFiles = [fN for fN in listTifFiles if filesWithTracks[fN]]
listTrackFiles = [fN[:-4] + '_Tracks.xml' for fN in listTifFiles if filesWithTracks[fN]]


listTifPaths = [os.path.join(dirPath, f) for f in listTifFiles]
listTrackPaths = [os.path.join(dirPath, f) for f in listTrackFiles]

MSD_res_df, MSD_dict = analyse_white_blobs_MSD(listTrackPaths[:], df_Pa, 
                                               SCALE, FPS)

if redo_all_files:
    ACF_res_df = new_ACF_res_df
    ACF_dict = new_ACF_dict
if not redo_all_files:
    ACF_res_df = pd.concat([ACF_res_df, new_ACF_res_df], axis=0)
    ACF_dict.update(new_ACF_dict)


ufun.dict2json(MSD_dict, dirPath + '/Results', 'MSD_dict')
MSD_res_df.to_csv(os.path.join(dirPath, 'Results', 'results_MSD.csv'), index=False)




# %% Plots

# %%% Datasets

SCALE = 0.222
FPS = 1

# "/26-02-09_UVonCytoplasm"
# "/26-02-11_UVonCytoplasmAndBeads"

dirPath = up.Path_AnalysisPulls + "/26-02-11_UVonCytoplasmAndBeads"
listAllFiles = os.listdir(dirPath)

df_ACF = pd.read_csv(os.path.join(dirPath, 'Results', 'results_ACF.csv'))
# if 'long_cell_id' not in df_ACF.columns:
#     get_long_cell_id = lambda x : '_'.join(x.split('_')[:4])# + [x.split('_')[-1]])
#     df_ACF['long_cell_id'] = df_ACF['id'].apply(get_long_cell_id)
#     # df_ACF.to_csv(os.path.join(dirPath, 'Results', 'results_ACF.csv'), index=False)

metrics_cols = [col for col in df_ACF.columns if col.startswith('t_')]
for mc in metrics_cols:
    mcn = mc + '_norm'
    df_ACF[mcn] = df_ACF[mc]
for k, cid in enumerate(df_ACF['long_cell_id'].unique()):
    index_cell = df_ACF[df_ACF['long_cell_id'] == cid].index
    index_cell_control = df_ACF[(df_ACF['long_cell_id'] == cid) & (df_ACF['Pa'] == 0)].index
    for mc in metrics_cols:
        mcn = mc + '_norm'
        val_ctrl = df_ACF.loc[index_cell_control, mc].values[0]
        df_ACF.loc[index_cell, mcn] /= val_ctrl
    

# df_MSD = pd.read_csv(os.path.join(dirPath, 'Results', 'results_MSD.csv'))

# df_merged = pd.merge(left=df_MSD, right=df_ACF, on='id', how='inner', suffixes=(None, '_2'))
# df_merged['Pa'] = df_merged['Pa'].apply(lambda x : str(x))

# %%% ACF

Id_cols = ['pos_id',]
Group_cols = ('Pa')
Yplot = 't_33p'

group_ACF = df_ACF.groupby(Group_cols)
agg_dict = {k:'first' for k in Id_cols}
agg_dict.update({Yplot:'mean'})
df_ACF_g = group_ACF.agg(agg_dict).reset_index()

fig, ax = plt.subplots(1, 1)
sns.swarmplot(data=df_ACF, ax=ax, x='Pa', y=Yplot)
ax.set_ylim([0, ax.get_ylim()[1]])
plt.tight_layout()
plt.show()


# %%% ACF - 2

Id_cols = ['pos_id']
Group_cols = ('Pa')
Xplot = 'Pa_total_power'
Yplot = 't_33p'
Hplot = 'Pa_irradiance'
hue_order=['0', '200', '200_200', '400', '400_400', '800', '800_800', 
           '1600', '1600_1600', '2400', '2400_2400', '2400_2400_2400']
hue_order = [h for h in hue_order if h in df_ACF[Hplot].unique()]

group_ACF = df_ACF.groupby(Xplot)
agg_dict = {k:'first' for k in Id_cols}
agg_dict.update({Yplot:'median', Yplot + '_norm':'median'})
df_ACF_g = group_ACF.agg(agg_dict).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
ax.grid(zorder=0)
sns.scatterplot(data=df_ACF, ax=ax, x=Xplot, y=Yplot, 
                hue=Hplot, hue_order = hue_order,
                alpha = 0.75, zorder=6)
sns.scatterplot(data=df_ACF_g, ax=ax, x=Xplot, y=Yplot, marker = 'o',
                color='None', edgecolor='k', s=75, zorder=6)
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel(r'Total energy (J/cm2)')
ax.set_ylabel(r'$T_{33\%}$ (s)')
ax.legend().set_visible(False)

ax = axes[1]
ax.grid(zorder=0)
sns.scatterplot(data=df_ACF, ax=ax, x=Xplot, y=Yplot + '_norm', 
                hue=Hplot, hue_order = hue_order,
                alpha = 0.75, zorder=6)
sns.scatterplot(data=df_ACF_g, ax=ax, x=Xplot, y=Yplot + '_norm', marker = 'o',
                color='None', edgecolor='k', s=75, zorder=6, label='Median values')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel(r'Total energy (J/cm2)')
ax.set_ylabel(r'$T_{33\%}$ - normalized')
ax.legend(title='Photo-activation\nsequence [mW/cm2]', 
          loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))


plt.tight_layout()
plt.show()


# %%% MSD

# Id_cols = ['pos_id',]
# Group_cols = ('Pa')
# Yplot = 'D'

# group_MSD = df_MSD.groupby(Group_cols)
# agg_dict = {k:'first' for k in Id_cols}
# agg_dict.update({Yplot:'mean'})
# df_MSD_g = group_MSD.agg(agg_dict).reset_index()

# fig, ax = plt.subplots(1, 1)
# ax.set_yscale('log')
# sns.swarmplot(data=df_MSD, ax=ax, x='Pa', y=Yplot)
# ax.set_ylim([0, ax.get_ylim()[1]])
# plt.tight_layout()
# plt.show()



# %%% Check consistency

# Xplot = 'D'
# Yplot = 't_33p'
# Hplot = 'Pa_total_power'

# fig, ax = plt.subplots(1, 1)
# # ax.set_yscale('log')
# sns.scatterplot(data=df_merged, ax=ax, x=Xplot, y=Yplot, 
#                 hue=Hplot,
#                 )
# plt.tight_layout()
# plt.show()


# %%% Check consistency 2

# Xplot = 'D'
# Yplot = 't_0p'
# Hplot = 'Pa_total_power'

# fig, ax = plt.subplots(1, 1)
# # ax.set_yscale('log')
# sns.scatterplot(data=df_merged, ax=ax, x=Xplot, y=Yplot, 
#                 hue=Hplot,
#                 )
# plt.tight_layout()
# plt.show()





# %% --------

# %% Small scripts

# %%% ACF - Add numeric value for Power and Irradiance

# dirPath = up.Path_AnalysisPulls + "/26-02-09_UVonCytoplasm"
# listAllFiles = os.listdir(dirPath)
# listTifFiles = [f for f in listAllFiles if f.endswith('.tif')]

# PaTableName = "MainIrradianceConditions.csv"
# PaTablePath = os.path.join(up.Path_AnalysisPulls, fileName)
# df_Pa = pd.read_csv(PaTablePath)

# ACF_res_df = pd.read_csv(os.path.join(dirPath, 'Results', 'results_ACF.csv'))
# ACF_dict = ufun.json2dict(dirPath + '/Results', 'ACF_dict')

# ACF_res_df_2 = add_irr_and_pow_to_table(ACF_res_df, df_Pa)
# ACF_res_df_2.to_csv(os.path.join(dirPath, 'Results', 'results_ACF.csv'), index=False)

# %%% MSD - Add numeric value for Power and Irradiance

# dirPath = up.Path_AnalysisPulls + "/26-02-09_UVonCytoplasm"
# listAllFiles = os.listdir(dirPath)
# listTifFiles = [f for f in listAllFiles if f.endswith('.tif')]

# PaTableName = "MainIrradianceConditions.csv"
# PaTablePath = os.path.join(up.Path_AnalysisPulls, fileName)
# df_Pa = pd.read_csv(PaTablePath)

# MSD_res_df = pd.read_csv(os.path.join(dirPath, 'Results', 'results_ACF.csv'))
# # ACF_dict = ufun.json2dict(dirPath + '/Results', 'ACF_dict')

# MSD_res_df_2 = add_irr_and_pow_to_table(MSD_res_df, df_Pa)
# MSD_res_df_2.to_csv(os.path.join(dirPath, 'Results', 'results_MSD.csv'), index=False)

# %%% Test a problematic cell

dirPath = up.Path_AnalysisPulls + "/26-02-09_UVonCytoplasm"
# fileName = "26-02-09_M1_Pos5_Pa66_C4_Film5min_Dt1sec.tif"
fileName = "26-02-09_M1_Pos6_Pa7_C7_Film5min_Dt1sec_1.tif"
filePath = os.path.join(dirPath, fileName)

print(pm.GREEN + f'Analysing {fileName}' + pm.NORMAL)
T0 = time.time()

# Get image and mask       
image_raw = skm.io.imread(filePath)
image = skm.util.img_as_float32(image_raw)
shape = image.shape

single_contour = get_reasonable_inner_cell_contour(image, PLOT = False)
single_mask = contour_to_mask([shape[1], shape[2]], single_contour)
single_mask = ndi.binary_erosion(single_mask, iterations = 50)

window_length = shape[0]//3

# ACF function
total_acor, image_acor = compute_acor(image, single_mask, 
                                      window_length, FPS,
                                      EQUALIZE = True, PLOT = False)

timescales = ['t_50p', 't_33p', 't_25p', 't_20p', 't_0p']
th_timescales = [1/2, 1/3, 1/4, 1/5, 0]
dict_timescales = {k:0 for k in timescales}
# print(timescales)
# print(th_timescales)
# print(dict_timescales)

interp_factor = 10
total_acor_interp = ufun.resize_1Dinterp(total_acor, 
                                         fx=interp_factor)
x = np.arange(0, len(total_acor), 1)
x_interp = np.arange(0, len(total_acor), 1/interp_factor)
# fig, ax = plt.subplots(1, 1)
# ax.plot(x_interp, total_acor_interp, 'r-')
# ax.plot(x, total_acor, 'b.')
# plt.show()

for ts, th in zip(timescales, th_timescales):
    test = (total_acor_interp < th)
    t = ufun.findFirst(1, test)/interp_factor
    dict_timescales[ts] = t
    
Dt = time.time() - T0
print(f'Done in Dt = {Dt:.4f}')

# %%% Compare MSDs

#### Import tracked trajectories

dirPath = up.Path_AnalysisPulls + "/26-02-09_UVonCytoplasm"

listFiles = [
             "26-02-09_M1_Pos3_Pa0_C5_Film5min_Dt1sec_VHighCut_Tracks.xml",
             "26-02-09_M1_Pos3_Pa0_C4_Film5min_Dt1sec_HighCut_Tracks.xml",
             "26-02-09_M1_Pos3_Pa0_C4_Film5min_Dt1sec_LowCut_Tracks.xml",
             # "26-02-09_M1_Pos3_Pa0_C5_Film5min_Dt1sec_Vth2-0_Tracks.xml",
             # "26-02-09_M1_Pos3_Pa0_C5_Film5min_Dt1sec_Vth2-5_Tracks.xml",
             # "26-02-09_M1_Pos3_Pa0_C5_Film5min_Dt1sec_Vth3-0_Tracks.xml",
             # "26-02-09_M1_Pos3_Pa0_C5_Film5min_Dt1sec_Vth3-5_Tracks.xml",
             ]

listPaths = [os.path.join(dirPath, fN) for fN in listFiles]
listTracks = [importTrackMateTracks(fP) for fP in listPaths]
listColors = pm.cL_Set21[:len(listTracks)]
listLabels = [
             "VHighCut",
             "HighCut",
             "LowCut",
             # "Vth2-0",
             # "Vth2-5",
             # "Vth3-0",
             # "Vth3-5",
             ]

#### Format as table & filter
Tables = []
column_names = ['frame', 'x', 'y', 'particle']
for Tracks, F in zip(listTracks, listFiles):
    T0 = time.time()
    all_tracks = []
    for i, track in enumerate(Tracks):
        nT = len(track)
        track = np.concat((track, np.ones((len(track[:,0]), 1), dtype=int) * (i+1)), axis = 1)
        track[:,0] = track[:,0].astype(int) + 1
        all_tracks.append(track)
    concat_tracks = np.concat(all_tracks, axis = 0)
    d = {column_names[k] : concat_tracks[:,k] for k in range(len(column_names))}
    df = pd.DataFrame(d)
    Tables.append(df)
    print(f'Done Formatting {F} in Dt = {time.time()-T0:.3f}')

#### Run msd
fig, ax = plt.subplots(1, 1)

for k in range(len(listTracks)):
    T0 = time.time()
    df = Tables[k]
    color, label = listColors[k], listLabels[k]
    
    # res_imsd = imsd(df, SCALE, FPS).reset_index()
    res_emsd = emsd(df, SCALE, FPS, max_lagtime=40).reset_index()
    print(f'Computed EMSD for {listFiles[k]} in Dt = {time.time()-T0:.3f}')
    
    T, MSD = res_emsd['lagt'], res_emsd['msd']
    parms, results = ufun.fitLineHuber(T, MSD, with_intercept = False)
    [D] = parms/4
    
    ax.plot(res_emsd['lagt'], res_emsd['msd'], color=color, marker='.', lw=0.5, label=label)
    ax.axline(xy1=(0,0), slope=D*4, color=pm.lighten_color(color, 0.5), ls='-', lw=1, label=f'D = {D:.2e} µm²/s')

ax.grid()
ax.set_xlabel('Lag time (s)')
ax.set_ylabel('MSD (µm²)')
ax.legend()

fig.tight_layout()
plt.show()
