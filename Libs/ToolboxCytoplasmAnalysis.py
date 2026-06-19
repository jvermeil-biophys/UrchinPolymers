# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:32:49 2026

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


# %% Imports & settings

#### Imports

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

import Libs.PlotMaker as pm
import Libs.UrchinPaths as up
import Libs.UtilityFunctions as ufun

#### Settings

SCALE_20X = 0.461
SCALE_40X = 0.229
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



def get_reasonable_inner_cell_contour(img, PLOT = False):
    nT, nY, nX = img.shape
    img_min = np.min(img, axis = 0)
    
    (Yc, Xc), Rc = ufun.find_cell_inner_circle(img_min, binarize = True, 
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
        mask = ufun.contour_to_mask([nY, nX], contour)
        axes[1].imshow(img[0]*mask, cmap='gray')
        plt.show()
    return(contour)


def get_numbers_following_text(text, target, output = 'integer'):
    if output == 'integer':
        m = re.search(r''+target, text)
        m_num = re.search(r'[\d]+', text[m.end():m.end()+10])
        res = int(text[m.end():m.end()+10][m_num.start():m_num.end()])
    elif output == 'string':
        m = re.search(r''+target, text)
        m_num = re.search(r'[\d-]+', text[m.end():m.end()+10])
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
        

def analyse_white_blobs_MSD(trackPathList, df_Pa, SCALE, FPS,
                            PLOT = False):
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
    
    if PLOT:
        fig, ax = plt.subplots(1, 1)
        Nt = len(trackPathList)
        Nc = len(pm.cL_Set21)
        if Nt <= Nc:
            listColors = pm.cL_Set21[:Nt]
        else:
            listColors = pm.cL_Set21
    
    for k, p in enumerate(trackPathList):
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
        res_emsd = emsd(df, SCALE, FPS, max_lagtime=30).reset_index()
        T, MSD = res_emsd['lagt'], res_emsd['msd']
        MSD_dict[full_id] = np.array([T, MSD]).T
        
        parms, results = ufun.fitLineHuber(T, MSD, with_intercept = False)
        D = parms.values[0]/4
        
        if PLOT:
            color = listColors[k%Nc]
            dark_color = pm.lighten_color(color, 0.5)
            ax.plot(res_emsd['lagt'], res_emsd['msd'], color=color, marker='.', lw=0.5, 
                    label=full_id)
            ax.axline(xy1=(0,0), slope=D*4, color=dark_color, ls='-', lw=1, 
                      label=f'D = {D:.2e} µm²/s')
        
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
        
    if PLOT:
        ax.set_xlabel('Lag times (s)')
        ax.set_ylabel('MSD (µm²)')
        ax.grid()
        ax.legend()
        fig.tight_layout()
        plt.show()
        
    res_df = pd.DataFrame(res_dict)
        
    return(res_df, MSD_dict)


