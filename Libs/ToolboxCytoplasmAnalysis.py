# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:32:49 2026

@author: Joseph Vermeil

UtilityFunctions.py - contains all kind of small functions used by CortExplore programs, 
to be imported with "import UtilityFunctions as ufun" and call with "ufun.my_function".
Joseph Vermeil, 2026

This program is free software: you can redistribute it and\\or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https:\\\\www.gnu.org\\licenses\\>.
"""


# %% Imports & settings

#### Imports

import os
import re
import cv2
import sys
import time
import alphashape

import numpy as np
import pandas as pd
import skimage as skm
import seaborn as sns
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from shapely import MultiPoint, Polygon
# from shapely.ops import polylabel
# from shapely.plotting import plot_polygon, plot_points # , plot_line

# from trackpy.motion import msd, imsd, emsd
from PIL import Image, ImageDraw
from scipy import signal # stats #, optimize, interpolate, 
from scipy.special import jv
from scipy.spatial import ConvexHull, Delaunay

import trackpy as tp

import Libs.PlotMaker as pm
import Libs.UrchinPaths as up
import Libs.UtilityFunctions as ufun

#### Settings

SCALE_20X = 0.461
SCALE_40X = 0.229
FPS = 1


# %% Imports 2

os.environ["JAVA_HOME"] = up.Path_JAVA_HOME

import imagej
import scyjava as sj

# import random
sj.config.add_options('-Xmx32g')



# initialize ImageJ
# ij = imagej.init()
# ij = imagej.init('sc.fiji:fiji')
ij = imagej.init(up.Path_Fiji, add_legacy=False)

print(f"ImageJ version: {ij.getVersion()}")


# %% I. Single Particle Tracking

# %%% Helper functions

def import_TrackMate_tracks(filepath):
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



def get_cell_inner_circle(img, PLOT = False):
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


def make_NbYolkCell_contour_and_mask(img, PixPerUm,
                                     mode = 'dark_background', 
                                     buffer_um = 0,
                                     PLOT = False):
    nT, nY, nX = img.shape
    k_th = 1.0
    
    if mode == 'dark_background':
        img_proj = np.max(img, axis = 0)
        th1 = skm.filters.threshold_li(img_proj) * k_th
        img_bin = (img_proj > th1)
        
    elif mode == 'light_background':
        img_proj = np.min(img, axis = 0)
        th1 = skm.filters.threshold_li(img_proj) * k_th
        img_bin = (img_proj < th1)
    
    img_bin = ndi.binary_opening(img_bin, iterations = 5)
    img_bin = ndi.binary_fill_holes(img_bin)
    
    # Get contours from first mask
    FoundContours = skm.measure.find_contours(img_bin, 0.5)
    
    # Get additionnal contours by "erroding" the mask
    N_it = 2
    MoreContours = FoundContours[:]
    for k in range(N_it):
        img_bin_bis = ndi.binary_erosion(img_bin, iterations=k+1)
        MoreContours += skm.measure.find_contours(img_bin_bis, 0.5)

    # Concatenating all contours from MoreContours gives a "thick" contour
    concat_contours = np.concatenate(MoreContours)
    # points = MultiPoint(concat_contours[:,::-1])
    # x_ch, y_ch = points.convex_hull.exterior.xy
    
    # Run the alpha shape
    # PixPerUm is the scale in Pix Per Um
    # 20x -> 2.2 ; 40x -> 4.5 ; 60x -> 9.2
    # -> One cell is more pixels at 60x than 20x
    
    # ALPHA sets the size of the "smoothing circle"
    # R = 1/ALPHA -> High alpha = high def; Low alpha = crude def
    
    # Need to check with other images
    
    R = PixPerUm # Radius of 1 µm
    ALPHA = 1/R
    alpha_shape = alphashape.alphashape(concat_contours[:,::-1], ALPHA)
    x_as, y_as = alpha_shape.exterior.xy
    Contour_alpha = np.array([y_as, x_as]).T
    
    inner_shape = alpha_shape.buffer(- buffer_um * PixPerUm)
    x_is, y_is = inner_shape.exterior.xy
    Contour_inner = np.array([y_is, x_is]).T
    
    Mask_inner = ufun.contour_to_mask((nY, nX), Contour_inner)
    
    
    if PLOT:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
        axes_f = axes.flatten()
        
        ax = axes_f[0]
        ax.set_aspect('equal', adjustable='box')
        ax.imshow(img_proj, cmap='gray')
        
        ax = axes_f[1]
        ax.set_aspect('equal', adjustable='box')
        ax.imshow(img_bin, cmap='gray')
        for c in FoundContours:
            ax.plot(c[:, 1], c[:, 0], lw=1)
        # ax.plot(x_ch, y_ch, lw=1)
            
        ax = axes_f[2]
        ax.set_aspect('equal', adjustable='box')
        for c in MoreContours:
            ax.plot(c[:, 1], c[:, 0], lw=1)
            
        ax = axes_f[3]
        ax.set_aspect('equal', adjustable='box')
        ax.imshow(Mask_inner, cmap='gray')
        
        for i in [0, 1, 2, 3]:
            ax = axes_f[i]
            ax.set_aspect('equal', adjustable='box')
            ax.plot(Contour_alpha[:, 1], Contour_alpha[:, 0], color='red', lw='0.75')
            ax.plot(Contour_inner[:, 1], Contour_inner[:, 0], color='cyan', lw='0.75')
            # ax.plot(x_is, y_is, lw='0.75')
        
        # list_geoms = list(alpha_shape.geoms)
        # for poly in list_geoms:
        #     x_as, y_as = poly.exterior.xy
        #     ax.plot(x_as, y_as, lw='0.75')
        
        plt.show()
        
    return(Contour_inner, Mask_inner)




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



def draw_circles(img, blobs, 
                 fig = None, ax = None):
    # this is the basic function that will be used to draw detected blobs 
    if fig == None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    ax.imshow(img, cmap='gray')
    for blob in blobs:
        y, x, radius = blob
        c = plt.Circle((x, y), radius*np.sqrt(2), color='white', linewidth=2, fill=False)
        ax.add_patch(c)

    # plt.show()  


def is_track_in_polygon(track, poly):
    xm = np.median(track[:, 1])
    ym = np.median(track[:, 2])
    point = MultiPoint([(ym, xm)])
    res = poly.contains(point)
    return(res)


def rawTracks_2_cleanTracks(rawTracks, dstDir, cleanTrackName,
                            Contour_cell, PixPerUm,
                            edge_buffer_cutoff_um = 2.5, nPoints_cuttoff = 30,
                            column_names = None,
                            RefImg = None, PLOT = False, SHOWPLOT = False, SAVEPLOT = False):
    
    if column_names is None:
        column_names = ['frame', 'x', 'y', 'particle']
    else:
        if len(column_names) != 4:
            raise ValueError("There should be 4 column names and they should be " + \
                   "roughly equivalent to: ['frame', 'x', 'y', 'particle']")
            
    Poly_cell = Polygon(shell=Contour_cell)
    Poly_inner_cell = Poly_cell.buffer(- edge_buffer_cutoff_um * PixPerUm)
    
    all_tracks = []
    for i, track in enumerate(rawTracks):
        nT = len(track)
        
        if is_track_in_polygon(track, Poly_inner_cell) and (nT >= nPoints_cuttoff):
            track = np.concat((track, np.ones((len(track[:,0]), 1), dtype=int) * (i+1)), axis = 1)
            track[:, 0] = track[:, 0].astype(int) + 1
            all_tracks.append(track)
            
    concat_tracks = np.concat(all_tracks, axis = 0)
    df = pd.DataFrame({column_names[k] : concat_tracks[:,k] for k in range(len(column_names))})
    df[column_names[0]] = df[column_names[0]].values.astype(int)
    df[column_names[3]] = df[column_names[3]].values.astype(int)
    df.to_csv(os.path.join(dstDir, cleanTrackName), index=False, sep = '\t')
    
    
    # Plot
    if PLOT:
        if not SHOWPLOT:
            plt.ioff()
        else:
            plt.ion()
            
        pm.setGraphicOptions(mode = 'screen')
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(cleanTrackName.split('.')[0])
        ColorList = pm.cL_Set21
        
        y_ic, x_ic = Poly_inner_cell.exterior.xy
        Contour_inner_cell = np.array([y_ic, x_ic]).T
        
        ax = axes[0]
        if not (RefImg is None):
            ax.imshow(RefImg, cmap='gray')
        
        
        ax = axes[1]
        if not (RefImg is None):
            ax.imshow(RefImg, cmap='gray')
        for k in range(len(rawTracks)):
            track = rawTracks[k]
            color = ColorList[k%len(ColorList)]
            ax.plot(track[:,1], track[:,2], ls='-', color=color, lw=0.25)
        ax.plot(Contour_cell[:,1], Contour_cell[:,0], ls='-', color='red', lw=1)
        ax.plot(Contour_inner_cell[:,1], Contour_inner_cell[:,0], ls='-', color='cyan', lw=1)
        
        ax = axes[2]
        if not (RefImg is None):
            ax.imshow(RefImg, cmap='gray')
            
        for k in range(len(all_tracks)):
            track = all_tracks[k]
            color = ColorList[k%len(ColorList)]
            ax.plot(track[:,1], track[:,2], ls='-', color=color, lw=0.25)
        
        if SHOWPLOT:
            plt.show()
        
        if SAVEPLOT:
            figName = '_'.join(cleanTrackName.split('_')[:-1]) + '_cleanTracks.png'
            figPath = os.path.join(dstDir, figName)
            fig.savefig(figPath, dpi=500, )
    
    if not SHOWPLOT:
        plt.ion()
    
    return(df)

# %%%% Pairwise MSD

def tri_to_short_edges(tri, points, thresh_d):
    # Extract all edges from each triangle
    edges = np.vstack([
        tri.simplices[:, [0, 1]],
        tri.simplices[:, [1, 2]],
        tri.simplices[:, [2, 0]]
    ])

    # Sort indices within each edge to make (i, j) and (j, i) identical
    edges = np.sort(edges, axis=1)

    # Remove duplicate edges
    edges = np.unique(edges, axis=0)

    # Convert to a Python list of pairs
    edges = np.array(edges)
    
    pairs = points[edges]
    dists = np.power(np.sum((pairs[:,1,:]-pairs[:,0,:])**2, axis=1), 0.5)
    idx_close_neighbours = (dists < thresh_d)
    edges_close_neighbours = edges[idx_close_neighbours]
    return(edges_close_neighbours, dists)


def get_pairs_for_TRanges_Delaunay(df, PixPerUm, FPS, Nframes,
                                  len_TRanges = 200, delta_TRanges = -1,
                                  dist_th_um = 5):
    df.frame = df.frame.astype(int)
    df.particle = df.particle.astype(int)
    dist_th = dist_th_um * PixPerUm
    
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
    
    dict_TRanges2particles = {f'{fi}_{ff}':{'pid':[], 'xm':[], 'ym':[]} \
                              for fi, ff in zip(FI, FF)}
        
    list_TRanges = [f'{fi}_{ff}' for fi, ff in zip(FI, FF)]
    list_Pairs = []
    
    PIDs = df.particle.unique()
    for pid in PIDs:
        pfi = np.min(df[df['particle'] == pid]['frame'].values) - 1
        pff = np.max(df[df['particle'] == pid]['frame'].values) - 1
        
        for fi, ff in zip(FI, FF):
            if (pfi <= fi) and (ff <= pff):
                df_p = df[df['particle'] == pid]
                df_p_TR = df_p[df_p['frame'].apply(lambda x : fi <= (x-1) < ff)]
                
                xm = np.median(df_p_TR['x'].values)
                ym = np.median(df_p_TR['y'].values)
                dict_TRanges2particles[f'{fi}_{ff}']['pid'].append(pid)
                dict_TRanges2particles[f'{fi}_{ff}']['xm'].append(xm)
                dict_TRanges2particles[f'{fi}_{ff}']['ym'].append(ym)
    
    for k, TRange in enumerate(list_TRanges):
        df_parts = pd.DataFrame(dict_TRanges2particles[TRange])
        XY = np.array([df_parts['xm'].values[:],
                       df_parts['ym'].values[:]]).T
        
        tri = Delaunay(XY)
        edges_short, _ = tri_to_short_edges(tri, XY, dist_th)
        close_pairs = df_parts['pid'].values[edges_short]
        
        list_Pairs.append(np.array(close_pairs))    
            
    return(list_TRanges, list_Pairs)




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


def get_relative_displacement_by_TRange(df, pairs, TRange):
    df.frame = df.frame.astype(int)
    df.particle = df.particle.astype(int)
    
    FI, FF = np.array(TRange.split('_')).astype(int)
    # T_array = np.arange(FI, FF)-1 
    T_array_shifted = np.arange(0, FF-FI)
    
    ids_in_pairs = np.unique(pairs.flatten())

    df_f = df
    df_f = df_f[df_f['frame'].apply(lambda x : FI <= (x-1) < FF)]
    df_f = df_f[df_f['particle'].apply(lambda x : x in ids_in_pairs)]
    
    # pair_2_pairId = {pairs[i] : i for i in range(len(pairs))}
    # pairs[pairId] = pair
    # pair_2_pairId[pair] = pairId
    
    # df_pairs = pd.DataFrame({'pair_id':[],'x':[],'y':[],'frame':[],})
    list_df_pairs = []
    
    for i in range(len(pairs)):
        pair = pairs[i]
        # i = pairId
        id1, id2 = pair
        idx1, idx2 = (df_f['particle']==id1), (df_f['particle']==id2)
        Xpair = df_f[idx2]['x'].values-df_f[idx1]['x'].values
        Ypair = df_f[idx2]['y'].values-df_f[idx1]['y'].values
        
        N = len(T_array_shifted)
        
        # df_pairs = pd.concat([df_pairs, pd.DataFrame(
        #                                              {'pair_id':np.ones(N, dtype=int)*i,
        #                                               'x':Xpair, 'y':Ypair,
        #                                               'frame':T_array_shifted,}
        #                                              )],
        #                      axis=0)
        list_df_pairs.append(pd.DataFrame({'pair_id':np.ones(N, dtype=int)*i,
                                           'x':Xpair, 'y':Ypair,
                                           'frame':T_array_shifted + 1,}
                                          ))
        
    df_pairs = pd.concat(list_df_pairs, axis=0)
    return(df_pairs)


# %%% Main functions


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
        
        # MSD
        Tracks = import_TrackMate_tracks(p)
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
        # res_imsd = tp.motion.imsd(df, SCALE, FPS).reset_index()
    
        #### Run msd
        res_emsd = tp.motion.emsd(df, SCALE, FPS, max_lagtime=30).reset_index()
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
                      label=f'D = {D:.2e} µm²\\s')
        
        parms, results = ufun.fitLineHuber(np.log(T), np.log(MSD), with_intercept = True)
        b, a = parms
        k_nl = a
        D_nl = np.exp(b)/4
        
        res_dict['id'].append(full_id)
        res_dict['pos_id'].append(pos_id)
        res_dict['cell_id'].append(cell_id)
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



def track_spots_in_cell(tifPath, dstDir):
    
    SCALE = SCALE_40X
    SIZE_UM = 1.5
    SIZE_PIX = SIZE_UM/SCALE
    print(SIZE_PIX)
    EQUALIZE = True
    N_ERODE = 40
    TOP_HAT = True
    MEDIAN_FILTER = True
    
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Get image and mask
    shape, dtype = ufun.tiff_inspect(tifPath)
    nT = shape[0]
    
    nT_subset = min(100, nT)
    subset_T = np.linspace(0, nT-1, num = nT_subset, dtype=int)
    image_subset = ufun.load_stack_region(tifPath, time_indices=subset_T, 
                                          x_slice=None, y_slice=None)
    image_subset = skm.util.img_as_float32(image_subset)
    
    inner_cell_contour = get_cell_inner_circle(image_subset, PLOT = False)
    mask = ufun.contour_to_mask([shape[1], shape[2]], inner_cell_contour)
    mask = ndi.binary_erosion(mask, iterations = N_ERODE)
    
    image = ufun.load_stack_region(tifPath, time_indices=None, 
                                   x_slice=None, y_slice=None)
    image = skm.util.img_as_float32(image)  
    # image = skm.util.img_as_ubyte(image)
    
    # image = image_subset
    # nT = nT_subset
    nT = 1
    
    image_pt = np.zeros_like(image)
    
    #### EQUALIZE
    if EQUALIZE:
        top = time.time()
        for t in range(nT):
            p1, p99 = np.percentile(image[t].flatten()[mask.flatten()], (1, 99))
            image_pt[t] = skm.exposure.rescale_intensity(image[t], in_range=(p1, p99))
        print(f'Equalize {time.time() - top:.1f} s')    
        
    else:
        for t in range(nT):
            image_pt[t] = image[t, :, :]
    
    #### CROP
    for t in range(nT):
        image_pt[t] = image_pt[t] * mask
            
    ax = axes[0]
    ax.imshow(image_pt[0], cmap='gray')
    
    ax = axes[1]
    ax.imshow(image_pt[0], cmap='gray')
    
    #### FILTER
    if MEDIAN_FILTER:
        top = time.time()
        for t in range(nT):
            k = 3
            image_pt[t] = cv2.medianBlur(image_pt[t], k)
        print(f'Median filter {time.time() - top:.1f} s') 
            
    ax = axes[2]
    ax.imshow(image_pt[0], cmap='gray')
    
    
    if TOP_HAT: # Applying the Top-Hat operation
        top = time.time()
        filterSize = (15, 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
        for t in range(nT):
            image_pt[t] = cv2.morphologyEx(image_pt[t], cv2.MORPH_TOPHAT, kernel)
            p1, p99 = np.percentile(image_pt[t].flatten()[mask.flatten()], (1, 99))
            image_pt[t] = skm.exposure.rescale_intensity(image_pt[t], in_range=(p1, p99))
        print(f'Top hat {time.time() - top:.1f} s') 
        
    ax = axes[3]
    ax.imshow(image_pt[0], cmap='gray')
    
    
    #### LOCATE
    list_df = []
    
    top = time.time()
    for t in range(nT):
        blobs = skm.feature.blob_dog(image_pt[t], min_sigma=7, max_sigma=7, 
                                     threshold=0.05, overlap=.5, exclude_border=True)
        X = blobs[:, 0]
        Y = blobs[:, 1]
        F = len(X) * [t]
        df = pd.DataFrame({'x':X, 'y':Y, 'frame':F})
        list_df.append(df)
        
        sigma1 = 0.35 * SIZE_PIX
        sigma2 = 1.6 * sigma1
        g1 = cv2.GaussianBlur(image_pt[t], (0,0), sigma1)
        g2 = cv2.GaussianBlur(image_pt[t], (0,0), sigma2)
        dog = g2.astype(np.float32) - g1.astype(np.float32)
    
    df_all = pd.concat(list_df)
    
    print(f'Blob_log {time.time() - top:.1f} s') 
    
    ax = axes[4]
    draw_circles(image_pt[t], blobs, fig = fig, ax = ax)
    
    ax = axes[5]
    ax.imshow(dog)
    
    
    #### LINK
    top = time.time()
    df_all = tp.link(df_all, 4, pos_columns=['y', 'x'], t_column='frame', memory=0, 
                 predictor=None, adaptive_stop=None, adaptive_step=0.95, 
                 neighbor_strategy=None, link_strategy=None, dist_func=None, to_eucl=None)
    print(f'Link {time.time() - top:.1f} s') 
    
    
    
    plt.show()
    out = (blobs, df_all,)
    
    return(out)
    
    # image_raw = skm.io.imread(tifPath)
    # image = skm.util.img_as_float32(image_raw)
    # if EQUALIZE:
    #     for t in range(image.shape[0]):
    #         p1, p99 = np.percentile(image[t].flatten()[mask.flatten()], (1, 99))
    #         image[t] = skm.exposure.rescale_intensity(image[t], in_range=(p1, p99))
    


# tifPath = "F:\\WorkingData\\26-06-19_FastAcq\\FilmBF_fastAcq_4000f_10Hz_C1.tif"
# tifPath = "C:\\Users\\josep\\Desktop\\Seafile\\AnalysisPulls\\" + \
#           "26-06-19_FastAcq\\FilmBF_fastAcq_4000f_10Hz_C1.tif"
tifPath = "C:\\Users\\josep\\Desktop\\Seafile\\AnalysisPulls\\" + \
          "26-06-10_Test-NileBlueYolk\\M1_40x-WI\\26-06-10_TestNileBlueYolk_C2_10fps_1min_L50p.tif"
dstDir = ""

# out, keypoints = TrackSpotsInCell(tifPath, dstDir)

# %%% Pipeline main functions


#### Function
def pretreat_image_for_TrackMate(tifPath, **kwargs):
    SETTINGS = {
        # 'SCALE' : SCALE_40X,
        # 'SIZE_UM' : 1.5,
        # 'SIZE_PIX' : 1.5/SCALE_40X,
        'EQUALIZE' : False,
        'N_ERODE' : 80,
        'TOP_HAT' : True,
        'MEDIAN_FILTER' : True,
        'SAVE_OUTPUT_IMAGE' : False,
        'RETURN_MASK' : False,
    }
    
    SETTINGS.update(kwargs)
    print(SETTINGS)
    
    # fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    # axes = axes.flatten()
    # iPlot = 0
    
    #### Get image and mask
    shape, dtype = ufun.tiff_inspect(tifPath)
    nT = shape[0]
    
    nT_subset = min(100, nT)
    subset_T = np.linspace(0, nT-1, num = nT_subset, dtype=int)
    image_subset = ufun.load_stack_region(tifPath, time_indices=subset_T, 
                                          x_slice=None, y_slice=None)
    image_subset = skm.util.img_as_float32(image_subset)
    
    inner_cell_contour = get_cell_inner_circle(image_subset, PLOT = False)
    mask = ufun.contour_to_mask([shape[1], shape[2]], inner_cell_contour)
    mask = ndi.binary_erosion(mask, iterations = SETTINGS['N_ERODE'])
    
    image = ufun.load_stack_region(tifPath, time_indices=None, 
                                   x_slice=None, y_slice=None)
    # image = skm.util.img_as_float32(image)  
    # image = skm.util.img_as_ubyte(image)
    
    # ax = axes[iPlot]
    # ax.imshow(image[0], cmap='gray')
    # iPlot += 1
    
    #### Pretreatments
    image_pt = np.zeros_like(image)
    
    #### i. EQUALIZE
    if SETTINGS['EQUALIZE']:
        top = time.time()
        for t in range(nT):
            p1, p99 = np.percentile(image[t].flatten()[mask.flatten()], (1, 99))
            image_pt[t] = skm.exposure.rescale_intensity(image[t], in_range=(p1, p99))
        print(f'Equalize {time.time() - top:.1f} s')    
        
    else:
        for t in range(nT):
            image_pt[t] = image[t, :, :]
    
    #### ii. CROP
    for t in range(nT):
        image_pt[t] = image_pt[t] * mask
            
    # ax = axes[iPlot]
    # ax.imshow(image_pt[0], cmap='gray')
    # iPlot += 1
    
    #### iii. FILTER
    if SETTINGS['TOP_HAT']: # Applying the Top-Hat operation
        top = time.time()
        filterSize = (15, 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
        for t in range(nT):
            image_pt[t] = cv2.morphologyEx(image_pt[t], cv2.MORPH_TOPHAT, kernel)
            p1, p99 = np.percentile(image_pt[t].flatten()[mask.flatten()], (1, 99))
            image_pt[t] = skm.exposure.rescale_intensity(image_pt[t], in_range=(p1, p99))
        print(f'Top hat {time.time() - top:.1f} s') 
        
    # ax = axes[iPlot]
    # ax.imshow(image_pt[0], cmap='gray')
    # iPlot += 1
    
    if SETTINGS['MEDIAN_FILTER']:
        top = time.time()
        for t in range(nT):
            k = 3
            image_pt[t] = cv2.medianBlur(image_pt[t], k)
        print(f'Median filter {time.time() - top:.1f} s') 
            
    # ax = axes[iPlot]
    # ax.imshow(image_pt[0], cmap='gray')
    # iPlot += 1
    
    # image_to_save = image_pt[:nT]
    # skm.io.imsave(dstDir + pretreatedName, image_to_save)
    # plt.show()
    
    image_output = skm.util.img_as_ubyte(image_pt)
    
    if SETTINGS['SAVE_OUTPUT_IMAGE']:
        srcDir, tifName = os.path.split(tifPath)
        tifRoot = tifName.split('.')[0]
        outputName = tifRoot + '_pretreated.tif'
        outputPath = os.path.join(srcDir, outputName)
        skm.io.imsave(outputPath, image_output)
    
    output = (image_output, )
    
    if SETTINGS['RETURN_MASK']:
        output += (mask, )
    
    if len(output) == 1:
        output == output[0]
        
    return(output)



def runTrackMate(tif_file, xmlPath, Pix_Per_Um,
                 IMG_UNITS = 'PIX',
                 RADIUS_UM = 1.0, 
                 THRESH_SPOT_QLT = 1.0,
                 THRESH_LINK_UM = 0.2, 
                 THRESH_MIN_DURATION = 40):
    
    imp = ij.py.to_imageplus(tif_file)
    dims = imp.getDimensions() # default order: XYCZT
    print(dims)
    
    if dims[4] == 1:
        print('need to change order')
        imp.setDimensions(dims[4], dims[3], dims[2])
    
    
    print(f" dims: {tif_file.dims if hasattr(tif_file, 'dims') else 'N/A'}")
    
    File = sj.jimport('java.io.File')
    Model = sj.jimport('fiji.plugin.trackmate.Model')
    Settings = sj.jimport('fiji.plugin.trackmate.Settings')
    TrackMate = sj.jimport('fiji.plugin.trackmate.TrackMate')
    FeatureFilter = sj.jimport('fiji.plugin.trackmate.features.FeatureFilter')
    LAPUtils = sj.jimport('fiji.plugin.trackmate.tracking.jaqaman.LAPUtils')
    # SelectionModel = sj.jimport('fiji.plugin.trackmate.SelectionModel')
    Logger = sj.jimport('fiji.plugin.trackmate.Logger')
    # DisplaySettingsIO = sj.jimport('fiji.plugin.trackmate.gui.displaysettings.DisplaySettingsIO')
    # HyperStackDisplayer = sj.jimport('fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer')
    
    LogDetectorFactory = sj.jimport('fiji.plugin.trackmate.detection.LogDetectorFactory')
    # DogDetectorFactory = sj.jimport('fiji.plugin.trackmate.detection.DogDetectorFactory')
    SparseLAPTrackerFactory = sj.jimport('fiji.plugin.trackmate.tracking.jaqaman.SparseLAPTrackerFactory')
    
    TrackAnalyzerProvider = sj.jimport('fiji.plugin.trackmate.providers.TrackAnalyzerProvider')
    FeatureFilter = sj.jimport('fiji.plugin.trackmate.features.FeatureFilter')
    
    # TmXmlWriter = sj.jimport('fiji.plugin.trackmate.io.TmXmlWriter')
    # CSVExporter = sj.jimport('fiji.plugin.trackmate.io.CSVExporter')
    # TrackTableView = sj.jimport('fiji.plugin.trackmate.visualization.table.TrackTableView')
    ExportTracksToXML = sj.jimport('fiji.plugin.trackmate.action.ExportTracksToXML')
    
    # from fiji.plugin.trackmate.io import TmXmlWriter
    # from fiji.plugin.trackmate.io import CSVExporter
    # from fiji.plugin.trackmate.visualization.table import TrackTableView
    # from fiji.plugin.trackmate.action import ExportTracksToXML
    
    
    # Initiate
    model = Model()
    # model.setLogger(Logger.IJ_LOGGER)
    model.setLogger(Logger.DEFAULT_LOGGER)
    
    settings = Settings(imp)
    
    # Convert thresholds from Um to Pix if necessary
    if IMG_UNITS == 'PIX':
        RADIUS = Pix_Per_Um * RADIUS_UM
        THRESH_LINK = Pix_Per_Um * THRESH_LINK_UM
    elif IMG_UNITS == 'UM':
        RADIUS = RADIUS_UM
        THRESH_LINK = THRESH_LINK_UM
    else:
        raise ValueError("Setting variable IMG_UNITS should be equal to 'PIX' or 'UM'")
    
    # Configure detector
    settings.detectorFactory = LogDetectorFactory()
    settings.detectorSettings = {
        'DO_SUBPIXEL_LOCALIZATION' : True,
        'RADIUS' : RADIUS,
        'TARGET_CHANNEL': ij.py.to_java(1),
        'DO_MEDIAN_FILTERING': False,
        'THRESHOLD': THRESH_SPOT_QLT # 0.01
    }
    
    # Configure tracker   
    settings.trackerFactory = SparseLAPTrackerFactory()
    settings.trackerSettings = LAPUtils.getDefaultSegmentSettingsMap()
    settings.trackerSettings['LINKING_MAX_DISTANCE'] = THRESH_LINK
    settings.trackerSettings['ALLOW_GAP_CLOSING'] = False
    settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = 1.0
    settings.trackerSettings['MAX_FRAME_GAP'] = ij.py.to_java(0)
    
    # Configure filtering
    
    # settings.addAllAnalyzers()
    trackAnalyzerProvider = TrackAnalyzerProvider()
    for key in trackAnalyzerProvider.getKeys():
        print(key)
        settings.addTrackAnalyzer(trackAnalyzerProvider.getFactory(key))
    
    filter1 = FeatureFilter('TRACK_DURATION', THRESH_MIN_DURATION, True)
    settings.addTrackFilter(filter1)
    
    # Run the model
    trackmate = TrackMate(model, settings)
    ok = trackmate.checkInput()
    if not ok:
        sys.exit(str(trackmate.getErrorMessage()))
    
    ok = trackmate.process()
    if not ok:
        sys.exit(str(trackmate.getErrorMessage()))
    
    model.getLogger().log('Found ' + str(model.getTrackModel().nTracks(True)) + ' tracks.')
    
    simple_xml_file = File(xmlPath)
    ExportTracksToXML.export(model, settings, simple_xml_file)
    
    print('\nDone!')


# runTrackMate(imp, xmlPath)

# %%% Function of the whole pipeline + test

def pretreatAndTrack(tifPath, dstDir):
    srcDir, tifName = os.path.split(tifPath)
    xmlName = tifName.split('.')[0] + '_PyTracks.xml'
    xmlPath = os.path.join(srcDir, xmlName)
    PtImage, mask = pretreat_image_for_TrackMate(tifPath, 
                                        N_ERODE = 50,
                                        SAVE_OUTPUT_IMAGE = True,
                                        RETURN_MASK = True)
    
    
    
    tif_file = ij.py.to_java(PtImage)
    # tif_file = ij.io().open(srcDir + tifPtName)
    runTrackMate(tif_file, xmlPath)
    
    Tracks = import_TrackMate_tracks(xmlPath)
    
    I0 = ufun.load_stack_region(tifPath, time_indices=[0])[0]
    Co = ufun.mask_to_contour(mask, keep_only_longest_contour = True)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ax = axes[0]
    ax.imshow(I0, cmap='gray')
    
    ax = axes[1]
    ax.imshow(PtImage[0], cmap='gray')
    
    ax = axes[2]
    ax.imshow(I0, cmap='gray')
    ax.plot(Co[:,1], Co[:,0], ls='-', color='darkorange', lw=1.5)
    CL = pm.cL_Set21
    for k in range(len(Tracks)):
        track = Tracks[k]
        color = CL[k%len(CL)]
        ax.plot(track[:,1], track[:,2], ls='-', color=color, lw=0.25)
    
    plt.show()
    
    return(Tracks)
    

def pretreatAndTrack_CropedYolk(tifPath, xmlName, dstDir,
                                PLOT = False, SAVEPLOT = False):
    srcDir, tifName = os.path.split(tifPath)
    # xmlName = tifName.split('.')[0] + '_PyTracks.xml'
    xmlPath = os.path.join(dstDir, xmlName)
    
    shape, dtype = ufun.tiff_inspect(tifPath)
    nT = shape[0]    
    image = ufun.load_stack_region(tifPath, time_indices=None, 
                                   x_slice=None, y_slice=None)
    
    #### Pretreatments
    for t in range(nT):
        k = 3
        image[t] = cv2.medianBlur(image[t], k)
    
    
    tif_file = ij.py.to_java(image)
    runTrackMate(tif_file, xmlPath)
    
    
    if PLOT:
        pm.setGraphicOptions(mode = 'screen')
        Tracks = import_TrackMate_tracks(xmlPath)
        I0 = ufun.load_stack_region(tifPath, time_indices=[0])[0]
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle('_'.join(tifName.split('_')[:5]))
        
        ax = axes[0]
        ax.imshow(I0, cmap='gray')
        
        ax = axes[1]
        ax.imshow(image[0], cmap='gray')
        
        ax = axes[2]
        ax.imshow(I0, cmap='gray')
        CL = pm.cL_Set21
        for k in range(len(Tracks)):
            track = Tracks[k]
            color = CL[k%len(CL)]
            ax.plot(track[:,1], track[:,2], ls='-', color=color, lw=0.25)
    
        plt.show()
        
        if SAVEPLOT:
            figName = tifName.split('.')[0] + '_FigTracks.png'
            figPath = os.path.join(dstDir, figName)
            fig.savefig(figPath, dpi=500, )
    
    return(Tracks)


def pretreat_and_track_NbYolk(tifPath, rawTrackName, dstDir,
                              Pix_Per_Um, Dict_TrackMate_Settings = None,
                              Mask_cell = None, mask_buffer_um = 0.0,
                              return_tracks = False,
                              PLOT = False, SHOWPLOT = False, SAVEPLOT = False):
    
    
    
    srcDir, tifName = os.path.split(tifPath)
    rawTrackPath = os.path.join(dstDir, rawTrackName)
    
    shape, dtype = ufun.tiff_inspect(tifPath)
    nT = shape[0]
    
    image = ufun.load_stack_region(tifPath, time_indices=None, 
                                   x_slice=None, y_slice=None)
    
    # nT = 100
    # image = ufun.load_stack_region(tifPath, time_indices=range(0, 100), 
    #                                x_slice=None, y_slice=None)
    
    if Mask_cell is None:
        pass
    else:
        mask_buffer_pix = round(mask_buffer_um * Pix_Per_Um)
        Mask_cell = ndi.binary_erosion(Mask_cell, iterations=mask_buffer_pix)
        image = image * Mask_cell
        
        
    # Pretreatments
    for t in range(nT):
        k = 3
        image[t] = cv2.medianBlur(image[t], k)
    
    image_0 = image[0,:,:]
    tif_file = ij.py.to_java(image)
    del(image)
      
    # Update TrackMate settings
    Dict_TrackMate_Settings_DEFAULTS = {
        'IMG_UNITS' : 'PIX',
        'RADIUS_UM' : 1.0, 
        'THRESH_SPOT_QLT' : 1.0,
        'THRESH_LINK_UM' : 0.2, 
        'THRESH_MIN_DURATION' : 40,
        }
    if Dict_TrackMate_Settings is None:
        Dict_TrackMate_Settings = Dict_TrackMate_Settings_DEFAULTS
    else:
        Dict_TrackMate_Settings_DEFAULTS.update(Dict_TrackMate_Settings)
        Dict_TrackMate_Settings = Dict_TrackMate_Settings_DEFAULTS
    print(pm.GREEN + 'Settings: ' + pm.NORMAL, Dict_TrackMate_Settings)
    
    
    # Run TrackMate
    runTrackMate(tif_file, rawTrackPath, Pix_Per_Um,
                 **Dict_TrackMate_Settings)
    
    
    # Plot
    if PLOT:
        if not SHOWPLOT:
            plt.ioff()
        else:
            plt.ion()
            
        pm.setGraphicOptions(mode = 'screen')
        Tracks = import_TrackMate_tracks(rawTrackPath)
        image_raw_0 = ufun.load_stack_region(tifPath, time_indices=[0])[0]
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle('_'.join(tifName.split('_')[:5]))
        
        ax = axes[0]
        ax.imshow(image_raw_0, cmap='gray')
        
        ax = axes[1]
        ax.imshow(image_0, cmap='gray')
        
        ax = axes[2]
        ax.imshow(image_raw_0, cmap='gray')
        CL = pm.cL_Set21
        for k in range(len(Tracks)):
            track = Tracks[k]
            color = CL[k%len(CL)]
            ax.plot(track[:,1], track[:,2], ls='-', color=color, lw=0.25)
        
        if SHOWPLOT:
            plt.show()
        
        if SAVEPLOT:
            figName = tifName.split('.')[0] + '_rawTracks.png'
            figPath = os.path.join(dstDir, figName)
            fig.savefig(figPath, dpi=500, )
    
    if not SHOWPLOT:
        plt.ion()
    
    
    # Return tracks
    if return_tracks:
        if not PLOT:
            Tracks = import_TrackMate_tracks(rawTrackPath)
        else:
            pass
        
        return(Tracks)



# %% II. Dynamic Differential Microscopy

# %%% DDM Analysis

class ImageStack(object):
    """
    A stack of images on disk with a name pattern like 'mydir/myfile_t{:03d}.tif'
    """
    
    def __init__(self, path):
        """The numbering can start at 0 or 1"""
        self.path = path
        self.t0 = 0
        # get the images shape while checking that the last image do exist
        self.shape, self.type = ufun.tiff_inspect(path)
        self.Nbimages = self.shape[0]
        
        # #for some monochrome image format, imread makes 4 channels out of one
        # self.enforceMono = len(self.shape)>2
        # if self.enforceMono:
        #     self.shape = self.shape[:-1]
            
    def __len__(self):
        return(self.Nbimages)
            
    def __getitem__(self, t):
        """
        returns the image at time t
        """
        if t<0: 
            t = len(self)+t
        assert t-self.t0 < self.Nbimages
        
        # im = imread(self.pattern.format(t + self.t0))
        im = ufun.load_stack_region(self.path, time_indices=[t])[0]
        # if self.enforceMono:
        #     im = im[...,0]
        return(im)

#### DDM step 1 - spectrumDiff()
# Define the function at the heart of DDM:
# $$\left|\widehat{\Delta I}\right|^2(\vec{q}, t, \Delta t) = \left|\mathcal{F}\left[I(\vec{r}, t+\Delta t) - I(\vec{r}, t)\right]\right|^2$$
# where $I(\vec{r}, t)$ is the intensity of the image at time $t$ at position $\vec{r}$ and $\mathcal{F}$ is the Fourier transform.

def spectrumDiff(im0, im1):
    """
    Compute the squared modulus of the 2D Fourier Transform of 
    the difference between im0 and im1
    """
    # diff = im1-im0.astype(float)
    # FT = np.abs(np.fft.fft2(diff))
    # FT = skm.util.img_as_ubyte(FT)
    # return(FT**2)
    return(np.abs(np.fft.fft2(im1-im0.astype(float)))**2)
    



#### DDM step 2 - timeAveraged()

# A single couple of images is not enough to get good statistics. 
# For a fixed time interval `dt`, we take at most `maxNCouples` 
# couples of images evenly spead in the available range of times.


def timeAveraged(stack, dt, maxNCouples=50):
    """
    Does at most maxNCouples spectreDiff 
    on regularly spaced couples of images. 
    Separation within couple is dt.
    """
    
    #Spread initial times over the available range
    increment = max([(len(stack)-dt)/maxNCouples, 1])
    # print(int(increment))
    initialTimes = np.arange(0, len(stack)-dt, increment, dtype=int)
    
    #perform the time average
    avgFFT = np.zeros(stack.shape[1:])
    for t in initialTimes:
        avgFFT += spectrumDiff(stack[t], stack[t+dt])
        
    return(avgFFT / len(initialTimes))


#### DDM step 3 - RadialAverager()

# Define a class able to perform radial averaging of FFT spectra. 
# For the sake of performance, a RadialAverager instance has a fixed shape 
# and can only process spectra of this shape. This is not a limitation since 
# all the images in a stack do have the same shape.

# Also, since some spectra have anomalously bright cross, we do not take this line 
# and this column into account.

class RadialAverager(object):
    """Radial average of a 2D array centred on (0,0), like the result of fft2d."""
    def __init__(self, shape):
        """A RadialAverager instance can process only arrays of a given shape, fixed at instanciation."""
        assert len(shape)==2
        #matrix of distances
        self.dists = np.sqrt(np.fft.fftfreq(shape[0])[:,None]**2 +  np.fft.fftfreq(shape[1])[None,:]**2)
        #dump the cross
        self.dists[0] = 0
        self.dists[:, 0] = 0
        #discretize distances into bins
        self.bins = np.arange(max(shape)/2 + 1)/float(max(shape))
        #number of pixels at each distance
        self.hd = np.histogram(self.dists, self.bins)[0]
    
    def __call__(self, im):
        """Perform and return(the radial average of the specrum 'im'"""
        assert im.shape == self.dists.shape
        hw = np.histogram(self.dists, self.bins, weights=im)[0]
        return(hw/self.hd)

#### DDM step 4 - logSpaced()

# We won't perform all those steps for every time interval, it would be 
# too time consuming. So we sample time intervals logarithmically.

def logSpaced(L, pointsPerDecade=15):
    """Generate an array of log spaced integers smaller than L"""
    nbdecades = np.log10(L)
    # print(nbdecades, nbdecades * pointsPerDecade)
    return(np.unique(np.logspace(
        start=0, stop=nbdecades, 
        num=int(nbdecades * pointsPerDecade), 
        base=10, endpoint=False
        ).astype(int)))

#### DDM step 5 - ddm()

# Finally, we put everything together to obtain 
# $$\mathcal{D}(\Delta t,q) = \left\langle \left|\widehat{\Delta I}\right|^2 (\vec{q}, t, \Delta t)\right\rangle$$ 
# were $\langle.\rangle$ is the average on initial time $t$ and the orientation of $\vec{q}$.

# Since this can be a long operation, we add a counter

def ddm(stack, idts, maxNCouples=100):
    """Perform time averaged and radial averaged DDM for given time intervals.
    Returns DDM"""
    ra = RadialAverager(stack.shape[1:])
    DDM = np.zeros((len(idts), len(ra.hd)))
    N = len(idts)
    progress_step = N/100
    for i, idt in enumerate(idts):
        DDM[i] = ra(timeAveraged(stack, idt, maxNCouples))
        if i//progress_step > (i-1)//progress_step:
            j = int(i//progress_step)
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*(j//5), j))
            sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*20, 100))
    return(DDM)

# %%% Post-treatment

#### DDM step 6 - merge different freqs

def mergeDDM(DDMs, dts, frequencies):
    # Then we merge the two sets of data by scaling the data at 4 Hz so that both values 
    # at 0.25 s are equal. Finally, we average the values of the curves at 4 Hz and 400 Hz 
    # in the first third of their overlap interval.
    
    # Find the closest time at 400Hz to the smallest time at 4Hz
    boundary = np.argmin(np.abs(dts[0] - dts[1][0]))
    
    # find the first third of their overlap
    overlap0 = (len(DDMs[0])-1 - boundary)//3
    overlap1 = np.argmin(np.abs(dts[1] - dts[0][boundary+overlap0]))
    
    # Rescale the value of radial average at 4 Hz according to the value at t=boundary for 400Hz
    overlap_full_1 = (len(DDMs[0])-1 - boundary)//2
    overlap_full_2 = np.argmin(np.abs(dts[1] - dts[0][boundary+overlap_full_1]))
    DDMs[1] *= DDMs[0][boundary+overlap_full_1] / DDMs[1][overlap_full_2]
    
    # interpolate on this first third the DDM at 4Hz on the times at 400Hz
    interpolated = np.transpose([
        np.interp(
            dts[0][boundary:boundary+overlap0],
            dts[1][:overlap1], 
            v)
        for v in DDMs[1][:overlap1].T])
    
    #do a smooth transition on this first third
    x = ((dts[0][boundary:boundary+overlap0]-dts[0][boundary])/(dts[0][boundary+overlap0]-dts[0][boundary]))[:,None]
    transition = (1-x) * DDMs[0][boundary:boundary+overlap0] + x * interpolated
    # Merge 400Hz, transition and 4Hz
    dtMerge = np.concatenate([dts[0][:boundary+overlap0], dts[1][overlap1:]])
    DDMMerge = np.concatenate([DDMs[0][:boundary], transition, DDMs[1][overlap1:]], axis=0)
    return(DDMMerge, dtMerge)
    
