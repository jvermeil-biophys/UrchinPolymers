# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 14:06:30 2026

@author: Joseph
"""


# %% Imports

import os
# import re
# import cv2
# import logging
import tifffile
# import traceback

import numpy as np
import pandas as pd
# import pyjokes as pj
import skimage as skm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

import shapely
from shapely.ops import polylabel
from shapely.plotting import plot_polygon, plot_points # , plot_line

# import scipy
import scipy.ndimage as ndi
from scipy import signal, stats #, optimize, interpolate, 

import PlotMaker as pm
import UrchinPaths as up
import UtilityFunctions as ufun


# %% Functions

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

     
def argmedian(x):
    """
    Find the argument of the median value in array x.
    """
    if len(x)%2 == 0:
        x = x[:-1]
    return(np.argpartition(x, len(x) // 2)[len(x) // 2])


def tiff_inspect(filepath):
    with tifffile.TiffFile(filepath) as tif:
        series = tif.series[0]  # first series
        shape = series.shape
        dtype = series.dtype
    return(shape, dtype)

def load_stack_region(filepath, time_indices=None, x_slice=None, y_slice=None):
    """
    Load a cropped region of a 3D TIFF (X, Y, time) with minimal memory usage.

    Parameters
    ----------
    filename : str
        Path to the TIFF file.
    time_indices : list[int] or slice, optional
        Which time points to load. Default = all.
    x_slice : slice, optional
        Cropping along X dimension (cols).
    y_slice : slice, optional
        Cropping along Y dimension (rows).

    Returns
    -------
    numpy.ndarray
        Cropped stack with shape (T, Y, X).
    """
    
    with tifffile.TiffFile(filepath) as tif:
        series = tif.series[0]   # the first image series
        pages = series.pages
        firstFrame = pages[0]
        if time_indices is None:
            time_indices = range(0, len(pages))
        if x_slice is None:
            x_slice = slice(0, firstFrame.shape[1])
        if y_slice is None:
            y_slice = slice(0, firstFrame.shape[0])

        # Collect requested frames without loading everything
        cropped_stack = []
        for i in time_indices:
            page = pages[i]
            arr = page.asarray()[y_slice, x_slice]  # crop directly
            cropped_stack.append(arr)

        return(np.stack(cropped_stack, axis=0))
    
def get_largest_object(img, mode = 'dark', out_type = 'contour'):
    th = skm.filters.threshold_otsu(img)
    if mode == 'dark':
        img_bin = (img < th)
    elif mode == 'bright':
        img_bin = (img > th)
    img_label, num_features = ndi.label(img_bin)
    
    df = pd.DataFrame(skm.measure.regionprops_table(img_label, img, properties = ['label', 'area']))
    df = df.sort_values(by='area', ascending=False)
    i_label = df.label.values[0]
    img_bin_object = (img_label == i_label)
    img_bin_object = ndi.binary_fill_holes(img_bin_object)
    if out_type == 'contour':
        FoundContours = skm.measure.find_contours(img_bin_object, 0.5)
        if len(FoundContours) == 1:
            contour = FoundContours[0]
        else:
            L = [len(c) for c in FoundContours]
            im = np.argmax(L)
            contour = FoundContours[im]    
        output = contour
    elif out_type == 'mask':
        output = img_bin_object
    return(output)
    
def find_cell_inner_circle(img, plot=False, k_th = 1.0):
    """
    On a picture of a cell in fluo, find the approximate position of the cell.
    The idea is to fit the largest circle which can be contained in a mask of the cell.
    It uses the library shapely, and more precisely the function polylabel,
    to find the "pole of inaccessibility" of the cell mask.
    See : https://github.com/mapbox/polylabel and https://sites.google.com/site/polesofinaccessibility/
    Interesting topic !
    """
    
    th1 = skm.filters.threshold_isodata(img) * k_th
    img_bin = (img < th1)
    img_bin = ndi.binary_opening(img_bin, iterations = 2)
    img_label, num_features = ndi.label(img_bin)
    df = pd.DataFrame(skm.measure.regionprops_table(img_label, img, properties = ['label', 'area']))
    df = df.sort_values(by='area', ascending=False)
    i_label = df.label.values[0]
    img_rawCell = (img_label == i_label)
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
    
    if plot:
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



def viterbi_path_finder(treillis):
    """
    To understand the idea, see : https://www.youtube.com/watch?v=6JVqutwtzmo
    """
    
    def node2node_distance(node1, node2):
        """Define a distance between nodes of the treillis graph"""
        d = (np.abs(node1['pos'] - node2['pos']))**2 #+ np.abs(node1['val'] - node2['val'])
        return(d)
    
    N_row = len(treillis)
    for i in range(1, N_row):
        current_row = treillis[i]
        previous_row = treillis[i-1]
        
        if len(current_row) == 0:
            current_row = [node.copy() for node in previous_row]
            treillis[i] = current_row
            
        for node1 in current_row:
            costs = []
            for node2 in previous_row:
                c = node2node_distance(node1, node2) + node2['accumCost']
                costs.append(c)
            best_previous_node = np.argmin(costs)
            accumCost = np.min(costs)
            node1['previous'] = best_previous_node
            node1['accumCost'] = accumCost
    
    final_costs = [node['accumCost'] for node in treillis[-1]]
    i_best_arrival = np.argmin(final_costs)
    best_path = [i_best_arrival]
    
    for i in range(N_row-1, 0, -1):
        node = treillis[i][best_path[-1]]
        i_node_predecessor = node['previous']
        best_path.append(i_node_predecessor)
    
    best_path = best_path[::-1]
    nodes_list = [treillis[k][best_path[k]] for k in range(len(best_path))]
            
    return(best_path, nodes_list)

#### Test of ViterbiPathFinder

# A "treillis" is a type of graph
# My understanding of it is the following
# > It has N "rows" of nodes
# > Each row has M nodes
# > Each node of a given row are linked with all the nodes of the row before
#   as well as with all the nodes of the row after
#   by vertices of different weights.
# > The weights of the vertices correspond to the cost computed by the home-made cost function.
# In the case of the Viterbi tracking, the goal is to go through the treillis graph, row after row,
# and find the best path leading to each node of the current row.
# At the end of the graph, the node with the lowest "accumCost" is the best arrival point.
# One just have to back track from there to find the best path.

# treillis = []

# node_A1 = {'pos':0,'val':10,'t':0,'previous':-1,'accumCost':0}
# row_A = [node_A1]

# node_B1 = {'pos':-35,'val':10,'t':1,'previous':-1,'accumCost':0}
# node_B2 = {'pos':10,'val':10,'t':1,'previous':-1,'accumCost':0}
# node_B3 = {'pos':5,'val':10,'t':1,'previous':-1,'accumCost':0}
# row_B = [node_B1, node_B2, node_B3]

# node_C1 = {'pos':-20,'val':10,'t':2,'previous':-1,'accumCost':0}
# node_C2 = {'pos':40,'val':10,'t':2,'previous':-1,'accumCost':0}
# row_C = [node_C1, node_C2]

# node_D1 = {'pos':-35,'val':10,'t':3,'previous':-1,'accumCost':0}
# node_D2 = {'pos':15,'val':10,'t':3,'previous':-1,'accumCost':0}
# row_D = [node_D1, node_D2]

# row_E = [node.copy() for node in row_A]

# treillis = [row_A, row_B, row_C, row_D, row_E]

# ViterbiPathFinder(treillis)

def viterbi_edge(warped, Rc, inPix, outPix, blur_parm, relative_height_virebi):
    """
    Wrapper around ViterbiPathFinder
    Use the principle of the Viterbi algorithm to smoothen the contour of the cell on a warped image.
    To understand the idea, see : https://www.youtube.com/watch?v=6JVqutwtzmo
    """
    # Create the TreillisGraph for Viterbi tracking
    Angles = np.arange(0, 360)
    AllPeaks = []
    TreillisGraph = []
    warped_filtered = skm.filters.gaussian(warped, sigma=(1, blur_parm), mode='wrap')
    inBorder = round(Rc) - inPix
    outBorder = round(Rc) + outPix
    for a in Angles:
        profile = warped_filtered[a, :] - np.min(warped_filtered[a, inBorder:outBorder])
        peaks, peaks_props = signal.find_peaks(profile[inBorder:outBorder], 
                                                # width = 4,
                                                height = relative_height_virebi*np.max(profile[inBorder:outBorder]))
        AllPeaks.append(peaks + inBorder)
        TreillisRow = [{'angle':a, 'pos':p+inBorder, 'val':profile[p+inBorder], 'previous':0, 'accumCost':0} for p in peaks]
        TreillisGraph.append(TreillisRow)
        
    for k in range(len(TreillisGraph)):
        row = TreillisGraph[k]
        if len(row) == 0:
            TreillisGraph[k] = [node.copy() for node in TreillisGraph[k-1]]
    
    # Get a reasonable starting point
    try:
        starting_candidates = []
        for R in TreillisGraph:
            if len(R) == 1:
                starting_candidates.append(R[0])
        # pos_start = np.median([p['pos'] for p in starting_candidates])
        # starting_i = argmedian([p['pos'] for p in starting_candidates])
        # starting_peak = starting_candidates[starting_i]
        starting_peak = starting_candidates[argmedian([p['pos'] for p in starting_candidates])]
        starting_i = starting_peak['angle']
        
    except:
        starting_candidates = []
        for R in TreillisGraph:
            if len(R) <= 3:
                R_pos = [np.abs(p['pos']-Rc) for p in R]
                i_best = np.argmin(R_pos)
                starting_candidates.append(R[i_best])
        starting_peak = starting_candidates[argmedian([p['pos'] for p in starting_candidates])]
        starting_i = starting_peak['angle']
    
    # Pretreatment of the TreillisGraph for Viterbi tracking
    TreillisGraph = TreillisGraph[starting_i:] + TreillisGraph[:starting_i] # Set the starting point
    TreillisGraph.append([node.copy() for node in TreillisGraph[0]]) # Make the graph cyclical
    
    # Viterbi tracking
    best_path, nodes_list = viterbi_path_finder(TreillisGraph)
    
    edge_viterbi = [p['pos'] for p in nodes_list[:-1]]
    edge_viterbi = edge_viterbi[1-starting_i:] + edge_viterbi[:1-starting_i] # Put it back in order
    return(edge_viterbi)


def warpXY(X, Y, Xcw, Ycw):
    """
    X, Y is the point of interest
    Xcw, Ycw are the coordinates of the center of the warp
    skimage.transform.warp_polar
    """
    R = ((Y-Ycw)**2 + (X-Xcw)**2)**0.5
    angle = (np.arccos((X-Xcw)/R * np.sign(Y-Ycw+0.001)) + np.pi*((Y-Ycw)<0)) * 180/np.pi  # Radian -> Degree
    return(R, angle)

def unwarpRA(R, A, Xcw, Ycw):
    """
    R, A is the point of interest
    Xcw, Ycw are the coordinates of the center of the warp
    skimage.transform.warp_polar
    """
    X = Xcw + (R*np.cos(A*np.pi/180)) # Degree -> Radian
    Y = Ycw + (R*np.sin(A*np.pi/180)) # Degree -> Radian
    return(X, Y)

def segment_single_cell(img, starting_contour = [], N_it_viterbi = 2, PLOT = False):
    #### Settings
    inPix_set, outPix_set = 20, 20
    blur_parm = 1
    relative_height_virebi = 0.2
    warp_radius = round(60/SCALE)
    
    nY, nX = img.shape
    Angles = np.arange(0, 360, 1)
       
    if len(starting_contour) == 0:
        mask = get_largest_object(img, mode = 'dark', out_type = 'mask').astype(bool)
        mask = ndi.binary_dilation(mask, iterations=8)
    
        #### 2.0 Locate cell
        (Yc, Xc), Rc = find_cell_inner_circle(img * mask, plot=False, k_th = 1.0)
        if Rc < (10/SCALE):
            (Xc, Yc, Rc) = (nX//2, nY//2, 45/SCALE)
        Yc0, Xc0, Rc0 = round(Yc), round(Xc), round(Rc)
        
        #### 2.1 Warp
        warped = skm.transform.warp_polar(img, center=(Yc, Xc), radius=warp_radius, #Rc*1.2, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        # max_values = np.max(warped[:,Rc0-inPix_set:Rc0+outPix_set+1], axis=1)
            
        #### 2.2 Viterbi Smoothing
        edge_viterbi = viterbi_edge(warped, Rc0, inPix_set, outPix_set, blur_parm, relative_height_virebi)
        edge_viterbi_unwarped = np.array(unwarpRA(np.array(edge_viterbi), Angles, Xc, Yc)) # X, Y
        starting_contour = np.array([edge_viterbi_unwarped[1], edge_viterbi_unwarped[0]]).T
        
    # N_it_viterbi = 1
    viterbi_contour = starting_contour
    for i in range(N_it_viterbi):
        # x.0 Locate cell
        (Yc, Xc), Rc = ufun.fitCircle(viterbi_contour, loss = 'huber')        
        Yc0, Xc0, Rc0 = round(Yc), round(Xc), round(Rc)
        # x.1 Warp
        warped = skm.transform.warp_polar(img, center=(Yc, Xc), radius=warp_radius, #Rc*1.4, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        warped = skm.util.img_as_uint(warped)
        w_ny, w_nx = warped.shape
        Angles = np.arange(0, 360, 1)
        max_values = np.max(warped[:,Rc0-inPix_set:Rc0+outPix_set+1], axis=1)
        inPix, outPix = 10, 10
        # x.2 Viterbi Smoothing
        edge_viterbi = viterbi_edge(warped, Rc0, inPix, outPix, blur_parm, relative_height_virebi)
        edge_viterbi_unwarped = unwarpRA(np.array(edge_viterbi), Angles, Xc, Yc)
        viterbi_contour = np.array([edge_viterbi_unwarped[1], edge_viterbi_unwarped[0]]).T
    
    if PLOT:
        fig, axes = plt.subplots(2, 2, figsize=(10,10))
        axes_f = axes.flatten()
        ax = axes_f[0]
        ax.imshow(img, cmap='gray')
        
        
        ax = axes_f[1]
        ax.imshow(warped, cmap='gray')
        
        
        ax = axes_f[2]
        ax.imshow(img, cmap='gray')
        ax.plot(viterbi_contour[:,1], viterbi_contour[:,0], 'r:')
        
        ax = axes_f[3]
        ax.imshow(warped, cmap='gray')
        ax.plot(edge_viterbi, Angles, ls='', c='cyan', marker='.', ms = 2)

    return(viterbi_contour)

def segment_single_cell_across_film(img, PLOT = False):
    nT, nY, nX = img.shape
    all_contours = np.zeros((nT, 360, 2))
    all_centroids = np.zeros((nT, 2))
    starting_contour = []
    for t in range(nT):
        contour_t = segment_single_cell(img[t], 
                            starting_contour = starting_contour, 
                            N_it_viterbi = 1, PLOT = False)
        P = shapely.Polygon(contour_t)
        C = P.centroid
        all_contours[t] = contour_t
        all_centroids[t] = [C.x, C.y] # In the right direction !! Don't inverse !
        starting_contour = contour_t
        
    # detect outliers
    centroid_avg = np.mean(all_centroids, axis=0)
    all_centroids_rel = all_centroids - (np.ones((1, nT)).T @ np.array([centroid_avg]))
    centroids_r = np.array([(c[0]**2 + c[1]**2)**0.5 for c in all_centroids_rel])
    Zs = stats.zscore(centroids_r)
    outlier_mask = (Zs > 3)
    # for t in range(nT):
    #     if outlier_mask[t]:
    #         segment_single_cell(img[t], starting_contour = [], N_it_viterbi = 4, PLOT = True)
        
    if PLOT:
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        ax.imshow(img[-1], cmap='gray')
        for t in range(nT): 
            ax.plot(all_contours[t,:,1], all_contours[t,:,0], ls='-', lw=0.5)
            ax.plot(all_centroids[t,1], all_centroids[t,0], marker='+')
        plt.show()
    return(all_contours, all_centroids)



        

# %% Scripts

# %%% Import a pair of films

dirPath = up.Path_AnalysisPulls + "/TestCytoRoutine"

# fileName1 = "26-02-09_M1_Pos6_Pa0_C1_Film5min_Dt1sec_1.tif"
# fileName2 = "26-02-09_M1_Pos6_Pa77_C1_Film5min_Dt1sec_1.tif"
fileName1 = "26-02-09_M1_Pos7_Pa0_C1_Film5min_Dt1sec_1-1.tif"
fileName2 = "26-02-09_M1_Pos7_Pa33_C1_Film5min_Dt1sec_1-1.tif"

filePath1 = os.path.join(dirPath, fileName1)
filePath2 = os.path.join(dirPath, fileName2)

shape1, type1 = tiff_inspect(filePath1)
shape2, type2 = tiff_inspect(filePath2)
img1 = skm.io.imread(filePath1)
img2 = skm.io.imread(filePath2)

img1_r = np.zeros_like(img1)
img2_r = np.zeros_like(img2)

# Equalize histograms
for I, I_r in zip([img1, img2], [img1_r, img2_r]):
    for t in range(I.shape[0]):
        p2, p98 = np.percentile(I[t], (1, 99))
        I_r[t] = skm.exposure.rescale_intensity(I[t], in_range=(p2, p98))

SCALE = 0.222
FPS = 1

shape = shape1
img1 = img1_r
img2 = img2_r


all_contours, all_centroids = segment_single_cell_across_film(img1, PLOT = True)

# %%%
centroid_avg = np.mean(all_centroids, axis=0)
all_centroids_rel = all_centroids - (np.ones((1, all_centroids.shape[0])).T @ np.array([centroid_avg]))
centroids_r = np.array([(c[0]**2 + c[1]**2)**0.5 for c in all_centroids_rel])
Zs = stats.zscore(centroids_r)
outlier_mask = (Zs > 3)
for t in range(len(outlier_mask)):
    if outlier_mask[t]:
        print(t)

# %%%

dT = 10
nT, nY, nX = shape




# %%% Import tracked trajectories

dirPath = up.Path_AnalysisPulls + "/TestCytoRoutine"

# fileName1 = "26-02-09_M1_Pos6_Pa0_C1_Film5min_Dt1sec_1.tif"
# fileName2 = "26-02-09_M1_Pos6_Pa77_C1_Film5min_Dt1sec_1.tif"
fileName1 = "26-02-09_M1_Pos6_Pa0_C1_Film5min_Dt1sec_1_Tracks.xml"
fileName2 = "26-02-09_M1_Pos6_Pa77_C1_Film5min_Dt1sec_1_Tracks.xml"


# fileName1 = "26-02-09_M1_Pos7_Pa0_C1_Film5min_Dt1sec_1_Tracks.xml"
# fileName2 = "26-02-09_M1_Pos7_Pa33_C1_Film5min_Dt1sec_1_Tracks.xml"

filePath1 = os.path.join(dirPath, fileName1)
filePath2 = os.path.join(dirPath, fileName2)

Tracks1 = importTrackMateTracks(filePath1)
Tracks2 = importTrackMateTracks(filePath2)






# %%% 

from trackpy.motion import msd, imsd, emsd

#### Format as table & filter

Tables = []
column_names = ['frame', 'x', 'y', 'particle']
for Tracks in [Tracks1, Tracks2]:
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

#### Run msd
fig, ax = plt.subplots(1, 1)
colors = ['cyan', 'r']
labels = ['Before UV', 'After UV']

for k in range(2):
    df = Tables[k]
    
    # res_imsd = imsd(df, SCALE, FPS).reset_index()
    res_emsd = emsd(df, SCALE, FPS, max_lagtime=40).reset_index()
    
    T, MSD = res_emsd['lagt'], res_emsd['msd']
    parms, results = ufun.fitLineHuber(T, MSD, with_intercept = False)
    [D] = parms
    
    ax.plot(res_emsd['lagt'], res_emsd['msd'], color=colors[k], marker='.', lw=0.5, label=labels[k])
    ax.axline(xy1=(0,0), slope=D, color=pm.lighten_color(colors[k], 0.5), ls='-', lw=1, label=f'D = {D:.2e} µm²/s')

ax.grid()
ax.set_xlabel('Lag time (s)')
ax.set_ylabel('MSD (µm²)')
ax.legend()

fig.tight_layout()
plt.show()





# %% Tests

# %%% Second segment cell



def segmentCell(img, starting_contour = [], PLOT = False):
    #### Settings
    inPix_set, outPix_set = 20, 20
    blur_parm = 1
    relative_height_virebi = 0.2
    warp_radius = round(55/SCALE)
    
    nY, nX = img.shape
    Angles = np.arange(0, 360, 1)
       
    if len(starting_contour) == 0:
        mask = get_largest_object(img, mode = 'dark', out_type = 'mask').astype(bool)
        mask = ndi.binary_dilation(mask, iterations=8)
    
        #### 2.0 Locate cell
        (Yc, Xc), Rc = find_cell_inner_circle(img * mask, plot=False, k_th = 1.0)
        if Rc < (10/SCALE):
            (Xc, Yc, Rc) = (nX//2, nY//2, 45/SCALE)
        Yc0, Xc0, Rc0 = round(Yc), round(Xc), round(Rc)
        
        #### 2.1 Warp
        warped = skm.transform.warp_polar(img, center=(Yc, Xc), radius=warp_radius, #Rc*1.2, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        # max_values = np.max(warped[:,Rc0-inPix_set:Rc0+outPix_set+1], axis=1)
            
        #### 2.2 Viterbi Smoothing
        edge_viterbi = viterbi_edge(warped, Rc0, inPix_set, outPix_set, blur_parm, relative_height_virebi)
        edge_viterbi_unwarped = np.array(unwarpRA(np.array(edge_viterbi), Angles, Xc, Yc)) # X, Y
        starting_contour = np.array([edge_viterbi_unwarped[1], edge_viterbi_unwarped[0]]).T
        
    N_it_viterbi = 1
    viterbi_contour = starting_contour
    for i in range(N_it_viterbi):
        # x.0 Locate cell
        (Yc, Xc), Rc = ufun.fitCircle(viterbi_contour, loss = 'huber')        
        Yc0, Xc0, Rc0 = round(Yc), round(Xc), round(Rc)
        # x.1 Warp
        warped = skm.transform.warp_polar(img, center=(Yc, Xc), radius=warp_radius, #Rc*1.4, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        warped = skm.util.img_as_uint(warped)
        w_ny, w_nx = warped.shape
        Angles = np.arange(0, 360, 1)
        max_values = np.max(warped[:,Rc0-inPix_set:Rc0+outPix_set+1], axis=1)
        inPix, outPix = inPix_set, outPix_set
        # x.2 Viterbi Smoothing
        edge_viterbi = viterbi_edge(warped, Rc0, inPix, outPix, blur_parm, relative_height_virebi)
        edge_viterbi_unwarped = unwarpRA(np.array(edge_viterbi), Angles, Xc, Yc)
        viterbi_contour = np.array([edge_viterbi_unwarped[1], edge_viterbi_unwarped[0]]).T
    
    if PLOT:
        fig, axes = plt.subplots(2, 2, figsize=(10,10))
        axes_f = axes.flatten()
        ax = axes_f[0]
        ax.imshow(img, cmap='gray')
        
        ax = axes_f[1]
        ax.imshow(warped, cmap='gray')
        
        ax = axes_f[2]
        ax.imshow(img, cmap='gray')
        ax.plot(viterbi_contour[:,1], viterbi_contour[:,0], 'r:')
        
        ax = axes_f[3]
        ax.imshow(warped, cmap='gray')
        ax.plot(edge_viterbi, Angles, ls='', c='cyan', marker='.', ms = 2)

    return(viterbi_contour)
    
viterbi_contour_0 = segmentCell(img1[0], starting_contour = [], PLOT = True)
viterbi_contour_1 = segmentCell(img1[50], starting_contour = viterbi_contour_0, PLOT = True)

# for I in [img1, img2]:
#     for t in range(0, nT-dT):
#         t1, t2 = t, t+dT
#         I_t1 = I[t1]
#         I_t2 = I[t2]
#         DeltaI = np.abs(I_t1 - I_t2)
#         if t == 0:
#             fig, ax = plt.subplots(1, 1)
#             ax.imshow(DeltaI)
#             plt.show()






# %%% Watershed

def segmentCell(img):
    img_edges = skm.filters.roberts(img)
    img_edges = skm.filters.gaussian(img_edges, sigma=2)
    th = skm.filters.threshold_otsu(img)
    img_bin = (img < th)
    # pad_width = 50
    # img_bin = np.pad(img_bin, pad_width, mode = 'symmetric')
    img_bin = ndi.binary_fill_holes(img_bin, structure=np.ones((3,3)).astype(int))
    img_bin_errode = ndi.binary_erosion(img_bin, iterations=20)
    
    img_dist = ndi.distance_transform_edt(img_bin)
    coords = skm.feature.peak_local_max(img_dist, labels=img_bin, 
                                        footprint=np.ones((5, 5)), min_distance = 50)
    mask = np.zeros_like(img_dist, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-img_dist, markers, mask=img_bin)
    
    
    fig, axes = plt.subplots(1, 4, figsize=(17,12))
    ax = axes[0]
    ax.imshow(img_edges)
    ax = axes[1]
    ax.imshow(img_bin)

        
    ax = axes[2]
    ax.imshow(img_dist)
    for p in coords:
        ax.plot(p[1], p[0], 'r+')
    ax = axes[3]
    ax.imshow(labels)
    
    fig.tight_layout()
    plt.show()

# %%% Skimage watershed example

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max


# Generate an initial image with two overlapping circles
x, y = np.indices((80, 80))
x1, y1, x2, y2 = 28, 28, 44, 52
r1, r2 = 16, 20
mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1**2
mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2**2
image = np.logical_or(mask_circle1, mask_circle2)

# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
distance = ndi.distance_transform_edt(image)
coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask=image)

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray)
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()