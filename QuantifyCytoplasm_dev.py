# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 14:06:30 2026

@author: Joseph
"""


# %% Imports

import os
import re
# import cv2
import time
# import logging
import tifffile
# import traceback


import numpy as np
import pandas as pd
# import pyjokes as pj
import skimage as skm
import seaborn as sns
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

import shapely
from shapely.ops import polylabel
from shapely.plotting import plot_polygon, plot_points # , plot_line

from trackpy.motion import msd, imsd, emsd

from PIL import Image, ImageDraw

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



def viterbi_path_finder(treillis, distance_power = 2):
    """
    To understand the idea, see : https://www.youtube.com/watch?v=6JVqutwtzmo
    """
    
    k = distance_power
    
    def node2node_distance(node1, node2, k):
        """Define a distance between nodes of the treillis graph"""
        d = (np.abs(node1['pos'] - node2['pos']))**k #+ np.abs(node1['val'] - node2['val'])
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
                c = node2node_distance(node1, node2, k) + node2['accumCost']
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

# Test of ViterbiPathFinder

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

def viterbi_edge(warped, mode, Rc, inPix, outPix, blur_parm, 
                 relative_height_viterbi, distance_power_viterbi):
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
    if mode == 'bright':
        pass
    elif mode == 'dark':
        warped = skm.util.invert(warped)
    inBorder = round(Rc) - inPix
    outBorder = round(Rc) + outPix
    for a in Angles:
        profile = warped_filtered[a, :] - np.min(warped_filtered[a, inBorder:outBorder])
        peaks, peaks_props = signal.find_peaks(profile[inBorder:outBorder], 
                                                # width = 4,
                                                height = relative_height_viterbi*np.max(profile[inBorder:outBorder]))
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
    best_path, nodes_list = viterbi_path_finder(TreillisGraph, 
                                                distance_power = distance_power_viterbi)
    
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


def contour_to_centroid(contour):
    P = shapely.Polygon(contour)
    C = P.centroid
    return((C.x, C.y)) # In the right direction !! Don't inverse !
    

def segment_single_cell(img, starting_contour = [], N_it_viterbi = 2, 
                        mode_edge = 'bright', PLOT = False):
    #### Settings
    inPix_set, outPix_set = 30, 30
    blur_parm = 2
    relative_height_viterbi = 0.2
    distance_power_viterbi = 3
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
        edge_viterbi = viterbi_edge(warped, mode_edge, Rc0, inPix_set, outPix_set, blur_parm, 
                                    relative_height_viterbi, distance_power_viterbi)
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
        inPix, outPix = int(inPix_set / (2*(i+1))), int(outPix_set / (2*(i+1)))
        # x.2 Viterbi Smoothing
        edge_viterbi = viterbi_edge(warped, mode_edge, Rc0, inPix, outPix, blur_parm, 
                                    relative_height_viterbi, distance_power_viterbi)
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


    

def segment_single_cell_across_film(img, mode_edge = 'bright', PLOT = False):
    nT, nY, nX = img.shape
    all_contours = np.zeros((nT, 360, 2))
    # all_centroids = np.zeros((nT, 2))
    starting_contour = []
    for t in range(nT):
        contour_t = segment_single_cell(img[t], 
                            starting_contour = starting_contour, 
                            N_it_viterbi = 2, PLOT = False, mode_edge = mode_edge)
        all_contours[t] = contour_t
        # all_centroids[t] = contour_to_centroid(contour_t)
        starting_contour = contour_t
    if PLOT:
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        ax.imshow(img[-1], cmap='gray')
        for t in range(nT): 
            ax.plot(all_contours[t,:,1], all_contours[t,:,0], ls='-', lw=0.5)
            # ax.plot(all_centroids[t,1], all_centroids[t,0], marker='+')
        plt.show()
    return(all_contours)


def get_reasonable_inner_cell_contour(img, PLOT = False):
    # nT, nY, nX = img.shape
    img_min = np.min(img, axis = 0)
    (Yc, Xc), Rc = find_cell_inner_circle(img_min, plot=PLOT, k_th = 1.0)
    Angles = np.linspace(0, 2*np.pi, 360)
    Xcontour = Xc + Rc*np.cos(Angles)
    Ycontour = Yc + Rc*np.sin(Angles)
    contour = np.array([Ycontour, Xcontour]).T
    if PLOT:
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(img_min, cmap='gray')
        axes[0].plot(contour[:,0], contour[:,1], 'r-')
        axes[1].imshow(img[0]*mask, cmap='gray')
        plt.show()
    return(contour)


def get_numbers_following_text(text, target):
    m = re.search(r''+target, text)
    m_num = re.search(r'[\d\.]+', text[m.end():m.end()+10])
    res = int(text[m.end():m.end()+10][m_num.start():m_num.end()])
    return(res)
    
def save_contours(fileName, contours, dstDir):
    fN_root = fileName.split('.')[0]
    fN_contour = fN_root + '_Contours.npy'
    # fN_mask = fN_root + '_Masks.npy'
    np.save(os.path.join(dstDir, fN_contour), contours)
    # np.save(os.path.join(dstDir, fN_mask), masks)

def save_contours_and_masks(fileName, contours, masks, dstDir):
    fN_root = fileName.split('.')[0]
    fN_contour = fN_root + '_Contours.npy'
    fN_mask = fN_root + '_Masks.npy'
    np.save(os.path.join(dstDir, fN_contour), contours)
    np.save(os.path.join(dstDir, fN_mask), masks)
    
def check_if_file_has_tracks(fileName, srcDir):
    fN_root = fileName.split('.')[0]
    fN_contour = fN_root + '_Tracks.xml'
    has_contours = os.path.isfile(os.path.join(srcDir, fN_contour))
    return(has_contours)

def check_if_file_has_contours(fileName, srcDir):
    fN_root = fileName.split('.')[0]
    fN_contour = fN_root + '_Contours.npy'
    has_contours = os.path.isfile(os.path.join(srcDir, fN_contour))
    return(has_contours)
    
def check_if_file_has_contours_and_masks(fileName, srcDir):
    fN_root = fileName.split('.')[0]
    fN_contour = fN_root + '_Contours.npy'
    fN_mask = fN_root + '_Masks.npy'
    has_contours = os.path.isfile(os.path.join(srcDir, fN_contour))
    has_masks = os.path.isfile(os.path.join(srcDir, fN_mask))
    return((has_contours and has_masks))

def import_contours(fileName, srcDir, mode='r'):
    fN_root = fileName.split('.')[0]
    fN_contour = fN_root + '_Contours.npy'
    contours = np.load(os.path.join(srcDir, fN_contour), mmap_mode=mode)
    return(contours)
    
def import_contours_and_masks(fileName, srcDir, mode='r'):
    fN_root = fileName.split('.')[0]
    fN_contour = fN_root + '_Contours.npy'
    fN_mask = fN_root + '_Masks.npy'
    contours = np.load(os.path.join(srcDir, fN_contour), mmap_mode=mode)
    masks = np.load(os.path.join(srcDir, fN_mask), mmap_mode=mode)
    return(contours, masks)

#### V1 of analysis functions

def compute_acor(image, mask, window_length, 
                 EQUALIZE = True, PLOT = False):
    if EQUALIZE:
        for t in range(image.shape[0]):
            p1, p99 = np.percentile(image[t].flatten()[mask.flatten()], (1, 99))
            image[t] = skm.exposure.rescale_intensity(image[t], in_range=(p1, p99))
    
    short_len = window_length
    long_len = image.shape[0] - short_len + 1
    image_acor = np.zeros((long_len, image.shape[1], image.shape[2]))
    
    for i in range(image.shape[1]):
        for j in range(image.shape[2]):
            if mask[i, j]:
                A = image[:, i, j]
                if np.std(A) == 0:
                    print(i, j)
                B = (A - np.mean(A))/np.std(A)
                acor = signal.correlate(B, B[:short_len], mode="valid")
                acor = acor / acor[0]
                image_acor[:, i, j] = acor
                    
    total_acor = np.zeros(long_len)
    lags = np.arange(long_len)
    for t in range(len(total_acor)):
        total_acor[t] = np.mean(image_acor[t].flatten()[mask.flatten()])
    
    if PLOT:
        fig, ax = plt.subplots(1, 1)
        ax.plot(lags, cell_acor, label = fileName)
        plt.show()
        
    return(total_acor, image_acor)


def analyse_cells_ACF(imagePathList, SCALE, FPS):
    res_dict = {
                'id':[],
                'pos_id':[],
                'cell_id':[],
                'Pa':[],
                't_50p':[],
                't_33p':[],
                't_25p':[],
                't_20p':[],
                't_0p':[],
                }
    tables_dict = {}
    ACF_dict = {}
    
    print(pm.BLUE + 'Starting ACF analysis' + pm.NORMAL)
    
    for p in imagePathList:
        T0 = time.time()
        
        # Ids
        _, fN = os.path.split(p)
        
        full_id = '_'.join(fN.split('_')[:5])
        pos_id = get_numbers_following_text(fN, '_Pos')
        cell_id = get_numbers_following_text(fN, '_C')
        Pa = get_numbers_following_text(fN, '_Pa')
        
        # Get image and mask       
        image_raw = skm.io.imread(p)
        image = skm.util.img_as_float32(image_raw)
        shape = image.shape

        single_contour = get_reasonable_inner_cell_contour(image, PLOT = False)
        single_mask = contour_to_mask([shape[1], shape[2]], single_contour)
        single_mask = ndi.binary_erosion(single_mask, iterations = 50)
        
        window_length = shape[0]//3
        
        # ACF function
        total_acor, image_acor = compute_acor(image, single_mask, window_length, 
                         EQUALIZE = True, PLOT = False)
        
        ACF_dict[full_id] = total_acor
        
        timescales = ['t_50p', 't_33p', 't_25p', 't_20p', 't_0p']
        th_timescales = [1/2, 1/3, 1/4, 1/5, 0]
        dict_timescales = {k:0 for k in timescales}
        # print(timescales)
        # print(th_timescales)
        # print(dict_timescales)
        
        for ts, th in zip(timescales, th_timescales):
            test = (total_acor < th)
            t = ufun.findFirst(1, test)
            dict_timescales[ts] = t
        
        res_dict['id'].append(full_id)
        res_dict['pos_id'].append(pos_id)
        res_dict['cell_id'].append(cell_id)
        res_dict['Pa'].append(Pa)
        for ts in dict_timescales.keys():
            res_dict[ts].append(dict_timescales[ts])
        
        
        
        Dt = time.time() - T0
        print(pm.GREEN + f'Done with {fN}' + pm.NORMAL + f' ; Dt = {Dt:.4f}')
        
    
    res_df = pd.DataFrame(res_dict)

        
    return(res_df, ACF_dict)
        
        
        

def analyse_white_blobs_MSD(trackPathList, SCALE, FPS):
    res_dict = {
                'id':[],
                'pos_id':[],
                'cell_id':[],
                'Pa':[],
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
        full_id = '_'.join(fN.split('_')[:5])
        pos_id = get_numbers_following_text(fN, '_Pos')
        cell_id = get_numbers_following_text(fN, '_C')
        Pa = get_numbers_following_text(fN, '_Pa')
        
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
        res_dict['D'].append(D)
        res_dict['k_nl'].append(k_nl)
        res_dict['D_nl'].append(D_nl)
        
        Dt = time.time() - T0
        print(pm.GREEN + f'Done with {fN}' + pm.NORMAL + f' ; Dt = {Dt:.4f}')
        
    res_df = pd.DataFrame(res_dict)
        
    return(res_df, MSD_dict)

# %% Scripts

# %%% Import a pair of films

dirPath = up.Path_AnalysisPulls + "/TestCytoRoutine"

fileName1 = "26-02-09_M1_Pos6_Pa0_C1_Film5min_Dt1sec_1.tif"
fileName2 = "26-02-09_M1_Pos6_Pa77_C1_Film5min_Dt1sec_1.tif"
# fileName1 = "26-02-09_M1_Pos7_Pa0_C1_Film5min_Dt1sec_1-1.tif"
# fileName2 = "26-02-09_M1_Pos7_Pa33_C1_Film5min_Dt1sec_1-1.tif"

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

# get_reasonable_inner_cell_contour(img1, PLOT = True)


# all_contours1  = segment_single_cell_across_film(img1, PLOT = False)
# all_contours2  = segment_single_cell_across_film(img2, PLOT = False)

# %%% Import a list of films

SCALE = 0.222
FPS = 1

dirPath = up.Path_AnalysisPulls + "/26-02-09_UVonCytoplasm"

fileNames = [
             "26-02-09_M1_Pos6_Pa0_C1_Film5min_Dt1sec_1.tif",
             # "26-02-09_M1_Pos6_Pa77_C1_Film5min_Dt1sec_1.tif",
             # "26-02-09_M1_Pos6_Pa7_C1_Film5min_Dt1sec_1.tif",
             ]
listAllFiles = os.listdir(dirPath)
listTifFiles = [f for f in listAllFiles if f.endswith('.tif')]

# fileName1 = "26-02-09_M1_Pos7_Pa0_C1_Film5min_Dt1sec_1-1.tif"
# fileName2 = "26-02-09_M1_Pos7_Pa33_C1_Film5min_Dt1sec_1-1.tif"

filePaths = [os.path.join(dirPath, fN) for fN in fileNames]

Images_raw = [skm.io.imread(fP) for fP in filePaths]
Images = [np.zeros_like(img) for img in Images_raw]

# Equalize histograms
for Ir, I in zip(Images_raw, Images):
    for t in range(I.shape[0]):
        p2, p98 = np.percentile(Ir[t], (1, 99))
        I[t] = skm.exposure.rescale_intensity(Ir[t], in_range=(p2, p98))

Images_contours = [segment_single_cell_across_film(img, PLOT = False) for img in Images]
Images_masks = np.array([[contour_to_mask((img_t.shape[1], img_t.shape[0]), contour_t) \
                          for (img_t, contour_t) in zip(img, contour)] \
                          for (img, contour) in zip(Images, Images_contours)])

Images_dict = {fileNames[k] : {'images'  :Images[k],
                               'contours':Images_contours[k],
                               'masks'   :Images_masks[k],
                               } \
               for k in range(len(fileNames))}

#### Test Save / import

# def save_contours_and_masks(Img_dict, dstDir):
#     for fN in Img_dict.keys():
#         fN_root = fN.split('.')[0]
#         fN_contour = fN_root + '_Contours.npy'
#         fN_mask = fN_root + '_Masks.npy'
#         np.save(os.path.join(dstDir, fN_contour), Img_dict[fN]['contours'])
#         np.save(os.path.join(dstDir, fN_mask), Img_dict[fN]['masks'])

# save_contours_and_masks(Images_dict, dirPath)

# %%% Get the contour & masks --> save them

SCALE = 0.222
FPS = 1

redo_all_files = True
dirPath = up.Path_AnalysisPulls + "/26-02-09_UVonCytoplasm"

listAllFiles = os.listdir(dirPath)
# listTifFiles = [f for f in listAllFiles if f.endswith('.tif')]
listTifFiles = [f for f in listAllFiles if (f.endswith('.tif') and ('M1_Pos6_Pa77_C2' in f))]

if not redo_all_files:
    filesToAnalyze = {fN:True for fN in listTifFiles}
    for fN in listTifFiles:
        check = check_if_file_has_contours_and_masks(fN, dirPath)
        filesToAnalyze[fN] = (not check)
    listTifFiles = [fN for fN in listTifFiles if filesToAnalyze[fN]]
    
listTifPaths = [os.path.join(dirPath, fN) for fN in listTifFiles]


# for k in range(len(listTifFiles)):
#     fN, fP = listTifFiles[k], listTifPaths[k]
#     img = skm.io.imread(fP)
    
#     for t in range(img.shape[0]):
#         p2, p98 = np.percentile(img[t], (1, 99))
#         img[t] = skm.exposure.rescale_intensity(img[t], in_range=(p2, p98))
    
#     img_contours = segment_single_cell_across_film(img, mode = 'dark', PLOT = False)
    #### ----
    # img_masks = np.array([contour_to_mask((img_t.shape[1], img_t.shape[0]), contour_t) \
                              # for (img_t, contour_t) in zip(img, img_contours)])
    
    # save_contours_and_masks(fileName, contours, masks, dstDir)
    # import_contours_and_masks(fileName, srcDir, mode='r')
    # save_contours_and_masks(fN, img_contours, img_masks, dirPath)
    #### ----
    # save_contours(fN, img_contours, dirPath)


# %%% Tracking & MSD V1

#### Import tracked trajectories

dirPath = up.Path_AnalysisPulls + "/TestCytoRoutine"

# fileName1 = "26-02-09_M1_Pos6_Pa0_C1_Film5min_Dt1sec_1.tif"
# fileName2 = "26-02-09_M1_Pos6_Pa77_C1_Film5min_Dt1sec_1.tif"
fileName1 = "26-02-09_M1_Pos6_Pa0_C1_Film5min_Dt1sec_1_Tracks.xml"
fileName2 = "26-02-09_M1_Pos6_Pa77_C1_Film5min_Dt1sec_1_Tracks.xml"
fileName1 = "26-02-09_M1_Pos7_Pa0_C1_Film5min_Dt1sec_1_Tracks.xml"
fileName2 = "26-02-09_M1_Pos7_Pa33_C1_Film5min_Dt1sec_1_Tracks.xml"

filePath1 = os.path.join(dirPath, fileName1)
filePath2 = os.path.join(dirPath, fileName2)

Tracks1 = importTrackMateTracks(filePath1)
Tracks2 = importTrackMateTracks(filePath2)

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
    
    res_imsd = imsd(df, SCALE, FPS).reset_index()
    res_emsd = emsd(df, SCALE, FPS, max_lagtime=40).reset_index()
    
    T, MSD = res_emsd['lagt'], res_emsd['msd']
    parms, results = ufun.fitLineHuber(T, MSD, with_intercept = False)
    [D] = parms/4
    
    ax.plot(res_emsd['lagt'], res_emsd['msd'], color=colors[k], marker='.', lw=0.5, label=labels[k])
    ax.axline(xy1=(0,0), slope=D*4, color=pm.lighten_color(colors[k], 0.5), ls='-', lw=1, label=f'D = {D:.2e} µm²/s')

ax.grid()
ax.set_xlabel('Lag time (s)')
ax.set_ylabel('MSD (µm²)')
ax.legend()

fig.tight_layout()
plt.show()

# %%% Tracking & MSD V2


        # fig, ax = plt.subplots(1, 1)
        # colors = ['cyan', 'r']
        # labels = ['Before UV', 'After UV']
        # ax.plot(res_emsd['lagt'], res_emsd['msd'], color=colors[k], marker='.', lw=0.5, label=labels[k])
        # ax.axline(xy1=(0,0), slope=D, color=pm.lighten_color(colors[k], 0.5), ls='-', lw=1, label=f'D = {D:.2e} µm²/s')
        # ax.grid()
        # ax.set_xlabel('Lag time (s)')
        # ax.set_ylabel('MSD (µm²)')
        # ax.legend()
        # fig.tight_layout()
        # plt.show()

SCALE = 0.222
FPS = 1
dirPath = up.Path_AnalysisPulls + "/26-02-09_UVonCytoplasm"
listAllFiles = os.listdir(dirPath)
listTifFiles = [f for f in listAllFiles if f.endswith('.tif')]

filesWithTracks = {fN:True for fN in listTifFiles}
for fN in listTifFiles:
    check = check_if_file_has_tracks(fN, dirPath)
    filesWithTracks[fN] = (check)
listTifFiles = [fN for fN in listTifFiles if filesWithTracks[fN]]
listTrackFiles = [fN[:-4] + '_Tracks.xml' for fN in listTifFiles if filesWithTracks[fN]]

listTifPaths = [os.path.join(dirPath, f) for f in listTifFiles]
listTrackPaths = [os.path.join(dirPath, f) for f in listTrackFiles]

# msd_df = analyse_white_blobs_MSD(listPaths[:], SCALE, FPS)


# %%% Autocorelation V1


dirPath = up.Path_AnalysisPulls + "/26-02-09_UVonCytoplasm"

listAllFiles = os.listdir(dirPath)
listTifFiles = [f for f in listAllFiles if f.endswith('.tif')]
# filesAnalyzed = {fN:True for fN in listTifFiles}
# for fN in listTifFiles:
#     check = check_if_file_has_contours_and_masks(fN, dirPath)
#     filesAnalyzed[fN] = (check)
# listTifFiles = [fN for fN in listTifFiles if filesAnalyzed[fN]]
listTifPaths = [os.path.join(dirPath, fN) for fN in listTifFiles]

fig, ax = plt.subplots(1, 1)

# index_list = [0, 6, 3] # Pa0, 6, 66
# index_list = [10, 17, 13] # Pa0, 7, 77
# index_list = [23, 26] # Pa0, 33
index_list = [10]
fileNames = [listTifFiles[i] for i in index_list]
filePaths = [listTifPaths[i] for i in index_list]

for fileName, filePath in zip(fileNames, filePaths):
    image_raw = skm.io.imread(filePath)
    image = skm.util.img_as_float32(image_raw)
    shape = image.shape
    
    # contours, masks = import_contours_and_masks(fileName, dirPath, mode='r')
    # contours = import_contours(fileName, dirPath, mode='r')
    # masks = [contour_to_mask([shape[1], shape[2]], C) for C in contours]
    # single_mask = ndi.binary_erosion(np.all(masks, axis=0), iterations = 20)

    single_contour = get_reasonable_inner_cell_contour(image, PLOT = False)
    single_mask = contour_to_mask([shape[1], shape[2]], single_contour)
    single_mask = ndi.binary_erosion(single_mask, iterations = 50)
    
    fig_i, ax_i = plt.subplots(3, 1)
    ax_i[0].imshow(image[10], cmap='gray')
    ax_i[0].plot(single_contour[:,1], single_contour[:,0], 'r--')
    ax_i[1].imshow(single_mask, cmap='gray')
    ax_i[2].imshow(image[10]*single_mask, cmap='gray')
    plt.show()

#### UNCOMMENT HERE FOR AUTOCORR !!
    for t in range(image.shape[0]):
        p1, p99 = np.percentile(image[t].flatten()[single_mask.flatten()], (1, 99))
        image[t] = skm.exposure.rescale_intensity(image[t], in_range=(p1, p99))
    
    # image_masked = image*single_mask
    short_len = image.shape[0]//3
    long_len = image.shape[0] - short_len + 1
    image_acor = np.zeros((long_len, image.shape[1], image.shape[2]))
    
    for i in range(image.shape[1]):
        for j in range(image.shape[2]):
            if single_mask[i, j]:
                A = image[:, i, j]
                if np.std(A) == 0:
                    print(i, j)
                B = (A - np.mean(A))/np.std(A)
                acor = signal.correlate(B, B[:short_len], mode="valid") #[len(A)//2:]
                acor = acor / acor[0]
                # lags = signal.correlation_lags(len(A), len(A), mode="full")[len(A):]
                image_acor[:, i, j] = acor
                
    # skm.io.imsave(filePath[:-4] + '_acor.tif', image_acor)
    
    cell_acor = np.zeros(long_len)
    lags = np.arange(0, long_len)
    for t in range(len(cell_acor)):
        cell_acor[t] = np.mean(image_acor[t].flatten()[single_mask.flatten()])
    
    ax.plot(lags, cell_acor, label = fileName)

ax.legend()
plt.show()


# %%% Better script for ACF and MSD

SCALE = 0.222
FPS = 1

redo_all_files = False

dirPath = up.Path_AnalysisPulls + "/26-02-09_UVonCytoplasm"
listAllFiles = os.listdir(dirPath)
listTifFiles = [f for f in listAllFiles if f.endswith('.tif')]

filesWithTracks = {fN:True for fN in listTifFiles}
for fN in listTifFiles:
    check = check_if_file_has_tracks(fN, dirPath)
    filesWithTracks[fN] = (check)
listTifFiles = [fN for fN in listTifFiles if filesWithTracks[fN]]
listTrackFiles = [fN[:-4] + '_Tracks.xml' for fN in listTifFiles if filesWithTracks[fN]]

listTifPaths = [os.path.join(dirPath, f) for f in listTifFiles]
listTrackPaths = [os.path.join(dirPath, f) for f in listTrackFiles]

ACF_res_df, ACF_dict = analyse_cells_ACF(listTifPaths[:], SCALE, FPS)
ufun.dict2json(ACF_dict, dirPath + '/Results', 'ACF_dict')
ACF_res_df.to_csv(os.path.join(dirPath, 'Results', 'results_ACF.csv'), index=False)

MSD_res_df, MSD_dict = analyse_white_blobs_MSD(listTrackPaths[:], SCALE, FPS)
ufun.dict2json(MSD_dict, dirPath + '/Results', 'MSD_dict')
MSD_res_df.to_csv(os.path.join(dirPath, 'Results', 'results_MSD.csv'), index=False)


# %%% Plot the results

SCALE = 0.222
FPS = 1

dirPath = up.Path_AnalysisPulls + "/26-02-09_UVonCytoplasm"
listAllFiles = os.listdir(dirPath)

df_ACF = pd.read_csv(os.path.join(dirPath, 'Results', 'results_ACF.csv'))
df_MSD = pd.read_csv(os.path.join(dirPath, 'Results', 'results_MSD.csv'))


#### ACF

Id_cols = ['pos_id',]
Group_cols = ('Pa')
Yplot = 't_33p'

group_ACF = df_ACF.groupby(Group_cols)
agg_dict = {k:'first' for k in Id_cols}
agg_dict.update({Yplot:'mean'})
df_ACF_g = group_ACF.agg(agg_dict).reset_index()

fig, ax = plt.subplots(1, 1)
sns.swarmplot(data=df_ACF, ax=ax, x='Pa', y=Yplot)
plt.tight_layout()
plt.show()


#### MSD

Id_cols = ['pos_id',]
Group_cols = ('Pa')
Yplot = 'D'

group_MSD = df_MSD.groupby(Group_cols)
agg_dict = {k:'first' for k in Id_cols}
agg_dict.update({Yplot:'mean'})
df_MSD_g = group_MSD.agg(agg_dict).reset_index()

fig, ax = plt.subplots(1, 1)
ax.set_yscale('log')
sns.swarmplot(data=df_MSD, ax=ax, x='Pa', y=Yplot)
plt.tight_layout()
plt.show()



#### Check consistency

df_merged = pd.merge(left=df_MSD, right=df_ACF, on='id', how='inner')
df_merged['Pa_x'] = df_merged['Pa_x'].apply(lambda x : str(x))
Xplot = 'D'
Yplot = 't_33p'

fig, ax = plt.subplots(1, 1)
# ax.set_yscale('log')
sns.scatterplot(data=df_merged, ax=ax, x=Xplot, y=Yplot, 
                hue='Pa_x',
                )
plt.tight_layout()
plt.show()


#### Check consistency 2

df_merged = pd.merge(left=df_MSD, right=df_ACF, on='id', how='inner')
df_merged['Pa_x'] = df_merged['Pa_x'].apply(lambda x : str(x))
Xplot = 'D'
Yplot = 't_0p'

fig, ax = plt.subplots(1, 1)
# ax.set_yscale('log')
sns.scatterplot(data=df_merged, ax=ax, x=Xplot, y=Yplot, 
                hue='Pa_x',
                )
plt.tight_layout()
plt.show()

# %% ------------





# %% Tests

# %%% Test with simple image difference

dirPath = up.Path_AnalysisPulls + "/TestCytoRoutine"

fileName1 = "26-02-09_M1_Pos6_Pa0_C1_Film5min_Dt1sec_1.tif"
fileName2 = "26-02-09_M1_Pos6_Pa77_C1_Film5min_Dt1sec_1.tif"
# fileName1 = "26-02-09_M1_Pos7_Pa0_C1_Film5min_Dt1sec_1-1.tif"
# fileName2 = "26-02-09_M1_Pos7_Pa33_C1_Film5min_Dt1sec_1-1.tif"

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

all_contours1  = segment_single_cell_across_film(img1, PLOT = False)
all_contours2  = segment_single_cell_across_film(img2, PLOT = False)

dT = 10
nT, nY, nX = shape
errode_it = 7

lag_times_set = np.arange(1, 25, 2)
deltas_set = np.zeros_like(lag_times_set)

fig, ax = plt.subplots(1, 1)
for img, all_contours in zip([img1, img2], [all_contours1, all_contours2]):
    nT, nY, nX = img.shape
    all_masks = [contour_to_mask((nY, nX), C) for C in all_contours]
    
    lag_times = lag_times_set
    deltas = deltas_set
    
    for i, dT in enumerate(lag_times):
        scores = np.zeros(nT - dT)
        for t in range(0, nT-dT):
            t1, t2 = t, t+dT
            
            contour_t1 = all_contours[t1]
            mask_t1 = all_masks[t1]
            mask_t1 = ndi.binary_erosion(mask_t1, iterations=errode_it)
            contour_t2 = all_contours[t2]
            mask_t2 = all_masks[t2]
            mask_t2 = ndi.binary_erosion(mask_t2, iterations=errode_it)
            
            mask = mask_t1 & mask_t2
            img_delta = np.abs((img[t2]*mask).astype(float) - (img[t1]*mask).astype(float))
            scores[t] = np.sum(img_delta) / np.sum(mask)       
            # fig, ax = plt.subplots(1, 1)
            # plt.imshow(img_delta, cmap='gray')
            # plt.show()
            
        deltas[i] = np.mean(scores)

    ax.plot(lag_times, deltas)

plt.show()
    

# %%% Test with ACF on checkerboard

image_raw = skm.data.checkerboard()
image = skm.util.img_as_float32(image_raw)
s = image.shape

nT = 200
film = np.zeros((nT, s[0], s[1]), dtype=float)
for t in range(nT):
    film[t] = np.roll(image, t, axis=1)
    
single_mask = np.ones((s[0], s[1]), dtype=bool)
image = film
    
# for t in range(image.shape[0]):
#     p1, p99 = np.percentile(image[t].flatten()[single_mask.flatten()], (1, 99))
#     image[t] = skm.exposure.rescale_intensity(image[t], in_range=(p1, p99))
    

# image_masked = image*single_mask
fig, ax = plt.subplots(1, 1)
short_len = 75
long_len = image.shape[0]-short_len + 1
image_acor = np.zeros((long_len, image.shape[1], image.shape[2]))

for i in range(single_mask.shape[0]):
    for j in range(single_mask.shape[1]):
        if single_mask[i, j]:
            A = image[:, i, j]
            if np.std(A) == 0:
                print(i, j)
            B = (A - np.mean(A))/np.std(A)
            acor = signal.correlate(B, B[:short_len], mode="valid")#[len(A)//2:]
            acor = acor / acor[0]
            # lags = signal.correlation_lags(len(A), len(A), mode="full")[len(A):]
            image_acor[:, i, j] = acor
            
# skm.io.imsave(filePath[:-4] + '_acor.tif', image_acor)

cell_acor = np.zeros(long_len)
lags = np.arange(0, long_len)
for t in range(len(cell_acor)):
    cell_acor[t] = np.mean(image_acor[t].flatten()[single_mask.flatten()])

ax.plot(lags, cell_acor, label = fileName)
    
# fig, axes = plt.subplots(1, 3)
# for k in range(len(axes)):
#     ax = axes[k]
#     ax.imshow(film[k*10], cmap='gray')

plt.show()

# %%% Test autocorr over an example

A = np.array([11, 10, 11, 10, 11, 12, 5, 2, 1, 0, 1, 0, 1])
A = A*10
B = (A - np.mean(A))/np.std(A)

print(B)
AC = signal.correlate(B, B, mode="full")

print(AC)

# %%% Test polygon to mask
# polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
# poly = np.round(viterbi_contour[:,::-1], 0).astype(int)
# poly = [(p[0], p[1]) for p in poly] + [(poly[0, 0], poly[0, 1])]
# height, width = img1[t].shape

# Im0 = Image.new('L', (width, height), 0)
# ImageDraw.Draw(Im0).polygon(poly, outline=1, fill=1)
# mask = np.array(Im0)
# plt.imshow(mask)



# %%% Test centroid & outliers detection
# centroid_avg = np.mean(all_centroids, axis=0)
# all_centroids_rel = all_centroids - (np.ones((1, all_centroids.shape[0])).T @ np.array([centroid_avg]))
# centroids_r = np.array([(c[0]**2 + c[1]**2)**0.5 for c in all_centroids_rel])
# Zs = stats.zscore(centroids_r)
# outlier_mask = (Zs > 3)
# for t in range(len(outlier_mask)):
#     if outlier_mask[t]:
#         print(t)

# %%% Warp test

X, Y = np.arange(200), np.arange(100)
XX, YY = np.meshgrid(X, Y)

Xcw, Ycw = 100, 50

R, A = warpXY(XX, YY, Xcw, Ycw)

XX2, YY2 = unwarpRA(R, A, Xcw, Ycw)

# %%% Warp test 2

warped = np.zeros((360, 100))
RR, AA = np.meshgrid(np.arange(warped.shape[1]), np.arange(warped.shape[0]))

B = (np.repeat(np.arange(5), 20))
C = np.resize(B, (5, 20))

# %%% Second segment cell



def segmentCell(img, starting_contour = [], PLOT = False):
    #### Settings
    inPix_set, outPix_set = 20, 20
    blur_parm = 1
    relative_height_viterbi = 0.2
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
        edge_viterbi = viterbi_edge(warped, Rc0, inPix_set, outPix_set, blur_parm, relative_height_viterbi)
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
        edge_viterbi = viterbi_edge(warped, Rc0, inPix, outPix, blur_parm, relative_height_viterbi)
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