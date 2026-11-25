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
import trackpy as tp
import skimage as skm
# import seaborn as sns
import matplotlib as mpl
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from shapely import MultiPoint, Polygon
# from shapely.ops import polylabel
# from shapely.plotting import plot_polygon, plot_points # , plot_line

# from trackpy.motion import msd, imsd, emsd
# from PIL import Image, ImageDraw
# from scipy.special import jv
from scipy.spatial import Delaunay, cKDTree # ConvexHull, 
from scipy.signal import savgol_filter

import trackpy as tp

import Libs.PlotMaker as pm
import Libs.UrchinPaths as up
import Libs.UtilityFunctions as ufun

# %% Imports 2

os.environ["JAVA_HOME"] = up.Path_JAVA_HOME

import imagej
import scyjava as sj

# import random
sj.config.add_options('-Xmx48g')



# initialize ImageJ
# ij = imagej.init()
# ij = imagej.init('sc.fiji:fiji')
ij = imagej.init(up.Path_Fiji, add_legacy=False)

print(f"ImageJ version: {ij.getVersion()}")


# %% I. Single Particle Tracking

# %%% Helper functions

# %%%% Files IO


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


def export_as_TrackMate_tracks(tracks,
                               filepath,
                               spaceUnits="pixels",
                               frameInterval=1.0,
                               timeUnits="frames",
                               generationDateTime=None,
                               from_version="TrackMate v7.14.0"):
    """
    Export a list of particle tracks to a TrackMate-style XML file.

    Parameters
    ----------
    tracks : list of numpy.ndarray
        Each array has shape (N, 3), with columns [t, x, y].
    filepath : str
        Output XML file path.
    spaceUnits : str
        Spatial units, e.g. "pixels" or "microns".
    frameInterval : float
        Time interval between frames.
    timeUnits : str
        Time units, e.g. "frames" or "seconds".
    generationDateTime : str or None
        Date/time string for the XML header.
    from_version : str
        TrackMate version string for the XML header.
    """

    if generationDateTime is None:
        generationDateTime = ""

    # Root element
    root = ET.Element("Tracks",
        {
            "nTracks": str(len(tracks)),
            "spaceUnits": str(spaceUnits),
            "frameInterval": str(frameInterval),
            "timeUnits": str(timeUnits),
            "generationDateTime": str(generationDateTime),
            "from": str(from_version),
        })

    # Add each particle and its detections
    for track in tracks:
        track = np.asarray(track)

        if track.size == 0:
            particle = ET.SubElement(root, "particle", {"nSpots": "0"})
            continue

        if track.ndim != 2 or track.shape[1] != 3:
            raise ValueError("Each track must be a 2D array with columns [t, x, y].")

        particle = ET.SubElement(
            root,
            "particle",
            {"nSpots": str(len(track))}
        )

        for t, x, y in track:
            ET.SubElement(
                particle,
                "detection",
                {
                    "t": str(t),
                    "x": str(x),
                    "y": str(y),
                    "z": "0.0",
                })

    # Pretty-print the XML with indentation
    ET.indent(root, space="  ")

    # Write XML to disk
    tree = ET.ElementTree(root)
    tree.write(
        filepath,
        encoding="UTF-8",
        xml_declaration=True
    )


# %%%% Segmentation

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

def get_contour_barycenter(contour):
    """"contour is a Nx2 array [[y0, x0], [y1, x1], ... ]"""
    poly = Polygon(contour[:, ::-1])
    y, x = poly.centroid.y, poly.centroid.x
    return([y, x])

def make_NbYolkCell_contour_and_mask(img, tifName, figDir, PixPerUm,
                                     mode = 'dark_background', buffer_um = 0.0,
                                     keep_only_largest = False, exclude_on_edges = True,
                                     PLOT = False, SHOWPLOT = False, SAVEPLOT = False):
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
    
    if exclude_on_edges:
        img_bin = skm.segmentation.clear_border(img_bin)
    
    img_bin = ndi.binary_opening(img_bin, iterations = 5)
    img_bin = ndi.binary_fill_holes(img_bin)
    
    # Get contours from first mask
    FoundContours = skm.measure.find_contours(img_bin, 0.5)
    if keep_only_largest:
        FoundContours = [FoundContours[np.argmax([len(c) for c in FoundContours])]]
    
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
    
    
    # Plot
    if PLOT:
        if not SHOWPLOT:
            plt.ioff()
        else:
            plt.ion()
            
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
        
        if SHOWPLOT:
            plt.show()
        
        if SAVEPLOT:
            figName = tifName.split('.')[0] + '_contour.png'
            figPath = os.path.join(figDir, figName)
            fig.savefig(figPath, dpi=300, )
    
    if not SHOWPLOT:
        plt.ion()
        
    return(Contour_inner, Mask_inner)



def make_NbYolkPronucleus_contour_and_mask(img, PixPerUm,
                                           Contour_cell, Mask_cell,
                                           mode = 'dark_background', 
                                           buffer_pix = 3,
                                           PLOT = False):

    # Contour_inner = Contour_cell
    Mask_inner = Mask_cell
    
    # ---
    
    # k_th = 0.9
    nT, nY, nX = img.shape
    th_fun = skm.filters.threshold_otsu
    Mask_inner_f = Mask_inner.flatten()
    
    step = nT//10
    img_masked = img[::step] * Mask_inner
    img_bin = np.zeros_like(img_masked)
    
    if mode == 'dark_background':
        for k, im in enumerate(img_masked):
            im_f = im.flatten()
            th1 = th_fun(im_f[Mask_inner_f]) # * k_th
            img_bin[k] = (im < th1)
            img_bin[k] = ndi.binary_opening(img_bin[k], iterations = 2)
    elif mode == 'bright_background':
        for k, im in enumerate(img_masked):
            im_f = im.flatten()
            th1 = th_fun(im_f[Mask_inner_f]) # * k_th
            img_bin[k] = (im > th1)
            img_bin[k] = ndi.binary_opening(img_bin[k], iterations = 2)
    
    img_bin = np.median(img_bin, axis = 0)
    img_bin = ndi.binary_opening(img_bin, iterations = 10)
    img_bin = ndi.binary_dilation(img_bin, iterations = 1)
    
    img_label, num_features = ndi.label(img_bin)
    properties = ['label', 'area', 'perimeter']
    df = pd.DataFrame(skm.measure.regionprops_table(img_label, img_bin, properties = properties))
    df['circularity'] = 4*np.pi*df['area']/(df['perimeter']**2)
    df = df[df['area'] > 6000]
    df = df[df['area'] < 15000]
    df = df[df['circularity'] > 0.55]

    if len(df) > 0:
        df = df.sort_values(by='area', ascending=False)
        best_label = df['label'].to_numpy()[0]
        mask_pronuc = (img_label == best_label)
        mask_pronuc = ndi.binary_dilation(mask_pronuc, iterations = buffer_pix)
        contour_pronuc = skm.measure.find_contours(mask_pronuc, 0.5)[0]
    else:
        mask_pronuc = None
        contour_pronuc = None
        

    if PLOT:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
        axes_f = axes.flatten()
        
        ax = axes_f[0]
        ax.set_aspect('equal', adjustable='box')
        ax.imshow(img[0], cmap='gray')
        if not (contour_pronuc is None):
            ax.plot(contour_pronuc[:,1], contour_pronuc[:,0], 'r-')
        
        ax = axes_f[1]
        ax.set_aspect('equal', adjustable='box')
        ax.imshow(img_bin, cmap='gray')
        
        if not (mask_pronuc is None):
            ax = axes_f[2]
            ax.set_aspect('equal', adjustable='box')
            ax.imshow(mask_pronuc, cmap='gray')
        
        plt.show()
    
    return contour_pronuc, mask_pronuc

def is_track_in_polygon(track, poly):
    xm = np.median(track[:, 1])
    ym = np.median(track[:, 2])
    point = MultiPoint([(ym, xm)])
    res = poly.contains(point)
    return(res)

def update_masks(yx, L, total_mask, allowed_mask, NyNx, 
                 force_distribution = False):
    Ny, Nx = NyNx
    y, x = yx
    y1, x1 = y+L, x+L   
    y0, x0 = y-L, x-L
    total_mask[y:y1, x:x1] = False
    
    y1, x1 = min(y1, Ny), min(x1, Nx)
    y0, x0 = max(y0, 0), max(x0, 0)
    allowed_mask[y0:y1, x0:x1] = False
    
    if force_distribution:
        allowed_mask[int(y-L/2):int(y+L/2), :] = False
        allowed_mask[:, int(x-L/2):int(x+L/2)] = False
    
    return(total_mask, allowed_mask)


def get_N_random_square_ROIs_in_cell(mask, L, N, 
                                     excluded_mask=None,
                                     force_distribution = False,
                                     seed=None, PLOT = False):
    Ny, Nx = mask.shape
    
    if not (excluded_mask is None):
        total_mask = mask & (~excluded_mask)
    else:
        total_mask = mask
    
    kernel = np.zeros((L*2 - 1, L*2 - 1), dtype=bool)
    kernel[L:, L:] = True
    
    allowed_mask = ndi.binary_erosion(total_mask, structure=kernel, iterations=1)
    # allowed_mask = cv2.erode(total_mask, kernel, iterations=1)
    
    if PLOT:
        fig, axes = plt.subplots(1, N+1, figsize=((N+1)*3, 3.5))
        ax = axes[0]
        ax.set_aspect('equal', adjustable='box')
        ax.imshow(total_mask*0.5 + allowed_mask*0.5, 
                  cmap='gray', vmin=0, vmax=1)
    
    list_rois = []
    
    for k in range(N):
    
        coords = np.argwhere(allowed_mask)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, coords.shape[0], 1, dtype=int)
        
        y0, x0 = coords[idx,:][0]
        y1, x1 = y0+L, x0+L
        list_rois.append([[int(y0), int(x0)], 
                          [int(y1), int(x1)]])
        
        # total_mask[y0:y1, x0:x1] = False
        # erroded_mask = ndi.binary_erosion(total_mask, structure=kernel, iterations=1)
        
        total_mask, allowed_mask = update_masks((y0, x0), L, 
                                                total_mask, allowed_mask, (Ny, Nx),
                                                force_distribution = force_distribution)
    
        if PLOT:
            ax = axes[k+1]
            ax.set_aspect('equal', adjustable='box')
            ax.imshow(total_mask*0.5 + allowed_mask*0.5, 
                      cmap='gray', vmin=0, vmax=1)
            for roi in list_rois:
                y0, x0 = roi[0]
                ax.plot([x0-0.5], [y0-0.5], 'ro', markersize=2)
                rect = mpl.patches.Rectangle((x0-0.5, y0-0.5), L, L,
                                             facecolor='None', edgecolor='g')
                ax.add_patch(rect)

    if PLOT:
        plt.show()
    
    return(list_rois)



# %%%% Data

def point_cell_center(img):
    QAtitle='Cell Microtubules Center',
    QAtext='Can the organizing center of the microtubules be pointed?',
    QAbuttons = ['Point center', 'No center']
    QA = ufun.showAndQA(img, strech = True, 
                        QAtitle = QAtitle, QAtext = QAtext, QAbuttons = QAbuttons)
    if QA == 'Point center':
        ui = plt.ginput(1, timeout=0)
        uiXY = ufun.ui2array(ui)
    else:
        uiXY = []
    fig = plt.gcf()
    plt.close(fig)
    return(QA, uiXY)


def rawTracks_2_cleanTracks(rawTracks, cleanTrackName, plotDir, 
                            PixPerUm, shape = None, Contour_cell = None, 
                            edge_buffer_cutoff_um = 2.5, nPoints_cuttoff = 30,
                            column_names = None,
                            RefImg = None, PLOT = False, SHOWPLOT = False, SAVEPLOT = False):
    
    if column_names is None:
        column_names = ['frame', 'x', 'y', 'particle']
    else:
        if len(column_names) != 4:
            raise ValueError("There should be 4 column names and they should be " + \
                   "roughly equivalent to: ['frame', 'x', 'y', 'particle']")
                
    if (Contour_cell is None) and (shape is None):
        raise ValueError("Please provide either a Contour for the ROI or the image shape")
    
    Valid_Tracks = []
    buffer_pix = edge_buffer_cutoff_um * PixPerUm
    
    if Contour_cell is not None:
        Poly_cell = Polygon(shell=Contour_cell)
        Poly_inner_cell = Poly_cell.buffer(-buffer_pix)
        for i, track in enumerate(rawTracks):
            nT = len(track)
            if is_track_in_polygon(track, Poly_inner_cell) and (nT >= nPoints_cuttoff):
                track = np.concat((track, np.ones((len(track[:,0]), 1), dtype=int) * (i+1)), axis = 1)
                track[:, 0] = track[:, 0].astype(int) + 1
                Valid_Tracks.append(track)
    
    else:
        height, width = shape
        for i, track in enumerate(rawTracks):
            nT = len(track)
            xm = np.median(track[:, 1])
            ym = np.median(track[:, 2])
            dist_x = min(np.abs(xm-0), np.abs(xm-width))
            dist_y = min(np.abs(ym-0), np.abs(ym-height))
            test_x = (dist_x < buffer_pix)
            test_y = (dist_y < buffer_pix)
            if (not (test_x or test_y)) and (nT >= nPoints_cuttoff):
                track = np.concat((track, np.ones((len(track[:,0]), 1), dtype=int) * (i+1)), axis = 1)
                track[:, 0] = track[:, 0].astype(int) + 1
                Valid_Tracks.append(track)
            
    concat_tracks = np.concat(Valid_Tracks, axis = 0)
    df = pd.DataFrame({column_names[k] : concat_tracks[:,k] for k in range(len(column_names))})
    df[column_names[0]] = df[column_names[0]].values.astype(int)
    df[column_names[3]] = df[column_names[3]].values.astype(int)
        
    
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
        
        
        
        ax = axes[0]
        if not (RefImg is None):
            ax.imshow(RefImg, cmap='gray')
        
        
        ax = axes[1]
        if not (RefImg is None):
            ax.imshow(RefImg, cmap='gray')
        if not (Contour_cell is None):
            y_ic, x_ic = Poly_inner_cell.exterior.xy
            Contour_inner_cell = np.array([y_ic, x_ic]).T
            ax.plot(Contour_cell[:,1], Contour_cell[:,0], ls='-', color='red', lw=1)
            ax.plot(Contour_inner_cell[:,1], Contour_inner_cell[:,0], ls='-', color='cyan', lw=1)
            
        for k in range(len(rawTracks)):
            track = rawTracks[k]
            color = ColorList[k%len(ColorList)]
            ax.plot(track[:,1], track[:,2], ls='-', color=color, lw=0.25)
        
        ax = axes[2]
        if not (RefImg is None):
            ax.imshow(RefImg, cmap='gray')
            
        for k in range(len(Valid_Tracks)):
            track = Valid_Tracks[k]
            color = ColorList[k%len(ColorList)]
            ax.plot(track[:,1], track[:,2], ls='-', color=color, lw=0.25)
        
        if SHOWPLOT:
            plt.show()
        
        if SAVEPLOT:
            figName = '_'.join(cleanTrackName.split('_')[:-1]) + '_cleanTracks.png'
            figPath = os.path.join(plotDir, figName)
            fig.savefig(figPath, dpi=500, )
    
    if not SHOWPLOT:
        plt.ion()
    
    return(df)

# %%%% MSD computation

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


# %%%% Tracks post-processing

def correct_drift_in_Tracks(df, fN, tP, dstDir, figDir,
                            PixPerUm, FPS, 
                            PLOT = False, SHOWPLOT = False, SAVEPLOT = False, 
                            export_image = False):
    if PLOT:
        i_eg = df['particle'].unique()[50]
        df_eg = df[df['particle']==i_eg]
        x_eg1 = df_eg['x'].to_numpy()
        y_eg1 = df_eg['y'].to_numpy()
        f_eg = df_eg['frame'].to_numpy()
    
    grouped = df.groupby('frame')
    list_df_f = [(f, df_f) for f, df_f in grouped]
    nF = len(list_df_f)
    Dxy = np.zeros((nF, 2))
    for (f1, df_f1), (f2, df_f2) in zip(list_df_f[:-1], list_df_f[1:]):
        df_f_m = pd.merge(df_f1, df_f2, on='particle', how='inner', suffixes=('_1', '_2'))
        df_f_m['dx'] = df_f_m['x_2'] - df_f_m['x_1']
        df_f_m['dy'] = df_f_m['y_2'] - df_f_m['y_1']
        Dxy[f1, 0] = np.mean(df_f_m['dx'].to_numpy()) #/ PixPerUm
        Dxy[f1, 1] = np.mean(df_f_m['dy'].to_numpy()) #/ PixPerUm
    
    sum_Dxy = np.array([[0, 0]] + [np.sum(Dxy[:k, :], axis=0) for k in range(1, nF)])
    
    sDxy = np.zeros_like(Dxy)
    sDxy = savgol_filter(Dxy, 5, 3, axis=0)
    sum_sDxy = np.array([[0, 0]] + [np.sum(sDxy[:k, :], axis=0) for k in range(1, nF)])
    
    Dxy_corr = np.copy(sum_sDxy)
    
    list_df_f = [(f, df_f) for f, df_f in grouped]
    list_results = []
    # list_results = [list_df_f[0][1]]
    for f, df_f in list_df_f: #[1:]:
        df_f['x'] = df_f['x'] - Dxy_corr[f-1, 0]
        df_f['y'] = df_f['y'] - Dxy_corr[f-1, 1]
        list_results.append(df_f)
    df2 = pd.concat(list_results, axis=0)
    df2.sort_values(by=["particle", "frame"], inplace=True)
    # print(df.head())
    # print(df2.head())
    
    if PLOT:
        if not SHOWPLOT:
            plt.ioff()
            
        # fig, axes = plt.subplots(2, 1, figsize=(7, 7), layout='compressed')
        pm.setGraphicOptions(mode='print')
        fig = plt.figure(layout="compressed", figsize=(8, 8))

        gs = mpl.gridspec.GridSpec(3, 3, figure=fig)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])
        ax3 = fig.add_subplot(gs[2:, 0])
        ax4 = fig.add_subplot(gs[2:, 1], sharex=ax3, sharey=ax3)
        ax5 = fig.add_subplot(gs[2:, 2], sharex=ax3, sharey=ax3)
        axes = [ax1, ax2, ax3, ax4, ax5]

        axes[0].set_ylabel('dx, dy')
        axes[1].set_ylabel('Smoothed dx, dy')
        ax = axes[0]
        ax.plot(np.arange(0, nF)*FPS, Dxy[:, 0]/PixPerUm, 'b-')
        ax.plot(np.arange(0, nF)*FPS, sum_Dxy[:, 0]/PixPerUm, ls='-', color='cyan')
        ax.plot(np.arange(0, nF)*FPS, Dxy[:, 1]/PixPerUm, 'r-')
        ax.plot(np.arange(0, nF)*FPS, sum_Dxy[:, 1]/PixPerUm, ls='-', color='orange')
        ax.grid()
        ax = axes[1]
        ax.plot(np.arange(0, nF)*FPS, sDxy[:, 0]/PixPerUm, 'b-')
        ax.plot(np.arange(0, nF)*FPS, sum_sDxy[:, 0]/PixPerUm, ls='-', color='cyan')
        ax.plot(np.arange(0, nF)*FPS, sDxy[:, 1]/PixPerUm, 'r-')
        ax.plot(np.arange(0, nF)*FPS, sum_sDxy[:, 1]/PixPerUm, ls='-', color='orange')
        ax.grid()
        ax.set_xlabel('Frame No.')
        
        
        # fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), layout='compressed',
        #                          sharex=True, sharey=True)
        df2_eg = df2[df2['particle']==i_eg]
        dx_eg, dy_eg = Dxy_corr[f_eg-1, 0], Dxy_corr[f_eg-1, 1]
        x_eg2 = df2_eg['x'].to_numpy()
        y_eg2 = df2_eg['y'].to_numpy()
        ax = axes[2]
        ax.set_aspect('equal', adjustable='box')
        ax.plot(x_eg1-x_eg1[0], y_eg1-y_eg1[0], 'b-')
        ax.grid()
        ax = axes[3]
        ax.set_aspect('equal', adjustable='box')
        ax.plot(dx_eg, dy_eg, 'k-')
        ax.grid()
        ax = axes[4]
        ax.set_aspect('equal', adjustable='box')
        ax.plot(x_eg2-x_eg2[0], y_eg2-y_eg2[0], 'r-')
        ax.grid()

        if SAVEPLOT:
            figName = fN + '_DriftCorr.png'
            figPath = os.path.join(figDir, figName)
            fig.savefig(figPath)
        if SHOWPLOT:
            plt.show()
        if not SHOWPLOT:
            plt.ion()

    if export_image:
        nT = nF
        img = ufun.load_stack_region(tP, time_indices=[i for i in range(nT)])
        tf_img = np.zeros_like(img)
        tf_img[0, :, :] = img[0, :, :]
        # tf_img = []
        # tf_img.append(img[0, :, :])
        for t in range(nT-1): # nF-1
            img_t = img[t+1, :, :]
            tform = skm.transform.EuclideanTransform(translation=(-sum_sDxy[t, 0], -sum_sDxy[t, 1]))
            tf_img[t+1] = skm.transform.warp(img_t, tform.inverse, preserve_range=True)
            # tf_img.append(skm.transform.warp(img_t, tform.inverse, preserve_range=True))
        tf_img = np.array(tf_img)
        fpath = os.path.join(dstDir, fN + '_DriftCorr.tif')
        skm.io.imsave(fpath, tf_img)
        
        
    return(df2, Dxy_corr)


#### Global modes QC & correction

# Measures three global (collective) modes per frame at lag tau_qc:
#   1. Translation  : mean (Δx, Δy) over all particles  [µm]
#   2. Rotation     : mean Δθ / r  = mean (Δy·er_x − Δx·er_y) / r   [rad/frame]
#   3. Breathing    : mean radial displacement Δr_r = Δr · ê_r         [µm/frame]
#
# These are measured in 'n_ann' equal-count annular bins for rotation and breathing,
# so a spatially inhomogeneous pattern (e.g. breathing only in the outer ring) is
# still detected and removed without polluting other rings.
#
# The function then optionally subtracts these modes from every displacement, writing
# corrected columns 'x_corr' / 'y_corr' back into the dataframe.
#
# Design choices (matching the report, section 2.4):
#   - lag = 1 frame (fastest timescale, so each original position contributes to its
#     own correction at most once per lag-step → auto-correlation bias is negligible)
#   - correction is cumulative: Δcorr(t) = Σ_{t'<t} [global-mode displacement at t']
#   - breathing and rotation are removed *per annulus* defined on the ORIGIN position,
#     consistent with how the report decouples radial inhomogeneity
#   - the centre (xC, yC) must be supplied in the same pixel units as the track coords
#
# Outputs (always):
#   dict_qc     – per-frame time series (DataFrames / arrays) and global scalars
#   df_corr     – dataframe with added 'x_corr' / 'y_corr' columns
#
# Outputs (if PLOT=True):
#   figure with four panels (a) positions, (b) global modes vs time,
#   (c) rotation vs time, (d) one example track before/after correction

def measure_and_correct_global_modes(df, fN, PixPerUm, FPS, xC, yC,
                                     tau_qc = 1,
                                     n_ann = 8,
                                     # correct = True,
                                     PLOT = False, SHOWPLOT = False, SAVEPLOT = False,
                                     figDir = None):
    """
    Measure the three collective motion modes (translation, rotation, breathing)
    per frame, then optionally subtract them from the track positions.

    Parameters
    ----------
    df          : DataFrame with columns ['frame', 'x', 'y', 'particle']
    fN          : file name root (used only for saving the figure)
    PixPerUm    : pixel-per-micron scale of the acquisition
    FPS         : frames per second
    xC, yC      : centre of the polar coordinate system (pixels, same units as df.x/y)
    tau_qc      : lag in frames used to measure the modes (default 1)
    n_ann       : number of equal-population annular bins for rotation/breathing removal
    # correct     : if True, write 'x_corr'/'y_corr' columns into the returned dataframe
    PLOT        : produce the QC figure
    SHOWPLOT    : call plt.show() (False → figure saved but not displayed)
    SAVEPLOT    : save the figure to figDir
    figDir      : directory for the figure (required when SAVEPLOT=True)

    Returns
    -------
    dict_qc     : dict with keys
                    'translation_pix'   – (nFrames, 2) array, mean (Δx, Δy) per frame [pix]
                    'rotation_rad'      – (nFrames,) array, mean Δθ/r per frame [rad]
                    'breathing_pix'     – (nFrames,) array, mean Δr_r per frame [pix]
                    'translation_cum_pix' – cumulative translation [pix]
                    'frames'            – frame index array
                    'rms_translation_um_per_s'  – scalar, rms translation speed [µm/s]
                    'mean_rotation_rad_per_s'   – scalar, mean |rotation| speed [rad/s]
                    'rms_breathing_um_per_s'    – scalar, rms breathing speed [µm/s]
    df_corr     : input df with two extra columns 'x_corr' and 'y_corr'
                    (copies of x, y when correct=False)
    """
    
    df = df.copy().sort_values(['particle', 'frame']).reset_index(drop=True)
    
    # ---- Build per-consecutive-frame displacement table ----
    # Merge frame t and frame t+tau_qc on particle; keeps only particles present at both.
    frames_sorted = np.sort(df['frame'].unique())
    nF = len(frames_sorted)
    
    # Pre-allocate per-frame mode arrays; indexed by position in frames_sorted
    Translation   = np.zeros((nF, 2))   # (Δx, Δy) [pix]
    Rotation      = np.zeros(nF)        # mean Δθ/r [rad / frame]
    Breathing     = np.zeros(nF)        # mean Δr_r  [pix / frame]
    
    # Annular bin edges in pixels, defined on the FULL set of positions
    # (consistent with the report's convention: bins have equal particle count)
    r_all = np.hypot(df['x'].values - xC, df['y'].values - yC)
    ann_quantiles = np.linspace(0, 1, n_ann + 1)
    ann_edges = np.quantile(r_all, ann_quantiles)
    ann_edges[0]  -= 1e-6
    ann_edges[-1] += 1e-6
    
    # Group by frame for fast merging
    grouped = {f: df_f for f, df_f in df.groupby('frame')}
    
    for idx, f1 in enumerate(frames_sorted[:-tau_qc]):
        f2 = frames_sorted[idx + tau_qc]
        df_f1 = grouped[f1]
        df_f2 = grouped[f2]
        
        merged = pd.merge(df_f1[['particle', 'x', 'y']],
                          df_f2[['particle', 'x', 'y']],
                          on='particle', how='inner', suffixes=('_1', '_2'))
        
        if len(merged) < 5:
            continue
        
        x1, y1 = merged['x_1'].values, merged['y_1'].values
        dx      = merged['x_2'].values - x1
        dy      = merged['y_2'].values - y1
        
        # 1. Translation: simple mean displacement
        Translation[idx, 0] = np.mean(dx)
        Translation[idx, 1] = np.mean(dy)
        
        # 2 & 3. Rotation and breathing in polar coords centred on (xC, yC)
        rx, ry  = x1 - xC, y1 - yC
        r       = np.hypot(rx, ry)
        ok      = r > 1e-9    # avoid division by zero right at centre
        er_x, er_y = rx / np.where(ok, r, 1), ry / np.where(ok, r, 1)
        et_x, et_y = -er_y, er_x              # orthoradial unit vector (90° CCW)
        
        d_r   = dx * er_x + dy * er_y         # radial component
        d_th  = (dx * et_x + dy * et_y) / np.where(ok, r, 1)  # angular velocity [rad]
        
        # Average per annulus, then take the global mean of annular means
        ann_idx = np.clip(np.digitize(r, ann_edges) - 1, 0, n_ann - 1)
        ann_mean_dr  = np.array([np.mean(d_r[ok & (ann_idx == a)])
                                 if np.any(ok & (ann_idx == a)) else 0.0
                                 for a in range(n_ann)])
        ann_mean_dth = np.array([np.mean(d_th[ok & (ann_idx == a)])
                                 if np.any(ok & (ann_idx == a)) else 0.0
                                 for a in range(n_ann)])
        
        Breathing[idx] = np.mean(ann_mean_dr)
        Rotation[idx]  = np.mean(ann_mean_dth)
    
    # ---- Cumulative correction ----
    # Cumulative sum of frame-to-frame translation gives the drift to subtract
    # from position at frame t.  Frame index starts at frames_sorted[0].
    Translation_cum = np.cumsum(Translation, axis=0)
    Translation_cum = np.vstack([[0, 0], Translation_cum[:-1]])  # shift: frame[0] stays put
    
    # ---- Apply correction to track positions ----
    df_corr = df.copy()
    # if correct:
    frame_to_idx = {f: i for i, f in enumerate(frames_sorted)}
    idx_col = df_corr['frame'].map(frame_to_idx).values
    df_corr['x_corr'] = df_corr['x'].values - Translation_cum[idx_col, 0]
    df_corr['y_corr'] = df_corr['y'].values - Translation_cum[idx_col, 1]
    # else:
        # df_corr['x_corr'] = df_corr['x'].values
        # df_corr['y_corr'] = df_corr['y'].values
    
    # ---- Scalar summary ----
    dt = tau_qc / FPS       # seconds per lag step
    rms_translation = np.sqrt(np.mean(Translation[:, 0]**2 + Translation[:, 1]**2)) / PixPerUm / dt
    mean_rotation   = np.mean(np.abs(Rotation)) / dt
    rms_breathing   = np.sqrt(np.mean(Breathing**2)) / PixPerUm / dt
    
    dict_qc = {
        'frames'                     : frames_sorted,
        'translation_pix'            : Translation,
        'translation_cum_pix'        : Translation_cum,
        'rotation_rad'               : Rotation,
        'breathing_pix'              : Breathing,
        'rms_translation_um_per_s'   : rms_translation,
        'mean_rotation_rad_per_s'    : mean_rotation,
        'rms_breathing_um_per_s'     : rms_breathing,
    }
    
    # ---- Figure ----
    if PLOT:
        if not SHOWPLOT:
            plt.ioff()
        else:
            plt.ion()
        
        pm.setGraphicOptions(mode='screen')
        t_axis = frames_sorted / FPS   # time in seconds
        
        fig, axes = plt.subplots(2, 3, figsize=(17, 8), layout='compressed')
        fig.suptitle(fN + ' — Global modes QC')
        axes[0, 2].set_visible(False)   # top-right cell unused in 2×3 layout
        
        # (a) particle positions + centre
        ax = axes[0, 0]
        sub = df.iloc[::max(1, len(df)//5000)]   # downsample for speed
        ax.scatter(sub['x'], sub['y'], s=0.3, c='0.6', rasterized=True)
        ax.plot(xC, yC, 'r+', ms=14, mew=2, label='centre')
        ax.set_aspect('equal', adjustable='datalim')
        ax.invert_yaxis()
        ax.set_xlabel('x (pix)')
        ax.set_ylabel('y (pix)')
        ax.set_title('(a) Positions & centre')
        ax.legend()
        
        # (b) translation (Δx, Δy) and cumulative drift over time
        ax = axes[0, 1]
        ax.plot(t_axis, Translation[:, 0] / PixPerUm, color='tab:blue',  lw=0.8, label='Δx (µm)')
        ax.plot(t_axis, Translation[:, 1] / PixPerUm, color='tab:red',   lw=0.8, label='Δy (µm)')
        ax.plot(t_axis, Translation_cum[:, 0] / PixPerUm, color='tab:blue',  lw=1.5, ls='--', alpha=0.7, label='Σ Δx')
        ax.plot(t_axis, Translation_cum[:, 1] / PixPerUm, color='tab:red',   lw=1.5, ls='--', alpha=0.7, label='Σ Δy')
        ax.axhline(0, c='k', lw=0.5)
        ax.grid()
        ax.set_xlabel('t (s)')
        ax.set_ylabel('displacement (µm)')
        ax.set_title(f'(b) Translation  rms={rms_translation*1e3:.1f} nm/s')
        ax.legend(fontsize=8, ncols=2)
        
        # (c) rotation and breathing vs time, each with its cumulative integral
        # The integrals (cumulative sums) reveal whether modes drift in one direction
        # over the film or merely fluctuate around zero — analogous to sum_Dxy for translation.
        # Rotation integral: cumulative angle [rad]  (like a slow spin of the whole cell)
        # Breathing integral: cumulative radial displacement [µm]  (like slow expansion/contraction)
        Rotation_cum  = np.cumsum(Rotation)                   # [rad]
        Breathing_cum = np.cumsum(Breathing) / PixPerUm       # [µm]
    
        ax = axes[1, 0]
        ax.set_title(f'(c) Rotation — per frame & integral')
        ax.plot(t_axis, Rotation / dt,   color='tab:green', lw=0.8, label=f'Ω (rad/s), rms={mean_rotation:.1e}')
        ax.axhline(0, c='k', lw=0.5)
        ax.set_xlabel('t (s)')
        ax.set_ylabel('Ω (rad/s)', color='tab:green')
        ax.tick_params(axis='y', colors='tab:green')
        ax.grid()
        ax2 = ax.twinx()
        ax2.plot(t_axis, Rotation_cum,   color='tab:green', lw=1.5, ls='--', alpha=0.7,
                 label='Σ Ω·Δt (rad)')
        ax2.set_ylabel('cumulative angle (rad)', color='tab:green', alpha=0.7)
        ax2.tick_params(axis='y', colors='tab:green')
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    
        ax = axes[1, 1]
        ax.set_title(f'(d) Breathing — per frame & integral')
        ax.plot(t_axis, Breathing / PixPerUm / dt, color='tab:orange', lw=0.8,
                label=f'ṙ (µm/s), rms={rms_breathing*1e3:.1f} nm/s')
        ax.axhline(0, c='k', lw=0.5)
        ax.set_xlabel('t (s)')
        ax.set_ylabel('ṙ (µm/s)', color='tab:orange')
        ax.tick_params(axis='y', colors='tab:orange')
        ax.grid()
        ax2 = ax.twinx()
        ax2.plot(t_axis, Breathing_cum, color='tab:orange', lw=1.5, ls='--', alpha=0.7,
                 label='Σ ṙ·Δt (µm)')
        ax2.set_ylabel('cumulative radial displacement (µm)', color='tab:orange', alpha=0.7)
        ax2.tick_params(axis='y', colors='tab:orange')
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
        
        # (e) one example track before and after correction
        ax = axes[1, 2]
        pid_example = df['particle'].unique()[len(df['particle'].unique()) // 2]
        df_eg  = df[df['particle'] == pid_example].sort_values('frame')
        df_eg2 = df_corr[df_corr['particle'] == pid_example].sort_values('frame')
        x1_eg  = df_eg['x'].values
        y1_eg  = df_eg['y'].values
        x2_eg  = df_eg2['x_corr'].values
        y2_eg  = df_eg2['y_corr'].values
        ax.plot((x1_eg - x1_eg[0]) / PixPerUm, (y1_eg - y1_eg[0]) / PixPerUm,
                color='tab:blue', lw=0.8, label='raw')
        ax.plot((x2_eg - x2_eg[0]) / PixPerUm, (y2_eg - y2_eg[0]) / PixPerUm,
                color='tab:red',  lw=0.8, label='corrected (x & y)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('Δx (µm)')
        ax.set_ylabel('Δy (µm)')
        ax.set_title(f'(e) Example track (particle {pid_example})')
        ax.legend()
        ax.grid()
        
        if SHOWPLOT:
            plt.show()
        
        if SAVEPLOT and figDir is not None:
            figName = fN + '_GlobalModesQC.png'
            figPath = os.path.join(figDir, figName)
            fig.savefig(figPath, dpi=150)
        
        if not SHOWPLOT:
            plt.ion()

    return(dict_qc, df_corr)


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

    grouped = df.groupby('particle')
    for pid, df_p in grouped:
        frames = df_p['frame'].to_numpy()
        x = df_p['x'].to_numpy()
        y = df_p['y'].to_numpy()

        pfi = frames.min()
        pff = frames.max()

        for fi, ff in zip(FI, FF):
            if pfi <= fi + 1 and ff <= pff:
                mask = (fi <= frames - 1) & (frames - 1 < ff)
                xm = np.median(x[mask])
                ym = np.median(y[mask])

                key = f'{fi}_{ff}'
                result = dict_TRanges2particles[key]
                result['pid'].append(int(pid))
                result['xm'].append(float(xm))
                result['ym'].append(float(ym))
    
    for k, TRange in enumerate(list_TRanges):
        df_parts = pd.DataFrame(dict_TRanges2particles[TRange])
        XY = np.array([df_parts['xm'].values[:],
                       df_parts['ym'].values[:]]).T
        
        tri = Delaunay(XY)
        edges_short, _ = tri_to_short_edges(tri, XY, dist_th)
        close_pairs = df_parts['pid'].values[edges_short]
        
        list_Pairs.append(np.array(close_pairs))    
            
    return(list_TRanges, list_Pairs)




def get_pairs_for_TRanges(df, PixPerUm, FPS, Nframes,
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
    FI, FF = map(int, TRange.split('_'))

    # Convert once
    frames = df['frame'].to_numpy(dtype=int)
    particles = df['particle'].to_numpy(dtype=int)
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()

    # Select requested time range
    frame_mask = (frames >= FI + 1) & (frames <= FF)

    frames = frames[frame_mask]
    particles = particles[frame_mask]
    x = x[frame_mask]
    y = y[frame_mask]

    # Keep only particles that occur in pairs
    ids_in_pairs = np.unique(pairs)
    particle_mask = np.isin(particles, ids_in_pairs)

    frames = frames[particle_mask]
    particles = particles[particle_mask]
    x = x[particle_mask]
    y = y[particle_mask]

    # Create a lookup: particle -> (x, y) arrays
    particle_data = {
        pid: (x[particles == pid], y[particles == pid])
        for pid in ids_in_pairs
    }

    N = FF - FI
    T_array = np.arange(1, N + 1)

    list_df_pairs = []

    for pair_id, (id1, id2) in enumerate(pairs):

        x1, y1 = particle_data[id1]
        x2, y2 = particle_data[id2]

        Xpair = x2 - x1
        Ypair = y2 - y1

        list_df_pairs.append(
            pd.DataFrame({
                'pair_id': pair_id,
                'x': Xpair,
                'y': Ypair,
                'frame': T_array
            })
        )
        
    output = pd.concat(list_df_pairs, ignore_index=True)
    return(output)



def make_film_of_pairs_V1(df, PixPerUm, FPS, shape, dirPlots,
                          plot_frames, dist_th_um = 4):  
    plt.ioff()
    pm.setGraphicOptions(mode='print')
    nT, nY, nX = shape
    df.frame = df.frame.astype(int)
    df.particle = df.particle.astype(int)
    dist_th = dist_th_um * PixPerUm
    
    # df_f = df[(df['frame']>fi) & (df['frame']<=ff)]
    df_f = df[df['frame'].apply(lambda x : x in plot_frames)]
    
    valid_particles = []
    
    grouped = df_f.groupby('particle')
    for pid, df_p in grouped:
        if len(df_p) == len(plot_frames):
        # if len(df_p) == (ff-fi):
            valid_particles.append(pid)
            
    valid_particles = np.array(valid_particles)
    df_f = df_f[df_f['particle'].apply(lambda x : x in valid_particles)]
    
    grouped = df_f.groupby('frame')
    for f, df_ff in grouped:
        XY = np.array([df_ff['x'].to_numpy(),
                       df_ff['y'].to_numpy()]).T
        
        tri = Delaunay(XY)
        edges_short, _ = tri_to_short_edges(tri, XY, dist_th)
        # close_pairs = df_ff['particle'].to_numpy()[edges_short]
        
        fig, axes = plt.subplots(1, 2, figsize = (6, 3), layout='compressed')
    
        ax = axes[0]
        ax.set_xlim([0, nX])
        ax.set_ylim([0, nY])
        ax.triplot(XY[:, 0], XY[:, 1], tri.simplices, 'g-', lw=0.75)
        ax.plot(XY[:, 0], XY[:, 1], ls='', marker='o', ms=1.5)
    
        ax = axes[1]
        ax.set_xlim([0, nX])
        ax.set_ylim([0, nY])
        ax.plot(XY[:, 0], XY[:, 1], ls='', marker='o', ms=1.5, zorder=8)
        for p in edges_short:
            i, j = p
            ax.plot([XY[i, 0], XY[j, 0]], [XY[i, 1], XY[j, 1]], 'g-', lw=0.75, zorder=7)
        
    
        fig.savefig(os.path.join(dirPlots, f'triA_{f:04d}.png'), dpi=200)
        plt.close('all')
        
    plt.ion()



def make_film_of_pairs_V2(df, PixPerUm, FPS, shape, dirPlots, 
                          plot_frames, dist_th_um = 4):
    plt.ioff()
    pm.setGraphicOptions(mode='print')
    # PixPerUm = cd.PixPerUm_60X_W1
    # FPS = 20
    # plot_frames = np.arange(1, 200, 5)
    # dist_th_um = 4
    nT, nY, nX = shape
    df.frame = df.frame.astype(int)
    df.particle = df.particle.astype(int)
    dist_th = dist_th_um * PixPerUm
    
    # df_f = df[(df['frame']>fi) & (df['frame']<=ff)]
    df_f = df[df['frame'].apply(lambda x : x in plot_frames)]
    
    valid_particles = []
    
    grouped = df_f.groupby('particle')
    for pid, df_p in grouped:
        if len(df_p) == len(plot_frames):
        # if len(df_p) == (ff-fi):
            valid_particles.append(pid)
            
    valid_particles = np.array(valid_particles)
    df_f = df_f[df_f['particle'].apply(lambda x : x in valid_particles)]
    
    grouped = df_f.groupby('particle')
    df_p = grouped.agg({'x':'median', 'y':'median'})
    XY = np.array([df_p['x'].to_numpy(),
                   df_p['y'].to_numpy()]).T
    tri = Delaunay(XY)
    edges_short, _ = tri_to_short_edges(tri, XY, dist_th)
    
    
    grouped = df_f.groupby('frame')
    for f, df_ff in grouped:
        XY = np.array([df_ff['x'].to_numpy(),
                       df_ff['y'].to_numpy()]).T
        
        # tri = Delaunay(XY)
        # edges_short, _ = tri_to_short_edges(tri, XY, dist_th)
        # close_pairs = df_ff['particle'].to_numpy()[edges_short]
        
        fig, axes = plt.subplots(1, 2, figsize = (6, 3), layout='compressed')
    
        ax = axes[0]
        ax.set_xlim([0, nX])
        ax.set_ylim([0, nY])
        ax.triplot(XY[:, 0], XY[:, 1], tri.simplices, 'g-', lw=0.75)
        ax.plot(XY[:, 0], XY[:, 1], ls='', marker='o', ms=1.5)
    
        ax = axes[1]
        ax.set_xlim([0, nX])
        ax.set_ylim([0, nY])
        ax.plot(XY[:, 0], XY[:, 1], ls='', marker='o', ms=1.5, zorder=8)
        for p in edges_short:
            i, j = p
            ax.plot([XY[i, 0], XY[j, 0]], [XY[i, 1], XY[j, 1]], 'g-', lw=0.75, zorder=7)
        
    
        fig.savefig(os.path.join(dirPlots, f'triB_{f:04d}.png'), dpi=200)
        plt.close('all')
        
    plt.ion()
    
    
    
def compute_pairwise_MSD(df, PixPerUm, FPS, Nframes,
                         len_TRanges = 100, delta_TRanges = -1, 
                         dist_th_um = 4, max_Dt_s = 2.5):
    
    list_TRanges, list_Pairs = get_pairs_for_TRanges_Delaunay(
        df, PixPerUm, FPS, Nframes,
        len_TRanges = 100, 
        delta_TRanges = -1,
        dist_th_um = 4
        )
    
    pairMSD = []
    max_lagtime = int(max_Dt_s * FPS)
    
    for k, TRange in enumerate(list_TRanges):
        pairs = list_Pairs[k]
        df_pairs = get_relative_displacement_by_TRange(df, pairs, TRange)
        res_pair_emsd = tp.motion.emsd(df_pairs.rename(columns={'pair_id':'particle'}), 
                                       1/PixPerUm, FPS, max_lagtime=max_lagtime).reset_index()
        pairMSD.append(res_pair_emsd)

    df_allTRange_pair_MSD = pd.concat([pairMSD[0]['lagt']] + [df['msd'] for df in pairMSD], axis=1)
    df_allTRange_pair_MSD.columns = ['lagt'] + [f'msd_{tr}' for tr in list_TRanges]
    median_msd = np.median(df_allTRange_pair_MSD.loc[:, [f'msd_{tr}' for tr in list_TRanges]].to_numpy(), 
                           axis = 1)

    df_allTRange_pair_MSD['msd_median'] = median_msd
    
    return(df_allTRange_pair_MSD)


# %%%% V1 - Displacement correlations C(tau, R) and rho(tau, R) 

# Computes, for each lag tau and inter-particle distance bin R:
#
#   For projection = 'r' or 'theta' (scalar projection a = u · Δr):
#     C_aa   = <a_i * a_j>  -  <a_i><a_j>
#     Var_a  = <a²> - <a>²
#     rho_aa = C_aa / sqrt(Var_i * Var_j)
#
#   For projection = 'full' (tensor trace, dot product):
#     C_full = <Δr_i · Δr_j>  -  <Δx_i><Δx_j>  -  <Δy_i><Δy_j>
#            = C_xx + C_yy  (lab-frame tensor trace)
#            ≈ C_rr + C_θθ  (barycenter polar trace — equal only if all barycenters
#              had the same orientation, which they don't; use 'r'+'theta' for that)
#     Var_i  = <|Δr_i|²>  -  <Δx_i>²  -  <Δy_i>²
#     rho    = C_full / sqrt(Var_i * Var_j)
#
#   # TAG: add 'L' and 'T' (pair-frame longitudinal/transverse) projections here
#
# Null model (section 4.4 of the report, "Modèle nul par décalage temporel"):
#   For each origin t, particle j's displacement is replaced by that of its nearest
#   spatial neighbour at a time-shifted origin t' = t + null_shift (mod film duration).
#
# Global mode subtraction (section 2.4 / 4.3):
#   Per origin: subtract mean (Δx, Δy) [translation], then per annulus subtract
#   mean Δr_r [breathing] and mean Δr_θ [rotation].  Each mode is independently togglable.


# def compute_displacement_correlations(df, PixPerUm, FPS, xC, yC,
#                                       tau_s_list     = [0.1, 0.5, 2.0, 5.0, 10.0, 20.0, 50.0],
#                                       R_bin_edges_um = np.arange(2, 22, 2),
#                                       projection     = 'full',
#                                       correct_modes  = True,
#                                       correct_xy     = True,
#                                       correct_r      = True,
#                                       correct_rot    = True,
#                                       n_ann_modes    = 8,
#                                       rbar_min_um    = 3.0,
#                                       R_min_um       = 2.0,
#                                       null_shift_s   = 30.0,
#                                       null_match_um  = 1.5,
#                                       origin_step    = 10):
#     """
#     Compute displacement correlation C(tau, R) and rho(tau, R).

#     Parameters
#     ----------
#     df              : DataFrame with columns ['frame', 'x', 'y', 'particle'], positions in pixels.
#     PixPerUm        : pixel-per-micron scale.
#     FPS             : frames per second.
#     xC, yC          : centre of the polar coordinate system (pixels).
#     tau_s_list      : list of lag times in seconds.
#     R_bin_edges_um  : 1-D array of bin edges for the inter-particle distance R (µm).
#     projection      : 'full'  – tensor trace C_xx + C_yy  (Δr_i · Δr_j dot product, lab frame)
#                       'r'     – project on ê_r at the pair barycenter → C_rr
#                       'theta' – project on ê_θ at the pair barycenter → C_θθ
#                       # TAG: 'L' and 'T' (pair-frame longitudinal/transverse) to be added here
#     correct_modes   : if True, subtract global modes before computing correlations.
#     correct_xy      : subtract mean (Δx, Δy) per origin [translation].
#     correct_r       : subtract mean Δr_r per annulus [breathing].
#     correct_rot     : subtract mean Δr_θ per annulus [rotation].
#     n_ann_modes     : number of equal-population annular bins for mode subtraction.
#     rbar_min_um     : exclude pairs whose barycenter is closer than this to the centre.
#     R_min_um        : exclude pairs closer than this (risk of detection overlaps).
#     null_shift_s    : time shift for the null model (seconds).
#     null_match_um   : max spatial distance for the nearest-neighbour match in the null model.
#     origin_step     : subsample origins; only frames where (frame - f_min) % origin_step == 0
#                       are used.

#     Returns
#     -------
#     tau_s_arr    : 1-D array, lag times (s), shape (n_tau,)
#     R_centers_um : 1-D array, bin centres (µm), shape (n_R,)
#     C            : 2-D array, covariance, shape (n_tau, n_R)
#     rho          : 2-D array, correlation coefficient, shape (n_tau, n_R)
#     C_null       : 2-D array, null-model covariance, shape (n_tau, n_R)
#     """

#     df = df.sort_values(['particle', 'frame']).reset_index(drop=True)
#     UmPerPix = 1.0 / PixPerUm

#     # Lag times in frames, deduplicated
#     seen = set()
#     tau_frames_unique = [t for t in [int(round(tau_s * FPS)) for tau_s in tau_s_list]
#                          if not (t in seen or seen.add(t))]

#     n_tau        = len(tau_frames_unique)
#     n_R          = len(R_bin_edges_um) - 1
#     R_centers_um = 0.5 * (R_bin_edges_um[:-1] + R_bin_edges_um[1:])
#     tau_s_arr    = np.array([t / FPS for t in tau_frames_unique])

#     # Positions in µm, centred on (xC, yC)
#     frames_all   = df['frame'].values
#     x_um_all     = (df['x'].values - xC) * UmPerPix
#     y_um_all     = (df['y'].values - yC) * UmPerPix
#     particle_all = df['particle'].values

#     frames_unique = np.unique(frames_all)
#     frame_to_rows = {f: np.where(frames_all == f)[0] for f in frames_unique}

#     null_shift_fr = int(round(null_shift_s * FPS))
#     f_min = int(frames_unique.min())
#     f_max = int(frames_unique.max())
#     f_range = f_max - f_min + 1

#     # Equal-population annular edges for mode subtraction
#     r_all        = np.hypot(x_um_all, y_um_all)
#     ann_edges_um = np.quantile(r_all[r_all > 0], np.linspace(0, 1, n_ann_modes + 1))
#     ann_edges_um[0]  -= 1e-6
#     ann_edges_um[-1] += 1e-6

#     # --- accumulators ---
#     # Scalar projections ('r', 'theta'):
#     #   S_ai, S_aj  : Σ a_i,  Σ a_j
#     #   S_ai2, S_aj2: Σ a_i², Σ a_j²
#     #   S_aiaj      : Σ a_i*a_j
#     #
#     # Full projection ('full'):
#     #   S_ai, S_aj  : Σ Δx_i, Σ Δx_j         (x-component mean for bias correction)
#     #   S_bi, S_bj  : Σ Δy_i, Σ Δy_j         (y-component mean for bias correction)
#     #   S_ai2, S_aj2: Σ |Δr_i|², Σ |Δr_j|²   (for variance)
#     #   S_aiaj      : Σ (Δr_i · Δr_j)         (dot product cross term)
#     # C_full = S_aiaj/n - (S_ai/n)(S_aj/n) - (S_bi/n)(S_bj/n)
#     # Var_i  = S_ai2/n  - (S_ai/n)²        - (S_bi/n)²

#     def _empty():
#         return np.zeros((n_tau, n_R), dtype=np.float64)

#     S_ai   = _empty(); S_aj   = _empty()
#     S_bi   = _empty(); S_bj   = _empty()   # y-component means (only used for 'full')
#     S_ai2  = _empty(); S_aj2  = _empty()
#     S_aiaj = _empty(); S_n    = _empty()

#     S_ai_null   = _empty(); S_aj_null   = _empty()
#     S_bi_null   = _empty(); S_bj_null   = _empty()
#     S_ai2_null  = _empty(); S_aj2_null  = _empty()
#     S_aiaj_null = _empty(); S_n_null    = _empty()

#     is_full = (projection == 'full')

#     # ----- main loop over lag times -----
#     for tau_idx, tau_fr in enumerate(tau_frames_unique):

#         origins = frames_unique[frames_unique + tau_fr <= f_max]
#         origins = origins[(origins - f_min) % origin_step == 0]

#         for o in origins:
#             rows_i = frame_to_rows.get(o,         None)
#             rows_j = frame_to_rows.get(o + tau_fr, None)
#             if rows_i is None or rows_j is None:
#                 continue

#             # particles present at both t=o and t=o+tau
#             pid_i = particle_all[rows_i]; pid_j = particle_all[rows_j]
#             pid_common = np.intersect1d(pid_i, pid_j)
#             if len(pid_common) < 2:
#                 continue

#             # sort both sets so particle ordering matches
#             si = np.argsort(pid_i[np.isin(pid_i, pid_common)])
#             sj = np.argsort(pid_j[np.isin(pid_j, pid_common)])
#             idx_i = rows_i[np.isin(pid_i, pid_common)][si]
#             idx_j = rows_j[np.isin(pid_j, pid_common)][sj]

#             x0 = x_um_all[idx_i]; y0 = y_um_all[idx_i]
#             dx = x_um_all[idx_j] - x0
#             dy = y_um_all[idx_j] - y0

#             # ---- global mode subtraction ----
#             if correct_modes:
#                 dx, dy = _subtract_global_modes(x0, y0, dx, dy, ann_edges_um,
#                                                 correct_xy, correct_r, correct_rot)

#             # ---- null-model origin o' ----
#             o_null_target = int((o - f_min + null_shift_fr) % f_range + f_min)
#             o_null = frames_unique[np.argmin(np.abs(frames_unique - o_null_target))]
#             if abs(o_null - o) < null_shift_fr // 2:
#                 o_null = None
#             if o_null is not None and frame_to_rows.get(o_null + tau_fr) is None:
#                 o_null = None

#             # ---- pair geometry ----
#             pts    = np.c_[x0, y0]
#             tree_i = cKDTree(pts)
#             pairs  = tree_i.query_pairs(R_bin_edges_um[-1], output_type='ndarray')
#             if len(pairs) == 0:
#                 continue

#             p, q = pairs[:, 0], pairs[:, 1]
#             bx = (x0[p] + x0[q]) / 2;  by = (y0[p] + y0[q]) / 2
#             rb = np.hypot(bx, by)
#             R  = np.hypot(x0[q] - x0[p], y0[q] - y0[p])

#             keep = (R >= R_min_um) & (rb >= rbar_min_um)
#             p, q, R, bx, by, rb = p[keep], q[keep], R[keep], bx[keep], by[keep], rb[keep]
#             if len(p) == 0:
#                 continue

#             iR       = np.digitize(R, R_bin_edges_um) - 1
#             in_range = (iR >= 0) & (iR < n_R)
#             p, q, iR, bx, by, rb = [v[in_range] for v in (p, q, iR, bx, by, rb)]
#             if len(p) == 0:
#                 continue

#             # ---- per-pair quantities ----
#             if is_full:
#                 # a = Δx (for mean bias), b = Δy (for mean bias)
#                 # ai2 = |Δr|² (for variance), cross = Δr_i · Δr_j
#                 ai = dx[p]; aj = dx[q]          # x-components (bias terms)
#                 bi = dy[p]; bj = dy[q]          # y-components (bias terms)
#                 ai2 = dx[p]**2 + dy[p]**2       # |Δr_i|²
#                 aj2 = dx[q]**2 + dy[q]**2       # |Δr_j|²
#                 cross = dx[p]*dx[q] + dy[p]*dy[q]  # Δr_i · Δr_j
#             else:
#                 a   = _project(x0, y0, dx, dy, projection)
#                 ai  = a[p]; aj = a[q]
#                 bi  = None; bj = None
#                 ai2 = ai**2; aj2 = aj**2
#                 cross = ai * aj

#             # ---- accumulate ----
#             for k_R in range(n_R):
#                 m = iR == k_R
#                 if not m.any():
#                     continue
#                 S_ai  [tau_idx, k_R] += ai[m].sum()
#                 S_aj  [tau_idx, k_R] += aj[m].sum()
#                 S_ai2 [tau_idx, k_R] += ai2[m].sum()
#                 S_aj2 [tau_idx, k_R] += aj2[m].sum()
#                 S_aiaj[tau_idx, k_R] += cross[m].sum()
#                 S_n   [tau_idx, k_R] += m.sum()
#                 if is_full:
#                     S_bi[tau_idx, k_R] += bi[m].sum()
#                     S_bj[tau_idx, k_R] += bj[m].sum()

#             # ---- null model ----
#             if o_null is not None:
#                 rows_null_0   = frame_to_rows[o_null]
#                 rows_null_tau = frame_to_rows[o_null + tau_fr]
 
#                 # Keep only particles present at both null frames (same intersection
#                 # logic as for the real origin above)
#                 pid_null_0   = particle_all[rows_null_0]
#                 pid_null_tau = particle_all[rows_null_tau]
#                 pid_null_common = np.intersect1d(pid_null_0, pid_null_tau)
 
#                 sn0  = np.argsort(pid_null_0  [np.isin(pid_null_0,   pid_null_common)])
#                 snt  = np.argsort(pid_null_tau [np.isin(pid_null_tau, pid_null_common)])
#                 idx_null_0   = rows_null_0  [np.isin(pid_null_0,   pid_null_common)][sn0]
#                 idx_null_tau = rows_null_tau[np.isin(pid_null_tau, pid_null_common)][snt]
 
#                 x0_no   = x_um_all[idx_null_0]; y0_no   = y_um_all[idx_null_0]
#                 dx_null = x_um_all[idx_null_tau] - x0_no
#                 dy_null = y_um_all[idx_null_tau] - y0_no

#                 if correct_modes:
#                     dx_null, dy_null = _subtract_global_modes(
#                         x0_no, y0_no, dx_null, dy_null, ann_edges_um,
#                         correct_xy, correct_r, correct_rot)

#                 # nearest neighbour in null frame for every particle i at origin o
#                 tree_null      = cKDTree(np.c_[x0_no, y0_no])
#                 dist_nn, idx_nn = tree_null.query(pts, k=1)
#                 matched        = dist_nn < null_match_um

#                 dx_nn = dx_null[idx_nn]; dy_nn = dy_null[idx_nn]

#                 if is_full:
#                     # null aj = Δx_null[j_match], bj = Δy_null[j_match]
#                     # cross_null = Δr_i(real) · Δr_j(null)
#                     pass    # computed inside the R-bin loop below
#                 else:
#                     a_null = _project(x0_no[idx_nn], y0_no[idx_nn], dx_nn, dy_nn, projection)

#                 for k_R in range(n_R):
#                     m   = iR == k_R
#                     if not m.any():
#                         continue
#                     p_m = p[m]; q_m = q[m]
#                     ok  = matched[p_m] & matched[q_m]
#                     if not ok.any():
#                         continue

#                     if is_full:
#                         ai_n   = dx[p_m[ok]];         aj_n  = dx_nn[q_m[ok]]
#                         bi_n   = dy[p_m[ok]];         bj_n  = dy_nn[q_m[ok]]
#                         ai2_n  = dx[p_m[ok]]**2 + dy[p_m[ok]]**2
#                         aj2_n  = dx_nn[q_m[ok]]**2 + dy_nn[q_m[ok]]**2
#                         cross_n = dx[p_m[ok]]*dx_nn[q_m[ok]] + dy[p_m[ok]]*dy_nn[q_m[ok]]
#                     else:
#                         ai_n    = ai[m][ok]
#                         aj_n    = a_null[q_m[ok]]
#                         bi_n    = None; bj_n = None
#                         ai2_n   = ai_n**2; aj2_n = aj_n**2
#                         cross_n = ai_n * aj_n

#                     S_ai_null  [tau_idx, k_R] += ai_n.sum()
#                     S_aj_null  [tau_idx, k_R] += aj_n.sum()
#                     S_ai2_null [tau_idx, k_R] += ai2_n.sum()
#                     S_aj2_null [tau_idx, k_R] += aj2_n.sum()
#                     S_aiaj_null[tau_idx, k_R] += cross_n.sum()
#                     S_n_null   [tau_idx, k_R] += ok.sum()
#                     if is_full:
#                         S_bi_null[tau_idx, k_R] += bi_n.sum()
#                         S_bj_null[tau_idx, k_R] += bj_n.sum()

#     # ----- compute C and rho -----
#     def _C_rho(S_a, S_b, S_a2, S_b2, S_ab, S_bx, S_by, S_bxb, S_byb, S_n, full):
#         """
#         S_a, S_b     : Σ a_i,   Σ a_j      (x-component for 'full', scalar for others)
#         S_bx, S_by   : Σ Δy_i,  Σ Δy_j     (y-component; only used when full=True)
#         S_a2, S_b2   : Σ a_i²,  Σ a_j²     (|Δr|² for 'full', a² for scalars)
#         S_ab         : Σ a_i*a_j or Σ Δr_i·Δr_j
#         """
#         nn   = np.where(S_n > 0, S_n, np.nan)
#         mi   = S_a  / nn; mj  = S_b  / nn
#         mi2  = S_a2 / nn; mj2 = S_b2 / nn
#         mij  = S_ab / nn
#         if full:
#             myi  = S_bx / nn; myj = S_by / nn
#             Ci   = mi2 - mi**2 - myi**2     # <|Δr_i|²> - <Δx_i>² - <Δy_i>²
#             Cj   = mj2 - mj**2 - myj**2
#             C    = mij - mi*mj - myi*myj    # <Δr_i·Δr_j> - <Δx_i><Δx_j> - <Δy_i><Δy_j>
#         else:
#             Ci   = mi2 - mi**2
#             Cj   = mj2 - mj**2
#             C    = mij - mi*mj
#         rho  = C / np.sqrt(np.abs(Ci * Cj))
#         rho  = np.where(np.isfinite(rho), rho, np.nan)
#         M = 0.5 * (mi2 + mj2)                   # ← INSERT HERE, before the return
#         return C, rho, denom, M

#     C,      rho,      denom, M      = _C_rho(S_ai,     S_aj,     S_bi,     S_bj,
#                                               S_ai2,     S_aj2,     S_aiaj,     S_n,      is_full)
#     C_null, rho_null, _, _          = _C_rho(S_ai_null, S_aj_null, S_bi_null, S_bj_null,
#                                               S_ai2_null, S_aj2_null, S_aiaj_null, S_n_null, is_full)

#     rho_null = np.where(denom > 0, C_null / denom, np.nan)   # ### TAG_RHO_NULL

#     return tau_s_arr, R_centers_um, C, rho, C_null, rho_null, M   # M added


# # ---- helpers ----------------------------------------------------------------

# def _subtract_global_modes(x0, y0, dx, dy, ann_edges_um,
#                             correct_xy=True, correct_r=True, correct_rot=True):
#     """
#     Subtract global collective modes from displacement arrays for a single origin time.

#     correct_xy  : subtract mean (Δx, Δy) [translation]
#     correct_r   : subtract mean Δr_r per annulus [breathing]
#     correct_rot : subtract mean Δr_θ per annulus [rotation]
#     """
#     dx = dx.copy(); dy = dy.copy()
#     if len(dx) == 0:
#         return dx, dy

#     if correct_xy:
#         dx -= dx.mean()
#         dy -= dy.mean()

#     if not (correct_r or correct_rot):
#         return dx, dy

#     r    = np.hypot(x0, y0)
#     ok   = r > 1e-9
#     er_x = np.where(ok, x0 / r, 0.0); er_y = np.where(ok, y0 / r, 0.0)
#     et_x = -er_y;                       et_y =  er_x

#     d_r = dx * er_x + dy * er_y
#     d_t = dx * et_x + dy * et_y

#     ann_idx = np.clip(np.digitize(r, ann_edges_um) - 1, 0, len(ann_edges_um) - 2)

#     for a in range(len(ann_edges_um) - 1):
#         m = ann_idx == a
#         if m.sum() < 2:
#             continue
#         if correct_r:
#             d_r[m] -= d_r[m].mean()
#         if correct_rot:
#             d_t[m] -= d_t[m].mean()

#     return d_r * er_x + d_t * et_x, d_r * er_y + d_t * et_y


# def _project(x0, y0, dx, dy, projection):
#     """
#     Scalar projection of displacements onto the chosen axis.
#     Not called for projection='full' (handled directly in the pair loop).

#     projection : 'r'     – ê_r at the barycenter
#                  'theta' – ê_θ at the barycenter
#                  # TAG: add 'L' (longitudinal) and 'T' (transverse) projections here.
#                  # Both require the pair separation vector as an extra argument.
#                  # uL = (x0_j - x0_i, y0_j - y0_i) / R  (longitudinal)
#                  # uT = (-uL_y, uL_x)                     (transverse)
#     """
#     r    = np.hypot(x0, y0)
#     ok   = r > 1e-9
#     er_x = np.where(ok, x0 / r, 0.0); er_y = np.where(ok, y0 / r, 0.0)

#     if projection == 'r':
#         return dx * er_x + dy * er_y

#     if projection == 'theta':
#         return dx * (-er_y) + dy * er_x

#     raise ValueError(f"Unknown projection '{projection}'. Valid: 'full', 'r', 'theta'.")
    
    


# %%%% V2 - Displacement correlations C(tau, R) and rho(tau, R) 

# Computes, for each lag tau and inter-particle distance bin R:
#
#   For projection = 'r' or 'theta' (scalar projection a = u · Δr):
#     C_aa   = <a_i * a_j>  -  <a_i><a_j>
#     Var_a  = <a²> - <a>²
#     rho_aa = C_aa / sqrt(Var_i * Var_j)
#
#   For projection = 'full' (tensor trace, dot product):
#     C_full = <Δr_i · Δr_j>  -  <Δx_i><Δx_j>  -  <Δy_i><Δy_j>
#            = C_xx + C_yy  (lab-frame tensor trace)
#            ≈ C_rr + C_θθ  (barycenter polar trace — equal only if all barycenters
#              had the same orientation, which they don't; use 'r'+'theta' for that)
#     Var_i  = <|Δr_i|²>  -  <Δx_i>²  -  <Δy_i>²
#     rho    = C_full / sqrt(Var_i * Var_j)
#
#   # TAG: add 'L' and 'T' (pair-frame longitudinal/transverse) projections here
#
# Null model (section 4.4 of the report, "Modèle nul par décalage temporel"):
#   For each origin t, particle j's displacement is replaced by that of its nearest
#   spatial neighbour at a time-shifted origin t' = t + null_shift (mod film duration).
#   C_null is NOT expected to be zero: the nearest-neighbour matching preserves individual
#   displacement statistics and spatial geometry, so residual spatial correlations survive.
#   The correct interpretation (as in the report, fig. 3) is that C_null defines a BAND
#   around which C fluctuates under the null hypothesis; the real signal is C - C_null,
#   and correlations are significant only where C >> C_null (or C_null band).
#
# Global mode subtraction (section 2.4 / 4.3):
#   Per origin: subtract mean (Δx, Δy) [translation], then per annulus subtract
#   mean Δr_r [breathing] and mean Δr_θ [rotation].  Each mode is independently togglable.


def compute_displacement_correlations(df, PixPerUm, FPS, xC, yC,
                                      tau_s_list     = [0.2, 0.5, 2.0, 5.0],
                                      R_bin_edges_um = np.arange(2, 22, 1),
                                      projection     = 'full',
                                      correct_modes  = True,
                                      correct_xy     = True,
                                      correct_r      = True,
                                      correct_rot    = True,
                                      n_ann_modes    = 8,
                                      rbar_min_um    = 3.0,
                                      R_min_um       = 2.0,
                                      null_shift_s   = 30.0,
                                      null_match_um  = 1.5,
                                      origin_step    = 20):
    """
    Compute displacement correlation C(tau, R) and rho(tau, R).

    Parameters
    ----------
    df              : DataFrame with columns ['frame', 'x', 'y', 'particle'], positions in pixels.
    PixPerUm        : pixel-per-micron scale.
    FPS             : frames per second.
    xC, yC          : centre of the polar coordinate system (pixels).
    tau_s_list      : list of lag times in seconds.
    R_bin_edges_um  : 1-D array of bin edges for the inter-particle distance R (µm).
    projection      : 'full'  – tensor trace C_xx + C_yy  (Δr_i · Δr_j dot product, lab frame)
                      'r'     – project on ê_r at the pair barycenter → C_rr
                      'theta' – project on ê_θ at the pair barycenter → C_θθ
                      # TAG: 'L' and 'T' (pair-frame longitudinal/transverse) to be added here
    correct_modes   : if True, subtract global modes before computing correlations.
    correct_xy      : subtract mean (Δx, Δy) per origin [translation].
    correct_r       : subtract mean Δr_r per annulus [breathing].
    correct_rot     : subtract mean Δr_θ per annulus [rotation].
    n_ann_modes     : number of equal-population annular bins for mode subtraction.
    rbar_min_um     : exclude pairs whose barycenter is closer than this to the centre.
    R_min_um        : exclude pairs closer than this (risk of detection overlaps).
    null_shift_s    : time shift for the null model (seconds).
    null_match_um   : max spatial distance for the nearest-neighbour match in the null model.
    origin_step     : subsample origins; only frames where (frame - f_min) % origin_step == 0
                      are used.

    Returns
    -------
    tau_s_arr    : 1-D array, lag times (s), shape (n_tau,)
    R_centers_um : 1-D array, bin centres (µm), shape (n_R,)
    C            : 2-D array, covariance, shape (n_tau, n_R)
    rho          : 2-D array, correlation coefficient, shape (n_tau, n_R)
    C_null       : 2-D array, null-model covariance, shape (n_tau, n_R)
    rho_null     : 2-D array, null-model rho = C_null / sqrt(Var_i * Var_j),
    M            : 2-D array, average raw second moment of pair members, shape (n_tau, n_R).
                   M = ½(<a_i²> + <a_j²>), i.e. the per-component MSD estimated on the
                   same pair sample as C and rho.  The ratio C/M gives the collective
                   fraction of the displacement variance for each (tau, R) bin (section 4.6).
                   where Var_i and Var_j come from the REAL data (same denominator
                   as rho), so that rho and rho_null are directly comparable.  ### TAG_RHO_NULL
    """

    df = df.sort_values(['particle', 'frame']).reset_index(drop=True)
    UmPerPix = 1.0 / PixPerUm

    # Lag times in frames, deduplicated
    seen = set()
    tau_frames_unique = [t for t in [int(round(tau_s * FPS)) for tau_s in tau_s_list]
                         if not (t in seen or seen.add(t))]

    n_tau        = len(tau_frames_unique)
    n_R          = len(R_bin_edges_um) - 1
    R_centers_um = 0.5 * (R_bin_edges_um[:-1] + R_bin_edges_um[1:])
    tau_s_arr    = np.array([t / FPS for t in tau_frames_unique])

    # Positions in µm, centred on (xC, yC)
    frames_all   = df['frame'].values
    x_um_all     = (df['x'].values - xC) * UmPerPix
    y_um_all     = (df['y'].values - yC) * UmPerPix
    particle_all = df['particle'].values

    frames_unique = np.unique(frames_all)
    frame_to_rows = {f: np.where(frames_all == f)[0] for f in frames_unique}

    null_shift_fr = int(round(null_shift_s * FPS))
    f_min = int(frames_unique.min())
    f_max = int(frames_unique.max())
    f_range = f_max - f_min + 1

    # Equal-population annular edges for mode subtraction
    r_all        = np.hypot(x_um_all, y_um_all)
    ann_edges_um = np.quantile(r_all[r_all > 0], np.linspace(0, 1, n_ann_modes + 1))
    ann_edges_um[0]  -= 1e-6
    ann_edges_um[-1] += 1e-6

    # --- accumulators ---
    # Scalar projections ('r', 'theta'):
    #   S_ai, S_aj  : Σ a_i,  Σ a_j
    #   S_ai2, S_aj2: Σ a_i², Σ a_j²
    #   S_aiaj      : Σ a_i*a_j
    #
    # Full projection ('full'):
    #   S_ai, S_aj  : Σ Δx_i, Σ Δx_j         (x-component mean for bias correction)
    #   S_bi, S_bj  : Σ Δy_i, Σ Δy_j         (y-component mean for bias correction)
    #   S_ai2, S_aj2: Σ |Δr_i|², Σ |Δr_j|²   (for variance)
    #   S_aiaj      : Σ (Δr_i · Δr_j)         (dot product cross term)
    # C_full = S_aiaj/n - (S_ai/n)(S_aj/n) - (S_bi/n)(S_bj/n)
    # Var_i  = S_ai2/n  - (S_ai/n)²        - (S_bi/n)²

    def _empty():
        return np.zeros((n_tau, n_R), dtype=np.float64)

    S_ai   = _empty(); S_aj   = _empty()
    S_bi   = _empty(); S_bj   = _empty()   # y-component means (only used for 'full')
    S_ai2  = _empty(); S_aj2  = _empty()
    S_aiaj = _empty(); S_n    = _empty()

    S_ai_null   = _empty(); S_aj_null   = _empty()
    S_bi_null   = _empty(); S_bj_null   = _empty()
    S_ai2_null  = _empty(); S_aj2_null  = _empty()
    S_aiaj_null = _empty(); S_n_null    = _empty()

    is_full = (projection == 'full')

    # ----- main loop over lag times -----
    for tau_idx, tau_fr in enumerate(tau_frames_unique):

        origins = frames_unique[frames_unique + tau_fr <= f_max]
        origins = origins[(origins - f_min) % origin_step == 0]

        for o in origins:
            rows_i = frame_to_rows.get(o,         None)
            rows_j = frame_to_rows.get(o + tau_fr, None)
            if rows_i is None or rows_j is None:
                continue

            # particles present at both t=o and t=o+tau
            pid_i = particle_all[rows_i]; pid_j = particle_all[rows_j]
            pid_common = np.intersect1d(pid_i, pid_j)
            if len(pid_common) < 2:
                continue

            # sort both sets so particle ordering matches
            si = np.argsort(pid_i[np.isin(pid_i, pid_common)])
            sj = np.argsort(pid_j[np.isin(pid_j, pid_common)])
            idx_i = rows_i[np.isin(pid_i, pid_common)][si]
            idx_j = rows_j[np.isin(pid_j, pid_common)][sj]

            x0 = x_um_all[idx_i]
            y0 = y_um_all[idx_i]
            dx = x_um_all[idx_j] - x0
            dy = y_um_all[idx_j] - y0

            # ---- global mode subtraction ----
            if correct_modes:
                dx, dy = _subtract_global_modes(x0, y0, dx, dy, ann_edges_um,
                                                correct_xy, correct_r, correct_rot)

            # ---- null-model origin o' ----
            o_null_target = int((o - f_min + null_shift_fr) % f_range + f_min)
            o_null = frames_unique[np.argmin(np.abs(frames_unique - o_null_target))]
            if abs(o_null - o) < null_shift_fr // 2:
                o_null = None
            if o_null is not None and frame_to_rows.get(o_null + tau_fr) is None:
                o_null = None

            # ---- pair geometry ----
            pts    = np.c_[x0, y0]
            tree_i = cKDTree(pts)
            pairs  = tree_i.query_pairs(R_bin_edges_um[-1], output_type='ndarray')
            if len(pairs) == 0:
                continue

            p, q = pairs[:, 0], pairs[:, 1]
            bx = (x0[p] + x0[q]) / 2;  by = (y0[p] + y0[q]) / 2
            rb = np.hypot(bx, by)
            R  = np.hypot(x0[q] - x0[p], y0[q] - y0[p])

            keep = (R >= R_min_um) & (rb >= rbar_min_um)
            p, q, R, bx, by, rb = p[keep], q[keep], R[keep], bx[keep], by[keep], rb[keep]
            if len(p) == 0:
                continue

            iR       = np.digitize(R, R_bin_edges_um) - 1
            in_range = (iR >= 0) & (iR < n_R)
            p, q, iR, bx, by, rb = [v[in_range] for v in (p, q, iR, bx, by, rb)]
            if len(p) == 0:
                continue

            # ---- per-pair quantities ----
            if is_full:
                # a = Δx (for mean bias), b = Δy (for mean bias)
                # ai2 = |Δr|² (for variance), cross = Δr_i · Δr_j
                ai = dx[p]; aj = dx[q]          # x-components (bias terms)
                bi = dy[p]; bj = dy[q]          # y-components (bias terms)
                ai2 = dx[p]**2 + dy[p]**2       # |Δr_i|²
                aj2 = dx[q]**2 + dy[q]**2       # |Δr_j|²
                cross = dx[p]*dx[q] + dy[p]*dy[q]  # Δr_i · Δr_j
            else:
                a   = _project(x0, y0, dx, dy, projection)
                ai  = a[p]; aj = a[q]
                bi  = None; bj = None
                ai2 = ai**2; aj2 = aj**2
                cross = ai * aj

            # ---- accumulate ----
            for k_R in range(n_R):
                m = iR == k_R
                if not m.any():
                    continue
                S_ai  [tau_idx, k_R] += ai[m].sum()
                S_aj  [tau_idx, k_R] += aj[m].sum()
                S_ai2 [tau_idx, k_R] += ai2[m].sum()
                S_aj2 [tau_idx, k_R] += aj2[m].sum()
                S_aiaj[tau_idx, k_R] += cross[m].sum()
                S_n   [tau_idx, k_R] += m.sum()
                if is_full:
                    S_bi[tau_idx, k_R] += bi[m].sum()
                    S_bj[tau_idx, k_R] += bj[m].sum()

            # ---- null model ----
            if o_null is not None:
                rows_null_0   = frame_to_rows[o_null]
                rows_null_tau = frame_to_rows[o_null + tau_fr]

                # Keep only particles present at both null frames (same intersection
                # logic as for the real origin above)
                pid_null_0   = particle_all[rows_null_0]
                pid_null_tau = particle_all[rows_null_tau]
                pid_null_common = np.intersect1d(pid_null_0, pid_null_tau)

                sn0  = np.argsort(pid_null_0  [np.isin(pid_null_0,   pid_null_common)])
                snt  = np.argsort(pid_null_tau [np.isin(pid_null_tau, pid_null_common)])
                idx_null_0   = rows_null_0  [np.isin(pid_null_0,   pid_null_common)][sn0]
                idx_null_tau = rows_null_tau[np.isin(pid_null_tau, pid_null_common)][snt]

                x0_no   = x_um_all[idx_null_0]; y0_no   = y_um_all[idx_null_0]
                dx_null = x_um_all[idx_null_tau] - x0_no
                dy_null = y_um_all[idx_null_tau] - y0_no

                if correct_modes:
                    dx_null, dy_null = _subtract_global_modes(
                        x0_no, y0_no, dx_null, dy_null, ann_edges_um,
                        correct_xy, correct_r, correct_rot)

                # nearest neighbour in null frame for every particle i at origin o
                tree_null      = cKDTree(np.c_[x0_no, y0_no])
                dist_nn, idx_nn = tree_null.query(pts, k=1)
                matched        = dist_nn < null_match_um

                dx_nn = dx_null[idx_nn]; dy_nn = dy_null[idx_nn]

                if is_full:
                    # null aj = Δx_null[j_match], bj = Δy_null[j_match]
                    # cross_null = Δr_i(real) · Δr_j(null)
                    pass    # computed inside the R-bin loop below
                else:
                    a_null = _project(x0_no[idx_nn], y0_no[idx_nn], dx_nn, dy_nn, projection)

                for k_R in range(n_R):
                    m   = iR == k_R
                    if not m.any():
                        continue
                    p_m = p[m]; q_m = q[m]
                    ok  = matched[p_m] & matched[q_m]
                    if not ok.any():
                        continue

                    if is_full:
                        ai_n    = dx[p_m[ok]];         aj_n  = dx_nn[q_m[ok]]
                        bi_n    = dy[p_m[ok]];         bj_n  = dy_nn[q_m[ok]]
                        ai2_n   = ai_n**2 + bi_n**2
                        aj2_n   = aj_n**2 + bj_n**2
                        cross_n = ai_n*aj_n + bi_n*bj_n
                    else:
                        ai_n    = ai[m][ok]
                        aj_n    = a_null[q_m[ok]]
                        bi_n    = None; bj_n = None
                        ai2_n   = ai_n**2; aj2_n = aj_n**2
                        cross_n = ai_n * aj_n

                    S_ai_null  [tau_idx, k_R] += ai_n.sum()
                    S_aj_null  [tau_idx, k_R] += aj_n.sum()
                    S_ai2_null [tau_idx, k_R] += ai2_n.sum()
                    S_aj2_null [tau_idx, k_R] += aj2_n.sum()
                    S_aiaj_null[tau_idx, k_R] += cross_n.sum()
                    S_n_null   [tau_idx, k_R] += ok.sum()
                    if is_full:
                        S_bi_null[tau_idx, k_R] += bi_n.sum()
                        S_bj_null[tau_idx, k_R] += bj_n.sum()

    # ----- compute C, rho, C_null, rho_null -----

    def _C_rho(S_ai, S_aj, S_bi, S_bj, S_ai2, S_aj2, S_aiaj, S_n, full):
        """
        C   = <a_i * a_j>  -  <a_i><a_j>
        Var = <a_i²> - <a_i>²
        rho = C / sqrt(Var_i * Var_j)
        For 'full': a = Δx (bias/variance use both Δx and Δy components).
        """
        nn   = np.where(S_n > 0, S_n, np.nan)
        mi   = S_ai  / nn;  mj  = S_aj  / nn
        mi2  = S_ai2 / nn;  mj2 = S_aj2 / nn
        mij  = S_aiaj / nn
        if full:
            myi  = S_bi / nn;  myj = S_bj / nn
            Ci   = mi2 - mi**2 - myi**2
            Cj   = mj2 - mj**2 - myj**2
            C    = mij - mi*mj - myi*myj
        else:
            Ci   = mi2 - mi**2
            Cj   = mj2 - mj**2
            C    = mij - mi*mj
        denom = np.sqrt(np.abs(Ci * Cj))
        rho   = np.where(denom > 0, C / denom, np.nan)
        # M = ½(<a_i²> + <a_j²>) : average raw second moment of the two pair members.
        # For 'full': mi2 = <|Δr_i|²>, so M = ½(<|Δr_i|²> + <|Δr_j|²>).
        # For 'r'/'theta': M = ½(<(u·Δr_i)²> + <(u·Δr_j)²>) = per-component MSD on the pair sample.
        # Uses the raw second moment (not centred), matching the report's definition (section 4.6).
        M     = 0.5 * (mi2 + mj2)
        return C, rho, denom, M

    C, rho, denom, M = _C_rho(S_ai,      S_aj,      S_bi,      S_bj,
                                         S_ai2,      S_aj2,      S_aiaj,      S_n,      is_full)
    C_null, _, _, _  = _C_rho(S_ai_null,  S_aj_null,  S_bi_null,  S_bj_null,
                                         S_ai2_null,  S_aj2_null,  S_aiaj_null,  S_n_null, is_full)

    # ### TAG_RHO_NULL: rho_null uses the REAL data denominator (same Var_i, Var_j as rho)
    # so that rho and rho_null are on the same scale and directly comparable.
    # Using the null denominator would give a different normalisation and make
    # the two curves harder to compare visually.
    rho_null = np.where(denom > 0, C_null / denom, np.nan)                  # ### TAG_RHO_NULL

    return tau_s_arr, R_centers_um, C, rho, C_null, rho_null, M


# ---- helpers ----------------------------------------------------------------

def _subtract_global_modes(x0, y0, dx, dy, ann_edges_um,
                            correct_xy=True, correct_r=True, correct_rot=True):
    """
    Subtract global collective modes from displacement arrays for a single origin time.

    correct_xy  : subtract mean (Δx, Δy) [translation]
    correct_r   : subtract mean Δr_r per annulus [breathing]
    correct_rot : subtract mean Δr_θ per annulus [rotation]
    """
    dx = dx.copy(); dy = dy.copy()
    if len(dx) == 0:
        return dx, dy

    if correct_xy:
        dx -= dx.mean()
        dy -= dy.mean()

    if not (correct_r or correct_rot):
        return dx, dy

    r    = np.hypot(x0, y0)
    ok   = r > 1e-9
    er_x = np.where(ok, x0 / r, 0.0)
    er_y = np.where(ok, y0 / r, 0.0)
    et_x = -er_y
    et_y =  er_x

    d_r = dx * er_x + dy * er_y
    d_t = dx * et_x + dy * et_y

    ann_idx = np.clip(np.digitize(r, ann_edges_um) - 1, 0, len(ann_edges_um) - 2)

    for a in range(len(ann_edges_um) - 1):
        m = (ann_idx == a)
        if m.sum() < 2:
            continue
        if correct_r:
            d_r[m] -= d_r[m].mean()
        if correct_rot:
            d_t[m] -= d_t[m].mean()

    return (d_r * er_x) + (d_t * et_x), (d_r * er_y) + (d_t * et_y)


def _project(x0, y0, dx, dy, projection):
    """
    Scalar projection of displacements onto the chosen axis.
    Not called for projection='full' (handled directly in the pair loop).

    projection : 'r'     – ê_r at the barycenter
                 'theta' – ê_θ at the barycenter
                 # TAG: add 'L' (longitudinal) and 'T' (transverse) projections here.
                 # Both require the pair separation vector as an extra argument.
                 # uL = (x0_j - x0_i, y0_j - y0_i) / R  (longitudinal)
                 # uT = (-uL_y, uL_x)                     (transverse)
    """
    r    = np.hypot(x0, y0)
    ok   = r > 1e-9
    er_x = np.where(ok, x0 / r, 0.0); er_y = np.where(ok, y0 / r, 0.0)

    if projection == 'r':
        return dx * er_x + dy * er_y

    if projection == 'theta':
        return dx * (-er_y) + dy * er_x

    raise ValueError(f"Unknown projection '{projection}'. Valid: 'full', 'r', 'theta'.")

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
    
    # # Convert thresholds from Um to Pix if necessary
    # if IMG_UNITS == 'PIX':
    #     RADIUS = Pix_Per_Um * RADIUS_UM
    #     THRESH_LINK = Pix_Per_Um * THRESH_LINK_UM
    # elif IMG_UNITS == 'UM':
    #     RADIUS = RADIUS_UM
    #     THRESH_LINK = THRESH_LINK_UM
    # else:
    #     raise ValueError("Setting variable IMG_UNITS should be equal to 'PIX' or 'UM'")
    RADIUS = Pix_Per_Um * RADIUS_UM
    THRESH_LINK = Pix_Per_Um * THRESH_LINK_UM
    
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
    
    print('\nDone!\n')


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


def pretreat_and_track_NbYolk(tifPath, rawTrackName, dstDir, figDir,
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
    
# %% Z. Older versions

# %%% Relative MSD

def get_relative_displacement_by_TRange_old(df, pairs, TRange):
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
